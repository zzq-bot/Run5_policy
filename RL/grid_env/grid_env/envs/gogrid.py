from typing import Any, Optional, Tuple, Union, Dict, List
from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

REL_POS = np.zeros((2, 12, 144), np.int32)
for k in range(12):
    for i in range(12):
        for j in range(12):
            REL_POS[0, k, i * 12 + j] = i - k + 11
            REL_POS[1, k, i * 12 + j] = j - k + 11

    
class GoGridWorldEnvV0(gym.Env):
    
    def __init__(
        self,
        size: int = 12,
        elim_n: int = 4,
        cls_n: int = 21,
        fixed_field: Union[np.ndarray, bool] = False,
        fixed_loc: Union[np.ndarray, bool] = False,
        use_aux_reward: bool = False,
        aux_reward_coef: float = 2.0,
        process_picked: bool = False,
        reduce: bool = False,
        force_all_char: bool = False,
        **kwargs, # maybe add noise to observations
    ) -> None:
        
        self.size = size
        self.elim_n = elim_n
        self.cls_n = cls_n
        self.grid_shape = self.size * self.size
        self._real_max_episode_steps = 4 * size ** 2 # 576 steps
        self._max_episode_steps = size ** 2 # select 144 positions
        
        self._process_picked = process_picked
        self._reduce = reduce
        self._force_all_char = force_all_char

        
        if isinstance(fixed_field, bool):
            if fixed_field:
                assert 0, "should not use bool fixed field unless you just want to check algo in a toy case"
                self.fixed_field = -np.ones((size, size), np.int32)
                all_char_types = np.random.randint(0, self.cls_n, 36)
                empty_pos = np.argwhere(self.fixed_field == -1).tolist() # len() = 144
            
                for char_type in all_char_types:
                    for _ in range(4):
                        pos = empty_pos.pop(np.random.randint(0, len(empty_pos)))
                        self.fixed_field[pos[0], pos[1]] = char_type
            else:
                self.fixed_field = None
        elif isinstance(fixed_field, np.ndarray):
            assert fixed_field.shape == (size, size)
            assert np.all(fixed_field >= 0) and np.all(fixed_field < cls_n)
            
            self.fixed_field = fixed_field
        else:
            assert 0
            
        if isinstance(fixed_loc, bool):
            if fixed_loc:
                self.fixed_loc = np.random.randint(0, self.size, 2)
            else:
                self.fixed_loc = None
        elif isinstance(fixed_loc, np.ndarray):
            assert fixed_loc.shape == (2,)
            assert np.all(fixed_loc >= 0) and np.all(fixed_loc < size)
            self.fixed_loc = fixed_loc
        else:
            assert 0
        
        if self.fixed_field is not None and self._reduce:
            # 直接reduce
            self.fixed_field = self.reduce_field(self.fixed_field)
            
        self.field = -np.ones((size, size), np.int32)

        self.observation_space = self._get_observation_space()
        
        self.use_aux_reward = use_aux_reward
        self.aux_reward_coef = aux_reward_coef
        
        self.action_space = spaces.Discrete(self.grid_shape)
        self.abs_pos_row = np.array(
            [[i] * self.size for i in range(self.size)]
        )
        self.abs_pos_col = np.array(
            [np.arange(self.size) for _ in range(self.size)]
        )
        
        
    def _get_observation_space(self) -> spaces.Box:

        low_obs = np.array([-np.inf] * (self.grid_shape * 4 + 2), dtype=np.float32)
        high_obs = np.array([np.inf] * (self.grid_shape * 4 + 2), dtype=np.float32)
        
        return spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
    
    def _get_avail_actions(self) -> np.ndarray:
        return self.valid_moves
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[:self.grid_shape] = self.field.flatten()
        if self._process_picked:
            obs[obs == -1] = 21
        # obs[self.grid_shape:self.grid_shape * 2] = self.abs_pos_row.flatten() - self.loc[0] + 11 # -11~11 -> 0~22
        # obs[self.grid_shape * 2:self.grid_shape * 3] = self.abs_pos_col.flatten() - self.loc[1] + 11 # -11~11 -> 0~22
        obs[self.grid_shape:self.grid_shape * 2] = REL_POS[0, self.loc[0], :]
        obs[self.grid_shape * 2:self.grid_shape * 3] = REL_POS[1, self.loc[1], :]

        obs[self.grid_shape * 3:self.grid_shape * 4] = self.pos2bagn.flatten() # 0~3
        obs[self.grid_shape * 4:self.grid_shape * 4 + 2] = self.loc
        

        return obs

    def spawn_char(self) -> None:
        if self._force_all_char:
            base_char_types = list(range(21))
            additional_char_types = np.random.randint(0, 21, 36-21)
            all_char_types = base_char_types + additional_char_types.tolist()
        else:
            all_char_types = np.random.randint(0, self.cls_n, 36)
        empty_pos = np.argwhere(self.field == -1).tolist() # len() = 144
        for char_type in all_char_types:
            for _ in range(4):
                pos = empty_pos.pop(np.random.randint(0, len(empty_pos)))
                self.field[pos[0], pos[1]] = char_type
    
    def reduce_field(self, field: np.ndarray) -> np.ndarray:
        cnt = {}
        for i in range(self.size):
            for j in range(self.size):
                if field[i, j] == -1:
                    continue
                if field[i, j] not in cnt:
                    cnt[field[i, j]] = len(cnt)
                field[i, j] = cnt[field[i, j]]
            
        assert len(cnt) <= 21, "sth wrong"
        return field

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed, options=options)

        # 1. set field
        if self.fixed_field is None:
            self.field = -np.ones((self.size, self.size), np.int32)
            self.spawn_char()
            if self._reduce:
                self.field = self.reduce_field(self.field)
        else:
            self.field = self.fixed_field.copy()
            
        self.char2pos = defaultdict(list)
        for i in range(self.size):
            for j in range(self.size):
                if self.field[i, j] != -1:
                    self.char2pos[self.field[i, j]].append((i, j))
        
        self.pos2bagn = np.zeros((self.size, self.size))
        # 2. set agent loc
        # self.loc = np.array([0, 0])
        if self.fixed_loc is None:
            self.loc = np.random.randint(0, self.size, 2)
        else:
            self.loc = self.fixed_loc.copy()
        # 3. set score
        self.score = 0
        
        # 4. set bag
        self.bag = np.zeros(self.cls_n, np.int32)
        
        self.current_step = 0
        self.real_current_step = 0
        self.valid_moves = np.ones(self.grid_shape, np.int32)
        
        self.path = [(self.loc[0], self.loc[1])]
        info = {
            "avail_actions": self._get_avail_actions(),
            "score": self.score,
        }
        
        return self._get_observation(), info

    def step_logic(self, obs: np.ndarray, action: Union[int, np.ndarray], valid_moves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # be used in MCTS
        if isinstance(action, np.ndarray):
            assert action.shape == (1,)
            action = action[0]
        field = obs[:144].reshape(12, 12).astype(int)
        field[field==21] = -1
        pos2bagn = obs[144*3:144*4].reshape(12, 12)
        loc = obs[-2:]
        bag = np.zeros(21)
        char_types, char_type_cnts = np.unique(field, return_counts=True)
        
        for idx, cnt in zip(char_types, char_type_cnts):
            if idx == -1: continue
            bag[idx] = (4 - cnt % 4) % 4
        
        terminated = truncated = False
        target_row, target_col = divmod(action, 12)
        n_move = abs(target_row - loc[0]) + abs(target_col - loc[1])
        reward = - (0.1 + np.sum(bag) / 144) * (n_move + 1) # cost of "move and pickup"
        loc[0] = target_row
        loc[1] = target_col
        
        char_type = field[target_row, target_col]
        assert char_type != -1
        bag[char_type] += 1
        field[target_row, target_col] = -1
        valid_moves[target_row * 12 + target_col] = 0
        if bag[char_type] == 4:
            bag[char_type] = 0
            reward += 1
        for i in range(12):
            for j in range(12):
                if field[i, j] == -1: continue
                pos2bagn[i, j] = bag[field[i, j]]
        
        succeed = False
        
        if np.all(field == -1):
            terminated = True

        info = {
            "avail_actions": valid_moves,
            "succeed": succeed,
            # "bag": bag,
        }
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self._process_picked:
            field[field==-1] = 21
        obs[:144] = field.flatten()
    
        obs[144:144 * 2] = REL_POS[0, int(loc[0]), :]
        obs[144 * 2:144 * 3] = REL_POS[1, int(loc[1]), :]

        obs[144 * 3:144 * 4] = pos2bagn.flatten() # 0~3
        obs[144 * 4:144 * 4 + 2] = loc
        
        
        return obs, reward, terminated, truncated, info
        
        
        
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if isinstance(action, np.ndarray):
            assert action.shape == (1,)
            action = action[0]

        terminated = truncated = False
        target_row, target_col = divmod(action, self.size)
        
        self.path.append((target_row, target_col))
        
        n_move = abs(target_row - self.loc[0]) + abs(target_col - self.loc[1])
        reward = - (0.1 + np.sum(self.bag) / self.grid_shape) * (n_move + 1) # cost of "move and pickup"
        self.current_step += 1
        self.real_current_step += (n_move + 1) 
        self.loc[0] = target_row
        self.loc[1] = target_col
        
        char_type = self.field[target_row, target_col]
        assert char_type != -1
        self.bag[char_type] += 1
        self.field[target_row, target_col] = -1
        self.valid_moves[target_row * self.size + target_col] = 0
        if self.bag[char_type] == 4:
            self.bag[char_type] = 0
            reward += 1            
        self.score += reward
        
        self.char2pos[char_type].remove((target_row, target_col))
        self.pos2bagn[target_row, target_col] = 0
        for pos in self.char2pos[char_type]:
            self.pos2bagn[pos[0], pos[1]] = self.bag[char_type]
        succeed = False
        
        if np.all(self.field == -1):
            if self.real_current_step <= self._real_max_episode_steps:
                self.score += 100
                succeed = True
            else:
                self.score += -3 * (np.sum(self.field != -1) + np.sum(self.bag))
            terminated = True
        

        info = {
            "avail_actions": self._get_avail_actions(),
            "score": self.score,
            "succeed": succeed,
            "real_length": self.real_current_step,
            "path": self.path
        }
        if self.use_aux_reward:
            raise NotImplementedError("GoGrid use_aux_reward, TBD")
    
        return self._get_observation(), reward, terminated, truncated, info
    
    def seed(self, seed: Union[int]=None) -> None:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def get_plan(self, path: List[Tuple[int]] | None = None) -> List[int]:
        # get actions from self.path
        if path is None:
            path = self.path
        plan = []
        
        cur_pos = path[0]
        for i in range(1, len(path)):
            nxt_pos = path[i]
            for _ in range(nxt_pos[0]-cur_pos[0]):
                plan.append(0)
            for _ in range(nxt_pos[1]-cur_pos[1]):
                plan.append(1)
            for _ in range(cur_pos[0]-nxt_pos[0]):
                plan.append(2)
            for _ in range(cur_pos[1]-nxt_pos[1]):
                plan.append(3)
            plan.append(4)
            cur_pos = nxt_pos
        return plan
    
    def load_ckpt(self, field: np.ndarray, loc: np.ndarray, bag: np.ndarray, score: float, step: int, real_current_step: int) -> None:
        self.field = field
        self.loc = loc
        # self.bag = bag
        self.bag = bag
        self.score = score
        self.current_step = step
        self.real_current_step = real_current_step
        self.char2pos = defaultdict(list)
        for i in range(self.size):
            for j in range(self.size):
                if self.field[i, j] != -1:
                    self.char2pos[self.field[i, j]].append((i, j))

        self.pos2bagn = np.zeros((self.size, self.size))
        self.valid_moves = np.ones(self.grid_shape, np.int32)
        for i in range(12):
            for j in range(12):
                if self.field[i, j] == -1:
                    self.valid_moves[i * 12 + j] = 0
                    continue
                self.pos2bagn[i, j] = self.bag[self.field[i, j]]
        self.path = [(loc[0], loc[1])]
    
    def get_obs_and_info(self) -> None:
        info = {
            "avail_actions": self._get_avail_actions(),
            "score": self.score,
        }
        
        return self._get_observation(), info

    