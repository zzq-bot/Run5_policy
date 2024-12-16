from typing import Any, Optional, Tuple, Union, Dict
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class Action(Enum):
    DOWN = 0
    RIGHT = 1
    UP = 2
    LEFT = 3
    LOAD = 4
    
class EvalGridWorldEnvV0(gym.Env):
    
    action_set = [Action.DOWN, Action.RIGHT, Action.UP, Action.LEFT, Action.LOAD]
    def __init__(
        self,
        size: int = 12,
        elim_n: int = 4,
        cls_n: int = 21,
        fixed_field: Union[np.ndarray, bool] = True,
        fixed_loc: Union[np.ndarray, bool] = False,
        process_picked: bool = False,
        reduce: bool = False,
        one_hot_pos: bool = False,
        **kwargs, # add noise to observations...?
    ) -> None:
        
        self.size = size
        self.grid_size = self.size * self.size
        self.elim_n = elim_n
        self.cls_n = cls_n
        self._max_episode_steps = 4 * size ** 2
        
        self._process_picked = process_picked
        self._reduce = reduce
        self._one_hot_pos = one_hot_pos

        
    
            
        self.fixed_field = fixed_field
        self.fixed_loc = fixed_loc

        
        if self.fixed_field is not None and self._reduce:
            self.fixed_field = self.reduce_field(self.fixed_field)
            
        self.field = -np.ones((size, size), np.int32)
            
        self.observation_space = self._get_observation_space()
        
        self.action_space = spaces.Discrete(5)
    
    def _get_observation_space(self) -> spaces.Box:
        self.grid_shape = 144
        if self._one_hot_pos:
            self.agent_info_shape = 12 * 2 + 21
        else:
            self.agent_info_shape = 2 + 21
    
        low_obs = np.array([-np.inf] * (self.grid_shape + self.agent_info_shape), dtype=np.float32)
        high_obs = np.array([np.inf] * (self.grid_shape + self.agent_info_shape), dtype=np.float32)
        
        return spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
    
    def _get_avail_actions(self) -> np.ndarray:
        valid_moves = np.ones(5, np.int32)
        if self.loc[0] == self.size - 1:
            valid_moves[0] = 0
        if self.loc[1] == self.size - 1:
            valid_moves[1] = 0
        if self.loc[0] == 0:
            valid_moves[2] = 0
        if self.loc[1] == 0:
            valid_moves[3] = 0
        if self.field[self.loc[0], self.loc[1]] == -1:
            valid_moves[4] = 0
        return valid_moves
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[:144] = self.field.flatten()
        
        if self._one_hot_pos:
            loc_x = np.zeros(12, np.int32)
            loc_x[self.loc[0]] = 1
            loc_y = np.zeros(12, np.int32)
            loc_y[self.loc[1]] = 1
            
            obs[144:144 + 12] = loc_x
            obs[144+12:144+12*2] = loc_y
            obs[144+12*2:] = self.bag % 4
        else:
            obs[144:144+2] = self.loc
            obs[144+2:] = self.bag % 4
        if self._process_picked:
            obs[obs == -1] = 21
        return obs

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
        
        self.field = self.fixed_field.copy()
        
        
        self.loc = self.fixed_loc.copy()
        
        # 3. set score
        self.score = 0
        
        # 4. set bag
        self.bag = np.zeros(21, np.int32)
        
        self.current_step = 0

        info = {
            "avail_actions": self._get_avail_actions(),
            "score": self.score,
        }
        
        return self._get_observation(), info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if isinstance(action, np.ndarray):
            assert action.shape == (1,)
            action = action[0]
        
        
        terminated = False
        truncated = False
        reward = 0
        
        self.current_step += 1
        reward += - (0.1 + np.sum(self.bag) / self.grid_shape)
        action = Action(action)
        if action == Action.DOWN:
            if self.loc[0] != self.size - 1:
                self.loc[0] += 1
        elif action == Action.RIGHT:
            if self.loc[1] != self.size - 1:
                self.loc[1] += 1
        elif action == Action.UP:
            if self.loc[0] != 0:
                self.loc[0] -= 1
        elif action == Action.LEFT:
            if self.loc[1] != 0:
                self.loc[1] -= 1
        elif action == Action.LOAD:
            if self.field[self.loc[0], self.loc[1]] == -1:
                assert 0, f"should not happen, loc: {self.loc}"
            char_type = self.field[self.loc[0], self.loc[1]]
            self.bag[char_type] += 1
            self.field[self.loc[0], self.loc[1]] = -1
            if self.bag[char_type] == 4:
                self.bag[char_type] = 0
                reward += 1
        else:
            assert 0, "should not happen"

        
        if np.all(self.field == -1):
            reward += 100.
            terminated = True
        elif self.current_step == self._max_episode_steps:
            reward = -3 * (np.sum(self.field != -1) + np.sum(self.bag))
            truncated = True
        
        
        self.score += reward
        info = {
            "avail_actions": self._get_avail_actions(),
            "score": self.score
        }
        return self._get_observation(), reward, terminated, truncated, info

    
    def seed(self, seed: Union[int]=None) -> None:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def load_ckpt(self, grid: np.ndarray, loc: np.ndarray, bag: np.ndarray, score: float, step: int) -> None:
        self.field = grid
        self.loc = loc
        self.bag = bag
        self.score = score
        self.current_step = step
    
    def get_obs_and_info(self):
        info = {
            "avail_actions": self._get_avail_actions(),
            "score": self.score,
            "action_seq": self.action_seq
        }
        return self._get_observation(), info