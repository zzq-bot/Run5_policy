import os
import random
import time
import math
from dataclasses import dataclass
from typing import Union, Tuple, Any


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from dotenv import load_dotenv  # pip install python-dotenv
from torch.distributions.categorical import Categorical
from utils.transformers import Transformer

import wandb

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "x_ubiquant_rl_zzq"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "grid_env/GoGridWorld-v0"
    
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    
    num_steps: int = 12 * 12
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    hidden_dim: int = 128
    """the hidden dimension of the neural networks"""
    fixed_field: bool = False
    """whether to use a fixed field for each episode"""
    use_aux_reward: bool = False
    """whether to use auxiliary reward"""
    reduce_field: bool = True
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    eval_only: bool = False
    render: bool = False
    eval_maps = list(range(100)) # will not be reserved in wandb
    
    # The following params might be changed often
    resume_path: str = "./policy_ckpt/ppo_allallin.pt"  # "ckpt/1112"
    num_envs: int = 200
    norm_rew: bool = True
    use_ret_rms: bool = False
    
    embed_dim: int = 128
    embed_type: str = "sinusoidal" # rope, sinusoidal
    # env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = int(2e9)
    save_interval: int = int(1e7)
    eval_interval: int = int(1e7)
    aux_reward_coef: float = 1.0
    one_hot_pos: bool = True
    gravity_reward: bool = False
    train_size: int = 12
    train_types: int = 21
    
    share_cnn: bool = True
    use_eval: bool = True
    baka_think_first: bool = False
    
    n_head: int = 8
    bias: bool = True
    drop_p: float = 0.1
    n_layer: int = 3

    num_simulations: int = 20
    exploration_constant: float = 1.0


def make_eval_env(env_id: str, idx: int, args: Args) -> Any:
    def thunk() -> gym.Env:
        fixed_field = np.load(f"./RL/new_maps/a{idx:05}_grid.npy")
        fixed_loc = np.load(f"./RL/new_maps/a{idx:05}_loc.npy")
        env = gym.make(
            env_id,
            fixed_field=fixed_field,
            fixed_loc=fixed_loc,
            reduce=True,
            process_picked=True,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device)) # masks[i]=False -> invalid action -> logit=-inf
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs # self.logits在torch里会调用probs_to_logits；or 自身已经归一化; logits-logits.logsumexp(dim=-1, keepdims=True)
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)

    
        
class Agent(nn.Module):
    def __init__(
        self,
        args: Args,
        hidden_dim: int = 64,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.cls_token = (torch.ones(1, 1) * 22).cuda()
        
        self.type_embed = nn.Embedding(21+1+1, embed_dim) # [cls]->22, 0~20, (-1)->21
        self.rel_row_pos_embed = nn.Embedding(23, embed_dim)
        self.rel_col_pos_embed = nn.Embedding(23, embed_dim)
        self.abs_row_pos_embed = nn.Embedding(12, embed_dim)
        self.abs_col_pos_embed = nn.Embedding(12, embed_dim)
        self.bagn_embed = nn.Embedding(4, embed_dim)
        
        self.transformer = Transformer(embed_dim, args.n_head, args.bias, args.drop_p, args.n_layer)
        
        self.critic = layer_init(nn.Linear(embed_dim, 1), std=1.0)
        self.actor = layer_init(nn.Linear(embed_dim, 144), std=0.01)
        
    def get_feature(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        # split obs
        obs_grid = x[:, :144] # (B, 144)
        rel_row = x[:, 144:144*2]
        rel_col = x[:, 144*2:144*3]
        bagn = x[:, 144*3:144*4] # (B, 144)
        abs_row, abs_col = x[:, -2:-1], x[:, -1:] 

        type_seq = torch.cat([self.cls_token.repeat(bs, 1), obs_grid], dim=1).long()
        type_embeddings = self.type_embed(type_seq) # (B, 145, n_embed)
        
        abs_row_embeddings = self.abs_row_pos_embed(abs_row.long()) # (B, 1, n_embed)
        abs_col_embeddings = self.abs_col_pos_embed(abs_col.long())
        rel_row_embeddings = self.rel_row_pos_embed(rel_row.long())
        rel_col_embeddings = self.rel_col_pos_embed(rel_col.long())
        
        row_embeddings = torch.cat([abs_row_embeddings, rel_row_embeddings], dim=1)
        col_embeddings = torch.cat([abs_col_embeddings, rel_col_embeddings], dim=1)
        
        bagn_embeddings = self.bagn_embed(bagn.long())
        
        embeddings = type_embeddings + row_embeddings + col_embeddings
        embeddings[:, 1:] = embeddings[:, 1:] + bagn_embeddings
        
        # action_masks: 0 if unavailable
        action_mask = torch.cat([torch.ones(bs, 1).to(device), action_mask], dim=-1)
        attn_masks = action_mask.unsqueeze(1).expand(-1, 145, -1).unsqueeze(1)
        
        return self.transformer(embeddings, attn_masks) # (B, 145, n_embed)
        
    
    def get_value(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        h = self.get_feature(x, action_mask)
        return self.critic(h[:, 0])
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        action: Union[torch.Tensor]=None
    ) -> Tuple[torch.Tensor,...]:
        # print(x.shape, action_mask.shape)
        h = self.get_feature(x, action_mask) # (B, 145, n_embed)
        query = h[:, 0]
        keys = h[:, 1:]
        logits = torch.einsum("bd, bnd -> bn", query, keys) # (B, 144)
    
        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(query), probs.probs
    
    def get_eval_action(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        h = self.get_feature(x, action_mask) # (B, 145, n_embed)
        query = h[:, 0]
        keys = h[:, 1:]
        logits = torch.einsum("bd, bnd -> bn", query, keys) # (B, 144)
        logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e8).to(device))
        # print(F.softmax(logits, dim=-1))
        return torch.argmax(logits, dim=-1)

    
if __name__ == "__main__":
    import grid_env
    
    args = tyro.cli(Args)
    # print(args.chosen_maps)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    



    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.share_cnn:
        agent = Agent(args, args.hidden_dim, embed_dim=args.embed_dim).to(device)
    else:
        assert 0
        
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.resume_path != "":
        print(f"Load param from {args.resume_path}")
        if args.resume_path.endswith(".pt"):
            loaded_params = torch.load(args.resume_path)
        # 遍历args.resume，根据split('_')找最大数字
        elif "ppo.pt" in os.listdir(args.resume_path):
            loaded_params = torch.load(os.path.join(args.resume_path, "ppo.pt"))
        else:
            save_step_lst = [int(x[:-3].split("_")[1]) for x in os.listdir(args.resume_path) if "ppo" in x]
            max_step = max(save_step_lst)
            print(f"Load param from {os.path.join(args.resume_path, f'ppo_{max_step}.pt')}")
            loaded_params = torch.load(os.path.join(args.resume_path, f"ppo_{max_step}.pt"))

        
        agent.load_state_dict(loaded_params)
        
    # agent.eval()
    sim_env = gym.make(
        "grid_env/GoGridWorld-v0",
        fixed_field=False,
        fixed_loc=False,
        reduce=True,
        process_picked=True,
    )
    # agent = MCTS(agent, sim_env, args)
    
    score_lst = []
    for idx in range(100):
        fixed_field = np.load(f"./RL/new_maps/a{idx:05}_grid.npy")
        fixed_loc = np.load(f"./RL/new_maps/a{idx:05}_loc.npy")
        go_env = gym.make(
            "grid_env/GoGridWorld-v0",
            fixed_field=fixed_field.copy(),
            fixed_loc=fixed_loc.copy(),
            reduce=True,
            process_picked=True,
        )
        best_plan = None
        best_score = -float('inf')
        next_obs, infos = go_env.reset()
        next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
        while True:
            action_mask = torch.Tensor(infos['avail_actions']).unsqueeze(0).to(device)
            action = agent.get_eval_action(next_obs, action_mask)
            next_obs, reward, terminated, truncated, infos = go_env.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
            if terminated or truncated:
                if infos["score"] > best_score:
                    best_score = infos["score"]
                    best_path = infos["path"]
                break
                
        go_eval_envs = gym.vector.SyncVectorEnv(
            [
                make_eval_env(
                    args.env_id,
                    idx,
                    args,
                )
                for _ in range(args.num_simulations)
            ],
        )
        next_obs, infos = go_eval_envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        for _ in range(144):
            action_mask = torch.Tensor(np.stack(infos["avail_actions"])).to(device)
            with torch.no_grad():
                action = agent.get_action_and_value(next_obs, action_mask)[0]
            next_obs, reward, terminated, truncated, infos = go_eval_envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
        assert np.all(terminated)
        for info in infos["final_info"]:
            if info["score"] > best_score:
                best_score = info["score"]
                best_path = info["path"]
        best_plan = go_env.get_plan(best_path)

        eval_env = gym.make(
            "grid_env/EvalGridWorld-v0",
            fixed_field=fixed_field.copy(),
            fixed_loc=fixed_loc.copy(),
        )
        next_obs, infos = eval_env.reset()
        while True:
            next_obs, reward, terminated, truncated, infos = eval_env.step(best_plan.pop(0))
            if terminated or truncated:
                print(f"Score in eval env {idx}:", infos['score'])
                score_lst.append(infos['score'])
                break
    print(np.mean(score_lst))