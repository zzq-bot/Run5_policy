import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from dotenv import load_dotenv  # pip install python-dotenv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from utils.normalize import NormalizeReward
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
    wandb_project_name: str = "Run5_RL"
    wandb_entity: str = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "grid_env/GoGridWorld-v0"

    num_steps: int = 12 * 12
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
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
    eval_maps = list(range(100))  # will not be reserved in wandb

    # The following params might be changed often
    resume_path: str = ""  # "ckpt/1112"

    total_timesteps: int = int(6e8)
    save_interval: int = int(1e7)
    eval_interval: int = int(1e7)
    aux_reward_coef: float = 1.0
    one_hot_pos: bool = True
    gravity_reward: bool = False
    train_size: int = 12
    train_types: int = 21

    force_all_char: bool = False # whether forcing every char_type occurs in a map
    use_eval: bool = False # train policy using evaluation maps or not
    num_envs: int = 200
    norm_rew: bool = False
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    embed_dim: int = 128
    n_head: int = 8
    bias: bool = True
    drop_p: float = 0.1
    n_layer: int = 6
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 4


def make_env(env_id: str, idx: int, args: Args) -> Any:
    def thunk() -> gym.Env:
        if args.use_eval and idx < 100:
            fixed_field = np.load(f"./RL/maps/a{idx:05}_grid.npy")
            fixed_loc = np.load(f"./RL/maps/a{idx:05}_loc.npy")
            env = gym.make(
                env_id,
                size=args.train_size,
                cls_n=args.train_types,
                fixed_field=fixed_field,
                fixed_loc=fixed_loc,
                use_aux_reward=args.use_aux_reward,
                aux_reward_coef=args.aux_reward_coef,
                reduce=args.reduce_field,
                process_picked=True,
                force_all_char=args.force_all_char,
            )
        else:
            env = gym.make(
                env_id,
                size=args.train_size,
                cls_n=args.train_types,
                fixed_field=False,
                fixed_loc=False,
                use_aux_reward=args.use_aux_reward,
                aux_reward_coef=args.aux_reward_coef,
                reduce=args.reduce_field,
                process_picked=True,
                force_all_char=args.force_all_char
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# def make_eval_env(env_id: str, idx: int, args: Args) -> Any:
#     def thunk() -> gym.Env:
#         fixed_field = np.load(f"./RL/maps/a{idx:05}_grid.npy")
#         fixed_loc = np.load(f"./RL/maps/a{idx:05}_loc.npy")
#         env = gym.make(
#             env_id,
#             fixed_field=fixed_field,
#             fixed_loc=fixed_loc,
#             use_aux_reward=args.use_aux_reward,
#             aux_reward_coef=args.aux_reward_coef,
#             reduce=args.reduce_field,
#             process_picked=True,
#         )
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         return env

#     return thunk


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def evaluate(eval_envs: gym.Env, agent: nn.Module, device: torch.device, global_step: int, writer: SummaryWriter) -> None:
    next_obs, infos = eval_envs.reset(seed=0)
    next_obs = torch.Tensor(next_obs).to(device)
    terminated = np.zeros(eval_envs.num_envs)

    record_returns = [0] * len(next_obs)
    while True:
        with torch.no_grad():
            action_mask = torch.Tensor(np.stack(infos["avail_actions"])).to(device)
            action = agent.get_eval_action(next_obs, action_mask)
        next_obs, reward, terminations, truncations, infos = eval_envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        terminated = np.logical_or(terminated, next_done)
        next_obs = torch.Tensor(next_obs).to(device)

        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if info and "episode" in info:
                    print(f"Eval: global_step={global_step}, idx={i} episodic_return={info['score']}, episodic_shaping_return={info['episode']['r']}")

                    record_returns[i] = info["score"]
                    if writer is not None:
                        writer.add_scalar("charts/eval_episodic_return", info["score"], global_step)
                        writer.add_scalar(
                            "charts/eval_episodic_shaping_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar("charts/eval_episodic_length", info["episode"]["l"], global_step)
        if np.all(terminated):
            break

    record_returns = np.array(record_returns)
    print(record_returns, np.mean(record_returns))
    return record_returns


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))  # masks[i]=False -> invalid action -> logit=-inf
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = (
            self.logits * self.probs
        )  # self.logits在torch里会调用probs_to_logits；or 自身已经归一化; logits-logits.logsumexp(dim=-1, keepdims=True)
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(
        self,
        args: Args,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.cls_token = (torch.ones(1, 1) * 22).cuda()

        self.type_embed = nn.Embedding(21 + 1 + 1, embed_dim)  # [cls]->22, 0~20, (-1)->21
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
        obs_grid = x[:, :144]  # (B, 144)
        rel_row = x[:, 144 : 144 * 2]
        rel_col = x[:, 144 * 2 : 144 * 3]
        bagn = x[:, 144 * 3 : 144 * 4]  # (B, 144)
        abs_row, abs_col = x[:, -2:-1], x[:, -1:]

        type_seq = torch.cat([self.cls_token.repeat(bs, 1), obs_grid], dim=1).long()
        type_embeddings = self.type_embed(type_seq)  # (B, 145, n_embed)

        abs_row_embeddings = self.abs_row_pos_embed(abs_row.long())  # (B, 1, n_embed)
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

        return self.transformer(embeddings, attn_masks)  # (B, 145, n_embed)

    def get_value(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        h = self.get_feature(x, action_mask)
        return self.critic(h[:, 0])

    def get_action_and_value(self, x: torch.Tensor, action_mask: torch.Tensor, action: Union[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        # print(x.shape, action_mask.shape)
        h = self.get_feature(x, action_mask)  # (B, 145, n_embed)
        query = h[:, 0]
        keys = h[:, 1:]
        logits = torch.einsum("bd, bnd -> bn", query, keys)  # (B, 144)

        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(query)

    def get_eval_action(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        h = self.get_feature(x, action_mask)  # (B, 145, n_embed)
        query = h[:, 0]
        keys = h[:, 1:]
        logits = torch.einsum("bd, bnd -> bn", query, keys)  # (B, 144)
        logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e8).to(device))
        # deterministic action
        return torch.argmax(logits, dim=-1)


if __name__ == "__main__":
    import grid_env

    args = tyro.cli(Args)
    # print(args.chosen_maps)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    print("current batch size:", args.minibatch_size)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"log/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    fixed_field = False
    if args.fixed_field:
        all_char_types = np.random.randint(0, 21, 36)
        fixed_field = -np.ones((12, 12), np.int32)
        empty_pos = np.argwhere(fixed_field == -1).tolist()  # len() = 144
        for char_type in all_char_types:
            for _ in range(4):
                pos = empty_pos.pop(np.random.randint(0, len(empty_pos)))
                fixed_field[pos[0], pos[1]] = char_type

        print(fixed_field)
    else:
        print("Random field will be generated for each episode.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args,
            )
            for i in range(args.num_envs)
        ],
    )

    if args.norm_rew:
        envs = NormalizeReward(envs, gamma=args.gamma)
    
    # eval_envs = gym.vector.SyncVectorEnv(
    #     [
    #         make_eval_env(
    #             args.env_id, map_idx, args,
    #         )
    #         for map_idx in args.eval_maps
    #     ],
    # )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    
    agent = Agent(args, embed_dim=args.embed_dim).to(device)
    

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.resume_path != "":
        print(f"Load param from {args.resume_path}")
        if "ppo.pt" in os.listdir(args.resume_path):
            loaded_params = torch.load(os.path.join(args.resume_path, "ppo.pt"))
        else:
            save_step_lst = [int(x[:-3].split("_")[1]) for x in os.listdir(args.resume_path) if "ppo" in x]
            max_step = max(save_step_lst)
            loaded_params = torch.load(os.path.join(args.resume_path, f"ppo_{max_step}.pt"))

        agent.load_state_dict(loaded_params, strict=False)
        missing_keys, unexpected_keys = agent.load_state_dict(loaded_params, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_masks = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    last_save_step = 0
    last_eval_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_action_masks = torch.Tensor(np.stack(infos["avail_actions"])).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            action_masks[step] = next_action_masks

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_masks[step])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_action_masks = torch.Tensor(np.stack(infos["avail_actions"])).to(device)

            if "final_info" in infos:
                log_episodic_return = []
                log_episodic_shaping_return = []
                log_episodic_real_length = []
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        log_episodic_return.append(info["score"])
                        log_episodic_shaping_return.append(info["episode"]["r"])
                        log_episodic_real_length.append(info["real_length"])
                if len(log_episodic_return) > 0:
                    log_episodic_return = np.array(log_episodic_return).mean()
                    log_episodic_shaping_return = np.array(log_episodic_shaping_return).mean()
                    log_episodic_real_length = np.array(log_episodic_real_length).mean()
                    print(
                        f"{args.exp_name}: global_step={global_step}, return={log_episodic_return:.3f}, shaping_return={log_episodic_shaping_return:.3f}, real_length={log_episodic_real_length:.1f}"
                    )
                    writer.add_scalar("charts/episodic_return", log_episodic_return, global_step)
                    writer.add_scalar("charts/episodic_shaping_return", log_episodic_shaping_return, global_step)
                    writer.add_scalar("charts/real_episodic_length", log_episodic_real_length, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_action_masks).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_action_masks[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(args.exp_name, "SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if (global_step - last_save_step) > args.save_interval:
            # save the model
            model_path = f"log/{run_name}/ppo_{global_step}.pt"
            torch.save(agent.state_dict(), model_path)

            last_save_step = global_step

    model_path = f"log/{run_name}/ppo.pt"
    torch.save(agent.state_dict(), model_path)

    envs.close()
    writer.close()
