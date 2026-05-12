from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import shimmy  # noqa: F401  # Registers dm_control Gymnasium environments.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ABRSCode.abrs import ABRSRewardModel
from utils.buffers import ReplayBuffer


def object_array(items: list[np.ndarray]) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def save_beta_snapshots(
    path: Path,
    steps: list[int],
    bucket_ids: list[np.ndarray],
    alphas: list[np.ndarray],
    betas: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        steps=np.asarray(steps, dtype=np.int64),
        bucket_ids=object_array(bucket_ids),
        alphas=object_array(alphas),
        betas=object_array(betas),
    )


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic is set"""
    cuda: bool = True
    """if toggled, cuda/mps will be enabled when available"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RLBase"
    """the wandb project name"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    env_id: str = "dm_control/cartpole-swingup_sparse-v0"
    """the environment id of the task"""
    total_timesteps: int = 500000
    """total timesteps of the experiment"""
    num_envs: int = 1
    """the number of parallel environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size sampled from replay memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of policy updates"""
    target_network_frequency: int = 1
    """the frequency of target network updates"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    abrs_lambda: float = 1.0
    """overall scale for ABRS auxiliary reward"""
    abrs_eta: float = 1.0
    """relative scale of uncertainty reward"""
    abrs_top_k_percent: float = 20.0
    """completed trajectories in the top K percent are high quality"""
    abrs_credit_gamma_w: float = 0.99
    """backward credit discount for high-quality trajectories"""
    abrs_hash_bits: int = 16
    """number of SimHash bits"""
    abrs_norm_clip: float = 5.0
    """clip range after running observation normalization; non-positive disables clipping"""
    abrs_return_tie_tol: float = 1e-8
    """return tolerance used when handling tied trajectory returns"""
    abrs_save_beta_snapshots: bool = True
    """whether to save periodic bucket-level Beta(alpha, beta) snapshots"""
    abrs_beta_snapshot_frequency: int = 0
    """frequency in environment steps for saving Beta snapshots; 0 means auto"""
    abrs_beta_snapshot_count: int = 100
    """target number of Beta snapshots when frequency is auto"""
    abrs_beta_snapshot_min_count: float = 1.0
    """only save buckets whose high+low count is at least this value"""
    abrs_reward_state: Literal["next", "current"] = "next"
    """state used for ABRS reward: next matches transition-arrival shaping"""
    abrs_log_frequency: int = 1000
    """frequency for ABRS diagnostic logging"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # dm_control tasks expose Dict observations and float64 arrays through Shimmy.
        # SAC and the replay buffer expect one flat float32 Box observation.
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.DtypeObservation(env, np.float32)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def get_final_observation(infos: dict, idx: int, fallback: np.ndarray) -> np.ndarray:
    final_obs = infos.get("final_obs", None)
    final_obs_mask = infos.get("_final_obs", None)
    if final_obs is None:
        final_obs = infos.get("final_observation", None)
        final_obs_mask = infos.get("_final_observation", None)

    if final_obs is not None and (final_obs_mask is None or final_obs_mask[idx]):
        obs = final_obs[idx]
        if obs is not None:
            return np.asarray(obs, dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    run_dir = Path("runs") / run_name
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    writer.add_text("device", str(device))

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    abrs = ABRSRewardModel(
        envs.single_observation_space.shape,
        hash_bits=args.abrs_hash_bits,
        top_k_percent=args.abrs_top_k_percent,
        credit_gamma_w=args.abrs_credit_gamma_w,
        norm_clip=args.abrs_norm_clip,
        return_tie_tol=args.abrs_return_tie_tol,
        seed=args.seed + 10007,
    )
    beta_snapshot_path = run_dir / "abrs_beta_snapshots.npz"
    beta_snapshot_steps: list[int] = []
    beta_snapshot_bucket_ids: list[np.ndarray] = []
    beta_snapshot_alphas: list[np.ndarray] = []
    beta_snapshot_betas: list[np.ndarray] = []
    beta_snapshot_frequency = args.abrs_beta_snapshot_frequency
    if args.abrs_save_beta_snapshots and beta_snapshot_frequency <= 0:
        beta_snapshot_frequency = max(1, args.total_timesteps // max(1, args.abrs_beta_snapshot_count))

    def record_beta_snapshot(step: int) -> None:
        bucket_ids, alphas, betas = abrs.beta_parameters(min_count=args.abrs_beta_snapshot_min_count)
        beta_snapshot_steps.append(int(step))
        beta_snapshot_bucket_ids.append(bucket_ids)
        beta_snapshot_alphas.append(alphas)
        beta_snapshot_betas.append(betas)
        save_beta_snapshots(
            beta_snapshot_path,
            beta_snapshot_steps,
            beta_snapshot_bucket_ids,
            beta_snapshot_alphas,
            beta_snapshot_betas,
        )
        writer.add_scalar("abrs/beta_snapshot_buckets", len(bucket_ids), step)

    if args.abrs_save_beta_snapshots:
        writer.add_text("abrs/beta_snapshot_file", str(beta_snapshot_path))
        writer.add_scalar("abrs/beta_snapshot_frequency", beta_snapshot_frequency, 0)

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    episode_states = [[obs[idx].copy()] for idx in range(envs.num_envs)]
    episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
    last_abrs_success = torch.tensor(0.0)
    last_abrs_uncertainty = torch.tensor(0.0)
    actor_loss = torch.tensor(float("nan"), device=device)
    alpha_loss = torch.tensor(float("nan"), device=device)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if done:
                real_next_obs[idx] = get_final_observation(infos, idx, next_obs[idx])
            elif truncations[idx]:
                real_next_obs[idx] = get_final_observation(infos, idx, next_obs[idx])

        final_info = infos.get("final_info", {})
        if "episode" in final_info:
            for idx, has_episode in enumerate(final_info["_episode"]):
                if has_episode:
                    episodic_return = final_info["episode"]["r"][idx]
                    episodic_length = final_info["episode"]["l"][idx]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        for idx in range(envs.num_envs):
            episode_returns[idx] += float(rewards[idx])
            episode_states[idx].append(real_next_obs[idx].copy())
            if dones[idx]:
                update_info = abrs.update_trajectory(np.asarray(episode_states[idx], dtype=np.float32), episode_returns[idx])
                writer.add_scalar("abrs/trajectory_return", update_info.env_return, global_step)
                writer.add_scalar("abrs/high_quality", float(update_info.is_high_quality), global_step)
                writer.add_scalar("abrs/rank", update_info.rank, global_step)
                writer.add_scalar("abrs/top_count", update_info.top_count, global_step)
                writer.add_scalar("abrs/unique_buckets", update_info.unique_buckets, global_step)
                writer.add_scalar(
                    "abrs/high_quality_ratio",
                    abrs.high_quality_episodes / max(1, len(abrs.episode_returns)),
                    global_step,
                )
                episode_states[idx] = [next_obs[idx].copy()]
                episode_returns[idx] = 0.0

        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                abrs_states = data.next_observations if args.abrs_reward_state == "next" else data.observations
                success_reward, uncertainty_reward = abrs.compute_bonus(abrs_states)
                shaped_rewards = data.rewards.flatten() + args.abrs_lambda * (
                    success_reward + args.abrs_eta * uncertainty_reward
                )
                last_abrs_success = success_reward.mean().detach().cpu()
                last_abrs_uncertainty = uncertainty_reward.mean().detach().cpu()

                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = shaped_rewards + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("abrs/success_reward", last_abrs_success.item(), global_step)
                writer.add_scalar("abrs/uncertainty_reward", last_abrs_uncertainty.item(), global_step)
                writer.add_scalar("abrs/shaped_reward_mean", shaped_rewards.mean().item(), global_step)
                writer.add_scalar("abrs/env_reward_mean", data.rewards.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            if args.abrs_log_frequency > 0 and global_step % args.abrs_log_frequency == 0:
                writer.add_scalar("abrs/high_quality_episodes", abrs.high_quality_episodes, global_step)
                writer.add_scalar("abrs/low_quality_episodes", abrs.low_quality_episodes, global_step)
                writer.add_scalar("abrs/total_episodes", len(abrs.episode_returns), global_step)

        if (
            args.abrs_save_beta_snapshots
            and beta_snapshot_frequency > 0
            and global_step > 0
            and global_step % beta_snapshot_frequency == 0
        ):
            record_beta_snapshot(global_step)

    if args.abrs_save_beta_snapshots:
        final_step = max(args.total_timesteps - 1, 0)
        if not beta_snapshot_steps or beta_snapshot_steps[-1] != final_step:
            record_beta_snapshot(final_step)
    envs.close()
    writer.close()
