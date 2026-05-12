from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict

import numpy as np
import torch


@dataclass
class TrajectoryUpdateInfo:
    env_return: float
    rank: int
    top_count: int
    is_high_quality: bool
    num_episodes: int
    unique_buckets: int


class RunningMeanStd:
    """Running observation normalizer used before SimHash projection."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        if x.ndim == len(self.mean.shape):
            x = x.reshape(1, *self.mean.shape)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x: np.ndarray, clip: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        if clip > 0:
            normalized = np.clip(normalized, -clip, clip)
        return normalized.astype(np.float32)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = float(total_count)


class ABRSRewardModel:
    """Beta-posterior reward shaping with SimHash state aggregation."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        *,
        hash_bits: int = 64,
        top_k_percent: float = 20.0,
        credit_gamma_w: float = 0.99,
        norm_clip: float = 5.0,
        return_tie_tol: float = 1e-8,
        seed: int = 1,
    ):
        if not 0 < top_k_percent <= 100:
            raise ValueError("top_k_percent must be in (0, 100].")
        if not 0 < credit_gamma_w <= 1:
            raise ValueError("credit_gamma_w must be in (0, 1].")
        if hash_bits <= 0:
            raise ValueError("hash_bits must be positive.")

        self.observation_shape = tuple(observation_shape)
        self.obs_dim = int(np.prod(self.observation_shape))
        self.hash_bits = int(hash_bits)
        self.top_k_percent = float(top_k_percent)
        self.credit_gamma_w = float(credit_gamma_w)
        self.norm_clip = float(norm_clip)
        self.return_tie_tol = float(return_tie_tol)

        rng = np.random.default_rng(seed)
        self.projection = rng.standard_normal((self.obs_dim, self.hash_bits)).astype(np.float32)
        self.obs_rms = RunningMeanStd((self.obs_dim,))

        self.high_counts: DefaultDict[int, float] = defaultdict(float)
        self.low_counts: DefaultDict[int, float] = defaultdict(float)
        self.episode_returns: list[float] = []
        self.high_quality_episodes = 0
        self.low_quality_episodes = 0
        self.last_update: TrajectoryUpdateInfo | None = None

    def update_trajectory(self, states: np.ndarray, env_return: float) -> TrajectoryUpdateInfo:
        """Classify one completed trajectory online and update bucket counts."""
        flat_states = self._flatten_obs(states)
        self.obs_rms.update(flat_states)

        env_return = float(env_return)
        num_episodes = len(self.episode_returns) + 1
        top_count = max(1, int(np.ceil(num_episodes * self.top_k_percent / 100.0)))
        rank, is_high_quality = self._classify_return(env_return, top_count)
        self.episode_returns.append(env_return)

        keys = self.hash_batch(flat_states)
        if is_high_quality:
            self.high_quality_episodes += 1
            terminal_index = len(keys) - 1
            for t, key in enumerate(keys):
                self.high_counts[key] += self.credit_gamma_w ** (terminal_index - t)
        else:
            self.low_quality_episodes += 1
            for key in keys:
                self.low_counts[key] += 1.0

        info = TrajectoryUpdateInfo(
            env_return=float(env_return),
            rank=rank,
            top_count=top_count,
            is_high_quality=is_high_quality,
            num_episodes=num_episodes,
            unique_buckets=self.num_unique_buckets,
        )
        self.last_update = info
        return info

    def _classify_return(self, env_return: float, top_count: int) -> tuple[int, bool]:
        """Online top-K% classification with explicit handling for tied sparse returns."""
        if not self.episode_returns:
            return 1, True

        returns = np.asarray(self.episode_returns, dtype=np.float64)
        higher_count = int(np.sum(returns > env_return + self.return_tie_tol))
        tied_count = int(np.sum(np.abs(returns - env_return) <= self.return_tie_tol))

        best_rank = higher_count + 1
        worst_rank = higher_count + tied_count + 1
        if worst_rank <= top_count:
            return best_rank, True
        if best_rank > top_count:
            return best_rank, False

        # The top-K boundary cuts through a group of tied returns. In sparse-reward
        # tasks this prevents every zero-return trajectory from becoming "high".
        return best_rank, self.high_quality_episodes < top_count

    def compute_bonus(self, observations: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return success-tendency and uncertainty rewards for a batch of states."""
        if isinstance(observations, torch.Tensor):
            device = observations.device
            obs_np = observations.detach().cpu().numpy()
        else:
            device = torch.device("cpu")
            obs_np = np.asarray(observations)

        keys = self.hash_batch(obs_np)
        high = np.fromiter((self.high_counts.get(key, 0.0) for key in keys), dtype=np.float32, count=len(keys))
        low = np.fromiter((self.low_counts.get(key, 0.0) for key in keys), dtype=np.float32, count=len(keys))

        alpha = high + 1.0
        beta = low + 1.0
        denom = alpha + beta
        success = alpha / denom
        variance = (alpha * beta) / (np.square(denom) * (denom + 1.0))
        uncertainty = success * np.sqrt(np.maximum(variance, 0.0))

        success_t = torch.as_tensor(success, dtype=torch.float32, device=device)
        uncertainty_t = torch.as_tensor(uncertainty, dtype=torch.float32, device=device)
        return success_t, uncertainty_t

    def beta_parameters(self, min_count: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return bucket IDs and their current Beta(alpha, beta) parameters."""
        keys = sorted(set(self.high_counts.keys()) | set(self.low_counts.keys()))
        bucket_ids: list[int] = []
        alphas: list[float] = []
        betas: list[float] = []
        for key in keys:
            high = self.high_counts.get(key, 0.0)
            low = self.low_counts.get(key, 0.0)
            if high + low < min_count:
                continue
            bucket_ids.append(key)
            alphas.append(high + 1.0)
            betas.append(low + 1.0)
        return (
            np.asarray(bucket_ids, dtype=object),
            np.asarray(alphas, dtype=np.float32),
            np.asarray(betas, dtype=np.float32),
        )

    def hash_batch(self, observations: np.ndarray) -> list[int]:
        flat_obs = self._flatten_obs(observations)
        norm_obs = self.obs_rms.normalize(flat_obs, self.norm_clip)
        bits = (norm_obs @ self.projection) >= 0.0
        return [self._bits_to_int(row) for row in bits]

    @property
    def num_unique_buckets(self) -> int:
        return len(set(self.high_counts.keys()) | set(self.low_counts.keys()))

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        key = 0
        for bit_index, bit in enumerate(bits):
            if bit:
                key |= 1 << bit_index
        return key

    def _flatten_obs(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape == self.observation_shape:
            obs = obs.reshape(1, -1)
        else:
            obs = obs.reshape(obs.shape[0], -1)
        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"Expected flattened obs dim {self.obs_dim}, got {obs.shape[1]}.")
        return obs
