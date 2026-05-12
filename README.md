# Adaptive Beta Reward Shaping (ABRS)

This folder contains the implementation of Adaptive Beta Reward Shaping (ABRS)
with a Soft Actor-Critic (SAC) training backbone. The current runnable example
targets the DeepMind Control Suite `cartpole-swingup_sparse` task through
Gymnasium and Shimmy.

An overview of the ABRS reward-shaping framework:

![](./ABRS.png)

ABRS maintains a state-level Beta posterior over high-quality and low-quality
trajectory evidence. During SAC critic-target construction, the replay buffer
keeps the raw environment reward, and ABRS recomputes auxiliary rewards online
from the current posterior:

```text
r_hat = r_e + abrs_lambda * (r_suc + abrs_eta * r_unc)
```

By default, the posterior is queried on `s_{t+1}`, so the auxiliary reward
scores the state reached by the current transition.

## Requirements

- Python 3.10+
- The parent RLBase project environment, including `utils.buffers.ReplayBuffer`
- Main Python packages: `gymnasium`, `shimmy[dm-control]`, `numpy`, `torch`,
  `tensorboard`, `tyro`, and `matplotlib`

If this folder is used inside the RLBase project, install dependencies from the
project-level environment file and run commands from the project root.

## Run ABRS

The main entry is `SAC_ABRS.py`, and the backbone algorithm is SAC.

Default ABRS + SAC run:

```bash
uv run python ABRSCode/SAC_ABRS.py
```

SAC-style ablation without ABRS auxiliary rewards:

```bash
uv run python ABRSCode/SAC_ABRS.py --abrs-lambda 0
```

Logs are saved to `runs/`. View them with:

```bash
tensorboard --logdir runs
```

## Key Arguments

- `--env-id`: Gymnasium environment ID. Default:
  `dm_control/cartpole-swingup_sparse-v0`.
- `--total-timesteps`: total environment interaction steps.
- `--seed`: random seed for NumPy, PyTorch, and the environment.
- `--num-envs`: number of vectorized environments. The script currently uses
  Gymnasium `SyncVectorEnv`.
- `--buffer-size`, `--batch-size`: replay-buffer capacity and sampled batch
  size.
- `--gamma`, `--tau`, `--policy-lr`, `--q-lr`: SAC discount, target smoothing,
  policy learning rate, and Q-network learning rate.
- `--autotune`, `--alpha`: entropy-temperature configuration.
- `--abrs-lambda`: overall ABRS auxiliary reward scale.
- `--abrs-eta`: uncertainty reward scale inside the ABRS bonus.
- `--abrs-top-k-percent`: percentage of completed trajectories classified as
  high-quality online.
- `--abrs-credit-gamma-w`: backward credit discount for states in high-quality
  trajectories.
- `--abrs-hash-bits`: number of SimHash bits used for state aggregation.
- `--abrs-norm-clip`: clipping range after running observation normalization.
- `--abrs-return-tie-tol`: tolerance for tied sparse trajectory returns.
- `--abrs-reward-state`: choose whether ABRS scores `next` or `current` states.
- `--abrs-save-beta-snapshots`: save periodic Beta-parameter snapshots.

## Beta Posterior Snapshots

Beta snapshots are enabled by default. When
`--abrs-beta-snapshot-frequency 0`, the script automatically uses:

```text
total_timesteps / abrs_beta_snapshot_count
```

Snapshot files are written under the corresponding run directory as:

```text
runs/<run_name>/abrs_beta_snapshots.npz
```

Each snapshot file stores:

```text
steps
bucket_ids
alphas
betas
```

Load a snapshot file with:

```python
import numpy as np

data = np.load("runs/<run_name>/abrs_beta_snapshots.npz", allow_pickle=True)
steps = data["steps"]
bucket_ids = data["bucket_ids"]
alphas = data["alphas"]
betas = data["betas"]
```

## Files

- `SAC_ABRS.py`: SAC training entry with ABRS reward shaping.
- `abrs.py`: SimHash state aggregation, online top-K trajectory ranking, and
  Beta-posterior reward computation.
- `plot_beta_snapshots.py`: utility for plotting Beta-distribution changes.
- `ABRS.png`: overview figure for the ABRS framework.
- `__init__.py`: package marker for `ABRSCode`.

## Implementation Notes

- DM Control observations are flattened and converted to `float32` before SAC
  and ABRS processing.
- High-quality trajectories update Beta alpha counts with discounted backward
  credit; low-quality trajectories update Beta beta counts.
- Sparse-return ties at the top-K boundary are handled explicitly so early
  zero-return trajectories do not all become high-quality examples.
- The script imports `ReplayBuffer` from the parent project module
  `utils.buffers`, so this folder is expected to sit under the RLBase project
  root or another project root that provides the same module.
