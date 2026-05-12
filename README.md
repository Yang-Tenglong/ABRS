# Asymmetric Beta-Posterior Reward Shaping (ABRS)

The code for the proposed Asymmetric Beta-Posterior Reward Shaping (ABRS)
algorithm.

[Paper Link to be updated]

An overview of the ABRS framework to shape rewards:

![](./ABRS.png)

This implementation uses SAC with two ABRS reward-shaping signals:
- Success reward from the mean of a state-level Beta posterior.
- Uncertainty reward from the variance of the same Beta posterior.

Both signals are computed from SimHash state aggregation and online trajectory
ranking. ABRS can be disabled for a SAC-style ablation, and the uncertainty term
can be disabled independently.

## Requirements

- Python 3.10+
- The parent RLBase project, since `SAC_ABRS.py` imports
  `utils.buffers.ReplayBuffer`.
- Main packages: `gymnasium`, `shimmy[dm-control]`, `numpy`, `torch`,
  `tensorboard`, `tyro`, and `matplotlib`.

All required packages can be installed from the project-level `pyproject.toml`:

```bash
uv sync
```

Run the commands below from the RLBase project root, with this folder named
`ABRSCode`.

## Run ABRS Algorithm

The main entry is `SAC_ABRS.py`, and the backbone algorithm is SAC.

ABRS + SAC:

```bash
uv run python ABRSCode/SAC_ABRS.py
```

SAC ablation without ABRS:

```bash
uv run python ABRSCode/SAC_ABRS.py --abrs-lambda 0
```

ABRS success reward only:

```bash
uv run python ABRSCode/SAC_ABRS.py --abrs-eta 0
```

Logs are saved to `runs/`. View with:

```bash
tensorboard --logdir runs
```

## Key Arguments

- `--exp-name`: experiment name for logging.
- `--env-id`: environment ID. Default:
  `dm_control/cartpole-swingup_sparse-v0`.
- `--seed`: random seed.
- `--total-timesteps`: total training steps.
- `--num-envs`: number of vectorized environments.
- `--buffer-size`, `--batch-size`: replay buffer and training batch sizes.
- `--policy-lr`, `--q-lr`: actor and critic optimizer learning rates.
- `--gamma`, `--tau`: SAC discount and target-network update coefficient.
- `--autotune`, `--alpha`: entropy-temperature configuration.
- `--abrs-lambda`: overall ABRS reward scale.
- `--abrs-eta`: uncertainty reward scale inside ABRS.
- `--abrs-top-k-percent`: top percentage of completed trajectories treated as
  high quality.
- `--abrs-credit-gamma-w`: backward credit discount for high-quality
  trajectories.
- `--abrs-hash-bits`: number of SimHash bits for state aggregation.
- `--abrs-reward-state`: use `next` or `current` states for ABRS rewards.
- `--abrs-save-beta-snapshots`: save Beta posterior snapshots to `runs/`.

## Files

- `SAC_ABRS.py`: training entry (SAC + ABRS).
- `abrs.py`: ABRS reward module with SimHash aggregation, online trajectory
  ranking, and Beta-posterior reward computation.
- `plot_beta_snapshots.py`: utility for plotting Beta-posterior changes.
- `ABRS.png`: overview figure for the ABRS framework.
- `__init__.py`: package marker for `ABRSCode`.
