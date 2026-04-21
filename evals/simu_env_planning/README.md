# Simu_env_planning Evaluation

> **See also**: [Main README](../../README.md) for training and general usage.

## Overview

Goal-conditioned trajectory optimization: we optimize over the action space to minimize the planning cost $C$:

$$
C(a, s_0, s_g) = \sum_{t=0}^H \| E_{\theta}(s_g) - P_{\theta}(\hat{z}_t, a_t) \|_2,
$$

where

$$
\hat{z}_0 = E_{\theta}(s_0)
$$

$$
\hat{z}_{t+1} = P_{\theta}(\hat{z}_t, a_t), \quad t=0, \dots, H-1.
$$

An **evaluation episode** is a pair $(s_0, s_g)$ (task definition) with the agent's plan, resulting in **Success** or **Failure**.

## Quick Start

**Interactive (single GPU)**:
```bash
python -m evals.main --fname configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml --debug
```

**Distributed (from login node)**:
```bash
python -m evals.main_distributed --fname configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml --account <account> --qos lowest --time 120
```

## Goal Sources

| Source | Description | Environments |
|--------|-------------|--------------|
| `dset` | Initial/goal states sampled from validation set | Push-T, RoboCasa, DROID |
| `expert` | Expert policy provides goal trajectory | Metaworld |
| `random_state` | 2D positions sampled via simulator | Maze, Wall |

The `random_actions` variant steps Gaussian-sampled actions from the initial state.

## Episode Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate** | % success across `cfg.meta.eval_episodes` episodes |
| `ep_end_dist` | Distance to goal at episode end |
| `reward` | Average cumulative reward (Metaworld, Maze, Push-T, Wall) |
| `success_dist` | Arm reached target location (Metaworld only) |
| `total_emb_l2` | L2 distance between agent/expert visual embeddings |
| `total_lpips` | LPIPS between agent/expert visual decodings |

**DROID offline only**: `end_distance_xyz`, `end_distance_orientation`, `end_distance_closure`

## Optional Episode Plots

Enable with `cfg.logging.optional_plots`. Generated via `evals/simu_env_planning/planning/episode_plot_utils.py`.

The original upstream image examples are not tracked in this focused Push-T release. Generate fresh plot assets from a local run when working with RoboCasa, DROID, or decoding-heavy evals.

## Architecture

### Distributed Episodes

Episodes are parallelized across GPUs via `main_distributed_episodes_eval()` in `evals/simu_env_planning/eval.py`.

### Planning Components

| Component | Location |
|-----------|----------|
| Planners (CEM, NeverGrad) | `evals/simu_env_planning/planning/planner.py` |
| Planning objectives | `evals/simu_env_planning/planning/objectives.py` |
| Goal sources | `evals/simu_env_planning/planning/plan_evaluator.py:set_episode()` |

### Model Wrapper

Default wrapper: `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.EncPredWM`

Initializes the **world model** (encoder, predictor, optional modules).

Required by `evals/simu_env_planning/eval.py`:
- **data_preprocessor**: Normalizes/denormalizes actions and proprioception
- **validation set**: Provides action/proprio dimensions, normalization stats, and task definition data (if `goal_source == dset`)

## Eval Configs

| Type | Location |
|------|----------|
| Full example configs | `configs/evals/simu_env_planning/` |
| Config templates | `evals/simu_env_planning/base_configs/` |

## Grid Evaluation

Run evaluations across multiple hyperparameters:

```bash
python -m evals.simu_env_planning.run_eval_grid \
    --env metaworld \
    --config configs/dump_online_evals/vjepa_wm/dset_ng_L2.yaml
```

Generates configs for all combinations of variants, planners (`ng`/`cem`), objectives (`L1`/`L2`), and epochs.

**Visualization notebook**: `app/plan_common/plot/logs_planning_joint.ipynb`
