# Skill-JEPA-WM Push-T Reliability Report

## Verdict

Use coverage success as the primary Push-T metric. The legacy locked artifact does not support a standard Push-T success claim: both flat and hierarchical have 0.00 coverage success.

## Legacy Locked Artifact Re-Score

| Method | Coverage success | Goal-state success | Unique episodes | Mean sampled-state distance | Mean latency |
|---|---:|---:|---:|---:|---:|
| flat | 0.00 | 0.07 | 1 | 355.30 | 0.564s |
| hierarchical | 0.00 | 0.07 | 1 | 264.43 | 0.249s |

The old `success_rate` column measured pose-to-sampled-goal success, not standard Push-T coverage success.
Phase A fresh eval: 1 sampled pairs, 1 unique episodes, requested_split=test, actual_split=test, subgoal_scope=train.

## Phase A Fresh Eval

| Method | Coverage success | Goal-state success | Mean sampled-state distance | Mean latency |
|---|---:|---:|---:|---:|
| flat | 0.00 | 0.00 | 321.15 | 0.292s |
| hierarchical | 0.00 | 0.00 | 464.56 | 0.129s |

## Figures

- Architecture: `docs/architecture/model_pipeline.svg`
- Experiment flow: `docs/architecture/experiment_flow.svg`
- Coverage vs goal-state plot: `artifacts/release/plots/coverage_vs_goal_state_success.png`
- Paired distance plot: `artifacts/release/plots/paired_state_distance.png`
- Latency plot: `artifacts/release/plots/planning_latency.png`
- Rollout montage: `artifacts/release/plots/rollout_montage.png`
