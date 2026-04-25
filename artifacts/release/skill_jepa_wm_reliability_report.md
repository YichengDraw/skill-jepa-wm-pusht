# Skill-JEPA-WM Push-T Reliability Report

## Verdict

Use task-aligned coverage success as the primary Push-T metric. The tracked debug and legacy locked artifacts are trajectory-goal diagnostics and do not support a standard Push-T task-success claim.

## Legacy Locked Artifact Re-Score

| Method | Coverage success | Goal-state success | Unique episodes | Mean sampled-state distance | Mean latency |
|---|---:|---:|---:|---:|---:|
| flat | 0.00 | 0.07 | 1 | 355.30 | 0.564s |
| hierarchical | 0.00 | 0.07 | 1 | 264.43 | 0.249s |

The old `success_rate` column measured sampled-trajectory goal-state success, not standard Push-T coverage success.
Phase A fresh eval: 1 sampled pairs, 1 unique episodes, requested_split=test, actual_split=test, subgoal_scope=train, goal_mode=trajectory, task_success_claim_supported=False.

## Phase A Fresh Eval

| Method | Fixed-task coverage diagnostic | Goal-state success | Mean sampled-state distance | Mean latency |
|---|---:|---:|---:|---:|
| flat | 0.00 | 0.00 | 321.15 | 0.319s |
| hierarchical | 0.00 | 0.00 | 464.56 | 0.131s |

## Figures

- Architecture: `docs/architecture/model_pipeline.svg`
- Experiment flow: `docs/architecture/experiment_flow.svg`
- Coverage vs goal-state plot: `artifacts/release/plots/coverage_vs_goal_state_success.png`
- Paired distance plot: `artifacts/release/plots/paired_state_distance.png`
- Latency plot: `artifacts/release/plots/planning_latency.png`
- Rollout montage: `artifacts/release/plots/rollout_montage.png`
