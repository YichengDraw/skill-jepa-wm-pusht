# Skill-JEPA-WM Push-T Reliability Report

## Verdict

Use task-aligned coverage success as the primary Push-T metric. The tracked debug and legacy locked artifacts are trajectory-goal diagnostics and do not support a standard Push-T task-success claim.

## Legacy Locked Artifact Re-Score

| Method | Coverage success | Goal-state diagnostic | Unique episodes | Mean sampled-state distance | Mean latency |
|---|---:|---:|---:|---:|---:|
| flat | 0.00 | 0.07 | 1 | 355.30 | 0.564s |
| hierarchical | 0.00 | 0.07 | 1 | 264.43 | 0.249s |

Legacy pre-rescore outputs used `success_rate` for sampled-trajectory goal-state success; current tracked eval exports use coverage-specific names.
Phase A fresh eval: 1 sampled pairs, 1 unique episodes, requested_split=test, actual_split=test, subgoal_scope=train, goal_mode=trajectory, task_success_claim_supported=False, under_sampled=True, provenance_warnings=6.

## Phase A Fresh Eval

| Method | Coverage diagnostic | Goal-state diagnostic | Mean sampled-state distance | Mean latency |
|---|---:|---:|---:|---:|
| flat | 0.00 | 0.00 | 321.15 | 0.561s |
| hierarchical | 0.00 | 0.00 | 464.56 | 0.227s |

## Figures

- Architecture: `docs/architecture/model_pipeline.svg`
- Experiment flow: `docs/architecture/experiment_flow.svg`
- Coverage vs goal-state plot: `artifacts/release/plots/coverage_vs_goal_state_success.png`
- Paired distance plot: `artifacts/release/plots/paired_state_distance.png`
- Latency plot: `artifacts/release/plots/planning_latency.png`
- Rollout montage: `artifacts/release/plots/rollout_montage.png`
