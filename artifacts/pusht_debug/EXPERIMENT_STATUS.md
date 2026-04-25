# Push-T Debug Status

Date: 2026-04-11

Historical note: the success and distance values in this debug status are sampled-trajectory goal-state diagnostics. They are not standard Push-T coverage-success evidence.

## What Was Completed

- Switched the frozen V-JEPA2 cache path from duplicated single frames to causal 2-frame clips.
- Fixed the online Push-T evaluator to:
  - replan from real observations,
  - use cache-consistent start and goal latents,
  - use a fair control horizon (`max_episode_steps = 32` for `goal_gap = 24`),
  - report sampled-state distance instead of mixing in velocity terms,
  - sample only reachable goals under the control budget,
  - reseed each eval episode deterministically so flat and hierarchical comparisons are order-independent,
  - support executing the first `M` primitive actions per plan.
- Rebuilt the cache and reran passive, low-level, joint, and online planning evals.

## Current Result

- No latent collapse on the joint model:
  - `skill_end = 0.0530`
  - `skill_roll = 0.0847`
  - `skill_std = 0.0295`
- The corrected online task sampler is valid:
  - expert replay sampled-goal success rate: `1.0`
  - expert replay sampled-state distance: `2.55`
- Best current debug result is the clip-based `K=4` model with `execute_actions_per_plan = 4`:
  - hierarchical sampled-goal success rate: `0.25`
  - hierarchical sampled-state distance: `254.61`
  - flat sampled-goal success rate: `0.0`
  - flat sampled-state distance: `459.92`
  - hierarchical beats flat on `75%` of eval episodes
- The 10%-label hierarchical method beats the labeled-only low-level baseline at the same label budget:
  - labeled-only flat baseline sampled-goal success rate: `0.0`
  - labeled-only flat baseline sampled-state distance: `793.81`
- The `K=2` ablation is no longer the recommended setting:
  - it introduced one-off successes earlier,
  - under the cache-consistent evaluator it is worse than the final `K=4, M=4` result.

## Remaining Limitation

- Absolute Push-T coverage success is unsupported by this 4-episode debug backup.
- The first complete-run criteria from the design spec should be treated as historical trajectory-diagnostic criteria.
- The main remaining gap is optimization quality at larger scale:
  - keep the `K=4, M=4` model as the current best setting,
  - scale the winning setting to a larger Push-T or DROID subset next.
