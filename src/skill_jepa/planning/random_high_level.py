from __future__ import annotations

import torch

from .high_level_cem import HighLevelPlan


class RandomHighLevelPlanner:
    def __init__(self, skill_wm, skill_prior, skill_dim: int, horizon: int = 4) -> None:
        self.skill_wm = skill_wm
        self.skill_prior = skill_prior
        self.skill_dim = skill_dim
        self.horizon = horizon

    @torch.no_grad()
    def plan(self, current_z: torch.Tensor, goal_z: torch.Tensor) -> HighLevelPlan:
        del goal_z
        means, stds = self._prior_rollout_stats(current_z)
        skill_sequence = means + stds * torch.randn(self.horizon, self.skill_dim, device=current_z.device)
        subgoal = self.skill_wm(current_z.unsqueeze(0), skill_sequence[0:1]).squeeze(0)
        return HighLevelPlan(
            first_skill=skill_sequence[0],
            subgoal=subgoal,
            skill_sequence=skill_sequence,
            costs=torch.zeros(self.horizon, device=current_z.device),
        )

    @torch.no_grad()
    def _prior_rollout_stats(self, current_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means = []
        stds = []
        rollout = current_z.unsqueeze(0)
        for _ in range(self.horizon):
            prior_mean, prior_logvar = self.skill_prior(rollout)
            step_mean = prior_mean.squeeze(0)
            step_std = torch.exp(0.5 * prior_logvar.squeeze(0)).clamp(0.05, 2.0)
            means.append(step_mean)
            stds.append(step_std)
            rollout = self.skill_wm(rollout, prior_mean)
        return torch.stack(means, dim=0), torch.stack(stds, dim=0)
