from __future__ import annotations

from dataclasses import dataclass

import torch

from skill_jepa.losses import negative_log_gaussian


@dataclass
class HighLevelPlan:
    first_skill: torch.Tensor
    subgoal: torch.Tensor
    skill_sequence: torch.Tensor
    costs: torch.Tensor


class HighLevelCEMPlanner:
    def __init__(
        self,
        skill_wm,
        skill_prior,
        skill_dim: int,
        horizon: int = 4,
        population: int = 256,
        elites: int = 32,
        iterations: int = 5,
        skill_penalty: float = 0.1,
        prior_penalty: float = 0.1,
    ) -> None:
        self.skill_wm = skill_wm
        self.skill_prior = skill_prior
        self.skill_dim = skill_dim
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iterations = iterations
        self.skill_penalty = skill_penalty
        self.prior_penalty = prior_penalty

    @torch.no_grad()
    def plan(self, current_z: torch.Tensor, goal_z: torch.Tensor) -> HighLevelPlan:
        device = current_z.device
        mean, std = self._prior_rollout_stats(current_z)
        best_sequence = None
        all_costs = []
        for _ in range(self.iterations):
            skills = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                self.horizon, self.population, self.skill_dim, device=device
            )
            rollout = current_z.unsqueeze(0).expand(self.population, -1)
            prior_cost = torch.zeros(self.population, device=device)
            magnitude_cost = torch.zeros(self.population, device=device)
            for step in range(self.horizon):
                step_skill = skills[step]
                prior_mean, prior_logvar = self.skill_prior(rollout)
                prior_cost = prior_cost + negative_log_gaussian(step_skill, prior_mean, prior_logvar)
                magnitude_cost = magnitude_cost + step_skill.pow(2).mean(dim=-1)
                rollout = self.skill_wm(rollout, step_skill)
            final_cost = (rollout - goal_z.unsqueeze(0)).pow(2).mean(dim=-1)
            cost = final_cost + self.prior_penalty * prior_cost + self.skill_penalty * magnitude_cost
            elite_idx = cost.topk(self.elites, largest=False).indices
            elite_skills = skills[:, elite_idx]
            mean = elite_skills.mean(dim=1)
            std = elite_skills.std(dim=1).clamp_min(1e-4)
            best_idx = cost.argmin()
            best_sequence = skills[:, best_idx]
            all_costs.append(cost[best_idx])
        subgoal = self.skill_wm(current_z.unsqueeze(0), best_sequence[0:1]).squeeze(0)
        return HighLevelPlan(
            first_skill=best_sequence[0],
            subgoal=subgoal,
            skill_sequence=best_sequence,
            costs=torch.stack(all_costs),
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
