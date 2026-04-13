from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LowLevelPlan:
    actions: torch.Tensor
    costs: torch.Tensor
    skill_consistency: torch.Tensor


class LowLevelCEMPlanner:
    def __init__(
        self,
        low_level_wm,
        action_chunk_encoder,
        action_dim: int,
        horizon: int = 4,
        population: int = 512,
        elites: int = 64,
        iterations: int = 5,
        skill_penalty: float = 1.0,
        action_penalty: float = 0.05,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        init_std: torch.Tensor | None = None,
        spatial_penalty: float = 1.0,
        global_penalty: float = 1.0,
    ) -> None:
        self.low_level_wm = low_level_wm
        self.action_chunk_encoder = action_chunk_encoder
        self.action_dim = action_dim
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iterations = iterations
        self.skill_penalty = skill_penalty
        self.action_penalty = action_penalty
        self.action_low = action_low
        self.action_high = action_high
        self.init_std = init_std
        self.spatial_penalty = spatial_penalty
        self.global_penalty = global_penalty

    @torch.no_grad()
    def plan(
        self,
        spatial_tokens: torch.Tensor,
        target_z: torch.Tensor,
        target_skill: torch.Tensor | None = None,
        target_s: torch.Tensor | None = None,
    ) -> LowLevelPlan:
        device = spatial_tokens.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        if self.init_std is None:
            std = torch.ones(self.horizon, self.action_dim, device=device)
        else:
            init_std = self.init_std.to(device)
            std = init_std.unsqueeze(0).expand(self.horizon, -1).clone()
        best_actions = None
        best_consistency = torch.tensor(0.0, device=device)
        costs = []
        for _ in range(self.iterations):
            actions = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                self.horizon, self.population, self.action_dim, device=device
            )
            if self.action_low is not None and self.action_high is not None:
                low = self.action_low.to(device).view(1, 1, -1)
                high = self.action_high.to(device).view(1, 1, -1)
                actions = actions.clamp(min=low, max=high)
            rollout = self.low_level_wm.rollout(
                spatial_tokens.unsqueeze(0).expand(self.population, -1, -1),
                actions.permute(1, 0, 2),
            )
            final_tokens = rollout.spatial_tokens[:, -1]
            final_global = rollout.global_states[:, -1]
            goal_cost = (final_global - target_z.unsqueeze(0)).pow(2).mean(dim=-1)
            if target_s is not None:
                spatial_cost = (final_tokens - target_s.unsqueeze(0)).pow(2).mean(dim=(-2, -1))
            else:
                spatial_cost = torch.zeros_like(goal_cost)
            action_cost = actions.pow(2).mean(dim=(0, 2))
            if target_skill is not None:
                skill_pred = self.action_chunk_encoder(actions.permute(1, 0, 2))
                skill_cost = (skill_pred - target_skill.unsqueeze(0)).pow(2).mean(dim=-1)
            else:
                skill_cost = torch.zeros_like(goal_cost)
            total_cost = (
                self.global_penalty * goal_cost
                + self.spatial_penalty * spatial_cost
                + self.skill_penalty * skill_cost
                + self.action_penalty * action_cost
            )
            elite_idx = total_cost.topk(self.elites, largest=False).indices
            elite_actions = actions[:, elite_idx]
            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1).clamp_min(1e-4)
            if self.action_low is not None and self.action_high is not None:
                mean = mean.clamp(min=self.action_low.to(device), max=self.action_high.to(device))
            best_idx = total_cost.argmin()
            best_actions = actions[:, best_idx]
            best_consistency = skill_cost[best_idx]
            costs.append(total_cost[best_idx])
        return LowLevelPlan(actions=best_actions, costs=torch.stack(costs), skill_consistency=best_consistency)
