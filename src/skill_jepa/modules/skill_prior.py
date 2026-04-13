from __future__ import annotations

import torch
import torch.nn as nn


class SkillPrior(nn.Module):
    def __init__(self, state_dim: int = 384, skill_dim: int = 8, hidden_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, skill_dim)
        self.logvar_head = nn.Linear(hidden_dim, skill_dim)

    def forward(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z_t)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-8.0, 8.0)
        return mean, logvar

