from __future__ import annotations

import torch
import torch.nn as nn


class _FiLMResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + gamma) + beta
        return x + self.mlp(h)


class SkillWorldModel(nn.Module):
    def __init__(self, state_dim: int = 384, skill_dim: int = 8, hidden_dim: int = 512, depth: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(state_dim, state_dim)
        self.blocks = nn.ModuleList(
            [_FiLMResidualBlock(dim=state_dim, cond_dim=skill_dim, hidden_dim=hidden_dim) for _ in range(depth)]
        )
        self.out = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim))

    def forward(self, z_t: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z_t)
        for block in self.blocks:
            h = block(h, skill)
        delta = self.out(h)
        return z_t + delta

