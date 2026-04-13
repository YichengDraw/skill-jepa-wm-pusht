from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SkillPosterior:
    sample: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class SkillIDM(nn.Module):
    def __init__(self, state_dim: int = 384, hidden_dim: int = 384, skill_dim: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.skill_dim = skill_dim
        self.encoder = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.mean_head = nn.Linear(hidden_dim * 2, skill_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, skill_dim)

    def encode(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.encoder(states)
        pooled = self.norm(outputs.mean(dim=1))
        mean = self.mean_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-8.0, 8.0)
        return mean, logvar

    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, states: torch.Tensor, deterministic: bool = False) -> SkillPosterior:
        mean, logvar = self.encode(states)
        sample = mean if deterministic else self.sample(mean, logvar)
        return SkillPosterior(sample=sample, mean=mean, logvar=logvar)

