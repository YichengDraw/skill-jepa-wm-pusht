from __future__ import annotations

import torch
import torch.nn as nn


class ActionChunkEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        proprio_dim: int = 0,
        hidden_dim: int = 128,
        skill_dim: int = 8,
    ) -> None:
        super().__init__()
        in_dim = action_dim + proprio_dim
        self.proprio_dim = proprio_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, skill_dim)

    def forward(self, actions: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        if proprio is not None and self.proprio_dim > 0:
            x = torch.cat([actions, proprio], dim=-1)
        else:
            x = actions
        x = x.transpose(1, 2)
        x = self.net(x).mean(dim=-1)
        return self.head(x)

