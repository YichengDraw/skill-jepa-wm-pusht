from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class _TokenConditionBlock(nn.Module):
    def __init__(self, token_dim: int, action_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.cond = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, token_dim * 2))
        self.mlp_norm = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, token_dim))

    def forward(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_out
        gamma, beta = self.cond(action).chunk(2, dim=-1)
        conditioned = self.mlp_norm(tokens) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return tokens + self.mlp(conditioned)


@dataclass
class LowLevelRollout:
    spatial_tokens: torch.Tensor
    global_states: torch.Tensor


class LowLevelWM(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        action_dim: int = 2,
        hidden_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TokenConditionBlock(token_dim, action_dim, hidden_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

    def forward(self, spatial_tokens: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = spatial_tokens
        h = spatial_tokens
        for block in self.blocks:
            h = block(h, action)
        next_tokens = residual + self.out(h)
        next_global = next_tokens.mean(dim=1)
        return next_tokens, next_global

    def rollout(self, spatial_tokens: torch.Tensor, actions: torch.Tensor) -> LowLevelRollout:
        cur_tokens = spatial_tokens
        pred_tokens = []
        pred_globals = []
        for step in range(actions.shape[1]):
            cur_tokens, cur_global = self(cur_tokens, actions[:, step])
            pred_tokens.append(cur_tokens)
            pred_globals.append(cur_global)
        return LowLevelRollout(
            spatial_tokens=torch.stack(pred_tokens, dim=1),
            global_states=torch.stack(pred_globals, dim=1),
        )
