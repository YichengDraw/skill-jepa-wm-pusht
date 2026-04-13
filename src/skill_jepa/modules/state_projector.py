from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 384, pool_grid: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_grid = pool_grid
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def _project_once(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, num_tokens, _ = patch_tokens.shape
        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"Patch tokens must form a square grid, got {num_tokens}")
        tokens = self.norm(patch_tokens)
        tokens = tokens.transpose(1, 2).reshape(batch, self.input_dim, side, side)
        pooled = F.adaptive_avg_pool2d(tokens, output_size=(self.pool_grid, self.pool_grid))
        pooled = pooled.flatten(2).transpose(1, 2)
        spatial = self.proj(pooled)
        global_state = spatial.mean(dim=1)
        return {"spatial_tokens": spatial, "global_state": global_state}

    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if patch_tokens.ndim == 3:
            return self._project_once(patch_tokens)
        if patch_tokens.ndim != 4:
            raise ValueError(f"Expected [B,N,D] or [B,T,N,D], got {tuple(patch_tokens.shape)}")
        bsz, time, num_tokens, hidden = patch_tokens.shape
        outputs = self._project_once(patch_tokens.reshape(bsz * time, num_tokens, hidden))
        return {
            "spatial_tokens": outputs["spatial_tokens"].reshape(bsz, time, -1, self.output_dim),
            "global_state": outputs["global_state"].reshape(bsz, time, self.output_dim),
        }

