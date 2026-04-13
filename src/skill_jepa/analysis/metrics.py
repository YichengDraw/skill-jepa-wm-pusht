from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def effect_retrieval(skill_embeddings: torch.Tensor, effect_embeddings: torch.Tensor, ks: Sequence[int] = (1, 5)) -> Dict[str, float]:
    skill_embeddings = F.normalize(skill_embeddings, dim=-1)
    effect_embeddings = F.normalize(effect_embeddings, dim=-1)
    scores = skill_embeddings @ effect_embeddings.t()
    order = scores.argsort(dim=-1, descending=True)
    target = torch.arange(scores.size(0), device=scores.device).unsqueeze(1)
    out: Dict[str, float] = {}
    for k in ks:
        hits = (order[:, :k] == target).any(dim=1).float().mean()
        out[f"R@{k}"] = float(hits.cpu())
    return out


def composition_residual(u_ij: torch.Tensor, u_jk: torch.Tensor, u_ik: torch.Tensor) -> torch.Tensor:
    return torch.abs(u_ik - (u_ij + u_jk)).mean()


def identity_residual(u_ii: torch.Tensor) -> torch.Tensor:
    return torch.abs(u_ii).mean()


def inverse_residual(u_ij: torch.Tensor, u_ji: torch.Tensor) -> torch.Tensor:
    return torch.abs(u_ij + u_ji).mean()


def train_leakage_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    steps: int = 100,
    lr: float = 1e-2,
) -> float:
    if features.numel() == 0:
        return 0.0
    device = features.device
    probe = nn.Linear(features.size(-1), num_classes, device=device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    for _ in range(steps):
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = probe(features).argmax(dim=-1)
    return float((preds == labels).float().mean().cpu())

