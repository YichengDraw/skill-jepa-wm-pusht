from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.t()
    logits = logits / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


def gaussian_kl(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor | None = None,
    logvar_p: torch.Tensor | None = None,
) -> torch.Tensor:
    if mean_p is None:
        mean_p = torch.zeros_like(mean_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * ((logvar_p - logvar_q) + (var_q + (mean_q - mean_p).pow(2)) / var_p - 1.0)
    return kl.sum(dim=-1).mean()


def rollout_weight(step: int, start_step: int, ramp_steps: int) -> float:
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    progress = min(1.0, (step - start_step) / float(ramp_steps))
    return float(progress)


def pairwise_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(a - b).mean()


def negative_log_gaussian(skill: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((skill - mean).pow(2) / logvar.exp()) + logvar + math.log(2 * math.pi)).sum(dim=-1)

