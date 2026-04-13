from __future__ import annotations

from typing import Dict, Tuple

import torch

from skill_jepa.analysis.metrics import composition_residual, identity_residual, inverse_residual
from skill_jepa.losses import gaussian_kl, info_nce_loss, pairwise_l1, rollout_weight


def _skill_chunks(z_seq: torch.Tensor, chunk_size: int, num_chunks: int) -> list[torch.Tensor]:
    return [z_seq[:, idx * chunk_size : idx * chunk_size + chunk_size + 1] for idx in range(num_chunks)]


def compute_passive_losses(
    batch: Dict[str, torch.Tensor],
    modules: Dict[str, torch.nn.Module],
    cfg: Dict,
    global_step: int,
    train: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    chunk_size = int(cfg["training"]["chunk_size"])
    num_chunks = int(cfg["training"]["rollout_chunks"])
    z_seq = batch["z"]
    skill_chunks = _skill_chunks(z_seq, chunk_size=chunk_size, num_chunks=num_chunks)
    posteriors = [modules["skill_idm"](chunk, deterministic=not train) for chunk in skill_chunks]

    skill_end_losses = []
    rollout_losses = []
    kl_losses = []
    effect_queries = []
    effect_keys = []
    pred = z_seq[:, 0]
    for idx, (chunk, posterior) in enumerate(zip(skill_chunks, posteriors)):
        chunk_start = z_seq[:, idx * chunk_size]
        chunk_end = z_seq[:, (idx + 1) * chunk_size]
        end_pred = modules["skill_wm"](chunk_start, posterior.sample)
        skill_end_losses.append(pairwise_l1(end_pred, chunk_end))
        pred = modules["skill_wm"](pred, posterior.sample)
        rollout_losses.append(pairwise_l1(pred, chunk_end))
        prior_mean, prior_logvar = modules["skill_prior"](chunk_start)
        kl_losses.append(gaussian_kl(posterior.mean, posterior.logvar, prior_mean, prior_logvar))
        effect = modules["effect_proj"]((chunk[:, 1:] - chunk[:, :-1]).sum(dim=1))
        skill = modules["skill_proj"](posterior.sample)
        effect_queries.append(skill)
        effect_keys.append(effect)

    if num_chunks >= 2:
        u_ij = posteriors[0].mean
        u_jk = posteriors[1].mean
        u_ik = modules["skill_idm"](z_seq[:, : 2 * chunk_size + 1], deterministic=not train).mean
        u_id = modules["skill_idm"](torch.stack([z_seq[:, 0], z_seq[:, 0]], dim=1), deterministic=not train).mean
        u_ji = modules["skill_idm"](torch.flip(z_seq[:, : chunk_size + 1], dims=[1]), deterministic=not train).mean
        comp_loss = composition_residual(u_ij, u_jk, u_ik)
        id_loss = identity_residual(u_id)
        inv_loss = inverse_residual(u_ij, u_ji)
        total_comp = comp_loss + 0.25 * id_loss + 0.25 * inv_loss
    else:
        comp_loss = z_seq.new_tensor(0.0)
        id_loss = z_seq.new_tensor(0.0)
        inv_loss = z_seq.new_tensor(0.0)
        total_comp = z_seq.new_tensor(0.0)

    effect_loss = info_nce_loss(torch.cat(effect_queries, dim=0), torch.cat(effect_keys, dim=0))
    skill_end = torch.stack(skill_end_losses).mean()
    skill_roll = torch.stack(rollout_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    skill_std = torch.cat([posterior.mean for posterior in posteriors], dim=0).std(dim=0).mean()
    posterior_var = torch.cat([posterior.logvar.exp() for posterior in posteriors], dim=0).mean()
    roll_weight = rollout_weight(
        step=global_step,
        start_step=int(cfg["training"].get("rollout_warmup_steps", 0)),
        ramp_steps=int(cfg["training"].get("rollout_ramp_steps", 0)),
    )
    loss_cfg = cfg["loss"]
    total = (
        loss_cfg["skill_end_weight"] * skill_end
        + loss_cfg["skill_roll_weight"] * roll_weight * skill_roll
        + loss_cfg["effect_weight"] * effect_loss
        + loss_cfg["composition_weight"] * total_comp
        + loss_cfg["kl_weight"] * kl_loss
    )
    metrics = {
        "skill_end": skill_end,
        "skill_roll": skill_roll,
        "effect": effect_loss,
        "composition": comp_loss,
        "identity": id_loss,
        "inverse": inv_loss,
        "kl": kl_loss,
        "skill_std": skill_std,
        "posterior_var": posterior_var,
        "rollout_weight": z_seq.new_tensor(roll_weight),
    }
    aux = {
        "u_obs": posteriors[0].mean,
        "u_obs_sample": posteriors[0].sample,
    }
    return total, metrics, aux


def compute_low_level_losses(
    batch: Dict[str, torch.Tensor],
    modules: Dict[str, torch.nn.Module],
    cfg: Dict,
    train: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    chunk_size = int(cfg["training"]["chunk_size"])
    rollout_steps = int(cfg["training"]["low_rollout_steps"])
    z_seq = batch["z"]
    s_seq = batch["s"]
    actions = batch["action"]
    post = modules["skill_idm"](z_seq[:, : chunk_size + 1], deterministic=not train)
    u_obs = post.mean
    u_act = modules["action_chunk_encoder"](actions[:, :chunk_size])
    align = pairwise_l1(u_obs.detach(), u_act) + pairwise_l1(u_obs, u_act.detach())
    action_skill = pairwise_l1(modules["skill_wm"](z_seq[:, 0], u_act), z_seq[:, chunk_size])

    next_tokens, next_global = modules["low_level_wm"](s_seq[:, 0], actions[:, 0])
    low_1 = pairwise_l1(next_tokens, s_seq[:, 1]) + pairwise_l1(next_global, z_seq[:, 1])
    rollout = modules["low_level_wm"].rollout(s_seq[:, 0], actions[:, :rollout_steps])
    low_roll = 0.0
    for step in range(rollout_steps):
        low_roll = low_roll + pairwise_l1(rollout.spatial_tokens[:, step], s_seq[:, step + 1])
        low_roll = low_roll + pairwise_l1(rollout.global_states[:, step], z_seq[:, step + 1])
    low_roll = low_roll / max(1, rollout_steps)
    total = (
        cfg["loss"]["align_weight"] * align
        + cfg["loss"]["action_skill_weight"] * action_skill
        + cfg["loss"]["low_level_weight"] * (low_1 + cfg["loss"]["low_roll_weight"] * low_roll)
    )
    metrics = {
        "align": align,
        "action_skill": action_skill,
        "low_1": low_1,
        "low_roll": low_roll,
    }
    aux = {"u_obs": u_obs, "u_act": u_act}
    return total, metrics, aux
