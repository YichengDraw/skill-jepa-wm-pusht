from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from skill_jepa.data import FeatureSequenceDataset
from skill_jepa.trainers.common import build_all_modules, load_checkpoint, modules_to_device, parameters_for, save_checkpoint
from skill_jepa.trainers.objectives import compute_low_level_losses, compute_passive_losses
from skill_jepa.utils import append_jsonl, choose_device, detach_metrics, ensure_dir, load_yaml, seed_everything, to_device


def _set_mode(modules, train: bool) -> None:
    for module in modules.values():
        module.train(train)


def _subset_batch(batch, mask):
    return {key: value[mask] if torch.is_tensor(value) and value.shape[0] == mask.shape[0] else value for key, value in batch.items()}


def evaluate(loader, modules, cfg, device, step):
    _set_mode(modules, False)
    totals = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            loss, metrics, _ = compute_passive_losses(batch, modules, cfg, global_step=step, train=False)
            joint_loss = loss
            labeled_mask = batch["is_labeled"].bool()
            if labeled_mask.any():
                low_loss, low_metrics, _ = compute_low_level_losses(_subset_batch(batch, labeled_mask), modules, cfg, train=False)
                joint_loss = joint_loss + low_loss
                metrics.update(low_metrics)
            metrics["joint_loss"] = joint_loss
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
            count += 1
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    device = choose_device(cfg.get("device"))
    out_dir = ensure_dir(cfg["training"]["joint_output_dir"])
    seq_len = cfg["training"]["chunk_size"] * cfg["training"]["rollout_chunks"] + 1

    train_set = FeatureSequenceDataset(
        cache_path=cfg["data"]["cache_path"],
        sequence_length=seq_len,
        split="train",
        labeled_fraction=cfg["data"]["labeled_fraction"],
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        stride=cfg["data"].get("stride", 1),
        seed=cfg["seed"],
    )
    val_set = FeatureSequenceDataset(
        cache_path=cfg["data"]["cache_path"],
        sequence_length=seq_len,
        split="val",
        labeled_fraction=cfg["data"]["labeled_fraction"],
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        stride=cfg["data"].get("val_stride", cfg["data"].get("stride", 1)),
        seed=cfg["seed"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    modules = build_all_modules(cfg, cfg["data"]["cache_path"])
    if cfg["training"].get("passive_checkpoint"):
        load_checkpoint(cfg["training"]["passive_checkpoint"], modules)
    if cfg["training"].get("low_level_checkpoint"):
        load_checkpoint(cfg["training"]["low_level_checkpoint"], modules)
    modules_to_device(modules, device)
    optimizer = torch.optim.AdamW(
        parameters_for(
            modules,
            ["skill_idm", "skill_wm", "skill_prior", "skill_proj", "effect_proj", "action_chunk_encoder", "low_level_wm"],
        ),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_metric = float("inf")
    global_step = 0
    try:
        for epoch in range(cfg["training"]["epochs"]):
            _set_mode(modules, True)
            for batch in train_loader:
                batch = to_device(batch, device)
                passive_loss, passive_metrics, _ = compute_passive_losses(batch, modules, cfg, global_step=global_step, train=True)
                loss = passive_loss
                metrics = dict(passive_metrics)
                labeled_mask = batch["is_labeled"].bool()
                if labeled_mask.any():
                    low_loss, low_metrics, _ = compute_low_level_losses(_subset_batch(batch, labeled_mask), modules, cfg, train=True)
                    loss = loss + low_loss
                    metrics.update(low_metrics)
                metrics["joint_loss"] = loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters_for(
                        modules,
                        ["skill_idm", "skill_wm", "skill_prior", "skill_proj", "effect_proj", "action_chunk_encoder", "low_level_wm"],
                    ),
                    cfg["training"]["grad_clip"],
                )
                optimizer.step()
                if global_step % cfg["training"]["log_every"] == 0:
                    append_jsonl(
                        out_dir / "train_metrics.jsonl",
                        {"epoch": epoch, "step": global_step, "loss": float(loss.detach().cpu()), **detach_metrics(metrics)},
                    )
                global_step += 1

            val_metrics = evaluate(val_loader, modules, cfg, device, global_step)
            append_jsonl(out_dir / "val_metrics.jsonl", {"epoch": epoch, "step": global_step, **val_metrics})
            save_checkpoint(out_dir / "joint_last.pt", modules, optimizer, global_step, cfg, metrics=val_metrics)
            joint_metric = val_metrics.get("joint_loss", float("inf"))
            if joint_metric < best_metric or not (out_dir / "joint_best.pt").exists():
                best_metric = joint_metric
                save_checkpoint(out_dir / "joint_best.pt", modules, optimizer, global_step, cfg, metrics=val_metrics)
    finally:
        train_set.close()
        val_set.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
