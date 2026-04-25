from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from skill_jepa.data import FeatureSequenceDataset
from skill_jepa.trainers.common import (
    PASSIVE_DATA_PROVENANCE_KEYS,
    assert_checkpoint_config_compatible,
    build_low_level_modules,
    build_skill_modules,
    load_checkpoint,
    modules_to_device,
    parameters_for,
    save_checkpoint,
)
from skill_jepa.trainers.objectives import compute_low_level_losses
from skill_jepa.utils import append_jsonl, choose_device, detach_metrics, ensure_dir, load_yaml, seed_everything, to_device


def _set_mode(modules, train: bool) -> None:
    for module in modules.values():
        module.train(train)


def _freeze(modules, names) -> None:
    for name in names:
        for param in modules[name].parameters():
            param.requires_grad_(False)


def evaluate(loader, modules, cfg, device):
    _set_mode(modules, False)
    totals = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            _, metrics, _ = compute_low_level_losses(batch, modules, cfg, train=False)
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
    out_dir = ensure_dir(cfg["training"]["low_level_output_dir"])
    seq_len = max(cfg["training"]["chunk_size"] + 1, cfg["training"]["low_rollout_steps"] + 1)
    split_seed = cfg["data"].get("split_seed", cfg["seed"])
    labeled_seed = cfg["data"].get("labeled_seed", cfg["seed"] + 17)

    train_set = FeatureSequenceDataset(
        cache_path=cfg["data"]["cache_path"],
        sequence_length=seq_len,
        split="train",
        labeled_fraction=cfg["data"]["labeled_fraction"],
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        stride=cfg["data"].get("stride", 1),
        seed=cfg["seed"],
        split_seed=split_seed,
        labeled_seed=labeled_seed,
        use_only_labeled=True,
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
        split_seed=split_seed,
        labeled_seed=labeled_seed,
        use_only_labeled=True,
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

    modules = build_skill_modules(cfg, cfg["data"]["cache_path"])
    modules.update(build_low_level_modules(cfg, cfg["data"]["cache_path"]))
    if cfg["training"].get("passive_checkpoint"):
        passive_names = ["skill_idm", "skill_wm", "skill_prior", "skill_proj", "effect_proj"]
        passive_payload = load_checkpoint(
            cfg["training"]["passive_checkpoint"],
            {name: modules[name] for name in passive_names},
            strict_modules=True,
        )
        assert_checkpoint_config_compatible(
            passive_payload,
            cfg,
            label="Passive checkpoint",
            data_value_keys=PASSIVE_DATA_PROVENANCE_KEYS,
            check_code=True,
        )
    _freeze(modules, ["skill_idm", "skill_wm", "skill_prior", "skill_proj", "effect_proj"])
    modules_to_device(modules, device)
    optimizer = torch.optim.AdamW(
        parameters_for(modules, ["action_chunk_encoder", "low_level_wm"]),
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
                loss, metrics, _ = compute_low_level_losses(batch, modules, cfg, train=True)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters_for(modules, ["action_chunk_encoder", "low_level_wm"]),
                    cfg["training"]["grad_clip"],
                )
                optimizer.step()
                if global_step % cfg["training"]["log_every"] == 0:
                    append_jsonl(
                        out_dir / "train_metrics.jsonl",
                        {"epoch": epoch, "step": global_step, "loss": float(loss.detach().cpu()), **detach_metrics(metrics)},
                    )
                global_step += 1

            val_metrics = evaluate(val_loader, modules, cfg, device)
            append_jsonl(out_dir / "val_metrics.jsonl", {"epoch": epoch, "step": global_step, **val_metrics})
            save_checkpoint(out_dir / "low_level_last.pt", modules, optimizer, global_step, cfg, metrics=val_metrics)
            low_roll = val_metrics.get("low_roll", float("inf"))
            if low_roll < best_metric or not (out_dir / "low_level_best.pt").exists():
                best_metric = low_roll
                save_checkpoint(out_dir / "low_level_best.pt", modules, optimizer, global_step, cfg, metrics=val_metrics)
    finally:
        train_set.close()
        val_set.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
