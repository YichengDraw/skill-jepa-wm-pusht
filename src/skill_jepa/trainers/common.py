from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from skill_jepa.data import cache_metadata
from skill_jepa.modules import ActionChunkEncoder, LowLevelWM, SkillIDM, SkillPrior, SkillWorldModel
from skill_jepa.utils import ensure_dir


def build_skill_modules(cfg: Dict, cache_path: str) -> Dict[str, nn.Module]:
    meta = cache_metadata(cache_path)
    model_cfg = cfg["model"]
    return {
        "skill_idm": SkillIDM(
            state_dim=meta["state_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            skill_dim=model_cfg["skill_dim"],
        ),
        "skill_wm": SkillWorldModel(
            state_dim=meta["state_dim"],
            skill_dim=model_cfg["skill_dim"],
            hidden_dim=model_cfg["wm_hidden_dim"],
            depth=model_cfg["wm_depth"],
        ),
        "skill_prior": SkillPrior(
            state_dim=meta["state_dim"],
            skill_dim=model_cfg["skill_dim"],
            hidden_dim=model_cfg["hidden_dim"],
        ),
        "skill_proj": nn.Sequential(
            nn.LayerNorm(model_cfg["skill_dim"]),
            nn.Linear(model_cfg["skill_dim"], model_cfg["skill_dim"]),
        ),
        "effect_proj": nn.Sequential(
            nn.LayerNorm(meta["state_dim"]),
            nn.Linear(meta["state_dim"], model_cfg["skill_dim"]),
        ),
    }


def build_low_level_modules(cfg: Dict, cache_path: str) -> Dict[str, nn.Module]:
    meta = cache_metadata(cache_path)
    model_cfg = cfg["model"]
    return {
        "action_chunk_encoder": ActionChunkEncoder(
            action_dim=meta["action_dim"],
            proprio_dim=0,
            hidden_dim=model_cfg["action_hidden_dim"],
            skill_dim=model_cfg["skill_dim"],
        ),
        "low_level_wm": LowLevelWM(
            token_dim=meta["token_dim"],
            action_dim=meta["action_dim"],
            hidden_dim=model_cfg["low_level_hidden_dim"],
            depth=model_cfg["low_level_depth"],
            num_heads=model_cfg["low_level_heads"],
        ),
    }


def build_all_modules(cfg: Dict, cache_path: str) -> Dict[str, nn.Module]:
    modules = build_skill_modules(cfg, cache_path)
    modules.update(build_low_level_modules(cfg, cache_path))
    return modules


def modules_to_device(modules: Dict[str, nn.Module], device: torch.device) -> None:
    for module in modules.values():
        module.to(device)


def parameters_for(modules: Dict[str, nn.Module], names) -> list:
    params = []
    for name in names:
        params.extend(list(modules[name].parameters()))
    return params


def save_checkpoint(
    path: str | Path,
    modules: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: Dict,
    metrics: Dict | None = None,
) -> None:
    ensure_dir(Path(path).parent)
    payload = {
        "step": step,
        "config": config,
        "metrics": metrics or {},
        "modules": {name: module.state_dict() for name, module in modules.items()},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    modules: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer | None = None,
    strict_modules: bool = False,
) -> Dict:
    payload = torch.load(path, map_location="cpu")
    checkpoint_modules = payload.get("modules", {})
    if strict_modules:
        missing_modules = sorted(set(modules) - set(checkpoint_modules))
        if missing_modules:
            raise RuntimeError(f"Checkpoint is missing required modules: {missing_modules}")
    for name, module in modules.items():
        if name in checkpoint_modules:
            incompatible = module.load_state_dict(checkpoint_modules[name], strict=strict_modules)
            if strict_modules and (incompatible.missing_keys or incompatible.unexpected_keys):
                raise RuntimeError(
                    f"Checkpoint module {name!r} does not match: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
