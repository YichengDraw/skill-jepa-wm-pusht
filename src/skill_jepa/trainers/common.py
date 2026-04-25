from __future__ import annotations

import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from skill_jepa.data import cache_metadata
from skill_jepa.modules import ActionChunkEncoder, LowLevelWM, SkillIDM, SkillPrior, SkillWorldModel
from skill_jepa.utils import ensure_dir


ROOT = Path(__file__).resolve().parents[3]
PASSIVE_DATA_PROVENANCE_KEYS = ("val_fraction", "test_fraction", "split_seed")
LOW_LEVEL_DATA_PROVENANCE_KEYS = ("labeled_fraction", "val_fraction", "test_fraction", "split_seed", "labeled_seed")
DATA_PROVENANCE_KEYS = LOW_LEVEL_DATA_PROVENANCE_KEYS


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalized_path(path: str | Path) -> str:
    return str(Path(str(path)).expanduser().resolve(strict=False)).replace("\\", "/")


def same_file_identity(recorded: str | Path, current: str | Path) -> bool:
    recorded_hash = sha256_file(recorded)
    current_hash = sha256_file(current)
    if recorded_hash is not None and current_hash is not None:
        return recorded_hash == current_hash
    return normalized_path(recorded) == normalized_path(current)


def artifact_hashes_for_config(config: Dict) -> Dict[str, str]:
    hashes = {}
    for section, keys in {
        "data": ["cache_path", "projector_ckpt"],
        "training": ["passive_checkpoint", "low_level_checkpoint"],
    }.items():
        for key in keys:
            path = config.get(section, {}).get(key)
            digest = sha256_file(path)
            if digest is not None:
                hashes[f"{section}.{key}"] = digest
    return hashes


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_status_porcelain() -> str | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _git_output_bytes(args: list[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def git_dirty() -> bool | None:
    status = git_status_porcelain()
    if status is None:
        return None
    return bool(status.strip())


def git_status_sha256() -> str | None:
    status = _git_output_bytes(["status", "--porcelain=v1", "--untracked-files=all", "-z"])
    if status is None:
        return None
    hasher = hashlib.sha256()
    hasher.update(b"STATUS\0")
    hasher.update(status)
    for args in (["diff", "--binary"], ["diff", "--cached", "--binary"]):
        diff = _git_output_bytes(args)
        if diff is None:
            return None
        hasher.update(b"\0DIFF\0")
        hasher.update(diff)
    untracked = _git_output_bytes(["ls-files", "--others", "--exclude-standard", "-z"])
    if untracked is None:
        return None
    for rel_bytes in sorted(part for part in untracked.split(b"\0") if part):
        rel_path = rel_bytes.decode("utf-8", errors="surrogateescape")
        file_path = ROOT / rel_path
        if not file_path.is_file():
            continue
        hasher.update(b"\0UNTRACKED\0")
        hasher.update(rel_bytes)
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def resolve_data_seed_config(config: Dict) -> Dict:
    resolved = deepcopy(config)
    data_cfg = resolved.setdefault("data", {})
    seed = int(resolved.get("seed", 0))
    if data_cfg.get("split_seed") is None:
        data_cfg["split_seed"] = seed
    if data_cfg.get("labeled_seed") is None:
        data_cfg["labeled_seed"] = seed + 17
    return resolved


def assert_checkpoint_code_compatible(
    checkpoint_payload: Dict,
    label: str = "Checkpoint",
    require_code_provenance: bool = True,
) -> bool:
    recorded_commit = checkpoint_payload.get("code_commit")
    has_dirty_field = "code_dirty" in checkpoint_payload
    if recorded_commit is None or not has_dirty_field:
        if require_code_provenance:
            raise RuntimeError(f"{label} does not record code provenance")
        return False
    current_commit = git_commit()
    if current_commit is not None and recorded_commit != current_commit:
        raise RuntimeError(
            f"{label} code_commit does not match the current code: "
            f"recorded={recorded_commit}, current={current_commit}"
        )
    current_dirty = git_dirty()
    recorded_dirty = bool(checkpoint_payload.get("code_dirty"))
    if current_dirty is not None and recorded_dirty != current_dirty:
        raise RuntimeError(
            f"{label} code_dirty does not match the current code: "
            f"recorded={recorded_dirty}, current={current_dirty}"
        )
    if recorded_dirty:
        current_status_sha = git_status_sha256()
        recorded_status_sha = checkpoint_payload.get("code_status_sha256")
        if current_status_sha is not None and recorded_status_sha != current_status_sha:
            raise RuntimeError(f"{label} code_status_sha256 does not match the current dirty tree")
    return True


def assert_checkpoint_config_compatible(
    checkpoint_payload: Dict,
    runtime_config: Dict,
    label: str = "Checkpoint",
    sections: tuple[str, ...] = ("encoder", "model"),
    data_keys: tuple[str, ...] = ("cache_path", "projector_ckpt"),
    data_value_keys: tuple[str, ...] = (),
    check_code: bool = False,
    require_code_provenance: bool = True,
) -> None:
    if check_code:
        assert_checkpoint_code_compatible(
            checkpoint_payload,
            label=label,
            require_code_provenance=require_code_provenance,
        )
    checkpoint_config = checkpoint_payload.get("config", {})
    if not checkpoint_config:
        raise RuntimeError(f"{label} does not record its training config")
    checkpoint_config = resolve_data_seed_config(checkpoint_config)
    runtime_config = resolve_data_seed_config(runtime_config)
    for section in sections:
        if checkpoint_config.get(section) != runtime_config.get(section):
            raise RuntimeError(f"{label} {section} config does not match the runtime config")
    checkpoint_data = checkpoint_config.get("data", {})
    runtime_data = runtime_config.get("data", {})
    for key in data_value_keys:
        if checkpoint_data.get(key) != runtime_data.get(key):
            raise RuntimeError(
                f"{label} data.{key} does not match the runtime config: "
                f"recorded={checkpoint_data.get(key)}, current={runtime_data.get(key)}"
            )
    checkpoint_hashes = checkpoint_payload.get("artifact_hashes", {})
    runtime_hashes = artifact_hashes_for_config(runtime_config)
    for key in data_keys:
        recorded = checkpoint_data.get(key)
        current = runtime_data.get(key)
        if recorded is None or current is None:
            continue
        hash_key = f"data.{key}"
        if hash_key in checkpoint_hashes and hash_key in runtime_hashes:
            if checkpoint_hashes[hash_key] != runtime_hashes[hash_key]:
                raise RuntimeError(f"{label} {key} hash does not match the runtime config")
            continue
        if not same_file_identity(recorded, current):
            raise RuntimeError(
                f"{label} {key} does not match the runtime config: "
                f"recorded={recorded}, current={current}"
            )


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
        "config": resolve_data_seed_config(config),
        "code_commit": git_commit(),
        "code_dirty": git_dirty(),
        "code_status_sha256": git_status_sha256(),
        "artifact_hashes": artifact_hashes_for_config(config),
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
        unexpected_modules = sorted(set(checkpoint_modules) - set(modules))
        if unexpected_modules:
            raise RuntimeError(f"Checkpoint has unexpected modules: {unexpected_modules}")
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


def load_checkpoint_subset(path: str | Path, modules: Dict[str, nn.Module], required_modules: list[str]) -> Dict:
    payload = torch.load(path, map_location="cpu")
    checkpoint_modules = payload.get("modules", {})
    missing_modules = sorted(set(required_modules) - set(checkpoint_modules))
    if missing_modules:
        raise RuntimeError(f"Checkpoint is missing required modules: {missing_modules}")
    for name in required_modules:
        modules[name].load_state_dict(checkpoint_modules[name], strict=True)
    return payload
