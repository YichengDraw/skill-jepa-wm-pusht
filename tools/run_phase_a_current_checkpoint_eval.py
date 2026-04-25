from __future__ import annotations

import argparse
import filecmp
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIBLING_DEBUG = ROOT.parent / "jepa-wms" / "outputs" / "skill_jepa" / "pusht_debug"
DEFAULT_CACHE = SIBLING_DEBUG / "cache" / "pusht_vjepa2_cache.h5"
DEFAULT_PROJECTOR = SIBLING_DEBUG / "cache" / "state_projector.pt"
DEFAULT_CHECKPOINT = SIBLING_DEBUG / "joint" / "joint_best.pt"
OUTPUT_ROOT = ROOT / "outputs" / "skill_jepa" / "phase_a_current_checkpoint"


def _path_str(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _write_config(device: str | None, num_eval_episodes: int, cache_path: Path, projector_path: Path) -> Path:
    for path in [cache_path, projector_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase A input: {path}")
    with open(ROOT / "configs" / "exp" / "pusht_debug.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if device is not None:
        cfg["device"] = device
    cfg["data"]["cache_path"] = _path_str(cache_path)
    cfg["data"]["projector_ckpt"] = _path_str(projector_path)
    cfg["planner"]["eval_split"] = "test"
    cfg["planner"]["num_eval_episodes"] = int(num_eval_episodes)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg_path = OUTPUT_ROOT / "phase_a_external_debug.yaml"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--projector", type=Path, default=DEFAULT_PROJECTOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-determinism-check", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing Phase A input: {args.checkpoint}")
    cfg_path = _write_config(args.device, args.num_eval_episodes, args.cache, args.projector)
    common = [
        sys.executable,
        "-m",
        "skill_jepa.analysis.eval_pusht_online",
        "--config",
        _path_str(cfg_path),
        "--checkpoint",
        _path_str(args.checkpoint),
        "--mode",
        "both",
        "--num-eval-episodes",
        str(args.num_eval_episodes),
        "--subgoal-scope",
        "train",
        "--goal-mode",
        "trajectory",
        "--allow-under-sampling",
        "--min-unique-episodes",
        "1",
    ]
    _run(
        common
        + [
            "--output",
            _path_str(OUTPUT_ROOT / "eval"),
            "--save-videos",
            "--video-limit",
            "1",
            "--force",
        ]
    )
    if args.skip_determinism_check:
        return
    for name in ["determinism_a", "determinism_b"]:
        _run(
            common
            + [
                "--output",
                _path_str(OUTPUT_ROOT / name),
                "--deterministic-timing",
                "--force",
            ]
        )
    left = OUTPUT_ROOT / "determinism_a" / "pusht_online_records.csv"
    right = OUTPUT_ROOT / "determinism_b" / "pusht_online_records.csv"
    if not filecmp.cmp(left, right, shallow=False):
        raise RuntimeError(f"Deterministic record mismatch: {left} != {right}")
    print(f"Deterministic records match: {left} == {right}", flush=True)


if __name__ == "__main__":
    main()
