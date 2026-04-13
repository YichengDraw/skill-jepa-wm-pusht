from __future__ import annotations

import csv
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_jepa.utils import ensure_dir, load_yaml

PYTHON = Path(sys.executable)


def _env_path(name: str, default: str) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else (ROOT / default)


DEBUG_CONFIG = _env_path("PUSHT_DEBUG_CONFIG", "configs/exp/pusht_debug.yaml")
DEBUG_CHECKPOINT = _env_path("PUSHT_DEBUG_CHECKPOINT", "outputs/skill_jepa/pusht_debug/joint/joint_best.pt")
DEBUG_PROJECTOR = _env_path("PUSHT_DEBUG_PROJECTOR", "outputs/skill_jepa/pusht_debug/cache/state_projector.pt")
FULL_RAW_H5 = _env_path("PUSHT_FULL_RAW_H5", "data/pusht_expert_train.h5")

OUTPUT_ROOT = ROOT / "outputs" / "skill_jepa" / "pusht_locked_suite"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
LOG_ROOT = OUTPUT_ROOT / "logs"
CACHE_ROOT = OUTPUT_ROOT / "cache"
EVAL_ROOT = OUTPUT_ROOT / "evals"
REPORT_ROOT = OUTPUT_ROOT / "reports"
PLOT_ROOT = OUTPUT_ROOT / "plots"

TOTAL_EPISODES = 13_024
TRAIN_EPISODES = 12_000
VAL_EPISODES = 1_024
VAL_FRACTION = VAL_EPISODES / TOTAL_EPISODES
SEEDS = [0, 1, 2]
GOAL_EVALS = [(24, 100, True), (16, 50, False), (32, 50, False)]
CONFIG_ORDER = [
    "joint_hier_10pct",
    "joint_flat_10pct",
    "labeled_only_flat_10pct",
    "labeled_only_flat_100pct",
    "random_skill_hier_10pct",
]
PRIMARY_GOAL_GAP = 24
HIER_MIN_SUCCESS = 0.15
PAIRWISE_MIN_DELTA = 0.05
LABELED100_MIN_SUCCESS = 0.10


def _path_str(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _run_logged(command: list[str], log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode == 0:
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    tail_text = "\n".join(tail)
    raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}\n{tail_text}")


def _write_yaml(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _base_scaled_cfg(seed: int, labeled_fraction: float, seed_root: Path) -> dict:
    cfg = load_yaml(DEBUG_CONFIG)
    cfg["seed"] = seed
    cfg["data"]["raw_h5_path"] = _path_str(FULL_RAW_H5)
    cfg["data"]["cache_path"] = _path_str(CACHE_ROOT / "pusht_vjepa2_cache_13024ep.h5")
    cfg["data"]["projector_ckpt"] = _path_str(DEBUG_PROJECTOR)
    cfg["data"]["max_steps"] = None
    cfg["data"]["max_episodes"] = TOTAL_EPISODES
    cfg["data"]["cache_batch_size"] = 16
    cfg["data"]["labeled_fraction"] = float(labeled_fraction)
    cfg["data"]["val_fraction"] = float(VAL_FRACTION)
    cfg["data"]["test_fraction"] = 0.0
    cfg["training"]["passive_output_dir"] = _path_str(seed_root / "passive")
    cfg["training"]["passive_checkpoint"] = _path_str(seed_root / "passive" / "passive_best.pt")
    cfg["training"]["low_level_output_dir"] = _path_str(seed_root / "low_level")
    cfg["training"]["low_level_checkpoint"] = _path_str(seed_root / "low_level" / "low_level_best.pt")
    cfg["training"]["joint_output_dir"] = _path_str(seed_root / "joint")
    cfg["planner"]["eval_split"] = "val"
    cfg["planner"]["num_eval_episodes"] = 100
    return cfg


def _validate_cache(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return int(handle["ep_len"].shape[0]) == TOTAL_EPISODES and int(handle["episode_idx"].shape[0]) > 0
    except OSError:
        return False


def ensure_scaled_cache() -> Path:
    cache_path = CACHE_ROOT / "pusht_vjepa2_cache_13024ep.h5"
    if _validate_cache(cache_path):
        return cache_path
    cfg = _base_scaled_cfg(SEEDS[0], labeled_fraction=0.1, seed_root=OUTPUT_ROOT / "seed_template")
    cfg_path = CONFIG_ROOT / "scaled_cache.yaml"
    _write_yaml(cfg_path, cfg)
    _run_logged(
        [str(PYTHON), "-m", "tools.cache_vjepa_features", "--config", _path_str(cfg_path)],
        LOG_ROOT / "cache_build.log",
    )
    if not _validate_cache(cache_path):
        raise RuntimeError(f"Scaled cache validation failed: {cache_path}")
    return cache_path


def run_debug_reeval() -> Path:
    out_dir = EVAL_ROOT / "current_best_checkpoint_100ep"
    summary_path = out_dir / "pusht_online_eval.json"
    if summary_path.exists():
        return summary_path
    _run_logged(
        [
            str(PYTHON),
            "-m",
            "skill_jepa.analysis.eval_pusht_online",
            "--config",
            _path_str(DEBUG_CONFIG),
            "--checkpoint",
            _path_str(DEBUG_CHECKPOINT),
            "--output",
            _path_str(out_dir),
            "--mode",
            "both",
            "--num-eval-episodes",
            "100",
            "--save-videos",
            "--video-limit",
            "4",
        ],
        LOG_ROOT / "current_best_checkpoint_100ep.log",
    )
    return summary_path


def _seed_paths(seed: int) -> dict[str, Path]:
    seed_root = OUTPUT_ROOT / f"seed_{seed}"
    return {
        "root": seed_root,
        "passive": seed_root / "passive" / "passive_best.pt",
        "low10": seed_root / "labeled_only_flat_10pct" / "low_level" / "low_level_best.pt",
        "joint10": seed_root / "joint_10pct" / "joint" / "joint_best.pt",
        "low100": seed_root / "labeled_only_flat_100pct" / "low_level" / "low_level_best.pt",
    }


def train_seed(seed: int) -> dict[str, Path]:
    paths = _seed_paths(seed)
    joint_root = OUTPUT_ROOT / f"seed_{seed}"
    cfg_joint = _base_scaled_cfg(seed, labeled_fraction=0.1, seed_root=joint_root / "joint_10pct")
    cfg_joint["training"]["passive_output_dir"] = _path_str(OUTPUT_ROOT / f"seed_{seed}" / "passive")
    cfg_joint["training"]["passive_checkpoint"] = _path_str(paths["passive"])
    cfg_joint["training"]["low_level_output_dir"] = _path_str(OUTPUT_ROOT / f"seed_{seed}" / "labeled_only_flat_10pct" / "low_level")
    cfg_joint["training"]["low_level_checkpoint"] = _path_str(paths["low10"])
    cfg_joint["training"]["joint_output_dir"] = _path_str(OUTPUT_ROOT / f"seed_{seed}" / "joint_10pct" / "joint")

    cfg_low100 = _base_scaled_cfg(seed, labeled_fraction=1.0, seed_root=joint_root / "labeled_only_flat_100pct")
    cfg_low100["training"]["passive_output_dir"] = _path_str(OUTPUT_ROOT / f"seed_{seed}" / "passive")
    cfg_low100["training"]["passive_checkpoint"] = _path_str(paths["passive"])
    cfg_low100["training"]["low_level_output_dir"] = _path_str(OUTPUT_ROOT / f"seed_{seed}" / "labeled_only_flat_100pct" / "low_level")
    cfg_low100["training"]["low_level_checkpoint"] = _path_str(paths["low100"])
    cfg_low100["training"]["joint_output_dir"] = _path_str(OUTPUT_ROOT / f"seed_{seed}" / "unused_joint")

    cfg_joint_path = CONFIG_ROOT / f"seed_{seed}_joint10.yaml"
    cfg_low100_path = CONFIG_ROOT / f"seed_{seed}_low100.yaml"
    _write_yaml(cfg_joint_path, cfg_joint)
    _write_yaml(cfg_low100_path, cfg_low100)

    if not paths["passive"].exists():
        _run_logged(
            [str(PYTHON), "-m", "skill_jepa.trainers.train_skill_passive", "--config", _path_str(cfg_joint_path)],
            LOG_ROOT / f"seed_{seed}_passive.log",
        )
    if not paths["low10"].exists():
        _run_logged(
            [str(PYTHON), "-m", "skill_jepa.trainers.train_low_level", "--config", _path_str(cfg_joint_path)],
            LOG_ROOT / f"seed_{seed}_low10.log",
        )
    if not paths["joint10"].exists():
        _run_logged(
            [str(PYTHON), "-m", "skill_jepa.trainers.train_joint", "--config", _path_str(cfg_joint_path)],
            LOG_ROOT / f"seed_{seed}_joint10.log",
        )
    if not paths["low100"].exists():
        _run_logged(
            [str(PYTHON), "-m", "skill_jepa.trainers.train_low_level", "--config", _path_str(cfg_low100_path)],
            LOG_ROOT / f"seed_{seed}_low100.log",
        )
    return {
        "joint_cfg": cfg_joint_path,
        "low100_cfg": cfg_low100_path,
        **paths,
    }


def evaluate_seed(seed: int, artifacts: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    episode_rows: list[dict] = []
    eval_specs = [
        ("joint_hier_10pct", artifacts["joint_cfg"], artifacts["joint10"], "hierarchical"),
        ("joint_flat_10pct", artifacts["joint_cfg"], artifacts["joint10"], "flat"),
        ("labeled_only_flat_10pct", artifacts["joint_cfg"], artifacts["low10"], "flat"),
        ("labeled_only_flat_100pct", artifacts["low100_cfg"], artifacts["low100"], "flat"),
        ("random_skill_hier_10pct", artifacts["joint_cfg"], artifacts["joint10"], "random_hierarchical"),
    ]
    for config_name, cfg_path, checkpoint_path, mode in eval_specs:
        for goal_gap, num_eval, save_videos in GOAL_EVALS:
            out_dir = EVAL_ROOT / f"seed_{seed}" / config_name / f"goal_gap_{goal_gap}"
            summary_path = out_dir / "pusht_online_eval.json"
            if not summary_path.exists():
                command = [
                    str(PYTHON),
                    "-m",
                    "skill_jepa.analysis.eval_pusht_online",
                    "--config",
                    _path_str(cfg_path),
                    "--checkpoint",
                    _path_str(checkpoint_path),
                    "--output",
                    _path_str(out_dir),
                    "--mode",
                    mode,
                    "--eval-split",
                    "val",
                    "--goal-gap",
                    str(goal_gap),
                    "--num-eval-episodes",
                    str(num_eval),
                ]
                if save_videos:
                    command.extend(["--save-videos", "--video-limit", "2"])
                _run_logged(command, LOG_ROOT / f"seed_{seed}_{config_name}_gap_{goal_gap}.log")
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            method_summary = summary[mode]
            rows.append(
                {
                    "seed": seed,
                    "config_name": config_name,
                    "goal_gap": goal_gap,
                    "num_eval_episodes": num_eval,
                    "mode": mode,
                    "success_rate": method_summary["success_rate"],
                    "state_dist": method_summary["state_dist"],
                    "final_latent_distance": method_summary["final_latent_distance"],
                    "planning_latency_sec": method_summary["planning_latency_sec"],
                    "skill_consistency": method_summary["skill_consistency"],
                    "summary_path": _path_str(summary_path),
                }
            )
            records_path = out_dir / "pusht_online_records.csv"
            with open(records_path, "r", encoding="utf-8", newline="") as handle:
                for record in csv.DictReader(handle):
                    episode_rows.append(
                        {
                            "seed": seed,
                            "config_name": config_name,
                            "goal_gap": goal_gap,
                            "mode": mode,
                            **record,
                        }
                    )
    return rows, episode_rows


def _save_current_best_csv(summary_path: Path) -> Path:
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    out_path = REPORT_ROOT / "current_best_checkpoint_comparison.csv"
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "success_rate", "state_dist", "final_latent_distance", "planning_latency_sec", "skill_consistency"],
        )
        writer.writeheader()
        for method in ["hierarchical", "flat"]:
            payload = summary[method]
            writer.writerow(
                {
                    "method": method,
                    "success_rate": payload["success_rate"],
                    "state_dist": payload["state_dist"],
                    "final_latent_distance": payload["final_latent_distance"],
                    "planning_latency_sec": payload["planning_latency_sec"],
                    "skill_consistency": payload["skill_consistency"],
                }
            )
    return out_path


def aggregate_results(rows: list[dict], episode_rows: list[dict]) -> dict[str, Path]:
    ensure_dir(REPORT_ROOT)
    ensure_dir(PLOT_ROOT)
    seed_df = pd.DataFrame(rows).sort_values(["goal_gap", "config_name", "seed"]).reset_index(drop=True)
    seed_csv = REPORT_ROOT / "aggregate_seed_metrics.csv"
    seed_df.to_csv(seed_csv, index=False)
    episode_df = pd.DataFrame(episode_rows).sort_values(["goal_gap", "config_name", "seed", "method", "episode_idx"]).reset_index(
        drop=True
    )
    episode_csv = REPORT_ROOT / "aggregate_episode_records.csv"
    episode_df.to_csv(episode_csv, index=False)

    summary_df = (
        seed_df.groupby(["config_name", "goal_gap"], as_index=False)
        .agg(
            success_rate_mean=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            state_dist_mean=("state_dist", "mean"),
            state_dist_std=("state_dist", "std"),
            final_latent_distance_mean=("final_latent_distance", "mean"),
            final_latent_distance_std=("final_latent_distance", "std"),
            planning_latency_sec_mean=("planning_latency_sec", "mean"),
            planning_latency_sec_std=("planning_latency_sec", "std"),
        )
        .fillna(0.0)
    )
    summary_df["config_name"] = pd.Categorical(summary_df["config_name"], categories=CONFIG_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["goal_gap", "config_name"]).reset_index(drop=True)
    summary_csv = REPORT_ROOT / "aggregate_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    _plot_metric(summary_df, "success_rate", "Success Rate", PLOT_ROOT / "success_rate_by_goal_gap.png")
    _plot_metric(summary_df, "state_dist", "Mean Pose Distance", PLOT_ROOT / "state_distance_by_goal_gap.png")

    decision_path = _write_decision_report(seed_df, summary_df)
    return {
        "episode_csv": episode_csv,
        "seed_csv": seed_csv,
        "summary_csv": summary_csv,
        "decision_report": decision_path,
        "success_plot": PLOT_ROOT / "success_rate_by_goal_gap.png",
        "state_plot": PLOT_ROOT / "state_distance_by_goal_gap.png",
    }


def _plot_metric(summary_df: pd.DataFrame, metric_prefix: str, ylabel: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for axis, goal_gap in zip(axes, [16, 24, 32]):
        gap_df = summary_df[summary_df["goal_gap"] == goal_gap].copy()
        gap_df["config_name"] = gap_df["config_name"].astype(str)
        means = gap_df[f"{metric_prefix}_mean"].tolist()
        stds = gap_df[f"{metric_prefix}_std"].tolist()
        axis.bar(gap_df["config_name"], means, yerr=stds, capsize=4, color="#1f4e79")
        axis.set_title(f"goal_gap={goal_gap}")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _summary_row(summary_df: pd.DataFrame, config_name: str, goal_gap: int) -> pd.Series:
    row = summary_df[(summary_df["config_name"].astype(str) == config_name) & (summary_df["goal_gap"] == goal_gap)]
    if row.empty:
        raise KeyError(f"Missing summary row for {config_name} gap {goal_gap}")
    return row.iloc[0]


def _seed_delta_support(seed_df: pd.DataFrame, left: str, right: str, goal_gap: int, min_delta: float) -> float:
    wins = []
    for seed in SEEDS:
        left_row = seed_df[
            (seed_df["seed"] == seed) & (seed_df["config_name"] == left) & (seed_df["goal_gap"] == goal_gap)
        ].iloc[0]
        right_row = seed_df[
            (seed_df["seed"] == seed) & (seed_df["config_name"] == right) & (seed_df["goal_gap"] == goal_gap)
        ].iloc[0]
        wins.append(float((left_row["success_rate"] - right_row["success_rate"]) >= min_delta))
    return float(sum(wins) / len(wins))


def _write_decision_report(seed_df: pd.DataFrame, summary_df: pd.DataFrame) -> Path:
    hier24 = _summary_row(summary_df, "joint_hier_10pct", PRIMARY_GOAL_GAP)
    flat24 = _summary_row(summary_df, "joint_flat_10pct", PRIMARY_GOAL_GAP)
    random24 = _summary_row(summary_df, "random_skill_hier_10pct", PRIMARY_GOAL_GAP)
    low10_24 = _summary_row(summary_df, "labeled_only_flat_10pct", PRIMARY_GOAL_GAP)
    low100_24 = _summary_row(summary_df, "labeled_only_flat_100pct", PRIMARY_GOAL_GAP)

    mean_hier_success_gate = hier24["success_rate_mean"] >= HIER_MIN_SUCCESS
    mean_hier_flat_gate = (hier24["success_rate_mean"] - flat24["success_rate_mean"]) >= PAIRWISE_MIN_DELTA
    mean_hier_low10_gate = (hier24["success_rate_mean"] - low10_24["success_rate_mean"]) >= PAIRWISE_MIN_DELTA
    mean_hier_random_gate = (hier24["success_rate_mean"] - random24["success_rate_mean"]) >= PAIRWISE_MIN_DELTA
    mean_low100_gate = low100_24["success_rate_mean"] >= LABELED100_MIN_SUCCESS

    seed_hier_support = float(
        sum(
            float(
                seed_df[
                    (seed_df["seed"] == seed)
                    & (seed_df["config_name"] == "joint_hier_10pct")
                    & (seed_df["goal_gap"] == PRIMARY_GOAL_GAP)
                ].iloc[0]["success_rate"]
                >= HIER_MIN_SUCCESS
            )
            for seed in SEEDS
        )
        / len(SEEDS)
    )
    seed_hier_flat_support = _seed_delta_support(
        seed_df, "joint_hier_10pct", "joint_flat_10pct", PRIMARY_GOAL_GAP, PAIRWISE_MIN_DELTA
    )
    seed_hier_low10_support = _seed_delta_support(
        seed_df, "joint_hier_10pct", "labeled_only_flat_10pct", PRIMARY_GOAL_GAP, PAIRWISE_MIN_DELTA
    )
    seed_hier_random_support = _seed_delta_support(
        seed_df, "joint_hier_10pct", "random_skill_hier_10pct", PRIMARY_GOAL_GAP, PAIRWISE_MIN_DELTA
    )
    seed_low100_support = float(
        sum(
            float(
                seed_df[
                    (seed_df["seed"] == seed)
                    & (seed_df["config_name"] == "labeled_only_flat_100pct")
                    & (seed_df["goal_gap"] == PRIMARY_GOAL_GAP)
                ].iloc[0]["success_rate"]
                >= LABELED100_MIN_SUCCESS
            )
            for seed in SEEDS
        )
        / len(SEEDS)
    )
    seed_support_gate = all(
        support >= (2.0 / 3.0)
        for support in [
            seed_hier_support,
            seed_hier_flat_support,
            seed_hier_low10_support,
            seed_hier_random_support,
            seed_low100_support,
        ]
    )
    recommendation = (
        "continue"
        if all(
            [
                mean_hier_success_gate,
                mean_hier_flat_gate,
                mean_hier_low10_gate,
                mean_hier_random_gate,
                mean_low100_gate,
                seed_support_gate,
            ]
        )
        else "stop"
    )

    lines = [
        "# Skill-JEPA-WM locked Push-T pilot decision report",
        "",
        "## Recommendation",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "The locked scaling criterion uses goal_gap 24 as the primary decision point.",
        "The continue gate requires all requested success-rate thresholds to pass on the mean results and to be supported by at least two of the three seeds.",
        "",
        "## Goal-gap 24 summary",
        "",
        f"- `joint_hier_10pct`: success {hier24['success_rate_mean']:.4f} ± {hier24['success_rate_std']:.4f}, pose distance {hier24['state_dist_mean']:.2f} ± {hier24['state_dist_std']:.2f}",
        f"- `joint_flat_10pct`: success {flat24['success_rate_mean']:.4f} ± {flat24['success_rate_std']:.4f}, pose distance {flat24['state_dist_mean']:.2f} ± {flat24['state_dist_std']:.2f}",
        f"- `random_skill_hier_10pct`: success {random24['success_rate_mean']:.4f} ± {random24['success_rate_std']:.4f}, pose distance {random24['state_dist_mean']:.2f} ± {random24['state_dist_std']:.2f}",
        f"- `labeled_only_flat_10pct`: success {low10_24['success_rate_mean']:.4f} ± {low10_24['success_rate_std']:.4f}, pose distance {low10_24['state_dist_mean']:.2f} ± {low10_24['state_dist_std']:.2f}",
        f"- `labeled_only_flat_100pct`: success {low100_24['success_rate_mean']:.4f} ± {low100_24['success_rate_std']:.4f}, pose distance {low100_24['state_dist_mean']:.2f} ± {low100_24['state_dist_std']:.2f}",
        "",
        "## Seed-level robustness",
        "",
        f"- `joint_hier_10pct` success >= 15% support rate: {seed_hier_support:.3f}",
        f"- `joint_hier_10pct - joint_flat_10pct` >= 5pp support rate: {seed_hier_flat_support:.3f}",
        f"- `joint_hier_10pct - labeled_only_flat_10pct` >= 5pp support rate: {seed_hier_low10_support:.3f}",
        f"- `joint_hier_10pct - random_skill_hier_10pct` >= 5pp support rate: {seed_hier_random_support:.3f}",
        f"- `labeled_only_flat_100pct` success >= 10% support rate: {seed_low100_support:.3f}",
        "",
        "## Decision basis",
        "",
        f"- Mean `joint_hier_10pct` success >= 15%: `{mean_hier_success_gate}`",
        f"- Mean `joint_hier_10pct - joint_flat_10pct` >= 5pp: `{mean_hier_flat_gate}`",
        f"- Mean `joint_hier_10pct - labeled_only_flat_10pct` >= 5pp: `{mean_hier_low10_gate}`",
        f"- Mean `joint_hier_10pct - random_skill_hier_10pct` >= 5pp: `{mean_hier_random_gate}`",
        f"- Mean `labeled_only_flat_100pct` success >= 10%: `{mean_low100_gate}`",
        f"- 2/3 seed support across all gates: `{seed_support_gate}`",
        f"- Continue threshold met: `{recommendation == 'continue'}`",
        "",
        "## Next action",
        "",
    ]
    if recommendation == "continue":
        lines.append("- Implement the subgoal feasibility critic as a planning-time scorer and rerun the reduced 3-config suite.")
    else:
        lines.append("- Stop the line here. Do not add the subgoal feasibility critic.")
    path = REPORT_ROOT / "decision_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _collect_existing_results(seeds: list[int]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    episode_rows: list[dict] = []
    for seed in seeds:
        for config_name in CONFIG_ORDER:
            for goal_gap, _, _ in GOAL_EVALS:
                out_dir = EVAL_ROOT / f"seed_{seed}" / config_name / f"goal_gap_{goal_gap}"
                summary_path = out_dir / "pusht_online_eval.json"
                records_path = out_dir / "pusht_online_records.csv"
                if not summary_path.exists() or not records_path.exists():
                    raise FileNotFoundError(f"Missing expected evaluation artifact: {out_dir}")
                with open(summary_path, "r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                mode = {
                    "joint_hier_10pct": "hierarchical",
                    "joint_flat_10pct": "flat",
                    "labeled_only_flat_10pct": "flat",
                    "labeled_only_flat_100pct": "flat",
                    "random_skill_hier_10pct": "random_hierarchical",
                }[config_name]
                method_summary = summary[mode]
                rows.append(
                    {
                        "seed": seed,
                        "config_name": config_name,
                        "goal_gap": goal_gap,
                        "num_eval_episodes": summary["num_eval_episodes"],
                        "mode": mode,
                        "success_rate": method_summary["success_rate"],
                        "state_dist": method_summary["state_dist"],
                        "final_latent_distance": method_summary["final_latent_distance"],
                        "planning_latency_sec": method_summary["planning_latency_sec"],
                        "skill_consistency": method_summary["skill_consistency"],
                        "summary_path": _path_str(summary_path),
                    }
                )
                with open(records_path, "r", encoding="utf-8", newline="") as handle:
                    for record in csv.DictReader(handle):
                        episode_rows.append(
                            {
                                "seed": seed,
                                "config_name": config_name,
                                "goal_gap": goal_gap,
                                "mode": mode,
                                **record,
                            }
                        )
    return rows, episode_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--skip-debug-reeval", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    seeds = list(args.seeds)
    ensure_dir(OUTPUT_ROOT)

    if not args.aggregate_only:
        ensure_scaled_cache()
        if not args.skip_debug_reeval:
            current_best_summary = run_debug_reeval()
            _save_current_best_csv(current_best_summary)

        rows: list[dict] = []
        episode_rows: list[dict] = []
        for seed in seeds:
            artifacts = train_seed(seed)
            seed_rows, seed_episode_rows = evaluate_seed(seed, artifacts)
            rows.extend(seed_rows)
            episode_rows.extend(seed_episode_rows)
    else:
        rows, episode_rows = _collect_existing_results(seeds)

    aggregate_results(rows, episode_rows)


if __name__ == "__main__":
    main()
