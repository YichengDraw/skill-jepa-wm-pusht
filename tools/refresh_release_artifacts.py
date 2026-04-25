from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "architecture"
RELEASE = ROOT / "artifacts" / "release"
LOCKED_EVAL = ROOT / "artifacts" / "pusht_locked_suite" / "evals" / "current_best_checkpoint_100ep"
LOCKED_REPORT = ROOT / "artifacts" / "pusht_locked_suite" / "reports" / "current_best_checkpoint_comparison.csv"
PHASE_A_ROOT = ROOT / "outputs" / "skill_jepa" / "phase_a_current_checkpoint"
PHASE_A_OUTPUT = PHASE_A_ROOT / "eval"
PHASE_A_ARTIFACT = ROOT / "artifacts" / "phase_a_current_checkpoint" / "evals"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


MODEL_PIPELINE_MMD = """flowchart TD
  raw["Raw Push-T HDF5: pixels, actions, states"] --> clips["Causal 2-frame clips"]
  clips --> encoder["Frozen V-JEPA2 encoder"]
  encoder --> projector["StateProjector"]
  projector --> z["z_t global latent"]
  projector --> s["s_t spatial tokens"]
  z --> skill["SkillIDM, SkillPrior, SkillWorldModel"]
  s --> lowwm["LowLevelWM"]
  raw --> actions["Primitive action chunks"]
  actions --> actionenc["ActionChunkEncoder"]
  skill --> highcem["HighLevelCEM: skill sequence to subgoal z"]
  highcem --> resolver["Train-split nearest cached s"]
  resolver --> hierlow["Hierarchical LowLevelCEM"]
  s --> flatlow["Flat LowLevelCEM with goal_s"]
  lowwm --> hierlow
  lowwm --> flatlow
  actionenc --> hierlow
  actionenc --> flatlow
  hierlow --> env["Execute first actions in PushTEnv"]
  flatlow --> env
  env --> live["Live visual observation"]
  live --> clips
"""


EXPERIMENT_FLOW_MMD = """flowchart TD
  data["Local Push-T HDF5"] --> cache["Build V-JEPA2 latent cache"]
  cache --> passive["Train passive skill model"]
  passive --> low10["Train low-level WM: 10 percent labels"]
  passive --> low100["Train low-level WM: 100 percent labels"]
  low10 --> joint["Joint finetune: 10 percent labels"]
  passive --> joint
  cache --> goals["Goal mode selection before rollout"]
  goals --> eval_joint["Forced online eval: joint checkpoint"]
  goals --> eval_low["Forced online eval: low-level checkpoints"]
  joint --> eval_joint
  low10 --> eval_low
  low100 --> eval_low
  eval_joint --> flat["Joint flat planner"]
  eval_joint --> hier["Joint hierarchical planner"]
  eval_joint --> random["Random-skill hierarchy ablation"]
  eval_low --> lowflat["Labeled-only flat baselines"]
  flat --> metrics["Coverage primary plus goal-state diagnostic"]
  hier --> metrics
  random --> metrics
  lowflat --> metrics
  metrics --> plots["Plots and rollout montage"]
  plots --> report["Reliability report"]
  report --> release["Clean GitHub release"]
"""


def write_mmd() -> None:
    ensure_dir(DOCS)
    (DOCS / "model_pipeline.mmd").write_text(MODEL_PIPELINE_MMD, encoding="utf-8")
    (DOCS / "experiment_flow.mmd").write_text(EXPERIMENT_FLOW_MMD, encoding="utf-8")


def draw_flow(title: str, nodes: dict[str, tuple[float, float, str]], edges: list[tuple[str, str]], out_base: Path) -> None:
    ensure_dir(out_base.parent)
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_axis_off()
    ax.set_xlim(-0.6, 14.8)
    ax.set_ylim(0, 7.4)
    colors = {
        "input": "#eef4f7",
        "latent": "#f8f0d9",
        "model": "#e7f1df",
        "planner": "#f5e7e2",
        "eval": "#e8e7f4",
    }
    for name, (x, y, label) in nodes.items():
        group = name.split("_", 1)[0]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=colors.get(group, "#ffffff"), edgecolor="#333333", linewidth=1.0),
        )
    for start, end in edges:
        x0, y0, _ = nodes[start]
        x1, y1, _ = nodes[end]
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#333333",
            shrinkA=35,
            shrinkB=35,
        )
        ax.add_patch(arrow)
    ax.set_title(title, fontsize=15, pad=12)
    svg_path = out_base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def render_diagrams() -> None:
    model_nodes = {
        "input_raw": (1.2, 6.3, "Push-T HDF5\npixels/actions/state"),
        "input_clips": (3.1, 6.3, "Causal 2-frame\nclip builder"),
        "model_encoder": (5.1, 6.3, "Frozen V-JEPA2\nencoder"),
        "model_projector": (7.0, 6.3, "StateProjector"),
        "latent_z": (8.9, 6.9, "z_t global\nlatent"),
        "latent_s": (8.9, 5.7, "s_t spatial\ntokens"),
        "model_skill": (10.9, 6.9, "SkillIDM / Prior /\nSkillWorldModel"),
        "model_low": (10.9, 5.7, "LowLevelWM"),
        "input_actions": (5.1, 4.5, "Action chunks"),
        "model_action": (7.0, 4.5, "ActionChunk\nEncoder"),
        "planner_high": (12.4, 6.9, "HighLevelCEM\nsubgoal z"),
        "planner_resolve": (12.4, 5.7, "Train-split\nnearest s"),
        "planner_hier_low": (12.4, 4.5, "Hierarchical\nLowLevelCEM"),
        "planner_flat_low": (10.2, 4.4, "Flat LowLevelCEM\ndirect goal_s"),
        "eval_env": (7.0, 3.2, "PushTEnv\nexecute actions"),
        "eval_live": (3.1, 3.2, "Live visual obs\nre-encode"),
    }
    model_edges = [
        ("input_raw", "input_clips"),
        ("input_clips", "model_encoder"),
        ("model_encoder", "model_projector"),
        ("model_projector", "latent_z"),
        ("model_projector", "latent_s"),
        ("latent_z", "model_skill"),
        ("latent_s", "model_low"),
        ("input_raw", "input_actions"),
        ("input_actions", "model_action"),
        ("model_skill", "planner_high"),
        ("planner_high", "planner_resolve"),
        ("planner_resolve", "planner_hier_low"),
        ("latent_s", "planner_flat_low"),
        ("model_low", "planner_hier_low"),
        ("model_low", "planner_flat_low"),
        ("model_action", "planner_hier_low"),
        ("model_action", "planner_flat_low"),
        ("planner_hier_low", "eval_env"),
        ("planner_flat_low", "eval_env"),
        ("eval_env", "eval_live"),
        ("eval_live", "input_clips"),
    ]
    flow_nodes = {
        "input_data": (0.9, 6.2, "Local Push-T\nHDF5"),
        "model_cache": (2.7, 6.2, "Build V-JEPA2\nlatent cache"),
        "model_passive": (4.7, 6.8, "Passive skill\ntraining"),
        "model_low10": (6.8, 6.8, "Low-level WM\n10% labels"),
        "model_low100": (6.8, 5.6, "Low-level WM\n100% labels"),
        "model_joint": (8.9, 6.5, "Joint finetune\n10% labels"),
        "eval_goals": (4.7, 5.0, "Goal mode before rollout\ntask or trajectory"),
        "eval_joint": (10.9, 6.5, "Forced eval\njoint ckpt"),
        "eval_low": (8.9, 4.7, "Forced eval\nlow-level ckpts"),
        "planner_flat": (12.8, 7.0, "Joint flat\nplanner"),
        "planner_hier": (12.8, 6.1, "Joint hierarchical\nplanner"),
        "planner_random": (12.8, 5.2, "Random-skill\nablation"),
        "planner_lowflat": (12.8, 4.3, "Labeled-only\nflat baselines"),
        "eval_metrics": (13.7, 3.3, "Coverage primary\nplus pose diagnostic"),
        "eval_plots": (11.6, 2.7, "Aggregate CSV\nand plots"),
        "eval_report": (9.4, 2.7, "Reliability\nreport"),
        "eval_release": (7.2, 2.7, "Tracked release\nartifacts"),
    }
    flow_edges = [
        ("input_data", "model_cache"),
        ("model_cache", "model_passive"),
        ("model_passive", "model_low10"),
        ("model_passive", "model_low100"),
        ("model_low10", "model_joint"),
        ("model_passive", "model_joint"),
        ("model_cache", "eval_goals"),
        ("eval_goals", "eval_joint"),
        ("eval_goals", "eval_low"),
        ("model_joint", "eval_joint"),
        ("model_low10", "eval_low"),
        ("model_low100", "eval_low"),
        ("eval_joint", "planner_flat"),
        ("eval_joint", "planner_hier"),
        ("eval_joint", "planner_random"),
        ("eval_low", "planner_lowflat"),
        ("planner_flat", "eval_metrics"),
        ("planner_hier", "eval_metrics"),
        ("planner_random", "eval_metrics"),
        ("planner_lowflat", "eval_metrics"),
        ("eval_metrics", "eval_plots"),
        ("eval_plots", "eval_report"),
        ("eval_report", "eval_release"),
    ]
    draw_flow("Skill-JEPA-WM Push-T Model Pipeline", model_nodes, model_edges, DOCS / "model_pipeline")
    draw_flow("Skill-JEPA-WM Push-T Experiment Flow", flow_nodes, flow_edges, DOCS / "experiment_flow")


def read_locked_records() -> list[dict]:
    records_path = LOCKED_EVAL / "pusht_online_records.csv"
    with open(records_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_records(records: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in records}):
        rows = [row for row in records if row["method"] == method]
        out[method] = {
            "n": float(len(rows)),
            "unique_episodes": float(len({row["episode_id"] for row in rows})),
            "coverage_success_rate": float(np.mean([float(row["max_coverage"]) >= 0.95 for row in rows])),
            "goal_state_success_diagnostic_rate": float(
                np.mean(
                    [
                        (row.get("goal_state_success") or row.get("success", "false")).lower() == "true"
                        for row in rows
                    ]
                )
            ),
            "state_dist": float(np.mean([float(row["state_dist"]) for row in rows])),
            "planning_latency_sec": float(np.mean([float(row["planning_latency_sec"]) for row in rows])),
        }
    return out


def strip_success_aliases(summary: dict) -> dict:
    summary.setdefault("deterministic_timing", False)
    summary["success_semantics"] = (
        "coverage_success and coverage_success_rate are Push-T coverage metrics; "
        "goal_state_success is diagnostic only"
    )
    for method in ["flat", "hierarchical", "random_hierarchical"]:
        payload = summary.get(method)
        if not isinstance(payload, dict):
            continue
        if "coverage_success_rate" not in payload and "success_rate" in payload:
            payload["coverage_success_rate"] = payload["success_rate"]
        if "goal_state_success_rate" in payload:
            payload["goal_state_success_diagnostic_rate"] = payload.pop("goal_state_success_rate")
        payload.pop("success_rate", None)
        payload["goal_state_success_is_task_metric"] = False
        for record in payload.get("records", []):
            record.pop("success", None)
    return summary


def strip_success_column(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "success"} for row in rows]


def _legacy_video_path(method: str, old_path: str | None) -> str | None:
    if not old_path:
        return None
    name = Path(old_path).name
    stem = Path(name).stem
    if stem.startswith("episode_"):
        return f"artifacts/pusht_locked_suite/visuals/{method}_{name}"
    return name


def sanitize_locked_in_place(records: list[dict], summary: dict[str, dict[str, float]]) -> list[dict]:
    summary_path = LOCKED_EVAL / "pusht_online_eval.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["cache_path"] = "external/legacy_debug_cache.h5"
        payload["checkpoint"] = "external/legacy_debug_joint_best.pt"
        payload["external_inputs_not_committed"] = True
        payload["legacy_artifact"] = True
        payload["coverage_threshold"] = 0.95
        payload["goal_mode"] = "trajectory"
        payload["task_success_claim_supported"] = False
        payload["deterministic_timing"] = False
        payload["unique_episode_count"] = int(max((row["unique_episodes"] for row in summary.values()), default=0))
        payload["provenance"] = {
            "warnings": ["Legacy locked artifact predates the current strict provenance schema; external inputs are not committed"]
        }
        payload["success_semantics"] = "coverage_success and coverage_success_rate are coverage metrics after reliability re-score"
        for method, method_summary in summary.items():
            if method not in payload:
                continue
            payload[method]["coverage_success_rate"] = method_summary["coverage_success_rate"]
            payload[method].pop("goal_state_success_rate", None)
            payload[method]["goal_state_success_diagnostic_rate"] = method_summary["goal_state_success_diagnostic_rate"]
            payload[method]["goal_state_success_is_task_metric"] = False
            payload[method]["goal_state_success_scope"] = "trajectory_full_state_diagnostic"
            payload[method]["unique_episode_count"] = int(method_summary["unique_episodes"])
            payload[method].pop("success_rate", None)
            for record in payload[method].get("records", []):
                old_success = bool(record.get("success", False))
                coverage_success = bool(float(record.get("max_coverage", 0.0)) >= 0.95)
                record["goal_state_success"] = old_success
                record["coverage_success"] = coverage_success
                record.pop("success", None)
                record["video_path"] = _legacy_video_path(method, record.get("video_path"))
        strip_success_aliases(payload)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    sanitized_records = []
    for row in records:
        cleaned = dict(row)
        method = cleaned["method"]
        goal_state_success = cleaned.get("goal_state_success") or cleaned.get("success", "False")
        coverage_success = str(float(cleaned["max_coverage"]) >= 0.95)
        cleaned["goal_state_success"] = goal_state_success
        cleaned["coverage_success"] = coverage_success
        cleaned.pop("success", None)
        cleaned["video_path"] = _legacy_video_path(method, cleaned.get("video_path")) or ""
        sanitized_records.append(cleaned)
    records_path = LOCKED_EVAL / "pusht_online_records.csv"
    fieldnames = [
        "method",
        "episode_idx",
        "episode_id",
        "start_index",
        "goal_index",
        "sampled_goal_gap",
        "coverage_success",
        "goal_state_success",
        "state_dist",
        "final_latent_distance",
        "start_latent_distance",
        "planning_latency_sec",
        "max_coverage",
        "final_coverage",
        "steps_taken",
        "skill_consistency",
        "video_path",
    ]
    with open(records_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sanitized_records:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    with open(LOCKED_REPORT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "coverage_success_rate",
                "goal_state_success_diagnostic_rate",
                "state_dist",
                "planning_latency_sec",
            ],
        )
        writer.writeheader()
        for method in ["hierarchical", "flat"]:
            row = summary[method]
            writer.writerow(
                {
                    "method": method,
                    "coverage_success_rate": row["coverage_success_rate"],
                    "goal_state_success_diagnostic_rate": row["goal_state_success_diagnostic_rate"],
                    "state_dist": row["state_dist"],
                    "planning_latency_sec": row["planning_latency_sec"],
                }
            )
    return sanitized_records


def write_sanitized_locked_artifacts(records: list[dict], summary: dict[str, dict[str, float]]) -> None:
    out_dir = ensure_dir(RELEASE / "sanitized_locked_artifacts")
    records_out = out_dir / "pusht_online_records_sanitized.csv"
    with open(records_out, "w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            cleaned = dict(row)
            if cleaned.get("video_path"):
                cleaned["video_path"] = Path(cleaned["video_path"]).name
            writer.writerow(cleaned)
    payload = {
        "source": "legacy locked artifact re-scored with coverage-first semantics",
        "coverage_threshold": 0.95,
        "goal_state_success_scope": "trajectory_full_state_diagnostic",
        "goal_state_success_is_task_metric": False,
        "methods": summary,
    }
    (out_dir / "pusht_online_eval_sanitized.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_plots(records: list[dict], summary: dict[str, dict[str, float]]) -> dict[str, Path]:
    plot_dir = ensure_dir(RELEASE / "plots")
    methods = ["flat", "hierarchical"]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - 0.18, [summary[m]["coverage_success_rate"] for m in methods], width=0.36, label="Coverage success")
    ax.bar(
        x + 0.18,
        [summary[m]["goal_state_success_diagnostic_rate"] for m in methods],
        width=0.36,
        label="Goal-state diagnostic",
    )
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 0.12)
    ax.set_ylabel("Rate")
    ax.set_title("Legacy Locked Eval: Coverage vs Goal-State Diagnostic")
    ax.legend()
    fig.tight_layout()
    coverage_plot = plot_dir / "coverage_vs_goal_state_success.png"
    fig.savefig(coverage_plot, dpi=180)
    plt.close(fig)

    paired = {}
    for method in methods:
        paired[method] = {
            int(row["episode_idx"]): float(row["state_dist"])
            for row in records
            if row["method"] == method
        }
    shared_idx = sorted(set(paired["flat"]) & set(paired["hierarchical"]))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(shared_idx, [paired["flat"][idx] for idx in shared_idx], label="Flat", linewidth=1.3)
    ax.plot(shared_idx, [paired["hierarchical"][idx] for idx in shared_idx], label="Hierarchical", linewidth=1.3)
    ax.set_xlabel("Sampled pair index")
    ax.set_ylabel("Final sampled-state distance")
    ax.set_title("Paired Final Sampled-State Distance")
    ax.legend()
    fig.tight_layout()
    distance_plot = plot_dir / "paired_state_distance.png"
    fig.savefig(distance_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(methods, [summary[m]["planning_latency_sec"] for m in methods], color=["#7f8b52", "#8f5f4f"])
    ax.set_ylabel("Seconds per planning call")
    ax.set_title("Mean Planning Latency")
    fig.tight_layout()
    latency_plot = plot_dir / "planning_latency.png"
    fig.savefig(latency_plot, dpi=180)
    plt.close(fig)

    return {
        "coverage_plot": coverage_plot,
        "distance_plot": distance_plot,
        "latency_plot": latency_plot,
    }


def write_montage() -> Path | None:
    visual_dir = ROOT / "artifacts" / "pusht_locked_suite" / "visuals"
    gifs = [
        ("Flat 000", visual_dir / "flat_episode_000.gif"),
        ("Flat 001", visual_dir / "flat_episode_001.gif"),
        ("Hierarchical 000", visual_dir / "hierarchical_episode_000.gif"),
        ("Hierarchical 001", visual_dir / "hierarchical_episode_001.gif"),
    ]
    existing = [(label, path) for label, path in gifs if path.exists()]
    if not existing:
        return None
    fig, axes = plt.subplots(1, len(existing), figsize=(3.2 * len(existing), 3.4))
    if len(existing) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, existing):
        frame = imageio.mimread(path, memtest=False)[0]
        ax.imshow(frame)
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    out_path = ensure_dir(RELEASE / "plots") / "rollout_montage.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def copy_phase_a_if_present(ingest_local_outputs: bool = False) -> dict | None:
    summary_path = PHASE_A_OUTPUT / "pusht_online_eval.json"
    records_path = PHASE_A_OUTPUT / "pusht_online_records.csv"
    if not ingest_local_outputs or not summary_path.exists() or not records_path.exists():
        artifact_summary = PHASE_A_ARTIFACT / "pusht_online_eval.json"
        if artifact_summary.exists():
            with open(artifact_summary, "r", encoding="utf-8") as handle:
                summary = strip_success_aliases(json.load(handle))
            artifact_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            artifact_records = PHASE_A_ARTIFACT / "pusht_online_records.csv"
            if artifact_records.exists():
                with open(artifact_records, "r", encoding="utf-8", newline="") as handle:
                    rows = strip_success_column(list(csv.DictReader(handle)))
                if rows:
                    with open(artifact_records, "w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
            return summary
        return None
    ensure_dir(PHASE_A_ARTIFACT)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["external_inputs_not_committed"] = True
    summary["code_commit_semantics"] = "code_commit records the evaluator source commit at run time; the artifact commit is necessarily later"
    summary["cache_path"] = "external/phase_a_debug_cache.h5"
    summary["checkpoint"] = "external/phase_a_debug_joint_best.pt"
    summary.setdefault("deterministic_timing", False)
    summary["success_semantics"] = "coverage_success and coverage_success_rate are Push-T coverage metrics; goal_state_success is diagnostic only"
    summary.setdefault("portable_paths", {})["cache_path"] = "external/phase_a_debug_cache.h5"
    summary.setdefault("portable_paths", {})["checkpoint"] = "external/phase_a_debug_joint_best.pt"
    summary.setdefault("portable_paths", {})["projector"] = "external/phase_a_debug_state_projector.pt"
    cfg_path = PHASE_A_ROOT / "phase_a_external_debug.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as handle:
            phase_cfg = yaml.safe_load(handle)
        phase_cfg["data"]["cache_path"] = summary.get("cache_path")
        phase_cfg["data"]["projector_ckpt"] = summary.get("portable_paths", {}).get("projector")
        with open(PHASE_A_ARTIFACT / "phase_a_external_debug.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(phase_cfg, handle, sort_keys=False)
        summary.setdefault("portable_paths", {})["config"] = (PHASE_A_ARTIFACT / "phase_a_external_debug.yaml").relative_to(ROOT).as_posix()
    video_map: dict[str, str] = {}
    source_video_root = PHASE_A_OUTPUT / "videos"
    target_video_root = ensure_dir(PHASE_A_ARTIFACT / "videos")
    if source_video_root.exists():
        for source in source_video_root.rglob("*.gif"):
            relative = source.relative_to(source_video_root)
            target = target_video_root / relative
            ensure_dir(target.parent)
            shutil.copy2(source, target)
            video_map[source.as_posix()] = target.relative_to(ROOT).as_posix()
            video_map[str(source.relative_to(ROOT).as_posix())] = target.relative_to(ROOT).as_posix()
    for method in ["flat", "hierarchical", "random_hierarchical"]:
        if method not in summary:
            continue
        if "goal_state_success_rate" in summary[method]:
            summary[method]["goal_state_success_diagnostic_rate"] = summary[method].pop("goal_state_success_rate")
        summary[method]["goal_state_success_is_task_metric"] = False
        for record in summary[method].get("records", []):
            video_path = record.get("video_path")
            if video_path:
                record["video_path"] = video_map.get(video_path, f"{PHASE_A_ARTIFACT.relative_to(ROOT).as_posix()}/videos/{method}/{Path(video_path).name}")
            record.pop("success", None)
        summary[method].pop("success_rate", None)
    strip_success_aliases(summary)
    (PHASE_A_ARTIFACT / "pusht_online_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with open(records_path, "r", encoding="utf-8", newline="") as handle:
        rows = strip_success_column(list(csv.DictReader(handle)))
    if rows:
        with open(PHASE_A_ARTIFACT / "pusht_online_records.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                video_path = row.get("video_path")
                if video_path:
                    row["video_path"] = video_map.get(
                        video_path,
                        f"{PHASE_A_ARTIFACT.relative_to(ROOT).as_posix()}/videos/{row['method']}/{Path(video_path).name}",
                    )
                writer.writerow(row)
    return summary


def write_report(summary: dict[str, dict[str, float]], plots: dict[str, Path], montage: Path | None, phase_a: dict | None) -> None:
    ensure_dir(RELEASE)
    md_path = RELEASE / "skill_jepa_wm_reliability_report.md"
    pdf_path = RELEASE / "skill_jepa_wm_reliability_report.pdf"
    phase_a_text = "Phase A fresh eval has not been copied into artifacts yet."
    phase_a_table: list[str] = []
    if phase_a is not None:
        phase_a_text = (
            f"Phase A fresh eval: {phase_a.get('num_eval_episodes', 0)} sampled pairs, "
            f"{phase_a.get('unique_episode_count', 0)} unique episodes, "
            f"requested_split={phase_a.get('requested_eval_split', phase_a.get('eval_split'))}, "
            f"actual_split={phase_a.get('eval_split')}, "
            f"subgoal_scope={phase_a.get('subgoal_scope')}, "
            f"goal_mode={phase_a.get('goal_mode', 'legacy')}, "
            f"task_success_claim_supported={phase_a.get('task_success_claim_supported', False)}, "
            f"under_sampled={phase_a.get('under_sampled', False)}, "
            f"provenance_warnings={len(phase_a.get('provenance', {}).get('warnings', []))}."
        )
        phase_a_table = [
            "",
            "## Phase A Fresh Eval",
            "",
            "| Method | Coverage diagnostic | Goal-state diagnostic | Mean sampled-state distance | Mean latency |",
            "|---|---:|---:|---:|---:|",
        ]
        for method in ["flat", "hierarchical"]:
            if method in phase_a:
                row = phase_a[method]
                goal_rate = row.get("goal_state_success_rate", row.get("goal_state_success_diagnostic_rate", 0.0))
                phase_a_table.append(
                    f"| {method} | {row['coverage_success_rate']:.2f} | {goal_rate:.2f} | "
                    f"{row['state_dist']:.2f} | {row['planning_latency_sec']:.3f}s |"
                )
    lines = [
        "# Skill-JEPA-WM Push-T Reliability Report",
        "",
        "## Verdict",
        "",
        "Use task-aligned coverage success as the primary Push-T metric. The tracked debug and legacy locked artifacts are trajectory-goal diagnostics and do not support a standard Push-T task-success claim.",
        "",
        "## Legacy Locked Artifact Re-Score",
        "",
        "| Method | Coverage success | Goal-state diagnostic | Unique episodes | Mean sampled-state distance | Mean latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["flat", "hierarchical"]:
        row = summary[method]
        lines.append(
            f"| {method} | {row['coverage_success_rate']:.2f} | {row['goal_state_success_diagnostic_rate']:.2f} | "
            f"{int(row['unique_episodes'])} | {row['state_dist']:.2f} | {row['planning_latency_sec']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "Legacy pre-rescore outputs used `success_rate` for sampled-trajectory goal-state success; current tracked eval exports use coverage-specific names.",
            phase_a_text,
            *phase_a_table,
            "",
            "## Figures",
            "",
            f"- Architecture: `docs/architecture/model_pipeline.svg`",
            f"- Experiment flow: `docs/architecture/experiment_flow.svg`",
            f"- Coverage vs goal-state plot: `{plots['coverage_plot'].relative_to(ROOT).as_posix()}`",
            f"- Paired distance plot: `{plots['distance_plot'].relative_to(ROOT).as_posix()}`",
            f"- Latency plot: `{plots['latency_plot'].relative_to(ROOT).as_posix()}`",
        ]
    )
    if montage is not None:
        lines.append(f"- Rollout montage: `{montage.relative_to(ROOT).as_posix()}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        for title, image_path in [
            ("Model Pipeline", DOCS / "model_pipeline.png"),
            ("Experiment Flow", DOCS / "experiment_flow.png"),
            ("Coverage vs Goal-State Diagnostic", plots["coverage_plot"]),
            ("Paired Sampled-State Distance", plots["distance_plot"]),
            ("Planning Latency", plots["latency_plot"]),
        ]:
            fig, ax = plt.subplots(figsize=(11, 7.2))
            ax.imshow(plt.imread(image_path))
            ax.set_title(title, fontsize=15)
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        if montage is not None:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.imshow(plt.imread(montage))
            ax.set_title("Tracked Rollout Frames", fontsize=15)
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 7.2))
        ax.set_axis_off()
        table_text = "\n".join(lines[:16] + phase_a_table)
        ax.text(0.02, 0.98, table_text, va="top", family="monospace", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest-local-phase-a",
        action="store_true",
        help="Copy Phase A outputs from ignored outputs/ into tracked artifacts; default is reproducible tracked-only refresh.",
    )
    args = parser.parse_args()
    write_mmd()
    render_diagrams()
    records = read_locked_records()
    summary = summarize_records(records)
    records = sanitize_locked_in_place(records, summary)
    write_sanitized_locked_artifacts(records, summary)
    plots = write_plots(records, summary)
    montage = write_montage()
    phase_a = copy_phase_a_if_present(ingest_local_outputs=bool(args.ingest_local_phase_a))
    write_report(summary, plots, montage, phase_a)


if __name__ == "__main__":
    main()
