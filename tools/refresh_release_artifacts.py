from __future__ import annotations

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
  resolver --> lowcem["LowLevelCEM: action sequence"]
  lowwm --> lowcem
  actionenc --> lowcem
  lowcem --> env["Execute first actions in PushTEnv"]
  env --> live["Live visual observation"]
  live --> clips
"""


EXPERIMENT_FLOW_MMD = """flowchart TD
  data["Local Push-T HDF5"] --> cache["Build V-JEPA2 latent cache"]
  cache --> passive["Train passive skill model"]
  cache --> low["Train low-level action WM"]
  passive --> joint["Joint finetune"]
  low --> joint
  joint --> eval["Forced online eval"]
  eval --> flat["Flat planner"]
  eval --> hier["Hierarchical planner"]
  eval --> random["Random-skill hierarchy ablation"]
  flat --> metrics["Coverage-first metrics"]
  hier --> metrics
  random --> metrics
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
    ax.set_xlim(-0.6, 13.8)
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
        "model_action": (7.0, 4.5, "ActionChunkEncoder"),
        "planner_high": (12.4, 6.9, "HighLevelCEM\nsubgoal z"),
        "planner_resolve": (12.4, 5.7, "Train-split\nnearest s"),
        "planner_low": (10.9, 4.5, "LowLevelCEM\nactions"),
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
        ("planner_resolve", "planner_low"),
        ("model_low", "planner_low"),
        ("model_action", "planner_low"),
        ("planner_low", "eval_env"),
        ("eval_env", "eval_live"),
        ("eval_live", "input_clips"),
    ]
    flow_nodes = {
        "input_data": (1.1, 5.9, "Local Push-T\nHDF5"),
        "model_cache": (3.0, 5.9, "Build V-JEPA2\nlatent cache"),
        "model_passive": (5.2, 6.6, "Passive skill\ntraining"),
        "model_low": (5.2, 5.2, "Low-level action\nWM training"),
        "model_joint": (7.2, 5.9, "Joint\nfinetune"),
        "eval_online": (9.1, 5.9, "Forced online\neval"),
        "planner_flat": (10.9, 6.8, "Flat"),
        "planner_hier": (10.9, 5.9, "Hierarchical"),
        "planner_random": (10.9, 5.0, "Random-skill\nhierarchy"),
        "eval_metrics": (12.4, 5.9, "Coverage-first\nmetrics"),
        "eval_plots": (9.1, 3.7, "Plots and\nmontage"),
        "eval_report": (7.2, 3.7, "Reliability\nreport"),
        "eval_release": (5.2, 3.7, "Clean GitHub\nrelease"),
    }
    flow_edges = [
        ("input_data", "model_cache"),
        ("model_cache", "model_passive"),
        ("model_cache", "model_low"),
        ("model_passive", "model_joint"),
        ("model_low", "model_joint"),
        ("model_joint", "eval_online"),
        ("eval_online", "planner_flat"),
        ("eval_online", "planner_hier"),
        ("eval_online", "planner_random"),
        ("planner_flat", "eval_metrics"),
        ("planner_hier", "eval_metrics"),
        ("planner_random", "eval_metrics"),
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
            "goal_state_success_rate": float(
                np.mean(
                    [
                        row.get("goal_state_success", row["success"]).lower() == "true"
                        for row in rows
                    ]
                )
            ),
            "state_dist": float(np.mean([float(row["state_dist"]) for row in rows])),
            "planning_latency_sec": float(np.mean([float(row["planning_latency_sec"]) for row in rows])),
        }
    return out


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
        payload["cache_path"] = "../jepa-wms/outputs/skill_jepa/pusht_debug/cache/pusht_vjepa2_cache.h5"
        payload["checkpoint"] = "../jepa-wms/outputs/skill_jepa/pusht_debug/joint/joint_best.pt"
        payload["coverage_threshold"] = 0.95
        for method, method_summary in summary.items():
            if method not in payload:
                continue
            payload[method]["coverage_success_rate"] = method_summary["coverage_success_rate"]
            payload[method]["goal_state_success_rate"] = method_summary["goal_state_success_rate"]
            payload[method]["unique_episode_count"] = int(method_summary["unique_episodes"])
            payload[method]["success_rate"] = method_summary["coverage_success_rate"]
            for record in payload[method].get("records", []):
                old_success = bool(record.get("success", False))
                coverage_success = bool(float(record.get("max_coverage", 0.0)) >= 0.95)
                record["goal_state_success"] = old_success
                record["coverage_success"] = coverage_success
                record["success"] = coverage_success
                record["video_path"] = _legacy_video_path(method, record.get("video_path"))
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    sanitized_records = []
    for row in records:
        cleaned = dict(row)
        method = cleaned["method"]
        goal_state_success = cleaned.get("goal_state_success", cleaned["success"])
        coverage_success = str(float(cleaned["max_coverage"]) >= 0.95)
        cleaned["goal_state_success"] = goal_state_success
        cleaned["coverage_success"] = coverage_success
        cleaned["success"] = coverage_success
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
        "success",
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
                "goal_state_success_rate",
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
                    "goal_state_success_rate": row["goal_state_success_rate"],
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
        "methods": summary,
    }
    (out_dir / "pusht_online_eval_sanitized.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_plots(records: list[dict], summary: dict[str, dict[str, float]]) -> dict[str, Path]:
    plot_dir = ensure_dir(RELEASE / "plots")
    methods = ["flat", "hierarchical"]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - 0.18, [summary[m]["coverage_success_rate"] for m in methods], width=0.36, label="Coverage success")
    ax.bar(x + 0.18, [summary[m]["goal_state_success_rate"] for m in methods], width=0.36, label="Goal-state success")
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 0.12)
    ax.set_ylabel("Rate")
    ax.set_title("Legacy Locked Eval: Coverage vs Goal-State Success")
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
    ax.set_ylabel("Final pose distance")
    ax.set_title("Paired Final Pose Distance")
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


def copy_phase_a_if_present() -> dict | None:
    summary_path = PHASE_A_OUTPUT / "pusht_online_eval.json"
    records_path = PHASE_A_OUTPUT / "pusht_online_records.csv"
    if not summary_path.exists() or not records_path.exists():
        artifact_summary = PHASE_A_ARTIFACT / "pusht_online_eval.json"
        if artifact_summary.exists():
            with open(artifact_summary, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return None
    ensure_dir(PHASE_A_ARTIFACT)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
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
        for record in summary[method].get("records", []):
            video_path = record.get("video_path")
            if video_path:
                record["video_path"] = video_map.get(video_path, f"{PHASE_A_ARTIFACT.relative_to(ROOT).as_posix()}/videos/{method}/{Path(video_path).name}")
    (PHASE_A_ARTIFACT / "pusht_online_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with open(records_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
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
            f"subgoal_scope={phase_a.get('subgoal_scope')}."
        )
        phase_a_table = [
            "",
            "## Phase A Fresh Eval",
            "",
            "| Method | Coverage success | Goal-state success | Mean pose distance | Mean latency |",
            "|---|---:|---:|---:|---:|",
        ]
        for method in ["flat", "hierarchical"]:
            if method in phase_a:
                row = phase_a[method]
                phase_a_table.append(
                    f"| {method} | {row['coverage_success_rate']:.2f} | {row['goal_state_success_rate']:.2f} | "
                    f"{row['state_dist']:.2f} | {row['planning_latency_sec']:.3f}s |"
                )
    lines = [
        "# Skill-JEPA-WM Push-T Reliability Report",
        "",
        "## Verdict",
        "",
        "Use coverage success as the primary Push-T metric. The legacy locked artifact does not support a standard Push-T success claim: both flat and hierarchical have 0.00 coverage success.",
        "",
        "## Legacy Locked Artifact Re-Score",
        "",
        "| Method | Coverage success | Goal-state success | Unique episodes | Mean pose distance | Mean latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ["flat", "hierarchical"]:
        row = summary[method]
        lines.append(
            f"| {method} | {row['coverage_success_rate']:.2f} | {row['goal_state_success_rate']:.2f} | "
            f"{int(row['unique_episodes'])} | {row['state_dist']:.2f} | {row['planning_latency_sec']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "The old `success_rate` column measured pose-to-sampled-goal success, not standard Push-T coverage success.",
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
            ("Coverage vs Goal-State Success", plots["coverage_plot"]),
            ("Paired Pose Distance", plots["distance_plot"]),
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
        table_text = "\n".join(lines[:16] + ["", phase_a_text])
        ax.text(0.02, 0.98, table_text, va="top", family="monospace", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    write_mmd()
    render_diagrams()
    records = read_locked_records()
    summary = summarize_records(records)
    records = sanitize_locked_in_place(records, summary)
    write_sanitized_locked_artifacts(records, summary)
    plots = write_plots(records, summary)
    montage = write_montage()
    phase_a = copy_phase_a_if_present()
    write_report(summary, plots, montage, phase_a)


if __name__ == "__main__":
    main()
