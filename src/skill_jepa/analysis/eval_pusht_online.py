from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
from skill_jepa.data import EpisodeGoalSampler, cache_metadata, split_episode_ids
from skill_jepa.encoders import FrozenVJEPA2Encoder
from skill_jepa.modules import StateProjector
from skill_jepa.planning import HighLevelCEMPlanner, HierarchicalPlanner, LowLevelCEMPlanner, RandomHighLevelPlanner
from skill_jepa.trainers.common import build_all_modules, load_checkpoint, modules_to_device
from skill_jepa.utils import dump_json, ensure_dir, load_yaml, seed_everything


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class OnlineEvalRecord:
    episode_idx: int
    episode_id: int
    start_index: int
    goal_index: int
    sampled_goal_gap: int
    success: bool
    coverage_success: bool
    goal_state_success: bool
    state_dist: float
    final_latent_distance: float
    start_latent_distance: float
    planning_latency_sec: float
    max_coverage: float
    final_coverage: float
    steps_taken: int
    skill_consistency: float
    video_path: str | None


def _load_cache_latents(cache_path: str, start_index: int, goal_index: int, device: torch.device) -> dict[str, torch.Tensor]:
    with h5py.File(cache_path, "r") as handle:
        return {
            "start_z": torch.from_numpy(handle["z"][start_index]).float().to(device),
            "start_s": torch.from_numpy(handle["s"][start_index]).float().to(device),
            "goal_z": torch.from_numpy(handle["z"][goal_index]).float().to(device),
            "goal_s": torch.from_numpy(handle["s"][goal_index]).float().to(device),
        }


def _set_eval_seed(seed: int, env: PushTEnv | None = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


def _sha256_file(path: str | Path | None) -> str | None:
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


def _git_commit() -> str | None:
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


def _portable_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    for base, prefix in [(ROOT, ""), (ROOT.parent, "../")]:
        try:
            relative = resolved.relative_to(base.resolve()).as_posix()
            return f"{prefix}{relative}" if prefix else relative
        except ValueError:
            continue
    return resolved.name


class NearestSubgoalResolver:
    def __init__(
        self,
        cache_path: str,
        device: torch.device,
        chunk_size: int = 4096,
        allowed_indices: np.ndarray | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.device = device
        self.chunk_size = chunk_size
        self.allowed_indices = None if allowed_indices is None else np.asarray(allowed_indices, dtype=np.int64)

    @torch.no_grad()
    def __call__(self, target_z: torch.Tensor) -> torch.Tensor:
        best_s = None
        best_cost = None
        with h5py.File(self.cache_path, "r") as handle:
            z_ds = handle["z"]
            s_ds = handle["s"]
            indices = self.allowed_indices
            if indices is None:
                indices = np.arange(z_ds.shape[0], dtype=np.int64)
            for start in range(0, len(indices), self.chunk_size):
                chunk_indices = indices[start : start + self.chunk_size]
                z_chunk = torch.from_numpy(z_ds[chunk_indices]).float().to(self.device)
                distances = (z_chunk - target_z.unsqueeze(0)).pow(2).mean(dim=-1)
                chunk_idx = int(torch.argmin(distances).item())
                chunk_cost = float(distances[chunk_idx].item())
                if best_cost is None or chunk_cost < best_cost:
                    best_cost = chunk_cost
                    best_s = torch.from_numpy(s_ds[int(chunk_indices[chunk_idx])]).float().to(self.device)
        return best_s


def _goal_state_eval(goal_state: np.ndarray, cur_state: np.ndarray) -> dict[str, float | bool]:
    pose_goal = goal_state[:5]
    pose_cur = cur_state[:5]
    pos_diff = float(np.linalg.norm(pose_goal[:4] - pose_cur[:4]))
    angle_delta = float(pose_cur[4] - pose_goal[4])
    angle_diff = float(abs((angle_delta + np.pi) % (2 * np.pi) - np.pi))
    pose_delta = np.concatenate([pose_goal[:4] - pose_cur[:4], np.asarray([angle_diff], dtype=np.float32)])
    return {
        "goal_state_success": bool(pos_diff < 20.0 and angle_diff < (np.pi / 9.0)),
        "state_dist": float(np.linalg.norm(pose_delta)),
    }


def _coverage_success(info: dict, threshold: float) -> bool:
    return bool(float(info.get("max_coverage", 0.0)) >= threshold)


@torch.no_grad()
def _encode_clip(
    prev_frame: np.ndarray,
    frame: np.ndarray,
    encoder: FrozenVJEPA2Encoder,
    projector: StateProjector,
) -> tuple[torch.Tensor, torch.Tensor]:
    clip = torch.from_numpy(np.stack([prev_frame, frame], axis=0)).unsqueeze(0)
    patch_tokens = encoder.encode_images(clip)
    states = projector(patch_tokens)
    return states["global_state"][0], states["spatial_tokens"][0]


def _split_step_indices(cache_path: str, split: str, cfg: dict) -> np.ndarray:
    with h5py.File(cache_path, "r") as handle:
        ep_len = handle["ep_len"][:]
        ep_offset = handle["ep_offset"][:]
    split_ids = split_episode_ids(
        len(ep_len),
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
        int(cfg["seed"]),
    )[split]
    indices = []
    for episode_id in split_ids.tolist():
        offset = int(ep_offset[episode_id])
        length = int(ep_len[episode_id])
        indices.append(np.arange(offset, offset + length, dtype=np.int64))
    if not indices:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(indices)


def _build_planners(
    cfg: dict,
    modules: dict,
    meta: dict,
    device: torch.device,
    subgoal_scope: str = "train",
) -> dict[str, object]:
    planner_cfg = cfg["planner"]
    action_low = torch.from_numpy(meta["action_low"]).float().to(device) if "action_low" in meta else None
    action_high = torch.from_numpy(meta["action_high"]).float().to(device) if "action_high" in meta else None
    action_std = torch.from_numpy(meta["action_std"]).float().to(device) if "action_std" in meta else None
    high_level_planner = HighLevelCEMPlanner(
        skill_wm=modules["skill_wm"],
        skill_prior=modules["skill_prior"],
        skill_dim=cfg["model"]["skill_dim"],
        horizon=planner_cfg["high_level_horizon"],
        population=planner_cfg["high_level_population"],
        elites=planner_cfg["high_level_elites"],
        iterations=planner_cfg["high_level_iters"],
        skill_penalty=planner_cfg["high_level_skill_penalty"],
        prior_penalty=planner_cfg["high_level_prior_penalty"],
    )
    random_high_level_planner = RandomHighLevelPlanner(
        skill_wm=modules["skill_wm"],
        skill_prior=modules["skill_prior"],
        skill_dim=cfg["model"]["skill_dim"],
        horizon=planner_cfg["high_level_horizon"],
    )
    low_level_planner = LowLevelCEMPlanner(
        low_level_wm=modules["low_level_wm"],
        action_chunk_encoder=modules["action_chunk_encoder"],
        action_dim=meta["action_dim"],
        horizon=planner_cfg["low_level_horizon"],
        population=planner_cfg["low_level_population"],
        elites=planner_cfg["low_level_elites"],
        iterations=planner_cfg["low_level_iters"],
        skill_penalty=planner_cfg["low_level_skill_penalty"],
        action_penalty=planner_cfg["action_penalty"],
        action_low=action_low,
        action_high=action_high,
        init_std=action_std,
        spatial_penalty=planner_cfg.get("subgoal_spatial_penalty", 1.0),
        global_penalty=planner_cfg.get("subgoal_global_penalty", 1.0),
    )
    flat_planner = LowLevelCEMPlanner(
        low_level_wm=modules["low_level_wm"],
        action_chunk_encoder=modules["action_chunk_encoder"],
        action_dim=meta["action_dim"],
        horizon=planner_cfg.get("flat_horizon", planner_cfg["low_level_horizon"] * planner_cfg["high_level_horizon"]),
        population=planner_cfg["low_level_population"],
        elites=planner_cfg["low_level_elites"],
        iterations=planner_cfg["low_level_iters"],
        skill_penalty=0.0,
        action_penalty=planner_cfg["action_penalty"],
        action_low=action_low,
        action_high=action_high,
        init_std=action_std,
        spatial_penalty=planner_cfg.get("flat_spatial_penalty", 1.0),
        global_penalty=planner_cfg.get("flat_global_penalty", 1.0),
    )
    subgoal_resolver = None
    if subgoal_scope != "none":
        allowed_indices = None
        if subgoal_scope == "train":
            allowed_indices = _split_step_indices(cfg["data"]["cache_path"], "train", cfg)
        subgoal_resolver = NearestSubgoalResolver(cfg["data"]["cache_path"], device, allowed_indices=allowed_indices)
    return {
        "flat": flat_planner,
        "hierarchical": HierarchicalPlanner(high_level_planner, low_level_planner, subgoal_resolver=subgoal_resolver),
        "random_hierarchical": HierarchicalPlanner(
            random_high_level_planner,
            low_level_planner,
            subgoal_resolver=subgoal_resolver,
        ),
    }


def _save_rollout_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    ensure_dir(video_path.parent)
    imageio.mimsave(
        str(video_path),
        [np.asarray(frame, dtype=np.uint8) for frame in frames],
        format="GIF",
        duration=1.0 / max(int(fps), 1),
        loop=0,
    )


@torch.no_grad()
def _run_episode(
    mode: str,
    planners: dict[str, object],
    encoder: FrozenVJEPA2Encoder,
    projector: StateProjector,
    env: PushTEnv,
    cache_path: str,
    device: torch.device,
    episode_idx: int,
    start_index: int,
    goal_index: int,
    episode_id: int,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    max_steps: int,
    execute_actions_per_plan: int,
    coverage_threshold: float,
    deterministic_timing: bool = False,
    video_path: Path | None = None,
    video_fps: int = 6,
) -> OnlineEvalRecord:
    latents = _load_cache_latents(cache_path, start_index, goal_index, device)
    env.reset_to_state = start_state.astype(np.float32)
    obs, _ = env.reset()
    goal_z = latents["goal_z"]
    goal_s = latents["goal_s"]
    start_z = latents["start_z"]
    current_z = latents["start_z"]
    current_s = latents["start_s"]
    prev_visual = obs["visual"]

    frames = [np.asarray(obs["visual"]).copy()] if video_path is not None else None
    latencies = []
    skill_consistency = []
    info = {"state": start_state, "max_coverage": 0.0, "final_coverage": 0.0}
    state_metrics = _goal_state_eval(goal_state, info["state"])
    steps_taken = 0
    for _ in range(max_steps):
        t0 = time.perf_counter()
        if mode == "flat":
            plan = planners["flat"].plan(spatial_tokens=current_s, target_z=goal_z, target_s=goal_s, target_skill=None)
        elif mode == "random_hierarchical":
            plan = planners["random_hierarchical"].plan(current_z=current_z, current_s=current_s, goal_z=goal_z)
        else:
            plan = planners["hierarchical"].plan(current_z=current_z, current_s=current_s, goal_z=goal_z)
        latencies.append(0.0 if deterministic_timing else time.perf_counter() - t0)
        skill_consistency.append(float(plan.skill_consistency.detach().cpu()))
        num_exec = min(execute_actions_per_plan, int(plan.actions.shape[0]), max_steps - steps_taken)
        for action_idx in range(num_exec):
            action = plan.actions[action_idx].detach().cpu().numpy()
            prev_visual = obs["visual"]
            obs, _, _, info = env.step(action)
            if frames is not None:
                frames.append(np.asarray(obs["visual"]).copy())
            steps_taken += 1
            state_metrics = _goal_state_eval(goal_state, info["state"])
            if _coverage_success(info, coverage_threshold):
                break
        if _coverage_success(info, coverage_threshold) or steps_taken >= max_steps:
            break
        current_z, current_s = _encode_clip(prev_visual, obs["visual"], encoder, projector)

    final_z, _ = _encode_clip(prev_visual, obs["visual"], encoder, projector)
    final_metrics = _goal_state_eval(goal_state, info["state"])
    coverage_success = _coverage_success(info, coverage_threshold)
    if video_path is not None and frames is not None:
        _save_rollout_video(video_path, frames, video_fps)
    return OnlineEvalRecord(
        episode_idx=episode_idx,
        episode_id=episode_id,
        start_index=start_index,
        goal_index=goal_index,
        sampled_goal_gap=goal_index - start_index,
        success=coverage_success,
        coverage_success=coverage_success,
        goal_state_success=bool(final_metrics["goal_state_success"]),
        state_dist=float(final_metrics["state_dist"]),
        final_latent_distance=float(torch.mean((final_z - goal_z) ** 2).cpu()),
        start_latent_distance=float(torch.mean((start_z - goal_z) ** 2).cpu()),
        planning_latency_sec=float(np.mean(latencies) if latencies else 0.0),
        max_coverage=float(info.get("max_coverage", 0.0)),
        final_coverage=float(info.get("final_coverage", 0.0)),
        steps_taken=int(steps_taken),
        skill_consistency=float(np.mean(skill_consistency) if skill_consistency else 0.0),
        video_path=_portable_path(video_path),
    )


def _make_env(with_velocity: bool) -> PushTEnv:
    return PushTEnv(with_velocity=with_velocity, with_target=True, render_size=224)


def _resolve_methods(mode: str) -> list[str]:
    if mode == "both":
        return ["flat", "hierarchical"]
    if mode == "all":
        return ["flat", "hierarchical", "random_hierarchical"]
    return [mode]


def _write_records_csv(path: Path, records_by_method: dict[str, list[dict]]) -> None:
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
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, records in records_by_method.items():
            for record in records:
                writer.writerow({"method": method, **record})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--mode",
        choices=["all", "both", "flat", "hierarchical", "random_hierarchical"],
        default="both",
    )
    parser.add_argument("--goal-gap", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-limit", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-replacement", action="store_true")
    parser.add_argument("--min-unique-episodes", type=int, default=0)
    parser.add_argument("--deterministic-timing", action="store_true")
    parser.add_argument("--subgoal-scope", choices=["train", "all", "none"], default="train")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    out_dir = ensure_dir(args.output)
    if (Path(out_dir) / "pusht_online_eval.json").exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing eval output without --force: {out_dir}")
    if args.force:
        for name in ["pusht_online_eval.json", "pusht_online_records.csv", "videos"]:
            target = Path(out_dir) / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    planner_cfg = dict(cfg["planner"])
    if args.goal_gap is not None:
        planner_cfg["goal_gap"] = int(args.goal_gap)
    if args.num_eval_episodes is not None:
        planner_cfg["num_eval_episodes"] = int(args.num_eval_episodes)
    if args.max_episode_steps is not None:
        planner_cfg["max_episode_steps"] = int(args.max_episode_steps)
    cfg["planner"] = planner_cfg

    modules = build_all_modules(cfg, cfg["data"]["cache_path"])
    load_checkpoint(args.checkpoint, modules, strict_modules=True)
    modules_to_device(modules, device)
    for module in modules.values():
        module.eval()

    encoder = FrozenVJEPA2Encoder(
        model_id=cfg["encoder"]["model_id"],
        dtype=cfg["encoder"].get("dtype", "bfloat16"),
        device=device,
    )
    projector = StateProjector(
        input_dim=encoder.hidden_size,
        output_dim=cfg["encoder"]["state_dim"],
        pool_grid=cfg["encoder"]["pool_grid"],
    )
    projector.load_state_dict(torch.load(cfg["data"]["projector_ckpt"], map_location="cpu"))
    projector.to(device)
    projector.eval()

    meta = cache_metadata(cfg["data"]["cache_path"])
    planners = _build_planners(cfg, modules, meta, device, subgoal_scope=args.subgoal_scope)
    eval_split = args.eval_split or planner_cfg.get("eval_split", "test")
    sampler = EpisodeGoalSampler(
        cache_path=cfg["data"]["cache_path"],
        split=eval_split,
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        seed=cfg["seed"],
        goal_gap=planner_cfg["goal_gap"],
    )
    goal_pairs = sampler.sample(
        planner_cfg["num_eval_episodes"],
        seed=cfg["seed"],
        max_goal_gap=planner_cfg.get("max_episode_steps"),
        allow_replacement=args.allow_replacement,
    )
    actual_eval_split = getattr(sampler, "actual_split", eval_split)
    unique_episode_count = len({int(pair["episode_id"]) for pair in goal_pairs})
    if unique_episode_count < int(args.min_unique_episodes):
        raise RuntimeError(
            f"Only sampled {unique_episode_count} unique episodes, below --min-unique-episodes={args.min_unique_episodes}"
        )

    with_velocity = bool(goal_pairs and np.asarray(goal_pairs[0]["start_state"]).shape[0] > 5)
    env = _make_env(with_velocity=with_velocity)
    coverage_threshold = float(getattr(env, "success_threshold", 0.95))
    methods = _resolve_methods(args.mode)
    execute_actions_per_plan = int(planner_cfg.get("execute_actions_per_plan", 1))
    records_by_method: dict[str, list[dict]] = {}
    summary: dict[str, object] = {
        "cache_path": _portable_path(cfg["data"]["cache_path"]),
        "checkpoint": _portable_path(args.checkpoint),
        "portable_paths": {
            "cache_path": _portable_path(cfg["data"]["cache_path"]),
            "checkpoint": _portable_path(args.checkpoint),
            "config": _portable_path(args.config),
            "projector": _portable_path(cfg["data"]["projector_ckpt"]),
        },
        "hashes": {
            "cache_sha256": _sha256_file(cfg["data"]["cache_path"]),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "config_sha256": _sha256_file(args.config),
            "projector_sha256": _sha256_file(cfg["data"]["projector_ckpt"]),
        },
        "code_commit": _git_commit(),
        "coverage_threshold": coverage_threshold,
        "eval_split": actual_eval_split,
        "requested_eval_split": eval_split,
        "execute_actions_per_plan": execute_actions_per_plan,
        "goal_gap": int(planner_cfg["goal_gap"]),
        "max_episode_steps": int(planner_cfg["max_episode_steps"]),
        "mode": args.mode,
        "num_eval_episodes": len(goal_pairs),
        "requested_num_eval_episodes": int(planner_cfg["num_eval_episodes"]),
        "unique_episode_count": unique_episode_count,
        "allow_replacement": bool(args.allow_replacement),
        "subgoal_scope": args.subgoal_scope,
    }
    try:
        for method in methods:
            method_records = []
            for episode_idx, pair in enumerate(goal_pairs):
                _set_eval_seed(int(cfg["seed"]) + episode_idx, env=env)
                video_path = None
                if args.save_videos and episode_idx < int(args.video_limit):
                    video_path = Path(out_dir) / "videos" / method / f"episode_{episode_idx:03d}.gif"
                record = _run_episode(
                    mode=method,
                    planners=planners,
                    encoder=encoder,
                    projector=projector,
                    env=env,
                    cache_path=cfg["data"]["cache_path"],
                    device=device,
                    episode_idx=episode_idx,
                    start_index=int(pair["start_index"]),
                    goal_index=int(pair["goal_index"]),
                    episode_id=int(pair["episode_id"]),
                    start_state=np.asarray(pair["start_state"], dtype=np.float32),
                    goal_state=np.asarray(pair["goal_state"], dtype=np.float32),
                    max_steps=int(planner_cfg["max_episode_steps"]),
                    execute_actions_per_plan=execute_actions_per_plan,
                    coverage_threshold=coverage_threshold,
                    deterministic_timing=bool(args.deterministic_timing),
                    video_path=video_path,
                    video_fps=int(args.video_fps),
                )
                method_records.append(asdict(record))
            records_by_method[method] = method_records
            summary[method] = {
                "final_latent_distance": float(np.mean([record["final_latent_distance"] for record in method_records]))
                if method_records
                else 0.0,
                "planning_latency_sec": float(np.mean([record["planning_latency_sec"] for record in method_records]))
                if method_records
                else 0.0,
                "records": method_records,
                "skill_consistency": float(np.mean([record["skill_consistency"] for record in method_records]))
                if method_records
                else 0.0,
                "state_dist": float(np.mean([record["state_dist"] for record in method_records])) if method_records else 0.0,
                "success_rate": float(np.mean([record["coverage_success"] for record in method_records]))
                if method_records
                else 0.0,
                "coverage_success_rate": float(np.mean([record["coverage_success"] for record in method_records]))
                if method_records
                else 0.0,
                "goal_state_success_rate": float(np.mean([record["goal_state_success"] for record in method_records]))
                if method_records
                else 0.0,
                "unique_episode_count": len({record["episode_id"] for record in method_records}),
            }
        if "hierarchical" in summary and "flat" in summary:
            summary["hierarchical_better_rate"] = float(
                np.mean(
                    [
                        float(hier["state_dist"] < flat["state_dist"])
                        for flat, hier in zip(records_by_method["flat"], records_by_method["hierarchical"])
                    ]
                )
            )
        if "hierarchical" in summary and "random_hierarchical" in summary:
            summary["hierarchical_vs_random_better_rate"] = float(
                np.mean(
                    [
                        float(hier["state_dist"] < rnd["state_dist"])
                        for rnd, hier in zip(records_by_method["random_hierarchical"], records_by_method["hierarchical"])
                    ]
                )
            )
    finally:
        env.close()

    dump_json(Path(out_dir) / "pusht_online_eval.json", summary)
    _write_records_csv(Path(out_dir) / "pusht_online_records.csv", records_by_method)


if __name__ == "__main__":
    main()
