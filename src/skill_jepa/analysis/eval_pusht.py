from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch

from skill_jepa.data import EpisodeGoalSampler, cache_metadata, split_episode_ids
from skill_jepa.planning import HighLevelCEMPlanner, HierarchicalPlanner, LowLevelCEMPlanner
from skill_jepa.trainers.common import build_all_modules, load_checkpoint, modules_to_device
from skill_jepa.utils import dump_json, ensure_dir, load_yaml, seed_everything


@dataclass
class CacheStep:
    z: torch.Tensor
    s: torch.Tensor


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
        if self.allowed_indices is not None and len(self.allowed_indices) == 0:
            raise ValueError("NearestSubgoalResolver received an empty allowed index set")

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


def _load_cache_step(cache_path: str, start_index: int, goal_index: int, device: torch.device) -> tuple[CacheStep, CacheStep]:
    with h5py.File(cache_path, "r") as handle:
        start = CacheStep(
            z=torch.from_numpy(handle["z"][start_index]).float().to(device),
            s=torch.from_numpy(handle["s"][start_index]).float().to(device),
        )
        goal = CacheStep(
            z=torch.from_numpy(handle["z"][goal_index]).float().to(device),
            s=torch.from_numpy(handle["s"][goal_index]).float().to(device),
        )
    return start, goal


@torch.no_grad()
def _rollout_hierarchical(pair, planners, modules, cache_path, device, num_chunks):
    start, goal = _load_cache_step(cache_path, int(pair["start_index"]), int(pair["goal_index"]), device)
    current_z = start.z
    current_s = start.s
    goal_z = goal.z
    start_dist = float(torch.mean((current_z - goal_z) ** 2).cpu())
    step_latencies = []
    skill_consistency = []
    for _ in range(num_chunks):
        t0 = time.perf_counter()
        plan = planners["hierarchical"].plan(current_z=current_z, current_s=current_s, goal_z=goal_z)
        step_latencies.append(time.perf_counter() - t0)
        skill_consistency.append(float(plan.skill_consistency.detach().cpu()))
        rollout = modules["low_level_wm"].rollout(current_s.unsqueeze(0), plan.actions.unsqueeze(0))
        current_s = rollout.spatial_tokens[0, -1]
        current_z = rollout.global_states[0, -1]
    final_dist = float(torch.mean((current_z - goal_z) ** 2).cpu())
    return {
        "start_distance": start_dist,
        "final_distance": final_dist,
        "improvement": start_dist - final_dist,
        "planning_latency_sec": float(np.mean(step_latencies) if step_latencies else 0.0),
        "skill_consistency": float(np.mean(skill_consistency) if skill_consistency else 0.0),
    }


@torch.no_grad()
def _rollout_flat(pair, planners, modules, cache_path, device):
    start, goal = _load_cache_step(cache_path, int(pair["start_index"]), int(pair["goal_index"]), device)
    current_z = start.z
    current_s = start.s
    goal_z = goal.z
    start_dist = float(torch.mean((current_z - goal_z) ** 2).cpu())
    t0 = time.perf_counter()
    plan = planners["flat"].plan(spatial_tokens=current_s, target_z=goal_z, target_s=goal.s, target_skill=None)
    latency = time.perf_counter() - t0
    rollout = modules["low_level_wm"].rollout(current_s.unsqueeze(0), plan.actions.unsqueeze(0))
    final_z = rollout.global_states[0, -1]
    final_dist = float(torch.mean((final_z - goal_z) ** 2).cpu())
    return {
        "start_distance": start_dist,
        "final_distance": final_dist,
        "improvement": start_dist - final_dist,
        "planning_latency_sec": float(latency),
        "skill_consistency": float(plan.skill_consistency.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    out_dir = ensure_dir(args.output)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    modules = build_all_modules(cfg, cfg["data"]["cache_path"])
    load_checkpoint(args.checkpoint, modules, strict_modules=True)
    modules_to_device(modules, device)
    for module in modules.values():
        module.eval()

    meta = cache_metadata(cfg["data"]["cache_path"])
    action_low = torch.from_numpy(meta["action_low"]).float().to(device) if "action_low" in meta else None
    action_high = torch.from_numpy(meta["action_high"]).float().to(device) if "action_high" in meta else None
    action_std = torch.from_numpy(meta["action_std"]).float().to(device) if "action_std" in meta else None

    planner_cfg = cfg["planner"]
    train_indices = _split_step_indices(cfg["data"]["cache_path"], "train", cfg)
    subgoal_resolver = NearestSubgoalResolver(cfg["data"]["cache_path"], device, allowed_indices=train_indices)
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
    planners = {
        "hierarchical": HierarchicalPlanner(high_level_planner, low_level_planner, subgoal_resolver=subgoal_resolver),
        "flat": flat_planner,
    }

    sampler = EpisodeGoalSampler(
        cache_path=cfg["data"]["cache_path"],
        split="test",
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        seed=cfg["seed"],
        goal_gap=planner_cfg["goal_gap"],
    )
    goal_pairs = sampler.sample(
        planner_cfg["num_eval_episodes"],
        seed=cfg["seed"],
        max_goal_gap=planner_cfg.get("max_episode_steps"),
    )
    if not goal_pairs:
        raise RuntimeError("No eval goal pairs were sampled; check split size, goal_gap, and max_episode_steps")

    summary = {}
    flat_records = [_rollout_flat(pair, planners, modules, cfg["data"]["cache_path"], device) for pair in goal_pairs]
    hierarchical_records = [
        _rollout_hierarchical(
            pair,
            planners,
            modules,
            cfg["data"]["cache_path"],
            device,
            num_chunks=planner_cfg["high_level_horizon"],
        )
        for pair in goal_pairs
    ]
    comparison = []
    for flat, hier in zip(flat_records, hierarchical_records):
        comparison.append(float(hier["final_distance"] < flat["final_distance"]))
    for method_name, records in [("flat", flat_records), ("hierarchical", hierarchical_records)]:
        summary[method_name] = {
            "final_latent_distance": float(np.mean([record["final_distance"] for record in records])) if records else 0.0,
            "latent_improvement": float(np.mean([record["improvement"] for record in records])) if records else 0.0,
            "planning_latency_sec": float(np.mean([record["planning_latency_sec"] for record in records])) if records else 0.0,
            "skill_consistency": float(np.mean([record["skill_consistency"] for record in records])) if records else 0.0,
            "records": records,
        }
    summary["hierarchical_sampled_state_better_rate"] = float(np.mean(comparison)) if comparison else 0.0

    dump_json(out_dir / "pusht_planning_eval.json", summary)


if __name__ == "__main__":
    main()
