import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from skill_jepa.analysis.eval_pusht_online import (
    NearestSubgoalResolver,
    _assign_task_goals,
    _build_planners,
    _coverage_success,
    _eval_split_summary_fields,
    _goal_state_eval,
    _prepare_output_dir,
    _run_episode,
    _select_task_goal_indices,
    _set_eval_seed,
    _same_record_rate,
    _split_step_indices,
    _summarize_method_records,
    _task_success_claim_supported,
    _validate_eval_provenance,
    _validate_goal_pairs,
)
from skill_jepa.analysis import eval_pusht as offline_eval
from skill_jepa.analysis import eval_pusht_online as online_eval
from skill_jepa.data import EpisodeGoalSampler, FeatureSequenceDataset, split_episode_ids
from skill_jepa.envs import PushTEnv
from skill_jepa.trainers.common import (
    LOW_LEVEL_DATA_PROVENANCE_KEYS,
    PASSIVE_DATA_PROVENANCE_KEYS,
    assert_checkpoint_config_compatible,
    load_checkpoint,
    load_checkpoint_subset,
)
from skill_jepa.trainers.train_joint import _assert_low_level_passive_lineage, main as train_joint_main
from tools import refresh_release_artifacts as release_artifacts
from tools import run_skill_jepa_pusht_locked_suite as locked_suite


def _write_goal_cache(path: Path, ep_len: np.ndarray) -> None:
    offsets = np.zeros_like(ep_len, dtype=np.int64)
    if len(offsets) > 1:
        offsets[1:] = np.cumsum(ep_len[:-1], dtype=np.int64)
    total = int(ep_len.sum())
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=ep_len.astype(np.int32))
        handle.create_dataset("ep_offset", data=offsets)
        handle.create_dataset("episode_idx", data=np.repeat(np.arange(len(ep_len)), ep_len))
        handle.create_dataset("step_idx", data=np.concatenate([np.arange(length) for length in ep_len]))
        handle.create_dataset("z", data=np.zeros((total, 4), dtype=np.float32))
        handle.create_dataset("s", data=np.zeros((total, 2, 4), dtype=np.float32))
        handle.create_dataset("action", data=np.zeros((total, 2), dtype=np.float32))
        state = np.zeros((total, 5), dtype=np.float32)
        state[:, 0] = np.arange(total, dtype=np.float32)
        handle.create_dataset("state", data=state)


def test_goal_sampler_replacement_is_explicit(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([6, 6, 6], dtype=np.int32))
    sampler = EpisodeGoalSampler(cache_path, split="train", val_fraction=0.0, test_fraction=0.0, seed=0, goal_gap=1)

    with pytest.raises(ValueError, match="without replacement"):
        sampler.sample(10, seed=0, allow_replacement=False)
    without_replacement = sampler.sample(10, seed=0, allow_replacement=False, allow_under_sampling=True)
    with_replacement = sampler.sample(10, seed=0, allow_replacement=True)

    assert len(without_replacement) == 3
    assert len({int(pair["episode_id"]) for pair in without_replacement}) == 3
    assert len(with_replacement) == 10
    assert len({int(pair["episode_id"]) for pair in with_replacement}) <= 3


def test_goal_sampler_rejects_empty_requested_split_by_default(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([6, 6], dtype=np.int32))

    with pytest.raises(ValueError, match="Requested split 'test' has no episodes"):
        EpisodeGoalSampler(cache_path, split="test", val_fraction=0.5, test_fraction=0.5, seed=0, goal_gap=1)


def test_goal_sampler_records_actual_split_after_explicit_fallback(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([6, 6], dtype=np.int32))
    sampler = EpisodeGoalSampler(
        cache_path,
        split="test",
        val_fraction=0.5,
        test_fraction=0.5,
        seed=0,
        goal_gap=1,
        fallback_empty_split=True,
    )

    assert sampler.requested_split == "test"
    assert sampler.actual_split == "val"
    assert len(sampler.episode_ids) == 1


def test_goal_sampler_records_train_fallback_when_eval_splits_empty(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([6, 6], dtype=np.int32))
    sampler = EpisodeGoalSampler(
        cache_path,
        split="test",
        val_fraction=0.0,
        test_fraction=0.0,
        seed=0,
        goal_gap=1,
        fallback_empty_split=True,
    )

    assert sampler.requested_split == "test"
    assert sampler.actual_split == "train"
    assert len(sampler.episode_ids) == 2


def test_split_episode_ids_keeps_train_val_test_disjoint_when_test_rounds_high():
    splits = split_episode_ids(num_episodes=3, val_fraction=0.0, test_fraction=0.84, seed=0)

    train_ids = set(splits["train"].tolist())
    val_ids = set(splits["val"].tolist())
    test_ids = set(splits["test"].tolist())
    assert train_ids
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == {0, 1, 2}


def test_feature_dataset_can_hold_split_and_labeled_subset_fixed_across_training_seeds(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([8, 8, 8, 8, 8, 8], dtype=np.int32))

    left = FeatureSequenceDataset(
        cache_path,
        sequence_length=2,
        split="train",
        labeled_fraction=0.5,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=1,
        split_seed=0,
        labeled_seed=0,
    )
    right = FeatureSequenceDataset(
        cache_path,
        sequence_length=2,
        split="train",
        labeled_fraction=0.5,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=2,
        split_seed=0,
        labeled_seed=0,
    )
    try:
        assert left.split_ids.tolist() == right.split_ids.tolist()
        assert left.labeled_ids == right.labeled_ids
    finally:
        left.close()
        right.close()


def test_goal_sampler_uses_valid_episodes_instead_of_dropping_short_draws(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([10, 10, 100, 100], dtype=np.int32))
    sampler = EpisodeGoalSampler(cache_path, split="train", val_fraction=0.0, test_fraction=0.0, seed=0, goal_gap=50)

    pairs = sampler.sample(2, seed=25, allow_replacement=False)

    assert len(pairs) == 2
    assert {int(pair["episode_id"]) for pair in pairs} <= {2, 3}


def test_goal_sampler_rejects_impossible_max_goal_gap(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([100], dtype=np.int32))
    sampler = EpisodeGoalSampler(cache_path, split="train", val_fraction=0.0, test_fraction=0.0, seed=0, goal_gap=50)

    with pytest.raises(ValueError, match="smaller than goal_gap"):
        sampler.sample(1, seed=0, max_goal_gap=10)


def test_task_goal_pool_uses_train_split_success_states(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([4, 4, 4], dtype=np.int32))
    with h5py.File(cache_path, "r+") as handle:
        states = handle["state"][:]
        states[:] = np.array([100.0, 100.0, 100.0, 100.0, 0.0], dtype=np.float32)
        train_id = int(split_episode_ids(3, val_fraction=0.0, test_fraction=0.34, seed=0)["train"][0])
        train_goal_index = int(handle["ep_offset"][train_id] + handle["ep_len"][train_id] - 1)
        states[train_goal_index] = np.array([100.0, 100.0, 256.0, 256.0, np.pi / 4.0], dtype=np.float32)
        handle["state"][:] = states

    env = PushTEnv(render_size=224, legacy=False, with_velocity=False)
    try:
        goal_indices, coverages = _select_task_goal_indices(
            cache_path,
            env=env,
            val_fraction=0.0,
            test_fraction=0.34,
            seed=0,
            coverage_threshold=0.95,
            tail_steps=2,
        )
    finally:
        env.close()

    assert goal_indices.tolist() == [train_goal_index]
    assert coverages[0] >= 0.95


def test_train_scoped_subgoal_indices_use_data_split_seed(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([4, 4, 4, 4, 4, 4], dtype=np.int32))
    cfg = {
        "seed": 999,
        "data": {
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "split_seed": 0,
        },
    }

    indices = _split_step_indices(str(cache_path), "train", cfg)
    with h5py.File(cache_path, "r") as handle:
        actual_episode_ids = set(handle["episode_idx"][indices].astype(int).tolist())
    expected_episode_ids = set(split_episode_ids(6, val_fraction=0.2, test_fraction=0.2, seed=0)["train"].tolist())
    wrong_episode_ids = set(split_episode_ids(6, val_fraction=0.2, test_fraction=0.2, seed=999)["train"].tolist())

    assert actual_episode_ids == expected_episode_ids
    assert actual_episode_ids != wrong_episode_ids


def test_offline_train_scoped_subgoal_indices_use_data_split_seed(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([4, 4, 4, 4, 4, 4], dtype=np.int32))
    cfg = {
        "seed": 999,
        "data": {
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "split_seed": 0,
        },
    }

    indices = offline_eval._split_step_indices(str(cache_path), "train", cfg)
    with h5py.File(cache_path, "r") as handle:
        actual_episode_ids = set(handle["episode_idx"][indices].astype(int).tolist())
    expected_episode_ids = set(split_episode_ids(6, val_fraction=0.2, test_fraction=0.2, seed=0)["train"].tolist())
    wrong_episode_ids = set(split_episode_ids(6, val_fraction=0.2, test_fraction=0.2, seed=999)["train"].tolist())

    assert actual_episode_ids == expected_episode_ids
    assert actual_episode_ids != wrong_episode_ids


def test_task_goal_pool_rejects_cache_without_train_success_states(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([4, 4, 4], dtype=np.int32))
    env = PushTEnv(render_size=224, legacy=False, with_velocity=False)
    try:
        with pytest.raises(RuntimeError, match="No train-split task goal states"):
            _select_task_goal_indices(
                cache_path,
                env=env,
                val_fraction=0.0,
                test_fraction=0.34,
                seed=0,
                coverage_threshold=0.95,
                tail_steps=2,
            )
    finally:
        env.close()


def test_task_goal_pool_falls_back_to_full_train_scan_when_tail_misses(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([5, 5, 5], dtype=np.int32))
    train_id = int(split_episode_ids(3, val_fraction=0.0, test_fraction=0.34, seed=0)["train"][0])
    with h5py.File(cache_path, "r+") as handle:
        states = handle["state"][:]
        states[:] = np.array([100.0, 100.0, 100.0, 100.0, 0.0], dtype=np.float32)
        early_train_goal_index = int(handle["ep_offset"][train_id])
        states[early_train_goal_index] = np.array([100.0, 100.0, 256.0, 256.0, np.pi / 4.0], dtype=np.float32)
        handle["state"][:] = states

    env = PushTEnv(render_size=224, legacy=False, with_velocity=False)
    try:
        goal_indices, coverages = _select_task_goal_indices(
            cache_path,
            env=env,
            val_fraction=0.0,
            test_fraction=0.34,
            seed=0,
            coverage_threshold=0.95,
            tail_steps=1,
        )
    finally:
        env.close()

    assert goal_indices.tolist() == [early_train_goal_index]
    assert coverages[0] >= 0.95


def test_task_goal_pool_keeps_early_train_states_out_when_tail_has_success(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([5, 5, 5], dtype=np.int32))
    train_id = int(split_episode_ids(3, val_fraction=0.0, test_fraction=0.34, seed=0)["train"][0])
    with h5py.File(cache_path, "r+") as handle:
        states = handle["state"][:]
        states[:] = np.array([100.0, 100.0, 100.0, 100.0, 0.0], dtype=np.float32)
        early_train_goal_index = int(handle["ep_offset"][train_id])
        tail_train_goal_index = int(handle["ep_offset"][train_id] + handle["ep_len"][train_id] - 1)
        success_state = np.array([100.0, 100.0, 256.0, 256.0, np.pi / 4.0], dtype=np.float32)
        states[early_train_goal_index] = success_state
        states[tail_train_goal_index] = success_state
        handle["state"][:] = states

    env = PushTEnv(render_size=224, legacy=False, with_velocity=False)
    try:
        goal_indices, coverages = _select_task_goal_indices(
            cache_path,
            env=env,
            val_fraction=0.0,
            test_fraction=0.34,
            seed=0,
            coverage_threshold=0.95,
            tail_steps=1,
        )
    finally:
        env.close()

    assert goal_indices.tolist() == [tail_train_goal_index]
    assert early_train_goal_index not in goal_indices.tolist()
    assert coverages[0] >= 0.95


def test_assign_task_goals_records_train_goal_identity(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([4, 4, 4], dtype=np.int32))
    goal_index = 1
    with h5py.File(cache_path, "r") as handle:
        expected_goal_state = handle["state"][goal_index].copy()
        expected_goal_episode = int(handle["episode_idx"][goal_index])
    pair = {
        "episode_id": np.int64(2),
        "start_index": np.int64(8),
        "goal_index": np.int64(9),
        "start_state": np.zeros(5, dtype=np.float32),
        "goal_state": np.ones(5, dtype=np.float32),
    }

    [aligned] = _assign_task_goals([pair], cache_path, np.array([goal_index], dtype=np.int64), seed=0)

    assert int(aligned["goal_index"]) == goal_index
    assert int(aligned["goal_episode_id"]) == expected_goal_episode
    assert aligned["goal_mode"] == "task"
    np.testing.assert_allclose(aligned["goal_state"], expected_goal_state)


def test_validate_goal_pairs_rejects_empty_eval():
    with pytest.raises(RuntimeError, match="No eval goal pairs"):
        _validate_goal_pairs([], min_unique_episodes=0)


def test_validate_goal_pairs_enforces_minimum_unique_episode_count():
    pairs = [{"episode_id": np.int64(3)}, {"episode_id": np.int64(3)}]

    with pytest.raises(RuntimeError, match="Only sampled 1 unique episodes"):
        _validate_goal_pairs(pairs, min_unique_episodes=2)


def test_same_record_rate_uses_metric_specific_direction():
    records = {
        "flat": [
            {"coverage_success": False, "state_dist": 10.0},
            {"coverage_success": True, "state_dist": 4.0},
        ],
        "hierarchical": [
            {"coverage_success": True, "state_dist": 9.0},
            {"coverage_success": False, "state_dist": 5.0},
        ],
    }

    assert _same_record_rate(records, "flat", "hierarchical", "coverage_success", True) == 0.5
    assert _same_record_rate(records, "flat", "hierarchical", "state_dist", False) == 0.5


def test_method_summary_keeps_success_rate_coverage_backed_when_goal_state_disagrees():
    records = [
        {
            "episode_id": 1,
            "coverage_success": True,
            "goal_state_success": False,
            "final_latent_distance": 1.0,
            "planning_latency_sec": 0.1,
            "skill_consistency": 0.2,
            "state_dist": 99.0,
        },
        {
            "episode_id": 2,
            "coverage_success": False,
            "goal_state_success": True,
            "final_latent_distance": 3.0,
            "planning_latency_sec": 0.3,
            "skill_consistency": 0.4,
            "state_dist": 1.0,
        },
    ]

    summary = _summarize_method_records(records, "sampled_train_task_goal_full_state_diagnostic")

    assert summary["coverage_success_rate"] == 0.5
    assert summary["goal_state_success_diagnostic_rate"] == 0.5
    assert "success_rate" not in summary
    assert "goal_state_success_rate" not in summary
    assert not summary["goal_state_success_is_task_metric"]


def test_task_success_claim_gate_rejects_smoke_or_confounded_evals():
    assert _task_success_claim_supported(
        goal_mode="task",
        actual_split="val",
        requested_count=3,
        sampled_count=3,
        unique_episode_count=3,
        allow_replacement=False,
        allow_under_sampling=False,
        allow_split_fallback=False,
        subgoal_scope="train",
    )
    assert not _task_success_claim_supported("trajectory", "val", 3, 3, 3, False, False, False, "train")
    assert not _task_success_claim_supported("task", "train", 3, 3, 3, False, False, False, "train")
    assert not _task_success_claim_supported("task", "val", 3, 1, 1, False, True, False, "train")
    assert not _task_success_claim_supported("task", "val", 3, 3, 1, True, False, False, "train")
    assert not _task_success_claim_supported("task", "val", 3, 3, 3, False, False, True, "train")
    assert not _task_success_claim_supported("task", "val", 3, 3, 3, False, False, False, "all")


def test_eval_split_summary_records_requested_and_actual_split(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    _write_goal_cache(cache_path, np.array([6, 6], dtype=np.int32))
    sampler = EpisodeGoalSampler(
        cache_path,
        split="test",
        val_fraction=0.5,
        test_fraction=0.5,
        seed=0,
        goal_gap=1,
        fallback_empty_split=True,
    )

    fields = _eval_split_summary_fields(sampler, "test")

    assert fields == {"requested_eval_split": "test", "eval_split": "val"}


def test_package_discovery_keeps_runtime_import_roots():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
        import tomli as tomllib

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    include = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])
    where = pyproject["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["src", "."]
    assert include == {"skill_jepa*", "tools*"}
    manifest = (Path(__file__).resolve().parents[2] / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include configs *.yaml" in manifest
    assert "recursive-include docs *.md *.mmd *.png *.svg" in manifest
    assert "recursive-include artifacts *.csv *.gif *.json *.md *.pdf *.png *.svg *.yaml" in manifest


def test_pusht_env_is_packaged_under_skill_jepa_namespace():
    assert PushTEnv.__name__ == "PushTEnv"


def test_success_metrics_keep_coverage_primary():
    assert _coverage_success({"max_coverage": 0.95}, threshold=0.95)
    assert not _coverage_success({"max_coverage": 0.949}, threshold=0.95)

    goal = np.array([100.0, 100.0, 200.0, 200.0, 0.0], dtype=np.float32)
    current = np.array([105.0, 101.0, 202.0, 199.0, 0.01], dtype=np.float32)
    metrics = _goal_state_eval(goal, current)
    assert metrics["goal_state_success"]
    assert metrics["state_dist"] > 0.0


def test_goal_state_diagnostic_uses_sampled_agent_and_block_state():
    goal = np.array([0.0, 0.0, 200.0, 200.0, 0.0], dtype=np.float32)
    current = np.array([500.0, 500.0, 202.0, 199.0, 0.01], dtype=np.float32)

    metrics = _goal_state_eval(goal, current)

    assert not metrics["goal_state_success"]
    assert metrics["state_dist"] > 700.0


def test_goal_state_angle_distance_wraps_periodically():
    goal = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.01], dtype=np.float32)
    current = np.array([0.0, 0.0, 0.0, 0.0, -np.pi + 0.01], dtype=np.float32)

    metrics = _goal_state_eval(goal, current)

    assert metrics["goal_state_success"]
    assert metrics["state_dist"] < 0.03


def test_run_episode_records_coverage_success_as_primary_metric(monkeypatch):
    class FakePlan:
        def __init__(self):
            self.actions = torch.zeros(1, 2)
            self.skill_consistency = torch.tensor(0.25)

    class FakePlanner:
        def plan(self, **kwargs):
            return FakePlan()

    class FakeEnv:
        def __init__(self):
            self.reset_to_state = None
            self.step_count = 0

        def reset(self):
            return {"visual": np.zeros((4, 4, 3), dtype=np.uint8)}, {}

        def step(self, action):
            self.step_count += 1
            obs = {"visual": np.ones((4, 4, 3), dtype=np.uint8)}
            info = {
                "state": np.array([500.0, 500.0, 10.0, 10.0, 0.0], dtype=np.float32),
                "max_coverage": 0.96,
                "final_coverage": 0.96,
            }
            return obs, 0.0, False, info

    monkeypatch.setattr(
        online_eval,
        "_load_cache_latents",
        lambda cache_path, start_index, goal_index, device: {
            "start_z": torch.zeros(2),
            "start_s": torch.zeros(1, 2),
            "goal_z": torch.ones(2),
            "goal_s": torch.ones(1, 2),
        },
    )
    monkeypatch.setattr(online_eval, "_encode_clip", lambda prev_frame, frame, encoder, projector: (torch.zeros(2), torch.zeros(1, 2)))

    record = _run_episode(
        mode="flat",
        planners={"flat": FakePlanner()},
        encoder=object(),
        projector=object(),
        env=FakeEnv(),
        cache_path="unused.h5",
        device=torch.device("cpu"),
        episode_idx=0,
        eval_seed=123,
        start_index=0,
        goal_index=1,
        episode_id=0,
        goal_episode_id=0,
        goal_mode="task",
        start_state=np.zeros(5, dtype=np.float32),
        goal_state=np.zeros(5, dtype=np.float32),
        max_steps=4,
        execute_actions_per_plan=1,
        coverage_threshold=0.95,
        deterministic_timing=True,
    )

    assert record.coverage_success
    assert not record.goal_state_success
    assert record.max_coverage == 0.96
    assert record.final_coverage == 0.96
    assert record.planning_latency_sec == 0.0


def test_set_eval_seed_calls_environment_seed():
    class FakeEnv:
        def __init__(self):
            self.seeds = []

        def seed(self, seed):
            self.seeds.append(seed)

    env = FakeEnv()
    _set_eval_seed(123, env=env)
    assert env.seeds == [123]


def test_strict_checkpoint_loading_rejects_missing_modules(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"modules": {"present": nn.Linear(1, 1).state_dict()}}, checkpoint)
    modules = {"present": nn.Linear(1, 1), "missing": nn.Linear(1, 1)}

    with pytest.raises(RuntimeError, match="missing required modules"):
        load_checkpoint(checkpoint, modules, strict_modules=True)


def test_strict_checkpoint_loading_rejects_unexpected_modules(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"modules": {"present": nn.Linear(1, 1).state_dict(), "stale_extra": nn.Linear(1, 1).state_dict()}},
        checkpoint,
    )
    modules = {"present": nn.Linear(1, 1)}

    with pytest.raises(RuntimeError, match="unexpected modules"):
        load_checkpoint(checkpoint, modules, strict_modules=True)


def test_strict_checkpoint_loading_rejects_incompatible_module_shapes(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"modules": {"present": nn.Linear(2, 1).state_dict()}}, checkpoint)

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(checkpoint, {"present": nn.Linear(1, 1)}, strict_modules=True)


def test_checkpoint_subset_loads_required_modules_and_allows_extra_modules(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    source = nn.Linear(1, 1)
    target = nn.Linear(1, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)
        target.weight.zero_()
        target.bias.zero_()
    torch.save(
        {
            "modules": {
                "needed": source.state_dict(),
                "extra": nn.Linear(1, 1).state_dict(),
            }
        },
        checkpoint,
    )

    load_checkpoint_subset(checkpoint, {"needed": target}, required_modules=["needed"])

    assert torch.allclose(target.weight, torch.full_like(target.weight, 3.0))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 4.0))


def test_checkpoint_subset_rejects_missing_required_modules(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"modules": {"other": nn.Linear(1, 1).state_dict()}}, checkpoint)

    with pytest.raises(RuntimeError, match="missing required modules"):
        load_checkpoint_subset(checkpoint, {"needed": nn.Linear(1, 1)}, required_modules=["needed"])


def test_checkpoint_config_compatibility_rejects_cache_hash_mismatch(tmp_path: Path):
    cache = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    cache.write_bytes(b"current-cache")
    projector.write_bytes(b"projector")
    cfg = {
        "data": {"cache_path": str(cache), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder"},
        "model": {"hidden_dim": 8},
    }
    payload = {
        "config": {
            "data": {"cache_path": str(cache), "projector_ckpt": str(projector)},
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(b"old-cache").hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
        },
    }

    with pytest.raises(RuntimeError, match="cache_path hash"):
        assert_checkpoint_config_compatible(payload, cfg)


def test_checkpoint_config_compatibility_rejects_data_split_provenance_mismatch(tmp_path: Path):
    cache = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    cache.write_bytes(b"cache")
    projector.write_bytes(b"projector")
    cfg = {
        "data": {
            "cache_path": str(cache),
            "projector_ckpt": str(projector),
            "labeled_fraction": 0.1,
            "val_fraction": 0.2,
            "test_fraction": 0.0,
            "split_seed": 0,
            "labeled_seed": 0,
        },
        "encoder": {"model_id": "encoder"},
        "model": {"hidden_dim": 8},
    }
    payload = {
        "config": {
            "data": {**cfg["data"], "split_seed": 99},
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(b"cache").hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
        },
    }

    with pytest.raises(RuntimeError, match="data.split_seed"):
        assert_checkpoint_config_compatible(
            payload,
            cfg,
            data_value_keys=("labeled_fraction", "val_fraction", "test_fraction", "split_seed", "labeled_seed"),
        )


def test_checkpoint_config_compatibility_resolves_implicit_split_seed_from_top_level_seed(tmp_path: Path):
    cache = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    cache.write_bytes(b"cache")
    projector.write_bytes(b"projector")
    payload = {
        "config": {
            "seed": 0,
            "data": {
                "cache_path": str(cache),
                "projector_ckpt": str(projector),
                "labeled_fraction": 0.1,
                "val_fraction": 0.2,
                "test_fraction": 0.0,
            },
            "encoder": {"model_id": "encoder"},
            "model": {"hidden_dim": 8},
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(b"cache").hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
        },
    }
    runtime_cfg = {
        "seed": 1,
        "data": {
            "cache_path": str(cache),
            "projector_ckpt": str(projector),
            "labeled_fraction": 0.1,
            "val_fraction": 0.2,
            "test_fraction": 0.0,
        },
        "encoder": {"model_id": "encoder"},
        "model": {"hidden_dim": 8},
    }

    with pytest.raises(RuntimeError, match="data.split_seed"):
        assert_checkpoint_config_compatible(
            payload,
            runtime_cfg,
            data_value_keys=LOW_LEVEL_DATA_PROVENANCE_KEYS,
        )


def test_passive_checkpoint_compatibility_allows_label_subset_drift_but_low_level_rejects_it(tmp_path: Path):
    cache = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    cache.write_bytes(b"cache")
    projector.write_bytes(b"projector")
    payload = {
        "config": {
            "seed": 0,
            "data": {
                "cache_path": str(cache),
                "projector_ckpt": str(projector),
                "labeled_fraction": 0.1,
                "val_fraction": 0.2,
                "test_fraction": 0.0,
                "split_seed": 0,
                "labeled_seed": 0,
            },
            "encoder": {"model_id": "encoder"},
            "model": {"hidden_dim": 8},
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(b"cache").hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
        },
    }
    runtime_cfg = {
        **payload["config"],
        "data": {**payload["config"]["data"], "labeled_fraction": 1.0, "labeled_seed": 99},
    }

    assert_checkpoint_config_compatible(payload, runtime_cfg, data_value_keys=PASSIVE_DATA_PROVENANCE_KEYS)
    with pytest.raises(RuntimeError, match="data.labeled_fraction"):
        assert_checkpoint_config_compatible(payload, runtime_cfg, data_value_keys=LOW_LEVEL_DATA_PROVENANCE_KEYS)


def test_joint_rejects_low_level_checkpoint_from_different_passive_lineage(tmp_path: Path):
    recorded_passive = tmp_path / "recorded_passive.pt"
    current_passive = tmp_path / "current_passive.pt"
    recorded_passive.write_bytes(b"recorded")
    current_passive.write_bytes(b"current")
    cfg = {"training": {"passive_checkpoint": str(current_passive)}}
    low_level_payload = {"config": {"training": {"passive_checkpoint": str(recorded_passive)}}}

    with pytest.raises(RuntimeError, match="passive lineage"):
        _assert_low_level_passive_lineage(cfg, low_level_payload)


def test_joint_rejects_low_level_checkpoint_when_passive_file_was_overwritten(tmp_path: Path):
    passive = tmp_path / "passive.pt"
    passive.write_bytes(b"current-passive")
    cfg = {"training": {"passive_checkpoint": str(passive)}}
    low_level_payload = {
        "config": {"training": {"passive_checkpoint": str(passive)}},
        "artifact_hashes": {"training.passive_checkpoint": hashlib.sha256(b"old-passive").hexdigest()},
    }

    with pytest.raises(RuntimeError, match="passive_checkpoint hash"):
        _assert_low_level_passive_lineage(cfg, low_level_payload)


def test_locked_suite_summary_freshness_rejects_missing_expected_artifacts(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "pusht_online_eval.json"
    summary_path.write_text(
        json.dumps(
            {
                "code_commit": "abc123",
                "goal_mode": "task",
                "hashes": {"checkpoint_sha256": "not-used"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: False)

    assert not locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": tmp_path / "missing.pt"},
    )


def test_locked_suite_summary_freshness_checks_hashes_and_eval_knobs(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    stale_checkpoint = tmp_path / "stale_checkpoint.pt"
    checkpoint.write_bytes(b"current")
    stale_checkpoint.write_bytes(b"stale")
    summary_path = tmp_path / "pusht_online_eval.json"
    summary_path.write_text(
        json.dumps(
            {
                "code_commit": "abc123",
                "goal_mode": "task",
                "subgoal_scope": "train",
                "requested_num_eval_episodes": 100,
                "deterministic_timing": False,
                "provenance": {"warnings": []},
                "hashes": {"checkpoint_sha256": hashlib.sha256(b"current").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: False)

    assert locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": checkpoint},
        expected_fields={"subgoal_scope": "train", "requested_num_eval_episodes": 100, "deterministic_timing": False},
    )
    assert not locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": stale_checkpoint},
        expected_fields={"subgoal_scope": "train", "requested_num_eval_episodes": 100, "deterministic_timing": False},
    )
    assert not locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": checkpoint},
        expected_fields={"subgoal_scope": "all", "requested_num_eval_episodes": 100, "deterministic_timing": False},
    )


def test_locked_suite_summary_freshness_rejects_dirty_state_mismatch(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current")
    summary_path = tmp_path / "pusht_online_eval.json"
    summary_path.write_text(
        json.dumps(
            {
                "code_commit": "abc123",
                "code_dirty": True,
                "code_status_sha256": "old-status",
                "goal_mode": "task",
                "hashes": {"checkpoint_sha256": hashlib.sha256(b"current").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: True)
    monkeypatch.setattr(locked_suite, "_git_status_sha256", lambda: "new-status")

    assert not locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": checkpoint},
    )


def test_locked_suite_summary_freshness_rejects_provenance_disabled(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current")
    summary_path = tmp_path / "pusht_online_eval.json"
    summary_path.write_text(
        json.dumps(
            {
                "code_commit": "abc123",
                "code_dirty": False,
                "goal_mode": "task",
                "provenance": {"warnings": ["Provenance validation was disabled by --allow-provenance-mismatch"]},
                "hashes": {"checkpoint_sha256": hashlib.sha256(b"current").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: False)

    assert not locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": checkpoint},
    )


def test_locked_suite_summary_freshness_checks_deterministic_timing_field(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"current")
    summary_path = tmp_path / "pusht_online_eval.json"
    summary_path.write_text(
        json.dumps(
            {
                "code_commit": "abc123",
                "code_dirty": False,
                "goal_mode": "task",
                "deterministic_timing": True,
                "provenance": {"warnings": []},
                "hashes": {"checkpoint_sha256": hashlib.sha256(b"current").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: False)

    assert not locked_suite._summary_matches_current(
        summary_path,
        expected_goal_mode="task",
        expected_artifacts={"checkpoint": checkpoint},
        expected_fields={"deterministic_timing": False},
    )


def test_locked_suite_checkpoint_reuse_rejects_commit_config_and_lineage_drift(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    passive = tmp_path / "passive.pt"
    low_level = tmp_path / "low_level.pt"
    checkpoint = tmp_path / "checkpoint.pt"
    for path, payload in [
        (cache, b"cache"),
        (projector, b"projector"),
        (passive, b"passive"),
        (low_level, b"low-level"),
    ]:
        path.write_bytes(payload)
    cfg = {
        "seed": 0,
        "data": {"cache_path": str(cache), "projector_ckpt": str(projector), "split_seed": 0, "labeled_seed": 17},
        "training": {"passive_checkpoint": str(passive), "low_level_checkpoint": str(low_level)},
    }
    torch.save(
        {
            "code_commit": "abc123",
            "code_dirty": False,
            "config": cfg,
            "artifact_hashes": {
                "data.cache_path": hashlib.sha256(b"cache").hexdigest(),
                "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
                "training.passive_checkpoint": hashlib.sha256(b"passive").hexdigest(),
                "training.low_level_checkpoint": hashlib.sha256(b"low-level").hexdigest(),
            },
        },
        checkpoint,
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: False)

    assert locked_suite._checkpoint_matches_config(
        checkpoint,
        cfg,
        ("data.cache_path", "data.projector_ckpt", "training.passive_checkpoint"),
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "newer")
    assert not locked_suite._checkpoint_matches_config(checkpoint, cfg, ("data.cache_path",))
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "abc123")
    drifted_cfg = {**cfg, "seed": 1}
    assert not locked_suite._checkpoint_matches_config(checkpoint, drifted_cfg, ("data.cache_path",))
    passive.write_bytes(b"overwritten-passive")
    assert not locked_suite._checkpoint_matches_config(checkpoint, cfg, ("training.passive_checkpoint",))


def test_locked_suite_aggregate_only_rejects_stale_existing_eval(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "suite"
    config_root = output_root / "configs"
    eval_root = output_root / "evals"
    cache_root = output_root / "cache"
    cfg_path = config_root / "seed_0_joint10.yaml"
    checkpoint = output_root / "seed_0" / "joint_10pct" / "joint" / "joint_best.pt"
    cache = cache_root / "pusht_vjepa2_cache_13024ep.h5"
    projector = tmp_path / "projector.pt"
    summary_dir = eval_root / "seed_0" / "joint_hier_10pct" / "goal_gap_24"
    for path in [cfg_path, checkpoint, cache, projector]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    summary_dir.mkdir(parents=True)
    (summary_dir / "pusht_online_eval.json").write_text(
        json.dumps(
            {
                "code_commit": "old",
                "code_dirty": False,
                "goal_mode": "task",
                "mode": "hierarchical",
                "requested_eval_split": "val",
                "subgoal_scope": "train",
                "goal_gap": 24,
                "requested_num_eval_episodes": 100,
                "allow_replacement": False,
                "allow_under_sampling": False,
                "allow_split_fallback": False,
                "deterministic_timing": False,
                "provenance": {"warnings": []},
                "hashes": {
                    "config_sha256": hashlib.sha256(b"artifact").hexdigest(),
                    "checkpoint_sha256": hashlib.sha256(b"artifact").hexdigest(),
                    "cache_sha256": hashlib.sha256(b"artifact").hexdigest(),
                    "projector_sha256": hashlib.sha256(b"artifact").hexdigest(),
                },
                "hierarchical": {"coverage_success_rate": 0.0, "records": []},
                "num_eval_episodes": 100,
            }
        ),
        encoding="utf-8",
    )
    (summary_dir / "pusht_online_records.csv").write_text("method,episode_idx\n", encoding="utf-8")
    monkeypatch.setattr(locked_suite, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(locked_suite, "CONFIG_ROOT", config_root)
    monkeypatch.setattr(locked_suite, "EVAL_ROOT", eval_root)
    monkeypatch.setattr(locked_suite, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(locked_suite, "DEBUG_PROJECTOR", projector)
    monkeypatch.setattr(locked_suite, "GOAL_EVALS", [(24, 100, False)])
    monkeypatch.setattr(
        locked_suite,
        "_eval_specs",
        lambda artifacts: [("joint_hier_10pct", cfg_path, checkpoint, "hierarchical")],
    )
    monkeypatch.setattr(locked_suite, "_git_commit", lambda: "current")
    monkeypatch.setattr(locked_suite, "_git_dirty", lambda: False)

    with pytest.raises(RuntimeError, match="stale or unverifiable"):
        locked_suite._collect_existing_results([0])


def test_release_sanitizer_converts_legacy_success_to_coverage_metric(tmp_path: Path, monkeypatch):
    eval_dir = tmp_path / "locked_eval"
    eval_dir.mkdir()
    report_path = tmp_path / "current_best_checkpoint_comparison.csv"
    summary_path = eval_dir / "pusht_online_eval.json"
    records_path = eval_dir / "pusht_online_records.csv"
    payload = {
        "cache_path": "private/old/cache.h5",
        "checkpoint": "private/old/joint.pt",
        "flat": {
            "goal_state_success_rate": 1.0,
            "records": [
                {
                    "success": True,
                    "max_coverage": 0.1,
                    "video_path": "private/old/videos/episode_000.gif",
                }
            ],
        },
        "hierarchical": {
            "goal_state_success_rate": 0.0,
            "records": [
                {
                    "success": False,
                    "max_coverage": 0.99,
                    "video_path": "private/old/videos/episode_000.gif",
                }
            ],
        },
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    records = [
        {
            "method": "flat",
            "episode_idx": "0",
            "episode_id": "2",
            "start_index": "1",
            "goal_index": "3",
            "sampled_goal_gap": "2",
            "success": "True",
            "max_coverage": "0.1",
            "state_dist": "5.0",
            "final_latent_distance": "0.0",
            "start_latent_distance": "0.0",
            "planning_latency_sec": "0.1",
            "final_coverage": "0.1",
            "steps_taken": "2",
            "skill_consistency": "0.0",
            "video_path": "private/old/videos/episode_000.gif",
        },
        {
            "method": "hierarchical",
            "episode_idx": "0",
            "episode_id": "2",
            "start_index": "1",
            "goal_index": "3",
            "sampled_goal_gap": "2",
            "success": "False",
            "max_coverage": "0.99",
            "state_dist": "1.0",
            "final_latent_distance": "0.0",
            "start_latent_distance": "0.0",
            "planning_latency_sec": "0.2",
            "final_coverage": "0.99",
            "steps_taken": "2",
            "skill_consistency": "0.0",
            "video_path": "private/old/videos/episode_000.gif",
        },
    ]
    summary = {
        "flat": {
            "unique_episodes": 1,
            "coverage_success_rate": 0.0,
            "goal_state_success_diagnostic_rate": 1.0,
            "state_dist": 5.0,
            "planning_latency_sec": 0.1,
        },
        "hierarchical": {
            "unique_episodes": 1,
            "coverage_success_rate": 1.0,
            "goal_state_success_diagnostic_rate": 0.0,
            "state_dist": 1.0,
            "planning_latency_sec": 0.2,
        },
    }
    monkeypatch.setattr(release_artifacts, "LOCKED_EVAL", eval_dir)
    monkeypatch.setattr(release_artifacts, "LOCKED_REPORT", report_path)

    sanitized = release_artifacts.sanitize_locked_in_place(records, summary)
    refreshed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_text = records_path.read_text(encoding="utf-8")

    assert refreshed_summary["cache_path"] == "external/legacy_debug_cache.h5"
    assert refreshed_summary["flat"]["coverage_success_rate"] == 0.0
    assert refreshed_summary["hierarchical"]["coverage_success_rate"] == 1.0
    assert "success_rate" not in refreshed_summary["flat"]
    assert "success_rate" not in refreshed_summary["hierarchical"]
    assert refreshed_summary["flat"]["goal_state_success_scope"] == "trajectory_full_state_diagnostic"
    assert "goal_state_success_rate" not in refreshed_summary["flat"]
    assert "success" not in refreshed_summary["flat"]["records"][0]
    assert sanitized[0]["coverage_success"] == "False"
    assert sanitized[0]["goal_state_success"] == "True"
    assert "success" not in csv_text.splitlines()[0].split(",")
    assert "private/old" not in summary_path.read_text(encoding="utf-8")


def test_joint_requires_passive_checkpoint_when_low_level_checkpoint_is_set(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "joint.yaml"
    checkpoint = tmp_path / "low_level.pt"
    output_dir = tmp_path / "joint_out"
    checkpoint.write_bytes(b"placeholder")
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 0",
                "device: cpu",
                "data:",
                "  cache_path: unused.h5",
                "  batch_size: 1",
                "  num_workers: 0",
                "  labeled_fraction: 0.1",
                "  val_fraction: 0.0",
                "  test_fraction: 0.0",
                "training:",
                f"  low_level_checkpoint: {checkpoint.as_posix()}",
                "  passive_checkpoint: null",
                f"  joint_output_dir: {output_dir.as_posix()}",
                "  chunk_size: 1",
                "  rollout_chunks: 1",
                "  low_rollout_steps: 1",
                "  lr: 0.001",
                "  weight_decay: 0.0",
                "  grad_clip: 1.0",
                "  epochs: 0",
                "  log_every: 1",
                "model: {}",
                "planner: {}",
                "loss: {}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["train_joint", "--config", str(cfg_path)])
    monkeypatch.setattr("skill_jepa.trainers.train_joint.FeatureSequenceDataset", lambda *args, **kwargs: [])
    monkeypatch.setattr("skill_jepa.trainers.train_joint.DataLoader", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "skill_jepa.trainers.train_joint.build_all_modules",
        lambda cfg, cache_path: {"skill_idm": nn.Linear(1, 1), "action_chunk_encoder": nn.Linear(1, 1)},
    )

    with pytest.raises(RuntimeError, match="requires an explicit passive_checkpoint"):
        train_joint_main()


def test_prepare_output_dir_rejects_partial_stale_outputs(tmp_path: Path):
    out_dir = tmp_path / "eval"
    out_dir.mkdir()
    (out_dir / "pusht_online_records.csv").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pusht_online_records.csv"):
        _prepare_output_dir(out_dir, force=False)


def test_prepare_output_dir_force_removes_known_outputs(tmp_path: Path):
    out_dir = tmp_path / "eval"
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True)
    (out_dir / "pusht_online_eval.json").write_text("{}", encoding="utf-8")
    (out_dir / "pusht_online_records.csv").write_text("stale", encoding="utf-8")
    (video_dir / "episode.gif").write_text("stale", encoding="utf-8")

    _prepare_output_dir(out_dir, force=True)

    assert not (out_dir / "pusht_online_eval.json").exists()
    assert not (out_dir / "pusht_online_records.csv").exists()
    assert not video_dir.exists()


def test_subgoal_resolver_respects_allowed_indices(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("z", data=np.array([[10.0, 0.0], [0.0, 0.0]], dtype=np.float32))
        handle.create_dataset(
            "s",
            data=np.array(
                [
                    [[1.0, 1.0]],
                    [[9.0, 9.0]],
                ],
                dtype=np.float32,
            ),
        )

    resolver = NearestSubgoalResolver(str(cache_path), torch.device("cpu"), allowed_indices=np.array([0]))
    subgoal_s = resolver(torch.tensor([0.0, 0.0]))

    assert torch.allclose(subgoal_s, torch.tensor([[1.0, 1.0]]))


def test_subgoal_resolver_maps_sparse_allowed_indices_to_matching_s(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    z = np.zeros((6, 2), dtype=np.float32)
    z[2] = np.array([10.0, 0.0], dtype=np.float32)
    z[5] = np.array([0.0, 10.0], dtype=np.float32)
    s = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("z", data=z)
        handle.create_dataset("s", data=s)

    resolver = NearestSubgoalResolver(str(cache_path), torch.device("cpu"), chunk_size=1, allowed_indices=np.array([2, 5]))
    subgoal_s = resolver(torch.tensor([0.0, 9.5]))

    assert torch.allclose(subgoal_s, torch.tensor([[5.0]]))


def test_subgoal_resolver_rejects_empty_allowed_indices(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("z", data=np.zeros((1, 2), dtype=np.float32))
        handle.create_dataset("s", data=np.zeros((1, 1, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="empty allowed index"):
        NearestSubgoalResolver(str(cache_path), torch.device("cpu"), allowed_indices=np.array([], dtype=np.int64))


def test_build_planners_wires_train_only_subgoal_indices(monkeypatch):
    calls = {}

    class DummyPlanner:
        def __init__(self, *args, **kwargs):
            pass

    class DummyHierarchicalPlanner:
        def __init__(self, high_level, low_level, subgoal_resolver=None):
            self.subgoal_resolver = subgoal_resolver

    class DummyResolver:
        def __init__(self, cache_path, device, chunk_size=4096, allowed_indices=None):
            self.allowed_indices = allowed_indices
            calls["allowed_indices"] = allowed_indices

    monkeypatch.setattr(online_eval, "HighLevelCEMPlanner", DummyPlanner)
    monkeypatch.setattr(online_eval, "RandomHighLevelPlanner", DummyPlanner)
    monkeypatch.setattr(online_eval, "LowLevelCEMPlanner", DummyPlanner)
    monkeypatch.setattr(online_eval, "HierarchicalPlanner", DummyHierarchicalPlanner)
    monkeypatch.setattr(online_eval, "NearestSubgoalResolver", DummyResolver)
    monkeypatch.setattr(online_eval, "_split_step_indices", lambda cache_path, split, cfg: np.array([5, 6], dtype=np.int64))

    cfg = {
        "seed": 999,
        "data": {"cache_path": "cache.h5", "val_fraction": 0.2, "test_fraction": 0.2, "split_seed": 0},
        "model": {"skill_dim": 2},
        "planner": {
            "high_level_horizon": 1,
            "high_level_population": 2,
            "high_level_elites": 1,
            "high_level_iters": 1,
            "high_level_skill_penalty": 0.0,
            "high_level_prior_penalty": 0.0,
            "low_level_horizon": 1,
            "low_level_population": 2,
            "low_level_elites": 1,
            "low_level_iters": 1,
            "low_level_skill_penalty": 0.0,
            "action_penalty": 0.0,
        },
    }

    modules = {
        "skill_wm": object(),
        "skill_prior": object(),
        "low_level_wm": object(),
        "action_chunk_encoder": object(),
    }
    planners = _build_planners(cfg, modules=modules, meta={"action_dim": 2}, device=torch.device("cpu"), subgoal_scope="train")
    np.testing.assert_array_equal(calls["allowed_indices"], np.array([5, 6], dtype=np.int64))
    assert planners["hierarchical"].subgoal_resolver.allowed_indices is calls["allowed_indices"]


def test_eval_provenance_rejects_projector_mismatch(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    projector_a = tmp_path / "projector_a.pt"
    projector_b = tmp_path / "projector_b.pt"
    projector_a.write_bytes(b"projector-a")
    projector_b.write_bytes(b"projector-b")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector_a)

    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector_b)},
        "encoder": {"model_id": "encoder", "state_dim": 4},
        "model": {"hidden_dim": 8},
    }
    checkpoint_payload = {
        "config": {
            "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector_a)},
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        }
    }

    with pytest.raises(RuntimeError, match="projector"):
        _validate_eval_provenance(cfg, checkpoint_payload)


def test_eval_provenance_requires_checkpoint_config(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    projector.write_bytes(b"projector")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector)
    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder", "state_dim": 4},
        "model": {"hidden_dim": 8},
    }

    with pytest.raises(RuntimeError, match="training config"):
        _validate_eval_provenance(cfg, checkpoint_payload={})


def test_eval_provenance_rejects_checkpoint_model_config_mismatch(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    projector.write_bytes(b"projector")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector)
    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder", "state_dim": 4},
        "model": {"hidden_dim": 8},
    }
    checkpoint_payload = {
        "config": {
            "data": cfg["data"],
            "encoder": cfg["encoder"],
            "model": {"hidden_dim": 16},
        }
    }

    with pytest.raises(RuntimeError, match="model config"):
        _validate_eval_provenance(cfg, checkpoint_payload)


def test_eval_provenance_rejects_checkpoint_projector_hash_mismatch(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    projector.write_bytes(b"current-projector")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector)
        handle.attrs["projector_ckpt_sha256"] = hashlib.sha256(b"current-projector").hexdigest()
    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder", "state_dim": 4},
        "model": {"hidden_dim": 8},
    }
    checkpoint_payload = {
        "config": {
            "data": cfg["data"],
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"old-projector").hexdigest(),
        },
    }

    with pytest.raises(RuntimeError, match="data.projector_ckpt hash"):
        _validate_eval_provenance(cfg, checkpoint_payload)


def test_eval_provenance_rejects_in_place_projector_overwrite(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    projector.write_bytes(b"current-projector")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector)
        handle.attrs["projector_ckpt_sha256"] = hashlib.sha256(b"old-projector").hexdigest()

    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder", "state_dim": 4},
        "model": {"hidden_dim": 8},
    }
    checkpoint_payload = {
        "config": {
            "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"current-projector").hexdigest(),
        },
    }

    with pytest.raises(RuntimeError, match="projector_ckpt_sha256"):
        _validate_eval_provenance(cfg, checkpoint_payload)


def test_eval_provenance_reports_cache_hash_check_when_available(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    projector.write_bytes(b"projector")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector)
        handle.attrs["projector_ckpt_sha256"] = hashlib.sha256(b"projector").hexdigest()

    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder", "state_dim": 4},
        "model": {"hidden_dim": 8},
    }
    checkpoint_payload = {
        "config": {
            "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
        },
    }

    summary = _validate_eval_provenance(cfg, checkpoint_payload)

    assert summary["cache_projector_hash_checked"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"encoder_model_id": "wrong-encoder"}, "encoder_model_id"),
        ({"encoder_state_dim": 5}, "encoder_state_dim"),
        ({"encoder_pool_grid": 3}, "encoder_pool_grid"),
    ],
)
def test_eval_provenance_rejects_cache_encoder_metadata_mismatch(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
):
    cache_path = tmp_path / "cache.h5"
    projector = tmp_path / "projector.pt"
    projector.write_bytes(b"projector")
    with h5py.File(cache_path, "w") as handle:
        handle.attrs["projector_ckpt"] = str(projector)
        handle.attrs["projector_ckpt_sha256"] = hashlib.sha256(b"projector").hexdigest()
        handle.attrs["encoder_model_id"] = "encoder"
        handle.attrs["encoder_state_dim"] = 4
        handle.attrs["encoder_pool_grid"] = 2
        for key, value in override.items():
            handle.attrs[key] = value
    cfg = {
        "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
        "encoder": {"model_id": "encoder", "state_dim": 4, "pool_grid": 2},
        "model": {"hidden_dim": 8},
    }
    checkpoint_payload = {
        "config": {
            "data": {"cache_path": str(cache_path), "projector_ckpt": str(projector)},
            "encoder": cfg["encoder"],
            "model": cfg["model"],
        },
        "artifact_hashes": {
            "data.cache_path": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "data.projector_ckpt": hashlib.sha256(b"projector").hexdigest(),
        },
    }

    with pytest.raises(RuntimeError, match=message):
        _validate_eval_provenance(cfg, checkpoint_payload)
