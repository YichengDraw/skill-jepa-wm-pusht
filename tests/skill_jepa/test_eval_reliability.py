from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from skill_jepa.analysis.eval_pusht_online import (
    NearestSubgoalResolver,
    _coverage_success,
    _eval_split_summary_fields,
    _goal_state_eval,
    _prepare_output_dir,
    _set_eval_seed,
)
from skill_jepa.data import EpisodeGoalSampler
from skill_jepa.envs import PushTEnv
from skill_jepa.trainers.common import load_checkpoint


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

    without_replacement = sampler.sample(10, seed=0, allow_replacement=False)
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
    import tomllib

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    include = set(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])
    where = pyproject["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["src"]
    assert include == {"skill_jepa*"}


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


def test_goal_state_diagnostic_uses_block_pose_not_agent_position():
    goal = np.array([0.0, 0.0, 200.0, 200.0, 0.0], dtype=np.float32)
    current = np.array([500.0, 500.0, 202.0, 199.0, 0.01], dtype=np.float32)

    metrics = _goal_state_eval(goal, current)

    assert metrics["goal_state_success"]
    assert metrics["state_dist"] < 3.0


def test_goal_state_angle_distance_wraps_periodically():
    goal = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.01], dtype=np.float32)
    current = np.array([0.0, 0.0, 0.0, 0.0, -np.pi + 0.01], dtype=np.float32)

    metrics = _goal_state_eval(goal, current)

    assert metrics["goal_state_success"]
    assert metrics["state_dist"] < 0.03


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
