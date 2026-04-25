import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from skill_jepa.analysis.eval_pusht_online import (
    NearestSubgoalResolver,
    _assign_task_goals,
    _coverage_success,
    _eval_split_summary_fields,
    _goal_state_eval,
    _prepare_output_dir,
    _select_task_goal_indices,
    _set_eval_seed,
    _same_record_rate,
    _validate_eval_provenance,
    _validate_goal_pairs,
)
from skill_jepa.data import EpisodeGoalSampler, split_episode_ids
from skill_jepa.envs import PushTEnv
from skill_jepa.trainers.common import load_checkpoint, load_checkpoint_subset
from skill_jepa.trainers.train_joint import _assert_low_level_passive_lineage, main as train_joint_main


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


def test_joint_rejects_low_level_checkpoint_from_different_passive_lineage(tmp_path: Path):
    recorded_passive = tmp_path / "recorded_passive.pt"
    current_passive = tmp_path / "current_passive.pt"
    recorded_passive.write_bytes(b"recorded")
    current_passive.write_bytes(b"current")
    cfg = {"training": {"passive_checkpoint": str(current_passive)}}
    low_level_payload = {"config": {"training": {"passive_checkpoint": str(recorded_passive)}}}

    with pytest.raises(RuntimeError, match="passive lineage"):
        _assert_low_level_passive_lineage(cfg, low_level_payload)


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
