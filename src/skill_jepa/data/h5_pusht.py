from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def cache_metadata(cache_path: str | Path) -> Dict[str, int]:
    with h5py.File(cache_path, "r") as handle:
        meta = {
            "num_steps": int(handle["episode_idx"].shape[0]),
            "num_episodes": int(handle["ep_len"].shape[0]),
            "state_dim": int(handle["z"].shape[-1]),
            "num_tokens": int(handle["s"].shape[-2]),
            "token_dim": int(handle["s"].shape[-1]),
            "action_dim": int(handle["action"].shape[-1]) if "action" in handle else 0,
            "proprio_dim": int(handle["proprio"].shape[-1]) if "proprio" in handle else 0,
        }
        for key in ["action_mean", "action_std", "action_low", "action_high"]:
            if key in handle.attrs:
                meta[key] = np.asarray(handle.attrs[key], dtype=np.float32)
        return meta


def split_episode_ids(num_episodes: int, val_fraction: float, test_fraction: float, seed: int) -> Dict[str, np.ndarray]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    rng = np.random.default_rng(seed)
    ids = np.arange(num_episodes)
    rng.shuffle(ids)
    n_test = max(1, int(round(num_episodes * test_fraction))) if test_fraction > 0 and num_episodes > 2 else 0
    n_test = min(n_test, max(0, num_episodes - 1))
    remaining_after_test = num_episodes - n_test
    n_val = max(1, int(round(num_episodes * val_fraction))) if val_fraction > 0 and remaining_after_test > 1 else 0
    n_val = min(n_val, max(0, remaining_after_test - 1))
    test_ids = np.sort(ids[:n_test])
    val_ids = np.sort(ids[n_test : n_test + n_val])
    train_ids = np.sort(ids[n_test + n_val :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _labeled_episode_ids(train_ids: np.ndarray, labeled_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    ids = train_ids.copy()
    rng.shuffle(ids)
    n_labeled = max(1, int(round(len(ids) * labeled_fraction)))
    return np.sort(ids[:n_labeled])


@dataclass
class SampleIndex:
    start: int
    episode_id: int
    local_start: int
    labeled: bool


class FeatureSequenceDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
        sequence_length: int,
        split: str = "train",
        labeled_fraction: float = 0.1,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        stride: int = 1,
        seed: int = 0,
        use_only_labeled: bool | None = None,
    ) -> None:
        self.cache_path = str(cache_path)
        self.sequence_length = sequence_length
        self.split = split
        self.use_only_labeled = use_only_labeled
        self._handle: h5py.File | None = None
        with h5py.File(self.cache_path, "r") as handle:
            self.ep_offset = handle["ep_offset"][:]
            self.ep_len = handle["ep_len"][:]
        splits = split_episode_ids(len(self.ep_len), val_fraction, test_fraction, seed)
        train_ids = splits["train"]
        self.split_ids = splits[split]
        self.labeled_ids = set(_labeled_episode_ids(train_ids, labeled_fraction, seed).tolist())
        self.samples: List[SampleIndex] = []
        for episode_id in self.split_ids.tolist():
            ep_len = int(self.ep_len[episode_id])
            offset = int(self.ep_offset[episode_id])
            labeled = True if split != "train" else episode_id in self.labeled_ids
            if self.use_only_labeled is True and not labeled:
                continue
            if self.use_only_labeled is False and labeled:
                continue
            max_start = ep_len - sequence_length
            for local_start in range(0, max_start + 1, stride):
                self.samples.append(
                    SampleIndex(
                        start=offset + local_start,
                        episode_id=episode_id,
                        local_start=local_start,
                        labeled=labeled,
                    )
                )

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.cache_path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        start = sample.start
        end = start + self.sequence_length
        handle = self.handle
        batch = {
            "z": torch.from_numpy(handle["z"][start:end]).float(),
            "s": torch.from_numpy(handle["s"][start:end]).float(),
            "action": torch.from_numpy(handle["action"][start : end - 1]).float() if "action" in handle else None,
            "proprio": torch.from_numpy(handle["proprio"][start:end]).float() if "proprio" in handle else None,
            "state": torch.from_numpy(handle["state"][start:end]).float() if "state" in handle else None,
            "episode_id": torch.tensor(sample.episode_id, dtype=torch.long),
            "local_start": torch.tensor(sample.local_start, dtype=torch.long),
            "is_labeled": torch.tensor(sample.labeled, dtype=torch.bool),
        }
        if batch["action"] is None:
            batch["action"] = torch.zeros(self.sequence_length - 1, 0)
        if batch["proprio"] is None:
            batch["proprio"] = torch.zeros(self.sequence_length, 0)
        if batch["state"] is None:
            batch["state"] = torch.zeros(self.sequence_length, 0)
        return batch

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()


class EpisodeGoalSampler:
    def __init__(
        self,
        cache_path: str | Path,
        split: str = "test",
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        seed: int = 0,
        goal_gap: int = 16,
        fallback_empty_split: bool = False,
    ) -> None:
        self.cache_path = str(cache_path)
        self.goal_gap = goal_gap
        with h5py.File(self.cache_path, "r") as handle:
            self.ep_offset = handle["ep_offset"][:]
            self.ep_len = handle["ep_len"][:]
        splits = split_episode_ids(len(self.ep_len), val_fraction, test_fraction, seed)
        self.requested_split = split
        self.actual_split = split
        self.episode_ids = splits[split]
        if len(self.episode_ids) == 0:
            if not fallback_empty_split:
                raise ValueError(
                    f"Requested split {split!r} has no episodes. "
                    "Use fallback_empty_split=True only for explicit debug fallback."
                )
            self.actual_split = "val" if len(splits["val"]) > 0 else "train"
            self.episode_ids = splits[self.actual_split]

    def sample(
        self,
        num_pairs: int,
        seed: int,
        max_goal_gap: int | None = None,
        allow_replacement: bool = False,
    ) -> List[Dict[str, np.ndarray]]:
        if max_goal_gap is not None and int(max_goal_gap) < int(self.goal_gap):
            raise ValueError(f"max_goal_gap={max_goal_gap} is smaller than goal_gap={self.goal_gap}")
        rng = np.random.default_rng(seed)
        result: List[Dict[str, np.ndarray]] = []
        with h5py.File(self.cache_path, "r") as handle:
            if len(self.episode_ids) == 0 or int(num_pairs) <= 0:
                return result
            valid_episode_ids = np.asarray(
                [int(episode_id) for episode_id in self.episode_ids.tolist() if int(self.ep_len[episode_id]) > self.goal_gap],
                dtype=np.int64,
            )
            if len(valid_episode_ids) == 0:
                return result
            sample_size = int(num_pairs)
            if not allow_replacement:
                sample_size = min(sample_size, len(valid_episode_ids))
            chosen_eps = rng.choice(
                valid_episode_ids,
                size=sample_size,
                replace=bool(allow_replacement and len(valid_episode_ids) < sample_size),
            )
            for episode_id in chosen_eps.tolist():
                ep_len = int(self.ep_len[episode_id])
                offset = int(self.ep_offset[episode_id])
                max_start = ep_len - self.goal_gap
                start_local = int(rng.integers(0, max_start))
                goal_upper = ep_len
                if max_goal_gap is not None:
                    goal_upper = min(goal_upper, start_local + max_goal_gap + 1)
                goal_lower = start_local + self.goal_gap
                if goal_upper <= goal_lower:
                    raise RuntimeError(
                        f"Invalid goal window for episode={episode_id}, start={start_local}, "
                        f"goal_gap={self.goal_gap}, max_goal_gap={max_goal_gap}"
                    )
                goal_local = int(rng.integers(goal_lower, goal_upper))
                start_idx = offset + start_local
                goal_idx = offset + goal_local
                result.append(
                    {
                        "episode_id": np.int64(episode_id),
                        "start_index": np.int64(start_idx),
                        "goal_index": np.int64(goal_idx),
                        "start_state": handle["state"][start_idx].copy(),
                        "goal_state": handle["state"][goal_idx].copy(),
                    }
                )
        return result
