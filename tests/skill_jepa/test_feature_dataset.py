from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

from skill_jepa.data import FeatureSequenceDataset


def test_feature_sequence_dataset_indexes_sequences(tmp_path: Path):
    cache_path = tmp_path / "cache.h5"
    ep_len = np.array([5, 6], dtype=np.int32)
    ep_offset = np.array([0, 5], dtype=np.int64)
    num_steps = int(ep_len.sum())
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("ep_len", data=ep_len)
        handle.create_dataset("ep_offset", data=ep_offset)
        handle.create_dataset("episode_idx", data=np.array([0] * 5 + [1] * 6))
        handle.create_dataset("step_idx", data=np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]))
        handle.create_dataset("z", data=np.zeros((num_steps, 384), dtype=np.float32))
        handle.create_dataset("s", data=np.zeros((num_steps, 16, 384), dtype=np.float32))
        handle.create_dataset("action", data=np.zeros((num_steps, 2), dtype=np.float32))
        handle.create_dataset("proprio", data=np.zeros((num_steps, 4), dtype=np.float32))
        handle.create_dataset("state", data=np.zeros((num_steps, 7), dtype=np.float32))

    dataset = FeatureSequenceDataset(
        cache_path=cache_path,
        sequence_length=4,
        split="train",
        labeled_fraction=0.5,
        val_fraction=0.0,
        test_fraction=0.0,
        stride=1,
        seed=0,
    )
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample["z"].shape == (4, 384)
    assert sample["s"].shape == (4, 16, 384)
    assert sample["action"].shape == (3, 2)
