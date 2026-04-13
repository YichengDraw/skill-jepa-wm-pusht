from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch

from skill_jepa.encoders import FrozenVJEPA2Encoder
from skill_jepa.modules import StateProjector
from skill_jepa.utils import ensure_dir, load_yaml, seed_everything


def _trim_episode_layout(ep_len: np.ndarray, max_steps: int | None, max_episodes: int | None):
    if max_episodes is not None:
        ep_len = ep_len[: int(max_episodes)]
    if max_steps is None:
        offsets = np.zeros_like(ep_len, dtype=np.int64)
        if len(offsets) > 1:
            offsets[1:] = np.cumsum(ep_len[:-1], dtype=np.int64)
        return ep_len.astype(np.int32), offsets
    new_lengths = []
    total = 0
    for length in ep_len.tolist():
        if total >= max_steps:
            break
        keep = min(int(length), max_steps - total)
        if keep <= 0:
            break
        new_lengths.append(keep)
        total += keep
    new_lengths = np.asarray(new_lengths, dtype=np.int32)
    new_offsets = np.zeros_like(new_lengths, dtype=np.int64)
    if len(new_lengths) > 1:
        new_offsets[1:] = np.cumsum(new_lengths[:-1], dtype=np.int64)
    return new_lengths, new_offsets


def _build_indices(ep_len: np.ndarray, ep_offset: np.ndarray):
    total = int(ep_len.sum())
    episode_idx = np.zeros(total, dtype=np.int64)
    step_idx = np.zeros(total, dtype=np.int64)
    for episode_id, (offset, length) in enumerate(zip(ep_offset.tolist(), ep_len.tolist())):
        episode_idx[offset : offset + length] = episode_id
        step_idx[offset : offset + length] = np.arange(length, dtype=np.int64)
    return episode_idx, step_idx


def _build_clips(
    pixel_window: np.ndarray,
    start: int,
    end: int,
    window_start: int,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
) -> np.ndarray:
    clips = []
    for current in range(start, end):
        prev = max(current - 1, 0)
        if episode_idx[prev] != episode_idx[current] or step_idx[current] == 0:
            prev = current
        prev_frame = pixel_window[prev - window_start]
        cur_frame = pixel_window[current - window_start]
        clips.append(np.stack([prev_frame, cur_frame], axis=0))
    return np.stack(clips, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))

    raw_h5 = Path(cfg["data"]["raw_h5_path"])
    cache_h5 = Path(cfg["data"]["cache_path"])
    projector_ckpt = Path(cfg["data"]["projector_ckpt"])
    ensure_dir(cache_h5.parent)
    ensure_dir(projector_ckpt.parent)

    device = cfg.get("device")
    encoder = FrozenVJEPA2Encoder(
        model_id=cfg["encoder"]["model_id"],
        dtype=cfg["encoder"].get("dtype", "bfloat16"),
        device=device,
    )
    projector = StateProjector(
        input_dim=encoder.hidden_size,
        output_dim=cfg["encoder"]["state_dim"],
        pool_grid=cfg["encoder"]["pool_grid"],
    ).to(encoder.device)
    if projector_ckpt.exists():
        projector.load_state_dict(torch.load(projector_ckpt, map_location="cpu"))
    else:
        torch.save({k: v.cpu() for k, v in projector.state_dict().items()}, projector_ckpt)
    projector.eval()

    with h5py.File(raw_h5, "r") as src:
        raw_ep_len = src["ep_len"][:]
        ep_len, ep_offset = _trim_episode_layout(
            raw_ep_len,
            cfg["data"].get("max_steps"),
            cfg["data"].get("max_episodes"),
        )
        num_steps = int(ep_len.sum())
        batch_size = int(cfg["data"]["cache_batch_size"])

        with h5py.File(cache_h5, "w") as dst:
            dst.attrs["source_h5"] = str(raw_h5)
            dst.attrs["projector_ckpt"] = str(projector_ckpt)
            dst.create_dataset("ep_len", data=ep_len)
            dst.create_dataset("ep_offset", data=ep_offset)
            if "episode_idx" in src and "step_idx" in src:
                episode_idx = src["episode_idx"][:num_steps]
                step_idx = src["step_idx"][:num_steps]
            else:
                episode_idx, step_idx = _build_indices(ep_len, ep_offset)
            dst.create_dataset("episode_idx", data=episode_idx)
            dst.create_dataset("step_idx", data=step_idx)
            if "action" in src:
                action_data = src["action"][:num_steps]
                dst.create_dataset("action", data=action_data)
                dst.attrs["action_mean"] = action_data.mean(axis=0)
                dst.attrs["action_std"] = action_data.std(axis=0) + 1e-6
                dst.attrs["action_low"] = action_data.min(axis=0)
                dst.attrs["action_high"] = action_data.max(axis=0)
            if "proprio" in src:
                dst.create_dataset("proprio", data=src["proprio"][:num_steps])
            if "state" in src:
                dst.create_dataset("state", data=src["state"][:num_steps])
            z_ds = dst.create_dataset(
                "z",
                shape=(num_steps, cfg["encoder"]["state_dim"]),
                dtype="float32",
                chunks=(min(batch_size, num_steps), cfg["encoder"]["state_dim"]),
                compression="lzf",
            )
            s_ds = dst.create_dataset(
                "s",
                shape=(num_steps, cfg["encoder"]["pool_grid"] ** 2, cfg["encoder"]["state_dim"]),
                dtype="float32",
                chunks=(min(batch_size, num_steps), cfg["encoder"]["pool_grid"] ** 2, cfg["encoder"]["state_dim"]),
                compression="lzf",
            )
            if cfg["data"].get("save_patch_tokens", False):
                patch_ds = dst.create_dataset(
                    "patch_tokens",
                    shape=(num_steps, encoder.patch_grid[0] * encoder.patch_grid[1], encoder.hidden_size),
                    dtype="float32",
                    chunks=(min(batch_size, num_steps), encoder.patch_grid[0] * encoder.patch_grid[1], encoder.hidden_size),
                    compression="lzf",
                )
            else:
                patch_ds = None

            for start in range(0, num_steps, batch_size):
                end = min(num_steps, start + batch_size)
                window_start = max(0, start - 1)
                pixel_window = src["pixels"][window_start:end]
                clips = torch.from_numpy(_build_clips(pixel_window, start, end, window_start, episode_idx, step_idx))
                patch_tokens = encoder.encode_images(clips)
                with torch.no_grad():
                    states = projector(patch_tokens)
                z_ds[start:end] = states["global_state"].detach().cpu().numpy()
                s_ds[start:end] = states["spatial_tokens"].detach().cpu().numpy()
                if patch_ds is not None:
                    patch_ds[start:end] = patch_tokens.detach().cpu().numpy()


if __name__ == "__main__":
    main()
