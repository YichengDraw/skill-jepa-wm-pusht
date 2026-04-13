from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoVideoProcessor, VJEPA2Model


def _center_crop(images: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = images.shape
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return images[:, :, top : top + size, left : left + size]


class FrozenVJEPA2Encoder(nn.Module):
    """Frozen V-JEPA2 wrapper for image batches."""

    def __init__(
        self,
        model_id: str = "facebook/vjepa2-vitl-fpc64-256",
        dtype: str = "bfloat16",
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        torch_dtype = torch.bfloat16 if dtype.lower() == "bfloat16" else torch.float32
        self.model = VJEPA2Model.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.resize_shortest_edge = int(self.processor.size["shortest_edge"])
        self.crop_size = int(self.processor.crop_size["height"])
        self.image_mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.autocast_dtype = torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float32

    @torch.no_grad()
    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW or BHWC tensor, got {tuple(images.shape)}")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        return images

    @torch.no_grad()
    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            images = self._normalize_images(frames)
            videos = images.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        elif frames.ndim == 5:
            if frames.shape[-1] == 3:
                frames = frames.permute(0, 1, 4, 2, 3)
            videos = torch.stack([self._normalize_images(step) for step in frames.unbind(dim=1)], dim=1)
        else:
            raise ValueError(f"Expected BCHW/BHWC or BTCHW/BTHWC tensor, got {tuple(frames.shape)}")
        _, _, _, h, w = videos.shape
        shortest = min(h, w)
        scale = self.resize_shortest_edge / shortest
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        flat = videos.reshape(-1, videos.shape[-3], h, w)
        flat = F.interpolate(flat, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        flat = _center_crop(flat, self.crop_size)
        mean = self.image_mean.to(flat.device)
        std = self.image_std.to(flat.device)
        flat = (flat - mean) / std
        return flat.reshape(videos.shape[0], videos.shape[1], flat.shape[1], flat.shape[2], flat.shape[3])

    @torch.no_grad()
    def encode_images(self, frames: torch.Tensor) -> torch.Tensor:
        videos = self.preprocess(frames).to(self.device)
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(pixel_values_videos=videos, skip_predictor=True)
        else:
            outputs = self.model(pixel_values_videos=videos, skip_predictor=True)
        return outputs.last_hidden_state.float().clone()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def patch_grid(self) -> Tuple[int, int]:
        size = self.crop_size // int(self.model.config.patch_size)
        return size, size
