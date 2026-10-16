"""Tensor helpers shared by backbone wrappers and the evaluation harness (torch)."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "pad_to_multiple", "upsample_logits", "normalize"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pad_to_multiple(images: torch.Tensor, multiple: int, mode: str = "replicate") -> torch.Tensor:
    """Pad (B, C, H, W) at the bottom/right so H and W are multiples of ``multiple``."""
    if multiple <= 1:
        return images
    h, w = images.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return images
    return F.pad(images, (0, pw, 0, ph), mode=mode)


def upsample_logits(
    logits: torch.Tensor, size: Tuple[int, int], output_stride: int, align_corners: bool
) -> torch.Tensor:
    """Bring backbone logits back to an ``(H, W)`` input resolution.

    Implements the :class:`models.interfaces.SegmentationBackbone` contract:
    the ``(h, w)`` logit grid covers the input padded to ``(h*s, w*s)``, so we
    interpolate to that padded size and crop -- exact for every ``H, W``. When
    ``H, W`` are multiples of ``s`` this is a plain interpolation to ``(H, W)``.
    """
    height, width = int(size[0]), int(size[1])
    h, w = logits.shape[-2:]
    s = int(output_stride)
    target = (h * s, w * s)
    if target[0] < height or target[1] < width:
        raise ValueError(f"logits {tuple(logits.shape[-2:])} at stride {s} cannot cover input {size}")
    if target != (h, w):
        logits = F.interpolate(logits, size=target, mode="bilinear", align_corners=align_corners)
    return logits[..., :height, :width]


def normalize(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.as_tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    sd = torch.as_tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images - m) / sd
