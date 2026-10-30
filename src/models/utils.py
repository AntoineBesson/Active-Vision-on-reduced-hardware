"""Tensor helpers shared by backbone wrappers and the evaluation harness (torch)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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
    logits: torch.Tensor,
    size: Tuple[int, int],
    output_stride: int,
    align_corners: bool,
    input_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Bring backbone logits to an ``(H, W)`` output resolution.

    Implements the :class:`models.interfaces.SegmentationBackbone` contract:
    the ``(gh, gw)`` logit grid covers the backbone input of size
    ``input_size = (h, w)`` (default: ``size``) padded to ``(gh*s, gw*s)``.
    That padded extent, mapped to output pixels (``x H/h``, ``x W/w``), is
    interpolated in one step and cropped to ``(H, W)``. Without padding this
    is a plain interpolation to ``(H, W)``; at the input resolution it is exact.
    """
    height, width = int(size[0]), int(size[1])
    in_h, in_w = (height, width) if input_size is None else (int(input_size[0]), int(input_size[1]))
    gh, gw = logits.shape[-2:]
    s = int(output_stride)
    if gh * s < in_h or gw * s < in_w:
        raise ValueError(f"logits {(gh, gw)} at stride {s} cannot cover input {(in_h, in_w)}")
    target = (int(round(gh * s * height / in_h)), int(round(gw * s * width / in_w)))
    if target != (gh, gw):
        logits = F.interpolate(logits, size=target, mode="bilinear", align_corners=align_corners)
    return logits[..., :height, :width]


def normalize(images: torch.Tensor, mean, std) -> torch.Tensor:
    """``(images - mean) / std`` per channel. ``mean``/``std`` are sequences or
    tensors; tensors already on the right device and dtype are used as is (no copy)."""
    m = torch.as_tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    sd = torch.as_tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return (images - m) / sd
