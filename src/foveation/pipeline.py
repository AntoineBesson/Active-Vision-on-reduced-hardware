"""Memory-free foveated segmentation of one frame: context view + k glimpses (torch).

    frame (1, 3, H, W)
      -> context = backbone_ctx(downsample(frame, context_scale))      (1, C, h, w)
      -> uncertainty map (entropy) -> one score per grid cell -> k cells
      -> crops (k, 3, size, size) from the padded frame -> backbone_glimpse  (one batch)
      -> labels = argmax(upsampled context), then the inner square of each
         glimpse's argmax pasted over it                                (H, W)

Fixed shapes for a fixed configuration and no host synchronisation (see
``foveation.ops``), so the whole call can be captured in a CUDA graph. The
context and glimpse backbones may differ (any ``SegmentationBackbone``).

This is the v0 used for the latency-floor experiment; selection rules and
recomposition are research knobs of the method that builds on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from models.utils import upsample_logits

from .ops import GlimpseGrid, cell_scores, downsample, entropy_map, gather_crops, paste_labels, select_greedy, select_topk

__all__ = ["FoveationConfig", "FoveatedSegmenter", "StubBackbone"]


@dataclass(frozen=True)
class FoveationConfig:
    context_scale: float = 0.25
    glimpse_size: int = 384
    glimpse_margin: int = 32
    k: int = 4
    selection: str = "topk"  # topk | greedy
    suppress: int = 1  # greedy only: neighbourhood (cells) excluded after a pick


class FoveatedSegmenter(nn.Module):
    def __init__(self, backbone_ctx: nn.Module, backbone_glimpse: nn.Module, cfg: FoveationConfig,
                 frame_size: Tuple[int, int] = (1024, 2048)):
        super().__init__()
        if cfg.selection not in ("topk", "greedy"):
            raise ValueError(f"unknown selection {cfg.selection!r}")
        self.ctx, self.glimpse, self.cfg = backbone_ctx, backbone_glimpse, cfg
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.grid = GlimpseGrid(self.frame_size, cfg.glimpse_size, cfg.glimpse_margin)
        if cfg.k > self.grid.num_cells:
            raise ValueError(f"k={cfg.k} exceeds the {self.grid.num_cells} grid cells")

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        cfg, grid = self.cfg, self.grid
        h, w = self.frame_size
        small = downsample(frame, cfg.context_scale)
        ctx_logits = self.ctx(small)
        scores = cell_scores(entropy_map(ctx_logits)[0], grid)
        idx = select_topk(scores, cfg.k) if cfg.selection == "topk" else select_greedy(scores, grid, cfg.k, cfg.suppress)
        crops = gather_crops(grid.pad_frame(frame), grid, idx)
        crop_logits = self.glimpse(crops)
        size = (grid.size, grid.size)
        crop_labels = upsample_logits(crop_logits, size, self.glimpse.output_stride, self.glimpse.align_corners).argmax(1)
        labels = upsample_logits(ctx_logits, (h, w), self.ctx.output_stride, self.ctx.align_corners,
                                 input_size=tuple(small.shape[-2:])).argmax(1)[0]
        canvas = paste_labels(grid.to_canvas(labels), crop_labels, grid, idx)
        return canvas[:h, :w]

    def describe(self) -> Dict[str, Any]:
        g = self.grid
        return {"config": asdict(self.cfg), "frame_size": list(self.frame_size),
                "grid": {"rows": g.rows, "cols": g.cols, "inner": g.inner, "cells": g.num_cells},
                "pixel_fraction": self.cfg.context_scale ** 2 + self.cfg.k * g.size ** 2 / (self.frame_size[0] * self.frame_size[1])}


class StubBackbone(nn.Module):
    """Zero-compute backbone: returns a preallocated logits buffer of the shape
    a real backbone with ``output_stride`` would produce. Isolates the
    pipeline's own overhead (resize, selection, crop, upsample, paste)."""

    def __init__(self, num_classes: int = 19, output_stride: int = 8, align_corners: bool = True):
        super().__init__()
        self.num_classes, self.output_stride, self.align_corners = num_classes, output_stride, align_corners
        self._buffers_by_shape: Dict[Tuple[int, ...], torch.Tensor] = {}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, _, hh, ww = images.shape
        s = self.output_stride
        key = (b, -(-hh // s), -(-ww // s), images.dtype, images.device)
        if key not in self._buffers_by_shape:
            g = torch.Generator(device="cpu").manual_seed(0)
            t = torch.randn(b, self.num_classes, key[1], key[2], generator=g)
            self._buffers_by_shape[key] = t.to(device=images.device, dtype=images.dtype)
        return self._buffers_by_shape[key]
