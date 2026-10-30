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
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.utils import upsample_logits

from .ops import (SCORE_KINDS, GlimpseGrid, cell_features, downsample, gather_crops, paste_labels, pool_scores,
                  samples_per_cell, select_greedy, select_topk)

__all__ = ["FoveationConfig", "FoveatedSegmenter", "StubBackbone", "context_pass", "glimpse_labels"]


def context_pass(backbone: nn.Module, frame: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(1, C, h, w) context logits and the (H, W) labels they give at full resolution."""
    small = downsample(frame, scale)
    logits = backbone(small)
    labels = upsample_logits(logits, tuple(frame.shape[-2:]), backbone.output_stride, backbone.align_corners,
                             input_size=tuple(small.shape[-2:])).argmax(1)[0]
    return logits, labels


def glimpse_labels(backbone: nn.Module, crops: torch.Tensor) -> torch.Tensor:
    """(k, 3, size, size) crops -> (k, size, size) labels."""
    size = tuple(crops.shape[-2:])
    return upsample_logits(backbone(crops), size, backbone.output_stride, backbone.align_corners).argmax(1)


@dataclass(frozen=True)
class FoveationConfig:
    context_scale: float = 0.25
    glimpse_size: int = 384
    glimpse_margin: int = 32
    k: int = 4
    selection: str = "topk"  # topk | greedy
    suppress: int = 1  # greedy only: neighbourhood (cells) excluded after a pick
    score: str = "entropy"  # entropy | margin | maxprob (foveation.ops.score_map)
    pool: str = "mean"  # mean | top10 (foveation.ops.pool_scores)
    class_weights: Optional[Tuple[float, ...]] = None  # mean pooling only: weight per predicted class


class FoveatedSegmenter(nn.Module):
    def __init__(self, backbone_ctx: nn.Module, backbone_glimpse: nn.Module, cfg: FoveationConfig,
                 frame_size: Tuple[int, int] = (1024, 2048)):
        super().__init__()
        if cfg.selection not in ("topk", "greedy"):
            raise ValueError(f"unknown selection {cfg.selection!r}")
        if cfg.score not in SCORE_KINDS:
            raise ValueError(f"unknown score {cfg.score!r}, expected one of {SCORE_KINDS}")
        if cfg.pool not in ("mean", "top10") or (cfg.pool == "top10" and cfg.class_weights is not None):
            raise ValueError(f"invalid pooling {cfg.pool!r} (class weights need mean pooling)")
        self.ctx, self.glimpse, self.cfg = backbone_ctx, backbone_glimpse, cfg
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.grid = GlimpseGrid(self.frame_size, cfg.glimpse_size, cfg.glimpse_margin)
        if cfg.k > self.grid.num_cells:
            raise ValueError(f"k={cfg.k} exceeds the {self.grid.num_cells} grid cells")
        weights = None if cfg.class_weights is None else torch.tensor(cfg.class_weights, dtype=torch.float32)
        self.register_buffer("_class_weights", weights, persistent=False)  # moves with .cuda(): no copy in forward

    def forward(self, frame: torch.Tensor,
                return_idx: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cfg, grid = self.cfg, self.grid
        h, w = self.frame_size
        ctx_logits, labels = context_pass(self.ctx, frame, cfg.context_scale)
        sums, top10 = cell_features(ctx_logits, grid, [cfg.score])[cfg.score]
        scores = pool_scores(sums, top10, samples_per_cell(ctx_logits.shape[-2], grid), cfg.pool, self._class_weights)
        idx = select_topk(scores, cfg.k) if cfg.selection == "topk" else select_greedy(scores, grid, cfg.k, cfg.suppress)
        crop_labels = glimpse_labels(self.glimpse, gather_crops(grid.pad_frame(frame), grid, idx))
        canvas = paste_labels(grid.to_canvas(labels), crop_labels, grid, idx)
        return (canvas[:h, :w], idx) if return_idx else canvas[:h, :w]

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
