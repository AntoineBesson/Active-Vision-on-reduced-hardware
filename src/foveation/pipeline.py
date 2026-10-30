"""Memory-free foveated segmentation of one frame: context view + k glimpses (torch).

    frame (1, 3, H, W)
      -> context = backbone_ctx(downsample(frame, context_scale))      (1, C, h, w)
      -> uncertainty map (entropy) -> one score per grid cell -> k cells
      -> crops of the padded frame (k, 3, size/glimpse_scale, ...), resized
         to (k, 3, size, size) when glimpse_scale < 1 -> backbone_glimpse  (one batch)
      -> labels = argmax(upsampled context), then the inner square of each
         glimpse pasted over it: its own argmax (``replace``) or its fusion
         with the context there (``mean``, ``conf``)                    (H, W)

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
import torch.nn.functional as F

from models.utils import upsample_logits

from .ops import (SCORE_KINDS, GlimpseGrid, cell_features, downsample, gather_crops, paste_labels, pool_scores,
                  samples_per_cell, select_greedy, select_topk)

__all__ = ["FoveationConfig", "FoveatedSegmenter", "StubBackbone", "FUSIONS", "context_pass", "glimpse_labels",
           "glimpse_logits", "labels_from_glimpse", "fuse_labels", "gather_inner", "glimpse_grid"]

FUSIONS = ("replace", "mean", "conf")


def context_pass(backbone: nn.Module, frame: torch.Tensor,
                 scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(1, C, h, w) context logits, the (H, W) labels they give at full resolution,
    and the (1, C, H, W) upsampled logits those labels come from."""
    small = downsample(frame, scale)
    logits = backbone(small)
    up = upsample_logits(logits, tuple(frame.shape[-2:]), backbone.output_stride, backbone.align_corners,
                         input_size=tuple(small.shape[-2:]))
    return logits, up.argmax(1)[0], up


def fuse_labels(glimpse: torch.Tensor, context: torch.Tensor, fusion: str) -> torch.Tensor:
    """Labels from (k, C, n, n) glimpse and context logits over the same pixels -> (k, n, n).

    ``replace``: the glimpse's argmax; ``mean``: argmax of the mean of the two
    softmax probabilities; ``conf``: per pixel, the argmax of whichever side has
    the higher top probability (the glimpse on ties).
    """
    if fusion == "replace":
        return glimpse.argmax(1)
    if fusion not in FUSIONS:
        raise ValueError(f"unknown fusion {fusion!r}, expected one of {FUSIONS}")
    pg, pc = torch.softmax(glimpse.float(), 1), torch.softmax(context.float(), 1)
    if fusion == "mean":
        return (pg + pc).argmax(1)
    (mg, ag), (mc, ac) = pg.max(1), pc.max(1)
    return torch.where(mg >= mc, ag, ac)


def gather_inner(up: torch.Tensor, grid: GlimpseGrid, idx: torch.Tensor) -> torch.Tensor:
    """Inner squares of the selected cells from (1, C, H, W) frame-sized logits -> (k, C, n, n).

    Pixels of a border cell that fall outside the frame repeat the border; they
    are cropped away with the canvas padding, so their value never matters.
    """
    h, w = grid.frame_size
    top, left = grid.cell_origin(idx)
    ar = torch.arange(grid.inner, device=up.device)
    rows = (top[:, None] + ar).clamp(max=h - 1)[:, :, None]
    cols = (left[:, None] + ar).clamp(max=w - 1)[:, None, :]
    return up[0][:, rows, cols].permute(1, 0, 2, 3)


def glimpse_logits(backbone: nn.Module, crops: torch.Tensor, input_size: Optional[int] = None) -> torch.Tensor:
    """(k, 3, S, S) crops of the frame -> (k, C, S, S) logits at frame resolution.

    With ``input_size`` < S the crops are resized (antialiased bilinear) to the
    network's input and its logits upsampled back to S.
    """
    size = tuple(crops.shape[-2:])
    x = crops
    if input_size is not None and input_size != size[0]:
        x = F.interpolate(crops, size=(input_size, input_size), mode="bilinear", align_corners=False, antialias=True)
    return upsample_logits(backbone(x), size, backbone.output_stride, backbone.align_corners,
                           input_size=tuple(x.shape[-2:]))


def glimpse_labels(backbone: nn.Module, crops: torch.Tensor, input_size: Optional[int] = None,
                   ctx_inner: Optional[torch.Tensor] = None, margin: int = 0, fusion: str = "replace") -> torch.Tensor:
    """(k, 3, S, S) crops of the frame -> (k, S, S) labels (:func:`glimpse_logits`, :func:`labels_from_glimpse`)."""
    return labels_from_glimpse(glimpse_logits(backbone, crops, input_size), ctx_inner, margin, fusion)


def labels_from_glimpse(up: torch.Tensor, ctx_inner: Optional[torch.Tensor] = None, margin: int = 0,
                        fusion: str = "replace") -> torch.Tensor:
    """(k, C, S, S) glimpse logits -> (k, S, S) labels. With a fusion other than
    ``replace``, the inner square (offset ``margin``) fuses with ``ctx_inner``,
    the (k, C, n, n) context logits there."""
    labels = up.argmax(1)
    if fusion != "replace":
        n = ctx_inner.shape[-1]
        labels[:, margin:margin + n, margin:margin + n] = fuse_labels(
            up[:, :, margin:margin + n, margin:margin + n], ctx_inner, fusion)
    return labels


def glimpse_grid(frame_size: Tuple[int, int], cfg: "FoveationConfig") -> GlimpseGrid:
    """The candidate grid in frame pixels: crops of ``glimpse_size / glimpse_scale`` px."""
    size, margin = cfg.glimpse_size / cfg.glimpse_scale, cfg.glimpse_margin / cfg.glimpse_scale
    if abs(size - round(size)) > 1e-6 or abs(margin - round(margin)) > 1e-6:
        raise ValueError(f"glimpse size {cfg.glimpse_size} and margin {cfg.glimpse_margin} must be whole frame "
                         f"pixels at glimpse_scale {cfg.glimpse_scale}")
    return GlimpseGrid(frame_size, round(size), round(margin))


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
    fusion: str = "replace"  # replace | mean | conf: how a glimpse combines with the context in its cell
    glimpse_scale: float = 1.0  # network input px per frame px: < 1 = wide glimpses at reduced resolution


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
        if cfg.fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {cfg.fusion!r}, expected one of {FUSIONS}")
        self.ctx, self.glimpse, self.cfg = backbone_ctx, backbone_glimpse, cfg
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.grid = glimpse_grid(self.frame_size, cfg)
        if cfg.k > self.grid.num_cells:
            raise ValueError(f"k={cfg.k} exceeds the {self.grid.num_cells} grid cells")
        weights = None if cfg.class_weights is None else torch.tensor(cfg.class_weights, dtype=torch.float32)
        self.register_buffer("_class_weights", weights, persistent=False)  # moves with .cuda(): no copy in forward

    def forward(self, frame: torch.Tensor,
                return_idx: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cfg, grid = self.cfg, self.grid
        h, w = self.frame_size
        ctx_logits, labels, ctx_up = context_pass(self.ctx, frame, cfg.context_scale)
        sums, top10 = cell_features(ctx_logits, grid, [cfg.score])[cfg.score]
        scores = pool_scores(sums, top10, samples_per_cell(ctx_logits.shape[-2], grid), cfg.pool, self._class_weights)
        idx = select_topk(scores, cfg.k) if cfg.selection == "topk" else select_greedy(scores, grid, cfg.k, cfg.suppress)
        ctx_inner = None if cfg.fusion == "replace" else gather_inner(ctx_up, grid, idx)
        crop_labels = glimpse_labels(self.glimpse, gather_crops(grid.pad_frame(frame), grid, idx), cfg.glimpse_size,
                                     ctx_inner, grid.margin, cfg.fusion)
        canvas = paste_labels(grid.to_canvas(labels), crop_labels, grid, idx)
        return (canvas[:h, :w], idx) if return_idx else canvas[:h, :w]

    def describe(self) -> Dict[str, Any]:
        g = self.grid
        return {"config": asdict(self.cfg), "frame_size": list(self.frame_size),
                "grid": {"rows": g.rows, "cols": g.cols, "inner": g.inner, "cells": g.num_cells},
                "pixel_fraction": self.cfg.context_scale ** 2 + self.cfg.k * self.cfg.glimpse_size ** 2 / (self.frame_size[0] * self.frame_size[1])}


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
