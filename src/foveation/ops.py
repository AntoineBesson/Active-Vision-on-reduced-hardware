"""Tensor building blocks of a foveated (context + glimpses) segmentation pass (torch).

Every op runs on the device of its inputs, has fixed output shapes for fixed
input shapes and a fixed glimpse count ``k``, and never synchronises with the
host (no ``.item()``, no data-dependent Python control flow). That keeps a
whole pipeline capturable in a CUDA graph and deterministic: ties in the
glimpse selection are broken by position (``stable`` sort / first ``argmax``).

Geometry. A glimpse is a ``size x size`` crop of the full-resolution frame.
Only its central ``inner = size - 2*margin`` square is written back to the
output; the margin gives the backbone context at the crop border. Candidate
glimpses lie on a grid of ``inner``-sized cells covering the frame, so the
inner squares of two different candidates never overlap and pasting is
order-independent. Crops are taken from the frame padded by ``margin``
(replicate) plus the round-up to whole cells, so border cells get a full crop.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "downsample",
    "entropy_map",
    "GlimpseGrid",
    "cell_scores",
    "select_topk",
    "select_greedy",
    "gather_crops",
    "paste_labels",
]


def downsample(frame: torch.Tensor, scale: float) -> torch.Tensor:
    """Low-resolution view: antialiased bilinear resize (the baselines' resize).

    Averages neighbouring pixels instead of dropping them, so thin structures
    (poles, signs) are attenuated rather than broken. ``frame``: (B, 3, H, W).
    """
    if scale == 1.0:
        return frame
    h, w = frame.shape[-2:]
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    return F.interpolate(frame, size=size, mode="bilinear", align_corners=False, antialias=True)


def entropy_map(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel softmax entropy (nats) of (B, C, h, w) logits -> (B, h, w), in fp32."""
    logp = torch.log_softmax(logits.float(), dim=1)
    return -(logp.exp() * logp).sum(dim=1)


class GlimpseGrid:
    """Candidate-glimpse grid of a ``frame_size`` frame (see module docstring)."""

    def __init__(self, frame_size: Tuple[int, int], size: int, margin: int):
        if size <= 2 * margin:
            raise ValueError(f"glimpse size {size} must exceed 2*margin {2 * margin}")
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.size, self.margin = int(size), int(margin)
        self.inner = self.size - 2 * self.margin
        h, w = self.frame_size
        self.rows, self.cols = math.ceil(h / self.inner), math.ceil(w / self.inner)
        # padding: margin on every side, plus the round-up to whole cells at the bottom/right
        self.pad = (self.margin, self.margin + self.cols * self.inner - w,
                    self.margin, self.margin + self.rows * self.inner - h)  # F.pad order: l, r, t, b

    @property
    def num_cells(self) -> int:
        return self.rows * self.cols

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return self.rows * self.inner, self.cols * self.inner

    def pad_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return F.pad(frame, self.pad, mode="replicate")

    def to_canvas(self, labels: torch.Tensor) -> torch.Tensor:
        """(H, W) labels -> (rows*inner, cols*inner), zero-padded at the bottom/right."""
        h, w = self.frame_size
        ch, cw = self.canvas_size
        return F.pad(labels, (0, cw - w, 0, ch - h))

    def cell_origin(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(top, left) of the cells' inner squares in frame pixels; idx: (k,) flat cell ids."""
        return (idx // self.cols) * self.inner, (idx % self.cols) * self.inner


def cell_scores(score_map: torch.Tensor, grid: GlimpseGrid) -> torch.Tensor:
    """Mean of a (h, w) score map over each grid cell -> (rows * cols,).

    The map may be at any resolution (e.g. the context logits'); it is resized
    to whole cells by area averaging first.
    """
    m = score_map.reshape(1, 1, *score_map.shape[-2:]).float()
    h, w = grid.frame_size
    # map to frame pixels' cell layout: frame (h, w) -> padded extent rows*inner x cols*inner
    target_h = max(grid.rows, round(m.shape[-2] * grid.rows * grid.inner / h))
    target_w = max(grid.cols, round(m.shape[-1] * grid.cols * grid.inner / w))
    m = F.interpolate(m, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(m, (grid.rows, grid.cols)).flatten()


def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """The ``k`` best cells (flat ids), best first; ties -> lower id first."""
    order = torch.sort(scores, descending=True, stable=True).indices
    return order[:k]


def select_greedy(scores: torch.Tensor, grid: GlimpseGrid, k: int, suppress: int = 1) -> torch.Tensor:
    """Greedy top-``k`` with suppression: after picking a cell, its neighbours
    within ``suppress`` cells (Chebyshev distance) are excluded, so glimpses
    spread out. Falls back to the best remaining cell when everything is
    suppressed. ``k`` sequential argmax steps, all on device."""
    s = scores.float().clone()
    ids = torch.arange(grid.num_cells, device=s.device)
    rr, cc = ids // grid.cols, ids % grid.cols
    taken = torch.zeros_like(s, dtype=torch.bool)
    picks = []
    neg = torch.finfo(s.dtype).min
    for _ in range(k):
        # suppressed-but-free cells keep a chance below every unsuppressed one
        i = torch.argmax(torch.where(taken, torch.full_like(s, neg), s))
        picks.append(i)
        taken |= ids == i
        # i // cols, i % cols rather than rr[i]: indexing with a device scalar syncs
        near = ((rr - i // grid.cols).abs() <= suppress) & ((cc - i % grid.cols).abs() <= suppress)
        s = torch.where(near & ~taken, s - 1e9, s)
    return torch.stack(picks)


def gather_crops(padded: torch.Tensor, grid: GlimpseGrid, idx: torch.Tensor) -> torch.Tensor:
    """Crops of ``grid.size`` around the selected cells: (1, C, Hp, Wp) -> (k, C, size, size)."""
    top, left = grid.cell_origin(idx)  # inner origin in frame px == crop origin in padded px
    ar = torch.arange(grid.size, device=padded.device)
    rows = (top[:, None] + ar)[:, :, None]  # (k, size, 1)
    cols = (left[:, None] + ar)[:, None, :]  # (k, 1, size)
    return padded[0][:, rows, cols].permute(1, 0, 2, 3)


def paste_labels(canvas: torch.Tensor, crop_labels: torch.Tensor, grid: GlimpseGrid, idx: torch.Tensor) -> torch.Tensor:
    """Write the inner squares of (k, size, size) crop labels into a label canvas, in place.

    ``canvas`` is (``grid.canvas_size``): the frame's labels padded at the
    bottom/right to whole cells (:meth:`GlimpseGrid.to_canvas`); crop it back
    with ``[:H, :W]``. Inner squares of distinct cells never overlap, so the
    result does not depend on the order of ``idx`` (which must not repeat a cell).
    """
    m, n = grid.margin, grid.inner
    top, left = grid.cell_origin(idx)
    ar = torch.arange(n, device=canvas.device)
    rows = (top[:, None] + ar)[:, :, None].expand(-1, n, n)
    cols = (left[:, None] + ar)[:, None, :].expand(-1, n, n)
    canvas.index_put_((rows, cols), crop_labels[:, m:m + n, m:m + n].to(canvas.dtype))
    return canvas
