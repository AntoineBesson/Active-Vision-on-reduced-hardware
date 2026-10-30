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
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "downsample",
    "entropy_map",
    "SCORE_KINDS",
    "score_map",
    "GlimpseGrid",
    "samples_per_cell",
    "cell_samples",
    "cell_features",
    "pool_scores",
    "cell_scores",
    "select_topk",
    "select_greedy",
    "select_greedy_batch",
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


SCORE_KINDS = ("entropy", "margin", "maxprob")


def score_map(logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-pixel uncertainty of (B, C, h, w) logits -> (B, h, w), in fp32 (higher = less sure).

    ``entropy``: softmax entropy (nats); ``margin``: ``1 - (p1 - p2)``, one minus
    the gap between the two most probable classes; ``maxprob``: ``1 - p1``.
    """
    if kind == "entropy":
        return entropy_map(logits)
    if kind not in SCORE_KINDS:
        raise ValueError(f"unknown score {kind!r}, expected one of {SCORE_KINDS}")
    p = torch.softmax(logits.float(), dim=1)
    if kind == "maxprob":
        return 1.0 - p.amax(dim=1)
    top2 = p.topk(2, dim=1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])


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


def samples_per_cell(map_h: int, grid: GlimpseGrid) -> int:
    """Samples per cell side when pooling a map of height ``map_h`` (about its own resolution)."""
    return max(1, round(map_h * grid.inner / grid.frame_size[0]))


def cell_samples(m: torch.Tensor, grid: GlimpseGrid, q: int, mode: str = "bilinear") -> torch.Tensor:
    """Sample a (h, w) map covering the frame on a ``q x q`` lattice per cell -> (cells, q*q).

    Sample points are the centres of a ``q x q`` subdivision of each cell's inner
    square, in frame coordinates, so every cell gets the same number of samples;
    points in the bottom/right padding take the border value (as the
    replicate-padded crops do). ``mode``: ``bilinear`` for scores, ``nearest``
    for class ids (passed as float).
    """
    h, w = grid.frame_size
    dev = m.device
    ys = (torch.arange(grid.rows * q, device=dev, dtype=torch.float32) + 0.5) * (grid.inner / q)
    xs = (torch.arange(grid.cols * q, device=dev, dtype=torch.float32) + 0.5) * (grid.inner / q)
    gy, gx = torch.meshgrid(2 * ys / h - 1, 2 * xs / w - 1, indexing="ij")
    pts = torch.stack([gx, gy], dim=-1)[None]  # (1, rows*q, cols*q, 2), x first
    out = F.grid_sample(m.reshape(1, 1, *m.shape[-2:]).float(), pts, mode=mode, padding_mode="border",
                        align_corners=False)[0, 0]
    return out.reshape(grid.rows, q, grid.cols, q).permute(0, 2, 1, 3).reshape(grid.num_cells, q * q)


def cell_features(ctx_logits: torch.Tensor, grid: GlimpseGrid,
                  kinds: Sequence[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Per-cell selection features of (1, C, h, w) context logits, for each score ``kind``.

    ``class_sums`` (cells, C): the score summed over a cell's samples, split by the
    context's predicted class at each sample; ``top10`` (cells,): the mean of the
    cell's top 10% samples. Every pooled score (:func:`pool_scores`) derives from these.
    """
    q = samples_per_cell(ctx_logits.shape[-2], grid)
    cls = cell_samples(ctx_logits[0].argmax(0).float(), grid, q, mode="nearest").long()
    k10 = math.ceil(0.1 * q * q)
    out = {}
    for kind in kinds:
        s = cell_samples(score_map(ctx_logits, kind)[0], grid, q)
        sums = torch.zeros(grid.num_cells, ctx_logits.shape[1], device=s.device).scatter_add_(1, cls, s)
        out[kind] = (sums, s.topk(k10, dim=1).values.mean(1))
    return out


def pool_scores(class_sums: torch.Tensor, top10: torch.Tensor, q: int, pool: str,
                class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One score per cell from :func:`cell_features` -> (cells,).

    ``mean``: the cell mean, optionally weighting each sample by the weight of its
    predicted class; ``top10``: the mean of the cell's top 10% samples (no weights).
    """
    if pool == "top10":
        if class_weights is not None:
            raise ValueError("class weights only apply to mean pooling")
        return top10
    if pool != "mean":
        raise ValueError(f"unknown pooling {pool!r}, expected 'mean' or 'top10'")
    # last axis = classes, so (frames, cells, C) features pool in one call too
    total = class_sums.sum(-1) if class_weights is None else class_sums @ class_weights.to(class_sums)
    return total / (q * q)


def cell_scores(score_map: torch.Tensor, grid: GlimpseGrid) -> torch.Tensor:
    """Mean of a (h, w) score map over each grid cell -> (rows * cols,).

    The map may be at any resolution (e.g. the context logits'); it is sampled
    on an equal ``q x q`` lattice per cell (:func:`cell_samples`).
    """
    q = samples_per_cell(score_map.shape[-2], grid)
    return cell_samples(score_map, grid, q).mean(1)


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


def select_greedy_batch(scores: torch.Tensor, grid: GlimpseGrid, k: int, suppress: int = 1) -> torch.Tensor:
    """:func:`select_greedy` on every row of (N, cells) scores -> (N, k)."""
    s = scores.float().clone()
    ids = torch.arange(grid.num_cells, device=s.device)
    rr, cc = ids // grid.cols, ids % grid.cols
    taken = torch.zeros_like(s, dtype=torch.bool)
    picks = []
    neg = torch.finfo(s.dtype).min
    for _ in range(k):
        i = torch.argmax(torch.where(taken, torch.full_like(s, neg), s), dim=1)  # (N,)
        picks.append(i)
        taken |= ids[None] == i[:, None]
        near = (((rr[None] - (i // grid.cols)[:, None]).abs() <= suppress)
                & ((cc[None] - (i % grid.cols)[:, None]).abs() <= suppress))
        s = torch.where(near & ~taken, s - 1e9, s)
    return torch.stack(picks, dim=1)


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
