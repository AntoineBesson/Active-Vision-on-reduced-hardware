"""Method 4, phase 1: a persistent canvas of past glimpses under a per-frame oracle selector.

Spec: docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md. Everything is
replayed from the Method 3 per-clip caches (:mod:`reuse.replay`); no network pass.

Per frame t (frame 0 included; there is no full pass):

1. the context prediction at ``ctx_scale`` (labels + confidence);
2. the canvas (labels, confidence, age of past glimpses) is aged by one frame and, in
   variant ``iclk``, warped by the cache's homography; pixels without a source become empty;
3. base = the more confident of context and canvas per pixel (:func:`reuse.method.fuse`,
   the context on ties); empty canvas pixels and those older than ``a_max`` count as 0;
4. the oracle picks cells: a cell's gain is the number of its pixels where the glimpse
   agrees with the scale-1 full pass, minus the number where the base does. Fixed ``k``:
   the k cells of highest gain. Adaptive: every cell whose gain is above ``gamma`` x cell pixels;
5. the picked cells' glimpses are pasted into the base and written into the canvas (age 0).

Variant ``off`` has no canvas (Method 2 with the oracle selector).

:func:`replay_canvas` is the reference, one configuration at a time, written with Method 3's
helpers. :func:`replay_canvas_many` runs configurations that share glimpse size and variant in
one batch per frame and must equal it exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from reuse.method import fuse, paste_cells
from reuse.motion import MOTION_FACTOR, warp
from reuse.replay import IGNORE, NUM_CLASSES, TEACHER_SCALE, ClipCache, _blocks, _confusion_t, _key

__all__ = ["CanvasConfig", "CanvasResult", "replay_canvas", "replay_canvas_many", "VARIANTS", "EMPTY_AGE"]

VARIANTS = {"off": "N", "iclk": "A", "none": "B"}  # config value -> the spec's letter
EMPTY_AGE = 255  # ages saturate here; a never-written pixel also has confidence 0


@dataclass(frozen=True)
class CanvasConfig:
    ctx_scale: float
    glimpse_size: int
    variant: str  # "off": no canvas (N); "iclk": canvas warped by the homography (A); "none": not warped (B)
    a_max: Optional[int] = None  # canvas pixels older than this many frames are ignored; None: no limit
    k: Optional[int] = None  # fixed number of glimpses per frame, or
    gamma: Optional[float] = None  # adaptive: every cell whose gain exceeds gamma x (cell pixels)
    margin: int = 32

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")
        if (self.k is None) == (self.gamma is None):
            raise ValueError("exactly one of k (fixed) and gamma (adaptive) must be set")
        if self.variant == "off" and self.a_max is not None:
            raise ValueError("variant 'off' has no canvas: a_max must be None")
        if self.a_max is not None and not 1 <= self.a_max < EMPTY_AGE:
            raise ValueError(f"a_max must be in 1..{EMPTY_AGE - 1} or None, got {self.a_max}")

    def name(self) -> str:
        age = "" if self.variant == "off" else f"/a{'inf' if self.a_max is None else self.a_max}"
        sel = f"k{self.k}" if self.k is not None else f"y{self.gamma:g}"
        return f"4o-{VARIANTS[self.variant]}/c{self.ctx_scale:g}/g{self.glimpse_size}{age}/{sel}"

    def gain_threshold(self, cell_pixels: int) -> int:
        """Adaptive selector: a cell is glimpsed when its gain (pixels) is above this integer."""
        return math.floor(self.gamma * cell_pixels)


@dataclass
class CanvasResult:
    conf: np.ndarray  # (labelled frames, 19, 19) int64
    k: np.ndarray  # (T,) int32: glimpses per frame
    canvas_share: np.ndarray  # (T,) float64: share of the base prediction taken from the canvas
    num_cells: int


def _check(cache: ClipCache, cfgs: Sequence[CanvasConfig], homography: str) -> None:
    for cfg in cfgs:
        if cfg.margin != cache.meta["margin"]:
            raise ValueError(f"config margin {cfg.margin} != cache margin {cache.meta['margin']}")
        if cfg.variant == "iclk" and cache.meta.get("homography") != homography:
            raise ValueError(f"variant A expects a cache built with homography {homography!r}; "
                             f"{cache.path.name} was built with {cache.meta.get('homography')!r}")
        if cfg.k is not None and not 0 <= cfg.k <= cache.grid(cfg.glimpse_size).num_cells:
            raise ValueError(f"k = {cfg.k} is outside 0..{cache.grid(cfg.glimpse_size).num_cells}")


def _aged(age: torch.Tensor) -> torch.Tensor:
    return (age.to(torch.int16) + 1).clamp(max=EMPTY_AGE).to(torch.uint8)


def replay_canvas(cache: ClipCache, cfg: CanvasConfig, device, return_predictions: bool = False,
                  homography: str = "iclk"):
    """Replay ``cfg`` over the clip (the reference). Returns a :class:`CanvasResult` (and the
    per-frame (H, W) long predictions with ``return_predictions``). ``homography``: the
    estimator variant A's cache must have been built with (the T4 cost counted is IC-LK's)."""
    _check(cache, [cfg], homography)
    dev = torch.device(device)
    grid = cache.grid(cfg.glimpse_size)
    n, s = grid.inner, grid.inner // MOTION_FACTOR
    h, w = cache.meta["frame_size"]
    t_of = lambda key, t: cache.tensor(key, dev)[t]  # noqa: E731
    all_idx = torch.arange(grid.num_cells, device=dev)
    labels = torch.zeros((h, w), dtype=torch.long, device=dev)
    conf = torch.zeros((h // MOTION_FACTOR, w // MOTION_FACTOR), dtype=torch.uint8, device=dev)
    age = torch.full_like(conf, EMPTY_AGE)
    row = {int(t): j for j, t in enumerate(cache.labeled_t)}
    confs: List[torch.Tensor] = []
    ks: List[int] = []
    shares: List[float] = []
    preds = [] if return_predictions else None
    for t in range(cache.num_frames):
        base = t_of(_key("ctx", cfg.ctx_scale), t).long()
        share = 0.0
        if cfg.variant != "off":
            if t > 0:
                age = _aged(age)
                if cfg.variant == "iclk":
                    H = torch.from_numpy(cache["H"][t])
                    labels = warp(labels[None, None].float(), H, "nearest")[0][0, 0].round().long()
                    warped, valid = warp(conf[None, None].float(), H, "nearest", frame_size=(h, w))
                    conf = torch.where(valid, warped[0, 0].round().to(torch.uint8), torch.zeros_like(conf))
                    age = warp(age[None, None].float(), H, "nearest", frame_size=(h, w))[0][0, 0].round().to(torch.uint8)
            usable = conf if cfg.a_max is None else torch.where(age <= cfg.a_max, conf, torch.zeros_like(conf))
            ctx_conf = t_of(_key("ctx_conf", cfg.ctx_scale), t)
            share = float((usable > ctx_conf).float().mean())
            base, _ = fuse(labels, usable, base, ctx_conf)
        cells = t_of(f"cells_{cfg.glimpse_size}", t).long()
        teacher = t_of(_key("full", TEACHER_SCALE), t).long()
        t_blocks = _blocks(teacher, n, grid.cols, all_idx)
        gain = ((_blocks(cells, n, grid.cols, all_idx) == t_blocks).flatten(1).sum(1)
                - (_blocks(base, n, grid.cols, all_idx) == t_blocks).flatten(1).sum(1))
        if cfg.k is not None:
            idx = torch.sort(torch.sort(-gain, stable=True).indices[:cfg.k]).values
        else:
            idx = torch.nonzero(gain > cfg.gain_threshold(n * n)).flatten()
        if idx.numel():
            blocks = _blocks(cells, n, grid.cols, idx)
            base = paste_cells(base, blocks, n, grid.cols, idx)
            if cfg.variant != "off":
                labels = paste_cells(labels, blocks, n, grid.cols, idx)
                conf = paste_cells(conf, _blocks(t_of(f"cells_conf_{cfg.glimpse_size}", t), s, grid.cols, idx), s,
                                   grid.cols, idx)
                age = paste_cells(age, torch.zeros((idx.numel(), s, s), dtype=torch.uint8, device=dev), s,
                                  grid.cols, idx)
        ks.append(int(idx.numel()))
        shares.append(share)
        if t in row:
            confs.append(_confusion_t(cache.tensor("labels", dev)[row[t]], base))
        if preds is not None:
            preds.append(base.clone())
    result = CanvasResult(conf=torch.stack(confs).cpu().numpy().astype(np.int64), k=np.array(ks, np.int32),
                          canvas_share=np.array(shares, np.float64), num_cells=grid.num_cells)
    return (result, preds) if return_predictions else result


def _cell_mask(mask: torch.Tensor, rows: int, cols: int, cell: int, size) -> torch.Tensor:
    """(B, cells) bool -> (B, h, w) bool, True over the pixels of the True cells."""
    m = mask.view(-1, rows, cols).repeat_interleave(cell, 1).repeat_interleave(cell, 2)
    return m[:, :size[0], :size[1]]


def _cell_counts(eq: torch.Tensor, rows: int, cols: int, cell: int) -> torch.Tensor:
    """(..., h, w) bool -> (..., cells) int32: True pixels per cell (the frame may end inside the last cells)."""
    h, w = eq.shape[-2:]
    x = torch.nn.functional.pad(eq.to(torch.uint8), (0, cols * cell - w, 0, rows * cell - h))
    return x.view(*eq.shape[:-2], rows, cell, cols, cell).sum(dim=(-3, -1), dtype=torch.int32).flatten(-2)


def replay_canvas_many(cache: ClipCache, cfgs: Sequence[CanvasConfig], device, return_predictions: bool = False,
                       homography: str = "iclk"):
    """:func:`replay_canvas` for several configs in lockstep, one batch per frame.

    The configs must share glimpse size, variant and margin (so the whole batch is warped, or
    not, by the same H). Warps, fusion, the per-cell counts and the pastes are elementwise, so
    each result equals :func:`replay_canvas`'s. Returns one :class:`CanvasResult` per config
    (and the per-frame predictions, per config, with ``return_predictions``)."""
    if len({(c.glimpse_size, c.variant, c.margin) for c in cfgs}) != 1:
        raise ValueError("replay_canvas_many: the configs must share glimpse size, variant and margin")
    _check(cache, cfgs, homography)
    dev = torch.device(device)
    g0, B = cfgs[0], len(cfgs)
    grid = cache.grid(g0.glimpse_size)
    n, s = grid.inner, grid.inner // MOTION_FACTOR
    h, w = cache.meta["frame_size"]
    small = (h // MOTION_FACTOR, w // MOTION_FACTOR)
    canvas = g0.variant != "off"
    t_of = lambda key, t: cache.tensor(key, dev)[t]  # noqa: E731
    scales = sorted({c.ctx_scale for c in cfgs})
    scale_of = torch.tensor([scales.index(c.ctx_scale) for c in cfgs], device=dev)
    a_max = torch.tensor([EMPTY_AGE if c.a_max is None else c.a_max for c in cfgs], device=dev)[:, None, None]
    fixed = torch.tensor([c.k is not None for c in cfgs], device=dev)
    k_of = torch.tensor([c.k if c.k is not None else 0 for c in cfgs], device=dev)
    thr = torch.tensor([c.gain_threshold(n * n) if c.gamma is not None else 0 for c in cfgs], device=dev)
    rank = torch.arange(grid.num_cells, device=dev)[None]
    labels = torch.zeros((B, h, w), dtype=torch.uint8, device=dev)
    conf = torch.zeros((B, *small), dtype=torch.uint8, device=dev)
    age = torch.full_like(conf, EMPTY_AGE)
    row = {int(t): j for j, t in enumerate(cache.labeled_t)}
    confs: List[torch.Tensor] = []
    ks: List[torch.Tensor] = []
    shares: List[torch.Tensor] = []
    preds: Optional[List[List[torch.Tensor]]] = [[] for _ in cfgs] if return_predictions else None
    offset = (torch.arange(B, device=dev) * NUM_CLASSES * NUM_CLASSES)[:, None]
    for t in range(cache.num_frames):
        base = torch.stack([t_of(_key("ctx", sc), t) for sc in scales])[scale_of]  # (B, h, w) uint8
        share = torch.zeros(B, device=dev)
        if canvas:
            if t > 0:
                age = _aged(age)
                if g0.variant == "iclk":
                    H = torch.from_numpy(cache["H"][t])
                    labels = warp(labels[:, None].float(), H, "nearest")[0][:, 0].round().to(torch.uint8)
                    warped, valid = warp(conf[:, None].float(), H, "nearest", frame_size=(h, w))
                    conf = torch.where(valid, warped[:, 0].round().to(torch.uint8), torch.zeros_like(conf))
                    age = warp(age[:, None].float(), H, "nearest", frame_size=(h, w))[0][:, 0].round().to(torch.uint8)
            usable = torch.where(age <= a_max, conf, torch.zeros_like(conf))
            ctx_conf = torch.stack([t_of(_key("ctx_conf", sc), t) for sc in scales])[scale_of]
            share = (usable > ctx_conf).float().mean((1, 2))
            base, _ = fuse(labels, usable, base, ctx_conf)
        cells = t_of(f"cells_{g0.glimpse_size}", t)
        teacher = t_of(_key("full", TEACHER_SCALE), t)
        gain = (_cell_counts(cells == teacher, grid.rows, grid.cols, n)[None]
                - _cell_counts(base == teacher[None], grid.rows, grid.cols, n))  # (B, cells) int32
        order = torch.sort(-gain, dim=1, stable=True).indices
        top = torch.zeros_like(gain, dtype=torch.bool).scatter_(1, order, (rank < k_of[:, None]).expand(B, -1))
        picked = torch.where(fixed[:, None], top, gain > thr[:, None])
        full_mask = _cell_mask(picked, grid.rows, grid.cols, n, (h, w))
        base = torch.where(full_mask, cells[None], base)
        if canvas:
            small_mask = _cell_mask(picked, grid.rows, grid.cols, s, small)
            labels = torch.where(full_mask, cells[None], labels)
            conf = torch.where(small_mask, t_of(f"cells_conf_{g0.glimpse_size}", t)[None], conf)
            age = torch.where(small_mask, torch.zeros_like(age), age)
        ks.append(picked.sum(1))
        shares.append(share)
        if t in row:
            label = cache.tensor("labels", dev)[row[t]]
            ok = label != IGNORE
            code = label[ok].long()[None] * NUM_CLASSES + base[:, ok].long() + offset
            confs.append(torch.bincount(code.flatten(), minlength=B * NUM_CLASSES * NUM_CLASSES)
                         .reshape(B, NUM_CLASSES, NUM_CLASSES))
        if preds is not None:
            for b in range(B):
                preds[b].append(base[b].long().clone())
    conf_all = torch.stack(confs, 1).cpu().numpy().astype(np.int64)  # (B, labelled frames, 19, 19)
    k_all = torch.stack(ks, 1).cpu().numpy().astype(np.int32)
    share_all = torch.stack(shares, 1).cpu().numpy().astype(np.float64)
    results = [CanvasResult(conf=conf_all[b], k=k_all[b], canvas_share=share_all[b], num_cells=grid.num_cells)
               for b in range(B)]
    return (results, preds) if return_predictions else results
