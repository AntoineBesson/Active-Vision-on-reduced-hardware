"""Exact replay of Method 3 from per-frame caches, and its bounds.

A frame's own passes do not depend on the clip's history: the full pass at each
scale, every cell glimpsed, the 3b context pass, the homography and the motion
map. :func:`build_clip_cache` computes them once per frame (on the GPU) and
stores them per clip; :func:`replay_clip` then replays any configuration's
memory, warps, fallbacks and fusion over the clip with the same functions as
:class:`reuse.method.MotionReuseMethod`, so it gives the same predictions pixel
for pixel, and scores the labelled frames.

Cache arrays (T frames; labels at full resolution, confidence and motion at 1/f):
``full_{s}`` / ``full_conf_{s}``, ``cells_{g}`` / ``cells_conf_{g}`` (every cell of
the g-px grid glimpsed, inner squares), ``ctx_{s}`` / ``ctx_conf_{s}``, ``H``
(T, 3, 3, ``H[0] = I``), ``inliers``, ``motion`` (aligned) and ``motion_raw``
(H = I, the no-homography ablation), ``labels`` and ``labeled_t``, ``meta``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.harness import PRECISIONS, frame_to_tensor
from foveation.ops import GlimpseGrid

from .method import (FRAME, MotionReuseConfig, cell_index, full_pass, fuse, glimpse_pass, grid_canvas, paste_cells,
                     warp_memory)
from .motion import IDENTITY, MOTION_FACTOR, estimate_homography, motion_map, moving_cells, to_gray_small, warp

__all__ = ["build_clip_cache", "ClipCache", "ClipResult", "replay_clip", "replay_many", "reuse_bound", "cache_file"]

NUM_CLASSES, IGNORE = 19, 255
TEACHER_SCALE = 1.0  # the zone oracle's reference: the scale-1 full-frame prediction


def _key(prefix: str, value: float) -> str:
    return f"{prefix}_{value:g}"


def cache_file(root: Path, clip_id: str) -> Path:
    return Path(root) / f"{clip_id}.npz"


def tmp_cache_file(path: Path) -> Path:
    """Where a cache is written before the rename (ends in ``.npz``: numpy would add it otherwise)."""
    return path.with_suffix(".tmp.npz")


def build_clip_cache(dataset, index: int, backbone: torch.nn.Module, out: Path, full_scales: Sequence[float],
                     glimpse_sizes: Sequence[int], ctx_scales: Sequence[float], device, precision: str = "fp32",
                     homography: str = "orb", margin: int = 32, allow_any_frame_size: bool = False,
                     extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``<out>/<clip_id>.npz`` (via a ``.tmp.npz`` renamed when complete); returns its path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clip_id = dataset.clip_id(index)
    path = cache_file(out, clip_id)
    dtype = PRECISIONS[precision]
    full_scales = sorted(set(full_scales) | {TEACHER_SCALE}, reverse=True)
    T = dataset.num_frames(index)
    labeled_t = np.flatnonzero(dataset.labeled_mask(index)).astype(np.int32)
    arrays: Dict[str, List[np.ndarray]] = defaultdict(list)
    hs: List[np.ndarray] = []
    inliers: List[int] = []
    labels: List[np.ndarray] = []
    prev_gray = None
    backbone = backbone.eval()
    with torch.no_grad():
        for t in range(T):
            frame = frame_to_tensor(dataset.load_frame(index, t), device)
            h, w = frame.shape[-2:]
            if not allow_any_frame_size and (h, w) != FRAME:
                raise ValueError(f"frame is {h}x{w}; the T4 latency tables are for {FRAME[0]}x{FRAME[1]}")
            small = (h // MOTION_FACTOR, w // MOTION_FACTOR)
            for s in full_scales:
                lab, conf, _ = full_pass(backbone, frame, s, small, dtype)
                arrays[_key("full", s)].append(lab.to(torch.uint8).cpu().numpy())
                arrays[_key("full_conf", s)].append(conf.cpu().numpy())
            for s in ctx_scales:
                lab, conf, _ = full_pass(backbone, frame, s, small, dtype)
                arrays[_key("ctx", s)].append(lab.to(torch.uint8).cpu().numpy())
                arrays[_key("ctx_conf", s)].append(conf.cpu().numpy())
            for g in glimpse_sizes:
                grid = GlimpseGrid((h, w), g, margin)
                idx = torch.arange(grid.num_cells, device=frame.device)
                g_lab, g_conf, _ = glimpse_pass(backbone, frame, grid, idx, dtype)
                lab = paste_cells(torch.zeros(h, w, dtype=torch.uint8, device=frame.device), g_lab, grid.inner,
                                  grid.cols, idx)
                conf = paste_cells(torch.zeros(small, dtype=torch.uint8, device=frame.device), g_conf,
                                   grid.inner // MOTION_FACTOR, grid.cols, idx)
                arrays[f"cells_{g}"].append(lab.cpu().numpy())
                arrays[f"cells_conf_{g}"].append(conf.cpu().numpy())
            gray = to_gray_small(frame)
            if prev_gray is None:
                H, n = IDENTITY, 0
                zero = np.zeros(small, np.uint8)
                arrays["motion"].append(zero)
                arrays["motion_raw"].append(zero)
            else:
                H, n = estimate_homography(prev_gray, gray, method=homography)
                arrays["motion"].append(motion_map(prev_gray, gray, H).cpu().numpy())
                arrays["motion_raw"].append(motion_map(prev_gray, gray, IDENTITY).cpu().numpy())
            hs.append(H.numpy())
            inliers.append(int(n))
            prev_gray = gray
            if t in labeled_t:
                labels.append(np.asarray(dataset.load_label(index, t), dtype=np.uint8))
    meta = {"clip_id": clip_id, "full_scales": full_scales, "glimpse_sizes": list(glimpse_sizes),
            "ctx_scales": list(ctx_scales), "margin": margin, "frame_size": [int(h), int(w)], "precision": precision,
            "homography": homography, **(extra_meta or {})}
    data = {k: np.stack(v) for k, v in arrays.items()}
    data.update(H=np.stack(hs), inliers=np.array(inliers, np.int32), labels=np.stack(labels),
                labeled_t=labeled_t, meta=json.dumps(meta))
    tmp = tmp_cache_file(path)
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    return path


class ClipCache:
    """One clip's cache; arrays are read on first use and kept."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._npz = np.load(self.path)
        self._arrays: Dict[str, np.ndarray] = {}
        self._tensors: Dict[Tuple[str, str], torch.Tensor] = {}
        self.meta: Dict[str, Any] = json.loads(str(self._npz["meta"]))
        self.labeled_t: np.ndarray = self._npz["labeled_t"]
        self.num_frames = int(self._npz["H"].shape[0])

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = self._npz[key]
        return self._arrays[key]

    def tensor(self, key: str, device) -> torch.Tensor:
        """The whole array on ``device``, uploaded once (a replay then slices it without host copies)."""
        k = (key, str(torch.device(device)))
        if k not in self._tensors:
            self._tensors[k] = torch.from_numpy(self[key]).to(device)
        return self._tensors[k]

    def grid(self, size: int) -> GlimpseGrid:
        return GlimpseGrid(tuple(self.meta["frame_size"]), size, self.meta["margin"])


@dataclass
class ClipResult:
    conf: np.ndarray  # (labelled frames, 19, 19) int64
    moved_cells: np.ndarray  # (T,) int32, frame 0 and fallbacks count every cell
    fallback: np.ndarray  # (T,) bool
    num_cells: int


def _confusion(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ok = label != IGNORE
    return np.bincount(label[ok].astype(np.int64) * NUM_CLASSES + pred[ok].astype(np.int64),
                       minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def _confusion_t(label: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """:func:`_confusion` on the device (the replay's predictions never leave it)."""
    ok = label != IGNORE
    return torch.bincount(label[ok].long() * NUM_CLASSES + pred[ok].long(),
                          minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def _blocks(image: torch.Tensor, cell: int, cols: int, idx: torch.Tensor) -> torch.Tensor:
    """(k, cell, cell) blocks of the cells ``idx`` of an (h, w) image (zero beyond the frame)."""
    return grid_canvas(image, cell, cols)[cell_index(cell, cols, idx)]


def replay_clip(cache: ClipCache, cfg: MotionReuseConfig, device, motion_key: str = "motion",
                selector: str = "motion", oracle_k: Optional[Sequence[int]] = None,
                return_predictions: bool = False):
    """Replay ``cfg`` over the clip. ``selector="oracle"`` recomputes, at frame t, the
    ``oracle_k[t]`` cells whose glimpse gains the most pixels of agreement with the scale-1
    full pass over the (warped, fused) memory. Returns a :class:`ClipResult` (and the
    per-frame predictions with ``return_predictions``)."""
    if cfg.margin != cache.meta["margin"]:
        raise ValueError(f"config margin {cfg.margin} != cache margin {cache.meta['margin']}")
    dev = torch.device(device)
    grid = cache.grid(cfg.glimpse_size)
    n, s = grid.inner, grid.inner // MOTION_FACTOR
    t_of = lambda key, t: cache.tensor(key, dev)[t]  # noqa: E731
    full, full_conf = _key("full", cfg.full_scale), _key("full_conf", cfg.full_scale)
    labels, conf = t_of(full, 0).long(), t_of(full_conf, 0)
    row = {int(t): j for j, t in enumerate(cache.labeled_t)}
    confs: List[torch.Tensor] = []
    preds = [labels.clone()] if return_predictions else None

    def score(t: int, pred: torch.Tensor) -> None:
        if t in row:
            confs.append(_confusion_t(cache.tensor("labels", dev)[row[t]], pred))

    score(0, labels)
    moved, fallback = [grid.num_cells], [False]
    for t in range(1, cache.num_frames):
        H = torch.from_numpy(cache["H"][t])
        if selector == "motion":
            moving = moving_cells(t_of(motion_key, t), grid, cfg.theta, cfg.rho)
            k = int(moving.sum())
        elif selector == "oracle":
            k = int(oracle_k[t])
        else:
            raise ValueError(f"unknown selector {selector!r}")
        if k / grid.num_cells > cfg.tau:
            labels, conf = t_of(full, t).long(), t_of(full_conf, t)
            moved.append(grid.num_cells)
            fallback.append(True)
        else:
            ctx = None
            if cfg.ctx_scale is not None:
                ctx = (t_of(_key("ctx", cfg.ctx_scale), t).long(), t_of(_key("ctx_conf", cfg.ctx_scale), t))
            cells, cells_conf = t_of(f"cells_{cfg.glimpse_size}", t).long(), t_of(f"cells_conf_{cfg.glimpse_size}", t)
            base, base_conf = warp_memory(labels, conf, H, ctx)
            if selector == "oracle":
                teacher = t_of(_key("full", TEACHER_SCALE), t).long()
                all_idx = torch.arange(grid.num_cells, device=dev)
                gain = ((_blocks(cells, n, grid.cols, all_idx) == _blocks(teacher, n, grid.cols, all_idx)).flatten(1).sum(1)
                        - (_blocks(base, n, grid.cols, all_idx) == _blocks(teacher, n, grid.cols, all_idx)).flatten(1).sum(1))
                order = torch.sort(-gain, stable=True).indices[:k]
                moving = torch.zeros(grid.num_cells, dtype=torch.bool, device=dev)
                moving[order] = True
            if bool(moving.any()):
                idx = torch.nonzero(moving).flatten().to(dev)
                base = paste_cells(base, _blocks(cells, n, grid.cols, idx), n, grid.cols, idx)
                base_conf = paste_cells(base_conf, _blocks(cells_conf, s, grid.cols, idx), s, grid.cols, idx)
            labels, conf = base, base_conf
            moved.append(k)
            fallback.append(False)
        score(t, labels)
        if preds is not None:
            preds.append(labels.clone())
    result = ClipResult(conf=torch.stack(confs).cpu().numpy().astype(np.int64), moved_cells=np.array(moved, np.int32),
                        fallback=np.array(fallback, bool), num_cells=grid.num_cells)
    return (result, preds) if return_predictions else result


def replay_many(cache: ClipCache, cfgs: Sequence[MotionReuseConfig], device, return_predictions: bool = False):
    """:func:`replay_clip` (motion selector) for several configs in lockstep, one batch per frame.

    The configs must share glimpse size, theta, rho and margin: they then see the same moving
    cells and the same H at every frame, so the warps and the glimpse pastes run once for the
    batch. Nearest warps, pastes and fusion are elementwise, so each result equals
    :func:`replay_clip`'s exactly. Returns one :class:`ClipResult` per config (and the per-frame
    predictions, per config, with ``return_predictions``)."""
    if len({(c.glimpse_size, c.theta, c.rho, c.margin) for c in cfgs}) != 1:
        raise ValueError("replay_many: the configs must share glimpse size, theta, rho and margin")
    if cfgs[0].margin != cache.meta["margin"]:
        raise ValueError(f"config margin {cfgs[0].margin} != cache margin {cache.meta['margin']}")
    dev = torch.device(device)
    g0 = cfgs[0]
    grid = cache.grid(g0.glimpse_size)
    n, s = grid.inner, grid.inner // MOTION_FACTOR
    B = len(cfgs)
    t_of = lambda key, t: cache.tensor(key, dev)[t]  # noqa: E731
    by_ctx: Dict[float, List[int]] = {}
    for b, c in enumerate(cfgs):
        if c.ctx_scale is not None:
            by_ctx.setdefault(c.ctx_scale, []).append(b)
    by_ctx_t = {k: torch.tensor(v, device=dev) for k, v in by_ctx.items()}
    labels = torch.empty((B, *cache.meta["frame_size"]), dtype=torch.uint8, device=dev)
    conf = torch.empty((B, labels.shape[1] // MOTION_FACTOR, labels.shape[2] // MOTION_FACTOR), dtype=torch.uint8,
                       device=dev)

    def reset(t: int, which: List[int]) -> None:
        for scale in {c.full_scale for c in cfgs}:
            sel = [b for b in which if cfgs[b].full_scale == scale]
            if sel:
                ii = torch.tensor(sel, device=dev)
                labels[ii] = t_of(_key("full", scale), t)
                conf[ii] = t_of(_key("full_conf", scale), t)

    row = {int(t): j for j, t in enumerate(cache.labeled_t)}
    confs: List[torch.Tensor] = []
    preds: Optional[List[List[torch.Tensor]]] = [[] for _ in cfgs] if return_predictions else None
    offset = (torch.arange(B, device=dev) * NUM_CLASSES * NUM_CLASSES)[:, None]

    def score(t: int) -> None:
        if preds is not None:
            for b in range(B):
                preds[b].append(labels[b].long().clone())
        if t in row:
            label = cache.tensor("labels", dev)[row[t]]
            ok = label != IGNORE
            code = label[ok].long()[None] * NUM_CLASSES + labels[:, ok].long() + offset
            confs.append(torch.bincount(code.flatten(), minlength=B * NUM_CLASSES * NUM_CLASSES)
                         .reshape(B, NUM_CLASSES, NUM_CLASSES))

    reset(0, list(range(B)))
    score(0)
    moved = [[grid.num_cells] for _ in cfgs]
    fallback = [[False] for _ in cfgs]
    hw = tuple(labels.shape[1:])
    for t in range(1, cache.num_frames):
        H = torch.from_numpy(cache["H"][t])
        moving = moving_cells(t_of("motion", t), grid, g0.theta, g0.rho)
        k = int(moving.sum())
        falls = [k / grid.num_cells > c.tau for c in cfgs]
        if not all(falls):
            labels = warp(labels[:, None].float(), H, "nearest")[0][:, 0].round().to(torch.uint8)
            conf = warp(conf[:, None].float(), H, "nearest", frame_size=hw)[0][:, 0].round().to(torch.uint8)
            for scale, ii in by_ctx_t.items():
                labels[ii], conf[ii] = fuse(labels[ii], conf[ii], t_of(_key("ctx", scale), t),
                                            t_of(_key("ctx_conf", scale), t))
            if k:
                idx = torch.nonzero(moving).flatten().to(dev)
                labels = paste_cells(labels, _blocks(t_of(f"cells_{g0.glimpse_size}", t), n, grid.cols, idx), n,
                                     grid.cols, idx)
                conf = paste_cells(conf, _blocks(t_of(f"cells_conf_{g0.glimpse_size}", t), s, grid.cols, idx), s,
                                   grid.cols, idx)
        reset(t, [b for b in range(B) if falls[b]])
        for b in range(B):
            moved[b].append(grid.num_cells if falls[b] else k)
            fallback[b].append(falls[b])
        score(t)
    conf_all = torch.stack(confs, 1).cpu().numpy().astype(np.int64)  # (B, labelled frames, 19, 19)
    results = [ClipResult(conf=conf_all[b], moved_cells=np.array(moved[b], np.int32),
                          fallback=np.array(fallback[b], bool), num_cells=grid.num_cells) for b in range(B)]
    return (results, preds) if return_predictions else results


def reuse_bound(cache: ClipCache, full_scale: float, device) -> np.ndarray:
    """(T-1, 19, 19): at each lag L = 1 .. T-1, the confusion summed over the labelled frames t
    (with t - L >= 0) of the full pass at t - L, warped once by the chained homographies to t."""
    dev = torch.device(device)
    T = cache.num_frames
    out = np.zeros((T - 1, NUM_CLASSES, NUM_CLASSES), np.int64)
    full = cache[_key("full", full_scale)]
    Hs = torch.from_numpy(cache["H"])
    for j, t in enumerate(cache.labeled_t):
        label = cache["labels"][j]
        chain = torch.eye(3, dtype=torch.float64)
        for lag in range(1, int(t) + 1):
            chain = chain @ Hs[int(t) - lag + 1]  # maps frame t - lag onto t
            src = torch.from_numpy(full[int(t) - lag]).to(dev)
            pred = warp(src[None, None].float(), chain, "nearest")[0][0, 0].round().to(torch.uint8).cpu().numpy()
            out[lag - 1] += _confusion(label, pred)
    return out
