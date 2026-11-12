"""Per-object recall of thing instances, split by size and camera-corrected motion (numpy only).

mIoU pools pixels per class, so large static regions set it and a small moving
pedestrian weighs almost nothing. This module scores every ground-truth thing
instance of a labelled frame on its own:

* **Object**: one instance id of a thing class (train ids 11..18) in one
  labelled frame, from the VPS ``panoptic_inst`` maps (``train_id * 1000 + obj``,
  255 = void and crowd). Objects smaller than ``MIN_AREA`` pixels are dropped.
* **Local IoU**: the backbones are semantic, so there is no predicted instance.
  With G the object's mask, P the pixels predicted as its class, R the object's
  bounding box grown by ``d = max(8, 0.25 sqrt(area))`` pixels, O the pixels of
  the *other* ground-truth objects of the same class and V the void pixels::

      IoU = |G & P| / (|G| + |P & R & ~G & ~O & ~V|)

  Spill of the right class around the object counts against it; pixels of a
  neighbour of the same class, and void, do not.
* **Found**: IoU > ``FOUND_IOU`` (0.5).
* **Motion**: VPS keeps instance ids across a video. For an object seen in two
  labelled frames a < b, the cached per-frame homographies (``H[t]``: frame
  t-1 -> t, full-resolution pixels) are chained from a to b, the centroid at a
  is warped with them, and the residual to the centroid at b, divided by
  ``b - a``, is its motion in px per frame. The backward and forward pairs are
  averaged when both exist. No pair: ``new``. A pair whose chain holds a failed
  homography (support 0): ``unknown``.
* **Recall** of a bin = found / objects, with a 95% bootstrap interval over
  clips (objects of one clip are correlated).

The bin edges are module defaults, overridable with a ``bins`` dict, and must be
frozen before any method is scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "LABEL_DIVISOR", "THING_CLASSES", "MIN_AREA", "FOUND_IOU", "DEFAULT_BINS", "MOTION_STATUSES",
    "FrameObjects", "frame_objects", "chain_homography", "warp_point", "clip_objects", "ring_width",
    "local_ious", "size_bins", "motion_bins", "recall_table", "recall_at_equal_latency",
]

LABEL_DIVISOR = 1000
THING_CLASSES = tuple(range(11, 19))  # person, rider, car, truck, bus, train, motorcycle, bicycle
IGNORE = 255
MIN_AREA = 64
FOUND_IOU = 0.5
MOTION_STATUSES = ("measured", "new", "unknown")

#: First-guess edges. Size: mask area in px at 1024 x 2048. Motion: px per frame.
DEFAULT_BINS: Dict[str, Any] = {
    "size_edges": [1024, 4096, 16384],
    "size_names": ["tiny", "small", "medium", "large"],
    "motion_edges": [2.0, 8.0],
    "motion_names": ["static", "slow", "fast"],
}


# --------------------------------------------------------------------------- #
# Objects of a frame
# --------------------------------------------------------------------------- #
@dataclass
class FrameObjects:
    """The thing objects of one labelled frame, sorted by id."""

    ids: np.ndarray  # (n,) int64 raw instance ids (train_id * 1000 + obj)
    cls: np.ndarray  # (n,) int64 train ids
    area: np.ndarray  # (n,) int64 pixels
    centroid: np.ndarray  # (n, 2) float64 (x, y), full-resolution pixels
    bbox: np.ndarray  # (n, 4) int64 (y0, x0, y1, x1), end exclusive

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def index(self) -> Dict[int, int]:
        return {int(i): j for j, i in enumerate(self.ids)}


def frame_objects(inst: np.ndarray, min_area: Optional[int] = None) -> FrameObjects:
    """Thing objects of an (H, W) ``panoptic_inst`` id map (one pass over the pixels).

    ``min_area`` defaults to the module's ``MIN_AREA`` at call time."""
    if min_area is None:
        min_area = MIN_AREA
    inst = np.asarray(inst, dtype=np.int64)
    if inst.ndim != 2:
        raise ValueError(f"expected an (H, W) id map, got shape {inst.shape}")
    cls_map = np.where(inst >= LABEL_DIVISOR, inst // LABEL_DIVISOR, -1)
    thing = np.isin(cls_map, THING_CLASSES)
    ys, xs = np.nonzero(thing)
    vals = inst[ys, xs]
    ids, inv, area = np.unique(vals, return_inverse=True, return_counts=True)
    n = len(ids)
    cx = np.bincount(inv, weights=xs, minlength=n) / np.maximum(area, 1)
    cy = np.bincount(inv, weights=ys, minlength=n) / np.maximum(area, 1)
    y0 = np.full(n, np.iinfo(np.int64).max)
    x0 = np.full(n, np.iinfo(np.int64).max)
    y1 = np.full(n, -1, dtype=np.int64)
    x1 = np.full(n, -1, dtype=np.int64)
    np.minimum.at(y0, inv, ys)
    np.minimum.at(x0, inv, xs)
    np.maximum.at(y1, inv, ys)
    np.maximum.at(x1, inv, xs)
    keep = area >= min_area
    return FrameObjects(ids=ids[keep], cls=(ids // LABEL_DIVISOR)[keep], area=area[keep].astype(np.int64),
                        centroid=np.stack([cx, cy], 1)[keep],
                        bbox=np.stack([y0, x0, y1 + 1, x1 + 1], 1)[keep].astype(np.int64))


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #
def chain_homography(H: np.ndarray, a: int, b: int) -> np.ndarray:
    """Frame a -> frame b (a < b) from per-frame ``H[t]`` (frame t-1 -> t)."""
    if not 0 <= a < b < len(H):
        raise ValueError(f"need 0 <= a < b < {len(H)}, got a={a}, b={b}")
    out = np.eye(3)
    for t in range(a + 1, b + 1):
        out = np.asarray(H[t], dtype=np.float64) @ out
    return out


def warp_point(Hab: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(n, 2) points (x, y) through a 3x3 homography."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    p = np.concatenate([xy, np.ones((len(xy), 1))], 1) @ np.asarray(Hab, dtype=np.float64).T
    return p[:, :2] / p[:, 2:3]


def clip_objects(clip_id: str, labeled_t: Sequence[int], frames: Sequence[FrameObjects], H: np.ndarray,
                 support: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Object table of one clip, columns as arrays, rows ordered by (t, id).

    ``frames[j]`` holds the objects of labelled frame ``labeled_t[j]``; ``H`` is
    (T, 3, 3) with ``H[t]``: t-1 -> t; ``support`` (T,) marks a failed fit with 0.
    Columns: ``clip``, ``t``, ``id``, ``cls``, ``area``, ``cx``, ``cy``, ``motion``
    (px/frame, NaN unless measured), ``status`` (index into ``MOTION_STATUSES``),
    ``matched`` (pairs found, 0..2) and ``raw_shift`` (mean centroid shift with no
    homography, px/frame; a check that ids persist).
    """
    labeled_t = [int(t) for t in labeled_t]
    if len(frames) != len(labeled_t):
        raise ValueError(f"{len(frames)} frames of objects for {len(labeled_t)} labelled frames")
    ok = np.ones(len(H), bool) if support is None else np.asarray(support) > 0
    index = [f.index() for f in frames]
    cols: Dict[str, List[Any]] = {k: [] for k in ("t", "id", "cls", "area", "cx", "cy", "motion", "status",
                                                  "matched", "raw_shift")}
    for j, (t, objs) in enumerate(zip(labeled_t, frames)):
        pairs = []  # the neighbouring labelled frames
        if j > 0:
            pairs.append(j - 1)
        if j + 1 < len(frames):
            pairs.append(j + 1)
        for k in range(len(objs)):
            oid = int(objs.ids[k])
            residuals, raws, matched = [], [], 0
            for o in pairs:
                m = index[o].get(oid)
                if m is None:
                    continue
                matched += 1
                a, b = sorted((labeled_t[o], t))
                ca = frames[o].centroid[m] if labeled_t[o] == a else objs.centroid[k]
                cb = objs.centroid[k] if t == b else frames[o].centroid[m]
                raws.append(float(np.linalg.norm(cb - ca)) / (b - a))
                if not ok[a + 1:b + 1].all():
                    continue
                pred = warp_point(chain_homography(H, a, b), ca)[0]
                residuals.append(float(np.linalg.norm(cb - pred)) / (b - a))
            if residuals:
                status, motion = 0, float(np.mean(residuals))
            elif matched == 0:
                status, motion = 1, math.nan
            else:
                status, motion = 2, math.nan
            cols["t"].append(t)
            cols["id"].append(oid)
            cols["cls"].append(int(objs.cls[k]))
            cols["area"].append(int(objs.area[k]))
            cols["cx"].append(float(objs.centroid[k, 0]))
            cols["cy"].append(float(objs.centroid[k, 1]))
            cols["motion"].append(motion)
            cols["status"].append(status)
            cols["matched"].append(matched)
            cols["raw_shift"].append(float(np.mean(raws)) if raws else math.nan)
    out = {k: np.asarray(v, dtype=np.float64 if k in ("cx", "cy", "motion", "raw_shift") else np.int64)
           for k, v in cols.items()}
    out["clip"] = np.asarray([clip_id] * len(out["t"]), dtype=object)
    return out


# --------------------------------------------------------------------------- #
# Local IoU
# --------------------------------------------------------------------------- #
def ring_width(area: np.ndarray) -> np.ndarray:
    """Pixels the bounding box grows by for an object of ``area`` pixels."""
    return np.maximum(8, np.ceil(0.25 * np.sqrt(np.asarray(area, dtype=np.float64)))).astype(np.int64)


def local_ious(inst: np.ndarray, pred: np.ndarray, objs: FrameObjects) -> np.ndarray:
    """(n,) local IoU of each object of ``objs`` against an (H, W) semantic prediction."""
    inst = np.asarray(inst, dtype=np.int64)
    pred = np.asarray(pred)
    if inst.shape != pred.shape:
        raise ValueError(f"prediction shape {pred.shape} != label shape {inst.shape}")
    h, w = inst.shape
    out = np.empty(len(objs), dtype=np.float64)
    d = ring_width(objs.area)
    for k in range(len(objs)):
        y0, x0, y1, x1 = objs.bbox[k]
        y0, x0 = max(0, y0 - d[k]), max(0, x0 - d[k])
        y1, x1 = min(h, y1 + d[k]), min(w, x1 + d[k])
        g = inst[y0:y1, x0:x1]
        p = pred[y0:y1, x0:x1] == objs.cls[k]
        mine = g == objs.ids[k]
        same_cls_other = (g >= LABEL_DIVISOR) & (g // LABEL_DIVISOR == objs.cls[k]) & ~mine
        free = ~mine & ~same_cls_other & (g != IGNORE)
        tp = int(np.count_nonzero(p & mine))
        fp = int(np.count_nonzero(p & free))
        out[k] = tp / (int(objs.area[k]) + fp)
    return out


# --------------------------------------------------------------------------- #
# Bins and recall
# --------------------------------------------------------------------------- #
def _bins(bins: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out = dict(DEFAULT_BINS)
    out.update(bins or {})
    if len(out["size_names"]) != len(out["size_edges"]) + 1:
        raise ValueError("size_names must have one more entry than size_edges")
    if len(out["motion_names"]) != len(out["motion_edges"]) + 1:
        raise ValueError("motion_names must have one more entry than motion_edges")
    return out


def size_bins(area: np.ndarray, bins: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """Size bin name per object."""
    b = _bins(bins)
    return np.asarray(b["size_names"], dtype=object)[np.searchsorted(b["size_edges"], area, side="right")]


def motion_bins(motion: np.ndarray, status: np.ndarray, bins: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """Motion bin name per object: the measured bins, then ``new`` and ``unknown``."""
    b = _bins(bins)
    motion = np.asarray(motion, dtype=np.float64)
    names = np.asarray(b["motion_names"], dtype=object)[
        np.searchsorted(b["motion_edges"], np.nan_to_num(motion, nan=0.0), side="right")]
    status = np.asarray(status)
    names[status == 1] = "new"
    names[status == 2] = "unknown"
    return names


def _ci(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return (math.nan, math.nan)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def recall_table(found: np.ndarray, clip: np.ndarray, groups: Mapping[str, np.ndarray], n_boot: int = 1000,
                 seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Recall of each group (a boolean mask over objects) with a clip bootstrap.

    Every group uses the same clip resamples, so intervals of different groups
    are comparable. A group with no object has recall NaN.
    """
    found = np.asarray(found, dtype=bool)
    clips, cidx = np.unique(np.asarray(clip, dtype=str), return_inverse=True)
    nc = len(clips)
    rng = np.random.default_rng(seed)
    picks = [np.bincount(rng.integers(0, nc, nc), minlength=nc) for _ in range(n_boot)]
    out: Dict[str, Dict[str, Any]] = {}
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        hits = np.bincount(cidx[mask], weights=found[mask], minlength=nc)
        tot = np.bincount(cidx[mask], minlength=nc).astype(np.float64)
        n = int(tot.sum())
        recall = float(hits.sum() / n) if n else math.nan
        boots = []
        for w in picks:
            d = float(w @ tot)
            boots.append(float(w @ hits) / d if d else math.nan)
        out[name] = {"objects": n, "found": int(hits.sum()), "recall": recall, "ci95": _ci(boots) if n else
                     (math.nan, math.nan)}
    return out


def recall_at_equal_latency(found: np.ndarray, latency_ms: float, ff_found: Sequence[np.ndarray],
                            ff_latency_ms: Sequence[float], clip: np.ndarray, mask: np.ndarray,
                            n_boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """Recall of a method minus the full-frame recall curve at its latency, on the objects of ``mask``.

    The curve is piecewise linear in latency between the full-frame scales and
    undefined (NaN) outside them, as :class:`eval.tradeoff.FullFrameCurve`. The
    interval resamples clips, the same clips for the method and every scale.
    """
    from .tradeoff import FullFrameCurve

    mask = np.asarray(mask, dtype=bool)
    clips, cidx = np.unique(np.asarray(clip, dtype=str), return_inverse=True)
    nc = len(clips)
    tot = np.bincount(cidx[mask], minlength=nc).astype(np.float64)

    def per_clip(f: np.ndarray) -> np.ndarray:
        return np.bincount(cidx[mask], weights=np.asarray(f, dtype=bool)[mask], minlength=nc)

    hits = per_clip(found)
    ff_hits = [per_clip(f) for f in ff_found]

    def reading(w: np.ndarray) -> Tuple[float, float]:
        d = float(w @ tot)
        if not d:
            return math.nan, math.nan
        r = float(w @ hits) / d
        curve = FullFrameCurve(ff_latency_ms, [float(w @ h) / d for h in ff_hits])
        return r, r - curve.miou_at(latency_ms)

    recall, delta = reading(np.ones(nc))
    rng = np.random.default_rng(seed)
    boots = [reading(np.bincount(rng.integers(0, nc, nc), minlength=nc)) for _ in range(n_boot)]
    return {"objects": int(tot.sum()), "recall": recall, "delta_recall_at_equal_latency": delta,
            "recall_ci95": _ci([b[0] for b in boots]), "delta_ci95": _ci([b[1] for b in boots]),
            "nan_resamples": int(sum(math.isnan(b[1]) for b in boots))}
