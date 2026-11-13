"""Method 5, phase 1: one clip of the bound, every configuration of the grid on its labelled frames.

Spec: docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md. Per labelled frame t:

1. the teacher (scale-1 full pass) of t and of t - 1 (t itself when t = 0), and, for the error
   saliency E, the uniform pass at each scale of t and t - 1, all with
   :func:`reuse.method.full_pass` (the full-frame baseline's resize, model and upsampling);
2. per configuration, the oracle saliency of its kind, timing and smoothing, the warp
   (:func:`warp.grid.make_grid` at its strength and scale) and one pass of the frozen model on
   the warped frame (:func:`warped_pass`), scored against the labels.

The uniform configurations (alpha = 0) go through the same warp and unwarp with a uniform grid:
they are the reference and the pipeline check. Nothing is replayed: every configuration is a
network pass on a different image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.harness import frame_to_tensor
from reuse.method import FRAME, full_pass
from reuse.motion import MOTION_FACTOR
from reuse.replay import IGNORE, NUM_CLASSES, TEACHER_SCALE, _confusion_t

from .grid import WarpGrid, make_grid, unwarp_logits, warp_frame
from .oracle import KINDS, SAL_CELL, saliency

__all__ = ["WarpConfig", "TIMINGS", "SUPERSAMPLE", "RESAMPLES", "RESAMPLE", "warped_pass", "warped_passes",
           "bound_clip", "resize_ref_name", "label_pixels"]

TIMINGS = ("now", "prev")
SUPERSAMPLE = 2  # spec (plan): the warped image is sampled at twice its size and averaged 2 x 2
# How the warped image is resampled: "supersample" is the accepted spec's (plan) detail;
# "antialias" (warp.grid._axis_filter) is the baseline's own antialiased filter with a local
# width, proposed on 2026-11-13 because the 2 x 2 mean does not reproduce the baseline at alpha 0.
RESAMPLES = ("supersample", "antialias")
RESAMPLE = "supersample"


@dataclass(frozen=True)
class WarpConfig:
    scale: float  # c: the warped image is c x the frame, so it costs what the full frame at c costs
    saliency: Optional[str] = None  # "E", "B" or "S"; None for the uniform pass
    alpha: float = 0.0  # strength of the warp; 0 is the uniform downscale
    sigma: float = 1.0  # smoothing of the saliency, in cells
    timing: str = "now"  # "now": saliency from the teacher of frame t; "prev": of frame t - 1

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        if self.saliency is not None and self.saliency not in KINDS:
            raise ValueError(f"saliency must be one of {KINDS} or None, got {self.saliency!r}")
        if self.timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, got {self.timing!r}")
        if (self.saliency is None) != (self.alpha == 0.0):
            raise ValueError("a saliency goes with alpha > 0, and alpha 0 (the uniform pass) with none")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")

    @property
    def uniform(self) -> bool:
        return self.alpha == 0.0

    def name(self) -> str:
        if self.uniform:
            return f"5o/c{self.scale:g}/uniform"
        return f"5o/c{self.scale:g}/{self.saliency}/a{self.alpha:g}/s{self.sigma:g}/{self.timing}"


def resize_ref_name(scale: float) -> str:
    """The row of the baseline's own resize at ``scale`` (``full_pass``), kept next to the uniform
    warp so that a pipeline-check gap can be traced to the resampling or to the reference file."""
    return f"ref/c{scale:g}/resize"


def _out_size(frame: torch.Tensor, scale: float) -> Tuple[int, int]:
    h, w = frame.shape[-2:]
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))  # FullFrameMethod.input_size


def _warp(frame: torch.Tensor, grid: WarpGrid, size: Tuple[int, int], scale: float, resample: str) -> torch.Tensor:
    if resample == "antialias":
        return warp_frame(frame, grid, size, antialias=True)
    if resample != "supersample":
        raise ValueError(f"resample must be one of {RESAMPLES}, got {resample!r}")
    # supersampling stands in for the antialiasing of a downscale; the full frame at scale 1 is not resized
    return warp_frame(frame, grid, size, supersample=SUPERSAMPLE if scale < 1 else 1)


def warped_passes(model: torch.nn.Module, frame: torch.Tensor, grids: Sequence[WarpGrid], scale: float,
                  dtype: torch.dtype = torch.float32, resample: str = RESAMPLE) -> List[torch.Tensor]:
    """:func:`warped_pass` for several grids of one scale, the network run once on their batch."""
    h, w = frame.shape[-2:]
    size = _out_size(frame, scale)
    x = torch.cat([_warp(frame, g, size, scale, resample) for g in grids]).to(dtype)
    logits = model(x).float()
    return [unwarp_logits(logits[i:i + 1], g, (h, w), model.align_corners, model.output_stride,
                          input_size=size).argmax(1)[0] for i, g in enumerate(grids)]


def warped_pass(model: torch.nn.Module, frame: torch.Tensor, grid: WarpGrid, scale: float,
                dtype: torch.dtype = torch.float32, resample: str = RESAMPLE) -> torch.Tensor:
    """(1, 3, H, W) frame -> (H, W) labels: warp to round(scale H) x round(scale W) (``resample``:
    see :data:`RESAMPLES`), one forward pass, unwarp the logits with the model's convention
    (:func:`warp.grid.unwarp_logits`), argmax."""
    return warped_passes(model, frame, [grid], scale, dtype, resample)[0]


class _Frame:
    """The passes and saliencies of one timing ("now" or "prev") of a labelled frame, made on demand."""

    def __init__(self, model: torch.nn.Module, frame: torch.Tensor, cell: int):
        self.model, self.frame, self.cell = model, frame, cell
        h, w = frame.shape[-2:]
        self.small = (max(1, h // MOTION_FACTOR), max(1, w // MOTION_FACTOR))
        self._passes: Dict[float, torch.Tensor] = {}
        self._sal: Dict[Tuple[str, float, Optional[float]], torch.Tensor] = {}

    def labels(self, scale: float) -> torch.Tensor:
        if scale not in self._passes:
            self._passes[scale] = full_pass(self.model, self.frame, scale, self.small)[0]
        return self._passes[scale]

    def saliency(self, kind: str, sigma: float, scale: float) -> torch.Tensor:
        key = (kind, sigma, scale if kind == "E" else None)
        if key not in self._sal:
            uniform = self.labels(scale) if kind == "E" else None
            self._sal[key] = saliency(kind, self.labels(TEACHER_SCALE), uniform, sigma, self.cell)
        return self._sal[key]


def bound_clip(dataset, index: int, model: torch.nn.Module, cfgs: Sequence[WarpConfig], device,
               allow_any_frame_size: bool = False, cell: int = SAL_CELL, batch: int = 1,
               resize_refs: bool = False, resample: str = RESAMPLE) -> Dict[str, np.ndarray]:
    """Configuration name -> (19, 19) int64 confusion summed over the clip's labelled frames.

    ``batch``: warped images of one scale run through the network together (the predictions are
    the same up to the floating-point noise of batched kernels). ``resize_refs``: also return, for
    every scale of ``cfgs``, the confusion of the baseline's own resize at that scale
    (:func:`resize_ref_name`). Refuses a frame that is not 1024 x 2048 unless
    ``allow_any_frame_size``, and a labelled frame whose previous frame is not on disk (a split
    with only the labelled frames, such as ``train``: ``prev`` would be 5 frames old)."""
    names = [c.name() for c in cfgs]
    if len(set(names)) != len(names):
        raise ValueError("bound_clip: a configuration is given twice")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if resample not in RESAMPLES:
        raise ValueError(f"resample must be one of {RESAMPLES}, got {resample!r}")
    clip_id = dataset.clip_id(index)
    labelled = np.flatnonzero(np.asarray(dataset.labeled_mask(index), dtype=bool))
    numbers = np.asarray(dataset.frame_indices(index))
    by_scale: Dict[float, List[WarpConfig]] = {}
    for c in cfgs:
        by_scale.setdefault(c.scale, []).append(c)
    conf = {n: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for n in names}
    if resize_refs:
        conf.update({resize_ref_name(s): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
                     for s in by_scale})
    model = model.eval()
    with torch.no_grad():
        for t in (int(v) for v in labelled):
            if t > 0 and numbers[t] - numbers[t - 1] != 1:
                raise ValueError(f"clip {clip_id}: the frame before labelled frame {t} is not on disk (frame "
                                 f"numbers {numbers[t - 1]} then {numbers[t]}): use a split with every frame "
                                 "(train_all)")
            now = frame_to_tensor(dataset.load_frame(index, t), device)
            size = tuple(now.shape[-2:])
            if not allow_any_frame_size and size != FRAME:
                raise ValueError(f"frame is {size[0]}x{size[1]}; the T4 latency tables are for {FRAME[0]}x{FRAME[1]}")
            frames = {"now": _Frame(model, now, cell)}
            frames["prev"] = frames["now"] if t == 0 else _Frame(
                model, frame_to_tensor(dataset.load_frame(index, t - 1), device), cell)
            label = torch.from_numpy(np.asarray(dataset.load_label(index, t), dtype=np.int64)).to(device)
            if tuple(label.shape) != size:
                raise ValueError(f"clip {clip_id} frame {t}: label {tuple(label.shape)} and frame {size} differ")
            for scale, group in by_scale.items():
                if resize_refs:
                    conf[resize_ref_name(scale)] += _confusion_t(label, frames["now"].labels(scale))
                grids = [make_grid(frames[c.timing].saliency(c.saliency, c.sigma, c.scale), c.alpha, c.scale)
                         if not c.uniform else make_grid(torch.zeros(1, 1), 0.0, c.scale) for c in group]
                for start in range(0, len(group), batch):
                    preds = warped_passes(model, now, grids[start:start + batch], scale, resample=resample)
                    for c, pred in zip(group[start:start + batch], preds):
                        conf[c.name()] += _confusion_t(label, pred)
    return {n: m.cpu().numpy() for n, m in conf.items()}


def label_pixels(dataset, index: int) -> int:
    """Labelled (not ignored) pixels of the clip: the total of every row of :func:`bound_clip`."""
    return int(sum(int((np.asarray(dataset.load_label(index, int(t))) != IGNORE).sum())
                   for t in np.flatnonzero(np.asarray(dataset.labeled_mask(index), dtype=bool))))
