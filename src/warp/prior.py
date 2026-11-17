"""Method 5, phase 2: fitting the static prior P on the train frames.

Spec: docs/superpowers/specs/2026-11-16-method5-phase2-design.md (stage B, saliency P). Plan:
docs/superpowers/plans/2026-11-17-method5-phase2.md, Task 4.

P is the mean of phase 1's error oracle E at the configuration's scale over the labelled
``train_all`` frames, one map per model and scale: per frame, the teacher (the model at scale 1)
and the uniform pass at the scale (both :func:`reuse.method.full_pass`), the pixels where they
differ, counted per 16-px cell. Counts are integers, so chunks add up exactly; the map is
count / (cell^2 x frames) (:func:`warp.policy.load_prior`).

``scored_only`` (amendment of 2026-11-17): count only the pixels the score reads, those whose
label is not ignored (255). The fit on every pixel put up to a quarter of P's mass on the ego
vehicle at the bottom of the frame, which is never scored.

Each clip also records how many rounds phase 1's density loop needs on its frames' saliencies
(E at the scale, B and S of the teacher, smoothed by each sigma, at each alpha, both axes): a
fixed number of rounds at least that large reproduces the loop (:func:`warp.grid.rounds_needed`).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from eval.harness import frame_to_tensor
from reuse.method import FRAME, full_pass
from reuse.motion import MOTION_FACTOR
from reuse.replay import IGNORE, TEACHER_SCALE

from .grid import rounds_needed
from .oracle import SAL_CELL, pixel_mask, smooth

__all__ = ["ROUND_ALPHAS", "ROUND_SIGMAS", "ROUND_KINDS", "prior_name", "prior_clip"]

ROUND_ALPHAS = (0.1, 0.15, 0.25)  # stage B's strengths
ROUND_SIGMAS = (1.0, 4.0)  # and smoothings
ROUND_KINDS = ("E", "B", "S")


def prior_name(scale: float) -> str:
    return f"prior/c{scale:g}"


def _counts(mask: torch.Tensor, cell: int) -> torch.Tensor:
    h, w = mask.shape
    if h % cell or w % cell:
        raise ValueError(f"frame {h}x{w} is not a whole number of {cell}-px saliency cells")
    return mask.reshape(h // cell, cell, w // cell, cell).sum((1, 3), dtype=torch.int64)


def prior_clip(dataset, index: int, model: torch.nn.Module, scales: Sequence[float], device,
               allow_any_frame_size: bool = False, cell: int = SAL_CELL,
               scored_only: bool = False) -> Dict[str, object]:
    """One clip: per scale, the (gh, gw) int64 count of E pixels per cell summed over the clip's
    labelled frames (``prior_name(scale)``), ``frames`` (the number of labelled frames), and
    ``rounds`` (the most rounds the density loop needed, per ``"<kind>/c<scale>"``). With
    ``scored_only``, an E pixel counts only where the frame's label is not ignored. Refuses a
    frame that is not 1024 x 2048 unless ``allow_any_frame_size``."""
    labelled = np.flatnonzero(np.asarray(dataset.labeled_mask(index), dtype=bool))
    counts: Dict[float, torch.Tensor] = {}
    rounds: Dict[str, int] = {f"{k}/c{s:g}": 0 for k in ROUND_KINDS for s in scales}
    model = model.eval()
    with torch.no_grad():
        for t in (int(v) for v in labelled):
            frame = frame_to_tensor(dataset.load_frame(index, t), device)
            size = tuple(frame.shape[-2:])
            if not allow_any_frame_size and size != FRAME:
                raise ValueError(f"frame is {size[0]}x{size[1]}; the T4 latency tables are for {FRAME[0]}x{FRAME[1]}")
            small = (max(1, size[0] // MOTION_FACTOR), max(1, size[1] // MOTION_FACTOR))
            scored = None
            if scored_only:
                label = torch.from_numpy(np.asarray(dataset.load_label(index, t), dtype=np.int64)).to(device)
                if tuple(label.shape) != size:
                    raise ValueError(f"clip {dataset.clip_id(index)} frame {t}: label {tuple(label.shape)} and "
                                     f"frame {size} differ")
                scored = label != IGNORE
            teacher = full_pass(model, frame, TEACHER_SCALE, small)[0]
            pooled = {k: _counts(pixel_mask(k, teacher), cell).cpu().double() / cell ** 2 for k in ("B", "S")}
            for s in scales:
                e = pixel_mask("E", teacher, full_pass(model, frame, s, small)[0])
                c = _counts(e if scored is None else e & scored, cell)
                counts[s] = c if s not in counts else counts[s] + c
                maps = {"E": c.cpu().double() / cell ** 2, **pooled}
                for k in ROUND_KINDS:
                    key = f"{k}/c{s:g}"
                    for sigma in ROUND_SIGMAS:
                        sal = smooth(maps[k], sigma)
                        for a in ROUND_ALPHAS:
                            for prof in (sal.mean(1), sal.mean(0)):
                                rounds[key] = max(rounds[key], rounds_needed(prof, a, s))
    if not counts:
        raise ValueError(f"clip {dataset.clip_id(index)} has no labelled frame")
    out: Dict[str, object] = {prior_name(s): counts[s].cpu() for s in scales}
    out["frames"] = int(len(labelled))
    out["rounds"] = rounds
    return out
