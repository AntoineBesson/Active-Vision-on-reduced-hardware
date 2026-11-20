"""Method 5, phase 2, stage B: one clip of the closed loop.

Spec: docs/superpowers/specs/2026-11-16-method5-phase2-design.md (stage B). Plan:
docs/superpowers/plans/2026-11-17-method5-phase2.md, Task 5.

Each configuration (:class:`warp.policy.StepConfig`) runs its own :class:`warp.policy.WarpStep`
over the clip's frames in order: frame 0 from its initial state (the prior for P, the uniform
grid otherwise), every later frame from the state its previous frame returned. Per frame and
scale, the warped images of the configurations go through the network together (``batch`` at a
time); each configuration keeps its own state, so no configuration reads another's output.
Scored against the labels of the labelled frames, one confusion per labelled position, so that
the total and the drift along the clip come from the same numbers. fp32 (the accuracy runs).

A fixed warp (P, uniform) never reads its state, so its output on a frame does not depend on the
frames before it. When every configuration is fixed (stage B after gate A: P only), only the
labelled frames are run by default; the confusions are those of running every frame (tested).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.harness import frame_to_tensor
from reuse.method import FRAME
from reuse.replay import NUM_CLASSES, _confusion_t

from .oracle import SAL_CELL
from .policy import StepConfig, WarpStep

__all__ = ["build_steps", "stream_clip"]


def build_steps(model: torch.nn.Module, cfgs: Sequence[StepConfig], priors: Mapping[float, torch.Tensor],
                cell: int = SAL_CELL) -> List[WarpStep]:
    """One fp32 :class:`WarpStep` per configuration; a P configuration takes ``priors[scale]``.
    Build them once per model and reuse them across clips: a fixed warp's operands are made on
    the first frame and kept."""
    names = [c.name() for c in cfgs]
    if len(set(names)) != len(names):
        raise ValueError("build_steps: a configuration is given twice")
    steps = []
    for c in cfgs:
        prior = None
        if c.saliency == "P":
            if c.scale not in priors:
                raise ValueError(f"no prior at scale {c.scale:g} for {c.name()}")
            prior = priors[c.scale]
        steps.append(WarpStep(model, c, prior=prior, cell=cell, precision="fp32"))
    return steps


def stream_clip(dataset, index: int, model: torch.nn.Module, steps: Sequence[WarpStep], device,
                batch: int = 4, allow_any_frame_size: bool = False,
                every_frame: Optional[bool] = None) -> Tuple[List[int], Dict[str, np.ndarray]]:
    """The clip's labelled positions and, per configuration name, an (n_labelled, 19, 19) int64
    confusion, one per labelled position.

    ``every_frame``: run every frame of the clip in order (the closed loop); None runs every frame
    unless every configuration is fixed, then only the labelled ones. Running every frame needs
    them all on disk (``train_all``, val ``img_all``). Refuses a frame that is not 1024 x 2048
    unless ``allow_any_frame_size``."""
    names = [s.cfg.name() for s in steps]
    if not steps:
        raise ValueError("stream_clip: no configuration")
    if len(set(names)) != len(names):
        raise ValueError("stream_clip: a configuration is given twice")
    if any(s.model is not model for s in steps):
        raise ValueError("stream_clip: every step must run the given model")
    if any(s.precision != "fp32" for s in steps):
        raise ValueError("stream_clip: the accuracy runs are fp32")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    clip_id = dataset.clip_id(index)
    mask = np.asarray(dataset.labeled_mask(index), dtype=bool)
    labelled = [int(t) for t in np.flatnonzero(mask)]
    if not labelled:
        raise ValueError(f"clip {clip_id} has no labelled frame")
    run_all = (not all(s.cfg.fixed for s in steps)) if every_frame is None else bool(every_frame)
    if run_all:
        numbers = np.asarray(dataset.frame_indices(index))
        if len(numbers) > 1 and not bool((np.diff(numbers) == 1).all()):
            raise ValueError(f"clip {clip_id}: the closed loop needs every frame, and frame numbers {list(numbers)} "
                             "have gaps (use a split with every frame: train_all)")
    frames = list(range(len(mask))) if run_all else labelled
    pos = {t: j for j, t in enumerate(labelled)}
    by_scale: Dict[float, List[int]] = {}
    for i, s in enumerate(steps):
        by_scale.setdefault(s.cfg.scale, []).append(i)
    conf = {n: torch.zeros(len(labelled), NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for n in names}
    states: List[Optional[torch.Tensor]] = [None] * len(steps)
    model = model.eval()
    with torch.no_grad():
        for t in frames:
            frame = frame_to_tensor(dataset.load_frame(index, t), device)
            size = (int(frame.shape[-2]), int(frame.shape[-1]))
            if not allow_any_frame_size and size != FRAME:
                raise ValueError(f"frame is {size[0]}x{size[1]}; the T4 latency tables are for {FRAME[0]}x{FRAME[1]}")
            if t == frames[0]:
                states = [s.initial_state(size, device) for s in steps]
            label = None
            if t in pos:
                label = torch.from_numpy(np.asarray(dataset.load_label(index, t), dtype=np.int64)).to(device)
                if tuple(label.shape) != size:
                    raise ValueError(f"clip {clip_id} frame {t}: label {tuple(label.shape)} and frame {size} differ")
            for idx in by_scale.values():
                for start in range(0, len(idx), batch):
                    part = idx[start:start + batch]
                    warped = [steps[i].warp(frame, states[i]) for i in part]
                    logits = model(torch.cat([x for x, _ in warped]))
                    for k, i in enumerate(part):
                        labels, states[i] = steps[i].finish(logits[k:k + 1], warped[k][1], states[i])
                        if label is not None:
                            conf[names[i]][pos[t]] += _confusion_t(label, labels)
    return labelled, {n: c.cpu().numpy() for n, c in conf.items()}
