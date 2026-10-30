"""Sub-clip (temporal window) enumeration -- numpy only, dataset-agnostic.

A sub-clip is a set of frame positions inside ONE clip::

    positions = start + frame_stride * arange(length)

Windows never cross clip boundaries, so any split made at the clip level
(:mod:`data.splits`) carries over to sub-clips unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["SubClipSpec", "enumerate_windows"]


@dataclass(frozen=True)
class SubClipSpec:
    """How to cut clips into sub-clips.

    Attributes
    ----------
    length:
        T, number of frames per sub-clip.
    frame_stride:
        Temporal step between consecutive frames of a sub-clip (1 = every
        frame; 5 = only Cityscapes-VPS-style labelled frames, if aligned).
    window_stride:
        Step between the start positions of consecutive windows. ``None``
        means ``length * frame_stride`` (non-overlapping windows); use 1 for
        every possible start (e.g. random sampling for training).
    min_labeled:
        Keep only windows with at least this many labelled frames.
    require_last_labeled:
        Keep only windows whose last frame is labelled (causal evaluation:
        predict a labelled frame from its past).
    """

    length: int
    frame_stride: int = 1
    window_stride: Optional[int] = None
    min_labeled: int = 0
    require_last_labeled: bool = False

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}")
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.window_stride is not None and self.window_stride < 1:
            raise ValueError(f"window_stride must be >= 1, got {self.window_stride}")
        if not 0 <= self.min_labeled <= self.length:
            raise ValueError(f"min_labeled must be in [0, length], got {self.min_labeled}")

    @property
    def span(self) -> int:
        """Number of clip frames covered from the first to the last position."""
        return (self.length - 1) * self.frame_stride + 1

    @property
    def step(self) -> int:
        return self.window_stride if self.window_stride is not None else self.length * self.frame_stride


def enumerate_windows(num_frames: int, labeled_mask: np.ndarray, spec: SubClipSpec) -> List[np.ndarray]:
    """All valid windows of a clip, as int64 position arrays of length T, in start order.

    Clips shorter than ``spec.span`` yield no window.
    """
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    if labeled_mask.shape != (num_frames,):
        raise ValueError(f"labeled_mask must have shape ({num_frames},), got {labeled_mask.shape}")
    offsets = spec.frame_stride * np.arange(spec.length, dtype=np.int64)
    windows = []
    for start in range(0, num_frames - spec.span + 1, spec.step):
        pos = start + offsets
        lab = labeled_mask[pos]
        if lab.sum() < spec.min_labeled:
            continue
        if spec.require_last_labeled and not lab[-1]:
            continue
        windows.append(pos)
    return windows
