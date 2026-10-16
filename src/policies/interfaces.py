"""Gaze-policy interface: which regions of a frame to inspect at high resolution.

Two families of policies must fit the same contract:

* **memory-free** policies (e.g. uncertainty-driven on the low-resolution
  context pass) ignore the ``memory`` argument;
* **canvas-aware** policies read the persistent scene memory
  (:class:`models.interfaces.MemoryState`), e.g. to revisit stale regions.

The runner passes ``memory`` only to policies with ``uses_memory = True``.

Coordinate convention (shared with :mod:`models.interfaces`): a :class:`Box`
is expressed in coordinates *normalised to the full-resolution frame*, so
policies are independent of the context, glimpse and canvas resolutions.
``x`` runs along the width, ``y`` along the height, the origin is the top-left
corner, ``(x0, y0)`` is inclusive and ``(x1, y1)`` exclusive, all in [0, 1].

Tensor shapes are per sample (no batch dimension): evaluation runs one clip
at a time, at batch size 1. This module defines contracts only -- no policy
is implemented here -- and imports without torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

try:  # Python >= 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

if TYPE_CHECKING:  # imported for annotations only, so torch stays optional
    import torch

    from models.interfaces import MemoryState

__all__ = ["Box", "GazeState", "GazePolicy"]


@dataclass(frozen=True)
class Box:
    """Axis-aligned glimpse region in normalised full-frame coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        vals = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"box coordinates must be finite, got {vals}")
        if not (0.0 <= self.x0 < self.x1 <= 1.0 and 0.0 <= self.y0 < self.y1 <= 1.0):
            raise ValueError(f"box must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {vals}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Fraction of the frame covered by the box."""
        return self.width * self.height

    def to_pixels(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """``(top, left, bottom, right)`` integer pixel bounds on an (H, W) grid.

        Start coordinates are floored and end coordinates ceiled (with a 1e-6
        pixel tolerance against float noise, e.g. ``0.7 * 10``), so the pixel
        box covers the normalised box and is never empty.
        """
        eps = 1e-6
        top = min(int(math.floor(self.y0 * height + eps)), height - 1)
        left = min(int(math.floor(self.x0 * width + eps)), width - 1)
        bottom = max(int(math.ceil(self.y1 * height - eps)), top + 1)
        right = max(int(math.ceil(self.x1 * width - eps)), left + 1)
        return top, left, min(bottom, height), min(right, width)

    def pixel_area(self, height: int, width: int) -> int:
        top, left, bottom, right = self.to_pixels(height, width)
        return (bottom - top) * (right - left)

    @classmethod
    def from_pixels(cls, top: int, left: int, bottom: int, right: int, height: int, width: int) -> "Box":
        """Inverse of :meth:`to_pixels` for pixel-aligned boxes."""
        return cls(left / width, top / height, right / width, bottom / height)


@dataclass
class GazeState:
    """What a policy observes when choosing the glimpses of frame ``t``.

    Attributes
    ----------
    t:
        0-based position of the frame within the clip.
    frame_size:
        ``(H, W)`` of the full-resolution frame.
    context_logits:
        ``(C, h, w)`` float tensor: class logits of the low-resolution
        *context* pass over the whole frame (any ``h, w``; cell ``(i, j)``
        covers the normalised region ``[j/w, (j+1)/w) x [i/h, (i+1)/h)``).
        None for policies that do not use a context pass.
    frame:
        ``(3, H, W)`` float tensor in [0, 1]: the full-resolution frame, for
        policies that look at pixels (e.g. motion). Optional.
    budget:
        Number of glimpses to return for this frame (policies may return fewer).
    glimpse_size:
        ``(h_g, w_g)`` pixels each glimpse is resized to before the backbone,
        for policies that reason about glimpse resolution. Optional.
    previous_boxes:
        Boxes selected at frame ``t - 1`` (empty at the start of a clip).
    extras:
        Free-form side information (e.g. optical flow), keyed by name.
    """

    t: int
    frame_size: Tuple[int, int]
    context_logits: Optional["torch.Tensor"] = None
    frame: Optional["torch.Tensor"] = None
    budget: int = 1
    glimpse_size: Optional[Tuple[int, int]] = None
    previous_boxes: Sequence[Box] = ()
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GazePolicy(Protocol):
    """Chooses where to look.

    ``select`` returns at most ``state.budget`` boxes for frame ``state.t``.
    ``memory`` is the scene memory as read *before* this frame's glimpses
    (the runner passes it only when ``uses_memory`` is True; memory-free
    policies must accept and ignore it). ``reset`` is called at the start of
    every clip so stateful (e.g. recurrent / RL) policies can clear state.
    """

    #: True for canvas-aware policies that need :class:`MemoryState`.
    uses_memory: bool

    def reset(self) -> None: ...

    def select(self, state: GazeState, memory: Optional["MemoryState"] = None) -> List[Box]: ...
