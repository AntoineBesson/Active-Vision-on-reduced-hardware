"""Model-side interfaces: segmentation backbone and persistent scene memory.

Contracts only (``typing.Protocol``, structural): any object with the right
attributes/methods conforms -- torch ``nn.Module`` subclasses need no special
base class. Shapes:

* :class:`SegmentationBackbone` is **batched**: ``(B, ...)``;
* :class:`SceneMemory` / :class:`MemoryState` are **per sample** (one clip,
  no batch dimension), like :mod:`policies.interfaces`.

Boxes use the normalised full-frame convention of :class:`policies.interfaces.Box`.
Torch is only referenced in annotations, so this module imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

try:  # Python >= 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from policies.interfaces import Box

if TYPE_CHECKING:
    import torch

__all__ = ["SegmentationBackbone", "MemoryState", "SceneMemory", "Box"]


@runtime_checkable
class SegmentationBackbone(Protocol):
    """An existing per-image segmentation network, wrapped for this project.

    ``forward(images) -> logits``

    images
        ``(B, 3, H, W)`` float tensor, RGB, values in [0, 1]. The backbone
        applies its own normalisation (the mean/std of its pre-training)
        internally, so callers never need to know it. Any ``H, W >= 1`` must
        be accepted; implementations pad internally when the architecture
        needs a size multiple.
    logits
        ``(B, num_classes, ceil(H / s), ceil(W / s))`` float tensor with
        ``s = output_stride``. Cell ``(i, j)`` covers input pixels
        ``[i*s, (i+1)*s) x [j*s, (j+1)*s)`` of the input padded at the
        bottom/right to a multiple of ``s``. To bring them back to
        ``(H, W)``: bilinearly interpolate to ``(ceil(H/s)*s, ceil(W/s)*s)``
        with ``align_corners``, then crop to ``(H, W)``.

    Attributes: ``num_classes`` (C), ``output_stride`` (s >= 1),
    ``align_corners`` (interpolation the model's own evaluation uses when
    upsampling its logits).
    """

    num_classes: int
    output_stride: int
    align_corners: bool

    def forward(self, images: "torch.Tensor") -> "torch.Tensor": ...

    def __call__(self, images: "torch.Tensor") -> "torch.Tensor": ...


@dataclass
class MemoryState:
    """Snapshot of a persistent scene memory (the "canvas").

    Attributes
    ----------
    canvas:
        ``(D, Hc, Wc)`` float tensor: what the memory currently believes about
        the whole frame, at canvas resolution ``(Hc, Wc)`` in full-frame
        coordinates (e.g. ``D = num_classes`` accumulated logits, or features).
    confidence:
        ``(1, Hc, Wc)`` in [0, 1]: how much each cell can be trusted. Optional.
    age:
        ``(1, Hc, Wc)``: frames elapsed since each cell was last written. Optional.
    t:
        Frame index of the most recent update (-1 before any update).
    extras:
        Implementation-specific state, keyed by name.
    """

    canvas: "torch.Tensor"
    confidence: Optional["torch.Tensor"] = None
    age: Optional["torch.Tensor"] = None
    t: int = -1
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SceneMemory(Protocol):
    """Persistent memory written by glimpses and read by policies / recomposition.

    ``reset(frame_size)``
        Start a new clip whose frames are ``(H, W)``; returns the empty state.
    ``update(glimpse_output, box, *, t=None)``
        Write one glimpse: ``glimpse_output`` is the ``(C, h_g, w_g)`` output
        of the backbone (logits or features) for the crop ``box`` (normalised
        full-frame coordinates). ``t`` is the frame index, letting the memory
        age stale cells when a new frame starts; memories that do not need it
        ignore it. Returns the updated state.
    ``read()``
        Current :class:`MemoryState`, without side effects.

    A full-frame prediction can be recomposed by upsampling ``read().canvas``
    to ``(H, W)``.
    """

    def reset(self, frame_size: Tuple[int, int]) -> MemoryState: ...

    def update(self, glimpse_output: "torch.Tensor", box: Box, *, t: Optional[int] = None) -> MemoryState: ...

    def read(self) -> MemoryState: ...
