"""Dataset-agnostic interface for sparsely-labelled video semantic segmentation.

Everything downstream of the data layer (the torch pipeline, foveation
preprocessing, gaze policies, scene memory and the evaluation harness) talks to
datasets *only* through :class:`VideoSegDataset`. Moving the project to another
video segmentation dataset therefore means writing one adapter; nothing else
changes. Dataset-specific knowledge (directory layout, file naming, label
encodings, class lists, colours) lives in the adapter (e.g.
:mod:`data.cityscapes_vps`) and its YAML config, never in this module.

Shared conventions
------------------
A dataset is an indexable collection of **clips**. Clip ``i`` is an ordered
sequence of ``T_i`` frames, a subset of which (possibly none) carry a dense
semantic annotation.

=================  ==================  =========================================
name               shape / type        meaning
=================  ==================  =========================================
``frames``         (T, H, W, 3) uint8  RGB frames in temporal order
``labels``         (T, H, W) integer   train ids in ``[0, num_classes)`` or
                                       ``ignore_index``; unlabelled frames are
                                       filled entirely with ``ignore_index``
``labeled_mask``   (T,) bool           True where the frame has an annotation
``frame_indices``  (T,) int64          dataset-native frame numbers (bookkeeping)
``class_names``    tuple[str]          ``num_classes`` names; index == train id
``ignore_index``   int                 excluded from losses and metrics
=================  ==================  =========================================

Only numpy is required here; torch wrappers live in :mod:`data.torch_dataset`.
"""

from __future__ import annotations

import abc
import colorsys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "VideoClip",
    "VideoSegDataset",
    "InMemoryVideoSeg",
    "default_palette",
    "label_dtype",
]


def label_dtype(num_classes: int, ignore_index: int) -> np.dtype:
    """Smallest integer dtype that holds every train id and ``ignore_index``.

    With the usual ``ignore_index=255`` and fewer than 256 classes this is
    ``uint8``, which keeps a 30-frame 1024x2048 label clip at ~63 MB.
    """
    lo = min(0, int(ignore_index))
    hi = max(int(num_classes) - 1, int(ignore_index))
    for dt in (np.uint8, np.int16, np.int32):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dt)
    return np.dtype(np.int64)


def default_palette(num_classes: int) -> np.ndarray:
    """Deterministic, well-spread (num_classes, 3) uint8 colours for visualisation."""
    colours = []
    for i in range(num_classes):
        hue = (i * 0.618033988749895) % 1.0  # golden-ratio hue spacing
        sat = 0.65 + 0.35 * ((i // 7) % 2)
        val = 0.95 - 0.25 * ((i // 3) % 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colours.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colours, dtype=np.uint8).reshape(num_classes, 3)


# --------------------------------------------------------------------------- #
# Clip container
# --------------------------------------------------------------------------- #
@dataclass
class VideoClip:
    """A loaded (sub-)clip. See the module docstring for the conventions."""

    clip_id: str
    frames: np.ndarray  # (T, H, W, 3) uint8
    labels: np.ndarray  # (T, H, W) train ids; ignore_index on unlabelled frames
    labeled_mask: np.ndarray  # (T,) bool
    frame_indices: np.ndarray  # (T,) int64
    class_names: Tuple[str, ...]
    ignore_index: int
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.labeled_mask.shape[0])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(H, W) of every frame in the clip."""
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    @property
    def labeled_indices(self) -> np.ndarray:
        """Positions (0-based, within this clip) of the labelled frames."""
        return np.flatnonzero(self.labeled_mask)

    def validate(self) -> None:
        """Cheap shape/dtype checks (no pass over the pixels)."""
        t = self.num_frames
        if self.labeled_mask.shape != (t,) or self.labeled_mask.dtype != bool:
            raise ValueError(f"labeled_mask must be a (T,) bool array, got {self.labeled_mask.shape}")
        if self.frames.ndim != 4 or self.frames.shape[0] != t or self.frames.shape[3] != 3:
            raise ValueError(f"frames must be (T,H,W,3) with T={t}, got {self.frames.shape}")
        if self.frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got {self.frames.dtype}")
        if self.labels.shape != self.frames.shape[:3]:
            raise ValueError(
                f"labels must be (T,H,W)={self.frames.shape[:3]}, got {self.labels.shape}"
            )
        if self.frame_indices.shape != (t,):
            raise ValueError(f"frame_indices must be (T,), got {self.frame_indices.shape}")


# --------------------------------------------------------------------------- #
# Abstract dataset
# --------------------------------------------------------------------------- #
class VideoSegDataset(abc.ABC):
    """Abstract clip-level dataset for (sparsely labelled) video semantic segmentation.

    Adapters implement the small set of abstract methods below; the concrete
    helpers (:meth:`get_clip`, iteration, :meth:`describe`) are shared.
    Metadata methods (:meth:`clip_id`, :meth:`labeled_mask`, ...) must not read
    pixels, so samplers and split utilities stay cheap.
    """

    #: Short identifier recorded in result files; adapters override it.
    name: str = "video_seg"

    # -- Label space ---------------------------------------------------------- #
    @property
    @abc.abstractmethod
    def class_names(self) -> Tuple[str, ...]:
        """Names of the ``num_classes`` train ids, in id order."""

    @property
    @abc.abstractmethod
    def ignore_index(self) -> int:
        """Label value that is excluded from losses and metrics."""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def palette(self) -> np.ndarray:
        """(num_classes, 3) uint8 colours for visualisation (adapters may override)."""
        return default_palette(self.num_classes)

    # -- Clip metadata (no pixel access) ------------------------------------ #
    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of clips."""

    @abc.abstractmethod
    def clip_id(self, index: int) -> str:
        """Stable, unique identifier of clip ``index``."""

    @abc.abstractmethod
    def labeled_mask(self, index: int) -> np.ndarray:
        """(T,) bool: which frames of clip ``index`` carry an annotation."""

    def num_frames(self, index: int) -> int:
        return int(len(self.labeled_mask(index)))

    def frame_indices(self, index: int) -> np.ndarray:
        """(T,) int64 dataset-native frame numbers; defaults to ``0..T-1``."""
        return np.arange(self.num_frames(index), dtype=np.int64)

    def clip_group(self, index: int) -> str:
        """Clips sharing a group (e.g. one continuous recording) are never split
        across train/val by :func:`data.splits.split_by_clip`. Defaults to the clip id."""
        return self.clip_id(index)

    # -- Pixels ---------------------------------------------------------------- #
    @abc.abstractmethod
    def load_frame(self, index: int, t: int) -> np.ndarray:
        """Frame ``t`` (0-based position) of clip ``index`` as (H, W, 3) uint8 RGB."""

    @abc.abstractmethod
    def load_label(self, index: int, t: int) -> Optional[np.ndarray]:
        """(H, W) train-id map of frame ``t`` of clip ``index``, or None if unlabelled."""

    # -- Optional hooks ------------------------------------------------------- #
    def diagnostics(self) -> Dict[str, Any]:
        """Adapter-specific indexing statistics (used by scripts/diagnose_real_data.py)."""
        return {}

    # -- Shared helpers ------------------------------------------------------- #
    def _check_index(self, index: int) -> int:
        n = len(self)
        if not -n <= index < n:
            raise IndexError(f"clip index {index} out of range for {n} clips")
        return index % n

    @property
    def clip_ids(self) -> List[str]:
        return [self.clip_id(i) for i in range(len(self))]

    def num_labeled_frames(self) -> int:
        return int(sum(int(self.labeled_mask(i).sum()) for i in range(len(self))))

    def get_clip(
        self,
        index: int,
        frames: Optional[Sequence[int]] = None,
        load_labels: bool = True,
    ) -> VideoClip:
        """Load clip ``index`` (or only the frame positions in ``frames``).

        ``frames`` are 0-based positions within the clip, in the order they
        should appear in the result (sub-clip samplers pass increasing runs).
        With ``load_labels=False`` the label tensor is all ``ignore_index``
        (but ``labeled_mask`` still reports which frames *have* annotations).
        """
        index = self._check_index(index)
        full_mask = np.asarray(self.labeled_mask(index), dtype=bool)
        n_frames = len(full_mask)
        positions = list(range(n_frames)) if frames is None else [int(t) for t in frames]
        for t in positions:
            if not 0 <= t < n_frames:
                raise IndexError(f"frame {t} out of range for clip {self.clip_id(index)} ({n_frames} frames)")
        if not positions:
            raise ValueError("get_clip needs at least one frame")

        images = [np.asarray(self.load_frame(index, t)) for t in positions]
        shapes = {im.shape for im in images}
        if len(shapes) != 1:
            raise ValueError(
                f"clip {self.clip_id(index)}: frames have differing shapes {sorted(shapes)}; "
                "the adapter must resize them to a common size"
            )
        frames_arr = np.stack(images, axis=0)
        height, width = frames_arr.shape[1:3]

        dtype = label_dtype(self.num_classes, self.ignore_index)
        labels = np.full((len(positions), height, width), self.ignore_index, dtype=dtype)
        mask = full_mask[positions]
        if load_labels:
            for k, t in enumerate(positions):
                if not mask[k]:
                    continue
                lbl = self.load_label(index, t)
                if lbl is None:
                    raise RuntimeError(
                        f"clip {self.clip_id(index)} frame {t} is marked labelled but load_label returned None"
                    )
                if lbl.shape != (height, width):
                    raise ValueError(
                        f"clip {self.clip_id(index)} frame {t}: label shape {lbl.shape} != frame shape {(height, width)}"
                    )
                labels[k] = lbl

        clip = VideoClip(
            clip_id=self.clip_id(index),
            frames=frames_arr,
            labels=labels,
            labeled_mask=mask,
            frame_indices=np.asarray(self.frame_indices(index), dtype=np.int64)[positions],
            class_names=tuple(self.class_names),
            ignore_index=int(self.ignore_index),
            meta={"index": index, "positions": positions, "group": self.clip_group(index)},
        )
        clip.validate()
        return clip

    def __getitem__(self, index: int) -> VideoClip:
        return self.get_clip(index)

    def __iter__(self) -> Iterator[VideoClip]:
        for i in range(len(self)):
            yield self.get_clip(i)

    def describe(self) -> Dict[str, Any]:
        """JSON-serialisable summary recorded alongside results."""
        n_frames = [self.num_frames(i) for i in range(len(self))]
        return {
            "name": self.name,
            "num_clips": len(self),
            "num_frames": int(sum(n_frames)),
            "num_labeled_frames": self.num_labeled_frames(),
            "num_classes": self.num_classes,
            "ignore_index": int(self.ignore_index),
            "class_names": list(self.class_names),
        }


# --------------------------------------------------------------------------- #
# Generic in-memory implementation
# --------------------------------------------------------------------------- #
class InMemoryVideoSeg(VideoSegDataset):
    """Clips held in memory: synthetic data, tests, or small pre-loaded datasets.

    ``clips`` is a sequence of mappings with keys:

    * ``clip_id`` (str), ``frames`` ((T,H,W,3) uint8 array or list of (H,W,3)),
    * ``labels``: length-T sequence of (H,W) integer arrays or None (unlabelled),
    * optional ``frame_indices`` ((T,) ints) and ``group`` (str).
    """

    def __init__(
        self,
        clips: Sequence[Mapping[str, Any]],
        class_names: Sequence[str],
        ignore_index: int = 255,
        name: str = "in_memory",
        palette: Optional[np.ndarray] = None,
    ):
        self.name = name
        self._class_names = tuple(class_names)
        self._ignore_index = int(ignore_index)
        self._palette = None if palette is None else np.asarray(palette, dtype=np.uint8)
        self._clips: List[Dict[str, Any]] = []
        seen = set()
        for c in clips:
            frames = [np.asarray(f, dtype=np.uint8) for f in c["frames"]]
            labels = list(c["labels"])
            if len(labels) != len(frames):
                raise ValueError(f"clip {c['clip_id']}: {len(frames)} frames but {len(labels)} label entries")
            if c["clip_id"] in seen:
                raise ValueError(f"duplicate clip id {c['clip_id']!r}")
            seen.add(c["clip_id"])
            idx = c.get("frame_indices")
            self._clips.append(
                {
                    "clip_id": str(c["clip_id"]),
                    "frames": frames,
                    "labels": [None if lb is None else np.asarray(lb) for lb in labels],
                    "frame_indices": np.arange(len(frames), dtype=np.int64)
                    if idx is None
                    else np.asarray(idx, dtype=np.int64),
                    "group": str(c.get("group", c["clip_id"])),
                }
            )

    @property
    def class_names(self) -> Tuple[str, ...]:
        return self._class_names

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    @property
    def palette(self) -> np.ndarray:
        return default_palette(self.num_classes) if self._palette is None else self._palette

    def __len__(self) -> int:
        return len(self._clips)

    def clip_id(self, index: int) -> str:
        return self._clips[index]["clip_id"]

    def labeled_mask(self, index: int) -> np.ndarray:
        return np.array([lb is not None for lb in self._clips[index]["labels"]], dtype=bool)

    def frame_indices(self, index: int) -> np.ndarray:
        return self._clips[index]["frame_indices"]

    def clip_group(self, index: int) -> str:
        return self._clips[index]["group"]

    def load_frame(self, index: int, t: int) -> np.ndarray:
        return self._clips[index]["frames"][t]

    def load_label(self, index: int, t: int) -> Optional[np.ndarray]:
        return self._clips[index]["labels"][t]
