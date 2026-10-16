"""Data loading and dataset utilities.

* :mod:`data.base` -- the dataset-agnostic :class:`VideoSegDataset` interface
  every downstream component uses, plus a generic in-memory implementation.
* :mod:`data.label_mapping` -- config-driven raw annotation -> train-id mapping.
* :mod:`data.cityscapes_vps` -- the Cityscapes-VPS adapter (the only place that
  knows Cityscapes conventions).
* :mod:`data.subclips`, :mod:`data.splits` -- sub-clip windows and clip-level
  (never frame-level) splits.
* :mod:`data.torch_dataset` -- torch Dataset/DataLoader wrappers (needs torch).

Importing this package needs numpy only; torch wrappers are imported explicitly
from :mod:`data.torch_dataset`.
"""

from .base import InMemoryVideoSeg, VideoClip, VideoSegDataset, default_palette, label_dtype

__all__ = ["VideoSegDataset", "VideoClip", "InMemoryVideoSeg", "default_palette", "label_dtype"]
