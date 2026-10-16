"""Segmentation backbones and scene memory.

* :mod:`models.interfaces` -- the ``SegmentationBackbone`` and ``SceneMemory``
  contracts (``typing.Protocol``) plus ``MemoryState``. Imports without torch.

We do not build a new segmentation model: existing backbones are wrapped
behind ``SegmentationBackbone`` and a persistent memory (canvas) is added on
top of them.
"""
