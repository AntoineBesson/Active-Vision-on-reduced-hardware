"""Segmentation backbones and scene memory.

* :mod:`models.interfaces` -- the ``SegmentationBackbone`` and ``SceneMemory``
  contracts (``typing.Protocol``) plus ``MemoryState``. Imports without torch.
* :mod:`models.pidnet` -- PIDNet-S/M/L (official architecture, ported).
* :mod:`models.bisenetv2` -- BiSeNetV2 (CoinCheung implementation, ported).
* :mod:`models.dummy` -- tiny deterministic stand-in for tests/smoke runs.
* :mod:`models.weights` -- weights from a path or URL, with clear failures.
* :mod:`models.registry` -- ``build_backbone(config)``.
* :mod:`models.utils` -- padding / logits upsampling helpers.

We do not build a new segmentation model: existing backbones are wrapped
behind ``SegmentationBackbone`` and a persistent memory (canvas) is added on
top of them. Everything except ``models.interfaces`` needs torch.
"""
