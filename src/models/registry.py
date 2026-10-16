"""Build a backbone from a config mapping (torch).

    backbone:
      type: pidnet_s          # pidnet_s | pidnet_m | pidnet_l | bisenetv2 | dummy
      num_classes: 19
      weights: /path/or/https://url.pt   # required for real backbones
      sha256: null            # optional pin of the weights file
      cache_dir: null         # where URL downloads go (default: torch hub cache)
      strict: true            # fail if the checkpoint does not fit exactly
      allow_random_init: false  # explicit opt-in to run without weights (smoke tests only)
      # any other key is passed to the wrapper (e.g. mean, std, output_stride)

``class_colors: dataset_palette`` (dummy only) is resolved from ``context``
(``{"palette": (C, 3) uint8}``) so the generic code never hard-codes colours.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch.nn as nn

from .bisenetv2 import BiSeNetV2Backbone
from .dummy import DummyBackbone
from .pidnet import PIDNetBackbone

__all__ = ["build_backbone", "BACKBONES"]

BACKBONES = ("pidnet_s", "pidnet_m", "pidnet_l", "bisenetv2", "dummy")
_WEIGHT_KEYS = ("weights", "sha256", "cache_dir", "strict", "trust_pickle", "allow_random_init")


def build_backbone(spec: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> nn.Module:
    """Instantiate (and load weights into) the backbone described by ``spec``; eval mode."""
    spec = dict(spec)
    kind = spec.pop("type", None)
    opts = {k: spec.pop(k) for k in _WEIGHT_KEYS if k in spec}
    if kind in ("pidnet_s", "pidnet_m", "pidnet_l"):
        model: nn.Module = PIDNetBackbone(variant=kind[-1], **spec)
    elif kind == "bisenetv2":
        model = BiSeNetV2Backbone(**spec)
    elif kind == "dummy":
        if spec.get("class_colors") == "dataset_palette":
            if not context or context.get("palette") is None:
                raise ValueError("class_colors: dataset_palette needs the dataset palette in `context`")
            spec["class_colors"] = (context["palette"] / 255.0).tolist()
        model = DummyBackbone(**spec)
    else:
        raise ValueError(f"unknown backbone type {kind!r}; known: {BACKBONES}")

    if kind != "dummy":
        weights = opts.get("weights")
        if weights:
            model.load_weights(  # type: ignore[operator]
                str(weights),
                cache_dir=opts.get("cache_dir"),
                sha256=opts.get("sha256"),
                strict=bool(opts.get("strict", True)),
                trust_pickle=bool(opts.get("trust_pickle", False)),
            )
        elif opts.get("allow_random_init"):
            model.weights_info = {"source": "random-init (allow_random_init: true) -- predictions are meaningless"}
        else:
            raise ValueError(
                f"backbone {kind!r} needs pretrained weights: set backbone.weights to a local path or "
                "URL of the checkpoint (see README, 'Backbone weights'), or set "
                "allow_random_init: true for a smoke test whose numbers are meaningless."
            )
    return model.eval()


def describe_backbone(model: nn.Module) -> Dict[str, Any]:
    describe = getattr(model, "describe", None)
    if callable(describe):
        return describe()
    return {"type": type(model).__name__, "params": sum(p.numel() for p in model.parameters())}
