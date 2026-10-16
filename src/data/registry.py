"""Build a :class:`VideoSegDataset` from a config mapping (numpy only).

    dataset:
      type: cityscapes_vps                 # short name, or "package.module:Class"
      config: configs/cityscapes_vps.yaml  # adapter-specific YAML (relative paths: base_dir)
      split: val
      root: null                           # optional; else the adapter's own default/env var
      overrides: {}                        # adapter config overrides (highest precedence)

The adapter class does the rest through its ``from_spec(spec, base_dir)``
classmethod, so this module holds no dataset-specific knowledge beyond the
short-name table.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .base import VideoSegDataset

__all__ = ["DATASETS", "build_dataset", "resolve_dataset_class"]

#: short name -> "module:Class" (adapters register here)
DATASETS: Dict[str, str] = {
    "cityscapes_vps": "data.cityscapes_vps:CityscapesVPS",
}


def resolve_dataset_class(kind: str):
    target = DATASETS.get(kind, kind)
    if ":" not in target:
        raise ValueError(f"unknown dataset type {kind!r}; known: {sorted(DATASETS)} or 'module:Class'")
    module_name, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, VideoSegDataset)):
        raise TypeError(f"{target} is not a VideoSegDataset subclass")
    if not hasattr(cls, "from_spec"):
        raise TypeError(f"{target} has no from_spec(spec, base_dir) classmethod")
    return cls


def build_dataset(spec: Mapping[str, Any], base_dir: Optional[Path] = None) -> VideoSegDataset:
    """Instantiate the dataset adapter described by ``spec``."""
    spec = dict(spec)
    kind = spec.get("type")
    if not kind:
        raise ValueError("dataset spec needs a 'type'")
    return resolve_dataset_class(str(kind)).from_spec(spec, base_dir=Path(base_dir) if base_dir else Path.cwd())
