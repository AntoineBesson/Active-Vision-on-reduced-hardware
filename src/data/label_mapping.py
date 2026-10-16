"""Config-driven conversion of raw annotation maps into semantic train ids.

This module is dataset-agnostic: a dataset adapter (or its YAML config) states
*how its annotations are encoded* through a :class:`LabelMappingConfig`, and
:class:`SemanticLabelMapper` executes it. Supporting a new dataset means writing
a new config, not new code.

Pipeline (every stage is configurable)::

    raw array --encoding--> id map --id_rule--> category ids --category_to_train_id--> train ids

``encoding``
    ``"rgb"``  -- (H, W, >=3) uint8 PNG, ``id = R + 256*G + 256**2*B`` (COCO panoptic);
    ``"gray"`` -- (H, W) integer array, ``id = pixel value`` (8/16/32-bit PNG);
    ``"auto"`` -- ``"rgb"`` for >= 3 channels, ``"gray"`` for 2-D arrays.
``id_rule``
    ``"identity"``      -- category = id (plain semantic maps);
    ``"divisor"``       -- category = ``id // label_divisor`` if ``id >= label_divisor`` else ``id``
                           (instance-id maps where things are ``class * divisor + instance``);
    ``"segments_json"`` -- category looked up per annotation in a COCO-panoptic JSON
                           (``annotations[].segments_info[].{id, category_id}``); ids missing
                           from it -- including COCO's VOID id 0 -- are void.
``category_to_train_id``
    ``{category id: train id}``; unlisted categories become ``ignore_index``.
    ``None`` means identity on ``[0, num_classes)``.
``void_ids``
    raw ids that are void before any lookup.

Only numpy (+ json) is needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .base import label_dtype

__all__ = [
    "LabelMappingConfig",
    "SegmentsIndex",
    "SemanticLabelMapper",
    "decode_rgb_ids",
]

ENCODINGS = ("auto", "rgb", "gray")
ID_RULES = ("identity", "divisor", "segments_json")

# Ids below this bound are mapped through a dense lookup table (fast path);
# larger ids (e.g. 24-bit RGB panoptic ids) go through np.unique.
_DENSE_LUT_LIMIT = 1 << 16


def decode_rgb_ids(rgb: np.ndarray) -> np.ndarray:
    """``id = R + 256*G + 256**2*B`` for an (H, W, >=3) array -> (H, W) int64."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected an (H, W, >=3) array, got shape {rgb.shape}")
    rgb = rgb.astype(np.int64)
    return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]


@dataclass
class LabelMappingConfig:
    """How one annotation format maps to the train-id label space."""

    class_names: List[str]
    ignore_index: int = 255
    encoding: str = "auto"
    id_rule: str = "identity"
    label_divisor: int = 1000
    # Path of the COCO-panoptic JSON for id_rule == "segments_json". The adapter
    # resolves relative paths / "{split}" placeholders before building a mapper.
    segments_json: Optional[str] = None
    category_to_train_id: Optional[Dict[int, int]] = None
    void_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.class_names = [str(c) for c in self.class_names]
        if not self.class_names:
            raise ValueError("label mapping needs at least one class name")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}, got {self.encoding!r}")
        if self.id_rule not in ID_RULES:
            raise ValueError(f"id_rule must be one of {ID_RULES}, got {self.id_rule!r}")
        if int(self.label_divisor) <= 0:
            raise ValueError(f"label_divisor must be positive, got {self.label_divisor}")
        self.label_divisor = int(self.label_divisor)
        self.ignore_index = int(self.ignore_index)
        if self.id_rule == "segments_json" and not self.segments_json:
            raise ValueError("id_rule 'segments_json' requires segments_json (path to a COCO-panoptic JSON)")
        if self.category_to_train_id is not None:
            table = {int(k): int(v) for k, v in dict(self.category_to_train_id).items()}
            for cat, tid in table.items():
                if not (0 <= tid < len(self.class_names) or tid == self.ignore_index):
                    raise ValueError(
                        f"category {cat} maps to train id {tid}, outside [0, {len(self.class_names)}) "
                        f"and not ignore_index={self.ignore_index}"
                    )
                if cat < 0:
                    raise ValueError(f"category ids must be >= 0, got {cat}")
            self.category_to_train_id = table
        self.void_ids = [int(v) for v in self.void_ids]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelMappingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown label_mapping keys: {sorted(unknown)} (known: {sorted(known)})")
        return cls(**dict(data))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SegmentsIndex:
    """Per-annotation ``{segment id: category id}`` tables from a COCO-panoptic JSON.

    Annotations are found by their PNG ``file_name`` (with or without extension)
    or their ``image_id``.
    """

    def __init__(self, path: "str | os.PathLike"):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"COCO-panoptic segments JSON not found: {self.path} (needed by id_rule='segments_json')"
            )
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.categories: List[Dict[str, Any]] = list(data.get("categories", []))
        self._tables: Dict[str, Dict[int, int]] = {}
        for ann in data.get("annotations", []):
            table = {int(s["id"]): int(s["category_id"]) for s in ann.get("segments_info", [])}
            for key in self._keys(ann):
                self._tables.setdefault(key, table)

    @staticmethod
    def _keys(ann: Mapping[str, Any]) -> List[str]:
        keys = []
        if ann.get("file_name"):
            name = str(ann["file_name"])
            keys += [name, os.path.splitext(name)[0]]
        if ann.get("image_id") is not None:
            keys.append(str(ann["image_id"]))
        return keys

    def __len__(self) -> int:
        return len({id(t) for t in self._tables.values()})

    def lookup(self, annotation_name: str) -> Dict[int, int]:
        for key in (annotation_name, os.path.splitext(annotation_name)[0]):
            if key in self._tables:
                return self._tables[key]
        raise KeyError(f"annotation {annotation_name!r} has no entry in {self.path}")


class SemanticLabelMapper:
    """Applies a :class:`LabelMappingConfig` to raw annotation arrays."""

    def __init__(self, config: LabelMappingConfig, segments: Optional[SegmentsIndex] = None):
        self.config = config
        self._segments = segments
        cfg = config
        self.out_dtype = label_dtype(cfg.num_classes, cfg.ignore_index)
        # category -> train id lookup table; -1 (void) and out-of-range -> ignore.
        if cfg.category_to_train_id is None:
            lut = np.arange(cfg.num_classes, dtype=np.int64)
        else:
            size = max(cfg.category_to_train_id) + 1 if cfg.category_to_train_id else 0
            lut = np.full(size, cfg.ignore_index, dtype=np.int64)
            for cat, tid in cfg.category_to_train_id.items():
                lut[cat] = tid
        self._train_lut = lut
        self._void = np.array(sorted(set(cfg.void_ids)), dtype=np.int64)

    @property
    def segments(self) -> SegmentsIndex:
        if self._segments is None:
            self._segments = SegmentsIndex(self.config.segments_json)  # type: ignore[arg-type]
        return self._segments

    # -- Stages ---------------------------------------------------------------- #
    def decode_ids(self, raw: np.ndarray) -> np.ndarray:
        """Raw annotation array -> (H, W) int64 id map, per ``encoding``."""
        raw = np.asarray(raw)
        enc = self.config.encoding
        if enc == "auto":
            enc = "rgb" if (raw.ndim == 3 and raw.shape[2] >= 3) else "gray"
        if enc == "rgb":
            return decode_rgb_ids(raw)
        if raw.ndim == 3 and raw.shape[2] == 1:
            raw = raw[..., 0]
        if raw.ndim != 2:
            raise ValueError(f"encoding 'gray' expects a 2-D array, got shape {raw.shape}")
        return raw.astype(np.int64)

    def categories_for_ids(
        self, ids: np.ndarray, annotation_name: Optional[str] = None
    ) -> np.ndarray:
        """Category id for each value in ``ids`` (any shape); -1 marks void."""
        ids = np.asarray(ids, dtype=np.int64)
        rule = self.config.id_rule
        if rule == "identity":
            cats = ids.copy()
        elif rule == "divisor":
            d = self.config.label_divisor
            cats = np.where(ids >= d, ids // d, ids)
        else:  # segments_json
            if annotation_name is None:
                raise ValueError("id_rule 'segments_json' needs the annotation file name")
            table = self.segments.lookup(annotation_name)
            flat = ids.reshape(-1)
            uniq, inv = np.unique(flat, return_inverse=True)
            cats_u = np.array([table.get(int(u), -1) for u in uniq], dtype=np.int64)
            cats = cats_u[inv].reshape(ids.shape)
        if self._void.size:
            cats = np.where(np.isin(ids, self._void), -1, cats)
        return np.where(ids < 0, -1, cats)

    def train_ids_for_categories(self, cats: np.ndarray) -> np.ndarray:
        cats = np.asarray(cats, dtype=np.int64)
        lut = self._train_lut
        inside = (cats >= 0) & (cats < len(lut))
        out = np.full(cats.shape, self.config.ignore_index, dtype=np.int64)
        out[inside] = lut[cats[inside]]
        return out

    def train_ids_for_ids(self, ids: np.ndarray, annotation_name: Optional[str] = None) -> np.ndarray:
        return self.train_ids_for_categories(self.categories_for_ids(ids, annotation_name))

    # -- Full pipeline --------------------------------------------------------- #
    def __call__(self, raw: np.ndarray, annotation_name: Optional[str] = None) -> np.ndarray:
        """Raw annotation array -> (H, W) train-id map (``uint8`` for ignore 255)."""
        ids = self.decode_ids(raw)
        if ids.size == 0:
            return ids.astype(self.out_dtype)
        lo, hi = int(ids.min()), int(ids.max())
        if lo >= 0 and hi < _DENSE_LUT_LIMIT:
            lut = self.train_ids_for_ids(np.arange(hi + 1, dtype=np.int64), annotation_name)
            return lut[ids].astype(self.out_dtype)
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        mapped = self.train_ids_for_ids(uniq, annotation_name)
        return mapped[inv].reshape(ids.shape).astype(self.out_dtype)

    def explain(
        self, ids: Sequence[int], annotation_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """One row per raw id: its category, train id and class name (diagnostics)."""
        ids_arr = np.unique(np.asarray(list(ids), dtype=np.int64))
        cats = self.categories_for_ids(ids_arr, annotation_name)
        tids = self.train_ids_for_categories(cats)
        names = self.config.class_names
        rows = []
        for i, c, t in zip(ids_arr.tolist(), cats.tolist(), tids.tolist()):
            rows.append(
                {
                    "id": int(i),
                    "category": None if c < 0 else int(c),
                    "train_id": int(t),
                    "class": names[t] if 0 <= t < len(names) else "<ignore>",
                }
            )
        return rows
