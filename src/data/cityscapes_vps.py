"""Cityscapes-VPS adapter: the ONLY module that knows Cityscapes(-VPS) conventions.

:class:`CityscapesVPS` implements the dataset-agnostic
:class:`data.base.VideoSegDataset` interface (frames, semantic train-id labels,
``labeled_mask``, class names, ignore index). Downstream code (torch pipeline,
policies, evaluation) only sees that interface. Every dataset-specific choice
below is a field of :class:`CityscapesVPSConfig`, set from YAML
(``configs/cityscapes_vps.yaml`` holds the official layout).

==============================================================================
FORMAT: VERIFIED vs ASSUMED
==============================================================================
Checked against the official VPSNet release (github.com/mcahny/vps, master):
docs/DATASET.md, prepare_data/{fetch_city_images, merge_datasets,
create_panoptic_labels, create_panoptic_video_labels, city_default}.py,
tools/eval_vpq.py, tools/dataset/cityscapes_vps.py, configs/cityscapes/
fusetrack.py. No real files were available when this was written:
run ``scripts/diagnose_real_data.py`` on the real data before trusting it.

Verified from the official code/docs:

* ``cityscapes-vps-dataset-1.0.zip`` ships annotations + JSON only. Frames are
  copied from Cityscapes ``leftImg8bit_sequence`` (``val/`` part only: all 500
  VPS videos are Cityscapes *val* sequences) by ``fetch_city_images.py``; the
  20th-frame labels come from Cityscapes ``gtFine/val``. Splits: 400/100/100
  videos = 2400/600/600 labelled frames; labels at the 5th, 10th, ..., 30th
  frame of each 30-frame video, i.e. clip positions 4, 9, 14, 19, 24, 29
  (``pred[(20 // 5)::5]`` in tools/dataset/cityscapes_vps.py).
* Layout after the official preparation (``<root>`` = ``data/cityscapes_vps``)::

      <root>/panoptic_gt_val_city_vps.json   # COCO-panoptic GT of val/panoptic_video
      <root>/{im_all_info,panoptic_im}_<split>_city_vps.json, instances_*_rle.json
      <root>/<split>/img/        # labelled frames only (train, val, test)
      <root>/<split>/img_all/    # ALL frames (val, test only -- no train/img_all)
      <root>/<split>/labelmap/       # train, val
      <root>/<split>/panoptic_inst/  # train, val
      <root>/val/panoptic_video/     # val only

* File names (``vid``: 4-digit VPS video id, ``fid``: 4-digit field whose
  meaning is not documented, then the Cityscapes ``city_seq_frame`` name)::

      img/        {vid}_{fid}_{city}_{seq:06d}_{frame:06d}_{newImg8bit|leftImg8bit}.png
      img_all/    {city}_{seq:06d}_{frame:06d}_leftImg8bit.png     # NO clip prefix
      annotations {vid}_{fid}_{city}_{seq:06d}_{frame:06d}_{final_mask|gtFine_color}.png

  ``leftImg8bit`` / ``gtFine_color`` mark the original Cityscapes 20th frame,
  ``newImg8bit`` / ``final_mask`` the frames annotated by VPS. ``img_all``
  keeps the original sequence names (``fetch_city_images.py`` copies
  ``leftImg8bit_sequence/val/<city>/<file_name>`` unchanged), so clips must
  be rebuilt from the Cityscapes ``city_seq`` + consecutive frame numbers.
* Category ids == Cityscapes train ids 0..18 (``city_default.py``: ``id`` ==
  ``trainid``; 11..18 are things). Label encodings:

  - ``labelmap/``: single channel, value = train id, 255 = void.
  - ``panoptic_inst/``: single channel, stuff = train id (0..10), things =
    ``train_id * 1000 + obj`` (``obj`` may be 0), 255 = void *and* crowd.
  - ``panoptic_video/``: COCO-panoptic RGB, ``id = R + 256 G + 256^2 B``. Ids
    are panopticapi ``IdGenerator`` colours (stuff: the class colour; things:
    the class colour or a random jitter of it), consistent within a video;
    0 = void. The class of a segment is ONLY in the JSON
    (``annotations[].file_name`` -> ``segments_info[].category_id``): it
    cannot be read from the colour (jittered thing colours of different
    classes overlap, e.g. bus (0,60,100) vs train (0,80,100)).
  - Crowd thing regions keep their class in ``labelmap/`` but are void in
    ``panoptic_inst/`` and ``panoptic_video/``.

Assumptions (each one is checked by ``scripts/diagnose_real_data.py``):

* A1. Two VPS videos never occupy consecutive frame numbers of the same
  Cityscapes sequence, so ``img_all`` runs of consecutive frames are clips.
  If a run nevertheless contains annotations of several videos it is split
  (with a warning) so that each video ends on its last labelled frame.
* A2. The ``fid`` field is ignored; frames are ordered by the Cityscapes frame
  number, which is monotonic in time by Cityscapes' naming convention.
* A3. Annotation PNGs may be 8-, 16- or 32-bit; values are read unchanged.

Legacy layout: the dataclass *defaults* still describe the earlier, unverified
guess (``{clip}_{frame}_...`` prefixed names inside ``img_all``) because the
original synthetic fixture and its tests use it. On the real data that guess
finds no clip at all (``img_all`` names have no prefix); use
``configs/cityscapes_vps.yaml`` instead.

This module has NO download logic (the data is license-gated).
Dependencies: numpy + pillow (+ pyyaml for :meth:`CityscapesVPSConfig.from_yaml`);
torch is imported lazily by :func:`clip_to_tensors` only.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import warnings
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .base import VideoSegDataset
from .label_mapping import LabelMappingConfig, SemanticLabelMapper

__all__ = [
    "CITYSCAPES_TRAIN_CLASSES",
    "CITYSCAPES_PALETTE",
    "IGNORE_INDEX",
    "VPS_ANNOTATION_FORMATS",
    "CityscapesVPSConfig",
    "FrameRecord",
    "ClipRecord",
    "CityscapesVPS",
    "decode_panoptic",
    "clip_to_tensors",
]

# Environment variable that points at the local dataset root.
ROOT_ENV_VAR = "CITYSCAPES_VPS_ROOT"

#: The 19 Cityscapes evaluation classes, in train-id order (cityscapesScripts
#: helpers/labels.py: the labels with ``trainId`` 0..18).
CITYSCAPES_TRAIN_CLASSES: Tuple[str, ...] = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)

#: Official Cityscapes colours of the 19 train ids (same source, same order).
CITYSCAPES_PALETTE = np.array(
    [
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
        (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
        (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
        (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
        (0, 0, 230), (119, 11, 32),
    ],
    dtype=np.uint8,
)

#: Cityscapes convention for "void / not evaluated".
IGNORE_INDEX = 255

#: Label encodings of the annotation folders written by the official VPSNet
#: preparation scripts (see module docstring). Used when the config gives no
#: explicit ``label_mapping``; an unknown folder name then raises at load time
#: rather than guessing. VPS category ids already are train ids, so no
#: ``category_to_train_id`` table is needed.
VPS_ANNOTATION_FORMATS: Dict[str, Dict[str, Any]] = {
    "labelmap": {"encoding": "gray", "id_rule": "identity"},
    "panoptic_inst": {"encoding": "gray", "id_rule": "divisor", "label_divisor": 1000},
    "panoptic_video": {
        "encoding": "rgb",
        "id_rule": "segments_json",
        "segments_json": "panoptic_gt_{split}_city_vps.json",
    },
}

#: Official side-car JSONs cross-checked by :meth:`CityscapesVPS.diagnostics`.
VPS_INFO_JSONS = (
    "im_all_info_{split}_city_vps.json",
    "panoptic_im_{split}_city_vps.json",
    "panoptic_gt_{split}_city_vps.json",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CityscapesVPSConfig:
    """Where the data is and how to parse its filenames.

    Every field is a knob you can retarget at the *real* dataset without
    touching the loader code. ``root`` may be left ``None`` and supplied via the
    ``CITYSCAPES_VPS_ROOT`` environment variable (see :meth:`resolve_root`).
    Defaults describe the legacy prefixed layout of the unit-test fixture; the
    official layout lives in ``configs/cityscapes_vps.yaml``.
    """

    root: Optional[str] = None
    split: str = "val"  # one of {"train", "val", "test"}

    # Sub-directory names, relative to <root>/<split>/. annotation_subdir=None
    # means "no annotations" (e.g. the test split).
    image_subdir: str = "img_all"
    annotation_subdir: Optional[str] = "panoptic_video"

    # Filename globbing / parsing.
    image_glob: str = "*.png"
    # Regex applied to the image filename. Named groups:
    #   frame    (required) integer used to order frames within a clip;
    #   clip     clip id -> frames are grouped by it;
    #   sequence used instead of ``clip``: frames are grouped by sequence and
    #            split into clips wherever the frame number jumps by more than
    #            ``max_frame_gap`` (Cityscapes img_all names carry no clip id);
    #   key      join key with the annotation regex (default: the "stem").
    filename_regex: str = r"^(?P<clip>\d{4})_(?P<frame>\d{4})_"
    # Image-filename suffix(es) stripped (before the extension) to get the
    # "stem" used by the legacy prefix/suffix annotation matching.
    image_suffix: Union[str, List[str]] = "_leftImg8bit"

    # Legacy annotation matching (used when annotation_regex is None):
    #   * annotation_suffix is None -> any file in annotation_subdir whose
    #     name starts with "<stem>" (robust to unknown annotation suffixes);
    #   * annotation_suffix is a string -> exactly
    #     "<stem><annotation_suffix><annotation_ext>".
    annotation_suffix: Optional[str] = None
    annotation_ext: str = ".png"
    # Regex mode: named group ``key`` (required, joined with the image key) and
    # optional ``clip`` (names the clip of grouped-by-sequence frames).
    annotation_regex: Optional[str] = None
    max_frame_gap: int = 1

    # Legacy load_clip(): decode COCO-panoptic RGB annotations into id maps.
    decode_panoptic_ids: bool = True

    # Informational only: "is_keyframe" hint next to the authoritative
    # (file-existence) label flag. Clip position p is a keyframe iff
    # p >= keyframe_offset and (p - keyframe_offset) % annotated_every == 0.
    annotated_every: int = 5
    keyframe_offset: int = 0

    # When loading a clip, stack per-frame images into one (T,H,W,3) array.
    # Falls back to a list (with a warning) if frame sizes differ.
    stack_frames: bool = True

    # Raw annotation -> train id mapping (data.label_mapping.LabelMappingConfig
    # keys). None -> VPS_ANNOTATION_FORMATS[annotation_subdir]. Keys given here
    # are merged over those defaults.
    label_mapping: Optional[Dict[str, Any]] = None

    # Per-split overrides of the fields above, e.g. {"train": {...}}; dict
    # fields (label_mapping) are merged, everything else is replaced.
    split_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = _overridable_fields()
        for split, over in (self.split_overrides or {}).items():
            if not isinstance(over, dict):
                raise ValueError(f"split_overrides[{split!r}] must be a mapping, got {type(over)}")
            bad = set(over) - allowed
            if bad:
                raise ValueError(f"split_overrides[{split!r}] has unknown/forbidden keys: {sorted(bad)}")
        if self.max_frame_gap < 1:
            raise ValueError(f"max_frame_gap must be >= 1, got {self.max_frame_gap}")

    def resolve_root(self) -> Path:
        """Return the dataset root, falling back to ``$CITYSCAPES_VPS_ROOT``."""
        root = self.root or os.environ.get(ROOT_ENV_VAR)
        if not root:
            raise ValueError(
                "Cityscapes-VPS root not set. Pass CityscapesVPSConfig(root=...) "
                f"or set the {ROOT_ENV_VAR} environment variable."
            )
        return Path(root).expanduser()

    @property
    def split_dir(self) -> Path:
        return self.resolve_root() / self.split

    @property
    def image_dir(self) -> Path:
        return self.split_dir / self.image_subdir

    @property
    def annotation_dir(self) -> Optional[Path]:
        if not self.annotation_subdir:
            return None
        return self.split_dir / self.annotation_subdir

    @property
    def image_suffixes(self) -> List[str]:
        if isinstance(self.image_suffix, str):
            return [self.image_suffix] if self.image_suffix else []
        return [s for s in self.image_suffix if s]

    # -- Construction helpers ------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: "str | os.PathLike", **overrides: Any) -> "CityscapesVPSConfig":
        """Build a config from a YAML file, with optional keyword overrides."""
        import yaml  # local import: pyyaml is a light, base dependency

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a top-level mapping, got {type(data)}")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        data.update(overrides)
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> "CityscapesVPSConfig":
        """Replace fields (plain dataclass replace)."""
        return replace(self, **overrides)

    def merged(self, **overrides: Any) -> "CityscapesVPSConfig":
        """Like :meth:`with_overrides`, but dict-valued fields are merged."""
        out: Dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            if isinstance(current, dict) and isinstance(value, dict):
                out[key] = {**current, **value}
            else:
                out[key] = value
        return replace(self, **out)

    def for_split(self, split: Optional[str] = None) -> "CityscapesVPSConfig":
        """This config for ``split`` with ``split_overrides[split]`` applied."""
        cfg = replace(self, split=self.split if split is None else split)
        over = (self.split_overrides or {}).get(cfg.split)
        return cfg.merged(**over) if over else cfg


def _overridable_fields() -> set:
    return {f.name for f in fields(CityscapesVPSConfig)} - {"root", "split", "split_overrides"}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class FrameRecord:
    """A single frame within a clip (metadata only; pixels loaded on demand)."""

    clip_id: str
    frame_index: int  # parsed frame counter from the filename (e.g. 25)
    order: int  # 0-based position within the clip after sorting
    stem: str
    image_path: Path
    annotation_path: Optional[Path] = None
    key: str = ""  # join key with the annotation (regex ``key`` group or stem)
    sequence: Optional[str] = None  # regex ``sequence`` group, if any
    annotation_clip_id: Optional[str] = None  # ``clip`` group of the annotation name

    @property
    def is_labeled(self) -> bool:
        return self.annotation_path is not None


@dataclass
class ClipRecord:
    """An ordered sequence of frames sharing a clip id."""

    clip_id: str
    split: str
    frames: List[FrameRecord] = field(default_factory=list)
    group: Optional[str] = None  # e.g. the Cityscapes sequence the clip comes from

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def labeled_mask(self) -> np.ndarray:
        return np.array([f.is_labeled for f in self.frames], dtype=bool)

    @property
    def labeled_orders(self) -> List[int]:
        return [f.order for f in self.frames if f.is_labeled]

    @property
    def frame_indices(self) -> np.ndarray:
        return np.array([f.frame_index for f in self.frames], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Annotation decoding
# --------------------------------------------------------------------------- #
def decode_panoptic(rgb: np.ndarray) -> np.ndarray:
    """Decode a COCO-panoptic RGB PNG into a 2-D segment-id map.

    Pixel id = R + 256*G + 256*256*B. Accepts (H,W,3) or (H,W,4) uint8 arrays.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected an (H,W,>=3) array to decode, got shape {rgb.shape}")
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    return r + 256 * g + 256 * 256 * b


class _AnnotationIndex:
    """One listing of the annotation directory, queried per image key."""

    def __init__(self, cfg: CityscapesVPSConfig, regex: Optional["re.Pattern[str]"]):
        self.directory = cfg.annotation_dir
        self.exists = self.directory is not None and self.directory.is_dir()
        names = (
            sorted(p.name for p in self.directory.iterdir() if p.is_file()) if self.exists else []
        )
        self.num_files = len(names)
        self.regex = regex
        self.suffix = cfg.annotation_suffix
        self.ext = cfg.annotation_ext
        self.unmatched: List[str] = []
        self.duplicates: List[str] = []
        self._names = names
        self._name_set = set(names)
        self._by_key: Dict[str, Tuple[str, Optional[str]]] = {}
        if regex is not None:
            for name in names:
                m = regex.match(name)
                if not m:
                    self.unmatched.append(name)
                    continue
                key = m.group("key")
                if key in self._by_key:
                    self.duplicates.append(name)
                    continue
                self._by_key[key] = (name, m.groupdict().get("clip"))
        self.used: set = set()

    @property
    def num_matched(self) -> Optional[int]:
        return None if self.regex is None else len(self._by_key)

    def find(self, key: str) -> Tuple[Optional[Path], Optional[str]]:
        """(annotation path or None, annotation clip id or None) for an image key."""
        if not self.exists:
            return None, None
        name: Optional[str] = None
        clip: Optional[str] = None
        if self.regex is not None:
            hit = self._by_key.get(key)
            if hit:
                name, clip = hit
        elif self.suffix is not None:
            candidate = f"{key}{self.suffix}{self.ext}"
            name = candidate if candidate in self._name_set else None
        else:
            # Smallest file name starting with the key (== sorted(glob(key*))[0]).
            i = bisect.bisect_left(self._names, key)
            if i < len(self._names) and self._names[i].startswith(key):
                name = self._names[i]
        if name is None:
            return None, None
        self.used.add(name)
        return self.directory / name, clip  # type: ignore[operator]

    def orphans(self) -> List[str]:
        """Annotation files (matching the regex, if any) never joined to an image."""
        pool = [n for n, _ in self._by_key.values()] if self.regex is not None else self._names
        return sorted(n for n in pool if n not in self.used)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CityscapesVPS(VideoSegDataset):
    """Index and load Cityscapes-VPS clips from a local copy of the dataset.

    Indexing is by clip: ``len(ds)`` is the number of clips and ``ds[i]`` returns
    a :class:`data.base.VideoClip` (frames + semantic train-id labels). The legacy
    :meth:`load_clip` still returns the raw dict with (decoded) annotation ids.
    Metadata-only access is available via :attr:`clips` and :meth:`clip_record`
    without touching pixels.
    """

    name = "cityscapes_vps"

    def __init__(self, config: Optional[CityscapesVPSConfig] = None, **overrides: Any):
        base = config or CityscapesVPSConfig()
        structural = {k: overrides.pop(k) for k in ("root", "split", "split_overrides") if k in overrides}
        #: The config as given (``split_overrides`` not yet applied).
        self.base_config = base.with_overrides(**structural) if structural else base
        #: Effective config. Precedence: dataclass defaults < YAML <
        #: ``split_overrides[split]`` < keyword ``overrides`` given here.
        self.config = self.base_config.for_split().merged(**overrides)
        self._regex = re.compile(self.config.filename_regex)
        if "frame" not in self._regex.groupindex:
            raise ValueError(f"filename_regex must define a named group 'frame': {self.config.filename_regex!r}")
        if not ({"clip", "sequence"} & set(self._regex.groupindex)):
            raise ValueError(
                f"filename_regex must define a 'clip' or a 'sequence' group: {self.config.filename_regex!r}"
            )
        self._ann_regex = re.compile(self.config.annotation_regex) if self.config.annotation_regex else None
        if self._ann_regex is not None and "key" not in self._ann_regex.groupindex:
            raise ValueError(
                f"annotation_regex must define a named group 'key': {self.config.annotation_regex!r}"
            )
        self._label_spec = self._label_mapping_spec()
        self._label_config: Optional[LabelMappingConfig] = (
            LabelMappingConfig.from_dict(self._label_spec) if "id_rule" in self._label_spec else None
        )
        self._mapper: Optional[SemanticLabelMapper] = None
        self._index_stats: Dict[str, Any] = {}
        self.clips: List[ClipRecord] = self._index_clips()

    # -- Indexing ------------------------------------------------------------ #
    def _index_clips(self) -> List[ClipRecord]:
        cfg = self.config
        image_dir = cfg.image_dir
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}\n"
                "Set config.root / $CITYSCAPES_VPS_ROOT to a local copy of "
                "Cityscapes-VPS, or adjust config.split / config.image_subdir."
            )

        image_paths = sorted(p for p in image_dir.glob(cfg.image_glob) if p.is_file())
        if not image_paths:
            warnings.warn(f"No images matched {cfg.image_glob!r} in {image_dir}")
        annotations = _AnnotationIndex(cfg, self._ann_regex)
        groups = set(self._regex.groupindex)

        records: List[FrameRecord] = []
        unmatched: List[str] = []
        for path in image_paths:
            m = self._regex.match(path.name)
            if not m:
                unmatched.append(path.name)
                continue
            gd = m.groupdict()
            stem = self._image_stem(path.name)
            key = gd.get("key") or stem
            ann_path, ann_clip = annotations.find(key)
            records.append(
                FrameRecord(
                    clip_id=gd.get("clip") or "",
                    frame_index=int(gd["frame"]),
                    order=-1,  # filled in after sorting
                    stem=stem,
                    image_path=path,
                    annotation_path=ann_path,
                    key=key,
                    sequence=gd.get("sequence"),
                    annotation_clip_id=ann_clip,
                )
            )

        if unmatched:
            warnings.warn(
                f"{len(unmatched)} file(s) in {image_dir} did not match "
                f"filename_regex={cfg.filename_regex!r} and were skipped."
            )

        split_warnings: List[str] = []
        if "clip" in groups:
            grouped: Dict[str, List[FrameRecord]] = {}
            for rec in records:
                grouped.setdefault(rec.clip_id, []).append(rec)
            raw_clips = [(cid, grouped[cid], _common_sequence(grouped[cid])) for cid in sorted(grouped)]
        else:
            raw_clips = self._clips_from_sequences(records, split_warnings)

        clips: List[ClipRecord] = []
        mismatched_clip_ids = 0
        for clip_id, frames, group in raw_clips:
            frames = sorted(frames, key=lambda f: f.frame_index)
            for order, frame in enumerate(frames):
                frame.order = order
                frame.clip_id = clip_id
                if frame.annotation_clip_id is not None and frame.annotation_clip_id != clip_id:
                    mismatched_clip_ids += 1
            clips.append(ClipRecord(clip_id=clip_id, split=cfg.split, frames=frames, group=group))

        orphans = annotations.orphans()
        self._index_stats = {
            "split": cfg.split,
            "image_dir": str(image_dir),
            "image_glob": cfg.image_glob,
            "image_files": len(image_paths),
            "image_regex": cfg.filename_regex,
            "image_regex_matched": len(records),
            "image_regex_unmatched": len(unmatched),
            "image_regex_unmatched_examples": unmatched[:5],
            "annotation_dir": None if annotations.directory is None else str(annotations.directory),
            "annotation_dir_exists": annotations.exists,
            "annotation_files": annotations.num_files,
            "annotation_regex": cfg.annotation_regex,
            "annotation_regex_matched": annotations.num_matched,
            "annotation_regex_unmatched": len(annotations.unmatched),
            "annotation_regex_unmatched_examples": annotations.unmatched[:5],
            "annotation_duplicate_keys": len(annotations.duplicates),
            "frames_with_annotation": sum(r.is_labeled for r in records),
            "orphan_annotations": len(orphans),
            "orphan_annotation_examples": orphans[:5],
            "annotation_clip_id_mismatches": mismatched_clip_ids,
            "clip_split_warnings": split_warnings,
        }
        return clips

    def _clips_from_sequences(
        self, records: List[FrameRecord], split_warnings: List[str]
    ) -> List[Tuple[str, List[FrameRecord], Optional[str]]]:
        """Group by ``sequence``, cut at frame-number gaps, name by annotation clip id."""
        by_seq: Dict[str, List[FrameRecord]] = {}
        for rec in records:
            by_seq.setdefault(rec.sequence or "", []).append(rec)
        out: List[Tuple[str, List[FrameRecord], Optional[str]]] = []
        for seq in sorted(by_seq):
            frames = sorted(by_seq[seq], key=lambda f: f.frame_index)
            runs: List[List[FrameRecord]] = [[frames[0]]]
            for prev, cur in zip(frames, frames[1:]):
                if cur.frame_index - prev.frame_index > self.config.max_frame_gap:
                    runs.append([cur])
                else:
                    runs[-1].append(cur)
            for run in runs:
                for sub in _split_run_by_annotation_clip(run, split_warnings, seq):
                    named = [f.annotation_clip_id for f in sub if f.annotation_clip_id is not None]
                    clip_id = named[0] if named else sub[0].key
                    out.append((clip_id, sub, seq or None))
        # Clip ids must be unique (a clip id seen twice means A1 is violated).
        seen: Dict[str, int] = {}
        unique = []
        for clip_id, frames, seq in sorted(out, key=lambda x: x[0]):
            if clip_id in seen:
                seen[clip_id] += 1
                new_id = f"{clip_id}#{seen[clip_id]}"
                split_warnings.append(f"clip id {clip_id} appears in several runs; renamed to {new_id}")
                clip_id = new_id
            else:
                seen[clip_id] = 1
            unique.append((clip_id, frames, seq))
        for msg in split_warnings:
            warnings.warn(msg)
        return unique

    def _image_stem(self, filename: str) -> str:
        """Strip the image suffix + extension to get the shared annotation stem."""
        name = Path(filename).stem  # drop extension
        for suffix in self.config.image_suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], base_dir: Optional[Path] = None) -> "CityscapesVPS":
        """Build from a registry spec (see :mod:`data.registry`)::

            {type, config: YAML path (optional), split, root, overrides: {...}}
        """
        known = {"type", "config", "split", "root", "overrides"}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"unknown dataset spec keys for cityscapes_vps: {sorted(unknown)}")
        base = Path(base_dir) if base_dir else Path.cwd()
        structural = {k: spec[k] for k in ("split", "root") if spec.get(k) is not None}
        if "root" in structural and not Path(structural["root"]).expanduser().is_absolute():
            structural["root"] = str(base / structural["root"])
        if spec.get("config"):
            path = Path(spec["config"])
            cfg = CityscapesVPSConfig.from_yaml(path if path.is_absolute() else base / path)
        else:
            cfg = CityscapesVPSConfig()
        return cls(cfg, **structural, **dict(spec.get("overrides") or {}))

    # -- Access -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.clips)

    def clip_record(self, index: int) -> ClipRecord:
        return self.clips[index]

    # -- VideoSegDataset interface ------------------------------------------- #
    # ``ds[i]`` / ``iter(ds)`` come from the base class and yield VideoClip.
    @property
    def class_names(self) -> Tuple[str, ...]:
        return tuple(self._label_spec["class_names"])

    @property
    def ignore_index(self) -> int:
        return int(self._label_spec["ignore_index"])

    @property
    def palette(self) -> np.ndarray:
        if self.class_names == CITYSCAPES_TRAIN_CLASSES:
            return CITYSCAPES_PALETTE.copy()
        return super().palette

    def clip_id(self, index: int) -> str:
        return self.clips[index].clip_id

    def labeled_mask(self, index: int) -> np.ndarray:
        return self.clips[index].labeled_mask

    def num_frames(self, index: int) -> int:
        return self.clips[index].num_frames

    def frame_indices(self, index: int) -> np.ndarray:
        return self.clips[index].frame_indices

    def clip_group(self, index: int) -> str:
        clip = self.clips[index]
        return clip.group or clip.clip_id

    def load_frame(self, index: int, t: int) -> np.ndarray:
        return self.load_image(self.clips[index].frames[t].image_path)

    def load_label(self, index: int, t: int) -> Optional[np.ndarray]:
        """Semantic train-id map of frame ``t`` (None if the frame is unlabelled)."""
        frame = self.clips[index].frames[t]
        if frame.annotation_path is None:
            return None
        raw = self.read_annotation_array(frame.annotation_path)
        return self.label_mapper(raw, annotation_name=frame.annotation_path.name)

    # -- Label mapping --------------------------------------------------------- #
    def _label_mapping_spec(self) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"class_names": list(CITYSCAPES_TRAIN_CLASSES), "ignore_index": IGNORE_INDEX}
        spec.update(VPS_ANNOTATION_FORMATS.get(self.config.annotation_subdir or "", {}))
        spec.update(self.config.label_mapping or {})
        return spec

    @property
    def label_mapping_config(self) -> Optional[LabelMappingConfig]:
        """The mapping in effect (relative ``segments_json`` not yet resolved)."""
        return self._label_config

    @property
    def label_mapper(self) -> SemanticLabelMapper:
        if self._label_config is None:
            raise ValueError(
                f"No label mapping for annotation_subdir={self.config.annotation_subdir!r}: set "
                "`label_mapping` (encoding, id_rule, ...) in the config; known folders: "
                f"{sorted(VPS_ANNOTATION_FORMATS)} (see configs/cityscapes_vps.yaml)."
            )
        if self._mapper is None:
            lc = self._label_config
            if lc.id_rule == "segments_json" and lc.segments_json:
                lc = replace(lc, segments_json=str(self._data_path(lc.segments_json)))
            self._mapper = SemanticLabelMapper(lc)
        return self._mapper

    def _data_path(self, template: str) -> Path:
        path = Path(template.format(split=self.config.split)).expanduser()
        return path if path.is_absolute() else self.config.resolve_root() / path

    def annotation_ids(self, index: int, t: int) -> Optional[np.ndarray]:
        """Raw (H, W) id map of frame ``t`` before class mapping (diagnostics)."""
        frame = self.clips[index].frames[t]
        if frame.annotation_path is None:
            return None
        return self.label_mapper.decode_ids(self.read_annotation_array(frame.annotation_path))

    def explain_annotation_ids(self, index: int, t: int, ids: Sequence[int]) -> List[Dict[str, Any]]:
        """id -> category -> train id -> class name for the given raw ids."""
        frame = self.clips[index].frames[t]
        name = frame.annotation_path.name if frame.annotation_path is not None else None
        return self.label_mapper.explain(ids, name)

    # -- Diagnostics ------------------------------------------------------------ #
    def diagnostics(self) -> Dict[str, Any]:
        """Indexing statistics + cross-checks against the official side-car JSONs."""
        out: Dict[str, Any] = {"index": dict(self._index_stats)}
        lc = self._label_config
        out["label_mapping"] = None if lc is None else lc.to_dict()
        out["side_car_jsons"] = self._check_side_car_jsons()
        return out

    def _check_side_car_jsons(self) -> List[Dict[str, Any]]:
        try:
            root = self.config.resolve_root()
        except ValueError as exc:
            return [{"error": str(exc)}]
        image_names = {f.image_path.name: (c.clip_id, f.order) for c in self.clips for f in c.frames}
        ann_names = {f.annotation_path.name for c in self.clips for f in c.frames if f.annotation_path}
        keys = {f.key for c in self.clips for f in c.frames}
        reports = []
        for template in VPS_INFO_JSONS:
            path = root / template.format(split=self.config.split)
            rep: Dict[str, Any] = {"file": path.name, "exists": path.is_file()}
            if not rep["exists"]:
                reports.append(rep)
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                images = data.get("images", [])
                anns = data.get("annotations", [])
                rep["num_images"] = len(images)
                rep["num_annotations"] = len(anns)
                rep["first_image_entry"] = images[0] if images else None
                names = [str(im.get("file_name", "")) for im in images]
                in_index = [n for n in names if n in image_names]
                # img/-style names carry the city_seq_frame key after the prefix.
                key_misses = [n for n in names if _key_of(n) not in keys]
                rep["image_file_names_found_in_index"] = len(in_index)
                rep["image_keys_found_in_index"] = len(names) - len(key_misses)
                rep["image_names_not_found_examples"] = key_misses[:5]
                if anns:
                    ann_files = [str(a.get("file_name", "")) for a in anns]
                    rep["annotation_files_found"] = sum(n in ann_names for n in ann_files)
                if in_index and len(in_index) == len(names):
                    # Do the JSON's images come in the same clip blocks as our index?
                    seq = [image_names[n][0] for n in names]
                    runs = [seq[0]]
                    for prev, cur in zip(seq, seq[1:]):
                        if cur != prev:
                            runs.append(cur)
                    rep["json_order_matches_clips"] = len(runs) == len(set(runs)) == len(self.clips)
            except Exception as exc:  # diagnostics must never crash
                rep["error"] = f"{type(exc).__name__}: {exc}"
            reports.append(rep)
        return reports

    # -- Pixel loading ------------------------------------------------------- #
    def load_image(self, path: Path) -> np.ndarray:
        """Load an RGB frame as an (H,W,3) uint8 array."""
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    @staticmethod
    def read_annotation_array(path: Path) -> np.ndarray:
        """The annotation PNG as stored (2-D for 8/16/32-bit gray, (H,W,C) for RGB)."""
        with Image.open(path) as img:
            return np.asarray(img)

    def load_annotation(self, path: Path, decode: Optional[bool] = None) -> np.ndarray:
        """Load an annotation as an integer label map (legacy, no class mapping).

        Single-channel (semantic) PNGs are returned as (H,W). Multi-channel
        (panoptic-color) PNGs are decoded to an (H,W) id map when
        ``decode`` (default: ``config.decode_panoptic_ids``) is True, else the
        raw (H,W,C) array is returned.
        """
        if decode is None:
            decode = self.config.decode_panoptic_ids
        arr = self.read_annotation_array(path)
        if arr.ndim == 3 and arr.shape[2] >= 3 and decode:
            return decode_panoptic(arr)
        return arr

    def is_keyframe(self, index: int) -> np.ndarray:
        """(T,) bool hint from ``annotated_every`` / ``keyframe_offset`` (not authoritative)."""
        every = max(1, self.config.annotated_every)
        offset = self.config.keyframe_offset
        pos = np.arange(self.clips[index].num_frames)
        return (pos >= offset) & ((pos - offset) % every == 0)

    def load_clip(self, index: int, load_annotations: bool = True) -> Dict[str, Any]:
        """Load one clip: frames, sparse annotations, and metadata (legacy dict API).

        Returns a dict with:
          clip_id           : str
          split             : str
          num_frames        : int (== T)
          images            : (T,H,W,3) uint8 array, or a list of arrays if
                              per-frame sizes differ / stacking is disabled
          labeled_mask      : (T,) bool  -- True where an annotation exists
          is_keyframe       : (T,) bool  -- hint, see :meth:`is_keyframe`
          annotations       : list length T; entry is an (H,W)[,3] array for
                              labeled frames and None for unlabeled ones
          frame_indices     : (T,) int64 -- parsed per-frame counters
          image_paths       : list[str] length T
          annotation_paths  : list[Optional[str]] length T
        """
        clip = self.clips[index]
        images = [self.load_image(f.image_path) for f in clip.frames]

        annotations: List[Optional[np.ndarray]] = []
        if load_annotations:
            for f in clip.frames:
                annotations.append(
                    self.load_annotation(f.annotation_path) if f.is_labeled else None
                )
        else:
            annotations = [None] * clip.num_frames

        images_out: Any = images
        if self.config.stack_frames:
            shapes = {im.shape for im in images}
            if len(shapes) == 1:
                images_out = np.stack(images, axis=0)
            elif images:
                warnings.warn(
                    f"Clip {clip.clip_id}: frames have differing shapes {shapes}; "
                    "returning a list instead of a stacked array."
                )

        return {
            "clip_id": clip.clip_id,
            "split": clip.split,
            "num_frames": clip.num_frames,
            "images": images_out,
            "labeled_mask": clip.labeled_mask,
            "is_keyframe": self.is_keyframe(index),
            "annotations": annotations,
            "frame_indices": clip.frame_indices,
            "image_paths": [str(f.image_path) for f in clip.frames],
            "annotation_paths": [
                str(f.annotation_path) if f.annotation_path else None for f in clip.frames
            ],
        }


def _common_sequence(frames: Sequence[FrameRecord]) -> Optional[str]:
    seqs = {f.sequence for f in frames if f.sequence}
    return seqs.pop() if len(seqs) == 1 else None


_KEY_RE = re.compile(r"([^_/]+_\d{6}_\d{6})")


def _key_of(name: str) -> str:
    """The Cityscapes ``city_seq_frame`` part of a file name ('' if none)."""
    m = _KEY_RE.search(name)
    return m.group(1) if m else ""


def _split_run_by_annotation_clip(
    run: List[FrameRecord], split_warnings: List[str], seq: str
) -> List[List[FrameRecord]]:
    """Split a run of consecutive frames that holds annotations of several clips.

    Each frame goes to the clip of the next labelled frame at or after it
    (frames after the last labelled one go to that clip): VPS clips end on a
    labelled frame (position 29), so this recovers clip boundaries exactly.
    """
    labelled = [(i, f.annotation_clip_id) for i, f in enumerate(run) if f.annotation_clip_id is not None]
    if len({c for _, c in labelled}) <= 1:
        return [run]
    split_warnings.append(
        f"sequence {seq}: frames {run[0].frame_index}..{run[-1].frame_index} are consecutive but carry "
        f"annotations of clips {sorted({c for _, c in labelled})}; splitting them (assumption A1 violated)"
    )
    owner: List[Optional[str]] = [None] * len(run)
    nxt: Optional[str] = labelled[-1][1]
    li = len(labelled) - 1
    for i in range(len(run) - 1, -1, -1):
        while li >= 0 and labelled[li][0] >= i:
            nxt = labelled[li][1]
            li -= 1
        owner[i] = nxt
    pieces: List[List[FrameRecord]] = [[run[0]]]
    for i in range(1, len(run)):
        if owner[i] != owner[i - 1]:
            pieces.append([run[i]])
        else:
            pieces[-1].append(run[i])
    return pieces


# --------------------------------------------------------------------------- #
# Optional torch bridge (lazy import so numpy-only environments still work)
# --------------------------------------------------------------------------- #
def clip_to_tensors(clip: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a clip dict from :meth:`CityscapesVPS.load_clip` to torch tensors.

    Images -> (T,3,H,W) float32 in [0,1]; labeled annotations -> (H,W) int64
    tensors (unlabeled entries stay None). Imports torch lazily; raises a clear
    error if torch is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "clip_to_tensors requires torch. Install it with `pip install .[torch]` "
            "(or reuse the CUDA torch preinstalled on Colab/Kaggle)."
        ) from exc

    images = clip["images"]
    if isinstance(images, np.ndarray):
        img_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0
    else:  # list of frames with differing sizes
        img_t = [
            torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0 for im in images
        ]

    ann_t: List[Any] = [
        torch.from_numpy(np.ascontiguousarray(a)).long() if a is not None else None
        for a in clip["annotations"]
    ]

    out = dict(clip)
    out["images"] = img_t
    out["annotations"] = ann_t
    out["labeled_mask"] = torch.from_numpy(clip["labeled_mask"])
    out["is_keyframe"] = torch.from_numpy(clip["is_keyframe"])
    out["frame_indices"] = torch.from_numpy(clip["frame_indices"])
    return out
