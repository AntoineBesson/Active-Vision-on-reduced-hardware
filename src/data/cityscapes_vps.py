"""Cityscapes-VPS clip loader.

==============================================================================
ASSUMED ON-DISK LAYOUT  (READ ME — these are ASSUMPTIONS, not verified facts)
==============================================================================

I do NOT have the real Cityscapes-VPS files in this environment, so the layout
below is my best reconstruction of the dataset released with "Video Panoptic
Segmentation" (Kim et al., CVPR 2020) and the official VPSNet repo
(https://github.com/mcahny/vps). Treat every path/name here as a configurable
DEFAULT. If the real files differ, you do not need to edit this module: point
the knobs on ``CityscapesVPSConfig`` (subdir names, filename regex, suffixes)
at the real thing. Please correct this comment against the real download.

Dataset facts I'm fairly confident about:
  * Built on Cityscapes video snippets. Each clip is a 30-frame snippet.
  * Annotations are SPARSE: every 5th frame is labeled -> 6 labeled frames per
    clip (local frame indices 0, 5, 10, 15, 20, 25). The other ~24 frames are
    unlabeled. Splits: 400 train / 100 val / 100 test clips (2400/600/600
    labeled frames). Labels are video-panoptic (instance ids are temporally
    consistent within a clip).

Assumed directory tree (root = $CITYSCAPES_VPS_ROOT or config.root):

    <root>/
        train/
            img_all/          # ALL frames of every clip (dense; most unlabeled)
            panoptic_video/   # panoptic PNG annotations, ONLY for labeled frames
            labelmap/         # (optional) semantic trainId PNGs for labeled frames
        val/
            img_all/
            panoptic_video/
            labelmap/
        test/
            img_all/          # frames only; test labels are withheld
        *.json                # COCO-panoptic-style side-car metadata (not needed here)

Assumed frame filename pattern (a Cityscapes-VPS-style name), e.g.:

    0005_0025_frankfurt_000000_001736_leftImg8bit.png
    ^^^^ ^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^
    clip frame  original Cityscapes id   image suffix
    id   idx    (city_seq_frame)

  * ``clip`` (4 digits) = which clip/snippet the frame belongs to. This is how
    frames are GROUPED into clips.
  * ``frame`` (4 digits) = the frame's position within the snippet (00..29).
    Frames are ORDERED within a clip by this number. Labeled frames are those
    whose index is a multiple of 5 (but see below — we do not rely on the rule).
  * The image "stem" is the filename minus ``_leftImg8bit.png``.

Assumed annotation naming: an annotation shares the image's stem, with a
different suffix, and lives in a sibling directory, e.g.

    0005_0025_frankfurt_000000_001736_gtFine_panoptic.png   (in panoptic_video/)

  Panoptic PNGs follow the COCO-panoptic convention: the segment id of each
  pixel is encoded as  id = R + 256*G + 256*256*B  (see ``decode_panoptic``).
  A ``labelmap/`` PNG, if present, is a single-channel semantic trainId map.

HOW "labeled vs unlabeled" IS DECIDED (important):
  A frame is considered LABELED iff a matching annotation file actually exists
  on disk for it. We do NOT hard-code the "every 5th frame" rule (real dumps
  sometimes renumber frames); instead we match by filename stem. The parsed
  frame index and the every-Nth heuristic are still exposed as metadata so you
  can cross-check.

This module deliberately has NO download logic. The dataset must already be
present locally (it is license-gated). See scripts/verify_data.py to eyeball a
clip once you have the real files.

Dependencies: numpy + pillow only. torch is optional and imported lazily by
:func:`clip_to_tensors`, so this module (and the tests) run on a CPU box with
no deep-learning stack installed.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "CityscapesVPSConfig",
    "FrameRecord",
    "ClipRecord",
    "CityscapesVPS",
    "decode_panoptic",
    "clip_to_tensors",
]

# Environment variable that points at the local dataset root.
ROOT_ENV_VAR = "CITYSCAPES_VPS_ROOT"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CityscapesVPSConfig:
    """Where the data is and how to parse its filenames.

    Every field is a knob you can retarget at the *real* dataset without
    touching the loader code. ``root`` may be left ``None`` and supplied via the
    ``CITYSCAPES_VPS_ROOT`` environment variable (see :meth:`resolve_root`).
    """

    root: Optional[str] = None
    split: str = "val"  # one of {"train", "val", "test"}

    # Sub-directory names, relative to <root>/<split>/.
    image_subdir: str = "img_all"
    annotation_subdir: str = "panoptic_video"

    # Filename globbing / parsing.
    image_glob: str = "*.png"
    # Regex applied to the image filename; must expose named groups
    # ``clip`` (clip id, kept as-is) and ``frame`` (integer frame index).
    filename_regex: str = r"^(?P<clip>\d{4})_(?P<frame>\d{4})_"
    # The image-filename suffix (before the extension) that is stripped to get
    # the shared "stem" used to find the matching annotation.
    image_suffix: str = "_leftImg8bit"

    # How to match an annotation to an image stem:
    #   * annotation_suffix is None -> match any file in annotation_subdir whose
    #     name starts with "<stem>" (robust to unknown annotation suffixes).
    #   * annotation_suffix is a string -> require exactly
    #     "<stem><annotation_suffix><annotation_ext>".
    annotation_suffix: Optional[str] = None
    annotation_ext: str = ".png"

    # If True and the annotation has 3+ channels, decode COCO-panoptic ids into
    # a single-channel id map. Single-channel (semantic) annotations pass through.
    decode_panoptic_ids: bool = True

    # Informational only: the sparse-labeling period. Used to expose an
    # "is_keyframe" hint alongside the authoritative (file-existence) label flag.
    annotated_every: int = 5

    # When loading a clip, stack per-frame images into one (T,H,W,3) array.
    # Falls back to a list (with a warning) if frame sizes differ.
    stack_frames: bool = True

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
    def annotation_dir(self) -> Path:
        return self.split_dir / self.annotation_subdir

    # -- Construction helpers ------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | os.PathLike, **overrides: Any) -> "CityscapesVPSConfig":
        """Build a config from a YAML file, with optional keyword overrides."""
        import yaml  # local import: pyyaml is a light, base dependency

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a top-level mapping, got {type(data)}")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        data.update(overrides)
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> "CityscapesVPSConfig":
        return replace(self, **overrides)


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

    @property
    def is_labeled(self) -> bool:
        return self.annotation_path is not None


@dataclass
class ClipRecord:
    """An ordered sequence of frames sharing a clip id."""

    clip_id: str
    split: str
    frames: List[FrameRecord] = field(default_factory=list)

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


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CityscapesVPS:
    """Index and load Cityscapes-VPS clips from a local copy of the dataset.

    Indexing is by clip: ``len(ds)`` is the number of clips and ``ds[i]`` loads
    the ``i``-th clip's frames + (sparse) annotations. Metadata-only access is
    available via :attr:`clips` and :meth:`clip_record` without touching pixels.
    """

    def __init__(self, config: Optional[CityscapesVPSConfig] = None, **overrides: Any):
        self.config = (config or CityscapesVPSConfig()).with_overrides(**overrides)
        self._regex = re.compile(self.config.filename_regex)
        self.clips: List[ClipRecord] = self._index_clips()

    # -- Indexing ------------------------------------------------------------ #
    def _index_clips(self) -> List[ClipRecord]:
        image_dir = self.config.image_dir
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}\n"
                "Set config.root / $CITYSCAPES_VPS_ROOT to a local copy of "
                "Cityscapes-VPS, or adjust config.split / config.image_subdir."
            )

        image_paths = sorted(image_dir.glob(self.config.image_glob))
        if not image_paths:
            warnings.warn(f"No images matched {self.config.image_glob!r} in {image_dir}")

        # Group frames by parsed clip id.
        grouped: Dict[str, List[FrameRecord]] = {}
        unmatched = 0
        for path in image_paths:
            m = self._regex.match(path.name)
            if not m:
                unmatched += 1
                continue
            clip_id = m.group("clip")
            frame_index = int(m.group("frame"))
            stem = self._image_stem(path.name)
            grouped.setdefault(clip_id, []).append(
                FrameRecord(
                    clip_id=clip_id,
                    frame_index=frame_index,
                    order=-1,  # filled in after sorting
                    stem=stem,
                    image_path=path,
                    annotation_path=self._find_annotation(stem),
                )
            )

        if unmatched:
            warnings.warn(
                f"{unmatched} file(s) in {image_dir} did not match "
                f"filename_regex={self.config.filename_regex!r} and were skipped."
            )

        clips: List[ClipRecord] = []
        for clip_id in sorted(grouped):
            frames = sorted(grouped[clip_id], key=lambda f: f.frame_index)
            for order, frame in enumerate(frames):
                frame.order = order
            clips.append(ClipRecord(clip_id=clip_id, split=self.config.split, frames=frames))
        return clips

    def _image_stem(self, filename: str) -> str:
        """Strip the image suffix + extension to get the shared annotation stem."""
        name = Path(filename).stem  # drop extension
        suffix = self.config.image_suffix
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
        return name

    def _find_annotation(self, stem: str) -> Optional[Path]:
        ann_dir = self.config.annotation_dir
        if not ann_dir.is_dir():
            return None
        if self.config.annotation_suffix is not None:
            candidate = ann_dir / f"{stem}{self.config.annotation_suffix}{self.config.annotation_ext}"
            return candidate if candidate.is_file() else None
        # Unknown suffix: match any annotation file whose name starts with the stem.
        matches = sorted(p for p in ann_dir.glob(f"{stem}*") if p.is_file())
        return matches[0] if matches else None

    # -- Access -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.clips)

    def clip_record(self, index: int) -> ClipRecord:
        return self.clips[index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.load_clip(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self.load_clip(i)

    # -- Pixel loading ------------------------------------------------------- #
    def load_image(self, path: Path) -> np.ndarray:
        """Load an RGB frame as an (H,W,3) uint8 array."""
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    def load_annotation(self, path: Path, decode: Optional[bool] = None) -> np.ndarray:
        """Load an annotation as an integer label map.

        Single-channel (semantic) PNGs are returned as (H,W). Multi-channel
        (panoptic-color) PNGs are decoded to an (H,W) id map when
        ``decode`` (default: ``config.decode_panoptic_ids``) is True, else the
        raw (H,W,C) array is returned.
        """
        if decode is None:
            decode = self.config.decode_panoptic_ids
        with Image.open(path) as img:
            arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] >= 3 and decode:
            return decode_panoptic(arr)
        return arr

    def load_clip(self, index: int, load_annotations: bool = True) -> Dict[str, Any]:
        """Load one clip: frames, sparse annotations, and metadata.

        Returns a dict with:
          clip_id           : str
          split             : str
          num_frames        : int (== T)
          images            : (T,H,W,3) uint8 array, or a list of arrays if
                              per-frame sizes differ / stacking is disabled
          labeled_mask      : (T,) bool  -- True where an annotation exists
          is_keyframe       : (T,) bool  -- (frame_index % annotated_every == 0)
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

        every = max(1, self.config.annotated_every)
        is_keyframe = np.array(
            [f.frame_index % every == 0 for f in clip.frames], dtype=bool
        )

        return {
            "clip_id": clip.clip_id,
            "split": clip.split,
            "num_frames": clip.num_frames,
            "images": images_out,
            "labeled_mask": clip.labeled_mask,
            "is_keyframe": is_keyframe,
            "annotations": annotations,
            "frame_indices": clip.frame_indices,
            "image_paths": [str(f.image_path) for f in clip.frames],
            "annotation_paths": [
                str(f.annotation_path) if f.annotation_path else None for f in clip.frames
            ],
        }


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
