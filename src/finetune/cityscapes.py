"""Cityscapes fine (train) images and labels for Method 11's fine-tuning.

Layout, as unpacked from the official ``leftImg8bit_trainvaltest.zip`` and ``gtFine_trainvaltest.zip``
(``scripts/get_val.py --train`` fetches just these members)::

    <root>/leftImg8bit/train/<city>/<city>_<seq>_<frame>_leftImg8bit.png
    <root>/gtFine/train/<city>/<city>_<seq>_<frame>_gtFine_labelIds.png

Labels are Cityscapes label ids, mapped to the 19 train ids with the official table
(cityscapesScripts ``labels.py``); everything else is 255 (ignored). The public weights of
PIDNet-S (``_val``) and BiSeNetV2 were trained on exactly this split, and no Cityscapes-VPS clip
comes from it (every VPS clip is a Cityscapes *val* sequence), so the gates on VPS ``train_all``
and ``val`` stay unseen.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

__all__ = ["LABEL_TO_TRAIN", "IGNORE", "to_train_ids", "CityscapesFine"]

IGNORE = 255
#: official label id -> train id (cityscapesScripts/helpers/labels.py); unlisted ids are ignored
LABEL_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                  26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

_LUT = np.full(256, IGNORE, dtype=np.uint8)
for _k, _v in LABEL_TO_TRAIN.items():
    _LUT[_k] = _v


def to_train_ids(label_ids: np.ndarray) -> np.ndarray:
    """(H, W) Cityscapes label ids (uint8 PNG values) -> (H, W) uint8 train ids, 255 ignored."""
    a = np.asarray(label_ids)
    if a.ndim != 2:
        raise ValueError(f"a label map must be (H, W), got {a.shape}")
    if a.min() < 0 or a.max() > 255:
        raise ValueError("label ids must be in [0, 255]")
    return _LUT[a.astype(np.int64)]


class CityscapesFine:
    """The (image, label) pairs of one Cityscapes split, sorted by file name.

    ``__getitem__`` returns a (3, H, W) uint8 array and an (H, W) int64 train-id map. Pairs are
    matched on the ``<city>_<seq>_<frame>`` key; a missing label is an error, not a skip."""

    def __init__(self, root: Union[str, Path], split: str = "train"):
        self.root = Path(root)
        img_dir = self.root / "leftImg8bit" / split
        lab_dir = self.root / "gtFine" / split
        if not img_dir.is_dir() or not lab_dir.is_dir():
            raise FileNotFoundError(f"need {img_dir} and {lab_dir} (scripts/get_val.py --train)")
        self.items: List[Tuple[Path, Path]] = []
        for img in sorted(img_dir.glob("*/*_leftImg8bit.png")):
            key = img.name[: -len("_leftImg8bit.png")]
            lab = lab_dir / img.parent.name / f"{key}_gtFine_labelIds.png"
            if not lab.exists():
                raise FileNotFoundError(f"no label for {img.name}: {lab}")
            self.items.append((img, lab))
        if not self.items:
            raise FileNotFoundError(f"no image under {img_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image

        img, lab = self.items[i]
        frame = np.asarray(Image.open(img).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1).copy()
        label = to_train_ids(np.asarray(Image.open(lab))).astype(np.int64)
        if frame.shape[1:] != label.shape:
            raise ValueError(f"{img.name}: image {frame.shape[1:]} and label {label.shape} differ")
        return frame, label
