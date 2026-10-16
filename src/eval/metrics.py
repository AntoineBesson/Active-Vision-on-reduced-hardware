"""Semantic segmentation metrics from an accumulated confusion matrix (numpy only).

One :class:`ConfusionMatrix` is accumulated over every labelled frame of every
clip (dataset-level IoU, as in the Cityscapes benchmark), never averaged per
frame or per clip.

* ``IoU_c = TP_c / (TP_c + FP_c + FN_c)``; NaN when class ``c`` appears in
  neither ground truth nor predictions (excluded from the mean).
* ``mIoU`` = mean IoU over classes with a defined IoU.
* Pixels whose ground truth is ``ignore_index`` are skipped.
* Predictions outside ``[0, C)`` (e.g. a method that leaves a region
  unpredicted) are counted as *misses* of the ground-truth class -- never
  ignored, so leaving pixels unpredicted cannot raise the score.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = ["ConfusionMatrix"]


class ConfusionMatrix:
    """(C, C) counts, rows = ground truth, columns = prediction."""

    def __init__(self, num_classes: int, ignore_index: int = 255, class_names: Optional[Sequence[str]] = None):
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError(f"{len(class_names)} class names for {num_classes} classes")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        #: ground-truth pixels whose prediction was outside [0, C), per class
        self.unpredicted = np.zeros(self.num_classes, dtype=np.int64)
        self.num_frames = 0

    # -- Accumulation ------------------------------------------------------ #
    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Add one frame (or any batch of pixels): ``pred`` and ``target`` share a shape."""
        pred = np.asarray(pred)
        target = np.asarray(target)
        if pred.shape != target.shape:
            raise ValueError(f"prediction shape {pred.shape} != target shape {target.shape}")
        valid = target != self.ignore_index
        t = target[valid].astype(np.int64)
        p = pred[valid].astype(np.int64)
        if t.size and (t.min() < 0 or t.max() >= self.num_classes):
            bad = np.unique(t[(t < 0) | (t >= self.num_classes)])
            raise ValueError(f"target contains ids {bad.tolist()[:10]} outside [0, {self.num_classes}) that are not ignore")
        inside = (p >= 0) & (p < self.num_classes)
        c = self.num_classes
        self.matrix += np.bincount(c * t[inside] + p[inside], minlength=c * c).reshape(c, c)
        if not inside.all():
            self.unpredicted += np.bincount(t[~inside], minlength=c)
        self.num_frames += 1

    def update_clip(self, preds: np.ndarray, labels: np.ndarray, labeled_mask: np.ndarray) -> int:
        """Add the labelled frames of a clip; ``preds``/``labels`` are (T, H, W). Returns #frames used."""
        labeled_mask = np.asarray(labeled_mask, dtype=bool)
        if len(labeled_mask) != len(preds) or len(preds) != len(labels):
            raise ValueError("preds, labels and labeled_mask must have the same length T")
        for t in np.flatnonzero(labeled_mask):
            self.update(preds[t], labels[t])
        return int(labeled_mask.sum())

    def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        if (other.num_classes, other.ignore_index) != (self.num_classes, self.ignore_index):
            raise ValueError("cannot merge confusion matrices with different label spaces")
        self.matrix += other.matrix
        self.unpredicted += other.unpredicted
        self.num_frames += other.num_frames
        return self

    __iadd__ = merge

    # -- Metrics ------------------------------------------------------------- #
    def iou(self) -> np.ndarray:
        tp = np.diag(self.matrix).astype(np.float64)
        gt = self.matrix.sum(1) + self.unpredicted
        pred = self.matrix.sum(0)
        union = gt + pred - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, tp / np.maximum(union, 1), np.nan)

    def compute(self) -> Dict[str, Any]:
        """JSON-friendly metrics (NaN marks undefined values)."""
        iou = self.iou()
        tp = np.diag(self.matrix).astype(np.float64)
        gt = (self.matrix.sum(1) + self.unpredicted).astype(np.float64)
        total = gt.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(gt > 0, tp / np.maximum(gt, 1), np.nan)
        defined = ~np.isnan(iou)
        return {
            "miou": float(np.mean(iou[defined])) if defined.any() else float("nan"),
            "per_class_iou": {n: float(v) for n, v in zip(self.class_names, iou)},
            "pixel_accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
            "mean_accuracy": float(np.nanmean(recall)) if (gt > 0).any() else float("nan"),
            "num_classes_evaluated": int(defined.sum()),
            "num_valid_pixels": int(total),
            "num_unpredicted_pixels": int(self.unpredicted.sum()),
            "num_frames": int(self.num_frames),
        }

    # -- Persistence ------------------------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "class_names": self.class_names,
            "matrix": self.matrix.tolist(),
            "unpredicted": self.unpredicted.tolist(),
            "num_frames": self.num_frames,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ConfusionMatrix":
        cm = cls(state["num_classes"], state["ignore_index"], state.get("class_names"))
        cm.matrix = np.asarray(state["matrix"], dtype=np.int64).reshape(cm.num_classes, cm.num_classes)
        cm.unpredicted = np.asarray(state["unpredicted"], dtype=np.int64)
        cm.num_frames = int(state["num_frames"])
        return cm
