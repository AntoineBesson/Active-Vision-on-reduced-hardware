"""Confusion-matrix metrics (numpy only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.metrics import ConfusionMatrix


def test_known_confusion_and_iou():
    cm = ConfusionMatrix(3, ignore_index=255, class_names=["a", "b", "c"])
    target = np.array([[0, 0, 1, 1, 255]])
    pred = np.array([[0, 1, 1, 1, 2]])  # the ignored pixel's prediction does not count
    cm.update(pred, target)
    np.testing.assert_array_equal(cm.matrix, [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
    iou = cm.iou()
    assert iou[0] == pytest.approx(1 / 2)  # TP 1, FN 1
    assert iou[1] == pytest.approx(2 / 3)  # TP 2, FP 1
    assert math.isnan(iou[2])  # never seen: excluded from the mean
    m = cm.compute()
    assert m["miou"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert m["pixel_accuracy"] == pytest.approx(3 / 4)
    assert m["mean_accuracy"] == pytest.approx((1 / 2 + 1) / 2)
    assert m["num_valid_pixels"] == 4 and m["num_classes_evaluated"] == 2 and m["num_frames"] == 1
    assert math.isnan(m["per_class_iou"]["c"])


def test_predicted_but_absent_class_counts():
    cm = ConfusionMatrix(2)
    cm.update(np.array([1, 1]), np.array([0, 0]))
    assert cm.iou().tolist() == [0.0, 0.0]  # class 1 has union > 0 -> IoU 0, not NaN


def test_accumulation_across_clips_is_dataset_level_not_per_frame_average():
    cm = ConfusionMatrix(2)
    cm.update(np.zeros(100, int), np.zeros(100, int))  # frame 1: 100 px all correct (class 0)
    cm.update(np.array([0]), np.array([1]))  # frame 2: 1 px wrong
    # dataset-level IoU_0 = 100/101, not the mean of per-frame IoUs
    assert cm.iou()[0] == pytest.approx(100 / 101)
    other = ConfusionMatrix(2)
    other.update(np.array([1]), np.array([1]))
    cm += other
    assert cm.num_frames == 3 and cm.iou()[1] == pytest.approx(1 / 2)


def test_update_clip_uses_labeled_frames_only():
    cm = ConfusionMatrix(2)
    preds = np.zeros((3, 2, 2), int)
    labels = np.stack([np.zeros((2, 2), int), np.ones((2, 2), int), np.full((2, 2), 255)])
    used = cm.update_clip(preds, labels, np.array([True, False, True]))
    assert used == 2 and cm.num_frames == 2
    assert cm.matrix.sum() == 4  # the unlabelled frame (all wrong) was skipped


def test_unpredicted_pixels_are_errors_not_ignored():
    cm = ConfusionMatrix(2)
    cm.update(np.array([0, 255, -1]), np.array([0, 0, 1]))
    np.testing.assert_array_equal(cm.unpredicted, [1, 1])
    assert cm.iou()[0] == pytest.approx(1 / 2) and cm.iou()[1] == 0.0
    m = cm.compute()
    assert m["num_unpredicted_pixels"] == 2 and m["pixel_accuracy"] == pytest.approx(1 / 3)


def test_invalid_inputs():
    cm = ConfusionMatrix(3)
    with pytest.raises(ValueError, match="shape"):
        cm.update(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="outside"):
        cm.update(np.zeros(2), np.array([0, 7]))  # a label bug must not pass silently
    with pytest.raises(ValueError):
        ConfusionMatrix(2).merge(ConfusionMatrix(3))
    with pytest.raises(ValueError):
        ConfusionMatrix(2, class_names=["x"])


def test_empty_and_round_trip():
    cm = ConfusionMatrix(2)
    assert math.isnan(cm.compute()["miou"])
    cm.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
    clone = ConfusionMatrix.from_state_dict(cm.state_dict())
    np.testing.assert_array_equal(clone.matrix, cm.matrix)
    assert clone.compute() == cm.compute()
