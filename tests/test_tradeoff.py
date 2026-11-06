"""Trade-off readings against the full-frame curve, and the clip bootstrap (CPU, numpy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.metrics import ConfusionMatrix
from eval.tradeoff import (
    FullFrameCurve,
    clip_bootstrap,
    delta_miou_at_equal_latency,
    latency_saved_at_equal_miou,
    miou_of,
)


def test_curve_interpolation():
    c = FullFrameCurve([10.0, 2.0], [0.70, 0.50])  # any order
    assert c.miou_at(6.0) == pytest.approx(0.60)
    assert c.latency_at(0.65) == pytest.approx(8.0)
    assert math.isnan(c.miou_at(12.0)) and math.isnan(c.latency_at(0.75))


def test_latency_at_needs_a_rising_curve():
    c = FullFrameCurve([2.0, 5.0, 10.0], [0.50, 0.72, 0.70])
    assert math.isnan(c.latency_at(0.6))
    assert c.miou_at(5.0) == pytest.approx(0.72)


def test_readings():
    c = FullFrameCurve([2.0, 10.0], [0.50, 0.70])
    assert delta_miou_at_equal_latency(0.65, 6.0, c) == pytest.approx(0.05)
    assert latency_saved_at_equal_miou(0.65, 6.0, c) == pytest.approx(2.0)


def test_absent_class_is_skipped_like_confusion_matrix():
    m = np.zeros((3, 3), np.int64)
    m[0, 0], m[1, 1], m[1, 0] = 5, 3, 1  # class 2 absent
    cm = ConfusionMatrix(3)
    cm.matrix = m
    assert miou_of(m) == pytest.approx(cm.compute()["miou"])


def test_bootstrap_of_the_reference_itself_gives_zero():
    rng = np.random.default_rng(0)
    clips = rng.integers(0, 50, (20, 4, 4))
    ff = np.stack([clips // 2 + np.eye(4, dtype=np.int64) * 10, clips])  # two scales, the second == the config
    # the configuration *is* the reference scale at 10 ms: every resample gives delta 0
    ci = clip_bootstrap(clips, ff, np.array([2.0, 10.0]), 10.0, n=200, seed=0)
    assert ci["delta_miou"] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert ci["miou"][0] <= miou_of(clips.sum(0)) <= ci["miou"][1]
    assert ci == clip_bootstrap(clips, ff, np.array([2.0, 10.0]), 10.0, n=200, seed=0)


def test_bootstrap_counts_resamples_without_a_reading():
    rng = np.random.default_rng(1)
    clips = rng.integers(0, 50, (10, 3, 3))
    ff = np.stack([clips, clips])
    ci = clip_bootstrap(clips, ff, np.array([2.0, 10.0]), 20.0, n=50, seed=0)  # latency outside the curve
    assert ci["nan_resamples"]["delta_miou"] == 50 and all(math.isnan(v) for v in ci["delta_miou"])
