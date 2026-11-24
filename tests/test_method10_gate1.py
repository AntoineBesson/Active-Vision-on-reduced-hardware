"""Method 10 gate 1: the fixture run end to end, and the scoring arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.method10_gate1 import ENVELOPE_SCALES, ff_name, m_name, miou, moving_recall, schedule_conf


def test_miou_ignores_absent_classes():
    conf = np.zeros((19, 19))
    conf[0, 0], conf[0, 1], conf[1, 1] = 3, 1, 2
    assert miou(conf) == pytest.approx((3 / 4 + 2 / 3) / 2)


def test_schedule_pools_staleness_zero_to_k_minus_one():
    eye = np.eye(19, dtype=np.int64)[None]
    run = {"scale": 0.5, "conf": {ff_name(0.5): eye * 10, m_name("N", 1): eye * 1, m_name("N", 2): eye * 2,
                                  m_name("N", 3): eye * 3}}
    assert schedule_conf(run, "N", 1)[0, 0, 0] == 10
    assert schedule_conf(run, "N", 3)[0, 0, 0] == 10 + 1 + 2


def _objects_case(stale_found: float):
    """Two clips, one headline object each; full frame finds them at every scale >= 0.5."""
    cols = {"clip": np.asarray(["a", "b"])}
    prices = {"m": [1.0 + i for i in range(len(ENVELOPE_SCALES))]}
    iou = {}
    for clip in ("a", "b"):
        d = {ff_name(s): [0.9 if s >= 0.5 else 0.1] for s in ENVELOPE_SCALES}
        d.update({m_name("N", k): [0.9 if stale_found else 0.1] for k in range(1, 5)})
        iou[clip] = d
    runs = {"m": {"scale": 0.5, "iou": iou}}
    return runs, cols, np.ones(2, bool), prices


def test_moving_recall_is_the_mean_over_phases_minus_the_envelope():
    runs, cols, mask, prices = _objects_case(stale_found=1)
    at = prices["m"][ENVELOPE_SCALES.index(0.5)]
    r = moving_recall(runs, cols, mask, prices, "m", "N", 3, at, n_boot=20)
    assert r["recall"] == 1.0 and r["delta_vs_envelope"] == pytest.approx(0.0)
    runs, cols, mask, prices = _objects_case(stale_found=0)
    r = moving_recall(runs, cols, mask, prices, "m", "N", 3, at, n_boot=20)
    assert r["recall"] == pytest.approx(1 / 3)  # found only at k = 0
    assert r["delta_vs_envelope"] == pytest.approx(1 / 3 - 1.0)
    assert r["delta_ci95"][1] < 0


def test_fixture_run_merge_and_score(tmp_path):
    pytest.importorskip("torch")
    from scripts.method10_gate1 import main

    out = str(tmp_path)
    assert main(["run", "--fixture", "--split", "val", "--models", "pidnet_s,bisenetv2", "--device", "cpu",
                 "--out-dir", out]) == 0
    assert main(["merge", "--split", "val", "--out-dir", out]) == 0
    report = tmp_path / "rep.json"
    assert main(["score", "--split", "val", "--out-dir", out, "--n-boot", "10", "--report", str(report)]) == 0
    import json

    rep = json.loads(report.read_text())
    rows = rep["models"]["pidnet_s"]["rows"]
    assert {(r["variant"], r["K"]) for r in rows} == {(v, k) for v in ("N", "H0") for k in (2, 3, 4, 5)}
    assert not rep["models"]["pidnet_s"]["continue"]  # random weights, no objects: nothing passes
