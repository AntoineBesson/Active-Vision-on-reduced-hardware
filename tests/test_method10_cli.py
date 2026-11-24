"""scripts/method10.py ceiling (pure python): the gate 0 arithmetic."""

from __future__ import annotations

import json

import pytest

from scripts.method10 import ceiling_report, curve_at, envelope_at, load_timing

CURVES = {"models": {
    "a": {"scales": [0.25, 0.5, 1.0], "miou": [0.50, 0.68, 0.76], "price_ms": [4.0, 7.0, 21.0]},
    "b": {"scales": [0.25, 0.5, 1.0], "miou": [0.59, 0.70, 0.73], "price_ms": [4.3, 10.0, 33.0]},
}}


def _timing(fresh_a1=14.0):
    t = {}
    for m, v in CURVES["models"].items():
        for s, ms in zip(v["scales"], v["price_ms"]):
            t[("ff", m, s)] = {"median_ms": ms, "sessions": ["s1"]}
    t[("key", "a", 1.0)] = {"median_ms": 21.5, "sessions": ["s1"]}
    t[("fresh", "a", 1.0)] = {"median_ms": fresh_a1, "sessions": ["s1"]}
    t[("key", "b", 0.5)] = {"median_ms": 10.0, "sessions": ["s1"]}
    t[("fresh", "b", 0.5)] = {"median_ms": 9.5, "sessions": ["s1"]}
    return t


def test_curve_and_envelope():
    pts = [(4.0, 50.0), (7.0, 68.0), (21.0, 76.0)]
    assert curve_at(pts, 5.5) == pytest.approx(59.0)
    assert curve_at(pts, 3.0) is None and curve_at(pts, 22.0) is None
    env, by = envelope_at({"a": pts, "b": [(4.3, 59.0), (10.0, 70.0), (33.0, 73.0)]}, 25.0)
    assert by == "b" and env == pytest.approx(70 + 15 / 23 * 3)  # a's curve ends at 21 ms: left out


def test_ceiling_reads_the_envelope_at_the_schedule_latency():
    rep = ceiling_report(_timing(), CURVES)
    row = rep["models"]["a"]["rows"][0]
    inf = row["schedules"][-1]
    assert inf["K"] == "inf" and inf["mean_ms"] == 14.0
    # envelope at 14 ms: a = 68 + 7/14 * 8 = 72; b = 70 + 4/23 * 3 = 70.52 -> 72
    assert inf["envelope"] == pytest.approx(72.0) and inf["envelope_model"] == "a"
    assert inf["ceiling"] == pytest.approx(76.0 - 72.0)
    k2 = row["schedules"][0]
    assert k2["K"] == 2 and k2["mean_ms"] == pytest.approx((21.5 + 14.0) / 2)
    assert rep["models"]["a"]["gate0_pass"]
    # b: fresh is 95 % of the key frame; at 9.5 ms the envelope is above its own 70 at 10 ms
    assert not rep["models"]["b"]["gate0_pass"]
    assert rep["continue"] == ["a"]


def test_a_fresh_pass_slower_than_every_curve_still_reads_the_slowest_point():
    rep = ceiling_report(_timing(fresh_a1=20.9), CURVES)
    inf = rep["models"]["a"]["rows"][0]["schedules"][-1]
    assert inf["ceiling"] < 1.0 and not rep["models"]["a"]["gate0_pass"]


def test_load_timing_takes_the_median_and_refuses_quick_runs(tmp_path):
    paths = []
    for i, ms in enumerate([10.0, 12.0, 11.0]):
        rec = {"kind": "fresh", "model": "a", "scale": 1.0, "level": "graph", "cuda_event": {"median_ms": ms}}
        p = tmp_path / f"gate0_s{i}.json"
        p.write_text(json.dumps({"session": f"s{i}", "quick": False, "records": [rec]}))
        paths.append(str(p))
    t = load_timing(paths)
    assert t[("fresh", "a", 1.0)]["median_ms"] == 11.0
    q = tmp_path / "gate0_q.json"
    q.write_text(json.dumps({"session": "q", "quick": True, "records": []}))
    with pytest.raises(SystemExit, match="smoke"):
        load_timing([str(q)])
