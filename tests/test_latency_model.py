"""T4 latency model of foveated configurations, from the three latency-floor sessions (CPU)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from foveation.latency_model import LatencyTables

REPO = Path(__file__).resolve().parents[1]
SESSIONS = sorted((REPO / "results" / "latency_floor").glob("latency_floor_t4-s*.json"))
TABLES = LatencyTables.load(SESSIONS)


def _graph_e2e():
    recs = {}
    for path in SESSIONS:
        for r in json.loads(path.read_text())["records"]["e2e"]:
            if r["level"] == "graph":
                recs.setdefault(r["name"], []).append(r)
    return [dict(rs[0], median_ms=statistics.median(r["cuda_event"]["median_ms"] for r in rs)) for rs in recs.values()]


def test_three_sessions_are_loaded():
    assert len(SESSIONS) == 3 and TABLES.sessions == 3


@pytest.mark.parametrize("rec", _graph_e2e(), ids=lambda r: r["name"])
def test_prediction_matches_every_measured_e2e_point(rec):
    p = TABLES.predict(rec["context_model"], rec["context_scale"], rec["glimpse_model"], rec["size"], rec["k"], "topk")
    assert abs(rec["median_ms"] / p["median_ms"] - 1) < 0.05
    assert p["p95_ms"] >= p["median_ms"]
    assert p["gflops"] == pytest.approx(rec["gflops"], rel=0.01)
    assert set(p["parts"]) == {"context", "glimpses", "overhead"}


def test_fullframe_values():
    assert TABLES.fullframe("pidnet_s", 1.0)["median_ms"] == pytest.approx(19.35, abs=0.01)
    assert TABLES.fullframe("bisenetv2", 0.5)["median_ms"] == pytest.approx(8.06, abs=0.01)
    assert TABLES.fullframe("pidnet_s", 1.0)["gflops"] > 90


def test_off_grid_point_names_the_missing_run():
    with pytest.raises(KeyError, match="latency_floor"):
        TABLES.predict("pidnet_s", 0.25, "pidnet_s", 384, 3, "topk")
    with pytest.raises(KeyError, match="latency_floor"):
        TABLES.fullframe("pidnet_s", 0.3)


def test_from_records_builds_a_table_without_files():
    def rec(name, ms, **kw):
        return {"name": name, "level": "graph", "gflops": 1.0, "cuda_event": {"median_ms": ms, "p95_ms": ms + 1}, **kw}

    t = LatencyTables.from_records({
        "area": [rec("a", 2.0, model="pidnet_s", kind="frame", scale=0.25)],
        "batch": [rec("b", 3.0, model="pidnet_s", size=64, k=2)],
        "overhead": [rec("o", 1.0, stride=8, context_scale=0.25, size=64, k=2, selection="greedy")],
    })
    p = t.predict("pidnet_s", 0.25, "pidnet_s", 64, 2, "greedy")
    assert p["median_ms"] == pytest.approx(6.0) and p["p95_ms"] == pytest.approx(9.0)


def test_tables_know_the_measured_margin():
    assert TABLES.margin == 32


def test_another_margin_names_the_missing_run():
    assert TABLES.predict("pidnet_s", 0.25, "pidnet_s", 384, 2, "topk", margin=32)["median_ms"] > 0
    with pytest.raises(KeyError, match="latency_floor"):
        TABLES.predict("pidnet_s", 0.25, "pidnet_s", 384, 2, "topk", margin=16)


def _reuse_tables():
    def rec(name, ms, part="reuse", level="eager", **kw):
        return {"name": name, "level": level, "gflops": 1.0, "cuda_event": {"median_ms": ms, "p95_ms": ms * 2}, **kw}

    reuse = [rec("g", 0.1, op="gray"), rec("h", 2.0, op="homography", homography="iclk"),
             rec("m", 0.3, op="motion", size=384), rec("w", 0.4, op="warp"), rec("f", 0.2, op="fusion"),
             rec("p8", 0.5, op="post", stride=8, scale=1.0), rec("p8c", 0.25, op="post", stride=8, scale=0.25)]
    reuse += [rec(f"gp{k}", 0.05 * k, op="glimpse_post", stride=8, size=384, k=k) for k in (1, 2, 3)]
    return LatencyTables.from_records({
        "area": [rec("a1", 19.0, part="area", level="graph", model="pidnet_s", kind="frame", scale=1.0),
                 rec("a25", 2.7, part="area", level="graph", model="pidnet_s", kind="frame", scale=0.25)],
        "batch": [rec(f"b{k}", 1.5 * k, part="batch", level="graph", model="pidnet_s", size=384, k=k) for k in (1, 2, 3)],
        "reuse": reuse,
    })


def test_reuse_frame_sums_its_parts():
    from reuse.method import MotionReuseConfig

    t = _reuse_tables()
    cfg = MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.05, tau=1.0, homography="iclk")
    first = t.predict_reuse_frame("pidnet_s", cfg, moved_cells=28, fallback=False, first=True)
    assert first["median_ms"] == pytest.approx(19.0 + 0.5)
    reuse3 = t.predict_reuse_frame("pidnet_s", cfg, moved_cells=3, fallback=False, first=False)
    assert reuse3["median_ms"] == pytest.approx(0.1 + 2.0 + 0.3 + 0.4 + 4.5 + 0.15)
    back = t.predict_reuse_frame("pidnet_s", cfg, moved_cells=28, fallback=True, first=False)
    assert back["median_ms"] == pytest.approx(0.1 + 2.0 + 0.3 + 19.0 + 0.5)
    none = t.predict_reuse_frame("pidnet_s", cfg, moved_cells=0, fallback=False, first=False)
    assert none["median_ms"] == pytest.approx(0.1 + 2.0 + 0.3 + 0.4)
    b = MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.05, tau=1.0, homography="iclk",
                          ctx_scale=0.25)
    assert t.predict_reuse_frame("pidnet_s", b, 2, False, False)["median_ms"] == pytest.approx(
        0.1 + 2.0 + 0.3 + 0.4 + 3.0 + 0.1 + 2.7 + 0.25 + 0.2)


def test_unmeasured_reuse_point_names_the_missing_run():
    from reuse.method import MotionReuseConfig

    t = _reuse_tables()
    cfg = MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.05, tau=1.0, homography="iclk")
    with pytest.raises(KeyError, match="latency_floor"):
        t.predict_reuse_frame("pidnet_s", cfg, moved_cells=4, fallback=False, first=False)  # k = 4 not measured
    with pytest.raises(KeyError, match="latency_floor"):
        t.predict_reuse_frame("pidnet_s", MotionReuseConfig(full_scale=0.6, glimpse_size=384, theta=20, rho=0.05,
                                                            tau=1.0, homography="iclk"), 28, False, True)


def test_summary_mean_and_p95():
    import numpy as np

    from foveation.latency_model import frame_latency_summary

    ms = [1.0] * 95 + [10.0] * 5
    out = frame_latency_summary(ms)
    assert out["mean_ms"] == pytest.approx(1.45) and out["p95_ms"] == pytest.approx(float(np.percentile(ms, 95)))


def test_reuse_frame_counts_the_network_gflops():
    from reuse.method import MotionReuseConfig

    t = _reuse_tables()
    a = MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.05, tau=1.0, homography="iclk")
    b = MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.05, tau=1.0, homography="iclk",
                          ctx_scale=0.25)
    assert t.predict_reuse_frame("pidnet_s", a, 28, False, True)["gflops"] == pytest.approx(1.0)  # the full pass
    assert t.predict_reuse_frame("pidnet_s", a, 3, False, False)["gflops"] == pytest.approx(1.0)  # 3 glimpses
    assert t.predict_reuse_frame("pidnet_s", a, 0, False, False)["gflops"] == 0
    assert t.predict_reuse_frame("pidnet_s", b, 2, False, False)["gflops"] == pytest.approx(2.0)  # glimpses + context


def test_canvas_frame_sums_its_parts():
    from canvas.replay import CanvasConfig

    t = _reuse_tables()
    ctx = 2.7 + 0.25  # context backbone + its post-processing
    two = 3.0 + 0.1  # 2 glimpses + their post-processing
    n, a, b = (CanvasConfig(ctx_scale=0.25, glimpse_size=384, variant=v, k=2) for v in ("off", "iclk", "none"))
    for first in (True, False):
        assert t.predict_canvas_frame("pidnet_s", n, 2, first)["median_ms"] == pytest.approx(ctx + two)
    assert t.predict_canvas_frame("pidnet_s", n, 0, False)["median_ms"] == pytest.approx(ctx)  # no glimpse: context only
    assert t.predict_canvas_frame("pidnet_s", b, 2, True)["median_ms"] == pytest.approx(ctx + two)  # empty canvas
    assert t.predict_canvas_frame("pidnet_s", b, 2, False)["median_ms"] == pytest.approx(ctx + two + 0.2)  # + fusion
    assert t.predict_canvas_frame("pidnet_s", a, 2, True)["median_ms"] == pytest.approx(ctx + two + 0.1)  # + gray
    later = t.predict_canvas_frame("pidnet_s", a, 3, False)
    assert later["median_ms"] == pytest.approx(ctx + 4.5 + 0.15 + 0.1 + 0.2 + 2.0 + 0.4)  # gray, fusion, H, warps
    assert later["p95_ms"] == pytest.approx(2 * later["median_ms"])
    assert set(later["parts"]) == {"context_backbone", "context_post", "glimpses", "glimpse_post", "gray", "fusion",
                                   "homography", "warp"}  # no policy, no motion map
    motion = t.predict_canvas_frame("pidnet_s", a, 3, False, with_motion=True)
    assert motion["median_ms"] == pytest.approx(later["median_ms"] + 0.3)
    for cfg, first in ((b, False), (n, False), (a, True)):  # the motion map belongs to variant A's later frames only
        assert "motion" not in t.predict_canvas_frame("pidnet_s", cfg, 2, first, with_motion=True)["parts"]
    assert later["gflops"] == pytest.approx(2.0)  # context + glimpses
    assert t.predict_canvas_frame("pidnet_s", a, 0, False)["gflops"] == pytest.approx(1.0)


def test_unmeasured_canvas_point_names_the_missing_run():
    from canvas.replay import CanvasConfig

    t = _reuse_tables()
    with pytest.raises(KeyError, match="latency_floor"):
        t.predict_canvas_frame("pidnet_s", CanvasConfig(ctx_scale=0.25, glimpse_size=384, variant="off", k=4), 4, False)
    with pytest.raises(KeyError, match="latency_floor"):
        t.predict_canvas_frame("pidnet_s", CanvasConfig(ctx_scale=0.5, glimpse_size=384, variant="off", k=1), 1, False)
