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
