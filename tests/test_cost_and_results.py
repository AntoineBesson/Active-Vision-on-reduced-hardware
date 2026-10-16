"""Cost accounting (ledger, FLOPs, latency, memory) and run-record JSON files."""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pytest

from eval.cost import CostLedger, latency_stats, skipped
from eval.results import environment_info, git_info, make_run_record, to_jsonable, write_json


# --- ledger (no torch) ------------------------------------------------------ #
def test_ledger_sums_components_into_total():
    ledger = CostLedger()
    for _ in range(2):  # two frames of 100 px
        ledger.begin_frame(100)
        ledger.add("context", flops=10, pixels=25)
        ledger.add("glimpses", flops=30, pixels=20, calls=2)
        ledger.add("policy", flops=4)
    s = ledger.summary()
    assert s["frames"] == 2
    assert s["components"]["glimpses"]["calls"] == 4  # accumulated over frames
    assert s["components"]["glimpses"]["calls_per_frame"] == 2
    assert s["components"]["context"]["flops_per_frame"] == 10
    assert s["total"]["flops_per_frame"] == 44 and s["total"]["gmacs_per_frame"] == pytest.approx(22e-9)
    assert s["pixel_fraction"] == pytest.approx(45 / 100)
    assert "FlopCounterMode" in s["flop_counter"]


def test_latency_stats_and_skipped():
    st = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert st["median_ms"] == 3.0 and st["iters"] == 5 and st["p95_ms"] == pytest.approx(80.8)
    assert skipped("why") == {"status": "skipped", "reason": "why"}


# --- results (no torch) ----------------------------------------------------- #
def test_to_jsonable_and_strict_json(tmp_path):
    obj = {"a": float("nan"), "b": np.float32(1.5), "c": np.arange(2), "d": (1, float("inf")), 3: tmp_path}
    out = to_jsonable(obj)
    assert out == {"a": None, "b": 1.5, "c": [0, 1], "d": [1, None], "3": str(tmp_path)}
    path = write_json(tmp_path / "sub" / "r.json", {"x": float("nan"), "y": [np.int64(3)]})
    text = path.read_text()
    assert "NaN" not in text and json.loads(text) == {"x": None, "y": [3]}
    assert not list(path.parent.glob("*.tmp"))  # atomic write leaves no temp file
    umask = os.umask(0)
    os.umask(umask)
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask  # readable like any file, not 0600


def test_git_and_environment_info():
    g = git_info()
    assert g["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", g["commit"])
    assert set(g) == {"commit", "branch", "dirty"}
    env = environment_info()
    assert env["python"] and env["numpy"] and "torch" in env


def test_make_run_record_fields():
    rec = make_run_record(
        run_name="r", method={"name": "full_frame", "scale": 0.5}, config={"k": 1}, seed=3,
        dataset={"name": "d"}, backbone={"type": "dummy"}, metrics={"miou": float("nan")},
        cost={"x": 1}, notes="smoke",
    )
    for key in ("schema_version", "created_utc", "git", "seed", "method", "dataset", "backbone",
                "metrics", "cost", "environment", "config", "notes"):
        assert key in rec
    assert rec["metrics"]["miou"] is None and rec["seed"] == 3
    json.dumps(rec, allow_nan=False)


# --- torch-dependent cost functions ---------------------------------------- #
def test_count_flops_and_ledger_count():
    torch = pytest.importorskip("torch")
    from eval.cost import count_flops

    conv = torch.nn.Conv2d(3, 8, 3, padding=1)
    x = torch.rand(1, 3, 10, 10)
    assert count_flops(lambda: conv(x)) == 2 * 8 * 3 * 9 * 100  # 2 FLOPs per MAC
    ledger = CostLedger()
    ledger.begin_frame(100)
    with ledger.count("policy"):
        conv(x)
    assert ledger.summary()["components"]["policy"]["flops_per_frame"] == 43200


def test_benchmark_latency_cpu_and_memory_skip():
    torch = pytest.importorskip("torch")
    from eval.cost import benchmark_latency, measure_peak_memory

    conv = torch.nn.Conv2d(3, 4, 3)
    x = torch.rand(1, 3, 16, 16)
    st = benchmark_latency(lambda: conv(x), "cpu", warmup=2, iters=5)
    assert st["status"] == "ok" and st["iters"] == 5 and st["synchronized"] is False
    assert 0 < st["median_ms"] <= st["max_ms"] and st["min_ms"] <= st["median_ms"]
    assert measure_peak_memory(lambda: conv(x), "cpu")["status"] == "skipped"


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="GPU-only: needs CUDA")
def test_gpu_latency_and_peak_memory():  # pragma: no cover - runs on GPU machines only
    import torch

    from eval.cost import benchmark_latency, measure_peak_memory

    conv = torch.nn.Conv2d(3, 8, 3).cuda().half()
    x = torch.rand(1, 3, 256, 256, device="cuda").half()
    st = benchmark_latency(lambda: conv(x), "cuda", warmup=3, iters=10)
    assert st["synchronized"] is True and st["median_ms"] > 0 and "device_name" in st
    mem = measure_peak_memory(lambda: conv(x), "cuda")
    assert mem["status"] == "ok" and mem["peak_allocated_mib"] > 0
    assert mem["peak_increment_mib"] == mem["peak_allocated_mib"] - mem["allocated_before_mib"]
    assert not math.isnan(st["p95_ms"])
