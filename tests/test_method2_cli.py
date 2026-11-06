"""Method 2 CLI end to end on the synthetic fixture (CPU): cache -> sweep -> tune-weights -> finalists -> report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts import method2, method2_fullframe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CFG = str(REPO / "configs" / "baseline_fixture.yaml")
GEOM = ["--scales", "0.25,0.5", "--geoms", "12:2:1,8:1:0.5"]  # a plain grid and a wide one (16 px over 8 px input)


def _fake_latency(path: Path) -> Path:
    """One latency-floor session for the dummy model on the fixture grid (graph level)."""
    def rec(name, ms, **kw):
        return {"name": name, "level": "graph", "gflops": ms / 10, "cuda_event": {"median_ms": ms, "p95_ms": ms * 1.1}, **kw}

    area = [rec(f"area/dummy/{s}", 20 * s * s + 1, model="dummy", kind="frame", scale=s) for s in (1.0, 0.5, 0.25)]
    batch = [rec(f"batch/dummy/{q}/k{k}", 0.5 + 0.4 * k, model="dummy", size=q, k=k) for q in (8, 12) for k in (1, 2, 4, 8)]
    over = [rec(f"overhead/{s}/{q}/{k}/{sel}", 0.3, stride=8, context_scale=s, size=q, k=k, selection=sel)
            for s in (0.25, 0.5) for q in (8, 12) for k in (1, 2, 4, 8) for sel in ("topk", "greedy")]
    path.write_text(json.dumps({"records": {"area": area, "batch": batch, "overhead": over}}))
    return path


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("m2")
    assert method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(d / "ff")]) == 0
    assert method2.main(["cache", "--fixture", "--split", "val", "--models", "dummy", "--pairs", "dummy+dummy", *GEOM,
                         "--root", str(d / "cache")]) == 0
    _fake_latency(d / "lat.json")
    return d


def _common(d: Path):
    return ["--fixture", "--pairs", "dummy+dummy", *GEOM, "--root", str(d / "cache"), "--fullframe", str(d / "ff"),
            "--latency", str(d / "lat.json"), "--prior-split", "val"]


def test_cache_writes_the_expected_files_and_skips_existing(work, capsys):
    names = sorted(p.name for p in (work / "cache" / "val").glob("*.npz"))
    grids = ("g12m2", "g16m2")
    expected = [f"ctx_dummy_s{s}_{g}.npz" for s in ("0.25", "0.5") for g in grids]
    expected += ["gl_dummy_g12m2.npz", "gl_dummy_g16m2_i8.npz"]
    expected += [f"fu_dummy_s{s}_dummy_{g}_{f}.npz" for s in ("0.25", "0.5") for g in ("g12m2", "g16m2_i8")
                 for f in ("mean", "conf")]
    assert names == sorted(expected)
    capsys.readouterr()
    method2.main(["cache", "--fixture", "--split", "val", "--models", "dummy", "--pairs", "dummy+dummy", *GEOM,
                  "--root", str(work / "cache")])
    assert "built 0" in capsys.readouterr().out


def test_interrupted_build_is_rebuilt(work, tmp_path):
    root = tmp_path / "cache"
    method2.main(["cache", "--fixture", "--split", "val", "--models", "dummy", "--scales", "0.25", "--geoms", "12:2:1",
                  "--fusions", "replace", "--root", str(root)])
    final = root / "val" / "gl_dummy_g12m2.npz"
    final.unlink()
    (root / "val" / "gl_dummy_g12m2.tmp.npz").write_bytes(b"partial")
    method2.main(["cache", "--fixture", "--split", "val", "--models", "dummy", "--scales", "0.25", "--geoms", "12:2:1",
                  "--fusions", "replace", "--root", str(root)])
    assert final.is_file() and not (root / "val" / "gl_dummy_g12m2.tmp.npz").exists()


@pytest.fixture(scope="module")
def sweep(work):
    out = work / "sweep.jsonl"
    assert method2.main(["sweep", "--split", "val", *_common(work), "--ks", "1,2,4", "--out", str(out)]) == 0
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_sweep_lines_and_fields(sweep):
    # per (scale, geometry, fusion): 3 ks x (oracle, random, prior, 12 uncertainty) + 1 tiling bound
    assert len(sweep) == 2 * 2 * 3 * (3 * 15 + 1)
    for r in sweep:
        m, c = r["method"], r["cost"]
        assert {"pair", "context_model", "glimpse_model", "context_scale", "glimpse_size", "margin", "glimpse_scale",
                "fusion", "k", "rule"} <= set(m)
        approximate = m["fusion"] != "replace" or m["glimpse_scale"] != 1.0
        assert c["t4_latency_is_lower_bound"] is approximate
        assert "miou" in r["metrics"] and "per_class_iou" in r["metrics"]
        assert {"pixel_fraction", "gflops", "t4_latency", "flops_implied_speedup", "realised_speedup",
                "delta_miou_at_equal_latency", "latency_saved_at_equal_miou"} <= set(c)
    tiling = [r for r in sweep if r["method"]["rule"]["kind"] == "all"]
    assert len(tiling) == 12 and all(r["cost"]["t4_latency"] is None for r in tiling)
    rand = [r for r in sweep if r["method"]["rule"]["kind"] == "random"]
    assert all(len(r["metrics"]["random_seeds_miou"]) == 5 for r in rand)


def test_oracle_is_best_at_each_point(sweep):
    groups = {}
    for r in sweep:
        m = r["method"]
        if m["rule"]["kind"] != "all":
            groups.setdefault((m["context_scale"], m["glimpse_scale"], m["fusion"], m["k"]), []).append(r)
    for rows in groups.values():
        oracle = next(r for r in rows if r["method"]["rule"]["kind"] == "oracle")
        correct = oracle["metrics"]["correct_pixels"]
        assert all(correct >= r["metrics"]["correct_pixels"] for r in rows if r["method"]["rule"]["kind"] != "random")


def test_tune_weights_is_deterministic_and_never_worse(work):
    outs = []
    for i in range(2):
        out = work / f"w{i}.json"
        assert method2.main(["tune-weights", "--split", "val", *_common(work), "--pair", "dummy+dummy",
                             "--scale", "0.25", "--geom", "12:2:1", "--fusion", "mean", "--score", "entropy",
                             "--ks", "1,2,4",
                             "--out", str(out)]) == 0
        outs.append(out.read_text())
    assert outs[0] == outs[1]
    entry = next(iter(json.loads(outs[0]).values()))
    trace = entry["objective_trace"]
    assert len(entry["weights"]) == 19 and all(b >= a for a, b in zip(trace, trace[1:]))


def test_finalists_and_report(work, sweep):
    fin = work / "finalists.json"
    assert method2.main(["finalists", "--sweep", str(work / "sweep.jsonl"), "--out", str(fin)]) == 0
    finalists = json.loads(fin.read_text())["finalists"]
    assert finalists
    keys = {json.dumps(r["method"], sort_keys=True) for r in sweep}
    for f in finalists:
        assert f["method"]["rule"]["kind"] not in ("oracle", "random", "all")
        assert json.dumps(f["method"], sort_keys=True) in keys
    # nothing on the front is dominated by another candidate
    cands = [r for r in sweep if r["method"]["rule"]["kind"] in ("uncertainty", "prior")]
    for f in (f for f in finalists if "pareto" in f["reasons"]):
        lat, miou = f["latency_ms"], f["miou"]
        assert not any(r["cost"]["t4_latency"]["median_ms"] < lat and r["metrics"]["miou"] > miou for r in cands)
    out = work / "val.json"
    assert method2.main(["report", "--split", "val", *_common(work), "--finalists", str(fin), "--out", str(out),
                         "--runs", str(work / "runs"), "--bootstrap", "50"]) == 0
    rep = json.loads(out.read_text())
    assert len(rep["finalists"]) == len(finalists)
    first = rep["finalists"][0]
    assert {"miou", "delta_miou", "latency_saved"} <= set(first["bootstrap"])
    assert {"oracle", "prior", "all"} <= set(first["references"])
    assert len(list((work / "runs").glob("*.json"))) == len(finalists)


def test_report_refuses_train(work):
    with pytest.raises(SystemExit, match="val"):
        method2.main(["report", "--split", "train", *_common(work), "--finalists", str(work / "finalists.json"),
                      "--out", str(work / "x.json")])


def test_cost_refuses_an_unmeasured_margin_of_a_plain_glimpse():
    from eval.tradeoff import FullFrameCurve
    from foveation.latency_model import LatencyTables

    tables = LatencyTables.load(sorted((REPO / "results" / "latency_floor").glob("latency_floor_t4-s*.json")))
    full = tables.fullframe("pidnet_s", 1.0)
    ref = {"full": full, "scales": [1.0], "latency_ms": [full["median_ms"]], "miou": [0.7],
           "curve": FullFrameCurve([2.0, full["median_ms"]], [0.5, 0.7])}
    args = ("pidnet_s", "pidnet_s", 0.5)
    tail = ("replace", 2, "topk", (1024, 2048), {"miou": 0.6}, tables, ref)
    assert method2.cost_of(*args, (384, 32, 1.0), *tail)["t4_latency"] is not None
    with pytest.raises(KeyError, match="latency_floor"):
        method2.cost_of(*args, (384, 16, 1.0), *tail)
    wide = method2.cost_of(*args, (384, 24, 0.75), *tail)  # a wide glimpse is an estimate, flagged as such
    assert wide["t4_latency"] is not None and wide["t4_latency_is_lower_bound"]


def test_an_interrupted_cli_build_resumes(tmp_path, monkeypatch):
    import foveation.cache as cache

    real, calls = cache.per_cell_confusion, {"n": 0, "fail_at": None}

    def counting(*a, **kw):
        calls["n"] += 1
        if calls["fail_at"] is not None and calls["n"] >= calls["fail_at"]:
            raise RuntimeError("simulated cuDNN failure")
        return real(*a, **kw)

    monkeypatch.setattr(cache, "per_cell_confusion", counting)
    argv = ["cache", "--fixture", "--split", "val", "--models", "dummy", "--scales", "0.25", "--geoms", "12:2:1",
            "--fusions", "replace"]
    assert method2.main([*argv, "--root", str(tmp_path / "ref")]) == 0
    full, calls["n"], calls["fail_at"] = calls["n"], 0, int(calls["n"] * 0.7)
    with pytest.raises(RuntimeError, match="simulated"):
        method2.main([*argv, "--root", str(tmp_path / "run")])
    calls["n"], calls["fail_at"] = 0, None
    assert method2.main([*argv, "--root", str(tmp_path / "run")]) == 0
    assert 0 < calls["n"] < full  # the clips done before the failure are not run again
    for p in (tmp_path / "ref" / "val").glob("*.npz"):
        with np.load(p) as a, np.load(tmp_path / "run" / "val" / p.name) as b:
            assert all(np.array_equal(a[k], b[k]) for k in a.files), p.name
