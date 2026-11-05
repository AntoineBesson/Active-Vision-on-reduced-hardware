"""Method 3 CLI end to end on the synthetic fixture (CPU):
cache -> stats -> bounds -> sweep -> merge -> score -> finalists -> report."""

from __future__ import annotations

import gzip
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from scripts import method2_fullframe, method3  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CFG = str(REPO / "configs" / "baseline_fixture.yaml")
GRID = ["--full-scales", "1,0.5", "--glimpse-sizes", "12", "--ctx-scales", "0.25", "--margin", "2",
        "--homography", "orb"]
SWEEP = [*GRID, "--rhos", "0,0.3", "--taus", "0.1,1"]  # tau 0.1: falls back once a cell moves
CELLS = 8  # GlimpseGrid((16, 32), 12, 2): inner 8 -> 2 x 4


def _fake_latency(path: Path) -> Path:
    """One session for the dummy model on the fixture grid: area, batch and the reuse parts."""
    def rec(name, ms, level="graph", **kw):
        return {"name": name, "level": level, "gflops": ms / 10, "cuda_event": {"median_ms": ms, "p95_ms": ms * 1.5},
                **kw}

    area = [rec(f"area/{s}", 20 * s * s + 1, model="dummy", kind="frame", scale=s) for s in (1.0, 0.5, 0.25)]
    batch = [rec(f"batch/{k}", 0.5 + 0.4 * k, model="dummy", size=12, k=k) for k in range(1, CELLS + 1)]
    reuse = [rec("gray", 0.1, "eager", op="gray"), rec("warp", 0.4, "eager", op="warp"),
             rec("fusion", 0.2, "eager", op="fusion"), rec("motion", 0.3, "eager", op="motion", size=12)]
    reuse += [rec(f"h/{m}", 2.0, "eager", op="homography", homography=m) for m in ("orb", "iclk")]
    reuse += [rec(f"post/{s}", 0.5, "eager", op="post", stride=8, scale=s) for s in (1.0, 0.5, 0.25)]
    reuse += [rec(f"gp/{k}", 0.05 * k, "eager", op="glimpse_post", stride=8, size=12, k=k) for k in range(1, CELLS + 1)]
    path.write_text(json.dumps({"records": {"area": area, "batch": batch, "reuse": reuse}}))
    return path


def _read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("m3")
    assert method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(d / "ff")]) == 0
    _fake_latency(d / "lat.json")
    common = ["--fixture", "--split", "val", "--models", "dummy", "--root", str(d / "cache")]
    assert method3.main(["cache", *common, *GRID]) == 0
    assert method3.main(["stats", *common, *GRID, "--out", str(d / "stats.json")]) == 0
    out = ["--out-dir", str(d / "out")]
    for part in ("0/2", "1/2"):  # the grid split over two sessions, merged back
        assert method3.main(["sweep", *common, *SWEEP, "--stats", str(d / "stats.json"), "--part", part,
                             "--chunk", "0/1", *out]) == 0
    assert method3.main(["bounds", *common, *SWEEP, "--stats", str(d / "stats.json"), "--chunk", "0/1", *out]) == 0
    assert method3.main(["merge", *common, *out]) == 0
    evaluation = ["--fullframe", str(d / "ff"), "--latency", str(d / "lat.json")]
    assert method3.main(["score", *common, *out, *evaluation]) == 0
    assert method3.main(["finalists", "--models", "dummy", "--split", "val", *out,
                         "--out", str(d / "finalists.json")]) == 0
    val = ["--out-dir", str(d / "val")]  # the fixture's only split stands in for train and val
    assert method3.main(["sweep", *common, *GRID, "--finalists", str(d / "finalists.json"), "--tag", "fin", *val]) == 0
    assert method3.main(["merge", *common, *val]) == 0
    assert method3.main(["report", *common, *evaluation, *val, "--finalists", str(d / "finalists.json"),
                         "--out", str(d / "val.json"), "--bootstrap", "50"]) == 0
    return d


def test_cache_writes_one_file_per_clip_and_skips_existing(work, capsys):
    names = sorted(p.name for p in (work / "cache" / "val" / "dummy").glob("*.npz"))
    assert names == ["0001.npz", "0002.npz", "0003.npz"]
    capsys.readouterr()
    method3.main(["cache", "--fixture", "--split", "val", "--models", "dummy", "--root", str(work / "cache"), *GRID])
    assert "built 0" in capsys.readouterr().out


def test_stats_histograms_and_theta_candidates(work):
    s = json.loads((work / "stats.json").read_text())
    frames, pixels = 3 * 9, (16 // 4) * (32 // 4)  # frames 1..9 of 3 clips, at 1/4 resolution
    assert sum(s["hist_aligned"]) == sum(s["hist_raw"]) == frames * pixels
    assert s["frames"] == frames and 0 <= s["identity_fallbacks"] <= frames
    assert sorted(s["percentiles"]) == ["95", "98", "99"]
    cdf = np.cumsum(s["hist_aligned"]) / frames / pixels
    for q, v in s["percentiles"].items():
        assert cdf[v] >= int(q) / 100 and (v == 0 or cdf[v - 1] < int(q) / 100)
    assert s["theta_candidates"] == sorted(set(s["percentiles"].values()))
    assert set(s["cells"]["12"]) == {str(t) for t in s["theta_candidates"]}


def _grid_names(work):
    thetas = json.loads((work / "stats.json").read_text())["theta_candidates"]
    return {f"{v}/f{f:g}/g12/t{t}/r{r:g}/u{u:g}" for v in ("3a", "3b-c0.25") for f in (1.0, 0.5) for t in thetas
            for r in (0, 0.3) for u in (0.1, 1)}


def test_sweep_covers_the_grid_and_equals_a_direct_replay(work):
    from reuse.method import MotionReuseConfig
    from reuse.replay import ClipCache, replay_clip

    rows = _read(work / "out" / "sweep_val_dummy.jsonl.gz")
    assert {r["name"] for r in rows} == _grid_names(work)
    assert len(rows) == len(_grid_names(work)) * 3
    caches = {}
    for r in rows:
        cfg = MotionReuseConfig(**r["config"])
        assert cfg.name() == r["name"]
        if r["clip_id"] not in caches:
            caches[r["clip_id"]] = ClipCache.load(work / "cache" / "val" / "dummy" / f"{r['clip_id']}.npz")
        ref = replay_clip(caches[r["clip_id"]], cfg, "cpu")  # tau groups replayed once must match
        assert np.array_equal(np.asarray(r["conf"]), ref.conf), r["name"]
        assert r["moved_cells"] == ref.moved_cells.tolist() and r["fallback"] == ref.fallback.tolist()
        assert np.asarray(r["conf"]).shape == (2, 19, 19) and len(r["moved_cells"]) == 10
    falls = [any(r["fallback"]) for r in rows]
    assert any(falls) and not all(falls)  # both paths of the tau grouping ran
    assert not any(any(r["fallback"]) for r in rows if r["config"]["tau"] == 1.0)


def test_merge_refuses_duplicates_and_lists_missing(work, tmp_path, capsys):
    src = sorted((work / "out").glob("sweep_val_dummy_*.jsonl.gz"))
    assert len(src) == 2
    dup = tmp_path / "dup"
    dup.mkdir()
    for p in src:
        (dup / p.name).write_bytes(p.read_bytes())
    (dup / "sweep_val_dummy_again.jsonl.gz").write_bytes(src[0].read_bytes())
    with pytest.raises(SystemExit, match="duplicate"):
        method3.main(["merge", "--fixture", "--split", "val", "--models", "dummy", "--out-dir", str(dup)])
    miss = tmp_path / "miss"
    miss.mkdir()
    for p in src:
        rows = [r for r in _read(p) if r["clip_id"] != "0002"]
        with gzip.open(miss / p.name, "wt", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    capsys.readouterr()
    assert method3.main(["merge", "--fixture", "--split", "val", "--models", "dummy", "--out-dir", str(miss)]) == 0
    out = capsys.readouterr().out
    assert "missing" in out and "0002" in out


def test_score_reads_both_judges(work):
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    assert {c["name"] for c in score["configs"]} == _grid_names(work)
    for c in score["configs"]:
        assert set(c["judges"]) == {"mean", "p95"}
        for j in c["judges"].values():
            assert {"latency_ms", "delta_miou_at_equal_latency", "latency_saved_at_equal_miou"} <= set(j)
        assert c["judges"]["mean"]["latency_ms"] == c["latency"]["mean_ms"]
        assert c["judges"]["p95"]["latency_ms"] == c["latency"]["p95_ms"]
        assert len(c["miou_by_position"]) == 2 and len(c["per_class_iou"]) == 19
        assert 0 <= c["fallback_share"] <= 1 and 0 < c["recomputed_share"] <= 1 and c["gflops_mean"] > 0
    ff = score["fullframe"]
    assert ff["scales"] == [1.0, 0.5, 0.25] and len(ff["median_ms"]) == len(ff["p95_ms"]) == 3
    b = score["bounds"]
    assert set(b["reuse_bound"]) == {"1", "0.5"} and len(b["reuse_bound"]["1"]) == 9  # lags 1..9
    for rule in b["rules"]:
        assert {"motion", "oracle", "raw"} <= set(rule)
        assert set(rule["motion"]["judges"]) == {"mean", "p95"}


def test_finalists_are_on_a_front_of_a_judge(work):
    configs = json.loads((work / "out" / "score_val_dummy.json").read_text())["configs"]
    finalists = json.loads((work / "finalists.json").read_text())["finalists"]
    assert finalists
    for f in finalists:
        assert f["model"] == "dummy" and f["name"] in {c["name"] for c in configs}
        for judge in ("mean", "p95"):
            if f"pareto_{judge}" in f["reasons"]:
                lat, miou = f["train"][f"{judge}_ms"], f["train"]["miou"]
                assert not any(c["judges"][judge]["latency_ms"] < lat and c["miou"] > miou for c in configs)


def test_report_has_both_judges_with_intervals(work):
    rep = json.loads((work / "val.json").read_text())
    finalists = json.loads((work / "finalists.json").read_text())["finalists"]
    assert rep["split"] == "val" and len(rep["finalists"]) == len(finalists)
    for f in rep["finalists"]:
        assert set(f["bootstrap"]) == {"mean", "p95"}
        for boot in f["bootstrap"].values():
            assert boot["n"] == 50 and len(boot["miou"]) == 2


def test_report_matches_the_finalists_scores(work):
    configs = {c["name"]: c for c in json.loads((work / "out" / "score_val_dummy.json").read_text())["configs"]}
    for f in json.loads((work / "val.json").read_text())["finalists"]:
        assert f["miou"] == pytest.approx(configs[f["name"]]["miou"])  # same split, same replay
        assert f["latency"] == pytest.approx(configs[f["name"]]["latency"])


def test_val_sweep_needs_the_finalists():
    with pytest.raises(SystemExit, match="finalists"):
        method3.main(["sweep", "--split", "val", "--models", "dummy"])


@pytest.mark.parametrize("split", ["train", "train_all"])
def test_report_refuses_train(work, split):
    with pytest.raises(SystemExit, match="val"):
        method3.main(["report", "--fixture", "--split", split, "--models", "dummy", "--root", str(work / "cache"),
                      "--finalists", str(work / "finalists.json")])


def test_chunk_selects_contiguous_blocks_of_sorted_clip_ids():
    ids = ["0003", "0001", "0002", "0004", "0005"]
    assert method3.select_clips(ids, None, "0/2") == [1, 2, 0]
    assert method3.select_clips(ids, None, "1/2") == [3, 4]
    assert method3.select_clips(ids, "0005,0001", None) == [1, 4]
    with pytest.raises(SystemExit, match="0009"):
        method3.select_clips(ids, "0009", None)


def test_notebook_calls_existing_commands(tmp_path):
    nb = tmp_path / "m3.ipynb"
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_method3_notebook.py"), str(nb)], check=True)
    src = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code")
    calls = re.findall(r'run\(f?"([^"]+)"\)', src)
    assert {c.split()[0] for c in calls} >= {"stats", "cache", "sweep", "bounds"}  # score, finalists, report run locally
    for call in calls:
        args = shlex.split(re.sub(r"\{[^}]+\}", "0", call))
        method3.parse_args(args)  # exits on an unknown command or flag


def test_notebook_accepts_pytest_quiet_summary(tmp_path):
    nb = tmp_path / "m3.ipynb"
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_method3_notebook.py"), str(nb)], check=True)
    src = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code")
    pattern = re.search(r're\.search\(r"([^"]+)", test\.stdout, re\.M\)', src)
    assert pattern, "the notebook must search the summary line with a regex"
    assert re.search(pattern.group(1), ".\n1 passed, 10 deselected in 71.57s (0:01:11)\n", re.M)  # the line of `pytest -q`
    assert not re.search(pattern.group(1), "11 passed in 1s\n", re.M)
