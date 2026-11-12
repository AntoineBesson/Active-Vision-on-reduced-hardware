"""Method 4 phase 1 CLI end to end on the synthetic fixture (CPU): bound -> merge -> score."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from scripts import method2_fullframe, method3, method4  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CFG = str(REPO / "configs" / "baseline_fixture.yaml")
CELLS = 8  # GlimpseGrid((16, 32), 12, 2): inner 8 -> 2 x 4
GRID = ["--ctx-scales", "0.25,0.5", "--glimpse-sizes", "12", "--margin", "2", "--a-max", "1,inf", "--ks", "1,3",
        "--gammas", "0.1", "--homography", "orb", "--batch", "4"]  # IC-LK needs frames larger than the fixture's
CONFIGS = 2 * (1 + 2 + 2) * 3  # context x (N, A x 2 ages, B x 2 ages) x (k1, k3, gamma)


def _fake_latency(path: Path) -> Path:
    def rec(name, ms, level="graph", **kw):
        return {"name": name, "level": level, "gflops": ms / 10, "cuda_event": {"median_ms": ms, "p95_ms": ms * 1.5},
                **kw}

    area = [rec(f"area/{s}", 20 * s * s + 1, model="dummy", kind="frame", scale=s) for s in (1.0, 0.5, 0.25)]
    batch = [rec(f"batch/{k}", 0.5 + 0.4 * k, model="dummy", size=12, k=k) for k in range(1, CELLS + 1)]
    reuse = [rec("gray", 0.1, "eager", op="gray"), rec("warp", 0.4, "eager", op="warp"),
             rec("fusion", 0.2, "eager", op="fusion"), rec("motion", 0.3, "eager", op="motion", size=12),
             rec("h", 2.0, "eager", op="homography", homography="iclk")]
    reuse += [rec(f"post/{s}", 0.5, "eager", op="post", stride=8, scale=s) for s in (1.0, 0.5, 0.25)]
    reuse += [rec(f"gp/{k}", 0.05 * k, "eager", op="glimpse_post", stride=8, size=12, k=k) for k in range(1, CELLS + 1)]
    path.write_text(json.dumps({"records": {"area": area, "batch": batch, "reuse": reuse}}))
    return path


def _fake_final(path: Path, model: str = "dummy", scales=(1.0, 0.5, 0.25), factor: float = 1.0) -> Path:
    recs = [{"name": f"final/fullframe/{model}/{s:g}", "kind": "fullframe", "model": model, "scale": s, "level": level,
             "cuda_event": {"median_ms": (22 * s * s + 2) * factor * (3 if level == "eager" else 1),
                            "p95_ms": (22 * s * s + 2) * factor * 1.2}}
            for s in scales for level in ("eager", "graph")]
    recs.append({"name": "final/x", "kind": "finalist", "level": "graph", "cuda_event": {"median_ms": 1, "p95_ms": 1}})
    path.write_text(json.dumps({"records": {"final": recs}}))
    return path


def _read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _common(d: Path):
    return ["--fixture", "--split", "val", "--models", "dummy", "--root", str(d / "cache")]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("m4")
    assert method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(d / "ff")]) == 0
    _fake_latency(d / "lat.json")
    _fake_final(d / "final-s1.json")
    _fake_final(d / "final-s2.json", factor=1.1)
    assert method3.main(["cache", *_common(d), "--full-scales", "1", "--glimpse-sizes", "12", "--ctx-scales",
                         "0.25,0.5", "--margin", "2", "--homography", "orb"]) == 0
    out = ["--out-dir", str(d / "out")]
    for chunk in ("0/2", "1/2"):  # 2 clips, then 1
        assert method4.main(["bound", *_common(d), *GRID, "--chunk", chunk, *out]) == 0
    assert method4.main(["merge", *_common(d), *out]) == 0
    assert method4.main(["score", *_common(d), *out, "--fullframe", str(d / "ff"), "--latency", str(d / "lat.json"),
                         "--e2e", str(d / "final-s*.json")]) == 0
    return d


def test_the_default_grid_is_the_specs():
    args = method4.parse_args(["bound"])
    cfgs = method4.grid_configs(args)
    assert len(cfgs) == len({c.name() for c in cfgs}) == 693
    assert sum(c.variant == "off" for c in cfgs) == 63 and sum(c.variant in ("iclk", "none") for c in cfgs) == 630
    assert args.homography == "iclk" and args.split == "train_all" and args.margin == 32


def test_bound_covers_the_grid_and_equals_a_direct_replay(work):
    from canvas.replay import CanvasConfig, replay_canvas
    from reuse.replay import ClipCache

    rows = _read(work / "out" / "bound_val_dummy.jsonl.gz")
    assert len({r["name"] for r in rows}) == CONFIGS and len(rows) == CONFIGS * 3
    assert {r["config"]["variant"] for r in rows} == {"off", "iclk", "none"}
    caches = {}
    for r in rows:
        cfg = CanvasConfig(**r["config"])
        assert cfg.name() == r["name"]
        if r["clip_id"] not in caches:
            caches[r["clip_id"]] = ClipCache(work / "cache" / "val" / "dummy" / f"{r['clip_id']}.npz")
        ref = replay_canvas(caches[r["clip_id"]], cfg, "cpu", homography="orb")
        assert np.array_equal(np.asarray(r["conf"]), ref.conf), r["name"]
        assert r["k"] == ref.k.tolist() and len(r["k"]) == 10 and r["num_cells"] == CELLS
        assert r["canvas_share"] == pytest.approx(ref.canvas_share[1:].mean(), abs=1e-5)
    assert any(r["canvas_share"] > 0 for r in rows)
    assert not any(r["canvas_share"] for r in rows if r["config"]["variant"] == "off")


def _sorted_lines(path: Path):
    return sorted(json.dumps(r, sort_keys=True) for r in _read(path))


def test_bound_resumes_after_a_cut_off_run(work, tmp_path, capsys):
    src = work / "out" / "bound_val_dummy_c0of2.jsonl.gz"
    cut = tmp_path / src.name
    data = src.read_bytes()
    cut.write_bytes(data[:len(data) - 40])  # the second clip's gzip member loses its end
    args = ["bound", *_common(work), *GRID, "--chunk", "0/2", "--out-dir", str(tmp_path)]
    capsys.readouterr()
    assert method4.main(args) == 0
    out = capsys.readouterr().out
    assert "dropped a cut-off tail, kept 1 clips" in out and "1 clips to run, 1 already" in out
    assert _sorted_lines(cut) == _sorted_lines(src)
    assert method4.main(args) == 0  # nothing left to do
    assert "0 clips to run, 2 already" in capsys.readouterr().out
    assert _sorted_lines(cut) == _sorted_lines(src)


def test_bound_refuses_a_file_written_with_another_grid(work, tmp_path):
    src = work / "out" / "bound_val_dummy_c0of2.jsonl.gz"
    (tmp_path / src.name).write_bytes(src.read_bytes())
    other = [a if a != "1,3" else "1,2" for a in GRID]
    with pytest.raises(SystemExit, match="another grid"):
        method4.main(["bound", *_common(work), *other, "--chunk", "0/2", "--out-dir", str(tmp_path)])


def test_bound_refuses_val_and_a_cache_with_another_homography(work, tmp_path):
    with pytest.raises(SystemExit, match="train-only"):
        method4.main(["bound", "--split", "val", "--models", "dummy"])
    iclk = [a for a in GRID if a not in ("--homography", "orb")]  # the default: IC-LK, but the fixture cache is ORB's
    with pytest.raises(ValueError, match="homography 'iclk'"):
        method4.main(["bound", *_common(work), *iclk, "--out-dir", str(tmp_path)])


def test_merge_refuses_duplicates_and_cut_off_chunks_and_lists_missing(work, tmp_path, capsys):
    src = sorted((work / "out").glob("bound_val_dummy_*.jsonl.gz"))
    assert len(src) == 2
    dup = tmp_path / "dup"
    dup.mkdir()
    for p in src:
        (dup / p.name).write_bytes(p.read_bytes())
    (dup / "bound_val_dummy_again.jsonl.gz").write_bytes(src[0].read_bytes())
    with pytest.raises(SystemExit, match="duplicate"):
        method4.main(["merge", *_common(work), "--out-dir", str(dup)])
    short = tmp_path / "short"
    short.mkdir()
    rows = _read(src[0])
    with gzip.open(short / src[0].name, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows[:-3])
    with pytest.raises(SystemExit, match="different numbers"):
        method4.main(["merge", *_common(work), "--out-dir", str(short)])
    assert not (short / "bound_val_dummy.jsonl.gz").exists()
    miss = tmp_path / "miss"
    miss.mkdir()
    (miss / src[0].name).write_bytes(src[0].read_bytes())
    capsys.readouterr()
    assert method4.main(["merge", *_common(work), "--out-dir", str(miss)]) == 0
    out = capsys.readouterr().out
    assert "missing clips" in out and "0003" in out


def test_score_reads_both_judges_against_both_references(work):
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    assert len(score["configs"]) == CONFIGS
    ff, e2e = score["fullframe"], score["fullframe_e2e"]
    assert ff["scales"] == [1.0, 0.5, 0.25] and e2e["sessions"] == 2 and e2e["miou"] == ff["miou"]
    assert e2e["median_ms"] == pytest.approx([(22 * s * s + 2) * 1.05 for s in ff["scales"]])  # graph level, median of 2
    for c in score["configs"]:
        for judges in ("judges", "judges_e2e"):
            assert set(c[judges]) == {"mean", "p95"}
            assert c[judges]["mean"]["latency_ms"] == c["latency"]["mean_ms"]
            assert c[judges]["p95"]["latency_ms"] == c["latency"]["p95_ms"]
        assert c["latency"]["frames"] == 30 and c["clips"] == 3
        assert len(c["miou_by_position"]) == 2 and len(c["per_class_iou"]) == 19
        budget, saved = c["policy_budget_ms"], c["judges"]["mean"]["latency_saved_at_equal_miou"]
        assert (math.isnan(budget) and math.isnan(saved)) or budget == saved
        assert ("with_motion" in c) == (c["config"]["variant"] == "iclk")
        if "with_motion" in c:  # 0.3 ms on every frame but the first of each clip
            assert c["with_motion"]["latency"]["mean_ms"] == pytest.approx(c["latency"]["mean_ms"] + 0.3 * 27 / 30)
        assert 0 <= c["glimpsed_share"] <= 1 and c["k_max"] <= CELLS
    assert set(score["best"]) == {f"{v}/{j}/{k}" for v in "NAB" for j in ("judges", "judges_e2e") for k in ("mean", "p95")}


def test_score_latency_is_the_frames_sum_of_parts(work):
    """4o-B, k = 3: context (2.25 + 0.5) + 3 glimpses (1.7 + 0.15), + fusion (0.2) after frame 0."""
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    c = next(c for c in score["configs"] if c["name"] == "4o-B/c0.25/g12/ainf/k3")
    frame = 20 * 0.25 * 0.25 + 1 + 0.5 + 0.5 + 0.4 * 3 + 0.05 * 3
    assert c["latency"]["mean_ms"] == pytest.approx(frame + 0.2 * 27 / 30)
    assert c["latency"]["p95_ms"] == pytest.approx(1.5 * (frame + 0.2))
    assert c["k_mean"] == 3 and c["glimpsed_share"] == pytest.approx(3 / CELLS)


def test_score_writes_the_verdict_on_the_whole_clip_set_only(work, tmp_path, capsys):
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    v = score["verdict"]
    assert v["complete"] and v["clips"] == 3 and isinstance(v["passes"], bool) and v["margin"] == 0.01
    overall = json.loads((work / "out" / "verdict_val.json").read_text())
    assert overall["passes"] == v["passes"] and set(overall["models"]) == {"dummy"}
    capsys.readouterr()
    assert method4.main(["score", *_common(work), "--out-dir", str(tmp_path), "--fullframe", str(work / "ff"),
                         "--latency", str(work / "lat.json"), "--e2e", str(tmp_path / "none*.json"),
                         "--bound", str(work / "out" / "bound_val_dummy_c0of2.jsonl.gz")]) == 0  # 2 of 3 clips
    part = json.loads((tmp_path / "score_val_dummy.json").read_text())
    assert part["verdict"]["passes"] is None and not part["verdict"]["complete"] and part["verdict"]["clips"] == 2
    assert part["fullframe_e2e"] is None and "judges_e2e" not in part["configs"][0]
    assert not (tmp_path / "verdict_val.json").exists()
    assert "not read (partial: 2 clips)" in capsys.readouterr().out


def _cfg(variant, delta, p95=float("nan")):
    def judge(d):
        return {"latency_ms": 9.0, "delta_miou_at_equal_latency": d, "latency_saved_at_equal_miou": 1.0}

    return {"name": f"{variant}/{delta}", "miou": 0.7, "config": {"variant": variant},
            "judges": {"mean": judge(delta), "p95": judge(p95)}}


def test_the_verdict_follows_the_specs_rule():
    nan = float("nan")
    passing = [_cfg("iclk", -0.02), _cfg("none", 0.0100001), _cfg("off", 0.5), _cfg("none", nan)]
    v = method4.verdict_of(passing, complete=True, clips=400)
    assert v["passes"] is True and v["best"]["name"] == "none/0.0100001" and v["best"]["readable"] == 2
    assert method4.verdict_of([_cfg("iclk", 0.0099), _cfg("off", 0.5)], True, 400)["passes"] is False  # N does not count
    assert method4.verdict_of([_cfg("iclk", -0.01, p95=0.5)], True, 400)["passes"] is False  # nor does p95
    unreadable = method4.verdict_of([_cfg("iclk", nan), _cfg("off", 0.5)], True, 400)
    assert unreadable["best"] is None and unreadable["passes"] is False  # nothing readable: fails
    assert method4.verdict_of(passing, complete=False, clips=100)["passes"] is None  # a chunk is not the check


def test_e2e_curve_needs_every_scale(tmp_path):
    ref = {"scales": [1.0, 0.5, 0.25], "miou": [0.7, 0.6, 0.5]}
    _fake_final(tmp_path / "f-s1.json", scales=(1.0, 0.5))
    assert method4.e2e_curves(str(tmp_path / "f-s*.json"), "dummy", ref) is None
    _fake_final(tmp_path / "f-s1.json")
    e2e = method4.e2e_curves(str(tmp_path / "f-s*.json"), "dummy", ref)
    assert e2e["median_ms"] == pytest.approx([24.0, 7.5, 3.375]) and e2e["p95_ms"] == pytest.approx([28.8, 9.0, 4.05])
    assert e2e["curves"]["mean"].miou_at(24.0) == pytest.approx(0.7)
    assert method4.e2e_curves(str(tmp_path / "f-s*.json"), "pidnet_s", ref) is None


def test_the_real_final_sessions_give_the_measured_curve():
    """The committed T4 `final` sessions hold the full frame at the five scales of the curve."""
    ref = {"scales": [1.0, 0.75, 0.5, 0.375, 0.25], "miou": [0.76, 0.75, 0.68, 0.6, 0.5]}
    for model, at1 in (("pidnet_s", 21.5), ("bisenetv2", 33.3)):
        e2e = method4.e2e_curves(method4.DEFAULT_E2E, model, ref)
        assert e2e is not None and e2e["sessions"] == 3
        assert e2e["median_ms"][0] == pytest.approx(at1, rel=0.05)
        assert e2e["median_ms"] == sorted(e2e["median_ms"], reverse=True)
