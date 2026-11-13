"""Method 5 phase 1 CLI end to end on the synthetic fixture (CPU): bound -> merge -> score."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import method2_fullframe, method5  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CFG = str(REPO / "configs" / "baseline_fixture.yaml")
GRID = ["--cell", "8", "--scales", "0.5", "--saliencies", "E,S", "--alphas", "0.5", "--sigmas", "1", "--batch", "2"]
CONFIGS = 2 * 2 + 1  # (E, S) x (now, prev) + the uniform pass
LINES = CONFIGS + 1  # + the baseline's resize at 0.5


def _fake_latency(path: Path) -> Path:
    def rec(name, ms, level="graph", **kw):
        return {"name": name, "level": level, "gflops": ms / 10, "cuda_event": {"median_ms": ms, "p95_ms": ms * 1.5},
                **kw}

    area = [rec(f"area/{s}", 20 * s * s + 1, model="dummy", kind="frame", scale=s) for s in (1.0, 0.5, 0.25)]
    reuse = [rec(f"post/{s}", 0.5, "eager", op="post", stride=8, scale=s) for s in (1.0, 0.5, 0.25)]
    path.write_text(json.dumps({"records": {"area": area, "reuse": reuse}}))
    return path


def _fake_final(path: Path, factor: float = 1.0) -> Path:
    recs = [{"name": f"final/fullframe/dummy/{s:g}", "kind": "fullframe", "model": "dummy", "scale": s, "level": "graph",
             "cuda_event": {"median_ms": (22 * s * s + 2) * factor, "p95_ms": (22 * s * s + 2) * factor * 1.2}}
            for s in (1.0, 0.5, 0.25)]
    path.write_text(json.dumps({"records": {"final": recs}}))
    return path


def _read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _common():
    return ["--fixture", "--split", "val", "--models", "dummy"]


def _score_args(work: Path, out: Path):
    return ["score", *_common(), "--out-dir", str(out), "--fullframe", str(work / "ff"), "--latency",
            str(work / "lat.json"), "--e2e", str(work / "final-s*.json")]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Two chunks (2 clips, then 1) with the antialiased resampling, which reproduces the baseline at
    alpha 0, so the fixture passes the pipeline check; merged and scored."""
    d = tmp_path_factory.mktemp("m5")
    assert method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(d / "ff")]) == 0
    _fake_latency(d / "lat.json")
    _fake_final(d / "final-s1.json")
    _fake_final(d / "final-s2.json", factor=1.1)
    out = ["--out-dir", str(d / "out")]
    for chunk in ("0/2", "1/2"):
        assert method5.main(["bound", *_common(), *GRID, "--resample", "antialias", "--chunk", chunk, *out]) == 0
    assert method5.main(["merge", *_common(), *out]) == 0
    assert method5.main(_score_args(d, d / "out")) == 0
    return d


def test_the_default_grid_is_the_specs():
    args = method5.parse_args(["bound"])
    cfgs = method5.grid_configs(args)
    assert len(cfgs) == len({c.name() for c in cfgs}) == 4 * 3 * 3 * 2 * 2 + 4 == 148
    assert sorted(c.scale for c in cfgs if c.uniform) == [0.25, 0.375, 0.5, 0.75]
    assert sum(not c.uniform and c.timing == "now" for c in cfgs) == 72
    assert {c.sigma for c in cfgs if not c.uniform} == {1.0, 4.0} and {c.saliency for c in cfgs} == {"E", "B", "S", None}
    assert args.split == "train_all" and args.cell == 16 and args.resample == "supersample"


def test_bound_writes_one_line_per_config_and_clip(work):
    rows = _read(work / "out" / "bound_val_dummy_c0of2.jsonl.gz")
    assert len(rows) == LINES * 2 and len({r["clip_id"] for r in rows}) == 2
    assert all(np.asarray(r["conf"]).shape == (19, 19) for r in rows)
    assert all(r["resample"] == "antialias" and r["frames"] == 2 for r in rows)
    assert {r["name"] for r in rows} == {"5o/c0.5/E/a0.5/s1/now", "5o/c0.5/E/a0.5/s1/prev", "5o/c0.5/S/a0.5/s1/now",
                                         "5o/c0.5/S/a0.5/s1/prev", "5o/c0.5/uniform", "ref/c0.5/resize"}
    assert len(_read(work / "out" / "bound_val_dummy_c1of2.jsonl.gz")) == LINES


def test_bound_rows_are_bound_clip(work):
    from scripts.method2 import load_dataset, model_config
    from models.registry import build_backbone
    from warp.bound import bound_clip

    rows = _read(work / "out" / "bound_val_dummy_c1of2.jsonl.gz")
    cfg = model_config("dummy", True)
    ds = load_dataset(cfg, "val")
    i = [ds.clip_id(k) for k in range(len(ds))].index(rows[0]["clip_id"])
    model = build_backbone(cfg["backbone"], context={"palette": ds.palette}).eval()
    cfgs = method5.grid_configs(method5.parse_args(["bound", *GRID]))
    ref = bound_clip(ds, i, model, cfgs, "cpu", allow_any_frame_size=True, cell=8, resize_refs=True,
                     resample="antialias")
    assert {r["name"]: np.asarray(r["conf"]).tolist() for r in rows} == {k: v.tolist() for k, v in ref.items()}


def _sorted_lines(path: Path):
    return sorted(json.dumps(r, sort_keys=True) for r in _read(path))


def test_bound_resumes_and_skips_finished_clips_and_drops_a_cut_off_tail(work, tmp_path, capsys):
    src = work / "out" / "bound_val_dummy_c0of2.jsonl.gz"
    cut = tmp_path / src.name
    data = src.read_bytes()
    cut.write_bytes(data[:len(data) - 40])  # the second clip's gzip member loses its end
    args = ["bound", *_common(), *GRID, "--resample", "antialias", "--chunk", "0/2", "--out-dir", str(tmp_path)]
    capsys.readouterr()
    assert method5.main(args) == 0
    out = capsys.readouterr().out
    assert "[method5 bound] bound_val_dummy_c0of2.jsonl.gz: dropped a cut-off tail, kept 1 clips" in out
    assert "1 clips to run, 1 already" in out and "s/clip" in out
    assert _sorted_lines(cut) == _sorted_lines(src)
    assert method5.main(args) == 0  # nothing left to do
    assert "0 clips to run, 2 already" in capsys.readouterr().out
    assert _sorted_lines(cut) == _sorted_lines(src)


def test_bound_refuses_another_grid_or_resampling(work, tmp_path):
    src = work / "out" / "bound_val_dummy_c0of2.jsonl.gz"
    (tmp_path / src.name).write_bytes(src.read_bytes())
    i = GRID.index("--alphas") + 1
    other = GRID[:i] + ["0.25"] + GRID[i + 1:]
    with pytest.raises(SystemExit, match="another grid"):
        method5.main(["bound", *_common(), *other, "--resample", "antialias", "--chunk", "0/2", "--out-dir",
                      str(tmp_path)])
    with pytest.raises(SystemExit, match="written with --resample antialias"):
        method5.main(["bound", *_common(), *GRID, "--resample", "supersample", "--chunk", "0/2", "--out-dir",
                      str(tmp_path)])


def test_bound_refuses_val_without_fixture():
    with pytest.raises(SystemExit, match="train-only"):
        method5.main(["bound", "--split", "val", "--models", "dummy"])


def test_merge_refuses_duplicates_and_cut_off_chunks_and_lists_missing(work, tmp_path, capsys):
    src = sorted((work / "out").glob("bound_val_dummy_c*.jsonl.gz"))
    assert len(src) == 2
    dup = tmp_path / "dup"
    dup.mkdir()
    for p in src:
        (dup / p.name).write_bytes(p.read_bytes())
    (dup / "bound_val_dummy_again.jsonl.gz").write_bytes(src[0].read_bytes())
    with pytest.raises(SystemExit, match="duplicate"):
        method5.main(["merge", *_common(), "--out-dir", str(dup)])
    short = tmp_path / "short"
    short.mkdir()
    rows = _read(src[0])
    with gzip.open(short / src[0].name, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows[:-2])
    with pytest.raises(SystemExit, match="different numbers"):
        method5.main(["merge", *_common(), "--out-dir", str(short)])
    assert not (short / "bound_val_dummy.jsonl.gz").exists()
    miss = tmp_path / "miss"
    miss.mkdir()
    (miss / src[0].name).write_bytes(src[0].read_bytes())
    capsys.readouterr()
    assert method5.main(["merge", *_common(), "--out-dir", str(miss)]) == 0
    out = capsys.readouterr().out
    assert "missing clips" in out and "0003" in out
    assert method5.main(["merge", *_common(), "--out-dir", str(miss)]) == 0  # the merged file is not an input


def test_merge_refuses_mixed_resamplings(work, tmp_path):
    src = work / "out" / "bound_val_dummy_c1of2.jsonl.gz"
    (tmp_path / src.name).write_bytes(src.read_bytes())
    rows = _read(work / "out" / "bound_val_dummy_c0of2.jsonl.gz")
    with gzip.open(tmp_path / "bound_val_dummy_c0of2.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(dict(r, resample="supersample")) + "\n" for r in rows)
    with pytest.raises(SystemExit, match="different resamplings"):
        method5.main(["merge", *_common(), "--out-dir", str(tmp_path)])


def test_score_reads_the_rule_on_the_complete_set(work):
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    v = score["verdict"]
    assert v["complete"] and v["clips"] == 3 and isinstance(v["passes"], bool) and v["margin"] == 0.01
    assert v["pipeline_check_ok"] and v["state"] in ("PASSES", "FAILS")
    check = score["pipeline_check"]
    assert list(check) == ["0.5"] and check["0.5"]["ok"]
    assert check["0.5"]["uniform_minus_resize"] == 0 and check["0.5"]["resize_minus_baseline"] == 0
    overall = json.loads((work / "out" / "verdict_val.json").read_text())
    assert overall["passes"] == v["passes"] and set(overall["models"]) == {"dummy"}
    assert score["resample"] == "antialias"


def test_score_prices_and_reads_every_config(work):
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    assert len(score["configs"]) == CONFIGS and not any(c["name"].startswith("ref/") for c in score["configs"])
    ff, e2e = score["fullframe"], score["fullframe_e2e"]
    assert ff["scales"] == [1.0, 0.5, 0.25] and e2e["sessions"] == 2
    for c in score["configs"]:
        assert c["latency"]["mean_ms"] == pytest.approx(20 * 0.25 + 1 + 0.5)  # area at 0.5 + post
        assert c["latency"]["p95_ms"] == pytest.approx(1.5 * (20 * 0.25 + 1 + 0.5)) and c["latency"]["frames"] == 6
        for judges in ("judges", "judges_e2e"):
            assert set(c[judges]) == {"mean", "p95"}
            assert c[judges]["mean"]["latency_ms"] == c["latency"]["mean_ms"]
        budget, saved = c["budget_ms"], c["judges"]["mean"]["latency_saved_at_equal_miou"]
        assert (math.isnan(budget) and math.isnan(saved)) or budget == saved
        assert len(c["per_class_iou"]) == 19 and c["clips"] == 3
        if c["config"]["alpha"] == 0:
            assert c["delta_vs_uniform"] == 0
            assert c["delta_vs_fullframe"] == pytest.approx(0, abs=1e-12)  # antialias: the uniform pass is the baseline
    assert set(score["best"]) == {f"{k}/{tm}/{j}/{m}" for k in "EBS" for tm in ("now", "prev")
                                  for j in ("judges", "judges_e2e") for m in ("mean", "p95")}
    assert score["best"]["B/now/judges/mean"] is None  # B was not in this grid


def test_score_on_a_partial_set_does_not_read_the_rule(work, tmp_path, capsys):
    capsys.readouterr()
    assert method5.main([*_score_args(work, tmp_path), "--bound",
                         str(work / "out" / "bound_val_dummy_c0of2.jsonl.gz")]) == 0  # 2 of 3 clips
    part = json.loads((tmp_path / "score_val_dummy.json").read_text())
    assert part["verdict"]["passes"] is None and not part["verdict"]["complete"] and part["verdict"]["clips"] == 2
    assert "partial" in part["verdict"]["state"] and not (tmp_path / "verdict_val.json").exists()
    assert "not read (partial: 2 clips)" in capsys.readouterr().out


def test_the_plans_supersampling_fails_the_pipeline_check_on_the_fixture(work, tmp_path, capsys):
    """With the 2 x 2 mean, the uniform warp is not the baseline's resize: on this sharp-edged
    fixture the gap is far above 0.1 point, and the verdict is not read."""
    out = ["--out-dir", str(tmp_path)]
    assert method5.main(["bound", *_common(), *GRID, *out]) == 0
    assert method5.main(["merge", *_common(), *out]) == 0
    capsys.readouterr()
    assert method5.main(_score_args(work, tmp_path)) == 0
    printed = capsys.readouterr().out
    score = json.loads((tmp_path / "score_val_dummy.json").read_text())
    check = score["pipeline_check"]["0.5"]
    assert not check["ok"] and abs(check["gap"]) > 0.001 and check["resize_minus_baseline"] == 0
    assert score["verdict"]["passes"] is None and "pipeline check failed at scales 0.5" in score["verdict"]["state"]
    assert json.loads((tmp_path / "verdict_val.json").read_text())["passes"] is None
    assert "c0.5 NOT ok" in printed and "is NOT READ" in printed


def _cfg(timing, delta, alpha=0.5, p95=float("nan")):
    def judge(d):
        return {"latency_ms": 9.0, "delta_miou_at_equal_latency": d, "latency_saved_at_equal_miou": 1.0}

    return {"name": f"{timing}/{alpha}/{delta}", "miou": 0.7,
            "config": {"scale": 0.5, "alpha": alpha, "saliency": None if alpha == 0 else "E", "timing": timing},
            "judges": {"mean": judge(delta), "p95": judge(p95)}}


OK = {"0.5": {"ok": True}}


def test_verdict_needs_the_margin():
    assert method5.verdict_of([_cfg("now", 0.009)], True, 400, OK)["passes"] is False
    v = method5.verdict_of([_cfg("now", 0.011), _cfg("now", -0.02), _cfg("now", float("nan"))], True, 400, OK)
    assert v["passes"] is True and v["best"]["delta_miou_at_equal_latency"] == 0.011 and v["best"]["readable"] == 2
    assert method5.verdict_of([_cfg("prev", 0.05)], True, 400, OK)["passes"] is False  # prev is reported only
    assert method5.verdict_of([_cfg("now", 0.05, alpha=0.0)], True, 400, OK)["passes"] is False  # nor the uniform
    assert method5.verdict_of([_cfg("now", -0.01, p95=0.5)], True, 400, OK)["passes"] is False  # nor the p95 judge
    assert method5.verdict_of([_cfg("now", 0.05)], False, 100, OK)["passes"] is None  # a chunk is not the check


def test_the_same_size_reading_is_reported_but_is_not_the_rule():
    """The configuration against the full frame at its own scale: information only."""
    a = {**_cfg("now", -0.03), "delta_vs_fullframe": 0.015}
    b = {**_cfg("now", -0.01), "delta_vs_fullframe": 0.004}
    v = method5.verdict_of([a, b, {**_cfg("prev", -0.05), "delta_vs_fullframe": 0.002}], True, 400, OK)
    assert v["passes"] is False  # the rule reads the primary curve
    same = v["same_size"]
    assert same["best"]["name"] == a["name"] and same["at_margin"] is True and same["best_prev"]["readable"] == 1


def test_the_memory_version_under_water_is_reported():
    v = method5.verdict_of([_cfg("now", 0.02), _cfg("prev", -0.004)], True, 400, OK)
    assert v["passes"] and "under water" in v["note"] and v["best_prev"]["delta_miou_at_equal_latency"] == -0.004
    assert method5.verdict_of([_cfg("now", 0.02), _cfg("prev", 0.001)], True, 400, OK)["note"] is None


def test_a_failed_pipeline_check_blocks_the_verdict():
    configs = [_cfg("now", 0.05), {**_cfg("now", 0.0, alpha=0.0), "miou": 0.602}]
    ref = {"scales": [1.0, 0.5], "miou": [0.7, 0.6]}
    check = method5.pipeline_check(configs, {}, ref)
    assert check["0.5"]["gap"] == pytest.approx(0.002) and not check["0.5"]["ok"]
    v = method5.verdict_of(configs, True, 400, check)
    assert v["passes"] is None and "pipeline check failed at scales 0.5" in v["state"]
    within = method5.pipeline_check([{**_cfg("now", 0.0, alpha=0.0), "miou": 0.601}], {0.5: 0.6}, ref)
    assert within["0.5"]["ok"] and within["0.5"]["resize_minus_baseline"] == 0
    missing = method5.pipeline_check([{**_cfg("now", 0.0, alpha=0.0), "config": {"scale": 0.375, "alpha": 0.0}}],
                                     {}, ref)
    assert not missing["0.375"]["ok"] and missing["0.375"]["baseline_miou"] is None
