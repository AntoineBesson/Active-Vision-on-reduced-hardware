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
    assert args.split == "train_all" and args.cell == 16 and args.resample == "antialias"  # spec, amended 2026-11-13


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
        for judges in ("judges", "judges_e2e", "judges_same_parts"):
            assert set(c[judges]) == {"mean", "p95"}
            assert c[judges]["mean"]["latency_ms"] == c["latency"]["mean_ms"]
        # the budget is read on the full frame priced with the post-processing, as the configuration is
        budget, saved = c["budget_ms"], c["judges_same_parts"]["mean"]["latency_saved_at_equal_miou"]
        # (on the frontier of that curve: a number where the judges' reading is one, and the same)
        assert math.isnan(saved) or budget == pytest.approx(saved)
        # on that curve, ΔmIoU at equal latency is exactly the rule's reading (mIoU above the full frame at c)
        assert c["judges_same_parts"]["mean"]["delta_miou_at_equal_latency"] == pytest.approx(c["delta_vs_fullframe"])
        assert len(c["per_class_iou"]) == 19 and c["clips"] == 3
        if c["config"]["alpha"] == 0:
            assert c["delta_vs_uniform"] == 0
            assert c["delta_vs_fullframe"] == pytest.approx(0, abs=1e-12)  # antialias: the uniform pass is the baseline
    assert set(score["best"]) == {f"{k}/{tm}/{j}/{m}" for k in "EBS" for tm in ("now", "prev")
                                  for j in ("judges", "judges_e2e", "judges_same_parts") for m in ("mean", "p95")}
    same = score["fullframe_same_parts"]
    assert same["miou"] == ff["miou"] and all(s > m for s, m in zip(same["median_ms"], ff["median_ms"]))
    assert score["best"]["B/now/judges/mean"] is None  # B was not in this grid


def test_score_on_a_partial_set_does_not_read_the_rule(work, tmp_path, capsys):
    capsys.readouterr()
    assert method5.main([*_score_args(work, tmp_path), "--bound",
                         str(work / "out" / "bound_val_dummy_c0of2.jsonl.gz")]) == 0  # 2 of 3 clips
    part = json.loads((tmp_path / "score_val_dummy.json").read_text())
    assert part["verdict"]["passes"] is None and not part["verdict"]["complete"] and part["verdict"]["clips"] == 2
    assert "partial" in part["verdict"]["state"] and not (tmp_path / "verdict_val.json").exists()
    assert "not read (partial: 2 clips)" in capsys.readouterr().out


def test_score_without_the_dataset_reads_the_check_not_the_rule(work, tmp_path, capsys):
    """The README's pipeline check: one file scored with --no-expect (no clip list): the check per
    scale is read and printed, the rule is not."""
    capsys.readouterr()
    assert method5.main([*_score_args(work, tmp_path), "--no-expect", "--bound",
                         str(work / "out" / "bound_val_dummy.jsonl.gz")]) == 0  # all 3 clips
    score = json.loads((tmp_path / "score_val_dummy.json").read_text())
    assert score["pipeline_check"]["0.5"]["ok"] and score["verdict"]["passes"] is None
    assert not score["verdict"]["complete"] and score["verdict"]["clips"] == 3
    assert list(score["configs"][0]["per_class_iou"]) == [str(c) for c in range(19)]
    assert "c0.5 ok" in capsys.readouterr().out and not (tmp_path / "verdict_val.json").exists()


def test_the_plans_supersampling_fails_the_pipeline_check_on_the_fixture(work, tmp_path, capsys):
    """With the 2 x 2 mean, the uniform warp is not the baseline's resize: on this sharp-edged
    fixture the gap is far above 0.1 point, and the verdict is not read."""
    out = ["--out-dir", str(tmp_path)]
    assert method5.main(["bound", *_common(), *GRID, "--resample", "supersample", *out]) == 0
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


def _cfg(timing, delta, alpha=0.5, curve=float("nan")):
    """``delta``: mIoU above the full frame at the same scale (the rule); ``curve``: ΔmIoU at equal
    mean latency on the primary curve (reported only)."""
    def judge(d):
        return {"latency_ms": 9.0, "delta_miou_at_equal_latency": d, "latency_saved_at_equal_miou": 1.0}

    return {"name": f"{timing}/{alpha}/{delta}/{curve}", "miou": 0.7, "latency": {"mean_ms": 9.0}, "budget_ms": 1.0,
            "config": {"scale": 0.5, "alpha": alpha, "saliency": None if alpha == 0 else "E", "timing": timing},
            "delta_vs_fullframe": delta, "judges": {"mean": judge(curve), "p95": judge(float("nan"))}}


OK = {"0.5": {"ok": True}}


def test_verdict_needs_the_margin_above_the_full_frame_at_the_same_scale():
    assert method5.verdict_of([_cfg("now", 0.009)], True, 400, OK)["passes"] is False
    v = method5.verdict_of([_cfg("now", 0.011), _cfg("now", -0.02), _cfg("now", float("nan"))], True, 400, OK)
    assert v["passes"] is True and v["best"]["delta_vs_fullframe"] == 0.011 and v["best"]["readable"] == 2
    assert v["best"]["scale"] == 0.5 and v["best"]["latency_ms"] == 9.0 and "amended 2026-11-13" in v["rule"]
    assert method5.verdict_of([_cfg("prev", 0.05)], True, 400, OK)["passes"] is False  # prev is reported only
    assert method5.verdict_of([_cfg("now", 0.05, alpha=0.0)], True, 400, OK)["passes"] is False  # nor the uniform
    assert method5.verdict_of([_cfg("now", 0.05)], False, 100, OK)["passes"] is None  # a chunk is not the check


def test_the_curves_are_reported_but_are_not_the_rule():
    """Amended 2026-11-13: the primary curve (backbone only) is information, not the rule."""
    assert method5.verdict_of([_cfg("now", -0.01, curve=0.5)], True, 400, OK)["passes"] is False
    a, b = _cfg("now", 0.004, curve=0.02), _cfg("now", 0.015, curve=-0.03)
    v = method5.verdict_of([a, b, _cfg("prev", 0.002, curve=0.01)], True, 400, OK)
    assert v["passes"] is True and v["best"]["name"] == b["name"]
    primary = v["curves"]["primary"]
    assert primary["best"]["name"] == a["name"] and primary["best"]["delta_miou_at_equal_latency"] == 0.02
    assert primary["best_prev"]["readable"] == 1
    assert v["curves"]["e2e"]["best"] is None and v["curves"]["same_parts"]["best"] is None  # not in these configs


def test_the_memory_version_under_water_is_reported():
    v = method5.verdict_of([_cfg("now", 0.02), _cfg("prev", -0.004)], True, 400, OK)
    assert v["passes"] and "under water" in v["note"] and v["best_prev"]["delta_vs_fullframe"] == -0.004
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


# --------------------------------------------------------------------------- #
# Phase 2: the static prior P (prior, prior-merge)
# --------------------------------------------------------------------------- #
PRIOR = ["--cell", "8", "--scales", "0.5,0.25"]


def test_prior_clip_counts_the_phase1_error_oracle(vps, oracle_backbone):
    """The counts are phase 1's E saliency (unsmoothed) times the cell area, summed over the clip's
    labelled frames; the rounds are those of the density loop on its saliencies."""
    from eval.harness import frame_to_tensor
    from reuse.method import full_pass
    from warp.oracle import saliency
    from warp.prior import prior_clip, prior_name

    out = prior_clip(vps, 0, oracle_backbone, (0.5, 0.25), "cpu", allow_any_frame_size=True, cell=8)
    labelled = np.flatnonzero(np.asarray(vps.labeled_mask(0), dtype=bool))
    assert out["frames"] == len(labelled) > 0
    for s in (0.5, 0.25):
        ref = torch.zeros(2, 4, dtype=torch.float64)
        for t in labelled:
            frame = frame_to_tensor(vps.load_frame(0, int(t)), "cpu")
            teacher, uniform = full_pass(oracle_backbone, frame, 1.0, (4, 8))[0], full_pass(oracle_backbone, frame, s,
                                                                                              (4, 8))[0]
            ref += saliency("E", teacher, uniform, 0, 8).double()
        c = out[prior_name(s)]
        assert c.dtype == torch.int64 and c.shape == (2, 4) and torch.allclose(c.double() / 64, ref)
    assert set(out["rounds"]) == {f"{k}/c{s:g}" for k in "EBS" for s in (0.5, 0.25)}
    assert all(isinstance(v, int) and 0 <= v <= 50 for v in out["rounds"].values())


def test_prior_cli_chunks_merge_to_the_whole_and_refuse_doubles(tmp_path, capsys):
    from warp.policy import load_prior

    whole, parts = tmp_path / "whole", tmp_path / "parts"
    assert method5.main(["prior", *_common(), *PRIOR, "--out-dir", str(whole)]) == 0
    for chunk in ("0/2", "1/2"):
        assert method5.main(["prior", *_common(), *PRIOR, "--chunk", chunk, "--out-dir", str(parts)]) == 0
    rows = _read(whole / "prior_val_dummy_all.jsonl.gz")
    clips = {r["clip_id"] for r in rows}
    assert len(rows) == 3 * len(clips) and {r["name"] for r in rows} == {"prior/c0.5", "prior/c0.25", "rounds"}
    # resumable: run again, nothing is added
    assert method5.main(["prior", *_common(), *PRIOR, "--out-dir", str(whole)]) == 0
    assert len(_read(whole / "prior_val_dummy_all.jsonl.gz")) == len(rows)
    for src in (whole, parts):
        assert method5.main(["prior-merge", *_common(), "--out-dir", str(src), "--dest", str(src / "res")]) == 0
    for s in (0.5, 0.25):
        a, b = load_prior(whole / "res" / "prior_dummy.pt", "dummy", s), load_prior(parts / "res" / "prior_dummy.pt",
                                                                                    "dummy", s)
        assert a.shape == (2, 4) and torch.equal(a, b) and 0 <= float(a.min()) and float(a.max()) <= 1
    summary = json.loads((whole / "res" / "prior_dummy.json").read_text())
    assert summary["clips"] == len(clips) and summary["cell"] == 8
    assert summary["max_rounds_all"] == max(summary["max_rounds"].values())
    # a chunk merged twice, and a missing clip, are refused
    (parts / "prior_val_dummy_dup.jsonl.gz").write_bytes((parts / "prior_val_dummy_c0of2.jsonl.gz").read_bytes())
    with pytest.raises(SystemExit, match="merged twice"):
        method5.main(["prior-merge", *_common(), "--out-dir", str(parts), "--dest", str(tmp_path / "x")])
    (parts / "prior_val_dummy_dup.jsonl.gz").unlink()
    (parts / "prior_val_dummy_c1of2.jsonl.gz").unlink()
    with pytest.raises(SystemExit, match="missing"):
        method5.main(["prior-merge", *_common(), "--out-dir", str(parts), "--dest", str(tmp_path / "x")])


def test_prior_is_fitted_on_train_all_only():
    with pytest.raises(SystemExit):
        method5.main(["prior", "--split", "val", "--models", "pidnet_s"])


# --------------------------------------------------------------------------- #
# Phase 2, stage B: stream -> merge --kind stream -> score2
# --------------------------------------------------------------------------- #
STREAM = ["--cell", "8", "--scales", "0.5,0.25", "--alphas", "0.1,0.25", "--sigmas", "1", "--batch", "3"]
STREAM_CONFIGS = 2 * 2 + 2  # (0.5, 0.25) x (0.1, 0.25) P + the uniform row at each scale


def _fake_gate(path: Path, variants=("r2u1", "r3u1"), verdict="continue") -> Path:
    groups = {}
    for s in (0.5, 0.25):
        vs = {}
        for i, v in enumerate(variants):
            ff = (22 * s * s + 2) * 0.97  # stage A's full frame a little faster than the final sessions'
            w = ff + (0.05 if v.startswith("r2") else -0.05) * (i + 1)
            per = [{"session": f"s{k}", "w_ms": w + k * 0.01, "ff_ms": ff + k * 0.02, "w_p95_ms": (w + k * 0.01) * 1.2,
                    "ff_p95_ms": (ff + k * 0.02) * 1.1, "delta_ms": w - ff - k * 0.01} for k in (-1, 0, 1)]
            vs[f"{v}/P/n50"] = {"delta_ms": w - ff, "w_ms": w, "w_p95_ms": w * 1.2, "ff_ms": ff, "ff_p95_ms": ff * 1.1,
                                "status": "fits", "per_session": per}
        groups[f"dummy/{s:g}"] = {"model": "dummy", "scale": s, "kept": sorted(vs), "variants": vs}
    path.write_text(json.dumps({"verdict": verdict, "groups": groups}))
    return path


@pytest.fixture(scope="module")
def stage_b(work, tmp_path_factory):
    """The prior fitted on the fixture, both variants streamed in two chunks, merged and scored."""
    d = tmp_path_factory.mktemp("m5b")
    assert method5.main(["prior", *_common(), "--cell", "8", "--scales", "0.5,0.25", "--out-dir", str(d / "pc")]) == 0
    assert method5.main(["prior-merge", *_common(), "--out-dir", str(d / "pc"), "--dest", str(d / "priors")]) == 0
    _fake_gate(d / "gate_a.json")
    run = [*_common(), *STREAM, "--priors", str(d / "priors"), "--gate", str(d / "gate_a.json"), "--out-dir",
           str(d / "s")]
    for v in ("r2u1", "r3u1"):
        for chunk in ("0/2", "1/2"):
            assert method5.main(["stream", *run, "--variant", v, "--chunk", chunk]) == 0
    assert method5.main(["merge", "--kind", "stream", *_common(), "--out-dir", str(d / "s")]) == 0
    assert method5.main(_score2_args(work, d, d / "s")) == 0
    return d


def _score2_args(work: Path, d: Path, out: Path):
    return ["score2", *_common(), "--out-dir", str(out), "--gate", str(d / "gate_a.json"), "--fullframe",
            str(work / "ff"), "--latency", str(work / "lat.json"), "--e2e", str(work / "final-s*.json"),
            "--oracle-scores", str(work / "out"), "--finalists-dir", str(d / "fin")]


def test_the_default_stage_b_grid_is_p_only_after_gate_a():
    for model, n in (("pidnet_s", 3 * 3 * 2 + 3), ("bisenetv2", 2 * 3 * 2 + 2)):
        cfgs = method5.stream_configs(method5.parse_args(["stream", "--variant", "r3u1"]), model)
        assert len(cfgs) == len({c.name() for c in cfgs}) == n
        assert {c.saliency for c in cfgs} == {"P", "uniform"} and {c.alpha for c in cfgs} == {0.0, 0.1, 0.15, 0.25}
        assert {(c.resample, c.unwarp) for c in cfgs} == {("r3", "u1")} and {c.sigma for c in cfgs if c.alpha} == {1, 4}
        assert sorted({c.scale for c in cfgs}) == sorted(method5.STAGE_B[model])


def test_stream_writes_one_row_per_config_and_clip_with_its_positions(stage_b):
    for v in ("r2u1", "r3u1"):
        rows = _read(stage_b / "s" / f"stream_val_dummy_{v}.jsonl.gz")
        assert len(rows) == 3 * STREAM_CONFIGS and {r["variant"] for r in rows} == {v}
        assert {r["clip_id"] for r in rows} == {"0001", "0002", "0003"}
        assert all(r["positions"] == [4, 9] and np.asarray(r["conf"]).shape == (2, 19, 19) for r in rows)
        assert {r["config"]["saliency"] for r in rows} == {"P", "uniform"}


def test_stream_resumes_and_refuses_another_variant(stage_b, tmp_path):
    run = [*_common(), *STREAM, "--priors", str(stage_b / "priors"), "--gate", str(stage_b / "gate_a.json")]
    out = ["--out-dir", str(stage_b / "s")]
    before = _read(stage_b / "s" / "stream_val_dummy_r2u1_c0of2.jsonl.gz")
    assert method5.main(["stream", *run, *out, "--variant", "r2u1", "--chunk", "0/2"]) == 0
    assert _read(stage_b / "s" / "stream_val_dummy_r2u1_c0of2.jsonl.gz") == before
    src = stage_b / "s" / "stream_val_dummy_r2u1_c0of2.jsonl.gz"
    (tmp_path / "stream_val_dummy_r3u1_c0of2.jsonl.gz").write_bytes(src.read_bytes())
    with pytest.raises(SystemExit, match="variant r2u1"):
        method5.main(["stream", *run, "--out-dir", str(tmp_path), "--variant", "r3u1", "--chunk", "0/2"])


def test_stream_refuses_a_variant_or_saliency_gate_a_did_not_keep(stage_b, tmp_path):
    run = [*_common(), *STREAM, "--priors", str(stage_b / "priors"), "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit, match="did not keep r2u2/P/n50"):
        method5.main(["stream", *run, "--gate", str(stage_b / "gate_a.json"), "--variant", "r2u2"])
    with pytest.raises(SystemExit, match="did not keep r2u1/B/n50"):
        method5.main(["stream", *run, "--gate", str(stage_b / "gate_a.json"), "--variant", "r2u1", "--saliencies", "B"])
    with pytest.raises(SystemExit, match="gate A is 'stop'"):
        method5.main(["stream", *run, "--gate", str(_fake_gate(tmp_path / "g.json", verdict="stop")),
                      "--variant", "r2u1"])


def test_val_is_read_only_with_finalists_that_passed_gate_b(tmp_path):
    with pytest.raises(SystemExit, match="--finalists"):
        method5.main(["stream", "--split", "val", "--models", "pidnet_s", "--variant", "r2u1"])
    fin = tmp_path / "finalists_pidnet_s.json"
    cfg = {"scale": 0.25, "saliency": "P", "alpha": 0.25, "sigma": 1.0, "resample": "r2", "unwarp": "u1", "rounds": 50,
           "taps": None}
    entry = {"model": "pidnet_s", "name": "x", "variant": "r2u1", "config": cfg}
    fin.write_text(json.dumps({"gate_b": "stop", "finalists": [entry]}))
    args = method5.parse_args(["stream", "--split", "val", "--models", "pidnet_s", "--variant", "r2u1",
                               "--finalists", str(fin)])
    with pytest.raises(SystemExit, match="gate B is 'stop'"):
        method5.stream_configs(args, "pidnet_s")
    fin.write_text(json.dumps({"gate_b": "continue", "finalists": [entry, dict(entry, variant="r3u1")]}))
    cfgs = method5.stream_configs(args, "pidnet_s")
    assert [c.name() for c in cfgs] == ["5p/c0.25/P/a0.25/s1/r2u1"]


def test_merge_stream_refuses_doubles_and_cut_off_chunks(stage_b, tmp_path):
    src = sorted((stage_b / "s").glob("stream_val_dummy_r2u1_c*.jsonl.gz"))
    assert len(src) == 2
    for p in src:
        (tmp_path / p.name).write_bytes(p.read_bytes())
    (tmp_path / "stream_val_dummy_r2u1_again.jsonl.gz").write_bytes(src[0].read_bytes())
    with pytest.raises(SystemExit, match="duplicate"):
        method5.main(["merge", "--kind", "stream", *_common(), "--out-dir", str(tmp_path)])
    (tmp_path / "stream_val_dummy_r2u1_again.jsonl.gz").unlink()
    rows = _read(src[0])
    with gzip.open(tmp_path / src[0].name, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows[:-1])
    with pytest.raises(SystemExit, match="different numbers"):
        method5.main(["merge", "--kind", "stream", *_common(), "--out-dir", str(tmp_path)])


def test_score2_prices_checks_and_reads_gate_b(stage_b):
    score = json.loads((stage_b / "s" / "score2_val_dummy.json").read_text())
    gate = json.loads((stage_b / "gate_a.json").read_text())
    assert len(score["configs"]) == 2 * STREAM_CONFIGS and score["positions"] == [4, 9]
    for c in score["configs"]:
        price = gate["groups"][f"dummy/{c['config']['scale']:g}"]["variants"][f"{c['variant']}/P/n50"]
        assert c["latency"]["mean_ms"] == price["w_ms"] and c["latency"]["p95_ms"] == price["w_p95_ms"]
        assert set(c["miou_by_position"]) == {"4", "9"} and len(c["per_class_iou"]) == 19
        assert c["delta_vs_fullframe"] is not None and "mean" in c["judges_e2e"]
    # the pipeline check holds r2 to the baseline and reports r3
    for s in ("0.5", "0.25"):
        r2, r3 = score["pipeline_check"]["r2u1"][s], score["pipeline_check"]["r3u1"][s]
        assert r2["checked"] and r2["ok"] and abs(r2["gap"]) <= method5.CHECK_TOLERANCE
        assert not r3["checked"] and r3["ok"] is None and r3["gap"] is not None
    g = score["gate_b"]
    assert g["complete"] and g["verdict"] in ("continue", "stop") and g["best"] is not None
    summary = json.loads((stage_b / "s" / "gate_b_val.json").read_text())
    assert summary["verdict"] == g["verdict"]
    fin = json.loads((stage_b / "fin" / "finalists_dummy.json").read_text())
    assert fin["gate_b"] == g["verdict"] and fin["finalists"]
    assert all(f["config"]["saliency"] != "uniform" and f["reasons"] for f in fin["finalists"])


def test_score2_on_a_partial_set_reads_no_gate(stage_b, work, tmp_path):
    for v in ("r2u1", "r3u1"):
        src = stage_b / "s" / f"stream_val_dummy_{v}_c0of2.jsonl.gz"
        (tmp_path / src.name).write_bytes(src.read_bytes())
    assert method5.main(["merge", "--kind", "stream", *_common(), "--out-dir", str(tmp_path)]) == 0
    assert method5.main(_score2_args(work, stage_b, tmp_path)[:-2] + ["--finalists-dir", str(tmp_path / "fin")]) == 0
    g = json.loads((tmp_path / "score2_val_dummy.json").read_text())["gate_b"]
    assert not g["complete"] and g["verdict"] is None and "partial" in g["state"]
    assert not (tmp_path / "gate_b_val.json").exists()


def _c2(name, sal, scale, miou, ms, delta, variant="r2u1", p95=None):
    return {"name": name, "variant": variant, "config": {"saliency": sal, "scale": scale}, "miou": miou,
            "latency": {"mean_ms": ms, "p95_ms": ms * 1.2 if p95 is None else p95},
            "judges_e2e": {"mean": {"delta_miou_at_equal_latency": delta}}, "delta_vs_fullframe": 0.0,
            "delta_vs_uniform": 0.0}


def test_gate_b_on_hand_made_scores():
    ok = {"r2u1": {"0.25": {"checked": True, "ok": True}}, "r3u1": {"0.25": {"checked": False, "ok": None}}}
    above = [_c2("u", "uniform", 0.25, 0.5, 4.0, 0.002), _c2("a", "P", 0.25, 0.52, 4.0, 0.003),
             _c2("b", "P", 0.25, 0.49, 4.0, -0.01)]
    g = method5.gate_b(above, ok, True, 400)
    assert g["verdict"] == "continue" and g["above"] == ["a"] and g["best"]["name"] == "a"
    assert g["best"]["uniform_delta_miou_at_equal_latency"] == 0.002  # the cheaper resampling alone, shown
    # a uniform row above the curve is a reference: it does not open the gate
    assert method5.gate_b(above[:1] + above[2:], ok, True, 400)["verdict"] == "stop"
    bad = {**ok, "r2u1": {"0.25": {"checked": True, "ok": False}}}
    assert method5.gate_b(above, bad, True, 400)["verdict"] is None
    r3bad = {**ok, "r3u1": {"0.25": {"checked": False, "ok": None}}}
    assert method5.gate_b(above, r3bad, True, 400)["verdict"] == "continue"
    assert method5.gate_b(above, ok, False, 100)["verdict"] is None


def test_finalists_on_hand_made_scores():
    cfgs = [_c2("u", "uniform", 0.25, 0.60, 1.0, 0.0),  # a reference, never a finalist
            _c2("a", "P", 0.25, 0.50, 2.0, 0.0), _c2("b", "P", 0.25, 0.55, 3.0, 0.0),
            _c2("c", "P", 0.25, 0.54, 3.5, 0.0), _c2("d", "P", 0.5, 0.65, 6.0, 0.0, p95=20.0)]
    e2e = {"scales": [0.25, 0.5, 1.0], "median_ms": [2.5, 6.5, 30.0], "p95_ms": [3.0, 8.0, 36.0]}
    fin = {f["name"]: f["reasons"] for f in method5.finalists_of(cfgs, e2e, "m")}
    assert set(fin) == {"a", "b", "d"}
    assert "pareto_mean" in fin["a"] and "best_within_fullframe_s0.25_mean" in fin["a"]
    assert "best_within_fullframe_s0.5_mean" in fin["d"] and "best_within_fullframe_s1_p95" in fin["d"]
    assert "best_within_fullframe_s0.5_p95" in fin["b"]


def test_score2_reads_the_share_of_the_oracles_gain_kept(stage_b, work, tmp_path):
    """Against the phase 1 oracle (E, now) at the same scale, alpha and sigma, from the registered or
    the extension scores."""
    score = json.loads((stage_b / "s" / "score2_val_dummy.json").read_text())
    base = dict(zip(score["fullframe"]["scales"], score["fullframe"]["miou"]))[0.5]
    (tmp_path / "ext").mkdir()
    (tmp_path / "ext" / "score_train_all_dummy.json").write_text(json.dumps(
        {"configs": [{"name": "5o/c0.5/E/a0.25/s1/now", "miou": base + 0.04}]}))
    args = _score2_args(work, stage_b, tmp_path / "out")
    args[args.index("--oracle-scores") + 1] = f"{tmp_path / 'none'},{tmp_path / 'ext'}"
    for v in ("r2u1", "r3u1"):
        (tmp_path / "out").mkdir(exist_ok=True)
        src = stage_b / "s" / f"stream_val_dummy_{v}.jsonl.gz"
        (tmp_path / "out" / src.name).write_bytes(src.read_bytes())
    assert method5.main(args) == 0
    configs = {c["name"]: c for c in json.loads((tmp_path / "out" / "score2_val_dummy.json").read_text())["configs"]}
    c = configs["5p/c0.5/P/a0.25/s1/r2u1"]
    assert c["oracle"]["delta_vs_fullframe"] == pytest.approx(0.04)
    assert c["oracle"]["share_kept"] == pytest.approx(c["delta_vs_fullframe"] / 0.04)
    assert configs["5p/c0.5/P/a0.1/s1/r2u1"]["oracle"] is None  # no oracle at alpha 0.1 in that file


def test_both_prices_are_reported_and_the_paired_one_uses_the_curves_own_full_frame(stage_b):
    score = json.loads((stage_b / "s" / "score2_val_dummy.json").read_text())
    gate = json.loads((stage_b / "gate_a.json").read_text())
    e2e = score["fullframe_e2e"]
    for c in score["configs"]:
        s = c["config"]["scale"]
        v = gate["groups"][f"dummy/{s:g}"]["variants"][f"{c['variant']}/P/n50"]
        j = e2e["scales"].index(s)
        assert c["latency_paired"]["mean_ms"] == pytest.approx(e2e["median_ms"][j] + v["delta_ms"])
        d95 = sorted(p["w_p95_ms"] - p["ff_p95_ms"] for p in v["per_session"])[1]
        assert c["latency_paired"]["p95_ms"] == pytest.approx(e2e["p95_ms"][j] + d95)
        assert c["latency"]["mean_ms"] == v["w_ms"] and "mean" in c["judges_paired"]
    # the fake stage A full frame is 3 % faster than the final one: the spec's price is the lower
    assert all(c["latency"]["mean_ms"] < c["latency_paired"]["mean_ms"] for c in score["configs"])


def test_score2_re_reads_committed_scores_as_it_reads_the_stream_files(stage_b, work, tmp_path):
    """--from-scores re-prices the configurations of score2 files: the same gate and finalists as
    scoring the stream files with the same --pricing and --rule-variants."""
    opts = ["--pricing", "paired", "--rule-variants", "r2u1", "--label", "amended"]
    direct = tmp_path / "direct"
    direct.mkdir()
    for v in ("r2u1", "r3u1"):
        src = stage_b / "s" / f"stream_val_dummy_{v}.jsonl.gz"
        (direct / src.name).write_bytes(src.read_bytes())
    args = _score2_args(work, stage_b, direct)
    args[args.index("--finalists-dir") + 1] = str(direct)
    assert method5.main(args + opts) == 0
    again = tmp_path / "again"
    args = _score2_args(work, stage_b, again)
    args[args.index("--finalists-dir") + 1] = str(again)
    assert method5.main(args + opts + ["--from-scores", str(stage_b / "s")]) == 0
    a = json.loads((direct / "gate_b_val_amended.json").read_text())
    b = json.loads((again / "gate_b_val_amended.json").read_text())
    assert a == b and a["pricing"] == "paired" and a["variants"] == ["r2u1"]
    assert all(n.endswith("r2u1") for n in a["models"]["dummy"]["above"])
    fa = json.loads((direct / "finalists_dummy_amended.json").read_text())
    fb = json.loads((again / "finalists_dummy_amended.json").read_text())
    assert fa == fb and all(f["variant"] == "r2u1" for f in fa["finalists"])
    assert json.loads((again / "score2_val_dummy_amended.json").read_text())["re_read_from"].endswith(
        "score2_val_dummy.json")
    # the spec's files are not touched by a labelled run
    assert not (direct / "gate_b_val.json").exists()


def test_gate_b_reads_only_the_variants_and_price_it_is_given():
    ok = {"r2u1": {"0.25": {"checked": True, "ok": True}}, "r3u1": {"0.25": {"checked": False, "ok": None}}}

    def c(name, variant, spec, paired):
        out = _c2(name, "P", 0.25, 0.5, 4.0, spec, variant)
        out["latency_paired"] = {"mean_ms": 4.1, "p95_ms": 5.0}
        out["judges_paired"] = {"mean": {"delta_miou_at_equal_latency": paired}}
        return out

    cfgs = [c("a", "r3u1", 0.02, 0.01), c("b", "r2u1", 0.004, -0.002)]
    assert method5.gate_b(cfgs, ok, True, 400)["above"] == ["a", "b"]
    assert method5.gate_b(cfgs, ok, True, 400, "paired")["above"] == ["a"]
    g = method5.gate_b(cfgs, ok, True, 400, "paired", ["r2u1"])
    assert g["verdict"] == "stop" and g["above"] == [] and g["best"]["name"] == "b"
    assert method5.gate_b(cfgs, ok, True, 400, "spec", ["r2u1"])["verdict"] == "continue"
