"""Method 6 CLI end to end on the synthetic fixture (CPU): bound -> merge -> score, with Method 5's
scores of the same fixture as the separable bar and the reproduction check."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import method2_fullframe, method5, method6  # noqa: E402
from tests.test_method5_cli import FIXTURE_CFG, _fake_final, _fake_latency  # noqa: E402

GRID = ["--cell", "8", "--scales", "0.5", "--saliencies", "E,S", "--alphas", "0.5", "--sigmas", "1",
        "--kernels", "1.5", "--batch", "2"]
M5_GRID = ["--cell", "8", "--scales", "0.5", "--saliencies", "E,S", "--alphas", "0.25,0.5", "--sigmas", "1",
           "--timings", "now", "--batch", "2"]
LINES = 2 + 1 + 1 + 1  # (E, S) local + uniform + separable + resize


def _read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _common():
    return ["--fixture", "--split", "val", "--models", "dummy"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("m6")
    assert method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(d / "ff")]) == 0
    _fake_latency(d / "lat.json")
    _fake_final(d / "final-s1.json")
    lat = ["--fullframe", str(d / "ff"), "--latency", str(d / "lat.json"), "--e2e", str(d / "final-s*.json")]
    m5 = ["--out-dir", str(d / "m5")]
    assert method5.main(["bound", *_common(), *M5_GRID, *m5]) == 0
    assert method5.main(["merge", *_common(), *m5]) == 0
    assert method5.main(["score", *_common(), *m5, *lat]) == 0
    out = ["--out-dir", str(d / "out")]
    for chunk in ("0/2", "1/2"):
        assert method6.main(["bound", *_common(), *GRID, "--chunk", chunk, *out]) == 0
    assert method6.main(["merge", *_common(), *out]) == 0
    assert method6.main(["score", *_common(), *out, *lat, "--method5-scores", str(d / "m5")]) == 0
    return d


def test_the_default_grid_is_the_specs():
    args = method6.parse_args(["bound"])
    for model in ("pidnet_s", "bisenetv2"):
        cfgs = method6.grid_configs(args, model)
        local = [c for c in cfgs if c.form == "2d" and not c.uniform]
        sep = [c for c in cfgs if c.form == "sep"]
        assert len(cfgs) == len({c.name() for c in cfgs}) == 144 + 4 + 4
        assert len(local) == 4 * 3 * 3 * 2 * 2
        assert sorted(c.scale for c in cfgs if c.uniform) == [0.25, 0.375, 0.5, 0.75]
        assert {c.kernel for c in local} == {4.0, 12.0} and {c.alpha for c in local} == {0.1, 0.25, 0.5}
        assert {(c.scale, c.saliency, c.alpha, c.sigma) for c in sep} == {
            (s, *v) for s, v in method6.SEP_BAR[model].items()}
    assert args.split == "train_all" and args.cell == 16


def test_the_separable_bars_are_method5s_files():
    """SEP_BAR (the rows bound runs) is Method 5's best now configuration per scale in its files."""
    for model, bars in method6.SEP_BAR.items():
        found = method6.separable_bars(method6.method5_rows(method6.M5_SCORES, "train_all", model))
        for s, (k, a, sig) in bars.items():
            assert found[s]["name"] == f"5o/c{s:g}/{k}/a{a:g}/s{sig:g}/now", (model, s)


def test_bound_rows(work):
    rows = _read(work / "out" / "bound_val_dummy_c0of2.jsonl.gz")
    assert len(rows) == LINES * 2
    assert {r["name"] for r in rows} == {"6o/c0.5/E/a0.5/s1/k1.5", "6o/c0.5/S/a0.5/s1/k1.5", "6o/c0.5/uniform",
                                         "6o/c0.5/sep/E/a0.25/s1", "ref/c0.5/resize"}
    for r in rows:
        assert np.asarray(r["conf"]).shape == (19, 19)
        assert ("dist" in r) == (r["config"].get("form") == "2d" and r["config"].get("alpha", 0) > 0)


def test_score_reads_both_checks_and_the_rule(work):
    score = json.loads((work / "out" / "score_val_dummy.json").read_text())
    assert all(v["ok"] for v in score["pipeline_check"].values())
    repro = score["reproduction_check"]
    assert list(repro) == ["6o/c0.5/sep/E/a0.25/s1"] and repro["6o/c0.5/sep/E/a0.25/s1"]["ok"]
    assert repro["6o/c0.5/sep/E/a0.25/s1"]["gap"] == pytest.approx(0.0, abs=1e-12)
    v = score["verdict"]
    assert v["complete"] and v["pipeline_check_ok"] and v["reproduction_ok"] and v["passes"] in (True, False)
    ps = v["per_scale"]["0.5"]
    bar = score["separable_bars"]["0.5"]
    assert ps["delta_vs_separable"] == pytest.approx(ps["delta_vs_fullframe"] - bar["delta_vs_fullframe"])
    assert ps["both"] == (ps["delta_vs_fullframe"] >= 0.01 and ps["delta_vs_separable"] >= 0.005)
    local = [c for c in score["configs"] if c["config"].get("form") == "2d" and c["config"]["alpha"] > 0]
    assert all(set(c["distortion"]) == {"aniso_mean", "aniso_p95", "mag_p5", "mag_p95", "t"} for c in local)
    verdict = json.loads((work / "out" / "verdict_val.json").read_text())
    assert verdict["passes"] == v["passes"]


def test_rule_is_not_read_without_the_reproduction_rows(work, tmp_path):
    out = ["--out-dir", str(tmp_path)]
    assert method6.main(["bound", *_common(), *GRID, "--no-sep", *out]) == 0
    assert method6.main(["merge", *_common(), *out]) == 0
    lat = ["--fullframe", str(work / "ff"), "--latency", str(work / "lat.json"), "--e2e", str(work / "final-s*.json")]
    assert method6.main(["score", *_common(), *out, *lat, "--method5-scores", str(work / "m5")]) == 0
    v = json.loads((tmp_path / "score_val_dummy.json").read_text())["verdict"]
    assert v["passes"] is None and "reproduction" in v["state"]


def test_val_is_refused():
    with pytest.raises(SystemExit, match="train-only"):
        method6.main(["bound", "--split", "val", "--models", "dummy"])
