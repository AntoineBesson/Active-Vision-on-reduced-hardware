"""Method 9: selector pieces (numpy) and the CLI end to end on the synthetic fixture (CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from upsample.selector import assign, crossfit, cut_points, fit_selector, past_features, Selector


def test_past_features_use_only_the_past():
    f = np.arange(12, dtype=float).reshape(4, 3)
    p = past_features(f, half_lives=(1.0,))
    assert np.isnan(p[0]).all()
    assert np.allclose(p[1], f[0])
    assert np.allclose(p[2], 0.5 * f[0] + 0.5 * f[1])
    f2 = f.copy()
    f2[3] = 1e6  # the current frame never enters its own features
    assert np.allclose(past_features(f2, half_lives=(1.0,))[3], p[3])


def test_cut_points_reproduce_shares():
    rng = np.random.default_rng(0)
    s = rng.normal(size=1000)
    shares = np.array([0.2, 0.0, 0.5, 0.3])
    k = assign(s, cut_points(s, shares))
    got = np.bincount(k, minlength=4) / 1000
    assert np.allclose(got, shares, atol=0.01)


def test_selectors_learn_a_signal_and_round_trip():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=(n, 44))
    target = np.clip(np.round(2 + 1.5 * x[:, 0] + 0.3 * rng.normal(size=n)), 0, 4).astype(int)
    for kind in ("S1", "S2"):
        sel = fit_selector(kind, x, target, 5)
        assert np.mean(sel.choose(x) == target) > 0.5
        again = Selector.from_json(json.loads(json.dumps(sel.to_json())))
        assert (again.choose(x) == sel.choose(x)).all()
    groups = np.repeat(np.arange(100), 6)
    oof = crossfit("S2", x, target, groups, 5)
    assert np.mean(oof == target) > 0.5


torch = pytest.importorskip("torch")


def test_cli_on_fixture(tmp_path):
    from scripts import method7, method9

    common = ["--fixture", "--split", "val"]
    run_dir, out9 = tmp_path / "m7", tmp_path / "m9"
    assert method7.main(["run", *common, "--out-dir", str(run_dir), "--device", "cpu", "--guided-scales", "none"]) == 0
    assert method7.main(["merge", *common, "--out-dir", str(run_dir)]) == 0
    oracle = tmp_path / "oracle.json"
    assert method7.main(["oracle", *common, "--out-dir", str(run_dir), "--out", str(oracle), "--budgets", "0.5",
                         "--n-boot", "20", "--seeds", "1"]) == 0
    assert method9.main(["features", *common, "--out-dir", str(out9), "--device", "cpu"]) == 0
    assert method9.main(["merge", *common, "--out-dir", str(out9)]) == 0
    assert method9.main(["select", *common, "--out-dir", str(out9), "--run-dir", str(run_dir), "--oracle", str(oracle),
                         "--n-boot", "20", "--folds", "2"]) == 0
    rep = json.loads((method9.EXP / "select_val_fixture.json").read_text())
    assert set(rep["selectors"]) == {"S1", "S2"}
    sel = method9.EXP / "selectors_dummy_fixture.json"
    assert method9.main(["closedloop", *common, "--out-dir", str(out9), "--device", "cpu", "--selectors", str(sel)]) == 0
    out = tmp_path / "score.json"
    assert method9.main(["score", *common, "--out-dir", str(out9), "--run-dir", str(run_dir), "--n-boot", "20",
                         "--out", str(out)]) == 0
    s = json.loads(out.read_text())
    assert set(s["selectors"]) == {"S1", "S2"}
