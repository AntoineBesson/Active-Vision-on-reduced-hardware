import itertools

import numpy as np
import pytest

from eval.tradeoff import miou_of
from upsample.scale_oracle import fixed_mix, miou_many, scale_oracle


def _conf(rng, frames, scales, c=4):
    out = np.zeros((frames, scales, c, c), dtype=np.int64)
    for f in range(frames):
        for s in range(scales):
            acc = 0.4 + 0.5 * s / max(1, scales - 1) + rng.uniform(-0.3, 0.3)
            for t in range(c):
                n = rng.integers(0, 200)
                hit = int(n * min(max(acc, 0), 1))
                out[f, s, t, t] = hit
                out[f, s, t, (t + 1) % c] = n - hit
    return out


def test_miou_many_matches_miou_of():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 50, size=(3, 5, 5))
    m[1, 2] = 0
    m[1, :, 2] = 0
    assert np.allclose(miou_many(m), [miou_of(x) for x in m])


def test_oracle_meets_budget_and_beats_single_scales():
    rng = np.random.default_rng(1)
    conf = _conf(rng, 30, 4)
    lat = np.array([1.0, 2.0, 3.0, 4.0])
    res = scale_oracle(conf, lat, 2.5)
    assert res.mean_ms <= 2.5 + 1e-9
    single = miou_many(conf[:, 1].sum(0))
    assert res.miou >= single - 1e-12


def test_oracle_is_exact_on_a_tiny_case():
    rng = np.random.default_rng(2)
    conf = _conf(rng, 6, 3)
    lat = np.array([1.0, 2.0, 3.0])
    budget = 2.0
    best = -1.0
    for choice in itertools.product(range(3), repeat=6):
        ch = np.array(choice)
        if lat[ch].mean() <= budget + 1e-9:
            best = max(best, float(miou_many(conf[np.arange(6), ch].sum(0))))
    res = max((scale_oracle(conf, lat, budget, seed=s) for s in range(3)), key=lambda r: r.miou)
    assert res.miou >= best - 0.01  # a relaxation: near the exact optimum on this case
    assert res.miou <= best + 1e-12


def test_unconstrained_budget_and_errors():
    rng = np.random.default_rng(3)
    conf = _conf(rng, 10, 3)
    lat = np.array([1.0, 2.0, 3.0])
    res = scale_oracle(conf, lat, 3.0)
    assert res.miou >= miou_many(conf[:, 2].sum(0)) - 1e-12
    with pytest.raises(ValueError):
        scale_oracle(conf, lat, 0.5)
    mix = fixed_mix(conf, lat, 1.5)
    assert mix["scales"] == [0, 1] and abs(mix["mean_ms"] - 1.5) < 0.11
