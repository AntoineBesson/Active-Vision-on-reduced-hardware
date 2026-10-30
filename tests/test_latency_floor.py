"""scripts/latency_floor.py: every end-to-end point has its floor in the measured grids."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from foveation.pipeline import FoveationConfig  # noqa: E402
from scripts.latency_floor import FRAME, FULL, QUICK, final_configs, fullframe_points  # noqa: E402


def _finalist(rule, **geom):
    m = {"name": "foveated", "pair": f"{geom['ctx']}+{geom['gl']}", "context_model": geom["ctx"],
         "glimpse_model": geom["gl"], "context_scale": geom["s"], "glimpse_size": geom["q"], "margin": 32,
         "k": geom["k"], "rule": rule}
    return {"method": m, "latency_ms": 10.0, "miou": 0.7, "reasons": ["pareto"]}


def test_final_configs_follow_the_finalists_file(tmp_path):
    import json

    rule_a = {"kind": "uncertainty", "score": "margin", "pool": "mean", "class_weights": [2.0] + [1.0] * 18,
              "selection": "greedy", "suppress": 1, "seed": 0}
    rule_b = {"kind": "prior", "score": "entropy", "pool": "mean", "class_weights": None, "selection": "topk",
              "suppress": 1, "seed": 0}
    path = tmp_path / "finalists.json"
    path.write_text(json.dumps({"finalists": [_finalist(rule_a, ctx="pidnet_s", gl="pidnet_s", s=0.25, q=384, k=4),
                                              _finalist(rule_b, ctx="bisenetv2", gl="pidnet_s", s=0.5, q=256, k=8)]}))
    cfgs = final_configs(path)
    assert len(cfgs) == 2
    a, b = cfgs
    assert (a["context_model"], a["glimpse_model"]) == ("pidnet_s", "pidnet_s")
    assert a["config"] == FoveationConfig(context_scale=0.25, glimpse_size=384, glimpse_margin=32, k=4,
                                          selection="greedy", suppress=1, score="margin", pool="mean",
                                          class_weights=tuple([2.0] + [1.0] * 18))
    # the prior rule needs no scoring: timed as entropy top-k, an upper bound, and labelled so
    assert b["config"].selection == "topk" and b["timed_as"] == "uncertainty/entropy/mean/topk"
    # the full-frame reference curve: both models at all five scales, whatever the grid
    assert sorted(fullframe_points()) == sorted((m, s) for m in ("pidnet_s", "bisenetv2")
                                                for s in (0.25, 0.375, 0.5, 0.75, 1.0))


@pytest.mark.parametrize("s", [FULL, QUICK], ids=["full", "quick"])
def test_e2e_points_are_covered_by_the_parts(s):
    for cs, q, k in s["e2e"]:
        assert cs in s["full_scales"]  # context backbone time (area part)
        assert q in s["batch_sizes"] and k in s["ks"] and k * q * q <= s["max_batch_pixels"]  # glimpse batch
        assert cs in s["ctx_scales"] and q in s["glimpses"] and k in s["overhead_ks"]  # pipeline overhead
        assert (FRAME[0] * cs) % 32 == 0 and (FRAME[1] * cs) % 32 == 0 and q % 32 == 0  # no padding in either model
