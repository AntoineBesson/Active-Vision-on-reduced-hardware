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
    wide = _finalist(rule_a, ctx="pidnet_s", gl="pidnet_s", s=0.25, q=384, k=4)
    wide["method"].update(glimpse_scale=0.5, fusion="conf")
    path.write_text(json.dumps({"finalists": [wide,
                                              _finalist(rule_b, ctx="bisenetv2", gl="pidnet_s", s=0.5, q=256, k=8)]}))
    cfgs = final_configs(path)
    assert len(cfgs) == 2
    a, b = cfgs
    assert (a["context_model"], a["glimpse_model"]) == ("pidnet_s", "pidnet_s")
    assert a["config"] == FoveationConfig(context_scale=0.25, glimpse_size=384, glimpse_margin=32, k=4,
                                          selection="greedy", suppress=1, score="margin", pool="mean",
                                          class_weights=tuple([2.0] + [1.0] * 18), fusion="conf",
                                          glimpse_scale=0.5)
    assert "/z0.5/conf/" in a["name"]
    # the prior rule needs no scoring: timed as entropy top-k, an upper bound, and labelled so
    assert b["config"].selection == "topk" and b["timed_as"] == "uncertainty/entropy/mean/topk"
    assert (b["config"].fusion, b["config"].glimpse_scale) == ("replace", 1.0)  # older finalists files
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


def test_batchk_covers_every_cell_count():
    from scripts.latency_floor import batchk_specs

    specs = batchk_specs(FULL)
    for size, cells in ((256, 66), (384, 28), (512, 15)):
        ks = sorted(sp["k"] for sp in specs if sp["size"] == size and sp["model"] == "pidnet_s")
        assert ks == list(range(1, cells + 1)), size
    assert len({sp["name"] for sp in specs}) == len(specs)


def test_reuse_parts_are_listed():
    from scripts.latency_floor import reuse_specs

    names = {sp["name"] for sp in reuse_specs(FULL)}
    assert {"reuse/gray", "reuse/homography/orb", "reuse/homography/iclk", "reuse/motion/384", "reuse/warp",
            "reuse/fusion", "reuse/post/stride8/1", "reuse/post/stride1/0.25",
            "reuse/glimpse_post/stride1/256/k66", "reuse/glimpse_post/stride8/512/k15"} <= names
    assert all(sp["part"] == "reuse" for sp in reuse_specs(QUICK))


def test_m5final_times_each_models_finalists_with_the_full_frame_curve(tmp_path):
    import json

    from scripts.latency_floor import M5_FF_SCALES, m5final_specs

    cfg = {"scale": 0.25, "saliency": "P", "alpha": 0.25, "sigma": 4.0, "resample": "r2", "unwarp": "u1", "rounds": 50,
           "taps": None}
    for model, n in (("pidnet_s", 2), ("bisenetv2", 1)):
        fins = [{"model": model, "name": f"5p/c0.25/P/a0.25/s4/r2u1/{i}", "variant": "r2u1",
                 "config": dict(cfg, scale=0.25 + 0.125 * i)} for i in range(n)]
        (tmp_path / f"finalists_{model}_scored.json").write_text(json.dumps(
            {"gate_b": "continue", "prior": "scored", "finalists": fins}))
    groups = m5final_specs(str(tmp_path / "finalists_{model}_scored.json"))
    assert [g[0]["model"] for g in groups] == ["pidnet_s", "bisenetv2"]
    for g, n in zip(groups, (2, 1)):
        assert g[0]["kind"] == "fullframe" and g[0]["scale"] == 1.0  # the reference of a group's chunks
        assert [c["scale"] for c in g if c["kind"] == "fullframe"] == list(M5_FF_SCALES)
        warps = [c for c in g if c["kind"] == "warp"]
        assert len(warps) == n and all(c["prior"] == "scored" and c["config"]["resample"] == "r2" for c in warps)
        assert [c["scale"] for c in warps] == [0.25 + 0.125 * i for i in range(n)]
    quick = m5final_specs(str(tmp_path / "finalists_{model}_scored.json"), quick=True)
    assert all(len([c for c in g if c["kind"] == "warp"]) == 1 and len(g) == 3 for g in quick)
    (tmp_path / "finalists_pidnet_s_scored.json").write_text(json.dumps({"gate_b": "stop", "finalists": []}))
    with pytest.raises(SystemExit, match="gate B"):
        m5final_specs(str(tmp_path / "finalists_{model}_scored.json"))
