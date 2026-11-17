"""Method 5, phase 2, stage A: the timing part's cases and chunks, and the gate A reader (no GPU)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from scripts import latency_floor as lf  # noqa: E402
from scripts import method5 as m5  # noqa: E402

BUDGETS = {"pidnet_s/0.25": 0.18, "pidnet_s/0.375": 0.26, "pidnet_s/0.5": 0.75, "bisenetv2/0.25": 0.55,
           "bisenetv2/0.375": 0.68, "bisenetv2/0.5": 0.28}


def _budgets(**over):
    out = {}
    for g, b in BUDGETS.items():
        model, scale = g.split("/")
        out[g] = {"model": model, "scale": float(scale), "budget_ms": over.get(g, b),
                  "in_stage_b": float(scale) in m5.STAGE_B[model]}
    return out


def test_warp_cases_group_ff_and_w_by_model_and_scale():
    groups = lf.warp_specs([0.25, 0.375, 0.5])
    assert [g[0]["group"] for g in groups] == [f"{m}/{c:g}" for m in lf.MODELS for c in (0.25, 0.375, 0.5)]
    for g in groups:
        ff, ws = g[0], g[1:]
        assert ff["kind"] == "fullframe" and len(ws) == 26 and all(w["kind"] == "warp" for w in ws)
        assert {(w["model"], w["scale"], w["group"]) for w in ws} == {(ff["model"], ff["scale"], ff["group"])}
        variants = {m5.variant_of(w) for w in ws}
        assert len(variants) == 26 and len({w["name"] for w in g}) == 27
        assert {v for v in variants if v.endswith("/n10")} == {"r2u2/B/n10", "r2u2/U/n10"}
        assert {(w["alpha"], w["sigma"]) for w in ws} == {(0.25, 4.0)}
    assert [len(g) for g in lf.warp_specs([0.25], quick=True)] == [3, 3]


@pytest.mark.parametrize("fits_per_chunk", [1, 5, 10, 100])
def test_time_groups_never_mix_groups_and_repeat_the_full_frame(fits_per_chunk):
    """Chunks hold one group each, start with its full frame, and cover every W case once; a
    group that does not fit is split with the full frame repeated."""
    groups = [[{"name": f"{g}/ff", "group": g}] + [{"name": f"{g}/w{i}", "group": g} for i in range(7)]
              for g in ("a", "b")]
    captured, chunks = [], []

    def capture(c):
        captured.append(c["name"])
        return lambda: None

    def benchmark(calls):
        chunks.append(list(calls))
        captured.clear()
        return {n: {"cuda_event": {"median_ms": 1.0}} for n in calls}

    recs = lf.time_groups(groups, "graph", {}, None, capture=capture, benchmark=benchmark,
                          memory_full=lambda: len(captured) > fits_per_chunk, release=lambda: None)
    for ch in chunks:
        g = ch[0].split("/")[0]
        assert ch[0] == f"{g}/ff" and all(n.startswith(g + "/") for n in ch) and len(ch) >= 2
    ws = [n for ch in chunks for n in ch[1:]]
    assert sorted(ws) == sorted(f"{g}/w{i}" for g in "ab" for i in range(7))
    assert len(chunks) == 2 * -(-7 // max(1, min(fits_per_chunk, 7)))
    assert [r["chunk"] for r in recs] == [i for i, ch in enumerate(chunks) for _ in ch]
    assert all(r["level"] == "graph" and "fn" not in r for r in recs)


def _session(name, deltas, ff_ms=4.0, drift=0.0, gpu="Tesla T4", quick=False, missing=()):
    """One session: per (group, variant) a W median ``ff + drift + delta``, the full frame ``ff + drift``,
    each group in its own chunk."""
    recs, chunk = [], 0
    for group, per_variant in deltas.items():
        model, scale = group.split("/")
        recs.append({"name": f"warp/fullframe/{group}", "kind": "fullframe", "model": model, "scale": float(scale),
                     "level": "graph", "chunk": chunk,
                     "cuda_event": {"median_ms": ff_ms + drift, "p95_ms": ff_ms + drift + 0.1}})
        for variant, d in per_variant.items():
            if (group, variant) in missing:
                continue
            ru, sal, rounds = variant.split("/")
            recs.append({"name": f"warp/{group}/{variant}", "kind": "warp", "model": model, "scale": float(scale),
                         "resample": ru[:2], "unwarp": ru[2:], "saliency": sal, "rounds": int(rounds[1:]),
                         "level": "graph", "chunk": chunk,
                         "cuda_event": {"median_ms": ff_ms + drift + d, "p95_ms": ff_ms + drift + d + 0.1}})
        chunk += 1
    return {"session": name, "quick": quick, "environment": {"gpu": gpu}, "records": {"warp": recs}}


def _all(delta):
    return {g: {"r2u2/B/n50": delta, "r3u1/P/n50": delta + 0.5} for g in BUDGETS}


def test_gate_a_pairs_per_session():
    """Sessions that run at different speeds: unpaired medians would cross PIDNet-S's 0.18 ms at
    scale 0.25, but every paired Δs is under it, so the variant fits."""
    drifts = (0.0, 0.3, -0.2)
    sessions = [_session(f"m5w-s{i + 1}", _all(0.15 + 0.01 * i), drift=d) for i, d in enumerate(drifts)]
    out = m5.gate_a(sessions, _budgets())
    v = out["groups"]["pidnet_s/0.25"]["variants"]["r2u2/B/n50"]
    assert v["status"] == "fits" and v["delta_ms"] == pytest.approx(0.16)
    assert v["delta_range_ms"] == pytest.approx([0.15, 0.17])
    assert max(p["w_ms"] for p in v["per_session"]) - min(p["ff_ms"] for p in v["per_session"]) > 0.18
    assert out["groups"]["pidnet_s/0.25"]["variants"]["r3u1/P/n50"]["status"] == "over"
    assert out["groups"]["pidnet_s/0.25"]["kept"] == ["r2u2/B/n50"]
    assert out["groups"]["pidnet_s/0.25"]["cheapest"] == "r2u2/B/n50"
    assert out["verdict"] == "continue" and json.dumps(out)


def test_gate_a_straddle_and_over_and_stop():
    # every variant over everywhere -> stop
    over = [_session(f"s{i}", _all(1.0 + 0.01 * i)) for i in range(3)]
    out = m5.gate_a(over, _budgets())
    assert out["verdict"] == "stop" and out["goes_on"] == []
    assert all(g["status"] == "over" for g in out["groups"].values())
    # PIDNet-S at 0.25 straddles its budget (0.10, 0.17, 0.30 around 0.18), the rest over -> continue
    sess = []
    for i, d in enumerate((0.10, 0.30, 0.17)):
        deltas = _all(1.0)
        deltas["pidnet_s/0.25"] = {"r2u2/B/n50": d, "r3u1/P/n50": 0.9}
        sess.append(_session(f"s{i}", deltas))
    out = m5.gate_a(sess, _budgets())
    g = out["groups"]["pidnet_s/0.25"]
    assert g["status"] == "straddles" and g["kept"] == ["r2u2/B/n50"]
    assert out["verdict"] == "continue" and [x["group"] for x in out["goes_on"]] == ["pidnet_s/0.25"]
    assert "not decided by the timing" in out["goes_on"][0]["note"]
    # a group outside stage B (BiSeNetV2 at 0.5) that fits does not keep the method alive
    deltas = _all(1.0)
    deltas["bisenetv2/0.5"] = {"r2u2/B/n50": 0.0, "r3u1/P/n50": 0.0}
    out = m5.gate_a([_session(f"s{i}", deltas) for i in range(3)], _budgets())
    assert out["groups"]["bisenetv2/0.5"]["status"] == "fits" and out["verdict"] == "stop"


def test_gate_a_is_not_read_without_three_full_t4_sessions():
    good = [_session(f"s{i}", _all(0.0)) for i in range(3)]
    assert m5.gate_a(good, _budgets())["verdict"] == "continue"
    for sessions in (good[:2], good[:2] + [_session("s3", _all(0.0), gpu="NVIDIA GeForce RTX 2060")],
                     good[:2] + [_session("s3", _all(0.0), quick=True)],
                     good[:2] + [_session("s3", _all(0.0), missing={("pidnet_s/0.25", "r2u2/B/n50")})]):
        out = m5.gate_a(sessions, _budgets())
        assert out["verdict"] is None and out["state"].startswith("not read")
    bad = _session("s1", _all(0.0))
    bad["records"]["warp"] = [r for r in bad["records"]["warp"] if r["kind"] == "warp"]
    with pytest.raises(ValueError):
        m5.gate_a([bad], _budgets())


def test_budget_reproduces_the_spec_table():
    """The phase 1 budgets of the spec (registered grid, measured end-to-end curve):
    0.18 / 0.26 / 0.75 ms for PIDNet-S at 0.25 / 0.375 / 0.5, 0.55 / 0.68 ms for BiSeNetV2."""
    b = m5.stage_a_budgets(m5.REPO_ROOT / "results" / "method5", m5.DEFAULT_E2E)
    want = {"pidnet_s/0.25": 0.18, "pidnet_s/0.375": 0.26, "pidnet_s/0.5": 0.75, "bisenetv2/0.25": 0.55,
            "bisenetv2/0.375": 0.68}
    for g, ms in want.items():
        assert b[g]["budget_ms"] == pytest.approx(ms, abs=0.01) and b[g]["in_stage_b"]
    assert not b["bisenetv2/0.5"]["in_stage_b"]
    assert b["pidnet_s/0.25"]["best"] == "5o/c0.25/E/a0.25/s1/now"


def test_cost_command_writes_gate_a(tmp_path):
    for i in range(3):
        (tmp_path / f"latency_floor_m5w-s{i + 1}.json").write_text(json.dumps(_session(f"m5w-s{i + 1}", _all(0.1))))
    out = tmp_path / "gate_a.json"
    assert m5.main(["cost", "--sessions", str(tmp_path / "latency_floor_m5w-s*.json"), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["verdict"] == "continue" and len(data["files"]) == 3
    assert data["groups"]["pidnet_s/0.25"]["budget_ms"] == pytest.approx(0.176, abs=0.001)
