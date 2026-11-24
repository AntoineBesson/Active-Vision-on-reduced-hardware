"""scripts/bigmodel_curve.py report (pure python) and the SegFormer wrapper (torch + transformers)."""

from __future__ import annotations

import json

import pytest

from scripts.bigmodel_curve import SCALES, input_shape, load_accuracy, load_latency, miou_at, report


def _lat(points):
    """{(model, scale): ms} -> the load_latency mapping, for both parts."""
    return {(part, m, s): {"median_ms": ms, "p95_ms": ms * 1.1, "sessions": ["s1"], "session_medians_ms": [ms]}
            for part in ("area", "e2e") for (m, s), ms in points.items()}


def _acc(points):
    return {(m, s): {"miou": v, "file": "x", "split": "val", "clips": 50} for (m, s), v in points.items()}


SMALL_LAT = {("pidnet_s", 0.5): 5.0, ("pidnet_s", 0.75): 10.0, ("pidnet_s", 1.0): 20.0,
             ("bisenetv2", 0.5): 8.0, ("bisenetv2", 0.75): 18.0, ("bisenetv2", 1.0): 31.0}
SMALL_ACC = {("pidnet_s", 0.5): 64.0, ("pidnet_s", 0.75): 70.0, ("pidnet_s", 1.0): 71.0,
             ("bisenetv2", 0.5): 67.0, ("bisenetv2", 0.75): 70.0, ("bisenetv2", 1.0): 70.2}


def test_scales_give_whole_inputs_for_every_stride():
    for s in SCALES:
        h, w = input_shape(s)
        assert h % 32 == 0 and w % 32 == 0  # no padding in any model (PIDNet 8, BiSeNetV2 32, SegFormer 32)


def test_miou_at_interpolates_and_never_extrapolates():
    pts = [(5.0, 64.0, 0.5), (10.0, 70.0, 0.75), (20.0, 71.0, 1.0)]
    assert miou_at(pts, 7.5)["miou"] == pytest.approx(67.0)
    assert miou_at(pts, 7.5)["between"] == [0.5, 0.75]
    assert miou_at(pts, 4.0) is None  # faster than the curve's fastest point
    assert miou_at(pts, 40.0)["miou"] == 71.0  # slower than its slowest point: that model at full frame


def test_miou_at_reads_the_upper_envelope():
    # scale 1 is slower but less accurate than 0.75: ignored, the curve stays at 70
    pts = [(5.0, 64.0, 0.5), (10.0, 70.0, 0.75), (20.0, 69.0, 1.0)]
    assert miou_at(pts, 15.0)["miou"] == 70.0


def test_screen_go_needs_both_closeness_and_steepness():
    lat = _lat({**SMALL_LAT, ("segformer_b2", 0.25): 9.0, ("segformer_b2", 0.375): 16.0, ("segformer_b2", 0.5): 30.0})
    # at 9 ms the best small model is PIDNet-S, 64 + 4/5 * 6 = 68.8
    acc = _acc({**SMALL_ACC, ("segformer_b2", 0.25): 67.5, ("segformer_b2", 0.375): 72.0, ("segformer_b2", 0.5): 74.0})
    rep = report(lat, acc)
    rows = {r["scale"]: r for r in rep["parts"]["e2e"]["large_points"]}
    assert rows[0.25]["best_small"]["model"] == "pidnet_s"
    assert rows[0.25]["delta_vs_best_small"] == pytest.approx(67.5 - 68.8)
    assert rows[0.25]["gain_to_next_scale"] == pytest.approx(4.5)
    assert rows[0.25]["screen_go"]
    assert rep["parts"]["e2e"]["screen"] == "GO"
    # flat large-model curve: close, but nothing to win by a warp
    acc_flat = _acc({**SMALL_ACC, ("segformer_b2", 0.25): 67.5, ("segformer_b2", 0.375): 67.9,
                     ("segformer_b2", 0.5): 68.0})
    assert report(lat, acc_flat)["parts"]["e2e"]["screen"] == "STOP"
    # steep but far below the small models
    acc_far = _acc({**SMALL_ACC, ("segformer_b2", 0.25): 60.0, ("segformer_b2", 0.375): 66.0,
                    ("segformer_b2", 0.5): 70.0})
    assert report(lat, acc_far)["parts"]["e2e"]["screen"] == "STOP"


def test_report_lists_points_without_accuracy():
    lat = _lat({**SMALL_LAT, ("segformer_b5", 0.25): 40.0})
    rep = report(lat, _acc(SMALL_ACC))
    assert rep["missing_accuracy"] == ["segformer_b5@0.25"]
    assert rep["parts"]["area"]["screen"] == "STOP"


def _run_record(tmp_path, name, model, scale, miou, fixture=False):
    rec = {"metrics": {"miou": miou}, "method": {"name": "full_frame", "scale": scale}, "backbone": {"type": model},
           "dataset": {"split": "val", "clips_evaluated": 50, "synthetic_fixture": fixture}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(rec))
    return str(p)


def test_load_accuracy_reads_run_baseline_records(tmp_path):
    a = _run_record(tmp_path, "a", "segformer_b2", 0.5, 0.7)
    acc = load_accuracy([a])
    assert acc[("segformer_b2", 0.5)]["miou"] == pytest.approx(70.0)
    with pytest.raises(SystemExit, match="two accuracy runs"):
        load_accuracy([a, _run_record(tmp_path, "b", "segformer_b2", 0.5, 0.71)])
    with pytest.raises(SystemExit, match="fixture"):
        load_accuracy([_run_record(tmp_path, "c", "segformer_b2", 0.25, 0.6, fixture=True)])


def test_load_latency_takes_the_median_over_sessions(tmp_path):
    paths = []
    for i, ms in enumerate([10.0, 12.0, 11.0]):
        rec = {"part": "e2e", "model": "segformer_b0", "scale": 0.5, "level": "graph",
               "cuda_event": {"median_ms": ms, "p95_ms": ms + 1}}
        eager = dict(rec, level="eager", cuda_event={"median_ms": 99.0, "p95_ms": 99.0})
        p = tmp_path / f"latency_s{i}.json"
        p.write_text(json.dumps({"experiment": "bigmodel_curve", "session": f"s{i}", "records": [rec, eager]}))
        paths.append(str(p))
    lat = load_latency(paths, "graph")
    assert lat[("e2e", "segformer_b0", 0.5)]["median_ms"] == 11.0
    assert lat[("e2e", "segformer_b0", 0.5)]["sessions"] == ["s0", "s1", "s2"]


# --------------------------------------------------------------------------- SegFormer wrapper

def _segformer(variant="b0"):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from models.registry import build_backbone

    return build_backbone({"type": f"segformer_{variant}", "num_classes": 19, "allow_random_init": True})


@pytest.mark.parametrize("hw", [(64, 128), (100, 130), (37, 61)])
def test_segformer_follows_the_backbone_contract(hw):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from models.interfaces import SegmentationBackbone
    from models.utils import upsample_logits

    net = _segformer()
    assert isinstance(net, SegmentationBackbone)
    assert (net.output_stride, net.align_corners) == (4, False)
    h, w = hw
    with torch.no_grad():
        logits = net(torch.rand(2, 3, h, w))
    assert logits.shape == (2, 19, -(-h // 4), -(-w // 4))
    assert upsample_logits(logits, (h, w), 4, False).shape == (2, 19, h, w)


def test_segformer_variants_have_the_published_sizes():
    pytest.importorskip("transformers")
    # parameter counts of the MiT encoders + decoder, in millions (paper, Table 1: B0 3.8, B2 27.5)
    sizes = {v: sum(p.numel() for p in _segformer(v).parameters()) / 1e6 for v in ("b0", "b2")}
    assert sizes["b0"] == pytest.approx(3.7, abs=0.3)
    assert sizes["b2"] == pytest.approx(27.4, abs=1.0)


def test_segformer_weights_round_trip_through_a_file(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    src = _segformer()
    path = tmp_path / "segformer_b0.pt"
    torch.save(src.net.state_dict(), path)
    dst = _segformer()
    report = dst.load_weights(str(path))
    assert not report["missing"] and not report["unexpected"]
    x = torch.rand(1, 3, 64, 128)
    with torch.no_grad():
        assert torch.equal(src.eval()(x), dst.eval()(x))
    assert dst.describe()["weights"]["file"] == "segformer_b0.pt"


def test_segformer_runs_in_the_full_frame_method():
    import numpy as np

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from eval.cost import CostLedger
    from eval.harness import FullFrameMethod

    method = FullFrameMethod(_segformer(), scale=0.25)
    pred = method.predict(torch.rand(1, 3, 128, 256), 0, CostLedger())
    assert tuple(pred.shape) == (128, 256)
    assert int(pred.min()) >= 0 and int(pred.max()) < 19
    assert np.isfinite(method.describe()["scale"])
