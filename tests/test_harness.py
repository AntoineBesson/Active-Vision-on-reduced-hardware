"""Generic evaluation loop + full-frame method (CPU, synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import generate_fixture as gf  # noqa: E402
from data.base import InMemoryVideoSeg  # noqa: E402
from data.cityscapes_vps import CityscapesVPS, CityscapesVPSConfig  # noqa: E402
from eval.cost import CostLedger  # noqa: E402
from eval.harness import FrameMethod, FullFrameMethod, evaluate, measure_latency  # noqa: E402
from models.dummy import DummyBackbone  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cityscapes_vps.yaml"


class Oracle:
    """Predicts the ground truth it was given (or a constant) -- to test the loop."""

    name = "oracle"
    requires_all_frames = False

    def __init__(self, dataset, wrong=False):
        self.dataset, self.wrong, self.seen = dataset, wrong, []

    def reset(self, clip):
        self.clip = clip

    def predict(self, frame, t, ledger):
        self.seen.append((self.clip["index"], t))
        ledger.add("oracle", flops=1, pixels=frame.shape[-1] * frame.shape[-2])
        label = self.dataset.load_label(self.clip["index"], t)
        if label is None or self.wrong:
            return torch.zeros(frame.shape[-2:], dtype=torch.int64)
        return torch.from_numpy(label.astype(np.int64))

    def describe(self):
        return {"name": self.name}

    def latency_fns(self, frame_size, device):
        return {}


def _toy():
    rng = np.random.default_rng(0)
    clips = []
    for c in range(2):
        frames = rng.integers(0, 255, size=(3, 4, 5, 3), dtype=np.uint8)
        lbl = rng.integers(0, 3, size=(4, 5)).astype(np.uint8)
        clips.append({"clip_id": f"c{c}", "frames": frames, "labels": [lbl, None, lbl]})
    return InMemoryVideoSeg(clips, class_names=["a", "b", "c"])


def test_evaluate_scores_only_labelled_frames():
    ds = _toy()
    oracle = Oracle(ds)
    assert isinstance(oracle, FrameMethod)
    out = evaluate(ds, oracle)
    assert out["metrics"]["miou"] == pytest.approx(1.0) and out["metrics"]["num_frames"] == 4
    assert oracle.seen == [(0, 0), (0, 2), (1, 0), (1, 2)]  # labelled frames only
    every = Oracle(ds)
    out_all = evaluate(ds, every, frames="all")
    assert len(every.seen) == 6 and out_all["metrics"]["num_frames"] == 4  # all run, labelled scored
    assert out_all["cost"]["frames"] == 6 and out_all["cost"]["pixel_fraction"] == pytest.approx(1.0)
    assert evaluate(ds, Oracle(ds, wrong=True))["metrics"]["miou"] < 1.0
    sub = evaluate(ds, Oracle(ds), clip_indices=[1])
    assert sub["num_clips"] == 1 and sub["metrics"]["num_frames"] == 2


def test_temporal_methods_must_see_all_frames():
    ds = _toy()
    temporal = Oracle(ds)
    temporal.requires_all_frames = True
    with pytest.raises(ValueError, match="frames='all'"):
        evaluate(ds, temporal)
    with pytest.raises(ValueError):
        evaluate(ds, Oracle(ds), frames="some")


@pytest.fixture(scope="module")
def vps():
    return CityscapesVPS(CityscapesVPSConfig.from_yaml(CONFIG, root=str(gf.OFFICIAL_ROOT), split="val"))


def test_full_frame_pipeline_is_aligned_end_to_end(vps):
    """Frames are painted with the class colours, so the nearest-colour dummy
    at stride 1 is perfect at full resolution: any resize/upsampling
    misalignment in the pipeline would cost IoU."""
    palette = vps.palette / 255.0
    backbone = DummyBackbone(num_classes=19, output_stride=1, class_colors=palette).eval()
    full = evaluate(vps, FullFrameMethod(backbone, scale=1.0))
    assert full["metrics"]["miou"] == pytest.approx(1.0)
    assert full["metrics"]["num_frames"] == 6 and full["frame_sizes"] == [(gf.OH, gf.OW)]
    assert full["frame_size_counts"] == {f"{gf.OH}x{gf.OW}": 6}
    assert full["cost"]["pixel_fraction"] == pytest.approx(1.0)
    flops = full["cost"]["components"]["backbone"]["flops_per_frame"]
    assert flops == 2 * 19 * 3 * gf.OH * gf.OW  # one 1x1 conv per pixel

    half = evaluate(vps, FullFrameMethod(backbone, scale=0.5))
    assert half["cost"]["pixel_fraction"] == pytest.approx(0.25)
    assert half["cost"]["components"]["backbone"]["flops_per_frame"] == pytest.approx(flops / 4)
    assert 0.3 < half["metrics"]["miou"] < 1.0  # fewer pixels, coarser boundaries


def test_full_frame_method_sizes_and_validation():
    m = FullFrameMethod(DummyBackbone(num_classes=2), scale=0.375)
    assert m.resident_bytes() == (3 * 2 + 2) * 4  # 1x1 conv weights + bias, fp32
    assert FullFrameMethod(DummyBackbone(num_classes=2), precision="fp16").resident_bytes() == (3 * 2 + 2) * 2
    assert m.input_size(1024, 2048) == (384, 768)
    assert m.input_size(3, 3) == (1, 1)
    with pytest.raises(ValueError):
        FullFrameMethod(DummyBackbone(num_classes=2), scale=0)
    with pytest.raises(ValueError):
        FullFrameMethod(DummyBackbone(num_classes=2), precision="int8")
    ledger = CostLedger()
    ledger.begin_frame(16 * 32)
    pred = m.predict(torch.rand(1, 3, 16, 32), 0, ledger)
    assert pred.shape == (16, 32) and pred.dtype == torch.int64


def test_measure_latency_skips_without_cuda_and_runs_on_cpu_when_allowed():
    method = FullFrameMethod(DummyBackbone(num_classes=3), scale=0.5)
    if not torch.cuda.is_available():
        latency, memory = measure_latency(method, (32, 64), "cpu")
        assert latency["status"] == "skipped" and "GPU" in latency["reason"]
        assert memory["status"] == "skipped"
    latency, memory = measure_latency(method, (32, 64), "cpu", warmup=1, iters=3, allow_cpu=True)
    assert latency["status"] == "ok" and latency["batch_size"] == 1
    assert latency["backbone"]["device"] == "cpu" and latency["end_to_end"]["iters"] == 3
    assert memory["status"] == "skipped"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only: needs CUDA")
def test_measure_latency_fp16_on_gpu():  # pragma: no cover - runs on GPU machines only
    method = FullFrameMethod(DummyBackbone(num_classes=3).cuda(), scale=1.0, precision="fp16")
    latency, memory = measure_latency(method, (64, 128), "cuda", warmup=2, iters=5)
    assert latency["precision"] == "fp16" and latency["backbone"]["synchronized"] is True
    assert memory["status"] == "ok" and memory["model_weights_mib"] > 0
    assert memory["estimated_footprint_mib"] == memory["model_weights_mib"] + memory["peak_increment_mib"]
