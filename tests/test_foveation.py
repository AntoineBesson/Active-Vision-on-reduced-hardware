"""Foveation ops and the memory-free context + glimpses pipeline (CPU; CUDA graph test on GPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from foveation.ops import (  # noqa: E402
    GlimpseGrid,
    cell_scores,
    downsample,
    entropy_map,
    gather_crops,
    paste_labels,
    select_greedy,
    select_topk,
)
from foveation.pipeline import FoveatedSegmenter, FoveationConfig, StubBackbone  # noqa: E402
from models.dummy import DummyBackbone  # noqa: E402


def test_downsample_averages_instead_of_dropping():
    frame = torch.zeros(1, 1, 8, 8)
    frame[..., ::2] = 1.0  # one column in two
    small = downsample(frame, 0.5)
    assert small.shape == (1, 1, 4, 4)
    # decimation would give all 1 or all 0; the triangle filter only shifts the border columns
    assert torch.allclose(small[..., 1:3], torch.full_like(small[..., 1:3], 0.5), atol=1e-6)
    assert ((small > 0.4) & (small < 0.6)).all()
    assert downsample(frame, 1.0) is frame


def test_entropy_map():
    logits = torch.zeros(1, 4, 2, 2)
    logits[0, 0, 0, 0] = 100.0  # certain pixel
    e = entropy_map(logits)
    assert e.shape == (1, 2, 2)
    assert e[0, 0, 0] < 1e-6
    assert torch.allclose(e[0, 1, 1], torch.log(torch.tensor(4.0)))


def test_grid_geometry_and_canvas():
    g = GlimpseGrid((100, 230), size=48, margin=8)  # inner 32
    assert (g.inner, g.rows, g.cols) == (32, 4, 8)
    assert g.canvas_size == (128, 256)
    assert g.pad_frame(torch.zeros(1, 3, 100, 230)).shape == (1, 3, 128 + 16, 256 + 16)
    top, left = g.cell_origin(torch.tensor([0, 9, 31]))
    assert top.tolist() == [0, 32, 96] and left.tolist() == [0, 32, 224]
    with pytest.raises(ValueError):
        GlimpseGrid((100, 100), size=16, margin=8)


def test_cell_scores_locate_the_hot_cell():
    g = GlimpseGrid((64, 128), size=32, margin=8)  # inner 16 -> 4 x 8 cells
    m = torch.zeros(8, 16)  # score map at 1/8 resolution
    m[5, 13] = 10.0  # frame pixel ~ (40..48, 104..112) -> cell row 2, col 6
    s = cell_scores(m, g)
    assert s.shape == (32,)
    assert int(s.argmax()) == 2 * 8 + 6


def test_topk_is_deterministic_on_ties():
    s = torch.tensor([1.0, 3.0, 3.0, 0.0, 3.0])
    assert select_topk(s, 3).tolist() == [1, 2, 4]


def test_greedy_spreads_and_never_repeats():
    g = GlimpseGrid((64, 128), size=32, margin=8)  # 4 x 8 cells
    s = torch.zeros(32)
    s[[9, 10, 17, 30]] = torch.tensor([5.0, 4.9, 4.8, 1.0])  # 10 and 17 are neighbours of 9
    picks = select_greedy(s, g, k=3, suppress=1).tolist()
    assert picks[:2] == [9, 30]
    assert len(set(select_greedy(s, g, k=32, suppress=1).tolist())) == 32  # all cells, each once


def test_gather_and_paste_roundtrip():
    g = GlimpseGrid((40, 72), size=24, margin=4)  # inner 16 -> 3 x 5 cells, canvas 48 x 80
    frame = torch.arange(40 * 72, dtype=torch.float32).view(1, 1, 40, 72)
    padded = g.pad_frame(frame)
    idx = torch.tensor([0, 7, 14])
    crops = gather_crops(padded, g, idx)
    assert crops.shape == (3, 1, 24, 24)
    # inner square of cell 7 (row 1, col 2) = frame[16:32, 32:48]
    assert torch.equal(crops[1, 0, 4:20, 4:20], frame[0, 0, 16:32, 32:48])
    canvas = g.to_canvas(torch.zeros(40, 72, dtype=torch.long))
    crop_labels = torch.stack([torch.full((24, 24), v) for v in (1, 2, 3)])
    out = paste_labels(canvas, crop_labels, g, idx)[:40, :72]
    assert (out[0:16, 0:16] == 1).all() and (out[16:32, 32:48] == 2).all() and (out[32:40, 64:72] == 3).all()
    assert int((out != 0).sum()) == 16 * 16 * 2 + 8 * 8  # cell 14 is cut by the frame border


def test_pipeline_shapes_and_glimpse_override():
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=64, glimpse_margin=8, k=3)
    ctx = DummyBackbone(num_classes=5)
    seg = FoveatedSegmenter(ctx, ctx, cfg, frame_size=(128, 256)).eval()
    frame = torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        out = seg(frame)
    assert out.shape == (128, 256) and out.dtype == torch.long
    d = seg.describe()
    assert d["grid"]["cells"] == 3 * 6 and d["pixel_fraction"] == pytest.approx(0.0625 + 3 * 64 * 64 / (128 * 256))
    with pytest.raises(ValueError):
        FoveatedSegmenter(ctx, ctx, FoveationConfig(glimpse_size=64, glimpse_margin=8, k=99), frame_size=(128, 256))


def test_stub_backbone_shapes():
    stub = StubBackbone(num_classes=7, output_stride=8)
    y = stub(torch.zeros(3, 3, 100, 64))
    assert y.shape == (3, 7, 13, 8)
    assert stub(torch.zeros(3, 3, 100, 64)) is y  # preallocated, zero compute


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only: needs CUDA")
def test_pipeline_is_sync_free_and_graph_capturable():
    from eval.gpu_timing import cuda_graph

    cfg = FoveationConfig(context_scale=0.25, glimpse_size=128, glimpse_margin=16, k=4, selection="greedy")
    stub = StubBackbone(num_classes=19, output_stride=8).cuda()
    seg = FoveatedSegmenter(stub, stub, cfg, frame_size=(256, 512)).cuda().eval()
    frame = torch.rand(1, 3, 256, 512, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        seg(frame)  # first call allocates the stub's buffers (a host->device copy)
    torch.cuda.set_sync_debug_mode("error")  # any host sync raises
    try:
        with torch.no_grad():
            eager = seg(frame)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    replay = cuda_graph(seg, frame)
    assert torch.equal(replay(), eager)
