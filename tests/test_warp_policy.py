"""Method 5, phase 2: the computable saliencies and the per-frame warp step (CPU; one CUDA test)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.harness import frame_to_tensor  # noqa: E402
from models.dummy import DummyBackbone  # noqa: E402
from reuse.method import full_pass  # noqa: E402
from warp.bound import warped_pass  # noqa: E402
from warp.grid import make_grid, unwarp_logits  # noqa: E402
from warp.oracle import saliency, smooth  # noqa: E402
from warp.policy import (StepConfig, WarpStep, load_prior, margin_map, prior_record, raw_saliency,  # noqa: E402
                         warped_size)

CELL = 8  # 64 x 128 test frames -> an 8 x 16 saliency grid


def _frames(vps, n=3):
    """Fixture frames (16 x 32, painted with the class colours) enlarged 4 times: 64 x 128."""
    out = []
    for t in range(n):
        f = frame_to_tensor(vps.load_frame(0, t), "cpu")
        out.append(f.repeat_interleave(4, -2).repeat_interleave(4, -1).contiguous())
    return out


def _models(vps):
    """Nearest-colour dummies with each convention: BiSeNetV2's (stride 1, align_corners False) and
    PIDNet-S's (stride 8, align_corners True)."""
    bise = DummyBackbone(19, output_stride=1, class_colors=vps.palette / 255.0).eval()
    pid = DummyBackbone(19, output_stride=8, class_colors=vps.palette / 255.0).eval()
    pid.align_corners = True
    return {"bisenetv2": bise, "pidnet_s": pid}


def _labels(seed=0, size=(64, 128)):
    g = torch.Generator().manual_seed(seed)
    small = torch.randint(0, 19, (size[0] // 4, size[1] // 4), generator=g)
    return small.repeat_interleave(4, 0).repeat_interleave(4, 1)


def _saliency(seed=1, shape=(8, 16)):
    s = torch.zeros(shape)
    s[2, 3], s[6, 12], s[1, 14] = 1.0, 0.6, 0.3
    return s + 0.05 * torch.rand(shape, generator=torch.Generator().manual_seed(seed))


def test_names_and_bad_configs():
    assert StepConfig(0.25, "B", 0.15, 1).name() == "5p/c0.25/B/a0.15/s1/r2u2"
    assert StepConfig(0.375, "U", 0.25, 4, "r1", "u1", rounds=10).name() == "5p/c0.375/U/a0.25/s4/r1u1/n10"
    assert StepConfig(0.5).name() == "5p/c0.5/uniform/r2u2" and StepConfig(0.5).uniform and StepConfig(0.5).fixed
    assert StepConfig(0.5, "P", 0.1, 1).fixed and not StepConfig(0.5, "S", 0.1, 1).fixed
    assert StepConfig(0.25).tap_count() == 19 and StepConfig(0.5, taps=16).tap_count() == 16
    for bad in (dict(alpha=0.25), dict(saliency="B"), dict(saliency="E", alpha=0.25), dict(saliency="B", alpha=1.0),
                dict(saliency="B", alpha=0.25, sigma=-1), dict(resample="r4"), dict(unwarp="u3"), dict(rounds=0),
                dict(taps=10)):
        with pytest.raises(ValueError):
            StepConfig(0.5, **bad)
    with pytest.raises(ValueError):
        StepConfig(1.5)


@pytest.mark.parametrize("kind", ["B", "S"])
@pytest.mark.parametrize("cell", [8, 16])
def test_b_and_s_match_the_phase1_oracle_on_the_same_labels(kind, cell):
    labels = _labels()
    for sigma in (0, 1, 4):
        ref = saliency(kind, labels, None, sigma, cell)
        out = smooth(raw_saliency(kind, labels=labels, cell=cell), sigma)
        assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-7)
    assert raw_saliency(kind, labels=labels.to(torch.uint8), cell=cell).equal(raw_saliency(kind, labels=labels,
                                                                                            cell=cell))


def test_u_is_in_unit_range_and_peaks_where_two_classes_tie():
    logits = torch.zeros(1, 3, 4, 8)
    logits[0, 0] = 10.0  # confident everywhere ...
    logits[0, :2, 1, 5] = 5.0  # ... but a tie of classes 0 and 1 here
    logits[0, :, 3, 0] = 0.0  # and all three tied here
    m = margin_map(logits)
    assert m.shape == (4, 8) and float(m.min()) >= 0 and float(m.max()) <= 1
    assert float(m[1, 5]) == pytest.approx(1.0) and float(m[3, 0]) == pytest.approx(1.0)
    assert float(m[0, 0]) < 1e-3
    # a uniform warp at stride 1, one cell per pixel, frame = logits: U is the margin map itself
    u = raw_saliency("U", logits=logits, grid=make_grid(torch.zeros(4, 8), 0.0, 1.0), cell=1, frame_size=(4, 8))
    assert torch.allclose(u, m, atol=1e-6)
    for bad in (dict(), dict(logits=logits), dict(logits=logits, grid=make_grid(torch.zeros(4, 8), 0.0, 1.0))):
        with pytest.raises(ValueError):
            raw_saliency("U", **bad)
    with pytest.raises(ValueError):
        raw_saliency("P", labels=_labels())


@pytest.mark.parametrize("stride,align", [(8, True), (1, False)])
def test_u_reads_the_unwarped_margin_at_the_cell_centres(stride, align):
    """With one cell per pixel the cell centres are the pixel centres, so U is the margin map
    unwarped with the logits' own convention (:func:`warp.grid.unwarp_logits`)."""
    scale, frame = 0.25, (64, 128)
    size = warped_size(frame, scale)
    hl, wl = -(-size[0] // stride), -(-size[1] // stride)
    logits = torch.randn(1, 19, hl, wl, generator=torch.Generator().manual_seed(3))
    grid = make_grid(_saliency(), 0.25, scale)
    u = raw_saliency("U", logits=logits, grid=grid, cell=1, frame_size=frame, align_corners=align,
                     output_stride=stride, input_size=size)
    ref = unwarp_logits(margin_map(logits)[None, None], grid, frame, align, stride, input_size=size)[0, 0]
    assert u.shape == frame and (u - ref).abs().max() < 1e-5
    coarse = raw_saliency("U", logits=logits, grid=grid, cell=16, frame_size=frame, align_corners=align,
                          output_stride=stride, input_size=size)
    assert coarse.shape == (4, 8) and float(coarse.min()) >= 0 and float(coarse.max()) <= 1


@pytest.mark.parametrize("model_kind", ["bisenetv2", "pidnet_s"])
@pytest.mark.parametrize("scale", [0.25, 0.5])
def test_uniform_step_reproduces_the_full_pass(vps, model_kind, scale):
    """The pipeline check: at alpha 0 the step is the full frame, label for label, with the
    antialiased resampling tap by tap (r1) or as matrices (r2) and either unwarp."""
    model = _models(vps)[model_kind]
    for frame in _frames(vps):
        ref = full_pass(model, frame, scale, (4, 8))[0]
        for r in ("r1", "r2"):
            for u in ("u1", "u2"):
                step = WarpStep(model, StepConfig(scale, resample=r, unwarp=u), cell=CELL)
                with torch.no_grad():
                    labels, nxt = step(frame, step.initial_state(frame.shape[-2:]))
                assert labels.dtype == torch.int64 and torch.equal(labels, ref), (r, u)
                assert torch.equal(nxt, torch.zeros(8, 16))


@pytest.mark.parametrize("model_kind", ["bisenetv2", "pidnet_s"])
@pytest.mark.parametrize("kind", ["B", "S", "U"])
def test_step_is_phase1_warped_pass_on_the_same_grid(vps, model_kind, kind):
    """r1/u1 is phase 1's pipeline: the step's labels are :func:`warp.bound.warped_pass` with the
    grid of the smoothed state (phase 1's density loop there, 50 fixed rounds here); r2/u2 gives
    the same labels up to float rounding."""
    model = _models(vps)[model_kind]
    scale, alpha, sigma = 0.25, 0.25, 1
    state = _saliency()
    grid = make_grid(smooth(state, sigma), alpha, scale)
    for frame in _frames(vps, 2):
        with torch.no_grad():
            ref = warped_pass(model, frame, grid, scale)
            a, _ = WarpStep(model, StepConfig(scale, kind, alpha, sigma, "r1", "u1"), cell=CELL)(frame, state)
            b, _ = WarpStep(model, StepConfig(scale, kind, alpha, sigma, "r2", "u2"), cell=CELL)(frame, state)
        assert torch.equal(a, ref)
        assert (b != ref).float().mean() < 1e-3


@pytest.mark.parametrize("model_kind", ["bisenetv2", "pidnet_s"])
def test_next_state_is_this_frames_saliency(vps, model_kind):
    model = _models(vps)[model_kind]
    frame, state = _frames(vps, 1)[0], _saliency()
    for kind in ("B", "S", "U"):
        step = WarpStep(model, StepConfig(0.5, kind, 0.25, 1), cell=CELL)
        with torch.no_grad():
            x, ctx = step.warp(frame, state)
            logits = model(x)
            labels, nxt = step.finish(logits, ctx, state)
            again, nxt2 = step(frame, state)
        assert torch.equal(labels, again) and torch.equal(nxt, nxt2) and nxt.shape == (8, 16)
        if kind == "U":
            ref = raw_saliency("U", logits=logits, grid=ctx["grid"], cell=CELL, frame_size=(64, 128),
                               align_corners=model.align_corners, output_stride=model.output_stride,
                               input_size=ctx["input_size"])
        else:
            ref = raw_saliency(kind, labels=labels, cell=CELL)
        assert torch.equal(nxt, ref)


def test_p_ignores_the_previous_output(vps):
    model = _models(vps)["bisenetv2"]
    prior = _saliency(seed=4)
    step = WarpStep(model, StepConfig(0.25, "P", 0.25, 1), prior=prior, cell=CELL)
    f0, f1 = _frames(vps, 2)
    with torch.no_grad():
        l0, n0 = step(f0, step.initial_state(f0.shape[-2:]))
        l1, n1 = step(f1, torch.rand(8, 16))  # any state: P's warp is the prior's
        ref = warped_pass(model, f1, make_grid(smooth(prior, 1), 0.25, 0.25), 0.25)
    assert torch.equal(n0, prior) and torch.equal(n1, prior) and torch.equal(step.initial_state((64, 128)), prior)
    assert (l1 != ref).float().mean() < 1e-3
    cached = dict(step._fixed)
    with torch.no_grad():
        step(f1, torch.zeros(8, 16))
    assert step._fixed.keys() == cached.keys() and all(step._fixed[k] is cached[k] for k in cached)
    with pytest.raises(ValueError):
        WarpStep(model, StepConfig(0.25, "P", 0.25, 1))
    with pytest.raises(ValueError):
        WarpStep(model, StepConfig(0.25, "B", 0.25, 1), prior=prior)
    with pytest.raises(ValueError):
        WarpStep(model, StepConfig(0.25, "P", 0.25, 1), prior=prior, cell=16).initial_state((64, 128))


def test_dynamic_steps_cache_nothing(vps):
    model = _models(vps)["bisenetv2"]
    step = WarpStep(model, StepConfig(0.25, "B", 0.25, 1), cell=CELL)
    frame = _frames(vps, 1)[0]
    with torch.no_grad():
        step(frame, _saliency())
    assert step._fixed == {} and torch.equal(step.initial_state((64, 128)), torch.zeros(8, 16))


def test_prior_records_load_by_model_and_scale(tmp_path):
    a, b = torch.rand(8, 16, dtype=torch.float64), torch.rand(8, 16, dtype=torch.float64)
    recs = [prior_record("pidnet_s", 0.25, a, 4), prior_record("pidnet_s", 0.375, b, 2)]
    path = tmp_path / "prior_pidnet_s.pt"
    torch.save(recs, path)
    assert torch.allclose(load_prior(path, "pidnet_s", 0.25), (a / 4).float())
    assert torch.allclose(load_prior(path, "pidnet_s", 0.375), (b / 2).float())
    assert load_prior(recs[0], "pidnet_s", 0.25).dtype == torch.float32
    for model, scale in (("pidnet_s", 0.5), ("bisenetv2", 0.25)):
        with pytest.raises(ValueError):
            load_prior(path, model, scale)
    with pytest.raises(ValueError):
        load_prior(recs + recs[:1], "pidnet_s", 0.25)
    with pytest.raises(ValueError):
        prior_record("pidnet_s", 0.25, a, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("saliency_kind", ["B", "S", "U", "P"])
@pytest.mark.parametrize("variant", [("r1", "u1"), ("r2", "u2"), ("r3", "u2")])
def test_step_captures_in_a_cuda_graph(saliency_kind, variant):
    """Nothing in a step reads back to the host: it captures, and two replays give the eager output."""
    from eval.gpu_timing import cuda_graph

    torch.manual_seed(0)
    model = DummyBackbone(19, output_stride=8).cuda().eval().half()
    model.align_corners = True
    prior = _saliency(shape=(64, 128)).cuda() if saliency_kind == "P" else None
    cfg = StepConfig(0.25, saliency_kind, 0.25, 4, *variant)
    step = WarpStep(model, cfg, prior=prior, precision="fp16")
    frame = torch.rand(1, 3, 1024, 2048, device="cuda").half()
    state = _saliency(shape=(64, 128)).cuda()
    with torch.no_grad():
        eager = step(frame, state)
    replay = cuda_graph(lambda f: step(f, state), frame)
    for _ in range(2):
        labels, nxt = replay()
        torch.cuda.synchronize()
        assert torch.equal(labels, eager[0]) and torch.allclose(nxt, eager[1])
