"""MotionReuseMethod (Method 3) on the synthetic fixture (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

import generate_fixture as gf  # noqa: E402
from eval.cost import CostLedger  # noqa: E402
from eval.harness import FullFrameMethod, frame_to_tensor  # noqa: E402
from reuse import method as rm  # noqa: E402
from reuse.method import MotionReuseConfig, MotionReuseMethod  # noqa: E402

GRID = dict(glimpse_size=12, margin=2)  # fixture frames are 16x32: inner 8, 2 x 4 cells


def _cfg(**kw):
    base = dict(full_scale=1.0, theta=30, rho=0.05, tau=1.0, **GRID)
    base.update(kw)
    return MotionReuseConfig(**base)


def _frames(vps, clip=0):
    return [frame_to_tensor(vps.load_frame(clip, t), "cpu") for t in range(vps.num_frames(clip))]


def _run(method, frames):
    method.reset({})
    out, steps = [], []
    for t, f in enumerate(frames):
        ledger = CostLedger()
        ledger.begin_frame(16 * 32)
        out.append(method.predict(f, t, ledger))
        steps.append(dict(method.last_step))
    return out, steps


class _Const(torch.nn.Module):
    """Very confident in one class everywhere."""

    output_stride, align_corners = 1, False

    def __init__(self, cls=5):
        super().__init__()
        self.cls = cls

    def forward(self, x):
        out = torch.zeros(x.shape[0], 19, *x.shape[-2:])
        out[:, self.cls] = 100.0
        return out


def test_frame_zero_is_the_full_pass(vps, oracle_backbone):
    frames = _frames(vps)
    method = MotionReuseMethod(oracle_backbone, _cfg(), allow_any_frame_size=True)
    pred, _ = _run(method, frames[:1])
    full = FullFrameMethod(oracle_backbone, scale=1.0)
    ledger = CostLedger()
    ledger.begin_frame(16 * 32)
    assert torch.equal(pred[0], full.predict(frames[0], 0, ledger))


def test_tau_zero_always_falls_back(vps, oracle_backbone):
    frames = _frames(vps)
    method = MotionReuseMethod(oracle_backbone, _cfg(tau=0.0, rho=-1.0), allow_any_frame_size=True)
    preds, steps = _run(method, frames)
    full = FullFrameMethod(oracle_backbone, scale=1.0)
    for t, (p, f) in enumerate(zip(preds, frames)):
        ledger = CostLedger()
        ledger.begin_frame(16 * 32)
        assert torch.equal(p, full.predict(f, t, ledger))
    assert all(s["fallback"] for s in steps[1:]) and not steps[0]["fallback"]


def test_a_static_clip_recomputes_nothing(vps, oracle_backbone):
    f0 = _frames(vps)[0]
    method = MotionReuseMethod(oracle_backbone, _cfg(), allow_any_frame_size=True)
    preds, steps = _run(method, [f0] * 5)
    assert all(s["moved_cells"] == 0 and not s["fallback"] for s in steps[1:])
    assert all(torch.equal(p, preds[0]) for p in preds)


def test_predictions_never_hold_ignore(vps, oracle_backbone, monkeypatch):
    """A warp that uncovers the left 4 px of every frame (Review Focus 2)."""
    shift = torch.tensor([[1, 0, 4], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    monkeypatch.setattr(rm, "estimate_homography", lambda prev, cur, method="orb": (shift.clone(), 99))
    frames = _frames(vps)
    method = MotionReuseMethod(oracle_backbone, _cfg(rho=0.9), allow_any_frame_size=True)  # border cells stay reused
    preds, steps = _run(method, frames)
    assert all(int(p.max()) < 19 and int(p.min()) >= 0 for p in preds)
    assert any(s["moved_cells"] < s["num_cells"] for s in steps[1:])


def test_3b_static_cells_take_the_more_confident_label(vps, oracle_backbone):
    f0 = _frames(vps)[0]
    method = MotionReuseMethod(oracle_backbone, _cfg(ctx_scale=0.25), ctx_backbone=_Const(5), allow_any_frame_size=True)
    preds, steps = _run(method, [f0] * 3)
    assert steps[1]["moved_cells"] == 0
    assert (preds[1] == 5).all() and (preds[2] == 5).all()


def test_reset_clears_the_memory(vps, oracle_backbone):
    frames = _frames(vps)
    method = MotionReuseMethod(oracle_backbone, _cfg(), allow_any_frame_size=True)
    _run(method, frames[:3])
    method.reset({})
    ledger = CostLedger()
    ledger.begin_frame(16 * 32)
    method.predict(frames[3], 0, ledger)
    assert method.last_step == {"fallback": False, "moved_cells": method.last_step["num_cells"],
                                "num_cells": method.last_step["num_cells"], "inliers": 0, "first": True}


def test_config_names_are_stable():
    assert _cfg().name() == "3a/f1/g12/t30/r0.05/u1"
    assert _cfg(ctx_scale=0.25, tau=0.4).name() == "3b-c0.25/f1/g12/t30/r0.05/u0.4"


def test_moving_cells_get_the_current_frames_labels(vps, oracle_backbone):
    """A car-coloured patch appears in one cell: some cells, not all, are recomputed, and with
    a per-pixel perfect backbone a recomputed cell equals the current frame's full pass there."""
    f0 = _frames(vps)[0]
    f1 = f0.clone()
    f1[0, :, 0:8, 0:8] = torch.tensor(vps.palette[13] / 255.0, dtype=f1.dtype)[:, None, None]
    method = MotionReuseMethod(oracle_backbone, _cfg(theta=30, rho=0.0), allow_any_frame_size=True)
    preds, steps = _run(method, [f0, f1])
    assert 0 < steps[1]["moved_cells"] < steps[1]["num_cells"]
    full = FullFrameMethod(oracle_backbone, scale=1.0)
    ledger = CostLedger()
    ledger.begin_frame(16 * 32)
    ref = full.predict(f1, 1, ledger)
    grid = method.grid
    moving = method.last_moving.tolist()
    assert moving[0]
    for c in range(grid.num_cells):
        r0, c0 = (c // grid.cols) * grid.inner, (c % grid.cols) * grid.inner
        cell = (slice(r0, r0 + grid.inner), slice(c0, c0 + grid.inner))
        if moving[c]:
            assert torch.equal(preds[1][cell], ref[cell]), c
        else:
            assert torch.equal(preds[1][cell], preds[0][cell]), c
