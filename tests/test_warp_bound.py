"""Method 5: the per-clip runner of the bound on the synthetic fixture (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.harness import frame_to_tensor  # noqa: E402
from reuse.method import full_pass  # noqa: E402
from reuse.replay import _confusion_t  # noqa: E402
from warp.bound import (RESAMPLES, WarpConfig, bound_clip, label_pixels, resize_ref_name,  # noqa: E402
                        warped_pass, warped_passes)
from warp.grid import make_grid  # noqa: E402

CELL = 8  # 16 x 32 fixture frames -> a 2 x 4 saliency grid


def _baseline_confusion(dataset, index, model, scale):
    """The full-frame baseline (``full_pass``) at ``scale`` against the labels, over the labelled frames."""
    total = torch.zeros(19, 19, dtype=torch.int64)
    with torch.no_grad():
        for t in np.flatnonzero(dataset.labeled_mask(index)):
            frame = frame_to_tensor(dataset.load_frame(index, int(t)), "cpu")
            labels = full_pass(model, frame, scale, (4, 8))[0]
            label = torch.from_numpy(np.asarray(dataset.load_label(index, int(t)), dtype=np.int64))
            total += _confusion_t(label, labels)
    return total.numpy()


def test_names_and_bad_configs():
    assert WarpConfig(0.5, "E", 0.5, 4, "now").name() == "5o/c0.5/E/a0.5/s4/now"
    assert WarpConfig(0.375, "S", 0.25, 1, "prev").name() == "5o/c0.375/S/a0.25/s1/prev"
    assert WarpConfig(0.75).name() == "5o/c0.75/uniform" and WarpConfig(0.75).uniform
    for bad in (dict(alpha=0.5), dict(saliency="E"), dict(saliency="Q", alpha=0.5),
                dict(saliency="E", alpha=1.0), dict(saliency="E", alpha=0.5, timing="next")):
        with pytest.raises(ValueError):
            WarpConfig(0.5, **bad)
    with pytest.raises(ValueError):
        WarpConfig(1.5)


def test_uniform_rows_equal_the_full_frame_baseline(vps, oracle_backbone):
    """At scale 1 the uniform warp is the identity: it matches the baseline exactly with either
    resampling. Below 1, the antialiased resampling matches it exactly too. The plan's 2x
    supersampling (a 2 x 2 box filter, not the baseline's triangle) does not: on this sharp-edged
    fixture with the nearest-colour dummy its confusions differ on far more than the plan's
    hoped-for 2 % of the pixels, so it is only checked to score the same pixels."""
    cfgs = [WarpConfig(0.25), WarpConfig(0.5), WarpConfig(1.0)]
    for resample in RESAMPLES:
        out = bound_clip(vps, 0, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL,
                         resample=resample)
        for cfg in cfgs:
            ref = _baseline_confusion(vps, 0, oracle_backbone, cfg.scale)
            if cfg.scale == 1.0 or resample == "antialias":
                assert np.array_equal(out[cfg.name()], ref), (resample, cfg.name())
            else:
                assert out[cfg.name()].sum() == ref.sum()


def test_resize_refs_are_the_baseline(vps, oracle_backbone):
    cfgs = [WarpConfig(0.5), WarpConfig(0.25, "S", 0.5, 1)]
    out = bound_clip(vps, 1, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL, resize_refs=True)
    assert set(out) == {c.name() for c in cfgs} | {resize_ref_name(0.5), resize_ref_name(0.25)}
    for s in (0.5, 0.25):
        assert np.array_equal(out[resize_ref_name(s)], _baseline_confusion(vps, 1, oracle_backbone, s))


def test_every_config_is_scored_on_every_labelled_pixel(vps, oracle_backbone):
    cfgs = [WarpConfig(0.5, k, 0.75, 1, tm) for k in "EBS" for tm in ("now", "prev")] + [WarpConfig(0.5)]
    out = bound_clip(vps, 0, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL)
    totals = {int(v.sum()) for v in out.values()}
    assert set(out) == {c.name() for c in cfgs} and len(totals) == 1 and totals.pop() == label_pixels(vps, 0) > 0
    assert all(v.dtype == np.int64 and v.shape == (19, 19) for v in out.values())


def test_bound_equals_one_warped_pass_per_config(vps, oracle_backbone):
    """bound_clip's rows are warped_pass on each labelled frame with the oracle saliency's grid,
    the batched passes included."""
    from warp.oracle import saliency

    cfgs = [WarpConfig(0.5, "E", 0.75, 1, "prev"), WarpConfig(0.5, "B", 0.5, 4, "now"), WarpConfig(0.5)]
    out = bound_clip(vps, 2, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL, batch=2)
    expected = {c.name(): np.zeros((19, 19), np.int64) for c in cfgs}
    with torch.no_grad():
        for t in np.flatnonzero(vps.labeled_mask(2)):
            t = int(t)
            now = frame_to_tensor(vps.load_frame(2, t), "cpu")
            prev = frame_to_tensor(vps.load_frame(2, t - 1), "cpu")
            label = torch.from_numpy(np.asarray(vps.load_label(2, t), dtype=np.int64))
            teach = {f: full_pass(oracle_backbone, x, 1.0, (4, 8))[0] for f, x in (("now", now), ("prev", prev))}
            unif_prev = full_pass(oracle_backbone, prev, 0.5, (4, 8))[0]
            grids = [make_grid(saliency("E", teach["prev"], unif_prev, 1, CELL), 0.75, 0.5),
                     make_grid(saliency("B", teach["now"], None, 4, CELL), 0.5, 0.5),
                     make_grid(torch.zeros(2, 4), 0.0, 0.5)]
            for c, g in zip(cfgs, grids):
                expected[c.name()] += _confusion_t(label, warped_pass(oracle_backbone, now, g, 0.5)).numpy()
    for c in cfgs:
        assert np.array_equal(out[c.name()], expected[c.name()]), c.name()


def test_batched_passes_equal_single_passes(vps, oracle_backbone):
    frame = frame_to_tensor(vps.load_frame(0, 4), "cpu")
    grids = [make_grid(torch.rand(2, 4, generator=torch.Generator().manual_seed(i)), 0.75, 0.5) for i in range(3)]
    with torch.no_grad():
        many = warped_passes(oracle_backbone, frame, grids, 0.5)
        for g, labels in zip(grids, many):
            assert torch.equal(labels, warped_pass(oracle_backbone, frame, g, 0.5))


def test_a_real_size_is_required(vps, oracle_backbone):
    with pytest.raises(ValueError, match="1024"):
        bound_clip(vps, 0, oracle_backbone, [WarpConfig(0.5)], "cpu")


def test_a_missing_previous_frame_is_refused(vps, oracle_backbone):
    """On a split with only the labelled frames, the frame before a labelled one is 5 frames old."""

    class OnlyLabelled:
        def __init__(self, ds):
            self.ds = ds

        def clip_id(self, i):
            return self.ds.clip_id(i)

        def labeled_mask(self, i):
            return np.array([False, True])

        def frame_indices(self, i):
            return np.asarray(self.ds.frame_indices(i))[[4, 9]]

        def load_frame(self, i, t):
            return self.ds.load_frame(i, [4, 9][t])

        def load_label(self, i, t):
            return self.ds.load_label(i, [4, 9][t])

    with pytest.raises(ValueError, match="not on disk"):
        bound_clip(OnlyLabelled(vps), 0, oracle_backbone, [WarpConfig(0.5)], "cpu", allow_any_frame_size=True,
                   cell=CELL)


def test_frame_zero_uses_itself_as_the_previous_frame(vps, oracle_backbone):
    """A labelled frame at t = 0 has no previous frame: prev is frame 0 itself (spec, plan)."""

    class FirstLabelled:
        def __init__(self, ds):
            self.ds = ds

        def clip_id(self, i):
            return self.ds.clip_id(i)

        def labeled_mask(self, i):
            return np.array([True, False])

        def frame_indices(self, i):
            return np.array([4, 5])

        def load_frame(self, i, t):
            return self.ds.load_frame(i, 4 + t)

        def load_label(self, i, t):
            return self.ds.load_label(i, 4)

    cfgs = [WarpConfig(0.5, "S", 0.75, 1, tm) for tm in ("now", "prev")]
    out = bound_clip(FirstLabelled(vps), 0, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL)
    assert np.array_equal(out[cfgs[0].name()], out[cfgs[1].name()])
