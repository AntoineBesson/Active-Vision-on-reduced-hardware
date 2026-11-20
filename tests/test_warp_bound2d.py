"""Method 6: the per-clip runner of the local-warp bound on the synthetic fixture (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.harness import frame_to_tensor  # noqa: E402
from reuse.method import full_pass  # noqa: E402
from reuse.replay import _confusion_t  # noqa: E402
from warp.bound import WarpConfig, bound_clip, label_pixels, resize_ref_name  # noqa: E402
from warp.bound2d import DIST_KEYS, LocalConfig, bound_clip2d, local_passes  # noqa: E402
from warp.grid2d import make_map2d  # noqa: E402
from warp.oracle import saliency  # noqa: E402

CELL = 8  # 16 x 32 fixture frames -> a 2 x 4 saliency grid


def test_names_and_bad_configs():
    assert LocalConfig(0.25, "E", 0.25, 1, 4).name() == "6o/c0.25/E/a0.25/s1/k4"
    assert LocalConfig(0.375, "B", 0.1, 4, 12).name() == "6o/c0.375/B/a0.1/s4/k12"
    assert LocalConfig(0.5).name() == "6o/c0.5/uniform"
    sep = LocalConfig(0.25, "E", 0.25, 1, form="sep")
    assert sep.name() == "6o/c0.25/sep/E/a0.25/s1" and sep.method5_name() == "5o/c0.25/E/a0.25/s1/now"
    for bad in (dict(alpha=0.5), dict(saliency="E"), dict(saliency="Q", alpha=0.5), dict(saliency="E", alpha=0.5,
                form="x"), dict(form="sep"), dict(saliency="E", alpha=0.5, kernel=0)):
        with pytest.raises(ValueError):
            LocalConfig(0.5, **bad)
    with pytest.raises(ValueError):
        LocalConfig(0.5, "E", 0.5).method5_name()


def test_uniform_rows_are_the_baseline_and_separable_rows_are_method5(vps, oracle_backbone):
    cfgs = [LocalConfig(0.25), LocalConfig(0.5), LocalConfig(0.5, "E", 0.5, 1, form="sep"),
            LocalConfig(0.5, "B", 0.25, 4, form="sep")]
    out, dist = bound_clip2d(vps, 0, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL)
    m5 = bound_clip(vps, 0, oracle_backbone, [WarpConfig(0.25), WarpConfig(0.5), WarpConfig(0.5, "E", 0.5, 1, "now"),
                                              WarpConfig(0.5, "B", 0.25, 4, "now")],
                    "cpu", allow_any_frame_size=True, cell=CELL, resize_refs=True)
    for s in (0.25, 0.5):
        assert np.array_equal(out[f"6o/c{s:g}/uniform"], m5[f"5o/c{s:g}/uniform"])
        assert np.array_equal(out[f"6o/c{s:g}/uniform"], out[resize_ref_name(s)])
        assert np.array_equal(out[resize_ref_name(s)], m5[resize_ref_name(s)])
    for c in cfgs[2:]:
        assert np.array_equal(out[c.name()], m5[c.method5_name()])
    assert dist == {}


def test_every_config_is_scored_on_every_labelled_pixel(vps, oracle_backbone):
    cfgs = [LocalConfig(0.5, k, a, 1, kern) for k in "EBS" for a in (0.25, 0.5) for kern in (1, 3)]
    cfgs += [LocalConfig(0.5), LocalConfig(0.5, "E", 0.25, 1, form="sep")]
    out, dist = bound_clip2d(vps, 1, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL, batch=3)
    totals = {int(v.sum()) for v in out.values()}
    assert set(out) == {c.name() for c in cfgs} | {resize_ref_name(0.5)}
    assert len(totals) == 1 and totals.pop() == label_pixels(vps, 1) > 0
    frames = int(np.asarray(vps.labeled_mask(1), dtype=bool).sum())
    assert set(dist) == {c.name() for c in cfgs if c.form == "2d" and not c.uniform}
    for d in dist.values():
        assert set(d) == set(DIST_KEYS) and 0 <= d["t"] <= frames and d["aniso_mean"] >= frames - 1e-6


def test_bound_equals_one_local_pass_per_config(vps, oracle_backbone):
    cfgs = [LocalConfig(0.5, "E", 0.5, 1, 1.5), LocalConfig(0.5, "S", 0.25, 0, 3)]
    out, _ = bound_clip2d(vps, 2, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=CELL, batch=2,
                          resize_refs=False)
    expected = {c.name(): np.zeros((19, 19), np.int64) for c in cfgs}
    with torch.no_grad():
        for t in np.flatnonzero(vps.labeled_mask(2)):
            now = frame_to_tensor(vps.load_frame(2, int(t)), "cpu")
            label = torch.from_numpy(np.asarray(vps.load_label(2, int(t)), dtype=np.int64))
            teach = full_pass(oracle_backbone, now, 1.0, (4, 8))[0]
            unif = full_pass(oracle_backbone, now, 0.5, (4, 8))[0]
            maps = [make_map2d(saliency("E", teach, unif, 1, CELL), 0.5, 0.5, 1.5),
                    make_map2d(saliency("S", teach, None, 0, CELL), 0.25, 0.5, 3)]
            for c, m in zip(cfgs, maps):
                expected[c.name()] += _confusion_t(label, local_passes(oracle_backbone, now, [m], 0.5)[0]).numpy()
    for c in cfgs:
        assert np.array_equal(out[c.name()], expected[c.name()]), c.name()


def test_a_real_size_is_required(vps, oracle_backbone):
    with pytest.raises(ValueError, match="latency tables"):
        bound_clip2d(vps, 0, oracle_backbone, [LocalConfig(0.5)], "cpu", cell=CELL)
