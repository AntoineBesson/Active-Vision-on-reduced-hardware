"""Method 6: the local (2-D) warp, its inverse, and its two resampling steps (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from models.utils import upsample_logits  # noqa: E402
from warp.grid import _forward64, _inverse64, make_grid, unwarp_logits, warp_frame  # noqa: E402
from warp.grid2d import (Map2D, corner_jacobians, displacement, distortion, eval_map, identity_map,  # noqa: E402
                         inverse_map, make_map2d, map_ok, resample_aa, singular_values, unwarp_logits2d,
                         unwarp_logits_at, warp_frame2d)
from warp.oracle import smooth  # noqa: E402


def _blobs(gh=16, gw=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(gh, gw, generator=g) > 0.93).double()
    return smooth(s, 1.0)


def _centres(n):
    return (torch.arange(n, dtype=torch.float64) + 0.5) / n


def test_alpha_zero_constant_or_empty_saliency_is_the_identity():
    ident = identity_map(16, 32)
    for sal, a in ((_blobs(), 0.0), (torch.zeros(16, 32), 0.5), (torch.full((16, 32), 3.0), 0.5)):
        m = make_map2d(sal, a, 0.25, 4)
        assert torch.allclose(m.ctrl, ident.ctrl, atol=1e-12), (a, float(sal.mean()))
    assert make_map2d(torch.zeros(16, 32), 0.5, 0.25, 4).is_identity()


def test_borders_map_to_borders_and_the_clip_holds():
    for seed in range(20):
        for scale in (0.25, 0.5):
            for alpha, kernel in ((0.25, 2.0), (0.5, 6.0), (1.0, 3.0)):
                m = make_map2d(_blobs(seed=seed), alpha, scale, kernel)
                c = m.ctrl
                assert torch.all(c[0, :, 0] == 0) and torch.all(c[0, :, -1] == 1)
                assert torch.all(c[1, 0, :] == 0) and torch.all(c[1, -1, :] == 1)
                assert map_ok(c, scale) and 0.0 <= m.t <= 1.0
                assert (c >= 0).all() and (c <= 1).all()


def test_the_backoff_is_used_only_when_needed():
    sal = _blobs(seed=3)
    weak = make_map2d(sal, 0.05, 0.25, 6)
    assert weak.t == 1.0
    strong = make_map2d(sal, 1.0, 0.5, 2)
    assert strong.t < 1.0
    full = torch.stack(torch.meshgrid(torch.arange(17) / 16, torch.arange(33) / 32, indexing="ij")[::-1]).double()
    assert not map_ok(full + displacement(sal, 1.0, 2), 0.5)


def test_singular_values_of_known_matrices():
    j = torch.tensor([[[2.0, 0.0], [0.0, 0.5]], [[0.0, -1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 1.0]]],
                     dtype=torch.float64)
    smax, smin, det = singular_values(j)
    ref = torch.linalg.svdvals(j)
    assert torch.allclose(smax, ref[:, 0]) and torch.allclose(smin, ref[:, 1])
    assert torch.allclose(det, torch.linalg.det(j))


def test_corner_jacobians_of_an_affine_map():
    a = torch.tensor([[1.2, 0.1], [-0.05, 0.8]], dtype=torch.float64)
    ident = identity_map(4, 8).ctrl
    ctrl = torch.einsum("rc,chw->rhw", a, ident)
    j = corner_jacobians(ctrl)
    assert j.shape == (4, 4, 8, 2, 2) and torch.allclose(j, a.expand_as(j))


def test_a_salient_block_is_magnified_locally_not_along_its_row_and_column():
    """The local warp gives a salient block more output pixels than the uniform map; a cell far
    away on the block's own row keeps about its share (the separable warp of Method 5 would
    magnify the whole band)."""
    sal = torch.zeros(16, 32, dtype=torch.float64)
    sal[7:9, 15:17] = 1.0
    m = make_map2d(sal, 0.5, 0.25, 2)
    j = corner_jacobians(m.ctrl).mean(0)  # (gh, gw, 2, 2)
    det = j[..., 0, 0] * j[..., 1, 1] - j[..., 0, 1] * j[..., 1, 0]
    mag = det.rsqrt()  # output pixels per source pixel, relative to the uniform map
    assert mag[7:9, 15:17].mean() > 1.2
    # far along the block's own rows, the local map keeps the uniform height and area...
    assert abs(float(mag[8, 1]) - 1) < 0.05 and abs(1 / float(j[8, 1, 1, 1]) - 1) < 0.05
    # ...while the separable warp stretches the whole band of rows, there too
    g = make_grid(sal, 0.5, 0.25)
    assert float(g.dy[8]) > 1.2


def test_inverse_undoes_the_forward_map():
    m = make_map2d(_blobs(seed=1), 0.5, 0.25, 3)
    assert not m.is_identity()
    y, x = torch.meshgrid(_centres(64), _centres(128), indexing="ij")
    uy, ux, res = inverse_map(m, y, x)
    fx, fy = eval_map(m, uy, ux)
    assert res < 1e-8 and (fx - x).abs().max() < 1e-8 and (fy - y).abs().max() < 1e-8
    # and the forward map at u, then back
    gy, gx = torch.meshgrid(_centres(40), _centres(80), indexing="ij")
    fx, fy = eval_map(m, gy, gx)
    by, bx, _ = inverse_map(m, fy, fx)
    assert (by - gy).abs().max() < 1e-7 and (bx - gx).abs().max() < 1e-7


@pytest.mark.parametrize("scale", [0.25, 0.375, 0.5])
def test_identity_warp_is_the_baseline_resize(scale):
    frame = torch.rand(2, 3, 64, 128, generator=torch.Generator().manual_seed(0))
    size = (int(round(64 * scale)), int(round(128 * scale)))
    ref = F.interpolate(frame, size=size, mode="bilinear", antialias=True, align_corners=False)
    out = warp_frame2d(frame, identity_map(4, 8), size)
    assert out.shape == ref.shape and (out - ref).abs().max() < 1e-5


@pytest.mark.parametrize("scale,alpha", [(0.25, 0.5), (0.375, 0.75), (0.5, 0.25)])
def test_given_a_separable_map_the_2d_resampling_is_method5s(scale, alpha):
    """resample_aa with a separable map's centres and footprints is warp_frame(antialias=True)."""
    frame = torch.rand(1, 3, 64, 128, generator=torch.Generator().manual_seed(1))
    g = make_grid(torch.rand(4, 8, generator=torch.Generator().manual_seed(2)), alpha, scale)
    hh, ww = int(round(64 * scale)), int(round(128 * scale))
    yc, xc = _forward64(g.dy, hh) * 64, _forward64(g.dx, ww) * 128
    from warp.grid import _interp, _knots

    def edges(d, n_out, n_src):
        xs, us = _knots(d)
        return _interp(torch.arange(n_out + 1, dtype=torch.float64) / n_out, us, xs) * n_src

    ey, ex = edges(g.dy, hh, 64), edges(g.dx, ww, 128)
    cy, cx = torch.meshgrid(yc, xc, indexing="ij")
    sy, sx = torch.meshgrid(ey[1:] - ey[:-1], ex[1:] - ex[:-1], indexing="ij")
    out = resample_aa(frame, cy, cx, sy, sx)
    ref = warp_frame(frame, g, (hh, ww), antialias=True)
    assert (out - ref).abs().max() < 1e-5


@pytest.mark.parametrize("align", [False, True])
def test_identity_unwarp_is_upsample_logits_with_a_stride_and_padding(align):
    logits = torch.randn(2, 19, 5, 9)
    ref = upsample_logits(logits, (72, 140), 8, align, input_size=(36, 70))
    out = unwarp_logits2d(logits, identity_map(9, 14), (72, 140), align, output_stride=8, input_size=(36, 70))
    assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("align", [False, True])
def test_given_a_separable_inverse_the_unwarp_is_method5s(align):
    g = make_grid(torch.rand(4, 8, generator=torch.Generator().manual_seed(4)), 0.5, 0.5)
    logits = torch.randn(1, 19, 8, 16)
    uy, ux = torch.meshgrid(_inverse64(g.dy, 64), _inverse64(g.dx, 128), indexing="ij")
    out = unwarp_logits_at(logits, uy, ux, align, output_stride=4)
    ref = unwarp_logits(logits, g, (64, 128), align, output_stride=4)
    assert torch.allclose(out, ref, atol=1e-5)


def test_warp_then_unwarp_keeps_a_salient_square_sharper():
    """A small square, salient, survives the round trip at scale 0.25 better with the local warp
    than with the uniform map (image used as 3 'logits')."""
    frame = torch.zeros(1, 3, 64, 128)
    frame[:, :, 28:36, 60:68] = 1.0
    sal = torch.zeros(8, 16)
    sal[3:5, 7:9] = 1.0
    m = make_map2d(sal, 0.75, 0.25, 1.5)
    errs = []
    for mm in (identity_map(8, 16), m):
        small = warp_frame2d(frame, mm, (16, 32))
        back = unwarp_logits2d(small, mm, (64, 128), align_corners=False)
        errs.append(float((back - frame)[:, :, 24:40, 56:72].abs().mean()))
    assert errs[1] < errs[0] * 0.9, errs


def test_distortion_figures():
    d = distortion(identity_map(8, 16))
    assert d == pytest.approx({"aniso_mean": 1.0, "aniso_p95": 1.0, "mag_p5": 1.0, "mag_p95": 1.0, "t": 1.0})
    d = distortion(make_map2d(_blobs(seed=2), 0.5, 0.25, 3))
    assert d["aniso_mean"] >= 1.0 and d["aniso_p95"] >= d["aniso_mean"] - 1e-6 and d["mag_p5"] < 1 < d["mag_p95"]


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        make_map2d(torch.ones(2, 3, 4), 0.5, 0.5, 2)
    with pytest.raises(ValueError):
        make_map2d(torch.ones(4, 8), 1.5, 0.5, 2)
    with pytest.raises(ValueError):
        make_map2d(torch.ones(4, 8), 0.5, 0.5, 0)
    with pytest.raises(ValueError):
        make_map2d(torch.ones(4, 8), 0.5, 1.5, 2)
    with pytest.raises(ValueError):
        make_map2d(-torch.ones(4, 8), 0.5, 0.5, 2)
    with pytest.raises(ValueError, match="cannot cover"):
        unwarp_logits_at(torch.randn(1, 2, 4, 4), *torch.meshgrid(_centres(8), _centres(8), indexing="ij"), False,
                         output_stride=2, input_size=(10, 10))
    assert isinstance(identity_map(2, 2), Map2D)
