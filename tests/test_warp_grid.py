"""Method 5: the separable warp, its inverse, and its two resampling steps (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from foveation.ops import downsample  # noqa: E402
from models.utils import upsample_logits  # noqa: E402
from warp.grid import (axis_density, forward_coords, inverse_coords, make_grid, unwarp_logits,  # noqa: E402
                       warp_frame)


def test_alpha_zero_or_empty_saliency_is_uniform():
    for sal, a in ((torch.rand(4, 8), 0.0), (torch.zeros(4, 8), 0.75)):
        g = make_grid(sal, a, 0.5)
        assert torch.allclose(g.dx, torch.ones(8)) and torch.allclose(g.dy, torch.ones(4))
        assert g.is_uniform() and torch.isfinite(g.dx).all() and torch.isfinite(g.dy).all()
    assert torch.allclose(forward_coords(torch.ones(8), 16), (torch.arange(16) + 0.5) / 16)
    assert torch.allclose(inverse_coords(torch.ones(8), 16), (torch.arange(16) + 0.5) / 16)


def test_density_respects_the_clip_and_has_mean_one():
    p = torch.zeros(128)
    p[40] = 1.0  # one hot cell
    for scale in (0.25, 0.375, 0.5, 0.75):
        d = axis_density(p, 0.75, scale)
        assert d.min() >= 0.5 - 1e-5 and d.max() <= 1 / scale + 1e-5
        assert abs(float(d.mean()) - 1) < 1e-5 and torch.isfinite(d).all()
        assert d.argmax() == 40


def test_density_on_many_random_profiles():
    gen = torch.Generator().manual_seed(0)
    for n in range(200):
        g = 64 if n % 2 else 128
        p = torch.rand(g, generator=gen) ** (1 + n % 30) * (torch.rand(g, generator=gen) < 0.3)
        for scale in (0.25, 0.75):
            d = axis_density(p, 0.75, scale)
            assert d.min() >= 0.5 - 1e-5 and d.max() <= 1 / scale + 1e-5 and abs(float(d.mean()) - 1) < 1e-5


def test_bad_density_inputs_raise():
    with pytest.raises(ValueError):
        axis_density(torch.ones(8), 1.5, 0.5)
    with pytest.raises(ValueError):
        axis_density(torch.ones(8), 0.5, 1.5)  # 1/scale < 1: no density of mean 1 fits the clip
    with pytest.raises(ValueError):
        axis_density(torch.tensor([1.0, -1.0]), 0.5, 0.5)
    with pytest.raises(ValueError):
        make_grid(torch.ones(2, 3, 4), 0.5, 0.5)


def test_coords_are_monotone_and_inverse_undoes_forward():
    d = axis_density(torch.rand(32), 0.75, 0.5)
    f, inv = forward_coords(d, 64), inverse_coords(d, 256)
    assert (f[1:] > f[:-1]).all() and (inv[1:] > inv[:-1]).all()
    # source x -> output u -> back to x through the forward map's interpolation
    x = (torch.arange(256) + 0.5) / 256
    back = torch.from_numpy(np.interp(inv.numpy(), (np.arange(64) + 0.5) / 64, f.numpy()))
    assert (back - x)[8:-8].abs().max() < 2 / 64


def test_forward_and_inverse_are_exact_inverses_of_the_cdf():
    d = axis_density(torch.rand(16), 0.5, 0.375)
    u = inverse_coords(d, 1000)
    x = (torch.arange(1000, dtype=torch.float64) + 0.5) / 1000
    # forward_coords at u = inverse(x) must give x back: sample the forward map at those u
    from warp.grid import _interp, _knots

    xs, us = _knots(d)
    assert (_interp(u.double(), us, xs) - x).abs().max() < 1e-6


def test_a_salient_block_is_magnified():
    sal = torch.zeros(4, 8)
    sal[1, 2] = 1.0
    g = make_grid(sal, 0.75, 0.5)
    frame = torch.zeros(1, 3, 64, 128)
    frame[..., 16:32, 32:48] = 1.0  # the salient cell
    warped = warp_frame(frame, g, (32, 64))
    assert warped.mean() > 1.5 * frame.mean()  # the block takes a larger share of the output


def test_warp_samples_the_forward_coordinates():
    """A frame that holds its own x coordinate comes out holding the forward map."""
    sal = torch.rand(4, 8)
    g = make_grid(sal, 0.75, 0.5)
    ramp = ((torch.arange(128) + 0.5) / 128).expand(1, 3, 64, 128).contiguous()
    out = warp_frame(ramp, g, (32, 64), supersample=1)
    fx = forward_coords(g.dx, 64)
    assert torch.allclose(out[0, 0, 5], fx, atol=1e-5)
    assert torch.allclose(out[0, 2, 17], fx, atol=1e-5)  # the same columns on every row


def test_alpha_zero_warp_is_close_to_the_antialiased_downscale():
    """The 2x supersampled warp approximates the baseline's antialiased (triangle) resize with
    box filters: at scale 0.5 it is exactly a 2 x 2 average. On this smooth noise it stays within
    0.04 of the baseline (mean under 0.01), and at every scale it is at least as close as a plain
    ``grid_sample`` (the plan's first guess of 0.02 was too tight for a box filter)."""
    frame = torch.rand(1, 3, 64, 128, generator=torch.Generator().manual_seed(0))
    frame = F.avg_pool2d(frame, 5, 1, 2, count_include_pad=False)  # smooth content
    uniform = make_grid(torch.zeros(4, 8), 0.0, 0.5)
    assert torch.allclose(warp_frame(frame, uniform, (32, 64)), F.avg_pool2d(frame, 2), atol=1e-6)
    for scale in (0.25, 0.375, 0.5, 0.75):
        ref = downsample(frame, scale)
        size = tuple(ref.shape[-2:])
        out = warp_frame(frame, make_grid(torch.zeros(4, 8), 0.0, scale), size)
        plain = warp_frame(frame, make_grid(torch.zeros(4, 8), 0.0, scale), size, supersample=1)
        err, err_plain = (out - ref).abs(), (plain - ref).abs()
        assert err.max() < 0.04 and err.mean() < 0.01, scale
        assert err.mean() <= err_plain.mean() + 1e-6, scale


@pytest.mark.parametrize("align", [False, True])
def test_alpha_zero_unwarp_is_the_baseline_upsampling(align):
    logits = torch.randn(1, 19, 8, 16)
    ref = F.interpolate(logits, size=(64, 128), mode="bilinear", align_corners=align)
    out = unwarp_logits(logits, make_grid(torch.zeros(4, 8), 0.0, 0.5), (64, 128), align)
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("align", [False, True])
def test_alpha_zero_unwarp_is_upsample_logits_with_a_stride_and_padding(align):
    """A stride-8 model on a 36 x 70 input padded to 40 x 72: 5 x 9 logits, frame 72 x 140."""
    logits = torch.randn(2, 19, 5, 9)
    ref = upsample_logits(logits, (72, 140), 8, align, input_size=(36, 70))
    out = unwarp_logits(logits, make_grid(torch.zeros(4, 8), 0.0, 0.5), (72, 140), align, output_stride=8,
                        input_size=(36, 70))
    assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5)
    with pytest.raises(ValueError, match="cannot cover"):
        unwarp_logits(logits, make_grid(torch.zeros(4, 8), 0.0, 0.5), (72, 140), align, output_stride=4,
                      input_size=(36, 70))


@pytest.mark.parametrize("align", [False, True])
def test_stride_one_unwarp_at_the_frame_size_is_the_identity(align):
    logits = torch.randn(1, 19, 16, 32)
    out = unwarp_logits(logits, make_grid(torch.zeros(2, 4), 0.0, 0.5), (16, 32), align)
    assert torch.allclose(out, upsample_logits(logits, (16, 32), 1, align), atol=1e-5)


def test_unwarp_reads_the_inverse_coordinates():
    """Logits that hold their own output x coordinate come back holding the inverse map."""
    sal = torch.rand(4, 8)
    g = make_grid(sal, 0.75, 0.5)
    w = 64
    ramp = ((torch.arange(w) + 0.5) / w).expand(1, 2, 32, w).contiguous()
    out = unwarp_logits(ramp, g, (64, 128), align_corners=False)
    inv = inverse_coords(g.dx, 128).clamp(0.5 / w, 1 - 0.5 / w)  # border: the outer half pixels clamp
    assert torch.allclose(out[0, 0, 3], inv, atol=1e-5) and torch.allclose(out[0, 1, 60], inv, atol=1e-5)


def test_warp_then_unwarp_keeps_a_salient_square_sharper():
    """A thin square in the salient cell survives a 0.25 downscale better warped than uniform."""
    frame = torch.zeros(1, 1, 128, 256)
    frame[..., 40:43, 80:83] = 1.0  # 3 x 3 px: below one pixel at scale 0.25
    sal = torch.zeros(8, 16)
    sal[2, 5] = 1.0  # cell (2, 5) of 16 px holds it
    errs = []
    for alpha in (0.0, 0.75):
        g = make_grid(sal, alpha, 0.25)
        small = warp_frame(frame, g, (32, 64))
        back = unwarp_logits(small, g, (128, 256), align_corners=False)
        errs.append(float((back - frame).abs()[..., 32:48, 72:96].sum()))
    assert errs[1] < 0.7 * errs[0]


@pytest.mark.parametrize("scale", [0.25, 0.375, 0.5, 0.75, 1.0])
def test_antialiased_warp_at_alpha_zero_is_the_baseline_resize(scale):
    """The antialiased resampling reproduces torch's antialiased bilinear resize (the baseline's)
    to float precision, at every scale of the grid (at 1, both are the identity)."""
    frame = torch.rand(2, 3, 64, 128, generator=torch.Generator().manual_seed(1))
    ref = downsample(frame, scale)
    out = warp_frame(frame, make_grid(torch.zeros(4, 8), 0.0, scale), tuple(ref.shape[-2:]), antialias=True)
    assert out.shape == ref.shape and (out - ref).abs().max() < 1e-4


def test_antialiased_warp_keeps_the_mean_and_magnifies_the_salient_block():
    sal = torch.zeros(4, 8)
    sal[1, 2] = 1.0
    g = make_grid(sal, 0.75, 0.25)
    flat = torch.full((1, 3, 64, 128), 0.3)
    assert torch.allclose(warp_frame(flat, g, (16, 32), antialias=True), torch.full((1, 3, 16, 32), 0.3), atol=1e-6)
    frame = torch.zeros(1, 3, 64, 128)
    frame[..., 16:32, 32:48] = 1.0
    assert warp_frame(frame, g, (16, 32), antialias=True).mean() > 1.5 * frame.mean()


def test_antialiased_filter_widens_where_the_warp_compresses():
    """The filter spans the source pixels each output pixel covers: 1 / (scale x density)."""
    from warp.grid import _axis_filter

    d = axis_density(torch.tensor([1.0, 0.0, 0.0, 0.0]), 0.75, 0.25)  # cell 0 magnified, the rest compressed
    idx, w = _axis_filter(d, 256, 64)
    taps = (w > 0).sum(1)
    assert torch.allclose(w.sum(1), torch.ones(64, dtype=w.dtype))
    assert taps[:8].max() < taps[-8:].min()  # fewer taps where magnified, more where compressed
    centre = ((idx.double() + 0.5) * w).sum(1)  # each output pixel's footprint moves forward
    assert (centre[1:] > centre[:-1]).all()
