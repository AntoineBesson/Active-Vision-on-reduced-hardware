"""Motion ops of Method 3: homography, warp, aligned frame difference, moving cells (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

import torch.nn.functional as F  # noqa: E402

from foveation.ops import GlimpseGrid  # noqa: E402
from reuse.motion import (  # noqa: E402
    MOTION_FACTOR,
    estimate_homography,
    motion_map,
    moving_cells,
    to_gray_small,
    warp,
)

FH, FW = 1024, 2048
SHIFT = (12, 20)  # (dy, dx): content moves down-right between the two frames


def _texture(h, w, seed):
    """Smooth random texture in [0, 1] (features survive the 1/4 downsample)."""
    g = torch.Generator().manual_seed(seed)
    small = torch.rand(1, 1, h // 16 + 2, w // 16 + 2, generator=g)
    return F.interpolate(small, size=(h + 32, w + 32), mode="bicubic", align_corners=False)[..., 16:16 + h, 16:16 + w].clamp(0, 1)


def _frames(dy=SHIFT[0], dx=SHIFT[1], square=None):
    """prev = big[pad:pad+FH, pad:pad+FW]; cur = the same content moved by (dy, dx)."""
    pad = 80
    big = _texture(FH + 2 * pad, FW + 2 * pad, 0)
    prev = big[..., pad:pad + FH, pad:pad + FW].clone()
    cur = big[..., pad - dy:pad - dy + FH, pad - dx:pad - dx + FW].clone()
    if square is not None:  # (top, left, extra dx): an object moving on its own
        top, left, extra = square
        obj = _texture(128, 128, 1)
        prev[..., top:top + 128, left:left + 128] = obj
        cur[..., top + dy:top + dy + 128, left + dx + extra:left + dx + extra + 128] = obj
    return prev.expand(1, 3, FH, FW), cur.expand(1, 3, FH, FW)


@pytest.mark.parametrize("method", ["orb", "iclk"])
def test_estimators_recover_a_known_translation(method):
    prev, cur = _frames()
    H, inliers = estimate_homography(to_gray_small(prev), to_gray_small(cur), method=method)
    assert H.shape == (3, 3) and H.dtype == torch.float64
    src = torch.tensor([[200.0, 300.0, 1.0], [1800.0, 900.0, 1.0]], dtype=torch.float64).T
    dst = H @ src
    dst = dst[:2] / dst[2]
    assert torch.allclose(dst - src[:2], torch.tensor([[SHIFT[1]], [SHIFT[0]]], dtype=torch.float64), atol=1.0)
    if method == "orb":
        assert inliers >= 50


@pytest.mark.parametrize("method", ["orb", "iclk"])
def test_estimates_are_deterministic(method):
    prev, cur = _frames()
    a = estimate_homography(to_gray_small(prev), to_gray_small(cur), method=method)
    b = estimate_homography(to_gray_small(prev), to_gray_small(cur), method=method)
    assert torch.equal(a[0], b[0]) and a[1] == b[1]


def test_textureless_frame_gives_identity():
    flat = torch.full((1, 3, FH, FW), 0.5)
    H, inliers = estimate_homography(to_gray_small(flat), to_gray_small(flat), method="orb")
    assert torch.equal(H, torch.eye(3, dtype=torch.float64)) and inliers == 0


def test_aligned_difference_flags_only_the_moving_object():
    prev, cur = _frames(square=(400, 1000, 40))
    gp, gc = to_gray_small(prev), to_gray_small(cur)
    H, _ = estimate_homography(gp, gc, method="orb")
    grid = GlimpseGrid((FH, FW), 384, 32)  # inner 320: 4 x 7 cells
    moving = moving_cells(motion_map(gp, gc, H), grid, theta=30, rho=0.01)
    expected = {c for c in range(grid.num_cells) if c // grid.cols == 0 or c % grid.cols == 0}  # uncovered borders
    expected.add(1 * grid.cols + 3)  # the square, before and after, lies inside cell (1, 3)
    assert set(torch.nonzero(moving).flatten().tolist()) == expected
    raw = moving_cells(motion_map(gp, gc, torch.eye(3, dtype=torch.float64)), grid, theta=30, rho=0.01)
    assert raw.sum() >= grid.num_cells // 2  # without the homography, ego-motion lights up the frame


def test_border_pixels_without_source_count_as_moving():
    prev, cur = _frames(dy=0, dx=64)
    gp, gc = to_gray_small(prev), to_gray_small(cur)
    H = torch.tensor([[1, 0, 64], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    m = motion_map(gp, gc, H)
    assert m.dtype == torch.uint8 and m.shape == (FH // MOTION_FACTOR, FW // MOTION_FACTOR)
    strip = 64 // MOTION_FACTOR
    assert (m[:, :strip] == 255).all() and (m[:, strip + 4:] < 30).all()


def test_warp_identity_is_exact():
    x = torch.randint(0, 19, (1, 1, 32, 64)).float()
    out, valid = warp(x, torch.eye(3, dtype=torch.float64), "nearest")
    assert torch.equal(out, x) and valid.all() and valid.shape == (32, 64)
