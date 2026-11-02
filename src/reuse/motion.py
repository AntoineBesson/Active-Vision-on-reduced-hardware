"""Motion cue of Method 3: a global homography cancels ego-motion, then the frame difference.

The car's own motion moves the whole image between two frames, so a raw
difference f_t - f_{t-1} lights up almost everything. One homography H, fitted
on downscaled grayscale frames, maps the previous frame onto the current one;
the difference after that warp is mostly the independently moving objects,
plus the parallax of close static structures and the content entering at the
borders (pixels the warp leaves without a source, which always count as moving).

Conventions: H maps *full-resolution* pixel coordinates of the previous frame
onto the current frame, pixel centres at integer coordinates. Motion runs at
1/MOTION_FACTOR of the frame resolution.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from foveation.ops import GlimpseGrid

__all__ = ["MOTION_FACTOR", "to_gray_small", "estimate_homography", "warp", "motion_map", "moving_cells"]

MOTION_FACTOR = 4
MIN_INLIERS = 8
IDENTITY = torch.eye(3, dtype=torch.float64)


def to_gray_small(frame: torch.Tensor, factor: int = MOTION_FACTOR) -> torch.Tensor:
    """(1, 3, H, W) RGB in [0, 1] -> (H/factor, W/factor) BT.601 luma, area-downsampled."""
    luma = (0.299 * frame[:, 0] + 0.587 * frame[:, 1] + 0.114 * frame[:, 2]).unsqueeze(1)
    return F.avg_pool2d(luma, factor)[0, 0].float()


def _scale(factor: float) -> torch.Tensor:
    """Pixel coordinates at 1/factor resolution -> full resolution (pixel centres)."""
    f = float(factor)
    return torch.tensor([[f, 0, (f - 1) / 2], [0, f, (f - 1) / 2], [0, 0, 1]], dtype=torch.float64)


def _to_norm(h: int, w: int) -> torch.Tensor:
    """Pixel coordinates -> grid_sample's normalised coordinates (align_corners=False)."""
    return torch.tensor([[2 / w, 0, 1 / w - 1], [0, 2 / h, 1 / h - 1], [0, 0, 1]], dtype=torch.float64)


def _sane(H: torch.Tensor) -> bool:
    det = float(torch.det(H[:2, :2]))
    return bool(torch.isfinite(H).all()) and 1e-3 <= abs(det) <= 1e3


def _orb(prev: torch.Tensor, cur: torch.Tensor) -> Tuple[Optional[np.ndarray], int]:
    import cv2

    a = (prev.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    b = (cur.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    orb = cv2.ORB_create(nfeatures=2000)  # 8 pyramid levels: robust to the zoom of forward driving
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < MIN_INLIERS:
        return None, 0
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    cv2.setRNGSeed(0)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
    inliers = 0 if mask is None else int(mask.sum())
    if inliers < MIN_INLIERS:
        return None, inliers
    # Coarse-level keypoints have coarse positions: refit on the inliers found at full resolution.
    fine = mask.ravel().astype(bool) & np.array([ka[m.queryIdx].octave == 0 and kb[m.trainIdx].octave == 0
                                                 for m in matches])
    if fine.sum() >= MIN_INLIERS:
        refit, _ = cv2.findHomography(src[fine], dst[fine], 0)
        if refit is not None:
            H = refit
    return H, inliers


def _iclk(prev: torch.Tensor, cur: torch.Tensor, iters: Tuple[int, ...] = (30, 15, 10)) -> Tuple[Optional[torch.Tensor], int]:
    """Inverse-compositional Lucas-Kanade for an 8-parameter homography, coarse to fine.

    Works in grid_sample's normalised coordinates, which are the same at every
    pyramid level; returns H in those coordinates (prev -> cur) and the number of
    valid pixels at the finest level."""
    dev = prev.device
    h0, w0 = prev.shape
    levels = [(h0 // 4, w0 // 4), (h0 // 2, w0 // 2), (h0, w0)]
    Hn = torch.eye(3, dtype=torch.float64, device=dev)
    valid_count = 0
    for (h, w), n in zip(levels, iters):
        T = F.interpolate(prev[None, None].double(), size=(h, w), mode="area")[0, 0]
        I = F.interpolate(cur[None, None].double(), size=(h, w), mode="area")[0, 0]
        ys = (torch.arange(h, dtype=torch.float64, device=dev) * 2 + 1) / h - 1
        xs = (torch.arange(w, dtype=torch.float64, device=dev) * 2 + 1) / w - 1
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        gy, gx = torch.gradient(T, spacing=(2 / h, 2 / w))
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        # dW/dp at the identity, p = (h11-1, h12, h13, h21, h22-1, h23, h31, h32)
        sd = torch.stack([gx * x, gx * y, gx * one, gy * x, gy * y, gy * one,
                          -(gx * x + gy * y) * x, -(gx * x + gy * y) * y], -1)  # (h, w, 8)
        pts = torch.stack([x, y, one], -1).reshape(-1, 3).T  # (3, h*w)
        for _ in range(n):
            q = Hn @ pts  # where the template's pixels land in cur
            qx, qy = q[0] / q[2], q[1] / q[2]
            grid = torch.stack([qx, qy], -1).reshape(1, h, w, 2).float()
            Iw = F.grid_sample(I[None, None].float(), grid, mode="bilinear", align_corners=False)[0, 0].double()
            valid = ((qx.abs() <= 1) & (qy.abs() <= 1)).reshape(h, w)
            if int(valid.sum()) < 64:
                return None, 0
            e = (Iw - T)[valid]
            J = sd[valid]
            dp = torch.linalg.lstsq(J.T @ J, J.T @ e).solution
            dH = torch.eye(3, dtype=torch.float64, device=dev) + torch.tensor(
                [[dp[0], dp[1], dp[2]], [dp[3], dp[4], dp[5]], [dp[6], dp[7], 0.0]], dtype=torch.float64, device=dev)
            Hn = Hn @ torch.linalg.inv(dH)
            Hn = Hn / Hn[2, 2]
            valid_count = int(valid.sum())
    return Hn, valid_count


def estimate_homography(prev: torch.Tensor, cur: torch.Tensor, method: str = "orb",
                        factor: int = MOTION_FACTOR) -> Tuple[torch.Tensor, int]:
    """H (3x3 float64, full-resolution pixels, prev -> cur) from two (h, w) gray frames at 1/factor.

    ``orb``: OpenCV ORB features + RANSAC (seeded), refit on the full-resolution inliers (CPU). ``iclk``: inverse-compositional
    Lucas-Kanade on a 3-level pyramid (torch, runs on the frames' device). The second
    value is the support (RANSAC inliers, or valid pixels for ``iclk``). A failed or
    degenerate fit gives the identity and 0."""
    h, w = prev.shape
    if method == "orb":
        Hs, inliers = _orb(prev, cur)
        if Hs is None:
            return IDENTITY.clone(), 0
        Hs = torch.from_numpy(Hs).double()
    elif method == "iclk":
        Hn, inliers = _iclk(prev, cur)
        if Hn is None:
            return IDENTITY.clone(), 0
        N = _to_norm(h, w).to(Hn.device)
        Hs = (torch.linalg.inv(N) @ Hn @ N).cpu()
    else:
        raise ValueError(f"unknown homography method {method!r} (orb, iclk)")
    S = _scale(factor)
    H = S @ Hs @ torch.linalg.inv(S)
    H = H / H[2, 2]
    if not _sane(H):
        return IDENTITY.clone(), 0
    return H, inliers


def warp(x: torch.Tensor, H: torch.Tensor, mode: str,
         frame_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp (N, C, h, w) from the previous frame onto the current one; returns (warped, valid).

    ``H``: full-resolution pixels of a ``frame_size`` frame (default: x's own size).
    Backward mapping; pixels whose source falls outside the frame replicate the
    border (``valid`` is False there). ``mode``: ``nearest`` or ``bilinear``."""
    n, c, h, w = x.shape
    fh, fw = frame_size or (h, w)
    Hn = _to_norm(fh, fw) @ H.double().cpu() @ torch.linalg.inv(_to_norm(fh, fw))  # same at every resolution
    inv = torch.linalg.inv(Hn).to(x.device)
    ys = (torch.arange(h, dtype=torch.float64, device=x.device) * 2 + 1) / h - 1
    xs = (torch.arange(w, dtype=torch.float64, device=x.device) * 2 + 1) / w - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    q = inv @ torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(-1, 3).T
    qx, qy = q[0] / q[2], q[1] / q[2]
    valid = ((qx.abs() <= 1) & (qy.abs() <= 1)).reshape(h, w)
    grid = torch.stack([qx, qy], -1).reshape(1, h, w, 2).to(x.dtype if x.is_floating_point() else torch.float32)
    out = F.grid_sample(x if x.is_floating_point() else x.float(), grid.expand(n, h, w, 2), mode=mode,
                        padding_mode="border", align_corners=False)
    return out.to(x.dtype), valid


def motion_map(prev: torch.Tensor, cur: torch.Tensor, H: torch.Tensor,
               factor: int = MOTION_FACTOR) -> torch.Tensor:
    """(h, w) uint8: |cur - prev warped by H| x 255, 5x5 box blur; 255 where the warp has no source."""
    h, w = cur.shape
    warped, valid = warp(prev[None, None].float(), H, "bilinear", frame_size=(h * factor, w * factor))
    diff = (cur[None, None].float() - warped).abs() * 255
    diff = F.avg_pool2d(diff, 5, stride=1, padding=2, count_include_pad=False)[0, 0]
    diff[~valid] = 255
    return diff.round().clamp(0, 255).to(torch.uint8)


def moving_cells(motion: torch.Tensor, grid: GlimpseGrid, theta: int, rho: float) -> torch.Tensor:
    """(num_cells,) bool: cells whose share of in-frame pixels with motion > theta exceeds rho.

    ``motion`` is at 1/f of the grid's frame size; the cell's inner square must be a
    whole number of motion pixels."""
    h, w = motion.shape
    f = grid.frame_size[0] // h
    if grid.frame_size != (h * f, w * f) or grid.inner % f:
        raise ValueError(f"motion map {h}x{w} does not tile grid {grid.frame_size} with inner {grid.inner}")
    s = grid.inner // f
    ch, cw = grid.rows * s, grid.cols * s
    hot = F.pad((motion > theta).float()[None, None], (0, cw - w, 0, ch - h))
    inside = F.pad(torch.ones(1, 1, h, w, device=motion.device), (0, cw - w, 0, ch - h))
    share = F.avg_pool2d(hot, s) / F.avg_pool2d(inside, s).clamp_min(1e-9)
    return (share > rho).flatten()
