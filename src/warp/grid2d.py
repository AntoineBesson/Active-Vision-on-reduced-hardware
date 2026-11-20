"""Method 6: a local (2-D) warp of the frame by kernel sampling, its inverse, and its two resampling steps.

Spec: docs/superpowers/specs/2026-11-17-method6-local-warp-bound-design.md ("The mechanism, per
frame"). Coordinates are normalised as in :mod:`warp.grid`: pixel i of n covers [i/n, (i+1)/n].

The map sends an output position u (normalised, in [0, 1]^2) to the source position x(u) it
samples. It is held at the (gh + 1) x (gw + 1) corners of the saliency's cell grid (the
*control points*) and is bilinear in between (:class:`Map2D`). At control point u (in cells)

    x(u) = u + t * (D_a(u) - D_1(u)),   D_a(u) = sum_v a(v) k(u - v) (v - u) / sum_v a(v) k(u - v),

with ``a = (1 - alpha) + alpha * s / mean(s)`` the attraction, v the cell centres, k a Gaussian of
``kernel`` cells (cells are square in pixels, so the kernel is isotropic), the attraction reflected
across the frame's edges, and D_1 the same sum for a constant attraction (the bias of the
truncated, reflected kernel, so that a constant attraction is exactly the identity: the
"Learning to Zoom" sampler, Recasens et al. 2018, used as a fixed formula). The component of the
displacement normal to an edge is set to 0 on that edge: the border maps to the border.

The clip (the 2-D version of Method 5's density clip): the singular values of J = dx/du stay in
[``scale``, 1 / ``d_min``] at the four corners of every control cell, which also keeps every
cell from folding. Where the map breaks it, t is reduced by bisection.

:func:`warp_frame2d` resamples with the baseline's antialiased triangle filter, its half-width
along each source axis being the extent of the output pixel's footprint on that axis (at least
1); :func:`resample_aa` takes the per-pixel centres and spans directly, so that it can be checked
against :func:`warp.grid.warp_frame` on a separable map. :func:`unwarp_logits2d` finds, for each
frame pixel centre, its output coordinate by Newton's method on the bilinear map
(:func:`inverse_map`) and samples the logits there once, with the model's convention, as
:func:`warp.grid.unwarp_logits` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .grid import D_MIN, _logit_positions

__all__ = ["Map2D", "BACKOFF_STEPS", "identity_map", "make_map2d", "displacement", "corner_jacobians",
           "singular_values", "map_ok", "eval_map", "inverse_map", "resample_aa", "warp_frame2d",
           "unwarp_logits2d", "unwarp_logits_at", "distortion"]

BACKOFF_STEPS = 12  # bisection steps of the backoff t (the spec)
NEWTON_ITERS = 50
NEWTON_TOL = 1e-9  # normalised: 2e-6 of a pixel on a 2048-px axis


@dataclass
class Map2D:
    """Source position of each control point: ``ctrl`` (2, gh + 1, gw + 1) float64, [x, y] in
    [0, 1]; ``t`` the backoff that was applied (1: none)."""

    ctrl: torch.Tensor
    t: float = 1.0

    @property
    def cells(self) -> Tuple[int, int]:
        return self.ctrl.shape[1] - 1, self.ctrl.shape[2] - 1

    def is_identity(self) -> bool:
        return bool((self.ctrl == identity_map(*self.cells, device=self.ctrl.device).ctrl).all())


def _identity_ctrl(gh: int, gw: int, device=None) -> torch.Tensor:
    ys = torch.arange(gh + 1, dtype=torch.float64, device=device) / gh
    xs = torch.arange(gw + 1, dtype=torch.float64, device=device) / gw
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy])


def identity_map(gh: int, gw: int, device=None) -> Map2D:
    return Map2D(_identity_ctrl(gh, gw, device), 1.0)


def _kernel_matrix(n: int, sigma: float, pad: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(n + 1, n + 2 pad) Gaussian weights from the n + 1 corners to the cell centres of the
    reflected axis, and the offsets v - u (cells)."""
    u = torch.arange(n + 1, dtype=torch.float64, device=device)
    v = torch.arange(-pad, n + pad, dtype=torch.float64, device=device) + 0.5
    off = v[None, :] - u[:, None]
    return torch.exp(-0.5 * (off / sigma) ** 2), off


def _reflect(a: torch.Tensor, pad: int) -> torch.Tensor:
    """Reflect ``pad`` cells on both sides of both axes, edge cell included (d c b a | a b c d |
    d c b a), repeating the mirrored copies when ``pad`` exceeds the axis (tiny test grids)."""
    for axis in (0, 1):
        n = a.shape[axis]
        k = torch.arange(-pad, n + pad, device=a.device) % (2 * n)
        a = a.index_select(axis, torch.where(k < n, k, 2 * n - 1 - k))
    return a


def displacement(saliency: torch.Tensor, alpha: float, kernel: float) -> torch.Tensor:
    """(2, gh + 1, gw + 1) displacement x(u) - u of the control points before the backoff,
    normalised ([x, y]); zeros when ``alpha`` is 0 or the saliency is empty."""
    if saliency.dim() != 2:
        raise ValueError(f"saliency must be (gh, gw), got shape {tuple(saliency.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if kernel <= 0:
        raise ValueError(f"kernel must be > 0 cells, got {kernel}")
    s = saliency.detach().to(torch.float64)
    if not torch.isfinite(s).all() or bool((s < 0).any()):
        raise ValueError("a saliency must be finite and non-negative")
    gh, gw = s.shape
    dev = s.device
    mean = float(s.mean())
    if alpha == 0.0 or mean <= 0.0:
        return torch.zeros(2, gh + 1, gw + 1, dtype=torch.float64, device=dev)
    a = (1.0 - alpha) + alpha * s / mean
    pad = int(math.ceil(3 * kernel))
    ky, oy = _kernel_matrix(gh, kernel, pad, dev)
    kx, ox = _kernel_matrix(gw, kernel, pad, dev)

    def disp(att: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ap = _reflect(att, pad)
        den = ky @ ap @ kx.t()
        nx = ky @ ap @ (kx * ox).t()
        ny = (ky * oy) @ ap @ kx.t()
        return nx / den, ny / den

    dx, dy = disp(a)
    bx, by = disp(torch.ones_like(a))
    dx, dy = dx - bx, dy - by
    dx[:, 0] = 0.0
    dx[:, -1] = 0.0
    dy[0, :] = 0.0
    dy[-1, :] = 0.0
    return torch.stack([dx / gw, dy / gh])


def corner_jacobians(ctrl: torch.Tensor) -> torch.Tensor:
    """(4, gh, gw, 2, 2) Jacobian dx/du (normalised) of the bilinear map at the four corners of
    every control cell; [..., r, c] = d x_r / d u_c with r, c in (x, y)."""
    gh, gw = ctrl.shape[1] - 1, ctrl.shape[2] - 1
    ex = (ctrl[:, :, 1:] - ctrl[:, :, :-1]) * gw  # (2, gh + 1, gw): d/du_x along each row of corners
    ey = (ctrl[:, 1:, :] - ctrl[:, :-1, :]) * gh  # (2, gh, gw + 1): d/du_y along each column
    out = []
    for top, left in ((0, 0), (0, 1), (1, 0), (1, 1)):
        jx = ex[:, top:top + gh, :]  # the horizontal edge on the corner's row
        jy = ey[:, :, left:left + gw]  # the vertical edge on the corner's column
        out.append(torch.stack([jx, jy], dim=-1).permute(1, 2, 0, 3))  # (gh, gw, 2 rows, 2 cols)
    return torch.stack(out)


def singular_values(j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Largest and smallest singular values and the determinant of (..., 2, 2) matrices."""
    det = j[..., 0, 0] * j[..., 1, 1] - j[..., 0, 1] * j[..., 1, 0]
    fro = (j ** 2).sum((-2, -1))
    root = (fro ** 2 - 4 * det ** 2).clamp(min=0).sqrt()
    smax = ((fro + root) / 2).clamp(min=0).sqrt()
    smin = ((fro - root) / 2).clamp(min=0).sqrt()
    return smax, smin, det


def map_ok(ctrl: torch.Tensor, scale: float, d_min: float = D_MIN, eps: float = 1e-9) -> bool:
    """The clip: at every control-cell corner, det J > 0 and the singular values of J lie in
    [``scale``, 1 / ``d_min``]."""
    smax, smin, det = singular_values(corner_jacobians(ctrl))
    return bool((det > 0).all() and (smin >= scale - eps).all() and (smax <= 1.0 / d_min + eps).all())


def make_map2d(saliency: torch.Tensor, alpha: float, scale: float, kernel: float, d_min: float = D_MIN,
               steps: int = BACKOFF_STEPS) -> Map2D:
    """The local warp of a (gh, gw) saliency (module docstring), with the largest backoff t in
    [0, 1] (bisection, ``steps`` steps) that keeps the clip."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    if not 0.0 < d_min <= 1.0:
        raise ValueError(f"d_min must be in (0, 1], got {d_min}")
    gh, gw = saliency.shape[-2:]
    ident = _identity_ctrl(gh, gw, saliency.device)
    disp = displacement(saliency, alpha, kernel)
    if not bool(disp.any()):
        return Map2D(ident, 1.0)
    if map_ok(ident + disp, scale, d_min):
        return Map2D(ident + disp, 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        if map_ok(ident + mid * disp, scale, d_min):
            lo = mid
        else:
            hi = mid
    return Map2D(ident + lo * disp, lo)


def _cell_coords(u: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    p = u * n
    i = p.floor().clamp(0, n - 1)
    return i.long(), p - i


def eval_map(m: Map2D, uy: torch.Tensor, ux: torch.Tensor,
             jacobian: bool = False) -> Tuple[torch.Tensor, ...]:
    """The bilinear map at normalised output positions (float64, any equal shapes): (X, Y), and
    with ``jacobian`` also (dX/dux, dX/duy, dY/dux, dY/duy)."""
    c = m.ctrl.to(uy.device)
    gh, gw = m.cells
    i, b = _cell_coords(uy, gh)
    j, a = _cell_coords(ux, gw)
    c00, c01, c10, c11 = c[:, i, j], c[:, i, j + 1], c[:, i + 1, j], c[:, i + 1, j + 1]
    val = (1 - a) * (1 - b) * c00 + a * (1 - b) * c01 + (1 - a) * b * c10 + a * b * c11
    if not jacobian:
        return val[0], val[1]
    da = ((1 - b) * (c01 - c00) + b * (c11 - c10)) * gw
    db = ((1 - a) * (c10 - c00) + a * (c11 - c01)) * gh
    return val[0], val[1], da[0], db[0], da[1], db[1]


def inverse_map(m: Map2D, y: torch.Tensor, x: torch.Tensor, iters: int = NEWTON_ITERS,
                tol: float = NEWTON_TOL) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Output position (uy, ux) whose source position is (y, x) (normalised, float64), by
    Newton's method on the bilinear map, started at u = x; and the largest residual |x(u) - x|.
    Raises if it does not reach ``tol`` * 1000 (a map that keeps the clip does not fold)."""
    y = y.to(torch.float64)
    x = x.to(torch.float64)
    uy, ux = y.clone(), x.clone()
    res = float("inf")
    for _ in range(int(iters)):
        fx, fy, xx, xy, yx, yy = eval_map(m, uy, ux, jacobian=True)
        rx, ry = fx - x, fy - y
        res = float(torch.maximum(rx.abs(), ry.abs()).max()) if rx.numel() else 0.0
        if res < tol:
            break
        det = xx * yy - xy * yx
        dux = (yy * rx - xy * ry) / det
        duy = (-yx * rx + xx * ry) / det
        ux = (ux - dux).clamp(0, 1)
        uy = (uy - duy).clamp(0, 1)
    else:
        fx, fy = eval_map(m, uy, ux)
        res = float(torch.maximum((fx - x).abs(), (fy - y).abs()).max())
    if res > tol * 1000:
        raise RuntimeError(f"inverse_map did not converge (largest residual {res:.3g})")
    return uy, ux, res


def _centres(n: int, device) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n


def _axis_taps(centre: torch.Tensor, span: torch.Tensor, n_src: int,
               taps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(..., K) source indices and normalised float64 triangle weights along one axis, per
    output pixel: :func:`warp.grid._axis_filter`'s formula with per-pixel centre and span."""
    k = int(torch.ceil(2 * span.max()).item()) + 2 if taps is None else int(taps)
    lo = torch.floor(centre - span + 0.5).clamp(min=0)
    hi = torch.floor(centre + span + 0.5).clamp(max=n_src)
    idx = lo[..., None] + torch.arange(k, dtype=torch.float64, device=centre.device)
    w = (1 - ((idx + 0.5 - centre[..., None]) / span[..., None]).abs()).clamp(min=0)
    w = torch.where(idx < hi[..., None], w, torch.zeros_like(w))
    return idx.clamp(max=n_src - 1).long(), w / w.sum(-1, keepdim=True)


def resample_aa(frame: torch.Tensor, cy: torch.Tensor, cx: torch.Tensor, sy: torch.Tensor,
                sx: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H', W'): output pixel (i, j) is the triangle-filtered frame at
    source pixel position (``cy``, ``cx``)[i, j] (in source pixels, edges at 0 and H or W),
    with half-widths (``sy``, ``sx``)[i, j] source pixels (clamped to at least 1); a product of
    two 1-D triangles, each normalised as torch's antialiased bilinear filter is."""
    h, w = frame.shape[-2:]
    iy, wy = _axis_taps(cy.to(torch.float64), sy.to(torch.float64).clamp(min=1.0), h)
    ix, wx = _axis_taps(cx.to(torch.float64), sx.to(torch.float64).clamp(min=1.0), w)
    flat = frame.reshape(frame.shape[0], frame.shape[1], h * w)
    wy, wx = wy.to(frame.dtype), wx.to(frame.dtype)
    out = None
    for a in range(iy.shape[-1]):
        row = iy[..., a] * w
        for b in range(ix.shape[-1]):
            term = flat[:, :, (row + ix[..., b]).flatten()].view(*frame.shape[:2], *cy.shape) \
                * (wy[..., a] * wx[..., b])
            out = term if out is None else out + term
    return out


def warp_frame2d(frame: torch.Tensor, m: Map2D, out_size: Tuple[int, int]) -> torch.Tensor:
    """(B, 3, H, W) frame -> (B, 3, H', W') warped image (``out_size`` = (H', W')): each output
    pixel filtered at the map's image of its centre, with the extent of the image of its four
    corners as the filter's span on each axis (:func:`resample_aa`)."""
    hh, ww = int(out_size[0]), int(out_size[1])
    h, w = frame.shape[-2:]
    dev = frame.device
    ey = torch.arange(hh + 1, dtype=torch.float64, device=dev) / hh
    ex = torch.arange(ww + 1, dtype=torch.float64, device=dev) / ww
    gy, gx = torch.meshgrid(ey, ex, indexing="ij")
    xe, ye = eval_map(m, gy, gx)
    gy, gx = torch.meshgrid(_centres(hh, dev), _centres(ww, dev), indexing="ij")
    xc, yc = eval_map(m, gy, gx)

    def extent(e: torch.Tensor) -> torch.Tensor:
        corners = torch.stack([e[:-1, :-1], e[:-1, 1:], e[1:, :-1], e[1:, 1:]])
        return corners.amax(0) - corners.amin(0)

    return resample_aa(frame, yc * h, xc * w, extent(ye) * h, extent(xe) * w)


def unwarp_logits_at(logits: torch.Tensor, uy: torch.Tensor, ux: torch.Tensor, align_corners: bool,
                     output_stride: int = 1, input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """(B, C, h, w) logits of the warped image -> (B, C, H, W): frame pixel (i, j) reads the
    logits at its output coordinate (``uy``, ``ux``)[i, j] (normalised, (H, W) each), with the
    model's stride, padding and corner convention (as :func:`warp.grid.unwarp_logits`)."""
    hl, wl = logits.shape[-2:]
    s = int(output_stride)
    in_h, in_w = (hl * s, wl * s) if input_size is None else (int(input_size[0]), int(input_size[1]))
    if hl * s < in_h or wl * s < in_w:
        raise ValueError(f"logits {(hl, wl)} at stride {s} cannot cover input {(in_h, in_w)}")
    big_h, big_w = uy.shape
    gy = _logit_positions(uy.to(torch.float64), big_h, hl, hl * s, in_h, align_corners)
    gx = _logit_positions(ux.to(torch.float64), big_w, wl, wl * s, in_w, align_corners)
    g = torch.stack([gx, gy], dim=-1).to(logits.dtype)[None].expand(logits.shape[0], -1, -1, -1)
    return F.grid_sample(logits, g, mode="bilinear", padding_mode="border", align_corners=align_corners)


def unwarp_logits2d(logits: torch.Tensor, m: Map2D, frame_size: Tuple[int, int], align_corners: bool,
                    output_stride: int = 1, input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """:func:`unwarp_logits_at` at the output coordinates of the frame's pixel centres
    (:func:`inverse_map`)."""
    h, w = int(frame_size[0]), int(frame_size[1])
    dev = logits.device
    y, x = torch.meshgrid(_centres(h, dev), _centres(w, dev), indexing="ij")
    if m.is_identity():
        uy, ux = y, x
    else:
        uy, ux, _ = inverse_map(m, y, x)
    return unwarp_logits_at(logits, uy, ux, align_corners, output_stride, input_size)


def distortion(m: Map2D) -> Dict[str, float]:
    """The spec's distortion figures of a map, over every control-cell corner: mean and 95th
    percentile of the anisotropy smax / smin of J, 5th and 95th percentiles of the local linear
    magnification 1 / sqrt(det J), and the backoff t."""
    smax, smin, det = singular_values(corner_jacobians(m.ctrl))
    aniso = (smax / smin.clamp(min=1e-12)).flatten().float()
    mag = det.clamp(min=1e-12).rsqrt().flatten().float()
    q = torch.tensor([0.05, 0.95], dtype=torch.float32, device=mag.device)
    mq = torch.quantile(mag, q)
    return {"aniso_mean": float(aniso.mean()), "aniso_p95": float(torch.quantile(aniso, 0.95)),
            "mag_p5": float(mq[0]), "mag_p95": float(mq[1]), "t": float(m.t)}
