"""Method 5, phase 1: a separable warp of the frame, set by a saliency map, and its inverse.

Spec: docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md (step 2 of "The
mechanism, per frame", and the two *(plan)* details of steps 3 and 5).

Coordinates along an axis are normalised to [0, 1] with the pixel edges at 0 and 1: pixel i
of n covers [i/n, (i+1)/n] and its centre is (i + 0.5)/n. This holds for the frame (the
source) and for the warped image (the output), whatever their sizes.

A saliency map (gh, gw) is reduced to two profiles, its mean over rows (gw values: the x
axis) and over columns (gh values: the y axis). Each profile gives a density d over the G
cells of its axis (:func:`axis_density`): output pixels per source pixel relative to the
uniform downscale, piecewise constant, mean 1, clipped to [d_min, 1/scale]. Its integral F
maps a source coordinate x to the output coordinate u = F(x), with F(0) = 0, F(1) = 1 and
slope d: where d > 1 the region gets more output pixels than the uniform downscale gives it.

* forward (:func:`forward_coords`): the source coordinate each output pixel centre samples,
  x = F^-1(u);
* inverse (:func:`inverse_coords`): the output coordinate of each source pixel centre, u = F(x).

Rows and columns are warped independently, so axis-aligned lines stay straight, the map
cannot fold, and the inverse is exact (two 1-D monotone lookups). A uniform density
(alpha = 0, or a saliency that is all zeros) gives F(x) = x: the uniform downscale.

:func:`warp_frame` resamples the frame with the forward map (``grid_sample``, bilinear) at
``supersample`` times the output size and averages ``supersample`` x ``supersample`` blocks,
because the baseline's downscale is antialiased and a plain ``grid_sample`` is not (the spec's
*(plan)* detail). With ``antialias=True`` it instead filters each axis with a triangle whose
half-width is the source span of the output pixel, which is exactly the baseline's antialiased
bilinear resize (``F.interpolate(..., antialias=True)``) for a uniform density, and keeps
antialiasing the regions a warp compresses (a proposed amendment, not the accepted spec).
:func:`unwarp_logits` samples the model's logits once, at the inverse map, with the model's
own corner convention and logit stride, so that with a uniform density it is exactly
:func:`models.utils.upsample_logits` (the baseline's upsampling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["D_MIN", "WarpGrid", "axis_density", "forward_coords", "inverse_coords", "make_grid", "warp_frame",
           "unwarp_logits"]

D_MIN = 0.5  # the spec's lower clip: no region gets less than half the uniform density
MAX_ITER = 50  # clip-and-renormalise rounds (the worst of 20 000 random profiles needs 32)
TOL = 1e-6  # the clip is "stable" when it moves no value by more than this


def axis_density(profile: torch.Tensor, alpha: float, scale: float, d_min: float = D_MIN) -> torch.Tensor:
    """Sampling density over the G cells of one axis, from a (G,) non-negative saliency profile.

    ``d = (1 - alpha) + alpha * p / mean(p)`` (all ones when ``alpha`` is 0 or ``p`` is all
    zeros), then clipped to [``d_min``, 1/``scale``] and renormalised to mean 1 until the clip
    changes nothing. 1/``scale`` is "never sample finer than the source pixels". Returns (G,)
    float32."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1] (the upper clip 1/scale must allow a mean of 1), got {scale}")
    if not 0.0 < d_min <= 1.0:
        raise ValueError(f"d_min must be in (0, 1], got {d_min}")
    p = profile.detach().to(torch.float64).flatten()
    if not torch.isfinite(p).all() or bool((p < 0).any()):
        raise ValueError("a saliency profile must be finite and non-negative")
    mean = float(p.mean())
    if alpha == 0.0 or mean <= 0.0:
        return torch.ones(p.numel(), dtype=torch.float32, device=profile.device)
    d = (1.0 - alpha) + alpha * p / mean
    for _ in range(MAX_ITER):
        clipped = d.clamp(d_min, 1.0 / scale)
        change = float((clipped - d).abs().max())
        d = clipped / clipped.mean()
        if change < TOL:
            break
    else:
        raise RuntimeError(f"the density clip did not settle in {MAX_ITER} rounds (alpha {alpha}, scale {scale})")
    return d.to(torch.float32)


def _knots(density: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The CDF's knots in float64: cell edges x_k = k/G and u_k = F(x_k), with F(1) = 1 exactly."""
    d = density.detach().to(torch.float64).flatten()
    g = d.numel()
    xs = torch.arange(g + 1, dtype=torch.float64, device=d.device) / g
    us = torch.cat([d.new_zeros(1), torch.cumsum(d, 0)])
    return xs, us / us[-1]


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """numpy's ``interp`` for a strictly increasing ``xp`` (float64, on ``x``'s device)."""
    i = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return f0 + (x - x0) / (x1 - x0) * (f1 - f0)


def _centres(n: int, device) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n


def _forward64(density: torch.Tensor, n_out: int) -> torch.Tensor:
    xs, us = _knots(density)
    return _interp(_centres(n_out, xs.device), us, xs)


def _inverse64(density: torch.Tensor, n_src: int) -> torch.Tensor:
    xs, us = _knots(density)
    return _interp(_centres(n_src, xs.device), xs, us)


def forward_coords(density: torch.Tensor, n_out: int) -> torch.Tensor:
    """(n_out,) source coordinate in [0, 1] sampled by each output pixel centre: x = F^-1(u)."""
    return _forward64(density, n_out).to(torch.float32)


def inverse_coords(density: torch.Tensor, n_src: int) -> torch.Tensor:
    """(n_src,) output coordinate in [0, 1] of each source pixel centre: u = F(x)."""
    return _inverse64(density, n_src).to(torch.float32)


@dataclass
class WarpGrid:
    """The two densities of a separable warp: ``dy`` over the rows' cells, ``dx`` over the columns'."""

    dy: torch.Tensor
    dx: torch.Tensor

    def is_uniform(self) -> bool:
        return bool((self.dy == 1).all()) and bool((self.dx == 1).all())


def make_grid(saliency: torch.Tensor, alpha: float, scale: float, d_min: float = D_MIN) -> WarpGrid:
    """The warp of a (gh, gw) saliency map: its mean over columns sets the rows' density, its
    mean over rows the columns'."""
    if saliency.dim() != 2:
        raise ValueError(f"saliency must be (gh, gw), got shape {tuple(saliency.shape)}")
    return WarpGrid(dy=axis_density(saliency.mean(1), alpha, scale, d_min),
                    dx=axis_density(saliency.mean(0), alpha, scale, d_min))


def _sampling_grid(gy: torch.Tensor, gx: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
    """(batch, len(gy), len(gx), 2) ``grid_sample`` grid, x first, from normalised 1-D positions."""
    yy, xx = torch.meshgrid(gy.to(dtype), gx.to(dtype), indexing="ij")
    return torch.stack([xx, yy], dim=-1)[None].expand(batch, -1, -1, -1)


def _axis_filter(density: torch.Tensor, n_src: int, n_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Antialiased resampling of one axis: (n_out, K) source indices and float64 weights.

    Output pixel j is centred on the forward map's source position and spans ``span`` source
    pixels (the forward map at its two edges; at least 1). Its weights are a triangle of
    half-width ``span`` over the source pixels whose centres fall inside it, normalised: torch's
    antialiased bilinear filter (``_compute_weights_aa``), with the scale made local."""
    dev = density.device
    xs, us = _knots(density)
    edges = _interp(torch.arange(n_out + 1, dtype=torch.float64, device=dev) / n_out, us, xs) * n_src
    centre = _interp(_centres(n_out, dev), us, xs) * n_src
    span = (edges[1:] - edges[:-1]).clamp(min=1.0)
    k = int(torch.ceil(2 * span.max()).item()) + 2
    lo = torch.floor(centre - span + 0.5).clamp(min=0)
    hi = torch.floor(centre + span + 0.5).clamp(max=n_src)
    idx = lo[:, None] + torch.arange(k, dtype=torch.float64, device=dev)
    w = (1 - ((idx + 0.5 - centre[:, None]) / span[:, None]).abs()).clamp(min=0)
    w = torch.where(idx < hi[:, None], w, torch.zeros_like(w))
    return idx.clamp(max=n_src - 1).long(), w / w.sum(1, keepdim=True)


def _filter_axis(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, axis: int) -> torch.Tensor:
    shape = [1] * x.dim()
    shape[axis] = -1
    out = x.index_select(axis, idx[:, 0]) * w[:, 0].to(x.dtype).view(shape)
    for t in range(1, idx.shape[1]):
        out += x.index_select(axis, idx[:, t]) * w[:, t].to(x.dtype).view(shape)
    return out


def warp_frame(frame: torch.Tensor, grid: WarpGrid, out_size: Tuple[int, int], supersample: int = 2,
               antialias: bool = False) -> torch.Tensor:
    """(B, 3, H, W) frame -> (B, 3, H', W') warped image, ``out_size`` = (H', W').

    Bilinear ``grid_sample`` at ``supersample`` x ``out_size`` (``align_corners=False``, border
    padding), then the mean of each ``supersample`` x ``supersample`` block. ``antialias``: the
    separable antialiased filter of :func:`_axis_filter` instead (``supersample`` is ignored)."""
    h, w = int(out_size[0]), int(out_size[1])
    dev = frame.device
    if antialias:
        iy, wy = _axis_filter(grid.dy.to(dev), frame.shape[-2], h)
        ix, wx = _axis_filter(grid.dx.to(dev), frame.shape[-1], w)
        return _filter_axis(_filter_axis(frame, iy, wy, -2), ix, wx, -1)
    s = int(supersample)
    if s < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")
    ys = _forward64(grid.dy.to(dev), s * h)
    xs = _forward64(grid.dx.to(dev), s * w)
    g = _sampling_grid(2 * ys - 1, 2 * xs - 1, frame.shape[0], frame.dtype)
    out = F.grid_sample(frame, g, mode="bilinear", padding_mode="border", align_corners=False)
    return F.avg_pool2d(out, s) if s > 1 else out


def _logit_positions(u: torch.Tensor, n_frame: int, n_logits: int, extent: int, n_input: int,
                     align_corners: bool) -> torch.Tensor:
    """``grid_sample`` positions (normalised with ``align_corners``) in a logit axis of
    ``n_logits`` cells, for frame pixels whose warped coordinates are ``u`` (float64).

    As :func:`models.utils.upsample_logits`: the logits cover ``extent`` input pixels (the
    input of ``n_input`` pixels padded to whole strides), brought in one step to ``T``
    output pixels with ``T = round(extent * n_frame / n_input)``, of which the first
    ``n_frame`` are the frame. A frame pixel whose warped coordinate is u stands at position
    ``v = u * n_frame - 0.5`` of those T pixels (v = i for pixel i of a uniform warp)."""
    t = int(round(extent * n_frame / n_input))
    v = u * n_frame - 0.5
    if align_corners:  # logit j sits at output pixel j * (T - 1) / (n_logits - 1)
        return 2 * v / max(t - 1, 1) - 1
    return 2 * (v + 0.5) / t - 1  # logit j covers output pixels [j T / n, (j + 1) T / n)


def unwarp_logits(logits: torch.Tensor, grid: WarpGrid, frame_size: Tuple[int, int], align_corners: bool,
                  output_stride: int = 1, input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """(B, C, h, w) logits of the warped image -> (B, C, H, W) on the frame, in one bilinear step.

    The logits cover the warped input of ``input_size`` (default: ``h * output_stride`` x
    ``w * output_stride``, i.e. no padding) padded to whole strides, as the
    :class:`models.interfaces.SegmentationBackbone` contract says; ``align_corners`` is the
    model's own convention. With a uniform grid this is :func:`models.utils.upsample_logits`."""
    hl, wl = logits.shape[-2:]
    s = int(output_stride)
    in_h, in_w = (hl * s, wl * s) if input_size is None else (int(input_size[0]), int(input_size[1]))
    if hl * s < in_h or wl * s < in_w:
        raise ValueError(f"logits {(hl, wl)} at stride {s} cannot cover input {(in_h, in_w)}")
    big_h, big_w = int(frame_size[0]), int(frame_size[1])
    dev = logits.device
    gy = _logit_positions(_inverse64(grid.dy.to(dev), big_h), big_h, hl, hl * s, in_h, align_corners)
    gx = _logit_positions(_inverse64(grid.dx.to(dev), big_w), big_w, wl, wl * s, in_w, align_corners)
    g = _sampling_grid(gy, gx, logits.shape[0], logits.dtype)
    return F.grid_sample(logits, g, mode="bilinear", padding_mode="border", align_corners=align_corners)
