"""Method 5, phase 1: the three oracle saliencies, from the teacher's prediction.

Spec: docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md ("The oracle saliencies").
The teacher is the scale-1 full-frame prediction of the same model. Each saliency is a pixel
mask at full resolution, average-pooled to cells of ``cell`` pixels (16: a 64 x 128 grid on a
1024 x 2048 frame) and smoothed by a Gaussian of ``sigma`` cells:

* ``E`` (error): pixels where the uniform pass at the configuration's own scale disagrees with
  the teacher;
* ``B`` (boundaries): teacher pixels whose label differs from the right or lower neighbour,
  marked on both pixels of the pair (so: differs from a 4-neighbour);
* ``S`` (small classes): teacher pixels of pole, traffic light, traffic sign, person, rider,
  motorcycle, bicycle.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["SMALL_CLASSES", "SAL_CELL", "KINDS", "pixel_mask", "smooth", "saliency"]

SMALL_CLASSES = (5, 6, 7, 11, 12, 17, 18)  # train ids: pole, traffic light, traffic sign, person, rider, motorcycle, bicycle
SAL_CELL = 16  # px per saliency cell: 64 x 128 cells on a 1024 x 2048 frame
KINDS = ("E", "B", "S")


def pixel_mask(kind: str, teacher: torch.Tensor, uniform: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(H, W) bool mask of saliency ``kind`` from the (H, W) teacher labels (and, for ``E``, the
    uniform pass's labels at the same frame)."""
    if kind == "E":
        if uniform is None:
            raise ValueError("saliency E needs the uniform pass's labels")
        if uniform.shape != teacher.shape:
            raise ValueError(f"uniform {tuple(uniform.shape)} and teacher {tuple(teacher.shape)} differ in shape")
        return uniform != teacher
    if kind == "B":
        m = torch.zeros_like(teacher, dtype=torch.bool)
        dx = teacher[:, 1:] != teacher[:, :-1]
        dy = teacher[1:, :] != teacher[:-1, :]
        m[:, 1:] |= dx
        m[:, :-1] |= dx
        m[1:, :] |= dy
        m[:-1, :] |= dy
        return m
    if kind == "S":
        return torch.isin(teacher, torch.tensor(SMALL_CLASSES, device=teacher.device, dtype=teacher.dtype))
    raise ValueError(f"unknown saliency {kind!r}, expected one of {KINDS}")


def _reflect_pad(x: torch.Tensor, r: int, axis: int) -> torch.Tensor:
    """Pad ``r`` cells on both ends of ``axis`` with the edge cells mirrored, edge included
    (d c b a | a b c d | d c b a: scipy.ndimage's "reflect"). Unlike ``F.pad(mode="reflect")``,
    which leaves the edge out, a symmetric kernel then keeps the map's total exactly."""
    n = x.shape[axis]
    return torch.cat([x.narrow(axis, 0, r).flip(axis), x, x.narrow(axis, n - r, r).flip(axis)], dim=axis)


def smooth(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian of ``sigma`` cells on a (gh, gw) map: radius ceil(3 sigma), reflect
    padding (:func:`_reflect_pad`), kernel normalised to 1, so the total is kept. The radius is
    capped at the map's side minus one (only maps smaller than the kernel, such as the test
    fixture's, are affected: a longer kernel truncated there would favour the reflected copies).
    ``sigma`` 0: no smoothing."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return grid
    x = grid[None, None].float()
    radius = math.ceil(3 * sigma)
    for axis in (-2, -1):
        r = min(radius, x.shape[axis] - 1)
        if r <= 0:
            continue
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float64, device=x.device) / sigma) ** 2)
        k = (k / k.sum()).to(x.dtype)
        shape = (1, 1, -1, 1) if axis == -2 else (1, 1, 1, -1)
        x = F.conv2d(_reflect_pad(x, r, axis), k.view(shape))
    return x[0, 0]


def saliency(kind: str, teacher: torch.Tensor, uniform: Optional[torch.Tensor], sigma: float,
             cell: int = SAL_CELL) -> torch.Tensor:
    """(H / cell, W / cell) float saliency of ``kind`` ("E", "B" or "S"): the share of the cell's
    pixels in the mask, smoothed by a Gaussian of ``sigma`` cells."""
    if kind not in KINDS:
        raise ValueError(f"unknown saliency {kind!r}, expected one of {KINDS}")
    h, w = teacher.shape[-2:]
    if h % cell or w % cell:
        raise ValueError(f"frame {h}x{w} is not a whole number of {cell}-px saliency cells")
    mask = pixel_mask(kind, teacher, uniform).float()
    return smooth(F.avg_pool2d(mask[None, None], cell)[0, 0], sigma)
