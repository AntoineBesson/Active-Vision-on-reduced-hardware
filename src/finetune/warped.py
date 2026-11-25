"""Method 11: the differentiable path frame -> (warp) -> network -> (unwarp) -> full-size logits.

Spec: docs/superpowers/specs/2026-11-25-method11-warp-finetune-design.md ("What is trained").

It is Method 5's :class:`warp.policy.WarpStep` for a fixed warp (a prior, or the uniform grid),
on a batch, with gradients: the same resampling (``r2``: two matrix products with the
antialiased filter) and the same unwarp (``u1``: ``grid_sample`` of the logits at the inverse
map), so that the loss is computed on exactly the logits the evaluation takes the argmax of.
Both operands are fixed for a given frame size and are built once.

A horizontal flip of the frame is matched by flipping the prior: the warp then follows the
content, and the flipped sample sees the mirror image of the unflipped one's warped input.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from warp.grid import make_grid
from warp.oracle import SAL_CELL, smooth
from warp.policy import StepConfig, _Resampler, _Unwarper, _cells, warped_size

__all__ = ["WarpedForward", "boundary_counts"]


class WarpedForward(nn.Module):
    """``(frames (B, 3, H, W) in [0, 1], flip) -> (B, C, H, W)`` float32 logits on the frame grid.

    ``cfg`` is a fixed configuration: uniform, or ``P`` with its ``prior`` (gh, gw) map (the error
    prior of Method 5, or Method 11's boundary prior). The network runs on the resampled batch;
    under autocast it runs in half precision, and the unwarp is done in float32."""

    def __init__(self, model: nn.Module, cfg: StepConfig, prior: Optional[torch.Tensor] = None,
                 cell: int = SAL_CELL):
        super().__init__()
        if not cfg.fixed:
            raise ValueError(f"{cfg.name()}: only a fixed warp (P or uniform) can be fine-tuned")
        if (cfg.saliency == "P") != (prior is not None):
            raise ValueError("saliency P needs its prior, and only P takes one")
        self.model, self.cfg, self.cell = model, cfg, int(cell)
        self.register_buffer("prior", None if prior is None else prior.detach().float().clone(), persistent=False)
        self._ops: Dict[Tuple[Any, ...], Any] = {}

    def grid(self, size: Tuple[int, int], flip: bool = False):
        cfg = self.cfg
        if cfg.uniform:
            return make_grid(torch.zeros(_cells(size, self.cell)), 0.0, cfg.scale)
        gh, gw = _cells(size, self.cell)
        if tuple(self.prior.shape) != (gh, gw):
            raise ValueError(f"prior {tuple(self.prior.shape)} does not fit a {gh}x{gw} saliency grid")
        src = self.prior.flip(-1) if flip else self.prior
        return make_grid(smooth(src.cpu(), cfg.sigma), cfg.alpha, cfg.scale, d_max=cfg.d_max, rounds=cfg.rounds)

    def _resampler(self, size, flip, device):
        key = ("res", size, bool(flip and not self.cfg.uniform), str(device))
        if key not in self._ops:
            g = self.grid(size, flip)
            out = warped_size(size, self.cfg.scale)
            self._ops[key] = (g, out, _Resampler(self.cfg.resample, g, size, out, self.cfg.tap_count(),
                                                 torch.float32, device))
        return self._ops[key]

    def _unwarper(self, size, flip, logit_size, out, g, device):
        key = ("unw", size, bool(flip and not self.cfg.uniform), logit_size, str(device))
        if key not in self._ops:
            m = self.model
            self._ops[key] = _Unwarper(self.cfg.unwarp, g, size, logit_size, out, m.output_stride, m.align_corners,
                                       torch.float32, device)
        return self._ops[key]

    def forward(self, frames: torch.Tensor, flip: bool = False) -> torch.Tensor:
        if frames.dim() != 4:
            raise ValueError(f"frames must be (B, 3, H, W), got {tuple(frames.shape)}")
        size = (int(frames.shape[-2]), int(frames.shape[-1]))
        g, out, res = self._resampler(size, flip, frames.device)
        x = res(frames.float())
        logits = self.model(x)
        unw = self._unwarper(size, flip, tuple(int(v) for v in logits.shape[-2:]), out, g, frames.device)
        return unw(logits.float())


def boundary_counts(labels: torch.Tensor, cell: int = SAL_CELL, ignore: int = 255) -> torch.Tensor:
    """(gh, gw) int64 count of boundary pixels per ``cell`` x ``cell`` cell of one (H, W) train-id
    map: a pixel is on a boundary when a 4-neighbour has another class, pairs with an ignored
    pixel left out. Summed over the train labels and divided by cell^2 x frames, this is
    Method 11's boundary prior (LZU's "average of the ground-truth semantic boundaries")."""
    lab = labels.long()
    h, w = lab.shape
    if h % cell or w % cell:
        raise ValueError(f"label {h}x{w} is not a whole number of {cell}-px cells")
    b = torch.zeros_like(lab, dtype=torch.bool)
    for dim in (0, 1):
        a, c = (lab[1:], lab[:-1]) if dim == 0 else (lab[:, 1:], lab[:, :-1])
        edge = (a != c) & (a != ignore) & (c != ignore)
        if dim == 0:
            b[1:] |= edge
            b[:-1] |= edge
        else:
            b[:, 1:] |= edge
            b[:, :-1] |= edge
    return b.view(h // cell, cell, w // cell, cell).sum((1, 3)).to(torch.int64)
