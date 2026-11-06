"""Method 3: keep the previous frame's prediction, recompute the cells where something moved.

Per clip (spec: docs/superpowers/specs/2026-11-02-method3-motion-reuse-design.md):

* frame 0: a full-frame pass at ``full_scale``; its labels and confidence are the memory;
* later frames: a homography H aligns the previous frame (:mod:`reuse.motion`), the
  aligned difference marks the moving cells of the glimpse grid. If more than ``tau``
  of the cells move, a full pass replaces the memory. Otherwise the memory is warped
  by H and the moving cells are recomputed as full-resolution glimpses (Method 2's
  replace rule: a cell's inner square comes from its glimpse).
* 3b (``ctx_scale`` set): every non-fallback frame also gets a low-res context pass;
  in the cells that did not move, each pixel keeps whichever of the warped memory and
  the context is more confident (the context on ties), and so does the memory.

Memory: labels (H, W) at full resolution, confidence (H/f, W/f) uint8 at the motion
resolution (round(255 * max prob) of the logits resized bilinearly to that size).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from eval.cost import CostLedger, count_flops
from eval.harness import PRECISIONS, FullFrameMethod
from foveation.ops import GlimpseGrid, gather_crops
from foveation.pipeline import glimpse_logits
from models.utils import upsample_logits

from .motion import MOTION_FACTOR, estimate_homography, motion_map, moving_cells, to_gray_small, warp

__all__ = ["MotionReuseConfig", "MotionReuseMethod", "FRAME", "full_pass", "glimpse_pass", "paste_cells", "fuse",
           "warp_memory", "advance"]

FRAME = (1024, 2048)  # the frame size the T4 latency tables were measured at


@dataclass(frozen=True)
class MotionReuseConfig:
    full_scale: float
    glimpse_size: int
    theta: int
    rho: float
    tau: float
    ctx_scale: Optional[float] = None  # None: 3a (reuse only); a scale: 3b (reuse + low-res pass)
    margin: int = 32
    homography: str = "orb"

    def name(self) -> str:
        variant = "3a" if self.ctx_scale is None else f"3b-c{self.ctx_scale:g}"
        return f"{variant}/f{self.full_scale:g}/g{self.glimpse_size}/t{self.theta}/r{self.rho:g}/u{self.tau:g}"


def _confidence(logits: torch.Tensor, size: Tuple[int, int], backbone: torch.nn.Module,
                input_size: Tuple[int, int]) -> torch.Tensor:
    """(1, C, h, w) logits -> (size) uint8 max probability, logits resized bilinearly to ``size``."""
    up = upsample_logits(logits.float(), size, backbone.output_stride, backbone.align_corners, input_size=input_size)
    return (torch.softmax(up, 1).amax(1)[0] * 255).round().to(torch.uint8)


def full_pass(model: torch.nn.Module, frame: torch.Tensor, scale: float, small: Tuple[int, int],
              dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-frame-style pass at ``scale``: (H, W) labels exactly as :class:`FullFrameMethod`, the
    (small) uint8 confidence, and the network input (for FLOP counting)."""
    h, w = frame.shape[-2:]
    x = FullFrameMethod(model, scale=scale)._resize(frame).to(dtype)
    logits = model(x)
    labels = upsample_logits(logits.float(), (h, w), model.output_stride, model.align_corners,
                             input_size=x.shape[-2:]).argmax(1)[0]
    return labels, _confidence(logits, small, model, tuple(x.shape[-2:])), x


def glimpse_pass(model: torch.nn.Module, frame: torch.Tensor, grid: GlimpseGrid, idx: torch.Tensor,
                 dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Glimpses of the ``idx`` cells: (k, n, n) inner-square labels, (k, n/f, n/f) uint8 confidence, crops."""
    crops = gather_crops(grid.pad_frame(frame), grid, idx).to(dtype)
    up = glimpse_logits(model, crops).float()
    m, n, s = grid.margin, grid.inner, grid.inner // MOTION_FACTOR
    inner = up[:, :, m:m + n, m:m + n]
    small = F.interpolate(inner, size=(s, s), mode="bilinear", align_corners=False)
    conf = (torch.softmax(small, 1).amax(1) * 255).round().to(torch.uint8)
    return inner.argmax(1), conf, crops


def grid_canvas(image: torch.Tensor, cell: int, cols: int) -> torch.Tensor:
    """(..., h, w) image zero-padded on the right and bottom to whole cells of a ``cols``-column grid."""
    h, w = image.shape[-2:]
    return F.pad(image, (0, cols * cell - w, 0, -(-h // cell) * cell - h))


def cell_index(cell: int, cols: int, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row and column indices, each (k, cell, cell), of the pixels of the cells ``idx``."""
    top, left = (idx // cols) * cell, (idx % cols) * cell
    ar = torch.arange(cell, device=idx.device)
    return ((top[:, None] + ar)[:, :, None].expand(-1, cell, cell),
            (left[:, None] + ar)[:, None, :].expand(-1, cell, cell))


def paste_cells(image: torch.Tensor, blocks: torch.Tensor, cell: int, cols: int, idx: torch.Tensor) -> torch.Tensor:
    """Write (k, cell, cell) blocks into the cells ``idx`` of an (..., h, w) image (grid of ``cols``
    columns, cells of ``cell`` px from the top-left), the same blocks in every image of a batch;
    returns a new tensor of the image's shape."""
    h, w = image.shape[-2:]
    canvas = grid_canvas(image, cell, cols)
    r, c = cell_index(cell, cols, idx)
    canvas[..., r, c] = blocks.to(canvas.dtype)
    return canvas[..., :h, :w].contiguous()


def fuse(labels: torch.Tensor, conf: torch.Tensor, c_labels: torch.Tensor,
         c_conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """3b: each pixel keeps the more confident of the memory and the context (the context on ties).
    Compared at the motion resolution, so uniform over each block of f x f pixels."""
    take = c_conf >= conf
    take_full = take.repeat_interleave(MOTION_FACTOR, -2).repeat_interleave(MOTION_FACTOR, -1)
    return torch.where(take_full, c_labels, labels), torch.where(take, c_conf, conf)


def warp_memory(labels: torch.Tensor, conf: torch.Tensor, H: torch.Tensor,
                ctx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The memory warped by H and, in 3b (``ctx`` = (labels, conf)), fused with the context."""
    h, w = labels.shape
    labels = warp(labels[None, None].float(), H, "nearest")[0][0, 0].round().long()
    conf = warp(conf[None, None].float(), H, "nearest", frame_size=(h, w))[0][0, 0].round().to(torch.uint8)
    return (labels, conf) if ctx is None else fuse(labels, conf, *ctx)


def advance(labels: torch.Tensor, conf: torch.Tensor, H: torch.Tensor, grid: GlimpseGrid, moving: torch.Tensor,
            ctx: Optional[Tuple[torch.Tensor, torch.Tensor]],
            glimpses: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """One non-fallback step: :func:`warp_memory`, then ``glimpses(idx)`` = (inner labels, inner
    conf) in the moving cells."""
    labels, conf = warp_memory(labels, conf, H, ctx)
    if bool(moving.any()):
        idx = torch.nonzero(moving).flatten().to(labels.device)
        g_labels, g_conf = glimpses(idx)
        labels = paste_cells(labels, g_labels, grid.inner, grid.cols, idx)
        conf = paste_cells(conf, g_conf, grid.inner // MOTION_FACTOR, grid.cols, idx)
    return labels.contiguous(), conf.contiguous()


class MotionReuseMethod:
    """A :class:`eval.harness.FrameMethod` that needs every frame of the clip, in order."""

    name = "motion_reuse"
    requires_all_frames = True

    def __init__(self, backbone: torch.nn.Module, cfg: MotionReuseConfig, precision: str = "fp32",
                 ctx_backbone: Optional[torch.nn.Module] = None, allow_any_frame_size: bool = False):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        self.cfg, self.precision, self.dtype = cfg, precision, PRECISIONS[precision]
        self.backbone = backbone if self.dtype == torch.float32 else copy.deepcopy(backbone).to(self.dtype)
        self.backbone.eval()
        ctx = backbone if ctx_backbone is None else ctx_backbone
        self.ctx_backbone = ctx if self.dtype == torch.float32 else copy.deepcopy(ctx).to(self.dtype)
        self.allow_any_frame_size = allow_any_frame_size
        self._flops: Dict[Any, Optional[int]] = {}
        self.grid: Optional[GlimpseGrid] = None
        self.last_step: Dict[str, Any] = {}
        self.last_moving: Optional[torch.Tensor] = None  # (num_cells,) bool of the last non-first frame
        self.reset({})

    # -- state --------------------------------------------------------------- #
    def reset(self, clip: Mapping[str, Any]) -> None:
        self.labels: Optional[torch.Tensor] = None
        self.conf: Optional[torch.Tensor] = None
        self.gray: Optional[torch.Tensor] = None

    def _check(self, frame: torch.Tensor) -> GlimpseGrid:
        size = tuple(frame.shape[-2:])
        if not self.allow_any_frame_size and size != FRAME:
            raise ValueError(f"frame is {size[0]}x{size[1]}; the T4 latency tables are for {FRAME[0]}x{FRAME[1]}")
        if self.grid is None or self.grid.frame_size != size:
            self.grid = GlimpseGrid(size, self.cfg.glimpse_size, self.cfg.margin)
            if size[0] % MOTION_FACTOR or size[1] % MOTION_FACTOR or self.grid.inner % MOTION_FACTOR:
                raise ValueError(f"frame {size} and cell {self.grid.inner} px must be multiples of {MOTION_FACTOR}")
        return self.grid

    def _count(self, key: Any, fn: Callable[[], Any]) -> int:
        if key not in self._flops:
            self._flops[key] = count_flops(fn)
        return self._flops[key] or 0

    # -- passes -------------------------------------------------------------- #
    def _pass(self, model: torch.nn.Module, scale: float, frame: torch.Tensor, small: Tuple[int, int],
              ledger: CostLedger, component: str) -> Tuple[torch.Tensor, torch.Tensor]:
        labels, conf, x = full_pass(model, frame, scale, small, self.dtype)
        ledger.add(component, flops=self._count((component, tuple(x.shape)), lambda: model(x)),
                   pixels=x.shape[-2] * x.shape[-1])
        return labels, conf

    def _glimpses(self, frame: torch.Tensor, grid: GlimpseGrid, idx: torch.Tensor,
                  ledger: CostLedger) -> Tuple[torch.Tensor, torch.Tensor]:
        labels, conf, crops = glimpse_pass(self.backbone, frame, grid, idx, self.dtype)
        k = int(idx.numel())
        ledger.add("glimpses", flops=self._count(("glimpses", k), lambda: self.backbone(crops)),
                   pixels=k * grid.size * grid.size)
        return labels, conf

    # -- one frame ----------------------------------------------------------- #
    def predict(self, frame: torch.Tensor, t: int, ledger: CostLedger) -> torch.Tensor:
        grid = self._check(frame)
        h, w = frame.shape[-2:]
        small = (h // MOTION_FACTOR, w // MOTION_FACTOR)
        with torch.no_grad():
            gray = to_gray_small(frame)
            if self.labels is None:
                self.labels, self.conf = self._pass(self.backbone, self.cfg.full_scale, frame, small, ledger, "full")
                self.gray = gray
                self.last_step = {"fallback": False, "moved_cells": grid.num_cells, "num_cells": grid.num_cells,
                                  "inliers": 0, "first": True}
                return self.labels.clone()
            H, inliers = estimate_homography(self.gray, gray, method=self.cfg.homography)
            ledger.add("homography")
            motion = motion_map(self.gray, gray, H)
            moving = moving_cells(motion, grid, self.cfg.theta, self.cfg.rho)
            ledger.add("motion")
            self.gray = gray
            self.last_moving = moving
            moved = int(moving.sum())
            step = {"fallback": False, "moved_cells": moved, "num_cells": grid.num_cells, "inliers": inliers,
                    "first": False}
            if moved / grid.num_cells > self.cfg.tau:
                self.labels, self.conf = self._pass(self.backbone, self.cfg.full_scale, frame, small, ledger, "full")
                step.update(fallback=True, moved_cells=grid.num_cells)
                self.last_step = step
                return self.labels.clone()
            ctx = None
            if self.cfg.ctx_scale is not None:
                ctx = self._pass(self.ctx_backbone, self.cfg.ctx_scale, frame, small, ledger, "context")
            self.labels, self.conf = advance(self.labels, self.conf, H, grid, moving, ctx,
                                             lambda idx: self._glimpses(frame, grid, idx, ledger))
            ledger.add("warp")
            self.last_step = step
            return self.labels.clone()

    # -- reporting ----------------------------------------------------------- #
    def latency_fns(self, frame_size: Tuple[int, int], device: torch.device) -> Dict[str, Callable[[], Any]]:
        """``end_to_end``: one non-first frame (memory primed with a first frame)."""
        gen = torch.Generator().manual_seed(0)
        frames = [torch.rand(1, 3, *frame_size, generator=gen).to(device) for _ in range(2)]
        ledger = CostLedger()

        def step():
            self.reset({})
            ledger.begin_frame(frame_size[0] * frame_size[1])
            self.predict(frames[0], 0, ledger)
            return self.predict(frames[1], 1, ledger)

        return {"end_to_end": step}

    def resident_bytes(self) -> int:
        models = {id(self.backbone): self.backbone, id(self.ctx_backbone): self.ctx_backbone}.values()
        return int(sum(t.numel() * t.element_size() for m in models for t in list(m.parameters()) + list(m.buffers())))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "config": asdict(self.cfg), "config_name": self.cfg.name(),
                "precision": self.precision}
