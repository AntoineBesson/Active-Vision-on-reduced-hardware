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
from foveation.ops import GlimpseGrid, gather_crops, paste_labels
from foveation.pipeline import glimpse_logits
from models.utils import upsample_logits

from .motion import MOTION_FACTOR, estimate_homography, motion_map, moving_cells, to_gray_small, warp

__all__ = ["MotionReuseConfig", "MotionReuseMethod", "FRAME"]

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
        self._full = FullFrameMethod(self.backbone, scale=cfg.full_scale, precision="fp32")  # resize rule only
        self._ctx = FullFrameMethod(self.ctx_backbone, scale=cfg.ctx_scale or 1.0, precision="fp32")
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
    def _pass(self, which: FullFrameMethod, model: torch.nn.Module, frame: torch.Tensor, small: Tuple[int, int],
              ledger: CostLedger, component: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-frame-style pass: (H, W) labels (as FullFrameMethod) and (small) uint8 confidence."""
        h, w = frame.shape[-2:]
        x = which._resize(frame).to(self.dtype)
        logits = model(x)
        labels = upsample_logits(logits.float(), (h, w), model.output_stride, model.align_corners,
                                 input_size=x.shape[-2:]).argmax(1)[0]
        flops = self._count((component, tuple(x.shape)), lambda: model(x))
        ledger.add(component, flops=flops, pixels=x.shape[-2] * x.shape[-1])
        return labels, _confidence(logits, small, model, tuple(x.shape[-2:]))

    def _glimpses(self, frame: torch.Tensor, grid: GlimpseGrid, idx: torch.Tensor,
                  ledger: CostLedger) -> Tuple[torch.Tensor, torch.Tensor]:
        """(k, S, S) labels and (k, n/f, n/f) uint8 confidence of the inner squares."""
        crops = gather_crops(grid.pad_frame(frame), grid, idx).to(self.dtype)
        up = glimpse_logits(self.backbone, crops).float()
        m, n, s = grid.margin, grid.inner, grid.inner // MOTION_FACTOR
        inner = F.interpolate(up[:, :, m:m + n, m:m + n], size=(s, s), mode="bilinear", align_corners=False)
        conf = (torch.softmax(inner, 1).amax(1) * 255).round().to(torch.uint8)
        k = int(idx.numel())
        ledger.add("glimpses", flops=self._count(("glimpses", k), lambda: self.backbone(crops)),
                   pixels=k * grid.size * grid.size)
        return up.argmax(1), conf

    # -- one frame ----------------------------------------------------------- #
    def predict(self, frame: torch.Tensor, t: int, ledger: CostLedger) -> torch.Tensor:
        grid = self._check(frame)
        h, w = frame.shape[-2:]
        small = (h // MOTION_FACTOR, w // MOTION_FACTOR)
        with torch.no_grad():
            gray = to_gray_small(frame)
            if self.labels is None:
                self.labels, self.conf = self._pass(self._full, self.backbone, frame, small, ledger, "full")
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
                self.labels, self.conf = self._pass(self._full, self.backbone, frame, small, ledger, "full")
                step.update(fallback=True, moved_cells=grid.num_cells)
                self.last_step = step
                return self.labels.clone()
            labels = warp(self.labels[None, None].float(), H, "nearest")[0][0, 0].round().long()
            conf = warp(self.conf[None, None].float(), H, "nearest", frame_size=(h, w))[0][0, 0].round().to(torch.uint8)
            ledger.add("warp")
            if self.cfg.ctx_scale is not None:
                c_labels, c_conf = self._pass(self._ctx, self.ctx_backbone, frame, small, ledger, "context")
                take = c_conf >= conf  # at the motion resolution; uniform over each block of f x f pixels
                take_full = take.repeat_interleave(MOTION_FACTOR, 0).repeat_interleave(MOTION_FACTOR, 1)
                labels = torch.where(take_full, c_labels, labels)
                conf = torch.where(take, c_conf, conf)
                ledger.add("fusion")
            if moved:
                idx = torch.nonzero(moving).flatten().to(frame.device)
                g_labels, g_conf = self._glimpses(frame, grid, idx, ledger)
                canvas = grid.to_canvas(labels)
                paste_labels(canvas, g_labels, grid, idx)
                labels = canvas[:h, :w]
                s = grid.inner // MOTION_FACTOR
                cgrid = GlimpseGrid(small, s + 2, 1)  # same cells at the motion resolution (inner s)
                ccanvas = cgrid.to_canvas(conf)
                pad = torch.zeros(g_conf.shape[0], s + 2, s + 2, dtype=torch.uint8, device=conf.device)
                pad[:, 1:1 + s, 1:1 + s] = g_conf
                paste_labels(ccanvas, pad, cgrid, idx)
                conf = ccanvas[:small[0], :small[1]]
            self.labels, self.conf = labels.contiguous(), conf.contiguous()
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
