"""Method 5, phase 2: saliencies the method can compute, and one frame of the method.

Spec: docs/superpowers/specs/2026-11-16-method5-phase2-design.md (stage A's pipelines, stage B's
saliencies). Plan: docs/superpowers/plans/2026-11-17-method5-phase2.md, Task 2.

The warp of frame t comes from the method's own output at t - 1, kept as a *state*: a raw
(unsmoothed) saliency on the 64 x 128 grid of 16-px cells.

* **B**, boundaries, and **S**, small classes: phase 1's masks (:func:`warp.oracle.pixel_mask`)
  on the method's own unwarped labels, pooled to cells;
* **U**, uncertainty: 1 minus the margin between the two largest softmax scores, on the warped
  logits at their own resolution, read at the frame's cell centres through the inverse map
  (the plan's call: the unwarped logits would be 19 x 2 M values per frame);
* **P**, static prior: a fixed map (the mean phase 1 error oracle E over the train frames), so
  its warp is fixed too and its resampling weights are computed once (:func:`load_prior`).

:class:`WarpStep` runs one frame: smooth the state by sigma, densities (a fixed number of
rounds), resampling (a fixed number of taps), the frozen network, unwarp, argmax, and the raw
saliency of this output as the next state. Nothing in it reads a value back to the host, so the
same module is timed in a CUDA graph (stage A) and run for accuracy (stages B and C).

Variants (stage A): resampling r1 (the antialiased filter tap by tap, phase 1's), r2 (the same
weights as two matrix products), r3 (plain bilinear ``grid_sample``, not antialiased); unwarp u1
(``grid_sample`` on the logits, phase 1's) and u2 (two 1-D lerps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import (WarpGrid, _axis_filter, _filter_axis, _forward64, _inverse64, _lerp_axis,
                   _logit_positions, _sampling_grid, axis_matrix, filter_taps, make_grid)
from .oracle import SAL_CELL, SMALL_CLASSES, pixel_mask, smooth

__all__ = ["POLICY_KINDS", "UNIFORM", "RESAMPLES", "UNWARPS", "ROUNDS", "PRECISIONS", "StepConfig", "WarpStep",
           "raw_saliency", "margin_map", "load_prior", "prior_record", "warped_size"]

POLICY_KINDS = ("B", "S", "U", "P")
UNIFORM = "uniform"  # the alpha 0 row: the full frame through the warp and unwarp (pipeline check)
RESAMPLES = ("r1", "r2", "r3")
UNWARPS = ("u1", "u2")
ROUNDS = 50  # density rounds: phase 1's MAX_ITER, its worst case
PRECISIONS = ("fp32", "fp16")
PRIOR_SOURCE = "E oracle, phase 1 teacher"


def warped_size(frame_size: Tuple[int, int], scale: float) -> Tuple[int, int]:
    """The network input of scale ``scale``: :meth:`eval.harness.FullFrameMethod.input_size`."""
    h, w = int(frame_size[0]), int(frame_size[1])
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _cells(frame_size: Tuple[int, int], cell: int) -> Tuple[int, int]:
    h, w = int(frame_size[0]), int(frame_size[1])
    if h % cell or w % cell:
        raise ValueError(f"frame {h}x{w} is not a whole number of {cell}-px saliency cells")
    return h // cell, w // cell


def _pool_mask(mask: torch.Tensor, cell: int) -> torch.Tensor:
    """(H, W) bool -> (H / cell, W / cell) float32 share of each cell's pixels in the mask: what
    ``avg_pool2d`` gives (the counts are exact in float32), without a full-size float copy."""
    gh, gw = _cells(mask.shape, cell)
    return mask.reshape(gh, cell, gw, cell).sum((1, 3), dtype=torch.float32) / float(cell * cell)


def _small_mask(labels: torch.Tensor) -> torch.Tensor:
    """Phase 1's S mask (:func:`warp.oracle.pixel_mask`), as compares: no tensor made from host data."""
    m = labels == SMALL_CLASSES[0]
    for c in SMALL_CLASSES[1:]:
        m |= labels == c
    return m


def margin_map(logits: torch.Tensor) -> torch.Tensor:
    """(1, C, h, w) logits -> (h, w) float32 uncertainty ``1 - (p1 - p2)``, p1 >= p2 the two largest
    softmax scores: 0 for a confident pixel, 1 where two classes tie."""
    if logits.dim() != 4 or logits.shape[0] != 1 or logits.shape[1] < 2:
        raise ValueError(f"logits must be (1, C >= 2, h, w), got {tuple(logits.shape)}")
    top = torch.softmax(logits.float(), 1).topk(2, dim=1).values[0]
    return 1.0 - (top[0] - top[1])


def _sample_cells(values: torch.Tensor, grid: WarpGrid, frame_size: Tuple[int, int], cell: int,
                  align_corners: bool, output_stride: int, input_size: Tuple[int, int]) -> torch.Tensor:
    """(h, w) map at the logits' resolution -> (gh, gw), read at the frame's cell centres through the
    warp's inverse map, bilinear with border clamping and the model's convention: the value the
    unwarped map (:func:`warp.grid.unwarp_logits`) has at each cell centre."""
    hl, wl = values.shape[-2:]
    gh, gw = _cells(frame_size, cell)
    big_h, big_w = int(frame_size[0]), int(frame_size[1])
    s = int(output_stride)
    dev = values.device
    # _inverse64 at n points reads F at (i + 0.5) / n: with n = gh, the cell centres
    py = _logit_positions(_inverse64(grid.dy.to(dev), gh), big_h, hl, hl * s, int(input_size[0]), align_corners)
    px = _logit_positions(_inverse64(grid.dx.to(dev), gw), big_w, wl, wl * s, int(input_size[1]), align_corners)
    y0, y1, wy = _lerp_axis(py, hl, align_corners)
    x0, x1, wx = _lerp_axis(px, wl, align_corners)
    wy = wy.to(values.dtype)[:, None]
    wx = wx.to(values.dtype)[None, :]
    rows = values.index_select(0, y0) * (1 - wy) + values.index_select(0, y1) * wy
    return rows.index_select(1, x0) * (1 - wx) + rows.index_select(1, x1) * wx


def raw_saliency(kind: str, labels: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None,
                 grid: Optional[WarpGrid] = None, cell: int = SAL_CELL, *,
                 frame_size: Optional[Tuple[int, int]] = None, align_corners: bool = False, output_stride: int = 1,
                 input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """(H / cell, W / cell) float32 saliency of ``kind``, unsmoothed (sigma is applied when the
    next frame reads it, as phase 1 smooths after pooling).

    * B, S: ``labels`` (H, W), the method's own unwarped output;
    * U: ``logits`` (1, C, h, w) of the warped image, the ``grid`` that warped it, the
      ``frame_size`` (H, W), and the model's ``align_corners``, ``output_stride`` and network
      ``input_size`` (default ``h * stride`` x ``w * stride``);
    * P is not computed per frame: see :func:`load_prior`."""
    if kind in ("B", "S"):
        if labels is None or labels.dim() != 2:
            raise ValueError(f"saliency {kind} needs (H, W) labels")
        lab = labels if labels.dtype == torch.uint8 else labels.to(torch.uint8)  # 19 classes: 8x less to read
        mask = pixel_mask("B", lab) if kind == "B" else _small_mask(lab)
        return _pool_mask(mask, cell)
    if kind == "U":
        if logits is None or grid is None or frame_size is None:
            raise ValueError("saliency U needs the warped logits, their grid and the frame size")
        hl, wl = logits.shape[-2:]
        s = int(output_stride)
        size = (hl * s, wl * s) if input_size is None else (int(input_size[0]), int(input_size[1]))
        return _sample_cells(margin_map(logits), grid, frame_size, cell, align_corners, s, size)
    if kind == "P":
        raise ValueError("saliency P is a stored prior, not computed per frame: use load_prior")
    raise ValueError(f"unknown saliency {kind!r}, expected one of {POLICY_KINDS}")


def prior_record(model: str, scale: float, total: torch.Tensor, frames: int,
                 source: str = PRIOR_SOURCE) -> Dict[str, Any]:
    """One prior of a prior file: the sum of ``frames`` raw (unsmoothed) E saliencies of ``model`` at
    ``scale``. A prior file holds a list of these (one per scale of the model)."""
    if total.dim() != 2:
        raise ValueError(f"a prior sum must be (gh, gw), got {tuple(total.shape)}")
    if int(frames) < 1:
        raise ValueError(f"a prior needs at least one frame, got {frames}")
    return {"model": str(model), "scale": float(scale), "sum": total.detach().to("cpu", torch.float64),
            "frames": int(frames), "source": str(source)}


def load_prior(path: Union[str, Path, Sequence[Mapping[str, Any]], Mapping[str, Any]], model: str,
               scale: float) -> torch.Tensor:
    """(gh, gw) float32 static prior P of ``model`` at ``scale``: sum / frames of its record in a
    prior file (a list of :func:`prior_record`, or one record), or of records already loaded.
    Exactly one record must match."""
    data: Any = path
    if isinstance(path, (str, Path)):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
    records = [data] if isinstance(data, Mapping) else list(data)
    hits = [r for r in records if r["model"] == model and abs(float(r["scale"]) - float(scale)) < 1e-9]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} priors for {model} at scale {scale:g} (expected 1)")
    r = hits[0]
    if int(r["frames"]) < 1:
        raise ValueError(f"the prior of {model} at scale {scale:g} has no frames")
    return (r["sum"].to(torch.float64) / int(r["frames"])).to(torch.float32)


@dataclass(frozen=True)
class StepConfig:
    """One configuration of the method: input ``scale``, ``saliency`` (B, S, U, P, or ``"uniform"``
    with ``alpha`` 0), warp strength ``alpha``, smoothing ``sigma`` (cells), ``resample`` (r1, r2,
    r3), ``unwarp`` (u1, u2), density ``rounds`` and filter ``taps`` (None: :func:`filter_taps`)."""

    scale: float
    saliency: str = UNIFORM
    alpha: float = 0.0
    sigma: float = 0.0
    resample: str = "r2"
    unwarp: str = "u2"
    rounds: int = ROUNDS
    taps: Optional[int] = None
    d_max: Optional[float] = None  # Method 11: upper density clip replacing 1/scale (magnification at scale 1)

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if self.saliency not in POLICY_KINDS + (UNIFORM,):
            raise ValueError(f"saliency must be one of {POLICY_KINDS + (UNIFORM,)}, got {self.saliency!r}")
        if (self.saliency == UNIFORM) != (self.alpha == 0.0):
            raise ValueError(f"alpha 0 is the uniform row and only it (saliency {self.saliency}, alpha {self.alpha})")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.resample not in RESAMPLES or self.unwarp not in UNWARPS:
            raise ValueError(f"resample/unwarp must be in {RESAMPLES}/{UNWARPS}, got {self.resample}/{self.unwarp}")
        if int(self.rounds) < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if self.d_max is not None and (self.uniform or float(self.d_max) < 1.0):
            raise ValueError(f"d_max must be >= 1 and only on a warped row, got {self.d_max} ({self.saliency})")
        if self.taps is not None and int(self.taps) < filter_taps(self.scale):
            raise ValueError(f"{self.taps} taps would cut the filter at scale {self.scale:g} "
                             f"(needs {filter_taps(self.scale)})")

    @property
    def uniform(self) -> bool:
        return self.saliency == UNIFORM

    @property
    def fixed(self) -> bool:
        """The warp does not depend on the frame (P, uniform): computed once, no policy step."""
        return self.saliency in ("P", UNIFORM)

    def tap_count(self) -> int:
        return filter_taps(self.scale) if self.taps is None else int(self.taps)

    def name(self) -> str:
        tail = f"{self.resample}{self.unwarp}"
        if self.uniform:
            return f"5p/c{self.scale:g}/{UNIFORM}/{tail}"
        if self.rounds != ROUNDS:
            tail += f"/n{self.rounds}"
        if self.d_max is not None:
            tail += f"/m{self.d_max:g}"
        return f"5p/c{self.scale:g}/{self.saliency}/a{self.alpha:g}/s{self.sigma:g}/{tail}"


class _Resampler:
    """The resampling operands of one grid and frame size: tap indices and weights (r1), the two
    matrices (r2) or the sampling grid (r3). Built per frame, or once for a fixed warp."""

    def __init__(self, kind: str, grid: WarpGrid, frame_size: Tuple[int, int], out_size: Tuple[int, int],
                 taps: int, dtype: torch.dtype, device: torch.device):
        h, w = int(out_size[0]), int(out_size[1])
        big_h, big_w = int(frame_size[0]), int(frame_size[1])
        dy, dx = grid.dy.to(device), grid.dx.to(device)
        self.kind = kind
        if kind == "r1":
            self.iy, self.wy = _axis_filter(dy, big_h, h, taps)
            self.ix, self.wx = _axis_filter(dx, big_w, w, taps)
            self.wy, self.wx = self.wy.to(dtype), self.wx.to(dtype)
        elif kind == "r2":
            self.my = axis_matrix(dy, big_h, h, taps, dtype)
            self.mx_t = axis_matrix(dx, big_w, w, taps, dtype).t()
        elif kind == "r3":
            self.g = _sampling_grid(2 * _forward64(dy, h) - 1, 2 * _forward64(dx, w) - 1, 1, dtype)
        else:
            raise ValueError(f"resample must be one of {RESAMPLES}, got {kind!r}")

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        if self.kind == "r1":
            return _filter_axis(_filter_axis(frame, self.iy, self.wy, -2), self.ix, self.wx, -1)
        if self.kind == "r2":
            return torch.matmul(torch.matmul(self.my, frame), self.mx_t)
        g = self.g.expand(frame.shape[0], -1, -1, -1)
        return F.grid_sample(frame, g, mode="bilinear", padding_mode="border", align_corners=False)


class _Unwarper:
    """The unwarp operands of one grid, frame size and logit size: the logits' sampling grid (u1) or
    two indices and a weight per output row and column (u2)."""

    def __init__(self, kind: str, grid: WarpGrid, frame_size: Tuple[int, int], logit_size: Tuple[int, int],
                 input_size: Tuple[int, int], output_stride: int, align_corners: bool, dtype: torch.dtype,
                 device: torch.device):
        hl, wl = int(logit_size[0]), int(logit_size[1])
        s = int(output_stride)
        in_h, in_w = int(input_size[0]), int(input_size[1])
        if hl * s < in_h or wl * s < in_w:
            raise ValueError(f"logits {(hl, wl)} at stride {s} cannot cover input {(in_h, in_w)}")
        big_h, big_w = int(frame_size[0]), int(frame_size[1])
        gy = _logit_positions(_inverse64(grid.dy.to(device), big_h), big_h, hl, hl * s, in_h, align_corners)
        gx = _logit_positions(_inverse64(grid.dx.to(device), big_w), big_w, wl, wl * s, in_w, align_corners)
        self.kind, self.align = kind, align_corners
        if kind == "u1":
            self.g = _sampling_grid(gy, gx, 1, dtype)
        elif kind == "u2":
            self.y0, self.y1, wy = _lerp_axis(gy, hl, align_corners)
            self.x0, self.x1, wx = _lerp_axis(gx, wl, align_corners)
            self.wy = wy.to(dtype).view(1, 1, -1, 1)
            self.wx = wx.to(dtype).view(1, 1, 1, -1)
        else:
            raise ValueError(f"unwarp must be one of {UNWARPS}, got {kind!r}")

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.kind == "u1":
            g = self.g.expand(logits.shape[0], -1, -1, -1)
            return F.grid_sample(logits, g, mode="bilinear", padding_mode="border", align_corners=self.align)
        # rows at the logits' width first (small), then one lerp kernel at full size: fewer passes
        # over the full-size logits than four multiply-adds
        rows = torch.lerp(logits.index_select(-2, self.y0), logits.index_select(-2, self.y1), self.wy)
        return torch.lerp(rows.index_select(-1, self.x0), rows.index_select(-1, self.x1), self.wx)


class WarpStep(nn.Module):
    """One frame of Method 5 with a computable saliency: ``(frame, state) -> (labels, next_state)``.

    ``frame`` (1, 3, H, W); ``state`` (H / cell, W / cell), the raw saliency of the previous
    output (:meth:`initial_state` for frame 0); ``labels`` (H, W) int64. ``precision`` "fp32"
    runs as :func:`reuse.method.full_pass` (network input cast to float32, logits unwarped in
    float32): stages B and C. "fp16" runs as the latency floor's full frame (``final`` part: the
    frame, network and upsampling all in half precision): stage A, with a half model and frame.

    For a fixed warp (P, uniform) the grid, resampling and unwarp operands are made on the first
    call for each frame size and reused, so a captured graph replays none of it (the spec: "its
    resampling weights are precomputed and it has no policy step"). Every other configuration
    makes them each frame from ``state``, with a fixed number of density rounds and filter taps.

    :meth:`warp` and :meth:`finish` are the two halves around the network, so that a runner can
    batch the network over several configurations of one scale."""

    def __init__(self, model: nn.Module, cfg: StepConfig, prior: Optional[torch.Tensor] = None,
                 cell: int = SAL_CELL, precision: str = "fp32"):
        super().__init__()
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        if (cfg.saliency == "P") != (prior is not None):
            raise ValueError("saliency P needs its prior, and only P takes one")
        if prior is not None and prior.dim() != 2:
            raise ValueError(f"a prior must be (gh, gw), got {tuple(prior.shape)}")
        self.model, self.cfg, self.cell, self.precision = model, cfg, int(cell), precision
        self.register_buffer("prior", None if prior is None else prior.detach().float().clone(), persistent=False)
        self._fixed: Dict[Tuple[Any, ...], Any] = {}

    # -- state -------------------------------------------------------------------------------
    def initial_state(self, frame_size: Tuple[int, int], device: Optional[torch.device] = None) -> torch.Tensor:
        """Frame 0's state: the prior for P (its warp), zeros (the uniform grid) otherwise."""
        gh, gw = _cells(frame_size, self.cell)
        if self.prior is not None:
            if tuple(self.prior.shape) != (gh, gw):
                raise ValueError(f"prior {tuple(self.prior.shape)} does not fit a {gh}x{gw} saliency grid")
            return self.prior.clone() if device is None else self.prior.to(device).clone()
        return torch.zeros(gh, gw, device=device)

    # -- the two halves ------------------------------------------------------------------------
    def _grid(self, state: Optional[torch.Tensor]) -> WarpGrid:
        cfg = self.cfg
        if cfg.uniform:
            sal = state if state is not None else self.prior
            return make_grid(torch.zeros_like(sal), 0.0, cfg.scale)
        src = self.prior if cfg.saliency == "P" else state
        return make_grid(smooth(src, cfg.sigma), cfg.alpha, cfg.scale, d_max=cfg.d_max, rounds=cfg.rounds)

    def _cached(self, key: Tuple[Any, ...], build: Callable[[], Any]) -> Any:
        if not self.cfg.fixed:
            return build()
        if key not in self._fixed:
            self._fixed[key] = build()
        return self._fixed[key]

    def warp(self, frame: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """(1, 3, H, W) frame and state -> the network input and what :meth:`finish` needs."""
        if frame.dim() != 4 or frame.shape[0] != 1:
            raise ValueError(f"frame must be (1, 3, H, W), got {tuple(frame.shape)}")
        cfg = self.cfg
        size = (int(frame.shape[-2]), int(frame.shape[-1]))
        out = warped_size(size, cfg.scale)
        dev, dt = frame.device, frame.dtype
        grid = self._cached(("grid", str(dev)), lambda: self._grid(state))
        res = self._cached(("resample", size, str(dev), dt),
                           lambda: _Resampler(cfg.resample, grid, size, out, cfg.tap_count(), dt, dev))
        x = res(frame)
        if self.precision == "fp32":
            x = x.to(torch.float32)
        return x, {"grid": grid, "frame_size": size, "input_size": out}

    def finish(self, logits: torch.Tensor, ctx: Mapping[str, Any],
               state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(1, C, h, w) logits of :meth:`warp`'s input -> (H, W) labels and the next state."""
        cfg, model = self.cfg, self.model
        if self.precision == "fp32":
            logits = logits.float()
        size, grid = ctx["frame_size"], ctx["grid"]
        key = ("unwarp", size, tuple(logits.shape[-2:]), str(logits.device), logits.dtype)
        unw = self._cached(key, lambda: _Unwarper(cfg.unwarp, grid, size, tuple(logits.shape[-2:]), ctx["input_size"],
                                                  model.output_stride, model.align_corners, logits.dtype,
                                                  logits.device))
        labels = unw(logits).argmax(1)[0]
        if cfg.saliency == "P":
            nxt = self.prior
        elif cfg.uniform:
            nxt = state if state is not None else torch.zeros(_cells(size, self.cell), device=labels.device)
        elif cfg.saliency == "U":
            nxt = raw_saliency("U", logits=logits, grid=grid, cell=self.cell, frame_size=size,
                               align_corners=model.align_corners, output_stride=model.output_stride,
                               input_size=ctx["input_size"])
        else:
            nxt = raw_saliency(cfg.saliency, labels=labels, cell=self.cell)
        return labels, nxt

    def forward(self, frame: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, ctx = self.warp(frame, state)
        return self.finish(self.model(x), ctx, state)

