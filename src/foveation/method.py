"""Method (2) of the study as a :class:`eval.harness.FrameMethod`: context + glimpses, no memory.

Wraps :class:`foveation.pipeline.FoveatedSegmenter` so the generic evaluation
loop, cost ledger and run records treat it like the full-frame baseline. Costs
per frame go to four ledger components: ``context`` (the low-resolution pass),
``glimpses`` (the k crops, margins included), ``policy`` (scoring and
selection) and ``recomposition`` (upsampling, argmax, paste). The last two are
memory-bound ops with no FLOPs in the counter's sense; their cost shows up in
latency only.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch

from eval.cost import CostLedger, count_flops
from eval.harness import PRECISIONS

from .ops import downsample
from .pipeline import FoveatedSegmenter, FoveationConfig

__all__ = ["FoveatedMethod"]


class FoveatedMethod:
    name = "foveated"
    requires_all_frames = False

    def __init__(self, backbone_ctx: torch.nn.Module, backbone_glimpse: torch.nn.Module, cfg: FoveationConfig,
                 precision: str = "fp32"):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        self.cfg, self.precision = cfg, precision
        self.dtype = PRECISIONS[precision]
        same = backbone_ctx is backbone_glimpse
        if self.dtype != torch.float32:
            backbone_ctx = copy.deepcopy(backbone_ctx).to(self.dtype)
            backbone_glimpse = backbone_ctx if same else copy.deepcopy(backbone_glimpse).to(self.dtype)
        self.ctx, self.glimpse = backbone_ctx.eval(), backbone_glimpse.eval()
        self._segmenters: Dict[Tuple[int, int], FoveatedSegmenter] = {}
        self._flops: Dict[Tuple[int, int], Tuple[Optional[int], Optional[int]]] = {}

    def segmenter(self, frame_size: Tuple[int, int]) -> FoveatedSegmenter:
        frame_size = (int(frame_size[0]), int(frame_size[1]))
        if frame_size not in self._segmenters:
            device = next(self.ctx.parameters()).device
            self._segmenters[frame_size] = FoveatedSegmenter(self.ctx, self.glimpse, self.cfg, frame_size).to(device).eval()
        return self._segmenters[frame_size]

    def reset(self, clip: Mapping[str, Any]) -> None:
        pass  # memory-free

    def predict_with_idx(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(H, W) train ids and the (k,) selected cell ids of one (1, 3, H, W) frame."""
        with torch.no_grad():
            return self.segmenter(frame.shape[-2:])(frame.to(self.dtype), return_idx=True)

    def predict(self, frame: torch.Tensor, t: int, ledger: CostLedger) -> torch.Tensor:
        size = tuple(int(s) for s in frame.shape[-2:])
        seg = self.segmenter(size)
        small = downsample(frame, self.cfg.context_scale)
        q = self.cfg.glimpse_size
        if size not in self._flops:
            x = small.to(self.dtype)
            crops = torch.zeros(self.cfg.k, 3, q, q, device=frame.device, dtype=self.dtype)
            self._flops[size] = (count_flops(lambda: self.ctx(x)), count_flops(lambda: self.glimpse(crops)))
        labels, _ = self.predict_with_idx(frame)
        ctx_flops, gl_flops = self._flops[size]
        ledger.add("context", flops=ctx_flops or 0, pixels=small.shape[-2] * small.shape[-1])
        ledger.add("glimpses", flops=gl_flops or 0, pixels=self.cfg.k * q * q)
        ledger.add("policy")
        ledger.add("recomposition")
        return labels

    def latency_fns(self, frame_size: Tuple[int, int], device: torch.device) -> Dict[str, Callable[[], Any]]:
        """``end_to_end``: one full-resolution frame through the whole pipeline."""
        ctx = copy.deepcopy(self.ctx).to(device).eval()
        glimpse = ctx if self.glimpse is self.ctx else copy.deepcopy(self.glimpse).to(device).eval()
        seg = FoveatedSegmenter(ctx, glimpse, self.cfg, frame_size).to(device).eval()
        frame = torch.rand(1, 3, *frame_size, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=self.dtype)

        def run() -> torch.Tensor:
            with torch.no_grad():
                return seg(frame)

        return {"end_to_end": run}

    def resident_bytes(self) -> int:
        """Bytes of weights (parameters + buffers) of both backbones (once if shared)."""
        mods = [self.ctx] if self.glimpse is self.ctx else [self.ctx, self.glimpse]
        tensors = [t for m in mods for t in list(m.parameters()) + list(m.buffers())]
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def describe(self) -> Dict[str, Any]:
        seg = next(iter(self._segmenters.values()), None)
        info: Dict[str, Any] = {"name": self.name, **asdict(self.cfg), "precision": self.precision,
                                "context_model": type(self.ctx).__name__, "glimpse_model": type(self.glimpse).__name__}
        if seg is not None:
            d = seg.describe()
            info.update(grid=d["grid"], pixel_fraction=d["pixel_fraction"], frame_size=d["frame_size"])
        return info
