"""T4 latency of a foveated configuration, predicted from the latency-floor tables.

The latency-floor experiment (``scripts/latency_floor.py``, results in
``results/latency_floor/``) timed each part of the pipeline on a T4 at batch 1,
fp16, as CUDA graphs: the context backbone at every context scale (``area``),
the glimpse backbone on k crops (``batch``), and the pipeline's own overhead
with a zero-compute backbone (``overhead``). Their sum predicted the measured
end-to-end time within a few percent, so a configuration's latency is
estimated here as

    context (area) + glimpses (batch) + overhead,

each the median over sessions of the per-session median. The p95 is the sum of
the parts' p95s: a conservative bound, not a measured p95. The overhead was
measured with entropy scoring, mean pooling and the logit stride of one model
for both passes; a pair that involves a stride-1 model (BiSeNetV2) uses the
stride-1 overhead. Only measured points are known: anything else raises,
including a glimpse margin other than the measured one (``settings.margin``).

Method 3 (:meth:`LatencyTables.predict_reuse_frame`) adds the ``reuse`` part:
gray downsample, homography, motion map, warps, the 3b fusion, the post-processing
of a full or context pass, and of k glimpses. It is timed eagerly (the CPU
homography and host-side warp matrices cannot be captured in a CUDA graph), so
its eager records are kept; every other part keeps its graph-level records.

Method 4's bound (:meth:`LatencyTables.predict_canvas_frame`) prices a canvas frame from
the same entries, with the policy at 0.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

__all__ = ["LatencyTables", "STRIDE", "frame_latency_summary"]

STRIDE = {"pidnet_s": 8, "bisenetv2": 1}


def _missing(part: str, args: Mapping[str, Any]) -> KeyError:
    shown = ", ".join(f"{k}={v}" for k, v in args.items())
    return KeyError(f"no T4 measurement for {part} ({shown}): run scripts/latency_floor.py for this point")


class LatencyTables:
    def __init__(self, records: Mapping[str, List[Dict[str, Any]]], sessions: int = 1,
                 strides: Optional[Mapping[str, int]] = None, margin: Optional[int] = None):
        """``records``: part -> one entry per measured point (graph level), already merged over sessions.
        ``strides``: logit stride per model name (default :data:`STRIDE`). ``margin``: the glimpse
        margin the tables were measured with (None: not recorded, not checked)."""
        self.records = {part: list(v) for part, v in records.items()}
        self.sessions = sessions
        self.strides = dict(STRIDE if strides is None else strides)
        self.margin = margin

    @classmethod
    def from_records(cls, records: Mapping[str, Iterable[Mapping[str, Any]]],
                     strides: Optional[Mapping[str, int]] = None) -> "LatencyTables":
        """One session's ``records`` (latency-floor JSON format); only graph-level entries are kept."""
        return cls._merge([records], strides)

    @classmethod
    def load(cls, paths: Iterable[Path], strides: Optional[Mapping[str, int]] = None) -> "LatencyTables":
        runs = [json.loads(Path(p).read_text()) for p in paths]
        margins = {r.get("settings", {}).get("margin") for r in runs}
        if len(margins) > 1:
            raise ValueError(f"latency-floor sessions measured with different glimpse margins: {sorted(map(str, margins))}")
        tables = cls._merge([r["records"] for r in runs], strides)
        tables.margin = margins.pop() if margins else None
        return tables

    @classmethod
    def _merge(cls, sessions: List[Mapping[str, Iterable[Mapping[str, Any]]]],
               strides: Optional[Mapping[str, int]] = None) -> "LatencyTables":
        by_name: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
        for recs in sessions:
            for part, items in recs.items():
                for r in items:
                    if r.get("level", "graph") == "graph" or part == "reuse":
                        by_name.setdefault(part, {}).setdefault(r["name"], []).append(r)
        merged: Dict[str, List[Dict[str, Any]]] = {}
        for part, names in by_name.items():
            for rs in names.values():
                base = {k: v for k, v in rs[0].items() if k not in ("cuda_event", "host", "telemetry", "round_medians_ms")}
                base["median_ms"] = statistics.median(r["cuda_event"]["median_ms"] for r in rs)
                base["p95_ms"] = statistics.median(r["cuda_event"]["p95_ms"] for r in rs)
                base["sessions"] = len(rs)
                merged.setdefault(part, []).append(base)
        return cls(merged, sessions=len(sessions), strides=strides)

    def _find(self, part: str, **match: Any) -> Dict[str, Any]:
        for r in self.records.get(part, []):
            if all(r.get(k) == v for k, v in match.items()):
                return r
        raise _missing(part, match)

    def fullframe(self, model: str, scale: float) -> Dict[str, float]:
        """The full frame resized by ``scale`` (the area table's frame entries)."""
        r = self._find("area", model=model, kind="frame", scale=scale)
        return {"median_ms": r["median_ms"], "p95_ms": r["p95_ms"], "gflops": r["gflops"]}

    def predict(self, ctx_model: str, ctx_scale: float, glimpse_model: str, size: int, k: int,
                selection: str, margin: Optional[int] = None) -> Dict[str, Any]:
        """``margin``: the glimpse margin (input pixels); checked against the measured one when both are known."""
        if margin is not None and self.margin is not None and margin != self.margin:
            raise _missing("glimpse margin", {"margin": margin, "measured": self.margin})
        stride = 1 if 1 in (self.strides[ctx_model], self.strides[glimpse_model]) else 8
        parts = {
            "context": self._find("area", model=ctx_model, kind="frame", scale=ctx_scale),
            "glimpses": self._find("batch", model=glimpse_model, size=size, k=k),
            "overhead": self._find("overhead", stride=stride, context_scale=ctx_scale, size=size, k=k,
                                   selection=selection),
        }
        return {
            "median_ms": sum(p["median_ms"] for p in parts.values()),
            "p95_ms": sum(p["p95_ms"] for p in parts.values()),
            "gflops": parts["context"]["gflops"] + parts["glimpses"]["gflops"],
            "parts": {name: {"median_ms": p["median_ms"], "p95_ms": p["p95_ms"]} for name, p in parts.items()},
            "overhead_stride": stride,
        }

    def _part_ms(self, part: str, **match: Any) -> Dict[str, float]:
        r = self._find(part, **match)
        return {"median_ms": r["median_ms"], "p95_ms": r["p95_ms"]}

    def predict_reuse_frame(self, model: str, cfg: Any, moved_cells: int, fallback: bool,
                            first: bool) -> Dict[str, Any]:
        """Predicted latency of one Method 3 frame (``cfg``: a ``reuse.method.MotionReuseConfig``).

        Frame 0: the full pass. Later frames: gray + homography + motion, then either the
        full pass (fallback) or the warps, the glimpses of the ``moved_cells`` cells and, in
        3b, the context pass and the fusion. A full or context pass = backbone (area) +
        its post-processing (upsample, argmax, confidence). ``gflops``: the backbone passes only."""
        stride = self.strides[model]
        gflops = 0.0

        def frame_pass(scale: float) -> Dict[str, Dict[str, float]]:
            nonlocal gflops
            gflops += self._find("area", model=model, kind="frame", scale=scale)["gflops"]
            return {"backbone": self._part_ms("area", model=model, kind="frame", scale=scale),
                    "post": self._part_ms("reuse", op="post", stride=stride, scale=scale)}

        parts: Dict[str, Dict[str, float]] = {}
        if first:
            parts.update({f"full_{k}": v for k, v in frame_pass(cfg.full_scale).items()})
        else:
            parts["gray"] = self._part_ms("reuse", op="gray")
            parts["homography"] = self._part_ms("reuse", op="homography", homography=cfg.homography)
            parts["motion"] = self._part_ms("reuse", op="motion", size=cfg.glimpse_size)
            if fallback:
                parts.update({f"full_{k}": v for k, v in frame_pass(cfg.full_scale).items()})
            else:
                parts["warp"] = self._part_ms("reuse", op="warp")
                if moved_cells:
                    parts["glimpses"] = self._part_ms("batch", model=model, size=cfg.glimpse_size, k=moved_cells)
                    gflops += self._find("batch", model=model, size=cfg.glimpse_size, k=moved_cells)["gflops"]
                    parts["glimpse_post"] = self._part_ms("reuse", op="glimpse_post", stride=stride,
                                                          size=cfg.glimpse_size, k=moved_cells)
                if cfg.ctx_scale is not None:
                    parts.update({f"context_{k}": v for k, v in frame_pass(cfg.ctx_scale).items()})
                    parts["fusion"] = self._part_ms("reuse", op="fusion")
        return {"median_ms": sum(p["median_ms"] for p in parts.values()),
                "p95_ms": sum(p["p95_ms"] for p in parts.values()), "gflops": gflops, "parts": parts}


    def predict_canvas_frame(self, model: str, cfg: Any, k: int, first: bool,
                             with_motion: bool = False) -> Dict[str, Any]:
        """Predicted latency of one frame of Method 4's bound (``cfg``: a ``canvas.replay.CanvasConfig``).

        Every frame: the context pass (backbone + post-processing) and, when ``k`` > 0, the k
        glimpses and their post-processing. A canvas (variants ``iclk`` and ``none``) adds the
        fusion on every frame but the first; variant ``iclk`` also adds the gray downsample
        (every frame) and, after the first frame, the homography and the warps. The policy
        costs nothing. ``with_motion`` adds the motion map (variant ``iclk``, after the first
        frame): a policy input, not a part of the mechanism. ``gflops``: the backbone passes only."""
        stride = self.strides[model]
        ctx = self._find("area", model=model, kind="frame", scale=cfg.ctx_scale)
        gflops = ctx["gflops"]
        parts: Dict[str, Dict[str, float]] = {
            "context_backbone": {"median_ms": ctx["median_ms"], "p95_ms": ctx["p95_ms"]},
            "context_post": self._part_ms("reuse", op="post", stride=stride, scale=cfg.ctx_scale)}
        if k:
            parts["glimpses"] = self._part_ms("batch", model=model, size=cfg.glimpse_size, k=k)
            gflops += self._find("batch", model=model, size=cfg.glimpse_size, k=k)["gflops"]
            parts["glimpse_post"] = self._part_ms("reuse", op="glimpse_post", stride=stride, size=cfg.glimpse_size, k=k)
        if cfg.variant == "iclk":
            parts["gray"] = self._part_ms("reuse", op="gray")
        if cfg.variant != "off" and not first:
            parts["fusion"] = self._part_ms("reuse", op="fusion")
            if cfg.variant == "iclk":
                parts["homography"] = self._part_ms("reuse", op="homography", homography="iclk")
                parts["warp"] = self._part_ms("reuse", op="warp")
                if with_motion:
                    parts["motion"] = self._part_ms("reuse", op="motion", size=cfg.glimpse_size)
        return {"median_ms": sum(p["median_ms"] for p in parts.values()),
                "p95_ms": sum(p["p95_ms"] for p in parts.values()), "gflops": gflops, "parts": parts}


def frame_latency_summary(per_frame_ms: Iterable[float]) -> Dict[str, float]:
    """The two judges of Method 3 over a set of frames: mean and p95 (numpy's linear percentile)."""
    ms = np.asarray(list(per_frame_ms), dtype=np.float64)
    return {"mean_ms": float(ms.mean()), "p95_ms": float(np.percentile(ms, 95)), "frames": int(ms.size)}
