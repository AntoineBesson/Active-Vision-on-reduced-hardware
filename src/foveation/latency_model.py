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
stride-1 overhead. Only measured points are known: anything else raises.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

__all__ = ["LatencyTables", "STRIDE"]

STRIDE = {"pidnet_s": 8, "bisenetv2": 1}


def _missing(part: str, args: Mapping[str, Any]) -> KeyError:
    shown = ", ".join(f"{k}={v}" for k, v in args.items())
    return KeyError(f"no T4 measurement for {part} ({shown}): run scripts/latency_floor.py for this point")


class LatencyTables:
    def __init__(self, records: Mapping[str, List[Dict[str, Any]]], sessions: int = 1):
        """``records``: part -> one entry per measured point (graph level), already merged over sessions."""
        self.records = {part: list(v) for part, v in records.items()}
        self.sessions = sessions

    @classmethod
    def from_records(cls, records: Mapping[str, Iterable[Mapping[str, Any]]]) -> "LatencyTables":
        """One session's ``records`` (latency-floor JSON format); only graph-level entries are kept."""
        return cls._merge([records])

    @classmethod
    def load(cls, paths: Iterable[Path]) -> "LatencyTables":
        return cls._merge([json.loads(Path(p).read_text())["records"] for p in paths])

    @classmethod
    def _merge(cls, sessions: List[Mapping[str, Iterable[Mapping[str, Any]]]]) -> "LatencyTables":
        by_name: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
        for recs in sessions:
            for part, items in recs.items():
                for r in items:
                    if r.get("level", "graph") == "graph":
                        by_name.setdefault(part, {}).setdefault(r["name"], []).append(r)
        merged: Dict[str, List[Dict[str, Any]]] = {}
        for part, names in by_name.items():
            for rs in names.values():
                base = {k: v for k, v in rs[0].items() if k not in ("cuda_event", "host", "telemetry", "round_medians_ms")}
                base["median_ms"] = statistics.median(r["cuda_event"]["median_ms"] for r in rs)
                base["p95_ms"] = statistics.median(r["cuda_event"]["p95_ms"] for r in rs)
                base["sessions"] = len(rs)
                merged.setdefault(part, []).append(base)
        return cls(merged, sessions=len(sessions))

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
                selection: str) -> Dict[str, Any]:
        stride = 1 if 1 in (STRIDE[ctx_model], STRIDE[glimpse_model]) else 8
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
