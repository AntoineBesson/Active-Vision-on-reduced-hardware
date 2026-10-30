"""Compute-cost accounting: FLOPs, latency, peak memory, fraction of pixels processed.

* :class:`CostLedger` accumulates per-frame costs by *named component*. The
  full-frame baseline uses one component ("backbone"); foveated methods add
  e.g. "context", "glimpses", "policy", "memory", "recomposition" -- they all
  sum into the total, so new pipeline stages only need to report to the
  ledger (``add`` or the FLOP-counting ``count`` context).
* :func:`count_flops` uses ``torch.utils.flop_counter.FlopCounterMode``, which
  ships with torch (>= 2.1): no extra dependency, and it counts at the
  dispatcher level, so it handles data-dependent control flow (gaze loops)
  that tracing-based counters (fvcore) can miss, and functional ops that
  module-hook counters (ptflops) skip. Convention: 2 FLOPs per
  multiply-accumulate of convolutions and matrix products; elementwise ops,
  normalisation, pooling, interpolation and bias additions are not counted.
  MACs = FLOPs / 2 are reported too.
* :func:`benchmark_latency` measures wall-clock latency: warm-up calls, then
  per-call timing between ``torch.cuda.synchronize()`` barriers; median and
  95th percentile reported. :func:`measure_peak_memory` reads
  ``torch.cuda.max_memory_allocated`` (absolute peak and increment over the
  memory already resident). Both are CUDA-only by design (the
  target is a T4) and report ``status: skipped`` elsewhere unless CPU timing
  is explicitly allowed.

The ledger needs only the standard library; torch is imported lazily.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

__all__ = [
    "FLOP_COUNTER_NOTE",
    "CostLedger",
    "count_flops",
    "latency_stats",
    "benchmark_latency",
    "measure_peak_memory",
    "skipped",
]

FLOP_COUNTER_NOTE = (
    "torch.utils.flop_counter.FlopCounterMode: 2 FLOPs per multiply-accumulate of convolutions "
    "and matrix products; elementwise ops, normalisation, pooling, interpolation and bias "
    "additions are not counted. MACs = FLOPs / 2."
)


def skipped(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #
def count_flops(fn: Callable[[], Any]) -> Optional[int]:
    """FLOPs of the torch ops executed by ``fn()`` (None if the counter is unavailable)."""
    import torch

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # torch < 2.1
        return None
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            fn()
    return int(counter.get_total_flops())


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class CostLedger:
    """Per-frame cost accounting by named pipeline component.

    Call :meth:`begin_frame` once per processed frame, then report each
    component's work with :meth:`add` (known FLOPs / pixels) or wrap it in
    :meth:`count` to count the FLOPs of the torch ops it runs.
    ``pixels`` = pixels fed to a backbone at its input resolution; the pixel
    fraction is their per-frame total over the full-resolution frame size.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.full_pixels = 0
        self._flops: Dict[str, float] = {}
        self._pixels: Dict[str, float] = {}
        self._calls: Dict[str, int] = {}
        self.latency: Dict[str, Any] = {}
        self.memory: Dict[str, Any] = {}

    def begin_frame(self, full_frame_pixels: int) -> None:
        self.frames += 1
        self.full_pixels += int(full_frame_pixels)

    def add(self, component: str, flops: float = 0.0, pixels: float = 0.0, calls: int = 1) -> None:
        self._flops[component] = self._flops.get(component, 0.0) + float(flops)
        self._pixels[component] = self._pixels.get(component, 0.0) + float(pixels)
        self._calls[component] = self._calls.get(component, 0) + int(calls)

    @contextmanager
    def count(self, component: str, pixels: float = 0.0) -> Iterator[None]:
        """Count the FLOPs of the torch ops run inside the block and add them."""
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            yield
        self.add(component, flops=counter.get_total_flops(), pixels=pixels)

    def summary(self) -> Dict[str, Any]:
        n = max(self.frames, 1)
        components = {}
        for name in sorted(self._calls):
            fl = self._flops[name]
            components[name] = {
                "flops_per_frame": fl / n,
                "gflops_per_frame": fl / n / 1e9,
                "gmacs_per_frame": fl / n / 2e9,
                "pixels_per_frame": self._pixels[name] / n,
                "calls": self._calls[name],
                "calls_per_frame": self._calls[name] / n,
            }
        total_flops = sum(self._flops.values())
        total_pixels = sum(self._pixels.values())
        return {
            "frames": self.frames,
            "components": components,
            "total": {
                "flops_per_frame": total_flops / n,
                "gflops_per_frame": total_flops / n / 1e9,
                "gmacs_per_frame": total_flops / n / 2e9,
                "pixels_per_frame": total_pixels / n,
            },
            "pixel_fraction": (total_pixels / self.full_pixels) if self.full_pixels else None,
            "full_frame_pixels_per_frame": self.full_pixels / n,
            "flop_counter": FLOP_COUNTER_NOTE,
            "latency": self.latency,
            "memory": self.memory,
        }


# --------------------------------------------------------------------------- #
# Latency / memory
# --------------------------------------------------------------------------- #
def latency_stats(times_ms: List[float]) -> Dict[str, float]:
    a = np.asarray(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "iters": int(a.size),
    }


def benchmark_latency(fn: Callable[[], Any], device: Any, warmup: int = 20, iters: int = 100) -> Dict[str, Any]:
    """Wall-clock latency of ``fn()``: ``warmup`` untimed calls, then ``iters``
    calls each timed between ``torch.cuda.synchronize()`` barriers on CUDA."""
    import torch

    device = torch.device(device)
    cuda = device.type == "cuda"

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            fn()
        sync()
        times = []
        for _ in range(max(1, iters)):
            sync()
            start = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - start) * 1e3)
    out = latency_stats(times)
    out.update({"status": "ok", "device": str(device), "warmup": int(warmup), "synchronized": cuda})
    if cuda:
        out["device_name"] = torch.cuda.get_device_name(device)
    return out


def measure_peak_memory(fn: Callable[[], Any], device: Any, repeats: int = 3) -> Dict[str, Any]:
    """Peak CUDA memory while running ``fn()``; CUDA only.

    ``peak_allocated_mib`` is absolute, so it includes everything already
    resident on the device (e.g. another copy of the model kept for the
    accuracy pass). ``peak_increment_mib`` is the growth over what was
    allocated before the call (activations + workspace), which, added to the
    weights of the measured model, estimates a deployment's footprint.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return skipped("peak memory is measured on CUDA only")
    with torch.no_grad():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(max(1, repeats)):
            fn()
        torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "status": "ok",
        "peak_allocated_mib": peak / 2**20,
        "allocated_before_mib": before / 2**20,
        "peak_increment_mib": (peak - before) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "device": str(device),
    }
