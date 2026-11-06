"""GPU latency protocol for streaming, batch-1 inference (torch).

The comparison between methods is judged on measured end-to-end latency per
frame on the target GPU (a T4), so the measurement itself is fixed here:

* :func:`time_frames` times each call alone, as a frame of a live stream:
  ``synchronize`` (the GPU is idle when the frame "arrives"), record a start
  CUDA event, run, record an end event, wait for it. The event interval is
  the reported latency; a host ``perf_counter`` interval over the same call
  is kept as a cross-check (it adds the final wait and Python overhead).
  Nothing may be loaded from disk inside ``fn``.
* :class:`GpuTelemetry` samples SM clock, temperature, power and the active
  throttle reasons in a background thread (NVML via ``pynvml`` /
  ``nvidia-ml-py`` when installed, else ``nvidia-smi``), because a T4 throttles
  at its 70 W limit and when hot, and a throttled sample is not comparable.
* :func:`interleaved_benchmark` warms every configuration, then runs them in
  rounds (A, B, C, A, B, C, ...) so slow drifts (clock, temperature) hit all
  configurations alike, and reports median / p95 over all rounds, the
  per-round medians and the telemetry of each configuration's time windows.
* :func:`cuda_graph` captures a callable into a CUDA graph (static input and
  output buffers). Capture fails on any host synchronisation, so a successful
  capture also proves the pipeline is sync-free.

Statistics come from :func:`eval.cost.latency_stats`.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .cost import latency_stats

__all__ = ["THROTTLE_REASONS", "time_frames", "GpuTelemetry", "interleaved_benchmark", "cuda_graph", "gpu_environment"]


def time_frames(fn: Callable[[], Any], iters: int, warmup: int = 0) -> Tuple[List[float], List[float]]:
    """Per-call latency in ms: ``(cuda_event_ms, host_ms)``, one entry per call.

    CUDA only. Each call starts on an idle GPU (``synchronize`` before it), as
    a frame arriving in a stream that is processed faster than it arrives.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("time_frames needs CUDA")
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            fn()
        torch.cuda.synchronize()
        ev_ms, host_ms = [], []
        for _ in range(max(1, iters)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start.record()
            fn()
            end.record()
            end.synchronize()
            host_ms.append((time.perf_counter() - t0) * 1e3)
            ev_ms.append(start.elapsed_time(end))
    return ev_ms, host_ms


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
_SMI_FIELDS = "clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"

#: NVML clocks-event (throttle) reason bits; any of the slowdown/cap ones in a
#: window means its timings ran below full clock.
THROTTLE_REASONS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


class GpuTelemetry:
    """Background sampler of SM clock (MHz), temperature (C), power (W) and
    throttle reasons (NVML bitmask). Use as a context manager; ``window(t0, t1)``
    summarises the samples taken between two ``time.monotonic()`` stamps."""

    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        self.interval_s = float(interval_s)
        self.device_index = int(device_index)
        self.samples: List[Tuple[float, float, float, float, int]] = []
        self.backend = "none"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None

    def _open(self) -> None:
        try:
            import pynvml  # nvidia-ml-py

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.backend = "nvml"
        except Exception:
            self.backend = "nvidia-smi" if shutil.which("nvidia-smi") else "none"

    def _read(self) -> Optional[Tuple[float, float, float, int]]:
        if self.backend == "nvml":
            n, h = self._nvml, self._handle
            return (
                float(n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                float(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                n.nvmlDeviceGetPowerUsage(h) / 1000.0,
                int(n.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
            )
        if self.backend == "nvidia-smi":
            out = subprocess.run(
                ["nvidia-smi", f"--id={self.device_index}", f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().split(",")
            clk, temp, power, reasons = (x.strip() for x in out[:4])
            return float(clk), float(temp), float(power), int(reasons, 16) if reasons.startswith("0x") else int(reasons)
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r = self._read()
            except Exception:
                r = None
            if r is not None:
                self.samples.append((time.monotonic(), *r))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuTelemetry":
        self._open()
        if self.backend != "none":
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def window(self, t0: float, t1: float) -> Dict[str, Any]:
        return self.summarize([x for x in self.samples if t0 <= x[0] <= t1])

    def windows(self, wins: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        return self.summarize([x for x in self.samples if any(t0 <= x[0] <= t1 for t0, t1 in wins)])

    def summarize(self, s: Sequence[Tuple[float, float, float, float, int]]) -> Dict[str, Any]:
        if not s:
            return {"backend": self.backend, "samples": 0}
        a = np.asarray([x[1:4] for x in s], dtype=np.float64)
        reasons = 0
        for x in s:
            reasons |= x[4]

        def summ(col: np.ndarray) -> Dict[str, float]:
            return {"min": float(col.min()), "median": float(np.median(col)), "max": float(col.max())}

        return {
            "backend": self.backend,
            "samples": len(s),
            "sm_clock_mhz": summ(a[:, 0]),
            "temperature_c": summ(a[:, 1]),
            "power_w": summ(a[:, 2]),
            "throttle_reasons": [name for bit, name in THROTTLE_REASONS.items() if reasons & bit],
        }


# --------------------------------------------------------------------------- #
# Interleaved benchmark
# --------------------------------------------------------------------------- #
def interleaved_benchmark(
    configs: Mapping[str, Callable[[], Any]],
    rounds: int = 5,
    iters_per_round: int = 40,
    warmup: int = 20,
    telemetry: Optional[GpuTelemetry] = None,
) -> Dict[str, Dict[str, Any]]:
    """Time every configuration in interleaved rounds; see the module docstring."""
    names = list(configs)
    with torch.no_grad():
        for name in names:  # warm everything first (cuDNN autotuning, allocator)
            for _ in range(max(0, warmup)):
                configs[name]()
        torch.cuda.synchronize()
        ev: Dict[str, List[float]] = {n: [] for n in names}
        host: Dict[str, List[float]] = {n: [] for n in names}
        per_round: Dict[str, List[float]] = {n: [] for n in names}
        windows: Dict[str, List[Tuple[float, float]]] = {n: [] for n in names}
        for _ in range(max(1, rounds)):
            for name in names:
                t0 = time.monotonic()
                e, h = time_frames(configs[name], iters_per_round)
                windows[name].append((t0, time.monotonic()))
                ev[name] += e
                host[name] += h
                per_round[name].append(float(np.median(e)))
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        rec: Dict[str, Any] = {
            "cuda_event": latency_stats(ev[name]),
            "host": latency_stats(host[name]),
            "round_medians_ms": per_round[name],
        }
        if telemetry is not None:
            rec["telemetry"] = telemetry.windows(windows[name])
        out[name] = rec
    return out


# --------------------------------------------------------------------------- #
# CUDA graphs
# --------------------------------------------------------------------------- #
def cuda_graph(fn: Callable[[torch.Tensor], Any], example: torch.Tensor, warmup: int = 3) -> Callable[[], Any]:
    """Capture ``fn(static_input)`` into a CUDA graph; return a replay callable.

    The returned callable replays the graph on the captured static input (its
    content does not matter for latency) and returns the static output.
    Each graph keeps a private memory pool: sharing one pool between graphs
    replayed in an interleaved order crashed with illegal memory accesses, so
    bound memory by timing graphs in chunks instead.
    """
    static_in = example.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        static_out = fn(static_in)

    def replay() -> Any:
        graph.replay()
        return static_out

    replay.graph, replay.static_in = graph, static_in  # the graph reads static_in on every replay
    return replay


def gpu_environment() -> Dict[str, Any]:
    """GPU / software description for a result record."""
    env: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu": p.name, "sm_count": p.multi_processor_count, "total_memory_mib": p.total_memory / 2**20,
                    "capability": f"{p.major}.{p.minor}"})
        if shutil.which("nvidia-smi"):
            q = "driver_version,clocks.max.sm,power.limit"
            try:
                out = subprocess.run(["nvidia-smi", "--id=0", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                env.update({"driver": out[0].strip(), "max_sm_clock_mhz": float(out[1]), "power_limit_w": float(out[2])})
            except Exception:
                pass
    return env
