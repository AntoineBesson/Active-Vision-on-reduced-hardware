"""GPU latency protocol: telemetry summaries (CPU) and interleaved CUDA-event timing (GPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eval.gpu_timing import GpuTelemetry, cuda_graph, interleaved_benchmark, time_frames  # noqa: E402

CUDA = torch.cuda.is_available()


def test_telemetry_summary_and_throttle_decoding():
    tel = GpuTelemetry()
    tel.backend = "test"
    tel.samples = [(1.0, 1500.0, 60.0, 50.0, 0x0), (2.0, 1200.0, 70.0, 70.0, 0x4), (9.0, 300.0, 40.0, 10.0, 0x1)]
    w = tel.window(0.5, 2.5)
    assert w["samples"] == 2
    assert w["sm_clock_mhz"] == {"min": 1200.0, "median": 1350.0, "max": 1500.0}
    assert w["throttle_reasons"] == ["sw_power_cap"]
    both = tel.windows([(0.5, 1.5), (8.5, 9.5)])
    assert both["samples"] == 2 and both["throttle_reasons"] == ["gpu_idle"]
    assert tel.window(3.0, 4.0) == {"backend": "test", "samples": 0}


@pytest.mark.skipif(CUDA, reason="checks the CPU-only failure")
def test_time_frames_requires_cuda():
    with pytest.raises(RuntimeError):
        time_frames(lambda: None, iters=1)


@pytest.mark.skipif(not CUDA, reason="GPU-only: needs CUDA")
def test_interleaved_benchmark_on_gpu():
    a = torch.rand(256, 256, device="cuda")
    big = torch.rand(1024, 1024, device="cuda")
    with GpuTelemetry(interval_s=0.05) as tel:
        res = interleaved_benchmark({"small": lambda: a @ a, "big": lambda: big @ big @ big},
                                    rounds=3, iters_per_round=5, warmup=2, telemetry=tel)
    assert set(res) == {"small", "big"}
    for r in res.values():
        assert r["cuda_event"]["iters"] == 15 and len(r["round_medians_ms"]) == 3
        assert r["host"]["median_ms"] >= r["cuda_event"]["median_ms"] * 0.5
        assert "telemetry" in r
    assert res["big"]["cuda_event"]["median_ms"] > res["small"]["cuda_event"]["median_ms"]


@pytest.mark.skipif(not CUDA, reason="GPU-only: needs CUDA")
def test_cuda_graph_replays_the_computation():
    conv = torch.nn.Conv2d(3, 4, 3, padding=1).cuda().eval()
    x = torch.rand(1, 3, 32, 32, device="cuda")
    replay = cuda_graph(conv, x)
    with torch.no_grad():
        assert torch.allclose(replay(), conv(x))
