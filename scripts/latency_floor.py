#!/usr/bin/env python3
"""Latency floor of foveated segmentation on the target GPU (batch 1, FP16, streaming).

    python -m scripts.latency_floor --session s1          # on the T4 (Colab)
    python -m scripts.latency_floor --session dev --quick # smoke run, any CUDA GPU

Needs no dataset: with fixed input shapes and a fixed glimpse count, latency
depends on shapes, dtype and kernels, not on pixel or weight values, so the
real architectures run with random weights (19 classes, eval mode, auxiliary
heads off) on random FP16 frames. Four parts, each timed with the protocol of
``eval.gpu_timing`` (per-frame CUDA events, configurations interleaved in
rounds, median / p95, SM clock / temperature / power / throttle reasons):

* ``area``     backbone latency vs input area, k = 1: square crops up to
               1024 px and the full 1024x2048 frame at 5 scales;
* ``batch``    backbone latency vs number k of same-size crops in one batch;
* ``overhead`` the pipeline without compute (``StubBackbone``): resize,
               uncertainty, selection, crop gather, upsampling, argmax, paste,
               for each context scale / glimpse size / k / selection rule and
               the two output strides of the backbones (8: PIDNet, 1: BiSeNetV2);
* ``e2e``      the full pipeline with real backbones at a few points, with the
               floor predicted from the three parts above
               (context backbone + glimpse batch + overhead) for comparison.

Every part runs at two runtime levels, the same for all methods within a
level: ``eager`` (cuDNN autotuning on) and ``graph`` (the whole call captured
in a CUDA graph, which removes per-kernel launch overhead). GFLOPs are counted
once per configuration (``eval.cost.count_flops``). Run the script in 3+
separate sessions; each writes ``<out>/latency_floor_<session>.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if __package__ in (None, ""):  # `python scripts/latency_floor.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path  # noqa: E402

ensure_src_on_path()

import torch  # noqa: E402

from eval.cost import count_flops  # noqa: E402
from eval.gpu_timing import GpuTelemetry, cuda_graph, gpu_environment, interleaved_benchmark  # noqa: E402
from foveation.pipeline import FoveatedSegmenter, FoveationConfig, StubBackbone  # noqa: E402
from models.registry import build_backbone  # noqa: E402

FRAME = (1024, 2048)
MODELS = ("pidnet_s", "bisenetv2")
STRIDE = {"pidnet_s": 8, "bisenetv2": 1}
ALIGN = {"pidnet_s": True, "bisenetv2": False}
MARGIN = 32

FULL = dict(
    full_scales=[0.25, 0.375, 0.5, 0.75, 1.0],
    squares=[128, 256, 384, 512, 768, 1024],
    batch_sizes=[128, 256, 384, 512, 768],
    ks=[1, 2, 4, 8, 16],
    max_batch_pixels=8 * 768 * 768,  # caps k for the largest crops
    ctx_scales=[0.25, 0.375, 0.5],
    glimpses=[256, 384, 512],
    overhead_ks=[1, 2, 4, 8],
    selections=["topk", "greedy"],
    e2e=[(0.25, 384, 4), (0.25, 512, 4), (0.375, 256, 8), (0.5, 512, 2)],
    rounds=5, iters=40, warmup=20,
)
QUICK = dict(
    full_scales=[0.25, 1.0], squares=[256, 512], batch_sizes=[384], ks=[1, 4], max_batch_pixels=8 * 768 * 768,
    ctx_scales=[0.25], glimpses=[384], overhead_ks=[4], selections=["topk", "greedy"],
    e2e=[(0.25, 384, 4)], rounds=2, iters=10, warmup=5,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="Session tag (one per separate GPU session).")
    p.add_argument("--out", default=str(REPO_ROOT / "experiments" / "latency_floor"))
    p.add_argument("--quick", action="store_true", help="Few configurations, few iterations (smoke test).")
    p.add_argument("--levels", default="eager,graph", help="Comma list of runtime levels: eager, graph.")
    p.add_argument("--parts", default="area,batch,overhead,e2e")
    return p.parse_args(argv)


def build(kind: str) -> torch.nn.Module:
    m = build_backbone({"type": kind, "num_classes": 19, "allow_random_init": True})
    return m.cuda().eval().half()


def frame_of(shape: Tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0)
    return torch.rand(shape, generator=g).to("cuda", torch.float16)


def gflops(fn: Callable[[torch.Tensor], Any], x: torch.Tensor) -> Optional[float]:
    f = count_flops(lambda: fn(x))
    return None if f is None else f / 1e9


def time_part(cases: List[Dict[str, Any]], level: str, s: Dict[str, Any], tel: GpuTelemetry,
              memory_fraction: float = 0.6) -> List[Dict[str, Any]]:
    """cases: dicts with 'name', 'fn' (input -> output), 'x' and metadata; returns records.

    eager: all cases interleaved together. graph: each CUDA graph keeps a
    private memory pool (graphs sharing one pool crashed on replay), so graphs
    are captured in chunks that fit in ``memory_fraction`` of the GPU and each
    chunk is timed interleaved before the next one is captured.
    """
    total = torch.cuda.get_device_properties(0).total_memory
    out: List[Dict[str, Any]] = []
    i, chunk_id = 0, 0
    while i < len(cases):
        chunk, callables = [], {}
        while i < len(cases):
            c = cases[i]
            if level == "eager":
                callables[c["name"]] = (lambda fn, x: lambda: fn(x))(c["fn"], c["x"])
            else:
                callables[c["name"]] = cuda_graph(c["fn"], c["x"])
            chunk.append(c)
            i += 1
            if level == "graph" and torch.cuda.memory_reserved() > memory_fraction * total:
                break
        res = interleaved_benchmark(callables, rounds=s["rounds"], iters_per_round=s["iters"], warmup=s["warmup"],
                                    telemetry=tel)
        for c in chunk:
            rec = {k: v for k, v in c.items() if k not in ("fn", "x")}
            rec.update({"level": level, "chunk": chunk_id, **res[c["name"]]})
            out.append(rec)
        del callables, res
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        chunk_id += 1
    return out


def area_cases(models: Dict[str, torch.nn.Module], s: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    shapes = [("frame", sc, (round(FRAME[0] * sc), round(FRAME[1] * sc))) for sc in s["full_scales"]]
    shapes += [("square", None, (q, q)) for q in s["squares"]]
    for name, m in models.items():
        for kind, sc, (h, w) in shapes:
            x = frame_of((1, 3, h, w))
            cases.append({"name": f"area/{name}/{h}x{w}", "model": name, "kind": kind, "scale": sc, "h": h, "w": w,
                          "pixels": h * w, "gflops": gflops(m, x), "fn": m, "x": x})
    return cases


def batch_cases(models: Dict[str, torch.nn.Module], s: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    for name, m in models.items():
        for q in s["batch_sizes"]:
            for k in s["ks"]:
                if k * q * q > s["max_batch_pixels"]:
                    continue
                x = frame_of((k, 3, q, q))
                cases.append({"name": f"batch/{name}/{q}/k{k}", "model": name, "size": q, "k": k, "pixels": k * q * q,
                              "gflops": gflops(m, x), "fn": m, "x": x})
    return cases


def overhead_cases(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    x = frame_of((1, 3, *FRAME))
    for model in MODELS:
        stub = StubBackbone(19, output_stride=STRIDE[model], align_corners=ALIGN[model]).cuda()
        for cs in s["ctx_scales"]:
            for q in s["glimpses"]:
                for k in s["overhead_ks"]:
                    for sel in s["selections"]:
                        cfg = FoveationConfig(context_scale=cs, glimpse_size=q, glimpse_margin=MARGIN, k=k, selection=sel)
                        seg = FoveatedSegmenter(stub, stub, cfg, FRAME).cuda().eval()
                        with torch.no_grad():
                            seg(x)  # allocate the stub's buffers outside the timed / captured region
                        cases.append({"name": f"overhead/stride{STRIDE[model]}/{cs}/{q}/k{k}/{sel}", "stride": STRIDE[model],
                                      "context_scale": cs, "size": q, "k": k, "selection": sel,
                                      "gflops": gflops(seg, x), "fn": seg, "x": x})
    return cases


def e2e_cases(models: Dict[str, torch.nn.Module], s: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    x = frame_of((1, 3, *FRAME))
    pairs = [(n, n) for n in models] + ([("bisenetv2", "pidnet_s")] if len(models) == 2 else [])
    for ctx, gl in pairs:
        for cs, q, k in s["e2e"]:
            cfg = FoveationConfig(context_scale=cs, glimpse_size=q, glimpse_margin=MARGIN, k=k, selection="topk")
            seg = FoveatedSegmenter(models[ctx], models[gl], cfg, FRAME).cuda().eval()
            d = seg.describe()
            cases.append({"name": f"e2e/{ctx}+{gl}/{cs}/{q}/k{k}", "context_model": ctx, "glimpse_model": gl,
                          "context_scale": cs, "size": q, "k": k, "pixel_fraction": d["pixel_fraction"],
                          "grid": d["grid"], "gflops": gflops(seg, x), "fn": seg, "x": x})
    return cases


def predict_e2e(records: Dict[str, List[Dict[str, Any]]]) -> None:
    """Add the floor predicted from area + batch + overhead to each e2e record."""
    def median(part: str, level: str, **match: Any) -> Optional[float]:
        for r in records.get(part, []):
            if r["level"] == level and all(r.get(k) == v for k, v in match.items()):
                return r["cuda_event"]["median_ms"]
        return None

    for r in records.get("e2e", []):
        if r["context_model"] != r["glimpse_model"]:
            continue
        m, lv = r["context_model"], r["level"]
        h, w = round(FRAME[0] * r["context_scale"]), round(FRAME[1] * r["context_scale"])
        parts = {
            "context": median("area", lv, model=m, h=h, w=w),
            "glimpses": median("batch", lv, model=m, size=r["size"], k=r["k"]),
            "overhead": median("overhead", lv, stride=STRIDE[m], context_scale=r["context_scale"], size=r["size"],
                               k=r["k"], selection="topk"),
        }
        if all(v is not None for v in parts.values()):
            r["predicted_ms"] = {**parts, "total": sum(parts.values())}
            r["measured_over_predicted"] = r["cuda_event"]["median_ms"] / r["predicted_ms"]["total"]


def summary(records: Dict[str, List[Dict[str, Any]]]) -> None:
    for part, recs in records.items():
        print(f"\n== {part} ==")
        for r in recs:
            t = r["cuda_event"]
            tel = r.get("telemetry", {})
            clk = tel.get("sm_clock_mhz", {}).get("median")
            extra = ""
            if "predicted_ms" in r:
                extra = f"  predicted {r['predicted_ms']['total']:7.2f}  ratio {r['measured_over_predicted']:.2f}"
            g = r.get("gflops")
            print(f"  {r['level']:5s} {r['name']:44s} median {t['median_ms']:7.2f}  p95 {t['p95_ms']:7.2f} ms"
                  f"  {'' if g is None else f'{g:7.1f} GF'}  clk {clk}{extra}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("latency_floor needs a CUDA GPU", file=sys.stderr)
        return 2
    s = QUICK if args.quick else FULL
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    torch.backends.cudnn.benchmark = True
    models = {name: build(name) for name in MODELS}
    started = _dt.datetime.now().isoformat(timespec="seconds")
    records: Dict[str, List[Dict[str, Any]]] = {}
    t0 = time.time()
    with GpuTelemetry() as tel:
        builders = {"area": lambda: area_cases(models, s), "batch": lambda: batch_cases(models, s),
                    "overhead": lambda: overhead_cases(s), "e2e": lambda: e2e_cases(models, s)}
        for part in parts:
            cases = builders[part]()
            for level in levels:
                print(f"[latency_floor] {part} / {level}: {len(cases)} configurations", flush=True)
                records.setdefault(part, []).extend(time_part(cases, level, s, tel))
            del cases
        session_tel = tel.window(float("-inf"), float("inf"))
    predict_e2e(records)
    out = {
        "experiment": "latency_floor",
        "session": args.session,
        "started": started,
        "duration_s": time.time() - t0,
        "quick": args.quick,
        "settings": {**s, "frame": list(FRAME), "margin": MARGIN, "levels": levels, "dtype": "float16", "batch": "1 frame",
                     "timing": "per-frame CUDA events after synchronize; configurations interleaved in rounds",
                     "weights": "random (latency does not depend on values)"},
        "environment": gpu_environment(),
        "session_telemetry": session_tel,
        "records": records,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"latency_floor_{args.session}.json"
    path.write_text(json.dumps(out, indent=1))
    summary(records)
    print(f"\n[latency_floor] {time.time() - t0:.0f} s, wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
