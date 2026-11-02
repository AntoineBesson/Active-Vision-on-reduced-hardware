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

Method 3 parts (``--parts batchk,reuse,reuse_e2e``; graph level for ``batchk``,
eager for the other two):

* ``batchk``    the ``batch`` table for every k up to the cell count of the
               256 / 384 / 512 px grids (66 / 28 / 15), written as ``batch``;
* ``reuse``     each op of a Method 3 frame: gray downsample, homography (ORB on
               the host CPU, IC-LK on the GPU), motion map + moving cells, the
               label and confidence warps, the 3b fusion, the post-processing of
               a full or context pass, and of k glimpses (zero-compute backbone);
* ``reuse_e2e`` a few whole Method 3 frames with real backbones, to check that
               the parts add up.

Every part runs at two runtime levels, the same for all methods within a
level: ``eager`` (cuDNN autotuning on) and ``graph`` (the whole call captured
in a CUDA graph, which removes per-kernel launch overhead). GFLOPs are counted
once per configuration (``eval.cost.count_flops``). Run the script in 3+
separate sessions; each writes ``<out>/latency_floor_<session>.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import asdict
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
from foveation.ops import GlimpseGrid, downsample  # noqa: E402
from foveation.pipeline import FoveatedSegmenter, FoveationConfig, StubBackbone  # noqa: E402
from models.registry import build_backbone  # noqa: E402
from reuse.method import full_pass, glimpse_pass, paste_cells  # noqa: E402
from reuse.motion import MOTION_FACTOR, estimate_homography, motion_map, moving_cells, to_gray_small, warp  # noqa: E402

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
    batchk_sizes=[256, 384, 512],
    reuse_scales=[0.25, 0.375, 0.5, 0.75, 1.0],  # full passes (0.5-1) and 3b context passes (0.25-0.5)
    homographies=["orb", "iclk"],
    reuse_e2e=[("pidnet_s", 384, 2, None), ("pidnet_s", 384, 8, None), ("bisenetv2", 384, 4, None),
               ("pidnet_s", 256, 12, 0.25)],  # (model, glimpse size, k, 3b context scale)
    rounds=5, iters=40, warmup=20,
)
QUICK = dict(
    full_scales=[0.25, 1.0], squares=[256, 512], batch_sizes=[384], ks=[1, 4], max_batch_pixels=8 * 768 * 768,
    ctx_scales=[0.25], glimpses=[384], overhead_ks=[4], selections=["topk", "greedy"],
    e2e=[(0.25, 384, 4)], batchk_sizes=[384], reuse_scales=[0.25, 1.0], homographies=["orb", "iclk"],
    reuse_e2e=[("pidnet_s", 384, 2, None)], rounds=2, iters=10, warmup=5,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="Session tag (one per separate GPU session).")
    p.add_argument("--out", default=str(REPO_ROOT / "experiments" / "latency_floor"))
    p.add_argument("--quick", action="store_true", help="Few configurations, few iterations (smoke test).")
    p.add_argument("--levels", default="eager,graph", help="Comma list of runtime levels: eager, graph.")
    p.add_argument("--parts", default="area,batch,overhead,e2e",
                   help="Comma list of parts: area, batch, overhead, e2e, final (needs --finalists); "
                        "method 3: batchk, reuse, reuse_e2e.")
    p.add_argument("--finalists", default=str(REPO_ROOT / "results" / "method2" / "finalists.json"),
                   help="Method 2 finalists to time with the full-frame curve (part 'final').")
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


def _texture(h: int, w: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """Smooth random texture in [0, 1] (gives the homography realistic features)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    small = torch.rand(1, 1, h // 16 + 2, w // 16 + 2, generator=g)
    up = torch.nn.functional.interpolate(small, size=(h + 32, w + 32), mode="bicubic", align_corners=False)
    return up[..., 16:16 + h, 16:16 + w].clamp(0, 1).to(device)


def num_cells(size: int) -> int:
    return GlimpseGrid(FRAME, size, MARGIN).num_cells


def batchk_specs(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Method 3's glimpse batches: every k up to the grid's cell count (named like ``batch``)."""
    return [{"name": f"batch/{name}/{q}/k{k}", "model": name, "size": q, "k": k, "pixels": k * q * q}
            for name in MODELS for q in s["batchk_sizes"] for k in range(1, num_cells(q) + 1)]


def batchk_cases(models: Dict[str, torch.nn.Module], s: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    for sp in batchk_specs(s):
        x = frame_of((sp["k"], 3, sp["size"], sp["size"]))
        cases.append({**sp, "gflops": gflops(models[sp["model"]], x), "fn": models[sp["model"]], "x": x})
    return cases


def reuse_specs(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ops of a Method 3 frame, without tensors (see :func:`reuse_cases`)."""
    specs = [{"name": "reuse/gray", "part": "reuse", "op": "gray"}]
    specs += [{"name": f"reuse/homography/{m}", "part": "reuse", "op": "homography", "homography": m}
              for m in s["homographies"]]
    specs += [{"name": f"reuse/motion/{q}", "part": "reuse", "op": "motion", "size": q} for q in s["batchk_sizes"]]
    specs += [{"name": "reuse/warp", "part": "reuse", "op": "warp"},
              {"name": "reuse/fusion", "part": "reuse", "op": "fusion"}]
    for stride in sorted(set(STRIDE.values()), reverse=True):
        specs += [{"name": f"reuse/post/stride{stride}/{sc:g}", "part": "reuse", "op": "post", "stride": stride,
                   "scale": sc} for sc in s["reuse_scales"]]
        specs += [{"name": f"reuse/glimpse_post/stride{stride}/{q}/k{k}", "part": "reuse", "op": "glimpse_post",
                   "stride": stride, "size": q, "k": k} for q in s["batchk_sizes"] for k in range(1, num_cells(q) + 1)]
    return specs


def _pair(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two gray frames at the motion resolution, the second the first shifted by (3, 5) px."""
    h, w = FRAME[0] // MOTION_FACTOR, FRAME[1] // MOTION_FACTOR
    big = _texture(h + 16, w + 16, seed)[0, 0]
    return big[8:8 + h, 8:8 + w].contiguous(), big[5:5 + h, 3:3 + w].contiguous()


def _glimpse_post(stub, frame, grid, idx, labels, conf):
    g_lab, g_conf, _ = glimpse_pass(stub, frame, grid, idx, torch.float16)
    return (paste_cells(labels, g_lab, grid.inner, grid.cols, idx),
            paste_cells(conf, g_conf, grid.inner // MOTION_FACTOR, grid.cols, idx))


def reuse_cases(s: Dict[str, Any]) -> List[Dict[str, Any]]:
    h, w = FRAME
    small = (h // MOTION_FACTOR, w // MOTION_FACTOR)
    frame = _texture(h, w, 1).expand(1, 3, h, w).contiguous()
    prev, cur = _pair()
    H = torch.tensor([[1, 0, 20], [0, 1, 12], [0, 0, 1]], dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(0)
    labels = torch.randint(0, 19, (h, w), generator=g).cuda()
    conf = torch.randint(0, 256, small, generator=g, dtype=torch.uint8).cuda()
    stubs = {st: StubBackbone(19, output_stride=st, align_corners=st == 8).cuda() for st in set(STRIDE.values())}

    def warps(_):
        return (warp(labels[None, None].float(), H, "nearest")[0][0, 0].round().long(),
                warp(conf[None, None].float(), H, "nearest", frame_size=FRAME)[0][0, 0].round().to(torch.uint8))

    def fusion(_):
        take = conf >= conf.flip(1)
        full = take.repeat_interleave(MOTION_FACTOR, 0).repeat_interleave(MOTION_FACTOR, 1)
        return torch.where(full, labels, labels.flip(1)), torch.where(take, conf, conf.flip(1))

    cases = []
    for sp in reuse_specs(s):
        op = sp["op"]
        if op == "gray":
            fn = to_gray_small
        elif op == "homography":
            fn = (lambda m: lambda _: estimate_homography(prev, cur, method=m))(sp["homography"])
        elif op == "motion":
            fn = (lambda grid: lambda _: moving_cells(motion_map(prev, cur, H), grid, 30, 0.05))(
                GlimpseGrid(FRAME, sp["size"], MARGIN))
        elif op == "warp":
            fn = warps
        elif op == "fusion":
            fn = fusion
        elif op == "post":
            fn = (lambda stub, sc: lambda x: full_pass(stub, x, sc, small, torch.float16)[:2])(stubs[sp["stride"]],
                                                                                          sp["scale"])
        else:  # glimpse_post
            grid = GlimpseGrid(FRAME, sp["size"], MARGIN)
            idx = torch.arange(sp["k"], device="cuda")
            fn = (lambda stub, grid, idx: lambda x: _glimpse_post(stub, x, grid, idx, labels, conf))(
                stubs[sp["stride"]], grid, idx)
        with torch.no_grad():
            fn(frame)  # allocate stub buffers outside the timed region
        cases.append({**sp, "gflops": 0.0, "fn": fn, "x": frame})
    return cases


def reuse_e2e_cases(models: Dict[str, torch.nn.Module], s: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Whole non-fallback Method 3 frames with real backbones and a fixed set of k moving cells."""
    h, w = FRAME
    small = (h // MOTION_FACTOR, w // MOTION_FACTOR)
    g = torch.Generator(device="cpu").manual_seed(0)
    labels = torch.randint(0, 19, (h, w), generator=g).cuda()
    conf = torch.randint(0, 256, small, generator=g, dtype=torch.uint8).cuda()
    prev_gray = to_gray_small(_texture(h, w, 2).expand(1, 3, h, w).contiguous())
    cases = []
    for model, q, k, ctx in s["reuse_e2e"]:
        net = models[model]
        grid = GlimpseGrid(FRAME, q, MARGIN)
        idx = torch.arange(k, device="cuda")

        def step(frame, net=net, grid=grid, idx=idx, ctx=ctx):
            gray = to_gray_small(frame)
            H, _ = estimate_homography(prev_gray, gray, method="iclk")
            moving_cells(motion_map(prev_gray, gray, H), grid, 30, 0.05)
            lab = warp(labels[None, None].float(), H, "nearest")[0][0, 0].round().long()
            cf = warp(conf[None, None].float(), H, "nearest", frame_size=FRAME)[0][0, 0].round().to(torch.uint8)
            if ctx is not None:
                c_lab, c_conf, _ = full_pass(net, frame, ctx, small, torch.float16)
                take = c_conf >= cf
                full = take.repeat_interleave(MOTION_FACTOR, 0).repeat_interleave(MOTION_FACTOR, 1)
                lab, cf = torch.where(full, c_lab, lab), torch.where(take, c_conf, cf)
            g_lab, g_conf, _ = glimpse_pass(net, frame, grid, idx, torch.float16)
            return (paste_cells(lab, g_lab, grid.inner, grid.cols, idx),
                    paste_cells(cf, g_conf, grid.inner // MOTION_FACTOR, grid.cols, idx))

        frame = _texture(h, w, 3).expand(1, 3, h, w).contiguous()
        name = f"reuse_e2e/{model}/{q}/k{k}" + ("" if ctx is None else f"/c{ctx:g}")
        cases.append({"name": name, "model": model, "size": q, "k": k, "ctx_scale": ctx, "homography": "iclk",
                      "fn": step, "x": frame,
                      "gflops": gflops(lambda x, net=net, grid=grid, idx=idx: glimpse_pass(net, x, grid, idx, torch.float16), frame)})
    return cases


FINAL_SCALES = (0.25, 0.375, 0.5, 0.75, 1.0)  # the full-frame reference curve of method 2


def final_configs(path: Path) -> List[Dict[str, Any]]:
    """Method 2 finalists (``results/method2/finalists.json``) as pipeline configs to time.

    A ``prior`` finalist glimpses fixed cells and needs no scoring; it is timed
    with entropy top-k scoring, an upper bound, and labelled ``timed_as``.
    """
    out = []
    for f in json.loads(Path(path).read_text())["finalists"]:
        m, rule = f["method"], f["method"]["rule"]
        timed_as = None
        if rule["kind"] != "uncertainty":
            timed_as = "uncertainty/entropy/mean/topk"
            rule = {"score": "entropy", "pool": "mean", "class_weights": None, "selection": "topk", "suppress": 1}
        weights = rule.get("class_weights")
        cfg = FoveationConfig(context_scale=m["context_scale"], glimpse_size=m["glimpse_size"],
                              glimpse_margin=m["margin"], k=m["k"], selection=rule["selection"],
                              suppress=rule["suppress"], score=rule["score"], pool=rule["pool"],
                              class_weights=None if weights is None else tuple(float(w) for w in weights),
                              fusion=m.get("fusion", "replace"), glimpse_scale=m.get("glimpse_scale", 1.0))
        name = (f"final/{m['context_model']}+{m['glimpse_model']}/{m['context_scale']:g}/{m['glimpse_size']}"
                f"/z{cfg.glimpse_scale:g}/{cfg.fusion}"
                f"/k{m['k']}/{f['method']['rule']['kind']}/{rule['score']}/{rule['pool']}/{rule['selection']}"
                + ("/weighted" if weights is not None else ""))
        out.append({"name": name, "context_model": m["context_model"], "glimpse_model": m["glimpse_model"],
                    "config": cfg, "timed_as": timed_as, "finalist": f})
    return out


def fullframe_points() -> List[Tuple[str, float]]:
    return [(model, scale) for model in MODELS for scale in FINAL_SCALES]


def final_cases(models: Dict[str, torch.nn.Module], finalists: Path) -> List[Dict[str, Any]]:
    """The finalists and, interleaved in the same rounds, both models' full frame at every scale.

    Both sides run end to end from a full-resolution frame (resize, network,
    upsampling to the frame, argmax), so the comparison is like for like.
    """
    from models.utils import upsample_logits

    cases = []
    x = frame_of((1, 3, *FRAME))
    for c in final_configs(finalists):
        seg = FoveatedSegmenter(models[c["context_model"]], models[c["glimpse_model"]], c["config"], FRAME).cuda().eval()
        cases.append({"name": c["name"], "kind": "finalist", "context_model": c["context_model"],
                      "glimpse_model": c["glimpse_model"], "config": asdict(c["config"]), "timed_as": c["timed_as"],
                      "train_miou": c["finalist"].get("miou"), "predicted_median_ms": c["finalist"].get("latency_ms"),
                      "gflops": gflops(seg, x), "fn": seg, "x": x})
    for model, scale in fullframe_points():
        net = models[model]

        def full(frame, net=net, scale=scale):
            small = downsample(frame, scale)
            logits = net(small)
            return upsample_logits(logits, FRAME, net.output_stride, net.align_corners,
                                   input_size=tuple(small.shape[-2:])).argmax(1)

        cases.append({"name": f"final/fullframe/{model}/{scale:g}", "kind": "fullframe", "model": model,
                      "scale": scale, "gflops": gflops(full, x), "fn": full, "x": x})
    return cases


def compare_final(records: Dict[str, List[Dict[str, Any]]]) -> None:
    """Measured / predicted median for each finalist (prediction: the latency-floor tables)."""
    for r in records.get("final", []):
        if r.get("predicted_median_ms"):
            r["measured_over_predicted"] = r["cuda_event"]["median_ms"] / r["predicted_median_ms"]


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
            if "predicted_median_ms" in r and "measured_over_predicted" in r:
                extra = f"  predicted {r['predicted_median_ms']:7.2f}  ratio {r['measured_over_predicted']:.2f}"
            elif "predicted_ms" in r:
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
                    "overhead": lambda: overhead_cases(s), "e2e": lambda: e2e_cases(models, s),
                    "final": lambda: final_cases(models, Path(args.finalists)),
                    "batchk": lambda: batchk_cases(models, s), "reuse": lambda: reuse_cases(s),
                    "reuse_e2e": lambda: reuse_e2e_cases(models, s)}
        for part in parts:
            cases = builders[part]()
            # batchk extends the batch table; the reuse ops cannot be captured in a CUDA graph
            key, part_levels = {"batchk": ("batch", ["graph"]), "reuse": ("reuse", ["eager"]),
                                "reuse_e2e": ("reuse_e2e", ["eager"])}.get(part, (part, levels))
            for level in part_levels:
                print(f"[latency_floor] {part} / {level}: {len(cases)} configurations", flush=True)
                records.setdefault(key, []).extend(time_part(cases, level, s, tel))
            del cases
        session_tel = tel.window(float("-inf"), float("inf"))
    predict_e2e(records)
    compare_final(records)
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
