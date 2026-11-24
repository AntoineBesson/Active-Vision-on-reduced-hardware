#!/usr/bin/env python3
"""Method 9, gate 0: is the slow branch a large enough share of the T4 latency to win anything?

Spec: docs/superpowers/specs/2026-11-23-method9-deep-feature-reuse-design.md.

    # 1. T4 timing (random weights, fp16, CUDA graphs, all cases interleaved; ~10 min)
    python -m scripts.method9 gate0 --session s1
    # 2. the ceiling (pure python): reads the timing JSON(s) and the train_all full-frame curves
    python -m scripts.method9 ceiling --timing 'experiments/method9/gate0_*.json'

``gate0`` times, end to end from the 1024 x 2048 frame (resize, network, upsampling, argmax):

* ``ff``: each model's full frame at the 11 scales of the Methods 7/8 envelope (so the envelope
  is priced in the same session as the method);
* ``key``: the key frame at each gate scale (the full model, also producing the cache);
* ``fresh``: a non-key frame (fresh branch + cached slow tensors, held as static inputs);
* ``slow``: the slow part alone (information only).

``ceiling``: per model and gate scale s, the schedule with interval K costs on average
``(T_key + (K - 1) T_fresh) / K``. Assuming stale context is never better than fresh (accuracy at
most the model's own full frame at s), the most it can gain is
``FF_model(s) - Envelope(mean latency)``, the envelope being the higher of both models' train_all
full-frame curves at that latency. Gate 0 passes for a model if that is >= +1.0 point at K -> inf
(``T_fresh``) for some s. K = 2..5 are reported too.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method9.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path  # noqa: E402

ensure_src_on_path()

FRAME = (1024, 2048)
MODELS = ("pidnet_s", "bisenetv2")
GATE_SCALES = (0.5, 1.0)
ENVELOPE_SCALES = (0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1.0)
KS = (2, 3, 4, 5)
MARGIN = 1.0  # mIoU points
CURVES = REPO_ROOT / "results" / "method9" / "fullframe_train_all.json"
FULL = dict(rounds=5, iters=40, warmup=20)
QUICK = dict(rounds=2, iters=5, warmup=3)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate0", help="T4 timing of key, fresh and full-frame passes")
    g.add_argument("--session", required=True)
    g.add_argument("--models", default=",".join(MODELS))
    g.add_argument("--scales", default=",".join(f"{s:g}" for s in GATE_SCALES))
    g.add_argument("--quick", action="store_true", help="few iterations (smoke test)")
    g.add_argument("--out", default=str(REPO_ROOT / "experiments" / "method9"))
    c = sub.add_parser("ceiling", help="gate 0 verdict from the timing JSONs")
    c.add_argument("--timing", nargs="+", required=True, help="gate0 JSONs (globs allowed)")
    c.add_argument("--curves", default=str(CURVES))
    c.add_argument("--level", default="graph")
    c.add_argument("--out", default=None, help="write the report here (JSON)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- timing (GPU)

def input_shape(scale: float) -> Tuple[int, int]:
    return round(FRAME[0] * scale), round(FRAME[1] * scale)


def gate0_cases(name: str, net, gate_scales: Sequence[float]) -> List[Dict[str, Any]]:
    import torch

    from foveation.ops import downsample
    from models.utils import upsample_logits
    from reuse.slowfast import slowfast_split
    from scripts.latency_floor import frame_of, gflops

    split = slowfast_split(net)
    frame = frame_of((1, 3, *FRAME))

    def e2e(body, scale):
        def fn(f):
            small = downsample(f, scale)
            logits = body(small)
            return upsample_logits(logits, FRAME, net.output_stride, net.align_corners,
                                   input_size=tuple(small.shape[-2:])).argmax(1)
        return fn

    out = []
    for s in ENVELOPE_SCALES:
        fn = e2e(net, s)
        out.append({"name": f"ff/{name}/{s:g}", "kind": "ff", "model": name, "scale": s,
                    "gflops": gflops(fn, frame), "fn": fn, "x": frame})
    for s in gate_scales:
        h, w = input_shape(s)
        with torch.no_grad():  # the cache of another frame, held as static inputs of the fresh pass
            other = torch.rand((1, 3, h, w), generator=torch.Generator().manual_seed(1)).to("cuda", torch.float16)
            cache = tuple(t.clone() for t in split.slow(other))
        key = e2e(lambda x: split.key(x)[0], s)
        fresh = e2e(lambda x, cache=cache: split.fresh(x, cache), s)
        small = frame_of((1, 3, h, w))
        out += [
            {"name": f"key/{name}/{s:g}", "kind": "key", "model": name, "scale": s,
             "gflops": gflops(key, frame), "fn": key, "x": frame},
            {"name": f"fresh/{name}/{s:g}", "kind": "fresh", "model": name, "scale": s,
             "gflops": gflops(fresh, frame), "fn": fresh, "x": frame,
             "cache_shapes": [list(t.shape) for t in cache]},
            {"name": f"slow/{name}/{s:g}", "kind": "slow", "model": name, "scale": s,
             "gflops": gflops(split.slow, small), "fn": split.slow, "x": small},
        ]
    torch.cuda.synchronize()
    return out


def run_gate0(args: argparse.Namespace) -> int:
    import torch

    from eval.gpu_timing import GpuTelemetry, gpu_environment
    from scripts.bigmodel_curve import build
    from scripts.latency_floor import time_part

    if not torch.cuda.is_available():
        print("method9 gate0 needs a CUDA GPU", file=sys.stderr)
        return 2
    s = QUICK if args.quick else FULL
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scales = [float(v) for v in args.scales.split(",") if v.strip()]
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    started = _dt.datetime.now().isoformat(timespec="seconds")
    with torch.no_grad():
        nets = {m: build(m) for m in models}
        cases = [c for m in models for c in gate0_cases(m, nets[m], scales)]
        print(f"[method9 gate0] {len(cases)} configurations, interleaved", flush=True)
        with GpuTelemetry() as tel:
            records = time_part(cases, "graph", s, tel)
            session_tel = tel.window(float("-inf"), float("inf"))
    out = {
        "experiment": "method9_gate0", "session": args.session, "started": started,
        "duration_s": time.time() - t0, "quick": args.quick,
        "settings": {**s, "frame": list(FRAME), "models": models, "gate_scales": scales,
                     "envelope_scales": list(ENVELOPE_SCALES), "dtype": "float16", "batch": "1 frame",
                     "level": "graph", "weights": "random (latency does not depend on values)"},
        "environment": gpu_environment(), "session_telemetry": session_tel, "records": records,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gate0_{args.session}.json"
    path.write_text(json.dumps(out, indent=1))
    for r in records:
        print(f"{r['name']:24s} {r['cuda_event']['median_ms']:8.2f} ms median  p95 {r['cuda_event']['p95_ms']:8.2f}"
              f"  {r.get('gflops') or float('nan'):7.1f} GFLOPs")
    print(f"\n[method9 gate0] {time.time() - t0:.0f} s, wrote {path}")
    return 0


# --------------------------------------------------------------------------- ceiling (pure python)

def _expand(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        out.extend(sorted(glob.glob(p)) or ([p] if Path(p).is_file() else []))
    return sorted(set(out))


def load_timing(paths: Sequence[str], level: str = "graph") -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    """{(kind, model, scale): median over sessions of the per-session medians}."""
    by: Dict[Tuple[str, str, float], List[float]] = {}
    sessions: Dict[Tuple[str, str, float], List[str]] = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        if d.get("quick"):
            raise SystemExit(f"{p} is a --quick smoke run; its timings are not a measurement")
        for r in d["records"]:
            if r.get("level", "graph") != level:
                continue
            k = (r["kind"], r["model"], float(r["scale"]))
            by.setdefault(k, []).append(r["cuda_event"]["median_ms"])
            sessions.setdefault(k, []).append(d.get("session", Path(p).stem))
    return {k: {"median_ms": statistics.median(v), "sessions": sessions[k]} for k, v in by.items()}


def curve_at(points: Sequence[Tuple[float, float]], at: float) -> Optional[float]:
    """Piecewise-linear (latency, mIoU) curve at ``at``; None outside the measured latencies."""
    pts = sorted(points)
    if not pts or not pts[0][0] - 1e-9 <= at <= pts[-1][0] + 1e-9:
        return None
    for (l0, m0), (l1, m1) in zip(pts, pts[1:]):
        if l0 <= at <= l1:
            return m0 if l1 == l0 else m0 + (at - l0) / (l1 - l0) * (m1 - m0)
    return pts[-1][1] if abs(at - pts[-1][0]) <= 1e-9 else pts[0][1]


def envelope_at(curves: Dict[str, Sequence[Tuple[float, float]]], at: float) -> Tuple[Optional[float], Optional[str]]:
    """The higher of the models' curves at ``at`` (a model whose curve does not reach it is left out,
    as in Method 7)."""
    vals = {m: curve_at(c, at) for m, c in curves.items()}
    vals = {m: v for m, v in vals.items() if v is not None}
    if not vals:
        return None, None
    m = max(vals, key=vals.get)
    return vals[m], m


def ceiling_report(timing: Dict[Tuple[str, str, float], Dict[str, Any]], curves_file: Dict[str, Any],
                   ks: Sequence[int] = KS, margin: float = MARGIN) -> Dict[str, Any]:
    acc = {m: dict(zip(v["scales"], (100 * u for u in v["miou"]))) for m, v in curves_file["models"].items()}
    models = sorted({m for (_, m, _) in timing} & set(acc))
    curves, missing = {}, []
    for m in models:
        pts = []
        for s, miou in acc[m].items():
            t = timing.get(("ff", m, float(s)))
            if t is None:
                missing.append(f"ff/{m}/{s:g}")
            else:
                pts.append((t["median_ms"], miou))
        curves[m] = pts
    rep: Dict[str, Any] = {"margin": margin, "ks": list(ks), "missing_ff": missing, "models": {}}
    for m in models:
        rows = []
        for (kind, mm, s), t in sorted(timing.items()):
            if kind != "fresh" or mm != m:
                continue
            key = timing.get(("key", m, s))
            ff = timing.get(("ff", m, s))
            if key is None or s not in acc[m]:
                continue
            t_key, t_fresh = key["median_ms"], t["median_ms"]
            row = {"scale": s, "ff_miou": acc[m][s], "t_ff_ms": ff["median_ms"] if ff else None,
                   "t_key_ms": t_key, "t_fresh_ms": t_fresh, "fresh_share": t_fresh / t_key,
                   "t_slow_ms": timing.get(("slow", m, s), {}).get("median_ms"), "schedules": []}
            for k in [*ks, None]:
                mean = t_fresh if k is None else (t_key + (k - 1) * t_fresh) / k
                env, by = envelope_at(curves, mean)
                row["schedules"].append({"K": k if k is not None else "inf", "mean_ms": mean, "envelope": env,
                                         "envelope_model": by,
                                         "ceiling": None if env is None else acc[m][s] - env})
            inf = row["schedules"][-1]
            # below every measured curve: nothing on the envelope is that fast, so the bound is unbounded
            row["gate0_pass"] = inf["envelope"] is None or inf["ceiling"] >= margin - 1e-9
            rows.append(row)
        rep["models"][m] = {"rows": rows, "gate0_pass": any(r["gate0_pass"] for r in rows)}
    rep["continue"] = [m for m, v in rep["models"].items() if v["gate0_pass"]]
    return rep


def print_report(rep: Dict[str, Any]) -> None:
    for m, v in rep["models"].items():
        print(f"\n== {m}: gate 0 {'PASSES' if v['gate0_pass'] else 'FAILS'} (ceiling >= +{rep['margin']:.1f} at K=inf)")
        for r in v["rows"]:
            print(f"  s={r['scale']:g}: FF {r['ff_miou']:.2f} mIoU; key {r['t_key_ms']:.2f} ms, fresh "
                  f"{r['t_fresh_ms']:.2f} ms ({100 * r['fresh_share']:.0f}% of key)")
            for sc in r["schedules"]:
                env = "below every curve" if sc["envelope"] is None else \
                    f"envelope {sc['envelope']:.2f} ({sc['envelope_model']}) -> ceiling {sc['ceiling']:+.2f}"
                print(f"     K={str(sc['K']):>3s}: mean {sc['mean_ms']:6.2f} ms, {env}")
    if rep["missing_ff"]:
        print("\nmissing full-frame timings:", ", ".join(rep["missing_ff"]))
    print(f"\n[method9 gate0] continue: {rep['continue'] or 'none (Method 9 stops)'}")


def run_ceiling(args: argparse.Namespace) -> int:
    paths = _expand(args.timing)
    if not paths:
        raise SystemExit(f"no timing files match {args.timing}")
    rep = ceiling_report(load_timing(paths, args.level), json.loads(Path(args.curves).read_text()))
    rep["timing_files"] = paths
    print_report(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1))
        print("wrote", args.out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_gate0(args) if args.cmd == "gate0" else run_ceiling(args)


if __name__ == "__main__":
    raise SystemExit(main())
