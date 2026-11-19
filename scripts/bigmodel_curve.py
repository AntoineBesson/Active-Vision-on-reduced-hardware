#!/usr/bin/env python3
"""Full-frame curve of larger frozen models (SegFormer) against PIDNet-S and BiSeNetV2.

The first check before any warp on a larger model: does a larger model, run
on a small input, come near the small models' full-frame curve at the same
measured T4 latency, and is its own curve steep there? Only then is there room
for a warp (Method 5/6 oracle bound) to lift it above every model's curve. The
rule stays accuracy at equal latency, against all models.

    # 1. latency on the T4 (random weights, fp16, CUDA graphs, all models interleaved)
    python -m scripts.bigmodel_curve latency --session s1
    # 2. accuracy on VPS val (real weights, fp32): one run_baseline per model, e.g.
    python -m scripts.run_baseline --config configs/baseline_fullframe_segformer.yaml
    # 3. the table and the screen
    python -m scripts.bigmodel_curve report \\
        --latency 'experiments/bigmodel/latency_*.json' --accuracy 'experiments/*_fullframe_*/*.json'

``latency`` times two parts for every model and scale, interleaved in rounds
with the protocol of ``eval.gpu_timing`` (as ``scripts/latency_floor.py``):

* ``area``: the backbone alone on the resized frame (the primary curve of
  Methods 2 to 5);
* ``e2e``: from the full 1024 x 2048 frame: antialiased resize, backbone,
  upsampling of the logits to the frame and argmax (a model's own output
  stride changes this part: 8 PIDNet-S, 1 BiSeNetV2, 4 SegFormer).

``report`` takes the median over sessions of each point, joins the VPS-val
mIoU of the ``run_baseline`` JSONs, and for every large-model point reads the
best small model's mIoU at the same latency (linear between that model's two
bracketing measured scales, never extrapolated). The screen (not a pass rule)
is GO when some large-model scale below 1 is within ``--gap`` points (2.0) of
that best small model AND its own curve is steep there: the next larger scale
gains at least ``--steep`` points (1.0). It is read on ``e2e`` and on
``area``; both are reported.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/bigmodel_curve.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path  # noqa: E402

ensure_src_on_path()

FRAME = (1024, 2048)
SMALL = ("pidnet_s", "bisenetv2")
LARGE_DEFAULT = ("segformer_b0", "segformer_b2", "segformer_b5")
SCALES = (0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0)
PARTS = ("area", "e2e")
FULL = dict(rounds=5, iters=40, warmup=20)
QUICK = dict(rounds=2, iters=5, warmup=3)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    lat = sub.add_parser("latency", help="time every model at every scale on this GPU")
    lat.add_argument("--session", required=True, help="Session tag (one per separate GPU session).")
    lat.add_argument("--models", default=",".join(SMALL + LARGE_DEFAULT))
    lat.add_argument("--scales", default=",".join(f"{s:g}" for s in SCALES))
    lat.add_argument("--parts", default=",".join(PARTS))
    lat.add_argument("--levels", default="graph", help="Comma list: graph (the reference level), eager.")
    lat.add_argument("--quick", action="store_true", help="Few iterations (smoke test).")
    lat.add_argument("--out", default=str(REPO_ROOT / "experiments" / "bigmodel"))
    rep = sub.add_parser("report", help="join latency and accuracy, read the screen")
    rep.add_argument("--latency", nargs="+", required=True, help="latency JSONs (globs allowed)")
    rep.add_argument("--accuracy", nargs="+", required=True, help="run_baseline JSONs (globs allowed)")
    rep.add_argument("--level", default="graph")
    rep.add_argument("--gap", type=float, default=2.0, help="max mIoU points below the best small model")
    rep.add_argument("--steep", type=float, default=1.0, help="min mIoU points gained at the next larger scale")
    rep.add_argument("--out", default=str(REPO_ROOT / "experiments" / "bigmodel" / "report.json"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------- latency

def build(kind: str):
    from models.registry import build_backbone

    return build_backbone({"type": kind, "num_classes": 19, "allow_random_init": True}).cuda().eval().half()


def input_shape(scale: float) -> Tuple[int, int]:
    return max(1, round(FRAME[0] * scale)), max(1, round(FRAME[1] * scale))


def cases_for(name: str, net, scales: Sequence[float], parts: Sequence[str]) -> List[Dict[str, Any]]:
    import torch

    from foveation.ops import downsample
    from models.utils import upsample_logits
    from scripts.latency_floor import frame_of, gflops

    out = []
    full_frame = frame_of((1, 3, *FRAME))
    for scale in scales:
        h, w = input_shape(scale)
        if "area" in parts:
            x = frame_of((1, 3, h, w))
            out.append({"name": f"area/{name}/{scale:g}", "part": "area", "model": name, "scale": scale,
                        "h": h, "w": w, "gflops": gflops(net, x), "fn": net, "x": x})
        if "e2e" in parts:
            def full(frame, net=net, scale=scale):
                small = downsample(frame, scale)
                logits = net(small)
                return upsample_logits(logits, FRAME, net.output_stride, net.align_corners,
                                       input_size=tuple(small.shape[-2:])).argmax(1)

            out.append({"name": f"e2e/{name}/{scale:g}", "part": "e2e", "model": name, "scale": scale,
                        "h": h, "w": w, "gflops": gflops(full, full_frame), "fn": full, "x": full_frame})
    torch.cuda.synchronize()
    return out


def run_latency(args: argparse.Namespace) -> int:
    import torch

    from eval.gpu_timing import GpuTelemetry, gpu_environment
    from scripts.latency_floor import time_part

    if not torch.cuda.is_available():
        print("bigmodel_curve latency needs a CUDA GPU", file=sys.stderr)
        return 2
    s = QUICK if args.quick else FULL
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scales = [float(v) for v in args.scales.split(",") if v.strip()]
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    torch.backends.cudnn.benchmark = True
    nets = {m: build(m) for m in models}
    cases = [c for m in models for c in cases_for(m, nets[m], scales, parts)]
    started = _dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    with GpuTelemetry() as tel:
        for level in levels:
            print(f"[bigmodel_curve] {level}: {len(cases)} configurations", flush=True)
            try:
                records.extend(time_part(cases, level, s, tel))
            except RuntimeError as exc:  # e.g. a model that cannot be captured in a CUDA graph
                print(f"[bigmodel_curve] {level} failed together; timing one configuration at a time: {exc}",
                      flush=True)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                for c in cases:
                    try:
                        records.extend(time_part([c], level, s, tel))
                    except RuntimeError as one:
                        failures.append({"name": c["name"], "level": level, "error": str(one)[:500]})
                        torch.cuda.empty_cache()
        session_tel = tel.window(float("-inf"), float("inf"))
    out = {
        "experiment": "bigmodel_curve",
        "session": args.session,
        "started": started,
        "duration_s": time.time() - t0,
        "quick": args.quick,
        "settings": {**s, "frame": list(FRAME), "models": models, "scales": scales, "parts": parts, "levels": levels,
                     "dtype": "float16", "batch": "1 frame",
                     "timing": "per-frame CUDA events after synchronize; configurations interleaved in rounds",
                     "weights": "random (latency does not depend on values)"},
        "environment": gpu_environment(),
        "session_telemetry": session_tel,
        "records": records,
        "failures": failures,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"latency_{args.session}.json"
    path.write_text(json.dumps(out, indent=1))
    for r in records:
        print(f"{r['level']:5s} {r['name']:28s} {r['cuda_event']['median_ms']:8.2f} ms median"
              f"  p95 {r['cuda_event']['p95_ms']:8.2f}")
    for f in failures:
        print(f"FAILED {f['level']} {f['name']}: {f['error'][:120]}")
    print(f"\n[bigmodel_curve] {time.time() - t0:.0f} s, wrote {path}")
    return 0


# --------------------------------------------------------------------------- report (numpy-free, pure python)

def _expand(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else ([p] if Path(p).is_file() else []))
    return sorted(set(out))


def _median(values: Sequence[float]) -> float:
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def load_latency(paths: Sequence[str], level: str) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    """(part, model, scale) -> median over sessions of the session medians (and of p95)."""
    per: Dict[Tuple[str, str, float], Dict[str, List[float]]] = {}
    sessions: Dict[Tuple[str, str, float], List[str]] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if data.get("experiment") != "bigmodel_curve":
            raise SystemExit(f"{path} is not a bigmodel_curve latency file")
        for r in data["records"]:
            if r["level"] != level:
                continue
            key = (r["part"], r["model"], float(r["scale"]))
            d = per.setdefault(key, {"median": [], "p95": []})
            d["median"].append(float(r["cuda_event"]["median_ms"]))
            d["p95"].append(float(r["cuda_event"]["p95_ms"]))
            sessions.setdefault(key, []).append(str(data.get("session")))
    return {k: {"median_ms": _median(v["median"]), "p95_ms": _median(v["p95"]), "sessions": sessions[k],
                "session_medians_ms": v["median"]} for k, v in per.items()}


def load_accuracy(paths: Sequence[str]) -> Dict[Tuple[str, float], Dict[str, Any]]:
    """(model, scale) -> mIoU (points) of a run_baseline JSON; refuses two runs of one point."""
    out: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for path in paths:
        rec = json.loads(Path(path).read_text())
        if "metrics" not in rec or "method" not in rec or "backbone" not in rec:
            continue  # not a run record (e.g. a report)
        if rec.get("dataset", {}).get("synthetic_fixture"):
            raise SystemExit(f"{path} was run on the synthetic fixture, not real data")
        key = (str(rec["backbone"]["type"]), float(rec["method"]["scale"]))
        if key in out:
            raise SystemExit(f"two accuracy runs for {key}: {out[key]['file']} and {path}")
        out[key] = {"miou": 100.0 * float(rec["metrics"]["miou"]), "file": str(path),
                    "split": rec.get("dataset", {}).get("split"),
                    "clips": rec.get("dataset", {}).get("clips_evaluated")}
    return out


def curve(points: Dict[Tuple[str, str, float], Dict[str, Any]], acc: Dict[Tuple[str, float], Dict[str, Any]],
          part: str, model: str) -> List[Tuple[float, float, float]]:
    """Sorted (latency ms, mIoU, scale) of one model, only scales with both numbers."""
    pts = [(v["median_ms"], acc[(m, s)]["miou"], s) for (p, m, s), v in points.items()
           if p == part and m == model and (m, s) in acc]
    return sorted(pts)


def miou_at(pts: Sequence[Tuple[float, float, float]], ms: float) -> Optional[Dict[str, Any]]:
    """mIoU of a full-frame curve at ``ms``: linear between the two measured points around it.

    Outside the measured latency range: None (no extrapolation). A curve whose
    latency does not rise with scale is read on its upper envelope (a point
    that is slower and less accurate than another is ignored).
    """
    env = []
    for lat, m, s in sorted(pts):
        if env and m <= env[-1][1]:
            continue  # dominated: slower and not more accurate
        env.append((lat, m, s))
    if not env or ms < env[0][0]:
        return None
    if ms >= env[-1][0]:
        return {"miou": env[-1][1], "between": [env[-1][2], None], "note": "above the curve's slowest point: its mIoU"}
    for (l0, m0, s0), (l1, m1, s1) in zip(env, env[1:]):
        if l0 <= ms <= l1:
            t = 0.0 if l1 == l0 else (ms - l0) / (l1 - l0)
            return {"miou": m0 + t * (m1 - m0), "between": [s0, s1]}
    return None  # pragma: no cover


def report(lat: Dict[Tuple[str, str, float], Dict[str, Any]], acc: Dict[Tuple[str, float], Dict[str, Any]],
           gap: float = 2.0, steep: float = 1.0) -> Dict[str, Any]:
    models = sorted({m for (_, m, _) in lat})
    small = [m for m in SMALL if m in models]
    large = [m for m in models if m not in SMALL]
    out: Dict[str, Any] = {"small": small, "large": large, "gap": gap, "steep": steep, "parts": {}}
    for part in sorted({p for (p, _, _) in lat}):
        curves = {m: curve(lat, acc, part, m) for m in models}
        rows = []
        go = []
        for m in large:
            pts = curves[m]
            by_scale = {s: (l, mi) for l, mi, s in pts}
            for l, mi, s in pts:
                best = None
                for sm in small:
                    r = miou_at(curves[sm], l)
                    if r is not None and (best is None or r["miou"] > best["miou"]):
                        best = {**r, "model": sm}
                larger = sorted(x for x in by_scale if x > s)
                gain = by_scale[larger[0]][1] - mi if larger else None
                row = {"model": m, "scale": s, "latency_ms": l, "miou": mi,
                       "best_small": best, "delta_vs_best_small": None if best is None else mi - best["miou"],
                       "gain_to_next_scale": gain, "next_scale": larger[0] if larger else None}
                row["screen_go"] = bool(s < 1.0 and best is not None and row["delta_vs_best_small"] >= -gap
                                        and gain is not None and gain >= steep)
                if row["screen_go"]:
                    go.append(f"{m}@{s:g}")
                rows.append(row)
        out["parts"][part] = {
            "curves": {m: [{"latency_ms": l, "miou": mi, "scale": s} for l, mi, s in curves[m]] for m in models},
            "large_points": rows,
            "screen": "GO" if go else "STOP",
            "go_points": go,
        }
    missing = sorted(f"{m}@{s:g}" for (_, m, s) in lat if (m, s) not in acc)
    out["missing_accuracy"] = sorted(set(missing))
    return out


def print_report(rep: Dict[str, Any]) -> None:
    for part, r in rep["parts"].items():
        print(f"\n== {part}: screen {r['screen']} {r['go_points'] or ''}")
        for m, pts in r["curves"].items():
            print(f"  {m:14s} " + "  ".join(f"{p['scale']:g}: {p['latency_ms']:.2f} ms {p['miou']:.1f}" for p in pts))
        print(f"  {'large model':14s} {'scale':>6s} {'ms':>8s} {'mIoU':>6s} {'best small':>16s} {'delta':>6s} {'+next':>6s}")
        for row in r["large_points"]:
            b = row["best_small"]
            bs = "-" if b is None else f"{b['miou']:.1f} {b['model']}"
            d = "-" if row["delta_vs_best_small"] is None else f"{row['delta_vs_best_small']:+.1f}"
            g = "-" if row["gain_to_next_scale"] is None else f"{row['gain_to_next_scale']:+.1f}"
            print(f"  {row['model']:14s} {row['scale']:6g} {row['latency_ms']:8.2f} {row['miou']:6.1f} {bs:>16s} "
                  f"{d:>6s} {g:>6s}{'  GO' if row['screen_go'] else ''}")
    if rep["missing_accuracy"]:
        print("\nno accuracy for:", ", ".join(rep["missing_accuracy"]))


def run_report(args: argparse.Namespace) -> int:
    lat_paths, acc_paths = _expand(args.latency), _expand(args.accuracy)
    if not lat_paths or not acc_paths:
        raise SystemExit(f"no files: latency {args.latency} -> {lat_paths}, accuracy {args.accuracy} -> {acc_paths}")
    rep = report(load_latency(lat_paths, args.level), load_accuracy(acc_paths), gap=args.gap, steep=args.steep)
    rep.update({"level": args.level, "latency_files": lat_paths, "accuracy_files": acc_paths})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print_report(rep)
    print(f"\n[bigmodel_curve] wrote {out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_latency(args) if args.cmd == "latency" else run_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
