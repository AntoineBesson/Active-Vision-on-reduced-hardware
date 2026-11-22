#!/usr/bin/env python3
"""Methods 7 and 8: guided upsampling of low-scale logits, and the per-frame scale oracle.

    python -m scripts.method7 run --split train_all --models pidnet_s --chunk 0/4   # network passes, per clip (GPU)
    python -m scripts.method7 merge --split train_all                               # chunk files -> one per model
    python -m scripts.method7 score --split train_all                               # Method 7: readings, gate 1, finalists
    python -m scripts.method7 oracle --split train_all                              # Method 8: the per-frame scale oracle
    python -m scripts.method7 screen --split train_all                              # Method 8 step 2: predictability
    python -m scripts.method7 objects --split train_all                             # moving-object recall (train_all)
    python -m scripts.method7 time --session s1                                     # T4 (Colab): dense full frame + finalists

Spec (grids, prices and pass rules, fixed before any run):
docs/superpowers/specs/2026-11-20-method7-8-guided-upsampling-scale-oracle-design.md

``run`` makes, on every labelled frame of each clip, one pass of the frozen model at each scale
of the dense full-frame menu (:data:`FF_SCALES`). Each pass is scored with the full frame's own
bilinear step (the reference; per-frame confusions are kept for Method 8) and, at the Method 7
scales, with every guided configuration (clip-summed confusions). ``--obj`` also records the
local IoU of every thing object (``eval.objects``) for the moving-object metric.
``--fixture`` runs on the synthetic fixture with the dummy backbone (a smoke test).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method7.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _floats, _strs, load_dataset, model_config  # noqa: E402
from scripts.method3 import _tag, select_clips  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method7"
RES = REPO_ROOT / "results" / "method7"
FRAME = (1024, 2048)  # the frame the T4 latency tables are measured at
NUM_CLASSES, IGNORE = 19, 255

# The spec's grids.
FF_SCALES = (0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1.0)
GUIDED_SCALES = (0.25, 0.375, 0.5, 0.625)
GUIDES = ("gray", "rgb")
TARGETS = ("logit", "prob")
RADII = (1, 2)
EPSS = (1e-4, 1e-3, 1e-2)
# Method 7, gate 1 (train, zero overhead) and the val pass rule.
GATE1_MARGIN = 0.005  # +0.5 mIoU point above the envelope at equal latency
MAX_FINALISTS = 2  # per model
# Method 8: budgets (the same model's full frame at these scales) and the bound's pass margin.
ORACLE_BUDGETS = (0.5, 0.625, 0.75)
ORACLE_MARGIN = 0.010  # +1.0 mIoU point above the envelope at equal mean latency
DEFAULT_E2E = "results/latency_floor/latency_floor_final-s*.json"
FIXTURE_MODELS = ("dummy",)


def _out_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if getattr(args, "out_dir", None) else EXP / args.split


def ff_name(scale: float) -> str:
    return f"ff/{scale:g}"


def guided_grid(scales: Sequence[float] = GUIDED_SCALES, guides: Sequence[str] = GUIDES,
                targets: Sequence[str] = TARGETS, radii: Sequence[int] = RADII, epss: Sequence[float] = EPSS):
    from upsample.guided import GuidedConfig

    return [GuidedConfig(c, g, t, r, e) for c, g, t, r, e in itertools.product(scales, guides, targets, radii, epss)]


def read_lines(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _append(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def _confusion(label, pred):
    import torch

    ok = label != IGNORE
    return torch.bincount(label[ok].long() * NUM_CLASSES + pred[ok].long(),
                          minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def wait_for_gpu(max_temp: Optional[int], resume_temp: int = 65, pause_s: float = 20.0) -> None:
    """Laptop rule: pause at >= max_temp (two readings) or any thermal slowdown, resume at <= resume_temp."""
    if not max_temp:
        return
    import subprocess

    def read() -> Tuple[int, bool]:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_throttle_reasons.hw_thermal_slowdown,"
                              "clocks_throttle_reasons.sw_thermal_slowdown", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=False).stdout.strip().split(",")
        try:
            return int(out[0]), any("Active" in v for v in out[1:])
        except (ValueError, IndexError):
            return 0, False

    hot = 0
    while True:
        t, slow = read()
        hot = hot + 1 if t >= max_temp else 0
        if hot < 2 and not slow:
            return
        print(f"[method7] GPU at {t} C (thermal slowdown: {slow}): pausing until <= {resume_temp} C", flush=True)
        while read()[0] > resume_temp:
            time.sleep(pause_s)
        hot = 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run_clip(dataset, index: int, backbone, ff_scales: Sequence[float], cfgs: Sequence[Any], device, obj: str,
             allow_any_frame_size: bool = False) -> Dict[str, Any]:
    """One clip: per labelled frame, one pass per full-frame scale; the bilinear reference per frame,
    the guided configurations of that scale summed over the clip; object IoUs on request."""
    import numpy as np
    import torch

    from eval.harness import FullFrameMethod, frame_to_tensor
    from upsample.guided import bilinear_labels, guided_labels

    labelled = [int(t) for t in np.flatnonzero(np.asarray(dataset.labeled_mask(index), dtype=bool))]
    scales = sorted(set(ff_scales) | {c.scale for c in cfgs}, reverse=True)
    by_scale: Dict[float, List[Any]] = {}
    for c in cfgs:
        by_scale.setdefault(c.scale, []).append(c)
    ff = {ff_name(s): [] for s in ff_scales}
    guided = {c.name(): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for c in cfgs}
    ious: Dict[str, List[float]] = {}
    gt = None
    if obj != "none":
        from scripts.objects import clip_ground_truth

        gt = clip_ground_truth(dataset, index, labelled)
    with torch.no_grad():
        for n, t in enumerate(labelled):
            frame = frame_to_tensor(dataset.load_frame(index, t), device)
            size = tuple(frame.shape[-2:])
            if not allow_any_frame_size and size != FRAME:
                raise ValueError(f"frame is {size}; the T4 latency tables are for {FRAME}")
            label = torch.from_numpy(np.asarray(dataset.load_label(index, t), dtype=np.int64)).to(device)

            def score(name: str, pred) -> None:
                if gt is None or (obj == "ff" and not name.startswith("ff/")):
                    return
                from eval.objects import local_ious

                inst, fo = gt[0][n], gt[1][n]
                ious.setdefault(name, []).extend(local_ious(inst, pred.to(torch.uint8).cpu().numpy(), fo).tolist())

            for s in scales:
                x = FullFrameMethod(backbone, scale=s)._resize(frame)
                logits = backbone(x)
                geo = (backbone.output_stride, backbone.align_corners, tuple(x.shape[-2:]))
                if s in ff_scales:
                    pred = bilinear_labels(logits, size, *geo)
                    ff[ff_name(s)].append(_confusion(label, pred).cpu().numpy().tolist())
                    score(ff_name(s), pred)
                for c in by_scale.get(s, []):
                    pred = guided_labels(logits, frame, c, *geo)
                    guided[c.name()] += _confusion(label, pred)
                    score(c.name(), pred)
    return {"frames": labelled, "ff": ff, "guided": {k: v.cpu().numpy().tolist() for k, v in guided.items()},
            "iou": ious}


def _done(path: Path) -> set:
    return {r["clip_id"] for r in read_lines(path)} if path.is_file() else set()


def cmd_run(args: argparse.Namespace) -> int:
    from models.registry import build_backbone
    from upsample.guided import GuidedConfig

    device = resolve_device(args.device)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
    clips = select_clips([dataset.clip_id(i) for i in range(len(dataset))], args.clip_ids, args.chunk)
    ff_scales = [] if args.ff_scales == "none" else _floats(args.ff_scales)
    if args.configs:
        cfgs = [GuidedConfig.from_name(n) for n in _strs(args.configs)]
    elif args.guided_scales == "none":
        cfgs = []
    else:
        cfgs = guided_grid(_floats(args.guided_scales))
    for model in _strs(args.models):
        path = _out_dir(args) / f"run_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        done = _done(path)
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method7 run] {model}: {len(ff_scales)} full-frame scales, {len(cfgs)} guided configs, obj={args.obj}; "
              f"{len(todo)} clips to run, {len(clips) - len(todo)} already in {path.name}", flush=True)
        if not todo:
            continue
        backbone = build_backbone(model_config(model, args.fixture)["backbone"],
                                  context={"palette": dataset.palette}).to(device).eval()
        t0 = time.time()
        for n, i in enumerate(todo):
            wait_for_gpu(args.max_temp)
            out = run_clip(dataset, i, backbone, ff_scales, cfgs, device, args.obj, allow_any_frame_size=args.fixture)
            _append(path, [{"split": args.split, "model": model, "clip_id": dataset.clip_id(i), **out}])
            dt = time.time() - t0
            print(f"[method7 run] {model} clip {n + 1}/{len(todo)} ({dt:.0f} s, {dt / (n + 1):.1f} s/clip)", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args: argparse.Namespace) -> int:
    """Chunk files of a model -> run_<split>_<model>.jsonl.gz. Lines of one clip in several files
    (e.g. a finalists' objects run next to the main run) are merged key by key."""
    out_dir = _out_dir(args)
    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    for model in _strs(args.models):
        merged = out_dir / f"run_{args.split}_{model}.jsonl.gz"
        files = sorted(f for f in out_dir.glob(f"run_{args.split}_{model}_*.jsonl.gz") if f != merged)
        if not files:
            print(f"[method7 merge] {model}: no chunk file in {out_dir}")
            continue
        clips: Dict[str, Dict[str, Any]] = {}
        for f in files:
            for r in read_lines(f):
                c = clips.setdefault(r["clip_id"], {k: r[k] for k in ("split", "model", "clip_id", "frames")})
                if c["frames"] != r["frames"]:
                    raise SystemExit(f"{f}: clip {r['clip_id']}: labelled frames differ between files")
                for key in ("ff", "guided", "iou"):
                    for name, v in r.get(key, {}).items():
                        if name in c.setdefault(key, {}):
                            raise SystemExit(f"{f}: clip {r['clip_id']}: {key} {name} seen twice")
                        c[key][name] = v
        tmp = merged.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        _append(tmp, (clips[k] for k in sorted(clips)))
        tmp.replace(merged)
        missing = sorted((expected or set()) - set(clips))
        print(f"[method7 merge] {model}: {len(clips)} clips from {len(files)} files -> {merged}"
              + (f"; missing {len(missing)} clips, e.g. {missing[:3]}" if missing else ""))
    return 0


# --------------------------------------------------------------------------- #
# loading and prices
# --------------------------------------------------------------------------- #
def load_run(path: Path) -> Dict[str, Any]:
    """Per model run file -> arrays: clip ids, ff (clips, S, C, C) clip sums, ff_frames (F, S, C, C)
    with frame_clip (F,), guided (clips, G, C, C), and the row names."""
    import numpy as np

    rows = sorted(read_lines(path), key=lambda r: r["clip_id"])
    if not rows:
        raise SystemExit(f"{path}: empty")
    ff_names = sorted(rows[0]["ff"], key=lambda n: float(n[3:]))
    g_names = sorted(rows[0].get("guided", {}))
    for r in rows:
        if sorted(r["ff"], key=lambda n: float(n[3:])) != ff_names or sorted(r.get("guided", {})) != g_names:
            raise SystemExit(f"{path}: clip {r['clip_id']} has other configurations than {rows[0]['clip_id']}")
    frames = np.asarray([np.stack([np.asarray(r["ff"][n][k]) for n in ff_names])
                         for r in rows for k in range(len(r["frames"]))], dtype=np.int64).reshape(
        -1, len(ff_names), NUM_CLASSES, NUM_CLASSES)
    frame_clip = np.asarray([j for j, r in enumerate(rows) for _ in r["frames"]])
    ff = np.zeros((len(rows), len(ff_names), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(ff, frame_clip, frames)
    guided = np.asarray([[r["guided"][n] for n in g_names] for r in rows], dtype=np.int64).reshape(
        len(rows), len(g_names), NUM_CLASSES, NUM_CLASSES)
    return {"clips": [r["clip_id"] for r in rows], "ff_names": ff_names,
            "ff_scales": [float(n[3:]) for n in ff_names], "ff": ff, "ff_frames": frames, "frame_clip": frame_clip,
            "g_names": g_names, "guided": guided, "iou": {r["clip_id"]: r.get("iou", {}) for r in rows},
            "frames": {r["clip_id"]: r["frames"] for r in rows}}


def _paths(spec: str) -> List[str]:
    return sorted(p for part in _strs(spec)
                  for p in glob.glob(part if Path(part).is_absolute() else str(REPO_ROOT / part)))


def e2e_prices(spec: str, model: str) -> Dict[float, float]:
    """Measured end-to-end full-frame medians (latency-floor ``final`` sessions, graph level, median
    over sessions) per scale."""
    by: Dict[float, List[float]] = {}
    for p in _paths(spec):
        for r in json.loads(Path(p).read_text())["records"].get("final", []):
            if r.get("kind") == "fullframe" and r.get("model") == model and r.get("level") == "graph":
                by.setdefault(float(r["scale"]), []).append(r["cuda_event"]["median_ms"])
    return {s: statistics.median(v) for s, v in by.items()}


def timing_prices(spec: Optional[str], model: str) -> Tuple[Dict[float, float], Dict[str, float]]:
    """From ``time`` sessions: full-frame median per scale and guided overhead per configuration
    (guided minus the full frame at its scale, paired within each session, then the median over sessions)."""
    ff: Dict[float, List[float]] = {}
    over: Dict[str, List[float]] = {}
    for p in _paths(spec) if spec else []:
        recs = [r for r in json.loads(Path(p).read_text())["records"] if r.get("model") == model]
        base = {float(r["scale"]): r["cuda_event"]["median_ms"] for r in recs if r["kind"] == "fullframe"}
        for s, v in base.items():
            ff.setdefault(s, []).append(v)
        for r in recs:
            if r["kind"] == "guided":
                over.setdefault(r["config"], []).append(r["cuda_event"]["median_ms"] - base[float(r["scale"])])
    return ({s: statistics.median(v) for s, v in ff.items()}, {n: statistics.median(v) for n, v in over.items()})


def price_table(scales: Sequence[float], measured: Dict[float, float], frame=FRAME) -> List[float]:
    """Price of each scale: its measurement when there is one, else linear in input pixels between
    the measured scales (latency is linear in pixels above ~0.25 MP on the T4)."""
    import numpy as np

    if not measured:
        raise SystemExit("no full-frame latency measurement to price from")
    ms = sorted(measured)
    px = np.asarray([round(frame[0] * s) * round(frame[1] * s) for s in ms], dtype=np.float64)
    val = np.asarray([measured[s] for s in ms])
    out = []
    for s in scales:
        if s in measured:
            out.append(float(measured[s]))
            continue
        p = round(frame[0] * s) * round(frame[1] * s)
        if not px[0] <= p <= px[-1]:
            raise SystemExit(f"scale {s:g} is outside the measured scales {ms}")
        out.append(float(np.interp(p, px, val)))
    return out


# --------------------------------------------------------------------------- #
# readings
# --------------------------------------------------------------------------- #
def _boot_weights(n: int, n_boot: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]).astype(np.float64)


def _curve_at(lat: Sequence[float], miou, at: float):
    """Piecewise-linear curve(s) at ``at``; NaN outside. ``miou``: (..., S) in ``lat`` order."""
    import numpy as np

    lat = np.asarray(lat, dtype=np.float64)
    order = np.argsort(lat)
    lat, miou = lat[order], np.asarray(miou)[..., order]
    if not lat[0] - 1e-9 <= at <= lat[-1] + 1e-9:
        return np.full(miou.shape[:-1], np.nan)
    j = int(np.clip(np.searchsorted(lat, at), 1, len(lat) - 1))
    w = (at - lat[j - 1]) / (lat[j] - lat[j - 1]) if lat[j] > lat[j - 1] else 0.0
    return miou[..., j - 1] * (1 - w) + miou[..., j] * w


def envelope_at(curves: Dict[str, Tuple[Sequence[float], Any]], at: float):
    """Max over models of each model's full-frame curve at ``at`` (models whose curve does not reach
    ``at`` are left out); also returns which model sets it (on the point estimate)."""
    import numpy as np

    vals = {m: _curve_at(lat, mi, at) for m, (lat, mi) in curves.items()}
    stack = np.stack([np.nan_to_num(v, nan=-np.inf) for v in vals.values()])
    best = stack.max(0)
    return np.where(np.isinf(best), np.nan, best), vals


def readings(runs: Dict[str, Dict[str, Any]], prices: Dict[str, List[float]], overhead: Dict[str, Dict[str, float]],
             default_overhead: float, n_boot: int, names: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Every guided configuration of every model against: the same model's full frame at its scale,
    the same model's full-frame curve and the envelope of all models' curves, at equal latency,
    with a clip bootstrap (the same resamples for all rows and models)."""
    import numpy as np

    from upsample.scale_oracle import miou_many

    clips = runs[next(iter(runs))]["clips"]
    for m, r in runs.items():
        if r["clips"] != clips:
            raise SystemExit(f"{m}: other clips than {next(iter(runs))}: score models on the same clips")
    W = _boot_weights(len(clips), n_boot)
    point_ff, boot_ff = {}, {}
    for m, r in runs.items():
        flat = r["ff"].reshape(len(clips), -1)
        point_ff[m] = miou_many(r["ff"].sum(0))
        boot_ff[m] = miou_many((W @ flat).reshape(n_boot, len(r["ff_names"]), NUM_CLASSES, NUM_CLASSES))
    curves_point = {m: (prices[m], point_ff[m]) for m in runs}
    curves_boot = {m: (prices[m], boot_ff[m]) for m in runs}
    out: Dict[str, Any] = {"clips": len(clips), "n_boot": n_boot, "fullframe": {
        m: {"scales": runs[m]["ff_scales"], "price_ms": prices[m], "miou": point_ff[m].tolist()} for m in runs},
        "configs": []}

    def ci(v):
        v = v[~np.isnan(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [math.nan, math.nan]

    for m, r in runs.items():
        if not r["g_names"]:
            continue
        keep = [j for j, n in enumerate(r["g_names"]) if names is None or n in names.get(m, [])]
        if not keep:
            continue
        g = r["guided"][:, keep]
        point = miou_many(g.sum(0))
        boot = miou_many((W @ g.reshape(len(clips), -1)).reshape(n_boot, len(keep), NUM_CLASSES, NUM_CLASSES))
        for col, j in enumerate(keep):
            name = r["g_names"][j]
            from upsample.guided import GuidedConfig

            c = GuidedConfig.from_name(name)
            k = r["ff_scales"].index(c.scale)
            over = overhead.get(m, {}).get(name, default_overhead)
            price = prices[m][k] + over
            env_p, per_p = envelope_at(curves_point, price)
            env_b, _ = envelope_at(curves_boot, price)
            same_b = _curve_at(prices[m], boot_ff[m], price)
            d_env, d_same = boot[:, col] - env_b, boot[:, col] - same_b
            d_scale = boot[:, col] - boot_ff[m][:, k]
            out["configs"].append({
                "model": m, "name": name, "scale": c.scale, "miou": float(point[col]),
                "price_ms": price, "overhead_ms": over, "overhead_source": "timing" if name in overhead.get(m, {}) else
                "default", "ff_same_scale": float(point_ff[m][k]),
                "delta_same_scale": float(point[col] - point_ff[m][k]), "delta_same_scale_ci95": ci(d_scale),
                "delta_same_curve": float(point[col] - per_p[m]), "delta_same_curve_ci95": ci(d_same),
                "delta_envelope": float(point[col] - env_p), "delta_envelope_ci95": ci(d_env),
                "envelope_model": max(per_p, key=lambda x: -np.inf if np.isnan(per_p[x]) else per_p[x])})
    return out


def _load_runs(args: argparse.Namespace, models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    runs = {}
    for m in models:
        path = Path(args.run_file.format(model=m, split=args.split)) if args.run_file else \
            _out_dir(args) / f"run_{args.split}_{m}.jsonl.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}: `run` then `merge` first")
        runs[m] = load_run(path)
    return runs


def _prices(args: argparse.Namespace, runs: Dict[str, Dict[str, Any]]):
    prices, overhead = {}, {}
    for m, r in runs.items():
        if args.fixture:
            prices[m] = [1.0 + 10.0 * s * s for s in r["ff_scales"]]  # any increasing price: smoke test only
            overhead[m] = {}
            continue
        measured = e2e_prices(args.e2e, m)
        t_ff, t_over = timing_prices(args.timing, m)
        measured.update(t_ff)  # a Method 7 timing session measures the dense scales themselves
        prices[m] = price_table(r["ff_scales"], measured)
        overhead[m] = t_over
    return prices, overhead


def cmd_score(args: argparse.Namespace) -> int:
    models = _strs(args.models)
    runs = _load_runs(args, models)
    prices, overhead = _prices(args, runs)
    names = None
    if args.finalists:
        names = {m: json.loads(Path(args.finalists.format(model=m)).read_text())["finalists"] for m in models}
        if args.split == "val" and not args.timing and not args.fixture:
            raise SystemExit("the val reading needs the T4 timing sessions (--timing): spec, stage 3")
    rep = readings(runs, prices, overhead, args.overhead_ms, args.n_boot, names)
    rep.update(split=args.split, models=models, default_overhead_ms=args.overhead_ms, timing=args.timing)
    cfgs = rep["configs"]
    if not names:  # gate 1 and the finalists (train)
        rep["gate1"] = {}
        for m in models:
            mine = sorted((c for c in cfgs if c["model"] == m), key=lambda c: -c["delta_envelope"])
            fin = [c["name"] for c in mine if c["delta_envelope"] >= GATE1_MARGIN - 1e-12][:MAX_FINALISTS]
            rep["gate1"][m] = {"margin": GATE1_MARGIN, "best": mine[0] if mine else None, "finalists": fin,
                               "continue": bool(fin)}
            if args.write_finalists and not args.fixture:
                _json_dump({"model": m, "split": args.split, "finalists": fin, "rule": "gate 1, zero overhead"},
                           RES / f"finalists_{m}.json")
    else:  # stage 3: the pass rule on the finalists
        rep["verdict"] = {m: [{"name": c["name"], "delta_envelope": c["delta_envelope"],
                               "ci95": c["delta_envelope_ci95"],
                               "pass": c["delta_envelope"] > 0 and c["delta_envelope_ci95"][0] > 0}
                              for c in cfgs if c["model"] == m] for m in models}
    out = Path(args.out) if args.out else (EXP if args.fixture else RES) / f"score_{args.split}{'_fixture' if args.fixture else ''}.json"
    _json_dump(rep, out)
    for m in models:
        ff = rep["fullframe"][m]
        print(f"[method7 score] {m} full frame: " + ", ".join(
            f"{s:g}: {100 * v:.2f} @ {p:.2f} ms" for s, v, p in zip(ff["scales"], ff["miou"], ff["price_ms"])))
        for c in sorted((c for c in cfgs if c["model"] == m), key=lambda c: -c["delta_envelope"])[:args.show]:
            print(f"  {c['name']:<34} {100 * c['miou']:.2f} @ {c['price_ms']:.2f} ms  same scale "
                  f"{100 * c['delta_same_scale']:+.2f}  vs own curve {100 * c['delta_same_curve']:+.2f}  vs envelope "
                  f"{100 * c['delta_envelope']:+.2f} [{100 * c['delta_envelope_ci95'][0]:+.2f}, "
                  f"{100 * c['delta_envelope_ci95'][1]:+.2f}]")
        if "gate1" in rep:
            g = rep["gate1"][m]
            print(f"  gate 1: {'CONTINUE' if g['continue'] else 'STOP'}; finalists {g['finalists']}")
        if "verdict" in rep:
            print(f"  verdict: {'PASS' if any(v['pass'] for v in rep['verdict'][m]) else 'FAIL'}")
    print(f"[method7 score] -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# Method 8: the per-frame scale oracle
# --------------------------------------------------------------------------- #
def cmd_oracle(args: argparse.Namespace) -> int:
    import numpy as np

    from upsample.scale_oracle import fixed_mix, miou_many, scale_oracle

    models = _strs(args.models)
    runs = _load_runs(args, models)
    prices, _ = _prices(args, runs)
    curves = {m: (prices[m], miou_many(runs[m]["ff"].sum(0))) for m in models}
    W = _boot_weights(len(runs[models[0]]["clips"]), args.n_boot)
    rep: Dict[str, Any] = {"split": args.split, "budgets": {}, "margin": ORACLE_MARGIN}
    for m in models:
        r = runs[m]
        lat = np.asarray(prices[m])
        rep["budgets"][m] = []
        for b in _floats(args.budgets):
            budget = float(lat[r["ff_scales"].index(b)])
            res = [scale_oracle(r["ff_frames"], lat, budget, seed=s, max_sweeps=args.max_sweeps)
                   for s in range(args.seeds)]
            best = max(res, key=lambda x: x.miou)
            env, per = envelope_at(curves, best.mean_ms)
            # bootstrap: the oracle's selection held fixed, clips resampled (a bound; no reselection)
            sel = np.zeros((len(r["clips"]), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            np.add.at(sel, r["frame_clip"], r["ff_frames"][np.arange(len(best.choice)), best.choice])
            boot_sel = miou_many((W @ sel.reshape(len(r["clips"]), -1)).reshape(-1, NUM_CLASSES, NUM_CLASSES))
            boot_curves = {mm: (prices[mm], miou_many((W @ runs[mm]["ff"].reshape(len(r["clips"]), -1)).reshape(
                args.n_boot, len(runs[mm]["ff_names"]), NUM_CLASSES, NUM_CLASSES))) for mm in models}
            env_b, _ = envelope_at(boot_curves, best.mean_ms)
            d = boot_sel - env_b
            d = d[~np.isnan(d)]
            share = np.bincount(best.choice, minlength=len(lat)) / len(best.choice)
            mix = fixed_mix(r["ff_frames"], lat, budget)
            entry = {"budget_scale": b, "budget_ms": budget, "oracle_miou": best.miou, "oracle_mean_ms": best.mean_ms,
                     "oracle_p95_ms": float(np.percentile(lat[best.choice], 95)),
                     "seeds_miou": [x.miou for x in res], "ff_same_scale": float(curves[m][1][r["ff_scales"].index(b)]),
                     "delta_same_scale": best.miou - float(curves[m][1][r["ff_scales"].index(b)]),
                     "delta_envelope": float(best.miou - env), "envelope_by_model": {k: float(v) for k, v in per.items()},
                     "delta_envelope_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else None,
                     "scale_share": {f"{s:g}": float(v) for s, v in zip(r["ff_scales"], share) if v > 0},
                     "fixed_mix": mix, "pass": float(best.miou - env) >= ORACLE_MARGIN - 1e-12,
                     "choice": best.choice.tolist(), "frame_clip": [r["clips"][j] for j in r["frame_clip"]]}
            rep["budgets"][m].append(entry)
            print(f"[method8 oracle] {m} budget = ff {b:g} ({budget:.2f} ms): oracle {100 * best.miou:.2f} at "
                  f"{best.mean_ms:.2f} ms (seeds {', '.join(f'{100 * x.miou:.2f}' for x in res)}); same scale "
                  f"{100 * entry['delta_same_scale']:+.2f}; vs envelope {100 * entry['delta_envelope']:+.2f}"
                  + (f" [{100 * entry['delta_envelope_ci95'][0]:+.2f}, {100 * entry['delta_envelope_ci95'][1]:+.2f}]"
                     if entry["delta_envelope_ci95"] else "") + f"; shares {entry['scale_share']}", flush=True)
    rep["pass"] = any(e["pass"] for v in rep["budgets"].values() for e in v)
    print(f"[method8 oracle] bound {'PASSES' if rep['pass'] else 'FAILS'} (margin +{100 * ORACLE_MARGIN:.1f} vs envelope)")
    out = Path(args.out) if args.out else (EXP if args.fixture else RES) / f"oracle_{args.split}{'_fixture' if args.fixture else ''}.json"
    _json_dump(rep, out)
    print(f"[method8 oracle] -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# Method 8, step 2: predictability screen (no GPU, the run files only)
# --------------------------------------------------------------------------- #
SCREEN_MARGIN = 0.010  # the stale oracle must keep +1.0 mIoU point above the envelope


def screen_selections(choice, frame_clip, n_scales: int, base: int, seed: int = 0) -> Dict[str, Any]:
    """Selections derived from the oracle's per-frame choice (frames in clip order):

    * ``stale``: each labelled frame takes the oracle's choice of the previous labelled frame of its
      clip (5 frames earlier on VPS); the first labelled frame of a clip takes ``base``. The most a
      selector that only sees the past could know, if it knew the past perfectly.
    * ``shuffle_clip``: the oracle's choices permuted within each clip (keeps each clip's mix).
    * ``shuffle_all``: permuted over all frames (keeps only the overall scale shares).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    choice, frame_clip = np.asarray(choice), np.asarray(frame_clip)
    stale = np.full_like(choice, base)
    shuffle_clip = choice.copy()
    for c in np.unique(frame_clip):
        idx = np.flatnonzero(frame_clip == c)
        stale[idx[1:]] = choice[idx[:-1]]
        shuffle_clip[idx] = rng.permutation(choice[idx])
    return {"stale": stale, "shuffle_clip": shuffle_clip, "shuffle_all": rng.permutation(choice)}


def cmd_screen(args: argparse.Namespace) -> int:
    """Is the oracle's gain predictable? Per model and budget: the frame oracle, the clip oracle (one
    scale per clip, with labels), the stale oracle and two shuffles, each read against the envelope
    at its own mean latency with a clip bootstrap. Pass rule (spec, Method 8 step 2): build a
    realisable selector only if the stale oracle keeps >= +1.0 vs the envelope at some budget."""
    import numpy as np

    from upsample.scale_oracle import miou_many, scale_oracle

    models = _strs(args.models)
    runs = _load_runs(args, models)
    prices, _ = _prices(args, runs)
    curves = {m: (prices[m], miou_many(runs[m]["ff"].sum(0))) for m in models}
    n_clips = len(runs[models[0]]["clips"])
    W = _boot_weights(n_clips, args.n_boot)
    boot_curves = {m: (prices[m], miou_many((W @ runs[m]["ff"].reshape(n_clips, -1)).reshape(
        args.n_boot, len(runs[m]["ff_names"]), NUM_CLASSES, NUM_CLASSES))) for m in models}
    rep: Dict[str, Any] = {"split": args.split, "margin": SCREEN_MARGIN, "models": {}}
    for m in models:
        r = runs[m]
        lat = np.asarray(prices[m])
        F = len(r["frame_clip"])
        rep["models"][m] = []
        for b in _floats(args.budgets):
            k = r["ff_scales"].index(b)
            budget = float(lat[k])
            frame = max((scale_oracle(r["ff_frames"], lat, budget, seed=s, max_sweeps=args.max_sweeps)
                         for s in range(args.seeds)), key=lambda x: x.miou)
            clip = scale_oracle(r["ff"], lat, budget, seed=0, max_sweeps=args.max_sweeps)
            sels = {"frame_oracle": frame.choice, "clip_oracle": np.asarray(clip.choice)[r["frame_clip"]]}
            sels.update(screen_selections(frame.choice, r["frame_clip"], len(lat), k, seed=0))
            entry: Dict[str, Any] = {"budget_scale": b, "budget_ms": budget}
            for name, ch in sels.items():
                per_clip = np.zeros((n_clips, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                np.add.at(per_clip, r["frame_clip"], r["ff_frames"][np.arange(F), ch])
                miou = float(miou_many(per_clip.sum(0)))
                mean_ms = float(lat[ch].mean())
                env, _ = envelope_at(curves, mean_ms)
                env_b, _ = envelope_at(boot_curves, mean_ms)
                d = miou_many((W @ per_clip.reshape(n_clips, -1)).reshape(-1, NUM_CLASSES, NUM_CLASSES)) - env_b
                d = d[~np.isnan(d)]
                entry[name] = {"miou": miou, "mean_ms": mean_ms, "delta_envelope": float(miou - env),
                               "delta_envelope_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                               "agree_with_oracle": float(np.mean(ch == frame.choice))}
            entry["pass"] = entry["stale"]["delta_envelope"] >= SCREEN_MARGIN - 1e-12
            rep["models"][m].append(entry)
            print(f"[method8 screen] {m} budget ff {b:g} ({budget:.2f} ms):")
            for name in ("frame_oracle", "clip_oracle", "stale", "shuffle_clip", "shuffle_all"):
                e = entry[name]
                print(f"  {name:<13} {100 * e['miou']:.2f} @ {e['mean_ms']:.2f} ms  vs envelope "
                      f"{100 * e['delta_envelope']:+.2f} [{100 * e['delta_envelope_ci95'][0]:+.2f}, "
                      f"{100 * e['delta_envelope_ci95'][1]:+.2f}]  same choice as oracle {100 * e['agree_with_oracle']:.0f}%")
    rep["pass"] = any(e["pass"] for v in rep["models"].values() for e in v)
    print(f"[method8 screen] stale oracle {'KEEPS' if rep['pass'] else 'LOSES'} the margin "
          f"(+{100 * SCREEN_MARGIN:.1f} vs envelope): {'build a selector' if rep['pass'] else 'close Method 8'}")
    out = Path(args.out) if args.out else (EXP if args.fixture else RES) / f"screen_{args.split}{'_fixture' if args.fixture else ''}.json"
    _json_dump(rep, out)
    print(f"[method8 screen] -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# moving-object recall (train_all)
# --------------------------------------------------------------------------- #
def cmd_objects(args: argparse.Namespace) -> int:
    """Headline recall (tiny + small moving objects) of every config with object IoUs, against the
    same model's full-frame curve at equal latency; Method 8's oracle selection too, when present."""
    import numpy as np

    from eval.objects import recall_at_equal_latency, recall_table
    from scripts.objects import groups_of, load_bins, load_table, table_path

    cols, _ = load_table(Path(args.table) if args.table else table_path(args.split))
    bins = load_bins(args.bins)
    groups = groups_of(cols, bins)
    head = next(k for k in groups if k.startswith("headline"))
    starts: Dict[str, Tuple[int, int]] = {}
    for c in dict.fromkeys(cols["clip"].tolist()):
        idx = np.flatnonzero(cols["clip"] == c)
        starts[c] = (int(idx[0]), int(idx[-1]) + 1)
    models = _strs(args.models)
    runs = _load_runs(args, models)
    prices, overhead = _prices(args, runs)
    oracle = json.loads(Path(args.oracle).read_text()) if args.oracle and Path(args.oracle).is_file() else None
    rep: Dict[str, Any] = {"split": args.split, "headline": head, "models": {}}
    for m in models:
        r = runs[m]
        names = sorted({n for d in r["iou"].values() for n in d})
        arr = {n: np.full(len(cols["clip"]), np.nan) for n in names}
        for clip, d in r["iou"].items():
            if clip not in starts:
                continue
            a, b = starts[clip]
            for n, v in d.items():
                if len(v) != b - a:
                    raise SystemExit(f"{clip} {n}: {len(v)} IoUs for {b - a} objects (table out of date?)")
                arr[n][a:b] = v
        full = [n for n in names if not np.isnan(arr[n]).any()]
        ff = sorted((n for n in full if n.startswith("ff/")), key=lambda n: float(n[3:]))
        if len(ff) < 2:
            raise SystemExit(f"{m}: the full-frame curve needs object IoUs at two scales at least")
        ff_lat = [prices[m][r["ff_scales"].index(float(n[3:]))] for n in ff]
        rows = []

        def entry(name: str, iou, lat: float) -> Dict[str, Any]:
            found = iou > 0.5
            return {"name": name, "latency_ms": lat, "recall": recall_table(found, cols["clip"], {head: groups[head]},
                                                                            n_boot=args.n_boot)[head],
                    "vs_fullframe": recall_at_equal_latency(found, lat, [arr[n] > 0.5 for n in ff], ff_lat, cols["clip"],
                                                            groups[head], n_boot=args.n_boot)}

        for n in full:
            if n.startswith("ff/"):
                lat = prices[m][r["ff_scales"].index(float(n[3:]))]
            else:
                from upsample.guided import GuidedConfig

                lat = prices[m][r["ff_scales"].index(GuidedConfig.from_name(n).scale)] + \
                    overhead.get(m, {}).get(n, args.overhead_ms)
            rows.append(entry(n, arr[n], lat))
        if oracle:
            for e in oracle["budgets"].get(m, []):
                iou = np.full(len(cols["clip"]), np.nan)
                pos: Dict[str, int] = {}
                for clip, k in zip(e["frame_clip"], e["choice"]):
                    j = pos.get(clip, 0)
                    pos[clip] = j + 1
                    if clip not in starts:  # a clip without any thing object
                        continue
                    name = r["ff_names"][k]
                    if name not in arr:
                        break
                    a, b = starts[clip]
                    mask = cols["t"][a:b] == r["frames"][clip][j]  # this labelled frame's objects
                    iou[a:b][mask] = arr[name][a:b][mask]
                if not np.isnan(iou).any():
                    rows.append(entry(f"8o/budget{e['budget_scale']:g}", iou, e["oracle_mean_ms"]))
        rep["models"][m] = {"fullframe": ff, "configs": rows}
        for e in sorted(rows, key=lambda e: e["latency_ms"]):
            v = e["vs_fullframe"]
            print(f"  {m} {e['name']:<36} {e['latency_ms']:6.2f} ms  recall {e['recall']['recall']:.3f}  vs own curve "
                  f"{100 * v['delta_recall_at_equal_latency']:+.2f} pts")
    out = Path(args.out) if args.out else (EXP if args.fixture else RES) / f"objects_{args.split}{'_fixture' if args.fixture else ''}.json"
    _json_dump(rep, out)
    print(f"[method7 objects] -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# time (T4)
# --------------------------------------------------------------------------- #
def cmd_time(args: argparse.Namespace) -> int:
    """End to end from a full-resolution FP16 frame, as CUDA graphs, interleaved in the same rounds:
    each model's full frame at every dense scale (resize, network, bilinear, argmax) and each
    finalist (resize, network, guided step, argmax)."""
    import torch

    from eval.gpu_timing import GpuTelemetry, gpu_environment
    from foveation.ops import downsample
    from models.utils import upsample_logits
    from scripts.latency_floor import build, frame_of, time_part
    from upsample.guided import GuidedConfig, guided_logits

    if not torch.cuda.is_available():
        raise SystemExit("time needs a CUDA GPU")
    settings = {"rounds": args.rounds, "iters": args.iters, "warmup": args.warmup}
    models = {m: build(m) for m in _strs(args.models)}
    x = frame_of((1, 3, *FRAME))
    cases = []
    for m, net in models.items():
        for s in _floats(args.scales):
            def full(frame, net=net, s=s):
                small = downsample(frame, s)
                return upsample_logits(net(small), FRAME, net.output_stride, net.align_corners,
                                       input_size=tuple(small.shape[-2:])).argmax(1)
            cases.append({"name": f"m7/fullframe/{m}/{s:g}", "kind": "fullframe", "model": m, "scale": s, "fn": full, "x": x})
        names = []
        if args.finalists:
            p = Path(args.finalists.format(model=m))
            names = json.loads(p.read_text())["finalists"] if p.is_file() else []
        names += [n for n in _strs(args.configs or "")]
        for n in dict.fromkeys(names):
            c = GuidedConfig.from_name(n)
            if c.scale not in _floats(args.scales):
                raise SystemExit(f"{n}: its scale must be among --scales (the overhead is paired with it)")

            def guided(frame, net=net, c=c):
                small = downsample(frame, c.scale)
                return guided_logits(net(small), frame, c, net.output_stride, net.align_corners,
                                     tuple(small.shape[-2:]), full_dtype=torch.float16).argmax(1)
            cases.append({"name": f"m7/guided/{m}/{n}", "kind": "guided", "model": m, "scale": c.scale, "config": n,
                          "fn": guided, "x": x})
    t0 = time.time()
    with GpuTelemetry() as tel:
        records = time_part(cases, "graph", settings, tel)
        session_tel = tel.summarize(tel.samples)
    out = Path(args.out) / f"timing_{args.session}.json"
    _json_dump({"session": args.session, "duration_s": time.time() - t0, "settings": settings,
                "environment": gpu_environment(), "session_telemetry": session_tel, "records": records}, out)
    for r in records:
        print(f"  {r['name']:<60} {r['cuda_event']['median_ms']:7.2f} ms")
    print(f"[method7 time] -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, models="pidnet_s,bisenetv2"):
        sp.add_argument("--split", default="train_all")
        sp.add_argument("--models", default=models)
        sp.add_argument("--out-dir", default=None, help="Run files directory (default experiments/method7/<split>).")
        sp.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)

    def scoring(sp):
        sp.add_argument("--run-file", default=None, help="Run file pattern with {model} / {split} (default: merged file).")
        sp.add_argument("--e2e", default=DEFAULT_E2E, help="Latency-floor `final` sessions: measured full frame.")
        sp.add_argument("--timing", default=None, help="Method 7 `time` sessions (glob): dense full frame + overheads.")
        sp.add_argument("--overhead-ms", type=float, default=0.0, help="Overhead of a guided config without timing.")
        sp.add_argument("--n-boot", type=int, default=1000)
        sp.add_argument("--out", default=None)

    r = sub.add_parser("run", help="network passes per clip: dense full frame, guided configs, object IoUs")
    common(r, models="pidnet_s")
    r.add_argument("--clip-ids", default=None)
    r.add_argument("--chunk", default=None, help="i/N: the i-th block of the sorted clip ids.")
    r.add_argument("--tag", default=None)
    r.add_argument("--ff-scales", default=",".join(f"{s:g}" for s in FF_SCALES), help="'none' to skip.")
    r.add_argument("--guided-scales", default=",".join(f"{s:g}" for s in GUIDED_SCALES), help="'none' to skip.")
    r.add_argument("--configs", default=None, help="Comma-separated guided names instead of the grid.")
    r.add_argument("--obj", choices=("none", "ff", "all"), default="none",
                   help="Object IoUs: none, the full-frame rows, or every row (train_all only).")
    r.add_argument("--device", default="auto")
    r.add_argument("--max-temp", type=int, default=None, help="Laptop: pause at this temperature (C), resume <= 65 C.")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("merge", help="chunk files -> one run file per model")
    common(m)
    m.add_argument("--no-expect", action="store_true", help="Do not check against the split's clip list.")
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("score", help="Method 7 readings; gate 1 and finalists on train, the pass rule on val")
    common(s)
    scoring(s)
    s.add_argument("--finalists", default=None, help="results/method7/finalists_{model}.json: score these only.")
    s.add_argument("--write-finalists", action="store_true", help="Write results/method7/finalists_<model>.json.")
    s.add_argument("--show", type=int, default=8)
    s.set_defaults(func=cmd_score)

    o = sub.add_parser("oracle", help="Method 8: per-frame scale oracle at the budgets")
    common(o)
    scoring(o)
    o.add_argument("--budgets", default=",".join(f"{b:g}" for b in ORACLE_BUDGETS))
    o.add_argument("--seeds", type=int, default=3)
    o.add_argument("--max-sweeps", type=int, default=20)
    o.set_defaults(func=cmd_oracle)

    q = sub.add_parser("screen", help="Method 8 step 2: is the oracle's gain predictable (clip / stale oracle)")
    common(q)
    scoring(q)
    q.add_argument("--budgets", default=",".join(f"{b:g}" for b in ORACLE_BUDGETS))
    q.add_argument("--seeds", type=int, default=3)
    q.add_argument("--max-sweeps", type=int, default=20)
    q.set_defaults(func=cmd_screen)

    b = sub.add_parser("objects", help="moving-object recall of the configs with object IoUs (train_all)")
    common(b)
    scoring(b)
    b.add_argument("--table", default=None)
    b.add_argument("--bins", default=None)
    b.add_argument("--oracle", default=str(RES / "oracle_train_all.json"))
    b.set_defaults(func=cmd_objects)

    t = sub.add_parser("time", help="T4 timing: dense full frame and the finalists, end to end")
    t.add_argument("--session", required=True)
    t.add_argument("--models", default="pidnet_s,bisenetv2")
    t.add_argument("--scales", default=",".join(f"{s:g}" for s in FF_SCALES))
    t.add_argument("--finalists", default=str(RES / "finalists_{model}.json"))
    t.add_argument("--configs", default=None, help="Extra guided names to time.")
    t.add_argument("--rounds", type=int, default=5)
    t.add_argument("--iters", type=int, default=40)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--out", default=str(EXP / "timing"))
    t.set_defaults(func=cmd_time)
    args = p.parse_args(argv)
    if getattr(args, "fixture", False) and args.cmd != "time":
        args.models = ",".join(FIXTURE_MODELS)
        if args.cmd == "run" and args.ff_scales == ",".join(f"{s:g}" for s in FF_SCALES):
            args.ff_scales = "0.25,0.5,0.75,1"
            args.guided_scales = "0.5"
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
