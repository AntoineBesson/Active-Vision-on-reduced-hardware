#!/usr/bin/env python3
"""Method 2 (foveation without memory): caches, rule sweep, class-weight tuning, finalists, val report.

    python -m scripts.method2 cache --split train            # per-cell caches (GPU, long)
    python -m scripts.method2 sweep --split train            # every rule x geometry x k, from the caches
    python -m scripts.method2 tune-weights --split train --pair pidnet_s+pidnet_s --scale 0.25 --size 384 --score entropy
    python -m scripts.method2 finalists                      # Pareto + equal-latency picks, from the train sweep
    python -m scripts.method2 report --split val             # the finalists, once, on val

Accuracy comes from the per-cell caches (``foveation.cache``: exact for any
selection), latency from the T4 latency-floor tables (``foveation.latency_model``),
and the reference is the full-frame curve of the glimpse model
(``scripts.method2_fullframe``). Rules and finalists are chosen on train; val
is evaluated by ``report`` only. ``--fixture`` runs everything on the synthetic
fixture with the dummy backbone (a smoke test, not results).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method2.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_fixture, ensure_src_on_path, expand_env, load_yaml, resolve_device  # noqa: E402

ensure_src_on_path()

MODEL_CONFIGS = {"pidnet_s": "configs/baseline_fullframe_pidnet_s.yaml",
                 "bisenetv2": "configs/baseline_fullframe_bisenetv2.yaml"}
FIXTURE_CONFIG = "configs/baseline_fixture.yaml"  # model "dummy"
FIXTURE_STRIDES = {"dummy": 8}
DEFAULT_PAIRS = "pidnet_s+pidnet_s,bisenetv2+bisenetv2,bisenetv2+pidnet_s"
# glimpse geometries "input size:input margin:glimpse scale" (scale < 1: a wider frame region resized to the input)
DEFAULT_GEOMS = "256:32:1,384:32:1,512:32:1,384:32:0.5,512:32:0.5,384:24:0.75"
DEFAULT_FUSIONS = "replace,mean,conf"
DEFAULT_LATENCY = "results/latency_floor/latency_floor_t4-s*.json"
SCORES = ("entropy", "margin", "maxprob")
POOLS = ("mean", "top10")
SELECTIONS = ("topk", "greedy")
RANDOM_SEEDS = range(5)
WEIGHT_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
EXCLUDED_FROM_FINALISTS = ("oracle", "random", "all")


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _floats(text: str) -> List[float]:
    return [float(v) for v in text.split(",") if v]


def _ints(text: str) -> List[int]:
    return [int(v) for v in text.split(",") if v]


def _pairs(text: str) -> List[Tuple[str, str]]:
    return [tuple(p.split("+")) for p in text.split(",") if p]  # type: ignore[misc]


Geom = Tuple[int, int, float]  # (network input size, input margin, glimpse scale)


def _geoms(text: str) -> List[Geom]:
    out = []
    for g in text.split(","):
        if g:
            size, margin, scale = g.split(":")
            out.append((int(size), int(margin), float(scale)))
    return out


def _strs(text: str) -> List[str]:
    return [v for v in text.split(",") if v]


def model_config(name: str, fixture: bool) -> Dict[str, Any]:
    if fixture:
        cfg = load_yaml(REPO_ROOT / FIXTURE_CONFIG)
        cfg["dataset"]["root"] = str(ensure_fixture())
        cfg["fixture"] = True
        return cfg
    if name not in MODEL_CONFIGS:
        raise SystemExit(f"unknown model {name!r}; known: {sorted(MODEL_CONFIGS)}")
    return expand_env(load_yaml(REPO_ROOT / MODEL_CONFIGS[name]))


def load_dataset(cfg: Dict[str, Any], split: str):
    from data.registry import build_dataset

    spec = dict(cfg["dataset"], split=split)
    dataset = build_dataset(spec, base_dir=REPO_ROOT)
    if len(dataset) == 0 or dataset.num_labeled_frames() == 0:
        raise SystemExit(f"split {split}: {len(dataset)} clips, {dataset.num_labeled_frames()} labelled frames")
    return dataset


def frame_size_of(dataset, fixture: bool) -> Tuple[int, int]:
    from foveation.cache import FRAME

    if not fixture:
        return FRAME
    import numpy as np

    t = int(np.flatnonzero(np.asarray(dataset.labeled_mask(0), dtype=bool))[0])
    return tuple(int(s) for s in dataset.load_frame(0, t).shape[:2])  # type: ignore[return-value]


def latency_tables(spec: str, fixture: bool):
    from foveation.latency_model import LatencyTables

    paths = sorted(p for part in spec.split(",") for p in glob.glob(str(REPO_ROOT / part) if not Path(part).is_absolute() else part))
    if not paths:
        raise SystemExit(f"no latency-floor result matches {spec!r}")
    return LatencyTables.load(paths, strides=FIXTURE_STRIDES if fixture else None)


def geom_grid(frame_size, geom: Geom):
    """The candidate grid, in frame pixels, of a glimpse geometry."""
    from foveation.pipeline import FoveationConfig, glimpse_grid

    size, margin, gscale = geom
    return glimpse_grid(frame_size, FoveationConfig(glimpse_size=size, glimpse_margin=margin, glimpse_scale=gscale))


def _cache_frame_size(root: Path, split: str) -> Tuple[int, int]:
    """The frame size the caches of ``split`` were built for (read from any cache's meta)."""
    import numpy as np

    for p in sorted((root / split).glob("*.npz")):
        if not p.name.endswith(".tmp.npz"):
            return tuple(json.loads(str(np.load(p)["meta"]))["frame_size"])  # type: ignore[return-value]
    raise SystemExit(f"no cache in {root / split}: run `python -m scripts.method2 cache --split {split}` first")


def load_pair(root: Path, split: str, ctx_model: str, gl_model: str, scale: float, geom: Geom, fusion: str):
    from foveation.cache import cache_path
    from foveation.cached_eval import CachedPair

    grid = geom_grid(_cache_frame_size(root, split), geom)
    ctx = cache_path(root, split, "context", ctx_model, grid, scale)
    gl = cache_path(root, split, "glimpse", gl_model, grid, scale, input_size=geom[0], fusion=fusion,
                    ctx_model=ctx_model)
    for p in (ctx, gl):
        if not p.is_file():
            raise SystemExit(f"missing cache {p}: run `python -m scripts.method2 cache --split {split}`")
    return CachedPair.load(ctx, gl)


def fullframe_curve(ff_dir: Path, model: str, split: str, tables) -> Dict[str, Any]:
    """The glimpse model's full-frame reference: per-clip confusions, mIoU and T4 latency per scale."""
    import numpy as np

    from eval.tradeoff import FullFrameCurve, miou_of

    path = Path(ff_dir) / f"fullframe_{model}_{split}.npz"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run scripts.method2_fullframe for {model} on {split}")
    d = np.load(path)
    scales = [float(s) for s in d["scales"]]
    lat = [tables.fullframe(model, s) for s in scales]
    miou = [miou_of(c.sum(0)) for c in d["confusion"]]
    return {"scales": scales, "latency_ms": [x["median_ms"] for x in lat], "miou": miou,
            "full": tables.fullframe(model, 1.0), "clips": d["confusion"],
            "curve": FullFrameCurve([x["median_ms"] for x in lat], miou)}


def rule_from(d: Dict[str, Any]):
    from foveation.cached_eval import Rule

    d = dict(d)
    if d.get("class_weights") is not None:
        d["class_weights"] = tuple(float(w) for w in d["class_weights"])
    return Rule(**d)


def rule_dict(rule) -> Dict[str, Any]:
    from dataclasses import asdict

    d = asdict(rule)
    d["class_weights"] = None if rule.class_weights is None else list(rule.class_weights)
    return d


def metrics_of(total, pair) -> Dict[str, Any]:
    import numpy as np

    from eval.metrics import ConfusionMatrix

    meta = pair.meta["context"]
    cm = ConfusionMatrix(meta["num_classes"], meta["ignore_index"], meta.get("class_names"))
    cm.matrix = np.asarray(total, dtype=np.int64)
    out = cm.compute()
    out["correct_pixels"] = int(np.trace(cm.matrix))
    return out


def cost_of(ctx_model, gl_model, scale, geom: Geom, fusion, k, selection, frame_size, metrics, tables,
            ref) -> Dict[str, Any]:
    """Pixels, FLOPs, predicted T4 latency and the two trade-off readings of one configuration.

    The latency tables were measured with plain glimpses (glimpse scale 1, no
    fusion): the crop resize and the fusion are missing from the overhead, so
    for the other configurations the prediction is a lower bound until the T4
    ``final`` run measures them.
    """
    from eval.tradeoff import delta_miou_at_equal_latency, latency_saved_at_equal_miou

    size = geom[0]
    pixel_fraction = scale * scale + k * size * size / (frame_size[0] * frame_size[1])
    cost: Dict[str, Any] = {"pixel_fraction": pixel_fraction, "t4_latency": None, "gflops": None,
                            "t4_latency_is_lower_bound": fusion != "replace" or geom[2] != 1.0,
                            "fullframe_gflops": ref["full"]["gflops"], "fullframe_latency_ms": ref["full"]["median_ms"],
                            "flops_implied_speedup": None, "realised_speedup": None,
                            "delta_miou_at_equal_latency": None, "latency_saved_at_equal_miou": None,
                            "fullframe_curve": {"scales": ref["scales"], "latency_ms": ref["latency_ms"], "miou": ref["miou"]}}
    if selection is None:
        return cost
    t4 = tables.predict(ctx_model, scale, gl_model, size, k, selection)
    cost.update(t4_latency=t4, gflops=t4["gflops"],
                flops_implied_speedup=ref["full"]["gflops"] / t4["gflops"],
                realised_speedup=ref["full"]["median_ms"] / t4["median_ms"],
                delta_miou_at_equal_latency=delta_miou_at_equal_latency(metrics["miou"], t4["median_ms"], ref["curve"]),
                latency_saved_at_equal_miou=latency_saved_at_equal_miou(metrics["miou"], t4["median_ms"], ref["curve"]))
    return cost


def method_of(ctx_model, gl_model, scale, geom: Geom, fusion, k, rule) -> Dict[str, Any]:
    return {"name": "foveated", "pair": f"{ctx_model}+{gl_model}", "context_model": ctx_model,
            "glimpse_model": gl_model, "context_scale": scale, "glimpse_size": geom[0], "margin": geom[1],
            "glimpse_scale": geom[2], "fusion": fusion, "k": k, "rule": rule_dict(rule)}


def geom_of(method: Dict[str, Any]) -> Geom:
    return (method["glimpse_size"], method["margin"], method.get("glimpse_scale", 1.0))


def run_name(method: Dict[str, Any], rule) -> str:
    return (f"{method['pair']}_s{method['context_scale']:g}_g{method['glimpse_size']}m{method['margin']}"
            f"z{method.get('glimpse_scale', 1.0):g}_{method.get('fusion', 'replace')}"
            f"_k{method['k']}_{rule.name().replace('/', '-')}")


def _weights_key(pair: str, scale: float, geom: Geom, fusion: str) -> str:
    return f"{pair}/s{scale:g}/g{geom[0]}m{geom[1]}z{geom[2]:g}/{fusion}"


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cmd_cache(args: argparse.Namespace) -> int:
    """Every missing cache of the requested models, geometries and pairs, in one pass over the frames.

    Per model: a context cache per (scale, grid) and a plain glimpse cache per
    geometry; per pair whose models are both requested: a fused glimpse cache
    per (scale, geometry, fusion other than replace).
    """
    from eval.results import git_info
    from foveation.cache import CacheJob, build_caches, cache_path
    from models.registry import build_backbone

    device = resolve_device(args.device)
    root = Path(args.root)
    models = _strs(args.models)
    cfg = model_config(models[0], args.fixture)
    dataset = load_dataset(cfg, args.split)
    frame_size = frame_size_of(dataset, args.fixture)
    clips = None if args.max_clips is None else list(range(min(args.max_clips, len(dataset))))
    meta = {"split": args.split, "git": git_info(REPO_ROOT), "dataset_fingerprint": dataset.fingerprint(),
            "class_names": list(dataset.class_names), "clips_limited_to": args.max_clips}
    scales, geoms = _floats(args.scales), _geoms(args.geoms)
    fusions = [f for f in _strs(args.fusions) if f != "replace"]
    wanted = []
    for model in models:
        for geom in geoms:
            grid = geom_grid(frame_size, geom)
            wanted.append(CacheJob("glimpse", model, grid, cache_path(root, args.split, "glimpse", model, grid,
                                                                      input_size=geom[0]), input_size=geom[0]))
            for scale in scales:
                wanted.append(CacheJob("context", model, grid, cache_path(root, args.split, "context", model, grid,
                                                                          scale), scale=scale))
    for ctx_model, gl_model in _pairs(args.pairs or ""):
        if ctx_model not in models or gl_model not in models:
            continue
        for geom in geoms:
            grid = geom_grid(frame_size, geom)
            for scale in scales:
                for fusion in fusions:
                    path = cache_path(root, args.split, "glimpse", gl_model, grid, scale, input_size=geom[0],
                                      fusion=fusion, ctx_model=ctx_model)
                    wanted.append(CacheJob("glimpse", gl_model, grid, path, scale=scale, input_size=geom[0],
                                           fusion=fusion, ctx_model=ctx_model))
    jobs, skipped = [], 0
    for job in {j.path: j for j in wanted}.values():  # a context cache is shared by grids of equal frame geometry
        stem = job.path.name[:-len(".npz")]
        for tmp in (job.path.with_name(stem + ".tmp.npz"), job.path.with_name(stem + ".conf.tmp.npy")):
            if tmp.exists():
                tmp.unlink()  # left by an interrupted build: never trusted
        if job.path.exists():
            skipped += 1
        else:
            jobs.append(job)
    if jobs:
        backbones = {}
        for name in sorted({j.model for j in jobs} | {j.ctx_model for j in jobs if j.ctx_model}):
            bcfg = model_config(name, args.fixture)
            backbones[name] = build_backbone(bcfg["backbone"], context={"palette": dataset.palette}).to(device).eval()
        print(f"[method2 cache] {len(jobs)} caches in one pass: {', '.join(j.path.name for j in jobs)}", flush=True)
        build_caches(dataset, backbones, jobs, device, clip_indices=clips, precision=args.precision,
                     allow_any_frame_size=args.fixture, extra_meta=meta, progress=True)
    print(f"[method2 cache] built {len(jobs)}, skipped {skipped} existing, in {root / args.split}")
    return 0


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def sweep_rules(weights: Dict[str, Any], key: str) -> List[Any]:
    from foveation.cached_eval import Rule

    rules = [Rule("oracle"), Rule("prior")]
    rules += [Rule("uncertainty", score=s, pool=p, selection=sel) for s in SCORES for p in POOLS for sel in SELECTIONS]
    if key in weights:
        w = tuple(weights[key]["weights"])
        rules += [Rule("uncertainty", score=weights[key]["score"], pool="mean", selection=sel, class_weights=w)
                  for sel in SELECTIONS]
    return rules


def _base_record(args: argparse.Namespace) -> Dict[str, Any]:
    from eval.results import make_run_record

    notes = "SYNTHETIC FIXTURE + DUMMY BACKBONE: smoke test, not a result." if args.fixture else None
    return make_run_record(run_name="", method={}, config=vars(args), seed=0, dataset={"split": args.split},
                           backbone={}, metrics={}, cost={}, notes=notes)


def _line(base, method, rule, metrics, cost) -> Dict[str, Any]:
    from eval.results import to_jsonable

    rec = dict(base, run_name=run_name(method, rule), method=method, metrics=metrics, cost=cost)
    return to_jsonable(rec)


def cmd_sweep(args: argparse.Namespace) -> int:
    import numpy as np

    from foveation.cached_eval import Rule, compose, prior_order, select_cells

    root = Path(args.root)
    tables = latency_tables(args.latency, args.fixture)
    weights = json.loads(Path(args.weights).read_text()) if args.weights else {}
    base = _base_record(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with open(out, "w", encoding="utf-8") as fh:
        def emit(rec):
            nonlocal lines
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
            lines += 1

        for ctx_model, gl_model in _pairs(args.pairs):
            ref = fullframe_curve(Path(args.fullframe), gl_model, args.split, tables)
            for scale, geom, fusion in [(s, g, f) for s in _floats(args.scales) for g in _geoms(args.geoms)
                                        for f in _strs(args.fusions)]:
                pair = load_pair(root, args.split, ctx_model, gl_model, scale, geom, fusion)
                prior_pair = pair if args.prior_split == args.split else \
                    load_pair(root, args.prior_split, ctx_model, gl_model, scale, geom, fusion)
                prior = prior_order(prior_pair)
                frame_size = pair.grid.frame_size
                key = _weights_key(f"{ctx_model}+{gl_model}", scale, geom, fusion)
                for k in _ints(args.ks):
                    if k >= pair.grid.num_cells:
                        continue
                    for rule in sweep_rules(weights, key):
                        total = compose(pair, select_cells(pair, rule, k, prior_order=prior)).sum(0)
                        metrics = metrics_of(total, pair)
                        sel = rule.selection if rule.kind == "uncertainty" else "topk"
                        cost = cost_of(ctx_model, gl_model, scale, geom, fusion, k, sel, frame_size, metrics,
                                       tables, ref)
                        emit(_line(base, method_of(ctx_model, gl_model, scale, geom, fusion, k, rule),
                                   rule, metrics, cost))
                    # random: one line, the mean over seeds
                    runs = [metrics_of(compose(pair, select_cells(pair, Rule("random", seed=s), k)).sum(0), pair)
                            for s in RANDOM_SEEDS]
                    metrics = dict(runs[0], miou=float(np.mean([r["miou"] for r in runs])),
                                   correct_pixels=float(np.mean([r["correct_pixels"] for r in runs])),
                                   random_seeds_miou=[r["miou"] for r in runs])
                    with warnings.catch_warnings():  # a class absent from the split stays NaN
                        warnings.simplefilter("ignore", RuntimeWarning)
                        metrics["per_class_iou"] = {n: float(np.nanmean([r["per_class_iou"][n] for r in runs]))
                                                    for n in runs[0]["per_class_iou"]}
                    rule = Rule("random")
                    cost = cost_of(ctx_model, gl_model, scale, geom, fusion, k, "topk", frame_size, metrics,
                                   tables, ref)
                    emit(_line(base, method_of(ctx_model, gl_model, scale, geom, fusion, k, rule), rule,
                               metrics, cost))
                # tiling bound: every cell glimpsed (no T4 point for k = cells)
                rule = Rule("all")
                metrics = metrics_of(compose(pair, select_cells(pair, rule, 0)).sum(0), pair)
                cost = cost_of(ctx_model, gl_model, scale, geom, fusion, pair.grid.num_cells, None, frame_size,
                               metrics, tables, ref)
                emit(_line(base, method_of(ctx_model, gl_model, scale, geom, fusion, pair.grid.num_cells, rule),
                           rule, metrics, cost))
                print(f"[method2 sweep] {ctx_model}+{gl_model} s{scale:g} {geom} {fusion}: {lines} lines",
                      flush=True)
    print(f"[method2 sweep] wrote {lines} lines to {out}")
    return 0


# --------------------------------------------------------------------------- #
# tune-weights
# --------------------------------------------------------------------------- #
def cmd_tune_weights(args: argparse.Namespace) -> int:
    """Coordinate ascent on per-class score weights (mean pooling, top-k), on the tuning split.

    The objective is the mean over ``--ks`` of the mIoU. For a fixed geometry and
    k the latency does not depend on the weights, so this ranks weightings exactly
    as the mean ΔmIoU at equal latency does, without its undefined values.
    """
    import numpy as np

    from eval.tradeoff import miou_of
    from foveation.cached_eval import Rule, compose, select_cells

    ctx_model, gl_model = _pairs(args.pair)[0]
    geom = _geoms(args.geom)[0]
    pair = load_pair(Path(args.root), args.split, ctx_model, gl_model, args.scale, geom, args.fusion)
    ks = [k for k in _ints(args.ks) if k < pair.grid.num_cells]
    c = pair.meta["context"]["num_classes"]

    def objective(w: Sequence[float]) -> float:
        rule = Rule("uncertainty", score=args.score, pool="mean", selection="topk", class_weights=tuple(w))
        return float(np.mean([miou_of(compose(pair, select_cells(pair, rule, k)).sum(0)) for k in ks]))

    w = [1.0] * c
    best = objective(w)
    trace = [best]
    for _ in range(args.passes):
        for cls in range(c):
            for v in WEIGHT_VALUES:
                if v == w[cls]:
                    continue
                trial = list(w)
                trial[cls] = v
                score = objective(trial)
                if score > best + 1e-12:
                    best, w = score, trial
            trace.append(best)
        print(f"[method2 tune-weights] pass done: objective {best:.5f}", flush=True)
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.is_file() else {}
    data[_weights_key(f"{ctx_model}+{gl_model}", args.scale, geom, args.fusion)] = {
        "pair": f"{ctx_model}+{gl_model}", "context_scale": args.scale, "glimpse_size": geom[0],
        "margin": geom[1], "glimpse_scale": geom[2], "fusion": args.fusion, "score": args.score, "ks": ks, "split": args.split, "values": list(WEIGHT_VALUES),
        "passes": args.passes, "weights": w, "objective": "mean mIoU over ks (top-k, mean pooling)",
        "objective_trace": trace,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"[method2 tune-weights] objective {trace[0]:.5f} -> {best:.5f}; wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# finalists
# --------------------------------------------------------------------------- #
def cmd_finalists(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.sweep).read_text().splitlines() if line.strip()]
    finalists: Dict[str, Dict[str, Any]] = {}

    def add(r: Dict[str, Any], reason: str) -> None:
        key = json.dumps(r["method"], sort_keys=True)
        entry = finalists.setdefault(key, {"method": r["method"], "latency_ms": r["cost"]["t4_latency"]["median_ms"],
                                           "miou": r["metrics"]["miou"], "split": r["dataset"]["split"],
                                           "reasons": []})
        entry["reasons"].append(reason)

    for pair in sorted({r["method"]["pair"] for r in rows}):
        cands = [r for r in rows if r["method"]["pair"] == pair and r["method"]["rule"]["kind"] not in
                 EXCLUDED_FROM_FINALISTS and r["cost"]["t4_latency"] is not None]
        cands.sort(key=lambda r: (r["cost"]["t4_latency"]["median_ms"], -r["metrics"]["miou"]))
        best = float("-inf")
        for r in cands:  # Pareto front: faster first, keep each strict mIoU improvement
            if r["metrics"]["miou"] > best:
                best = r["metrics"]["miou"]
                add(r, "pareto")
        if not cands:
            continue
        curve = cands[0]["cost"]["fullframe_curve"]
        for scale, lat in zip(curve["scales"], curve["latency_ms"]):
            within = [r for r in cands if r["cost"]["t4_latency"]["median_ms"] <= lat]
            if within:
                add(max(within, key=lambda r: r["metrics"]["miou"]), f"best_within_fullframe_s{scale:g}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": str(args.sweep), "rule": "per pair: Pareto front over (T4 median latency, mIoU) and the "
               "best mIoU within each full-frame scale's latency; oracle, random and tiling excluded",
               "finalists": list(finalists.values())}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[method2 finalists] {len(finalists)} configurations -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace) -> int:
    if args.split != "val":
        raise SystemExit("report evaluates the finalists on val, the reporting split; tune on train with sweep")
    from eval.results import to_jsonable, write_json
    from eval.tradeoff import clip_bootstrap
    from foveation.cached_eval import Rule, compose, prior_order, select_cells

    root = Path(args.root)
    tables = latency_tables(args.latency, args.fixture)
    finalists = json.loads(Path(args.finalists).read_text())["finalists"]
    base = _base_record(args)
    refs: Dict[str, Dict[str, Any]] = {}
    pairs: Dict[Tuple, Any] = {}
    priors: Dict[Tuple, Any] = {}
    results = []
    for f in finalists:
        m = f["method"]
        rule = rule_from(m["rule"])
        key = (m["context_model"], m["glimpse_model"], m["context_scale"], geom_of(m), m.get("fusion", "replace"))
        if key not in pairs:
            pairs[key] = load_pair(root, args.split, *key)
            priors[key] = prior_order(load_pair(root, args.prior_split, *key))
        pair, prior = pairs[key], priors[key]
        if m["glimpse_model"] not in refs:
            refs[m["glimpse_model"]] = fullframe_curve(Path(args.fullframe), m["glimpse_model"], args.split, tables)
        ref = refs[m["glimpse_model"]]
        per_clip = compose(pair, select_cells(pair, rule, m["k"], prior_order=prior))
        metrics = metrics_of(per_clip.sum(0), pair)
        sel = rule.selection if rule.kind == "uncertainty" else "topk"
        cost = cost_of(m["context_model"], m["glimpse_model"], m["context_scale"], geom_of(m),
                       m.get("fusion", "replace"), m["k"], sel, pair.grid.frame_size, metrics, tables, ref)
        boot = clip_bootstrap(per_clip, ref["clips"][:, pair.clip_ids], ref["latency_ms"],
                              cost["t4_latency"]["median_ms"], n=args.bootstrap, seed=0)
        references = {kind: metrics_of(compose(pair, select_cells(pair, Rule(kind), m["k"], prior_order=prior)).sum(0),
                                       pair)["miou"] for kind in ("oracle", "prior", "all")}
        rec = _line(base, m, rule, metrics, dict(cost, bootstrap=boot))
        write_json(Path(args.runs) / f"{rec['run_name']}.json", rec)
        results.append(to_jsonable({"method": m, "reasons": f.get("reasons"), "train_miou": f.get("miou"),
                                    "miou": metrics["miou"], "per_class_iou": metrics["per_class_iou"], **cost,
                                    "bootstrap": boot, "references": references}))
    out = Path(args.out)
    write_json(out, {"split": args.split, "finalists_file": str(args.finalists), "bootstrap_resamples": args.bootstrap,
                     "fullframe": {m: {k: v for k, v in r.items() if k in ("scales", "latency_ms", "miou", "full")}
                                   for m, r in refs.items()},
                     "finalists": results})
    print(f"[method2 report] {len(results)} finalists on {args.split} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    exp = REPO_ROOT / "experiments" / "method2"
    res = REPO_ROOT / "results" / "method2"

    def geometry(sp, scales="0.25,0.375,0.5"):
        sp.add_argument("--scales", default=scales)
        sp.add_argument("--geoms", default=DEFAULT_GEOMS,
                        help="Glimpse geometries input_size:input_margin:glimpse_scale, comma-separated.")
        sp.add_argument("--fusions", default=DEFAULT_FUSIONS, help="replace, mean, conf (comma-separated).")
        sp.add_argument("--root", default=str(exp / "cache"))
        sp.add_argument("--fixture", action="store_true", help="Synthetic fixture + dummy backbone (smoke test).")

    def evaluation(sp):
        sp.add_argument("--fullframe", default=str(exp / "fullframe"))
        sp.add_argument("--latency", default=DEFAULT_LATENCY, help="Latency-floor JSONs (glob, comma-separated).")
        sp.add_argument("--prior-split", default="train", help="Split whose oracle gains rank the prior rule's cells.")

    c = sub.add_parser("cache", help="Build the per-cell context and glimpse caches.")
    c.add_argument("--split", required=True)
    c.add_argument("--models", default="pidnet_s,bisenetv2")
    c.add_argument("--pairs", default=DEFAULT_PAIRS, help="Pairs whose fused glimpse caches to build.")
    geometry(c)
    c.add_argument("--max-clips", type=int, default=None)
    c.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    c.add_argument("--device", default="auto")
    c.set_defaults(fn=cmd_cache)

    s = sub.add_parser("sweep", help="Evaluate every rule x geometry x k from the caches.")
    s.add_argument("--split", default="train")
    s.add_argument("--pairs", default=DEFAULT_PAIRS)
    s.add_argument("--ks", default="1,2,4,8")
    s.add_argument("--weights", default=None, help="class_weights.json from tune-weights.")
    s.add_argument("--out", default=str(exp / "sweep_train.jsonl"))
    geometry(s)
    evaluation(s)
    s.set_defaults(fn=cmd_sweep)

    t = sub.add_parser("tune-weights", help="Coordinate ascent on per-class score weights.")
    t.add_argument("--split", default="train")
    t.add_argument("--pair", required=True)
    t.add_argument("--scale", type=float, required=True)
    t.add_argument("--geom", required=True, help="input_size:input_margin:glimpse_scale, e.g. 384:32:1")
    t.add_argument("--fusion", default="replace")
    t.add_argument("--score", default="entropy", choices=list(SCORES))
    t.add_argument("--ks", default="1,2,4,8")
    t.add_argument("--passes", type=int, default=2)
    t.add_argument("--pairs", default=None, help=argparse.SUPPRESS)
    t.add_argument("--out", default=str(res / "class_weights.json"))
    geometry(t)
    evaluation(t)
    t.set_defaults(fn=cmd_tune_weights)

    f = sub.add_parser("finalists", help="Pareto front + equal-latency picks from a sweep.")
    f.add_argument("--sweep", default=str(exp / "sweep_train.jsonl"))
    f.add_argument("--out", default=str(res / "finalists.json"))
    f.set_defaults(fn=cmd_finalists)

    r = sub.add_parser("report", help="Evaluate the finalists once on val, with clip-bootstrap intervals.")
    r.add_argument("--split", default="val")
    r.add_argument("--finalists", default=str(res / "finalists.json"))
    r.add_argument("--out", default=str(res / "val.json"))
    r.add_argument("--runs", default=str(exp / "val"))
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--pairs", default=None, help=argparse.SUPPRESS)
    geometry(r)
    evaluation(r)
    r.set_defaults(fn=cmd_report)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
