#!/usr/bin/env python3
"""Method 5, phase 1: the feasibility bound of one pass of a frozen model on an oracle-warped frame.

    python -m scripts.method5 bound --split train_all --models pidnet_s --chunk 0/4   # network passes, per clip
    python -m scripts.method5 merge --split train_all                                # chunk files -> one per model
    python -m scripts.method5 score --split train_all                                # mIoU, latency, judges, verdict

Spec (grid, latency parts and pass rule, fixed before any run):
docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md

``bound`` runs, on every labelled frame of each clip, the teacher and uniform passes and one
pass of the frozen model per configuration of the grid on the warped frame
(:func:`warp.bound.bound_clip`), and appends the clip's confusions to a ``.jsonl.gz`` file;
started again, it skips the clips already there. Each clip also gets the baseline's own resize
at each scale (``ref/c*/resize``), which traces a pipeline-check gap to the resampling or to the
reference file. ``score`` prices a configuration as the full frame at its scale
(``LatencyTables.predict_warp_frame``: backbone + post-processing; resampling, unwarping and
saliency at 0), checks the pipeline (each uniform row within 0.1 mIoU point of the full-frame
baseline at its scale) and applies the spec's pass rule as amended on 2026-11-13: the mIoU above
the full frame at the configuration's own scale, which pays the same parts. It also reads every
configuration against three full-frame curves under both judges (mean and p95): that same curve
priced with the post-processing (which gives the latency budget), the primary curve (backbone
only, the spec's first reading) and the full frame's measured end-to-end latency.

Train only: the bound is a go/no-go check, val is never read here. ``--fixture`` runs on the
synthetic fixture with the dummy backbone (a smoke test, not results).

Phase 2 (spec docs/superpowers/specs/2026-11-16-method5-phase2-design.md):

    python -m scripts.method5 cost --sessions "results/method5/phase2/latency_floor_m5w-s*.json"   # gate A
    python -m scripts.method5 prior --split train_all --models pidnet_s --chunk 0/4    # static prior P, per clip
    python -m scripts.method5 prior-merge --split train_all                           # -> phase2/prior_<model>.pt
    python -m scripts.method5 stream --split train_all --models pidnet_s --variant r2u1 --chunk 0/4   # stage B
    python -m scripts.method5 merge --kind stream --split train_all                  # one file per variant
    python -m scripts.method5 score2 --split train_all                               # gate B, finalists

``cost`` reads the latency-floor sessions of the ``warp`` part (Colab T4): per model, scale and
variant of the warp step, the overhead over the full frame paired within each session, against
the phase 1 budget at that scale, and gate A's verdict (``results/method5/phase2/gate_a.json``).
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
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method5.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _floats, _strs, latency_tables, load_dataset, model_config  # noqa: E402
from scripts.method3 import (DEFAULT_LATENCY, _dataset_and_clips, _public, _tag, _write_lines,  # noqa: E402
                             fullframe_ref, judge_readings, read_lines)
from scripts.method4 import DEFAULT_E2E, _append, e2e_curves, resume  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method5"
PASS_MARGIN = 0.010  # the spec's rule: +1.0 mIoU point above the full frame at the same scale (amended 2026-11-13)
CHECK_TOLERANCE = 0.001  # the pipeline check: each uniform row within 0.1 mIoU point of the baseline
RULE_TIMING = "now"  # the rule is read on the oracle of the current frame; "prev" is reported only


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _out_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else EXP / args.split


def grid_configs(args: argparse.Namespace) -> List[Any]:
    """The spec's grid: scale x saliency x strength x smoothing x timing, then the uniform pass
    (alpha 0) at each scale."""
    from warp.bound import WarpConfig

    scales = _floats(args.scales)
    warped = [WarpConfig(c, k, a, s, tm) for c, k, a, s, tm in itertools.product(
        scales, _strs(args.saliencies), _floats(args.alphas), _floats(args.sigmas), _strs(args.timings))]
    return warped + [WarpConfig(c) for c in scales]


def _first_resample(path: Path) -> Optional[str]:
    """The resampling the first row of a bound file was written with (None: no readable row)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return json.loads(line).get("resample")
    except (EOFError, OSError, zlib.error, json.JSONDecodeError):
        pass
    return None


# --------------------------------------------------------------------------- #
# bound
# --------------------------------------------------------------------------- #
def cmd_bound(args: argparse.Namespace) -> int:
    if args.split == "val" and not args.fixture:
        raise SystemExit("the bound is a train-only check (spec): val is not read in phase 1")
    from models.registry import build_backbone
    from warp.bound import bound_clip, resize_ref_name

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    cfgs = grid_configs(args)
    scales = sorted({c.scale for c in cfgs}, reverse=True)
    expected = {c.name() for c in cfgs} | {resize_ref_name(s) for s in scales}
    for model in _strs(args.models):
        path = _out_dir(args) / f"bound_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        if path.exists():
            written = _first_resample(path)
            if written is not None and written != args.resample:
                raise SystemExit(f"{path} was written with --resample {written}: use another --tag or delete it")
        done = resume(path, expected, label="method5 bound")
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method5 bound] {model}: {len(cfgs)} configs + {len(scales)} resize refs, {len(todo)} clips to run, "
              f"{len(clips) - len(todo)} already in {path.name}", flush=True)
        if not todo:
            continue
        backbone = build_backbone(model_config(model, args.fixture)["backbone"],
                                  context={"palette": dataset.palette}).to(device).eval()
        t0 = time.time()
        for n, i in enumerate(todo):
            clip_id = dataset.clip_id(i)
            frames = int(sum(bool(v) for v in dataset.labeled_mask(i)))
            out = bound_clip(dataset, i, backbone, cfgs, device, allow_any_frame_size=args.fixture, cell=args.cell,
                             batch=args.batch, resize_refs=True, resample=args.resample, d_max=args.d_max)
            base = {"split": args.split, "model": model, "clip_id": clip_id, "resample": args.resample,
                    "frames": frames}
            if args.d_max is not None:
                base["d_max"] = args.d_max
            rows = [{**base, "name": c.name(), "config": asdict(c), "conf": out[c.name()].tolist()} for c in cfgs]
            rows += [{**base, "name": resize_ref_name(s), "ref": "resize", "config": {"scale": s},
                      "conf": out[resize_ref_name(s)].tolist()} for s in scales]
            _append(path, rows)
            dt = time.time() - t0
            print(f"[method5 bound] {model} clip {n + 1}/{len(todo)} {clip_id} ({dt:.0f} s, "
                  f"{dt / (n + 1):.1f} s/clip)", flush=True)
        print(f"[method5 bound] {model}: {len(clips)} clips in {path}")
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
_PLURAL = {"resample": "resamplings", "variant": "variants"}


def _merge_one(files: Sequence[Path], merged: Path, field: str, label: str) -> set:
    """``files`` -> ``merged``; the clips written. Refuses a (config, clip) seen twice, a clip
    without every config of the others and files whose rows differ in ``field`` (``resample``
    for bound files, ``variant`` for stream files)."""
    seen: Dict[Tuple[str, str], str] = {}
    per_clip: Counter = Counter()
    kinds = set()

    def rows():  # one line in memory at a time
        for f in files:
            for r in read_lines(f):
                key = (r["name"], r["clip_id"])
                if key in seen:
                    raise SystemExit(f"duplicate line {key} in {f.name} and {seen[key]}: a chunk merged twice?")
                seen[key] = f.name
                per_clip[r["clip_id"]] += 1
                kinds.add(r.get(field))
                yield r

    n = _write_lines(merged, rows())
    if len(set(per_clip.values())) > 1 or len(kinds) > 1:
        merged.unlink()
        what = (f"clips with different numbers of configs ({sorted(set(per_clip.values()))}): a cut-off chunk? "
                "Run it again" if len(set(per_clip.values())) > 1 else
                f"files written with different {_PLURAL[field]} {sorted(map(str, kinds))}")
        raise SystemExit(f"[{label}] {merged.name}: {what}")
    print(f"[{label}] {len(files)} files, {n} lines, {len(per_clip)} clips -> {merged}")
    return set(per_clip)


def cmd_merge(args: argparse.Namespace) -> int:
    """Chunk files -> ``bound_<split>_<model>.jsonl.gz`` (``--kind bound``) or, per variant,
    ``stream_<split>_<model>_<variant>.jsonl.gz`` (``--kind stream``; folder
    experiments/method5/phase2/<split> by default). Refuses a (config, clip) seen twice, a clip
    without every config of the others and files written with different resamplings or variants;
    lists the clips missing against the dataset's."""
    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    stream = getattr(args, "kind", "bound") == "stream"
    out_dir = _stream_dir(args) if stream else _out_dir(args)
    for model in _strs(args.models):
        if stream:
            jobs = [(out_dir / f"stream_{args.split}_{model}_{v}.jsonl.gz", fs, "variant")
                    for v, fs in _variant_files(out_dir, args.split, model).items()]
        else:
            merged = out_dir / f"bound_{args.split}_{model}.jsonl.gz"
            fs = sorted(f for f in out_dir.glob(f"bound_{args.split}_{model}_*.jsonl.gz") if f != merged)
            jobs = [(merged, fs, "resample")] if fs else []
        if not jobs:
            print(f"[method5 merge] {model}: no chunk file in {out_dir}")
            continue
        for merged, fs, field in jobs:
            clips = _merge_one(fs, merged, field, f"method5 merge {model}")
            missing = sorted((expected or set()) - clips)
            if missing:
                print(f"[method5 merge] {model}: missing clips {missing}")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _iou(total) -> List[float]:
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        return list(np.diag(total) / (total.sum(0) + total.sum(1) - np.diag(total)))


def same_parts_curve(ref: Dict[str, Any], tables, model: str) -> Dict[str, Any]:
    """The full-frame curve with each scale priced as a pass of the bound is (backbone and
    post-processing, :meth:`LatencyTables.predict_warp_frame`), so that both sides of a comparison
    pay the same parts (spec amendment of 2026-11-13). At a configuration's own scale, its ΔmIoU at
    equal latency on this curve is its mIoU minus the full frame's at that scale."""
    from eval.tradeoff import FullFrameCurve

    lat = [tables.predict_warp_frame(model, s) for s in ref["scales"]]
    med, p95 = [x["median_ms"] for x in lat], [x["p95_ms"] for x in lat]
    return {"scales": list(ref["scales"]), "miou": list(ref["miou"]), "median_ms": med, "p95_ms": p95,
            "curves": {"mean": FullFrameCurve(med, ref["miou"]), "p95": FullFrameCurve(p95, ref["miou"])}}


def summarise(name: str, config: Dict[str, Any], total, clips: int, frames: int, tables, model: str,
              class_names: Sequence[str], ref: Dict[str, Any], e2e: Optional[Dict[str, Any]],
              same: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One configuration over its clips: mIoU, per-class IoU, latency (the same on every frame:
    the mean judge reads the median, the p95 judge the p95), both judges against each reference:
    ``judges`` the primary curve, ``judges_e2e`` the measured end-to-end one, ``judges_same_parts``
    the full frame priced with the post-processing (:func:`same_parts_curve`)."""
    from eval.tradeoff import miou_of

    lat = tables.predict_warp_frame(model, float(config["scale"]))
    miou = miou_of(total)
    latency = {"mean_ms": lat["median_ms"], "p95_ms": lat["p95_ms"], "frames": frames}
    out = {"name": name, "model": model, "config": config, "miou": miou,
           "per_class_iou": {n: (None if math.isnan(v) else float(v)) for n, v in zip(class_names, _iou(total))},
           "latency": latency, "latency_parts": lat["parts"], "gflops": lat["gflops"], "clips": clips,
           "judges": judge_readings(miou, latency["mean_ms"], latency["p95_ms"], ref)}
    if e2e is not None:
        out["judges_e2e"] = judge_readings(miou, latency["mean_ms"], latency["p95_ms"], e2e)
    # what the resampling, the unwarping and a real policy may cost per frame before the config is
    # back on the full frame's curve, both paying the backbone and the post-processing (mean judge;
    # the primary curve's reading, which does not count the full frame's post-processing, is in
    # ``judges``)
    if same is not None:
        out["judges_same_parts"] = judge_readings(miou, latency["mean_ms"], latency["p95_ms"], same)
    # Read on the frontier of that curve: the cheapest full frame that reaches the config's mIoU.
    # ``latency_saved_at_equal_miou`` in the judges is NaN as soon as mIoU does not rise with
    # scale (BiSeNetV2 on the 400 clips: 0.7339 at 0.75, 0.7335 at 1); the frontier drops the
    # beaten scale and is the same number on a rising curve (decided 2026-11-16, before the
    # reading on the 400 clips; the budget is not in the pass rule).
    curve = (same if same is not None else ref)["curves"]["mean"]
    out["budget_ms"] = curve.latency_at(miou, frontier=True) - latency["mean_ms"]
    return out


def pipeline_check(configs: Sequence[Dict[str, Any]], resize: Dict[float, float],
                   ref: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per scale of the bound: the uniform row's mIoU against the full-frame baseline's at that
    scale (``ok`` when within :data:`CHECK_TOLERANCE`; no baseline at that scale is not ok), with
    the run's own resize (``resize``: scale -> mIoU) to tell a resampling gap from a reference gap."""
    out: Dict[str, Dict[str, Any]] = {}
    for c in configs:
        if c["config"].get("alpha", 0.0) != 0.0:
            continue
        s = float(c["config"]["scale"])
        base = ref["miou"][ref["scales"].index(s)] if s in ref["scales"] else None
        gap = None if base is None else c["miou"] - base
        res = resize.get(s)
        out[f"{s:g}"] = {"scale": s, "uniform_miou": c["miou"], "baseline_miou": base, "gap": gap,
                         "ok": gap is not None and abs(gap) <= CHECK_TOLERANCE + 1e-12, "tolerance": CHECK_TOLERANCE,
                         "resize_miou": res, "uniform_minus_resize": None if res is None else c["miou"] - res,
                         "resize_minus_baseline": None if res is None or base is None else res - base}
    return out


def _best(configs: Sequence[Dict[str, Any]], judges: str, judge: str, saliency: Optional[str] = None,
          timing: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The warped config (alpha > 0) of ``saliency`` / ``timing`` (None: any) with the highest
    ΔmIoU at equal latency that has a reading."""
    cands = [c for c in configs if c["config"].get("alpha", 0.0) > 0 and judges in c
             and (saliency is None or c["config"]["saliency"] == saliency)
             and (timing is None or c["config"]["timing"] == timing)
             and not math.isnan(c[judges][judge]["delta_miou_at_equal_latency"])]
    if not cands:
        return None
    c = max(cands, key=lambda c: c[judges][judge]["delta_miou_at_equal_latency"])
    return {"name": c["name"], "miou": c["miou"], "latency_ms": c[judges][judge]["latency_ms"],
            "delta_miou_at_equal_latency": c[judges][judge]["delta_miou_at_equal_latency"],
            "latency_saved_at_equal_miou": c[judges][judge]["latency_saved_at_equal_miou"],
            "delta_vs_uniform": c.get("delta_vs_uniform"), "readable": len(cands)}


def _best_same_size(configs: Sequence[Dict[str, Any]], timing: str) -> Optional[Dict[str, Any]]:
    """The warped config of ``timing`` with the highest mIoU above the full frame at its own scale."""
    cands = [c for c in configs if c["config"].get("alpha", 0.0) > 0 and c["config"]["timing"] == timing
             and c.get("delta_vs_fullframe") is not None and not math.isnan(c["delta_vs_fullframe"])]
    if not cands:
        return None
    c = max(cands, key=lambda c: c["delta_vs_fullframe"])
    return {"name": c["name"], "scale": c["config"]["scale"], "miou": c["miou"],
            "delta_vs_fullframe": c["delta_vs_fullframe"], "delta_vs_uniform": c.get("delta_vs_uniform"),
            "latency_ms": c.get("latency", {}).get("mean_ms"), "budget_ms": c.get("budget_ms"),
            "readable": len(cands)}


def verdict_of(configs: Sequence[Dict[str, Any]], complete: bool, clips: int,
               check: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The spec's pass rule for one model, as amended on 2026-11-13: the best warped config with
    timing ``now``, read as its mIoU minus the full frame's at the same scale. Both pay the same
    backbone and post-processing there (resampling and unwarping at 0), so this is the comparison
    at equal latency. Read only on the whole clip set and a passed pipeline check.

    ``curves`` is information only, not the rule: the best ΔmIoU at equal mean latency against the
    primary curve (backbone only: the spec's first reading), the measured end-to-end curve and the
    full frame priced with the post-processing."""
    best = _best_same_size(configs, RULE_TIMING)
    prev = _best_same_size(configs, "prev")
    delta = best["delta_vs_fullframe"] if best else None
    failed = sorted((k for k, v in check.items() if not v["ok"]), key=float)
    if not complete:
        passes, why = None, f"not read (partial: {clips} clips)"
    elif failed or not check:
        passes, why = None, (f"not read (pipeline check failed at scales {', '.join(failed)}: the uniform warp is "
                             f"more than {CHECK_TOLERANCE * 100:g} mIoU point from the full-frame baseline)"
                             if failed else "not read (no uniform row: no pipeline check)")
    else:
        passes = delta is not None and delta >= PASS_MARGIN
        why = "PASSES" if passes else "FAILS"
    note = None
    if passes and (prev is None or prev["delta_vs_fullframe"] < 0):
        note = "passes, but the memory version starts under water (best prev config below the full frame at its scale)"
    curves = {}
    for key, judges in (("primary", "judges"), ("e2e", "judges_e2e"), ("same_parts", "judges_same_parts")):
        curves[key] = {"best": _best(configs, judges, "mean", timing=RULE_TIMING),
                       "best_prev": _best(configs, judges, "mean", timing="prev")}
    return {"rule": f"best warped config with timing {RULE_TIMING} >= +{PASS_MARGIN * 100:g} mIoU points above the "
                    "full frame at the same scale (same backbone and post-processing: equal latency), on every clip "
                    "of train_all (spec, amended 2026-11-13)",
            "margin": PASS_MARGIN, "clips": clips, "complete": complete, "pipeline_check_ok": not failed and bool(check),
            "best": best, "best_prev": prev, "passes": passes, "state": why, "note": note,
            "curves": {"reading": "best ΔmIoU at equal mean latency on each full-frame curve (information only, "
                                  "not the rule)", **curves}}


def cmd_score(args: argparse.Namespace) -> int:
    import numpy as np

    tables = latency_tables(args.latency, args.fixture)
    out_dir = _out_dir(args)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split) if not args.no_expect else None
    names = list(dataset.class_names) if dataset is not None else [str(c) for c in range(19)]
    expected = {dataset.clip_id(i) for i in range(len(dataset))} if dataset is not None else None
    verdicts: Dict[str, Any] = {}
    for model in _strs(args.models):
        path = Path(args.bound) if args.bound else out_dir / f"bound_{args.split}_{model}.jsonl.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run `python -m scripts.method5 merge --split {args.split}`")
        totals: Dict[str, Any] = {}
        clips_of: Dict[str, set] = {}
        configs_of: Dict[str, Dict[str, Any]] = {}
        frames: Dict[str, int] = {}
        resamples = set()
        for r in read_lines(path):
            name = r["name"]
            conf = np.asarray(r["conf"], dtype=np.int64)
            totals[name] = conf if name not in totals else totals[name] + conf
            clips_of.setdefault(name, set()).add(r["clip_id"])
            configs_of.setdefault(name, {**r["config"], **({"ref": r["ref"]} if "ref" in r else {})})
            frames[r["clip_id"]] = r.get("frames", 0)
            resamples.add(r.get("resample"))
        if len(resamples) > 1:
            raise SystemExit(f"{path}: rows written with different resamplings {sorted(map(str, resamples))}")
        clip_ids = set().union(*clips_of.values())
        short = [n for n, c in clips_of.items() if c != clip_ids]
        if short:
            raise SystemExit(f"{path}: configs {short[:3]} lack clips: merge again from complete chunks")
        ref = fullframe_ref(Path(args.fullframe), model, args.split, tables, sorted(clip_ids))
        e2e = e2e_curves(args.e2e, model, ref)
        same = same_parts_curve(ref, tables, model)
        n_frames = sum(frames.values())
        from eval.tradeoff import miou_of

        resize = {float(configs_of[n]["scale"]): miou_of(t) for n, t in totals.items() if "ref" in configs_of[n]}
        configs = [summarise(n, configs_of[n], t, len(clip_ids), n_frames, tables, model, names, ref, e2e, same)
                   for n, t in totals.items() if "ref" not in configs_of[n]]
        uniform = {float(c["config"]["scale"]): c["miou"] for c in configs if c["config"].get("alpha", 0.0) == 0.0}
        baseline = dict(zip(ref["scales"], ref["miou"]))
        for c in configs:
            u, b = uniform.get(float(c["config"]["scale"])), baseline.get(float(c["config"]["scale"]))
            c["delta_vs_uniform"] = None if u is None else c["miou"] - u
            c["delta_vs_fullframe"] = None if b is None else c["miou"] - b  # the full frame at the same scale
        check = pipeline_check(configs, resize, ref)
        complete = expected is not None and clip_ids == expected
        verdict = verdicts[model] = verdict_of(configs, complete, len(clip_ids), check)
        best = {f"{k}/{tm}/{judges}/{judge}": _best(configs, judges, judge, k, tm)
                for k in ("E", "B", "S") for tm in ("now", "prev")
                for judges in ("judges", "judges_e2e", "judges_same_parts") for judge in ("mean", "p95")}
        payload = {"split": args.split, "model": model, "bound": str(path), "resample": resamples.pop(),
                   "fullframe": _public(ref),
                   "fullframe_e2e": None if e2e is None else {k: v for k, v in e2e.items() if k != "curves"},
                   "fullframe_same_parts": {k: v for k, v in same.items() if k != "curves"},
                   "pipeline_check": check, "verdict": verdict, "best": best, "configs": configs}
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"score_{args.split}_{model}.json"
        out.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
        b = verdict["best"]
        shown = "no readable warped config" if b is None else (
            f"best now {b['name']}: {b['delta_vs_fullframe'] * 100:+.2f} pts vs the full frame at c{b['scale']:g} "
            f"({b['latency_ms']:.2f} ms)")
        gaps = ", ".join(f"c{k} {'ok' if v['ok'] else 'NOT ok'}"
                         + ("" if v["gap"] is None else f" ({v['gap'] * 100:+.2f} pts)") for k, v in check.items())
        print(f"[method5 score] {model}: {len(configs)} configs over {len(clip_ids)} clips; pipeline check: {gaps}; "
              f"{shown}; rule {verdict['state']} -> {out}")
    if verdicts and all(v["complete"] for v in verdicts.values()):
        states = [v["passes"] for v in verdicts.values()]
        passes = None if any(p is None for p in states) else any(states)
        summary = {"split": args.split, "passes": passes, "models": verdicts}
        (out_dir / f"verdict_{args.split}.json").write_text(json.dumps(summary, indent=1, allow_nan=True) + "\n")
        state = {True: "PASSES", False: "FAILS", None: "is NOT READ (a pipeline check failed)"}[passes]
        print(f"[method5 score] feasibility check {state} (models: {', '.join(verdicts)})")
    return 0


# --------------------------------------------------------------------------- #
# Phase 2: the static prior P (prior, prior-merge)
# --------------------------------------------------------------------------- #
EXP2 = EXP / "phase2"


def _prior_scales(args: argparse.Namespace, model: str) -> List[float]:
    return _floats(args.scales) if args.scales else list(STAGE_B.get(model, COST_SCALES))


def cmd_prior(args: argparse.Namespace) -> int:
    """Per clip, the E counts per cell at each stage B scale of the model and the density rounds its
    frames need (:func:`warp.prior.prior_clip`), appended to a ``.jsonl.gz`` file (resumable)."""
    if args.split != "train_all" and not args.fixture:
        raise SystemExit("the prior is fitted on train_all (spec); val is its out-of-sample reading, never fitted on")
    from models.registry import build_backbone
    from warp.prior import prior_clip, prior_name

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    out_dir = Path(args.out_dir) if args.out_dir else EXP2
    for model in _strs(args.models):
        scales = _prior_scales(args, model)
        expected = {prior_name(s) for s in scales} | {"rounds"}
        path = out_dir / f"prior_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        done = resume(path, expected, label="method5 prior")
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method5 prior] {model}: scales {scales}, {len(todo)} clips to run, {len(clips) - len(todo)} already "
              f"in {path.name}", flush=True)
        if not todo:
            continue
        backbone = build_backbone(model_config(model, args.fixture)["backbone"],
                                  context={"palette": dataset.palette}).to(device).eval()
        t0 = time.time()
        for n, i in enumerate(todo):
            clip_id = dataset.clip_id(i)
            out = prior_clip(dataset, i, backbone, scales, device, allow_any_frame_size=args.fixture, cell=args.cell)
            base = {"split": args.split, "model": model, "clip_id": clip_id, "frames": out["frames"],
                    "cell": args.cell}
            rows = [{**base, "name": prior_name(s), "scale": s, "counts": out[prior_name(s)].tolist()} for s in scales]
            rows.append({**base, "name": "rounds", "rounds": out["rounds"]})
            _append(path, rows)
            dt = time.time() - t0
            print(f"[method5 prior] {model} clip {n + 1}/{len(todo)} {clip_id} ({dt:.0f} s, {dt / (n + 1):.1f} s/clip)",
                  flush=True)
        print(f"[method5 prior] {model}: {len(clips)} clips in {path}")
    return 0


def cmd_prior_merge(args: argparse.Namespace) -> int:
    """Chunk files -> ``prior_<model>.pt`` (a list of :func:`warp.policy.prior_record`, one per scale)
    and ``prior_<model>.json`` (clips, frames, mean E share, the most density rounds needed).
    Refuses a (row, clip) seen twice, a clip without every row, mixed cells, and missing clips."""
    import torch

    from warp.policy import prior_record

    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    src = Path(args.out_dir) if args.out_dir else EXP2
    dest = Path(args.dest) if args.dest else PHASE2
    for model in _strs(args.models):
        files = sorted(src.glob(f"prior_{args.split}_{model}_*.jsonl.gz"))
        if not files:
            raise SystemExit(f"[method5 prior-merge] {model}: no chunk file in {src}")
        seen: Dict[Tuple[str, str], str] = {}
        names: Dict[str, set] = {}
        sums: Dict[float, Any] = {}
        frames: Dict[float, int] = {}
        rounds: Dict[str, int] = {}
        cells = set()
        for f in files:
            for r in read_lines(f):
                key = (r["name"], r["clip_id"])
                if key in seen:
                    raise SystemExit(f"[method5 prior-merge] {key} in {f.name} and {seen[key]}: a chunk merged twice?")
                seen[key] = f.name
                names.setdefault(r["clip_id"], set()).add(r["name"])
                cells.add(r["cell"])
                if r["name"] == "rounds":
                    for k, v in r["rounds"].items():
                        rounds[k] = max(rounds.get(k, 0), int(v))
                    continue
                s = float(r["scale"])
                c = torch.tensor(r["counts"], dtype=torch.int64)
                sums[s] = c if s not in sums else sums[s] + c
                frames[s] = frames.get(s, 0) + int(r["frames"])
        if len({frozenset(v) for v in names.values()}) > 1:
            raise SystemExit(f"[method5 prior-merge] {model}: clips with different rows: a cut-off chunk?")
        if len(cells) != 1:
            raise SystemExit(f"[method5 prior-merge] {model}: chunks written with different cells {sorted(cells)}")
        if expected is not None and set(names) != expected:
            missing = sorted(expected - set(names))
            raise SystemExit(f"[method5 prior-merge] {model}: {len(missing)} clips missing, e.g. {missing[:5]}")
        cell = cells.pop()
        records = [prior_record(model, s, sums[s].to(torch.float64) / cell ** 2, frames[s])
                   for s in sorted(sums)]
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(records, dest / f"prior_{model}.pt")
        summary = {"model": model, "split": args.split, "clips": len(names), "cell": cell, "files": [f.name for f in files],
                   "scales": {f"{s:g}": {"frames": frames[s],
                                         "mean_e_share": float(sums[s].sum()) / (cell ** 2 * frames[s] * sums[s].numel())}
                              for s in sorted(sums)},
                   "max_rounds": dict(sorted(rounds.items())), "max_rounds_all": max(rounds.values(), default=0)}
        (dest / f"prior_{model}.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"[method5 prior-merge] {model}: {len(files)} files, {len(names)} clips, "
              + ", ".join(f"c{k} {v['frames']} frames E {v['mean_e_share'] * 100:.1f} %" for k, v in summary["scales"].items())
              + f"; density rounds needed at most {summary['max_rounds_all']} -> {dest / f'prior_{model}.pt'}")
    return 0


# --------------------------------------------------------------------------- #
# Phase 2, stage A: the cost of the warp step on a T4 (gate A)
# --------------------------------------------------------------------------- #
PHASE2 = REPO_ROOT / "results" / "method5" / "phase2"
STAGE_B = {"pidnet_s": (0.25, 0.375, 0.5), "bisenetv2": (0.25, 0.375)}  # spec decisions 4 and 5
GATE_SESSIONS = 3  # the spec: three separate Colab T4 sessions
COST_SCALES = (0.25, 0.375, 0.5)


def _read_json(path: Path) -> Any:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        return json.load(f)


def _score_file(scores: Path, model: str) -> Path:
    for name in (f"score_train_all_{model}.json.gz", f"score_train_all_{model}.json"):
        if (scores / name).is_file():
            return scores / name
    raise SystemExit(f"no phase 1 score of {model} in {scores}")


def stage_a_budgets(scores: Path, e2e: str, scales: Sequence[float] = COST_SCALES) -> Dict[str, Dict[str, Any]]:
    """Per ``model/scale``: phase 1's budget, the time a real policy may spend per frame. The
    registered grid's best ``now`` configuration at that scale (``scores``: its committed score
    files, alpha 0.25 / 0.5 / 0.75) read on the measured end-to-end full-frame curve
    (``latency_floor_final`` sessions, :func:`scripts.method4.e2e_curves`): the latency at which
    the curve reaches its mIoU (frontier), minus the full frame's at the same scale."""
    out: Dict[str, Dict[str, Any]] = {}
    for model in STAGE_B:
        data = _read_json(_score_file(scores, model))
        curves = e2e_curves(e2e, model, data["fullframe"])
        if curves is None:
            raise SystemExit(f"no latency_floor final session in {e2e} covers {model}'s scales")
        for c in scales:
            cands = [x for x in data["configs"] if x["config"].get("alpha", 0.0) > 0
                     and x["config"].get("timing") == RULE_TIMING and float(x["config"]["scale"]) == c]
            if not cands:
                raise SystemExit(f"no phase 1 configuration of {model} at scale {c:g}")
            best = max(cands, key=lambda x: x["miou"])
            ff_ms = curves["median_ms"][curves["scales"].index(c)]
            out[f"{model}/{c:g}"] = {
                "model": model, "scale": c, "best": best["name"], "best_miou": best["miou"],
                "fullframe_miou": data["fullframe"]["miou"][data["fullframe"]["scales"].index(c)],
                "fullframe_e2e_ms": ff_ms,
                "budget_ms": curves["curves"]["mean"].latency_at(best["miou"], frontier=True) - ff_ms,
                "in_stage_b": c in STAGE_B[model]}
    return out


def variant_of(r: Dict[str, Any]) -> str:
    """A variant of the warp step, as gate A reads it: resampling, unwarp, saliency, rounds."""
    return f"{r['resample']}{r['unwarp']}/{r['saliency']}/n{r['rounds']}"


def _status(lo: float, hi: float, budget: float) -> str:
    if hi <= budget:
        return "fits"
    return "straddles" if lo <= budget else "over"


def gate_a(sessions: Sequence[Dict[str, Any]], budgets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The spec's gate A on latency-floor sessions of the ``warp`` part.

    Per session, each W case is paired with the full frame of its own chunk (same model and
    scale, timed in the same interleaved rounds): Δs = median W - median FF. Per variant: Δ, the
    median of the Δs, their range, and ``fits`` (largest Δs within the budget), ``straddles``
    (smallest within, largest over) or ``over`` (smallest over). A (model, scale) takes the best
    status of its variants and keeps the variants that fit or straddle. Verdict: ``stop`` when
    every (model, scale) of stage B is ``over``, else ``continue``; not read (None) without
    :data:`GATE_SESSIONS` complete T4 sessions."""
    rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    info = []
    for sess in sessions:
        recs = [r for r in sess["records"].get("warp", []) if r.get("level") == "graph"]
        ff = {(r["model"], float(r["scale"]), r["chunk"]): r for r in recs if r["kind"] == "fullframe"}
        for r in recs:
            if r["kind"] != "warp":
                continue
            f = ff.get((r["model"], float(r["scale"]), r["chunk"]))
            if f is None:
                raise ValueError(f"session {sess['session']}: {r['name']} has no full frame in its chunk {r['chunk']}")
            w, b = r["cuda_event"], f["cuda_event"]
            rows.setdefault((f"{r['model']}/{float(r['scale']):g}", variant_of(r)), []).append(
                {"session": sess["session"], "chunk": r["chunk"], "w_ms": w["median_ms"], "ff_ms": b["median_ms"],
                 "w_p95_ms": w["p95_ms"], "ff_p95_ms": b["p95_ms"], "delta_ms": w["median_ms"] - b["median_ms"]})
        env = sess.get("environment", {})
        info.append({"session": sess["session"], "gpu": env.get("gpu"), "quick": bool(sess.get("quick")),
                     "started": sess.get("started"), "t4": "T4" in str(env.get("gpu")),
                     "sm_clock_mhz": sess.get("session_telemetry", {}).get("sm_clock_mhz")})
    n = len(sessions)
    variants: Dict[str, Dict[str, Any]] = {}
    for (group, variant), per in sorted(rows.items()):
        if group not in budgets:
            raise ValueError(f"no phase 1 budget for {group}")
        deltas = [p["delta_ms"] for p in per]
        lo, hi, budget = min(deltas), max(deltas), budgets[group]["budget_ms"]
        full = len(per) == n
        variants.setdefault(group, {})[variant] = {
            "delta_ms": statistics.median(deltas), "delta_range_ms": [lo, hi], "sessions": len(per),
            "w_ms": statistics.median(p["w_ms"] for p in per), "ff_ms": statistics.median(p["ff_ms"] for p in per),
            "w_p95_ms": statistics.median(p["w_p95_ms"] for p in per),
            "ff_p95_ms": statistics.median(p["ff_p95_ms"] for p in per),
            "status": _status(lo, hi, budget) if full else None, "per_session": per}
    groups: Dict[str, Dict[str, Any]] = {}
    for group, b in budgets.items():
        vs = variants.get(group, {})
        read = {k: v for k, v in vs.items() if v["status"] is not None}
        cheapest = min(read, key=lambda k: read[k]["delta_ms"]) if read else None
        statuses = {v["status"] for v in read.values()}
        status = next((s for s in ("fits", "straddles", "over") if s in statuses), None)
        groups[group] = {**b, "status": status, "cheapest": cheapest,
                         "cheapest_delta_ms": None if cheapest is None else read[cheapest]["delta_ms"],
                         "kept": sorted(k for k, v in read.items() if v["status"] in ("fits", "straddles")),
                         "variants": vs}
    stage_b = [g for g, v in groups.items() if v["in_stage_b"]]
    problems = []
    if n < GATE_SESSIONS:
        problems.append(f"{n} sessions (the spec needs {GATE_SESSIONS})")
    if any(not s["t4"] or s["quick"] for s in info):
        problems.append("a session is not a full T4 run")
    if any(groups[g]["status"] is None or any(v["status"] is None for v in groups[g]["variants"].values())
           for g in stage_b):
        problems.append("a variant is missing from a session")
    if problems:
        verdict, state, goes_on = None, "not read (" + "; ".join(problems) + ")", []
    elif all(groups[g]["status"] == "over" for g in stage_b):
        verdict, state, goes_on = "stop", "STOP: even the cheapest variant is over the budget everywhere", []
    else:
        verdict = "continue"
        goes_on = [{"group": g, "status": groups[g]["status"], "variants": groups[g]["kept"],
                    "note": "not decided by the timing (the sessions straddle the budget)"
                    if groups[g]["status"] == "straddles" else None}
                   for g in stage_b if groups[g]["status"] != "over"]
        state = "CONTINUE with " + ", ".join(f"{x['group']} ({x['status']})" for x in goes_on)
    return {"rule": "gate A (spec, stage A): Δs = median W - median FF per session, paired within its chunk; stop "
                    "when the smallest Δs of every variant is over the phase 1 budget at every model and scale of "
                    "stage B; a straddle goes on, not decided by the timing",
            "sessions": info, "verdict": verdict, "state": state, "goes_on": goes_on, "groups": groups}


def cmd_cost(args: argparse.Namespace) -> int:
    paths = sorted(p for part in args.sessions.split(",")
                   for p in glob.glob(part if Path(part).is_absolute() else str(REPO_ROOT / part)))
    if not paths:
        raise SystemExit(f"no session matches {args.sessions}")
    sessions = [_read_json(Path(p)) for p in paths]
    out = gate_a(sessions, stage_a_budgets(Path(args.scores), args.e2e))
    out["files"] = [Path(p).relative_to(REPO_ROOT).as_posix() if Path(p).is_relative_to(REPO_ROOT) else p
                    for p in paths]
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    for group, g in out["groups"].items():
        tag = "" if g["in_stage_b"] else "  (not in stage B)"
        print(f"[method5 cost] {group}: budget {g['budget_ms']:.3f} ms, status {g['status']}, cheapest {g['cheapest']}"
              + ("" if g["cheapest_delta_ms"] is None else f" (Δ {g['cheapest_delta_ms']:+.3f} ms)") + tag)
        for name, v in sorted(g["variants"].items(), key=lambda kv: kv[1]["delta_ms"]):
            lo, hi = v["delta_range_ms"]
            print(f"    {name:18s} Δ {v['delta_ms']:+.3f} ms [{lo:+.3f}, {hi:+.3f}]  W {v['w_ms']:.2f}  "
                  f"FF {v['ff_ms']:.2f}  {v['status']}")
    print(f"[method5 cost] gate A: {out['state']} -> {dest}")
    return 0


# --------------------------------------------------------------------------- #
# Phase 2, stage B: the closed loop (stream, merge --kind stream, score2)
# --------------------------------------------------------------------------- #
STREAM_VARIANTS = ("r1u1", "r1u2", "r2u1", "r2u2", "r3u1", "r3u2")
CHECKED_RESAMPLES = ("r1", "r2")  # the baseline's own filter: the pipeline check holds them to it; r3 is reported


def _split_variant(variant: str) -> Tuple[str, str]:
    if variant not in STREAM_VARIANTS:
        raise SystemExit(f"--variant must be one of {STREAM_VARIANTS}, got {variant!r}")
    return variant[:2], variant[2:]


def _priced_as(cfg: Dict[str, Any], variant: str) -> str:
    """The stage A variant that prices a configuration. A uniform row is a fixed warp like P's
    (same resampling and unwarp with operands made once), so it is priced as P."""
    sal = "P" if cfg["saliency"] == "uniform" else cfg["saliency"]
    return f"{variant}/{sal}/n{int(cfg['rounds'])}"


def stream_configs(args: argparse.Namespace, model: str) -> List[Any]:
    """The stage B grid of ``model`` for one variant: scale x saliency x strength x smoothing, then
    the uniform row at each scale; or, with ``--finalists``, the model's finalists of that variant."""
    from warp.policy import StepConfig

    resample, unwarp = _split_variant(args.variant)
    if args.finalists:
        data = json.loads(Path(args.finalists).read_text())
        if data.get("gate_b") != "continue" and not args.fixture:
            raise SystemExit(f"{args.finalists}: gate B is {data.get('gate_b')!r}, not 'continue': val is not read")
        picked = [f for f in data["finalists"] if f["model"] == model and f["variant"] == args.variant]
        return [StepConfig(**f["config"]) for f in picked]
    scales = _floats(args.scales) if args.scales else list(STAGE_B.get(model, COST_SCALES))
    warped = [StepConfig(c, k, a, s, resample, unwarp, args.rounds) for c, k, a, s in itertools.product(
        scales, _strs(args.saliencies), _floats(args.alphas), _floats(args.sigmas))]
    return warped + [StepConfig(c, resample=resample, unwarp=unwarp, rounds=args.rounds) for c in scales]


def _check_gate_a(gate: Dict[str, Any], model: str, cfgs: Sequence[Any], variant: str) -> None:
    """Refuses a configuration whose variant gate A did not keep at its model and scale."""
    if gate.get("verdict") != "continue":
        raise SystemExit(f"gate A is {gate.get('verdict')!r}: stage B does not run")
    for c in cfgs:
        group = gate["groups"].get(f"{model}/{c.scale:g}")
        want = _priced_as(asdict(c), variant)
        if group is None or want not in group.get("kept", []):
            raise SystemExit(f"{c.name()}: gate A did not keep {want} for {model} at scale {c.scale:g}")


def _first_field(path: Path, field: str) -> Optional[Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return json.loads(line).get(field)
    except (EOFError, OSError, zlib.error, json.JSONDecodeError):
        pass
    return None


def _stream_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else EXP2 / args.split


def cmd_stream(args: argparse.Namespace) -> int:
    """Per clip, every configuration of the grid (or the finalists) in its own closed loop
    (:func:`warp.stream.stream_clip`), one confusion per labelled position, appended to
    ``stream_<split>_<model>_<variant>_<tag>.jsonl.gz`` (resumable)."""
    if args.split == "val" and not (args.finalists or args.fixture):
        raise SystemExit("val is read once, with the finalists chosen on train: stream --split val needs --finalists")
    import torch

    from models.registry import build_backbone
    from warp.policy import load_prior
    from warp.stream import build_steps, stream_clip

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    gate = None if args.fixture and not args.gate else _read_json(Path(args.gate or PHASE2 / "gate_a.json"))
    for model in _strs(args.models):
        cfgs = stream_configs(args, model)
        if not cfgs:
            print(f"[method5 stream] {model}: no configuration of variant {args.variant}")
            continue
        if gate is not None:
            _check_gate_a(gate, model, cfgs, args.variant)
        prior_file = Path(args.priors) / f"prior_{model}.pt"
        priors = {c.scale: load_prior(prior_file, model, c.scale) for c in cfgs if c.saliency == "P"}
        path = _stream_dir(args) / f"stream_{args.split}_{model}_{args.variant}_{_tag(args)}.jsonl.gz"
        if path.exists():
            written = _first_field(path, "variant")
            if written is not None and written != args.variant:
                raise SystemExit(f"{path} was written with variant {written}: use another --tag or delete it")
        done = resume(path, {c.name() for c in cfgs}, label="method5 stream")
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        every = True if args.every_frame else None
        print(f"[method5 stream] {model} {args.variant}: {len(cfgs)} configs, {len(todo)} clips to run, "
              f"{len(clips) - len(todo)} already in {path.name}", flush=True)
        if not todo:
            continue
        backbone = build_backbone(model_config(model, args.fixture)["backbone"],
                                  context={"palette": dataset.palette}).to(device).eval()
        steps = build_steps(backbone, cfgs, {s: p.to(torch.float32) for s, p in priors.items()}, cell=args.cell)
        t0 = time.time()
        for n, i in enumerate(todo):
            clip_id = dataset.clip_id(i)
            positions, out = stream_clip(dataset, i, backbone, steps, device, batch=args.batch,
                                         allow_any_frame_size=args.fixture, every_frame=every)
            base = {"split": args.split, "model": model, "clip_id": clip_id, "variant": args.variant,
                    "frames": len(positions), "positions": positions, "cell": args.cell}
            _append(path, [{**base, "name": c.name(), "config": asdict(c), "conf": out[c.name()].tolist()}
                           for c in cfgs])
            dt = time.time() - t0
            print(f"[method5 stream] {model} {args.variant} clip {n + 1}/{len(todo)} {clip_id} ({dt:.0f} s, "
                  f"{dt / (n + 1):.1f} s/clip)", flush=True)
        print(f"[method5 stream] {model} {args.variant}: {len(clips)} clips in {path}")
    return 0


def _variant_files(out_dir: Path, split: str, model: str) -> Dict[str, List[Path]]:
    """Chunk files of the stream runs per variant (``stream_<split>_<model>_<variant>_<tag>``)."""
    prefix = f"stream_{split}_{model}_"
    out: Dict[str, List[Path]] = {}
    for f in sorted(out_dir.glob(f"{prefix}*.jsonl.gz")):
        rest = f.name[len(prefix):-len(".jsonl.gz")]
        variant, _, tag = rest.partition("_")
        if variant in STREAM_VARIANTS and tag:
            out.setdefault(variant, []).append(f)
    return out


def score_stream(path: Path, model: str, variant: str, gate: Dict[str, Any], ref: Dict[str, Any],
                 e2e: Dict[str, Any], oracle: Dict[str, float], class_names: Sequence[str]) -> Dict[str, Any]:
    """Every configuration of one merged stream file: mIoU, per-class IoU, mIoU per labelled
    position, the measured price of its variant (stage A's W, median and p95 over the sessions),
    both judges against the measured end-to-end full-frame curve, the full frame and the uniform
    row at the same scale, and the share of the phase 1 oracle's gain kept."""
    import numpy as np

    from eval.tradeoff import miou_of

    totals: Dict[str, Any] = {}
    clips_of: Dict[str, set] = {}
    cfg_of: Dict[str, Dict[str, Any]] = {}
    positions = set()
    for r in read_lines(path):
        if r.get("variant") != variant:
            raise SystemExit(f"{path}: a row of variant {r.get('variant')} in the {variant} file")
        conf = np.asarray(r["conf"], dtype=np.int64)
        totals[r["name"]] = conf if r["name"] not in totals else totals[r["name"]] + conf
        clips_of.setdefault(r["name"], set()).add(r["clip_id"])
        cfg_of.setdefault(r["name"], r["config"])
        positions.add(tuple(r["positions"]))
    clip_ids = set().union(*clips_of.values()) if clips_of else set()
    short = [n for n, c in clips_of.items() if c != clip_ids]
    if short:
        raise SystemExit(f"{path}: configs {short[:3]} lack clips: merge again from complete chunks")
    if len(positions) > 1:
        raise SystemExit(f"{path}: clips with different labelled positions {sorted(positions)[:3]}")
    pos = list(positions.pop()) if positions else []
    baseline = dict(zip(ref["scales"], ref["miou"]))
    configs = []
    for name, total in totals.items():
        cfg = cfg_of[name]
        scale = float(cfg["scale"])
        group = gate["groups"].get(f"{model}/{scale:g}", {})
        price = group.get("variants", {}).get(_priced_as(cfg, variant))
        if price is None:
            raise SystemExit(f"{name}: no stage A timing of {_priced_as(cfg, variant)} for {model} at {scale:g}")
        miou = miou_of(total.sum(0))
        mean_ms, p95_ms = float(price["w_ms"]), float(price["w_p95_ms"])
        judges = judge_readings(miou, mean_ms, p95_ms, e2e)
        for judge, ms in (("mean", mean_ms), ("p95", p95_ms)):
            judges[judge]["latency_saved_frontier"] = e2e["curves"][judge].latency_at(miou, frontier=True) - ms
        base = baseline.get(scale)
        orc = oracle.get(f"5o/c{scale:g}/E/a{float(cfg['alpha']):g}/s{float(cfg['sigma']):g}/now")
        delta = None if base is None else miou - base
        orc_delta = None if orc is None or base is None else orc - base
        configs.append({
            "name": name, "model": model, "variant": variant, "config": cfg, "miou": miou,
            "per_class_iou": {n: (None if math.isnan(v) else float(v)) for n, v in zip(class_names, _iou(total.sum(0)))},
            "miou_by_position": {str(t): miou_of(total[j]) for j, t in enumerate(pos)},
            "latency": {"mean_ms": mean_ms, "p95_ms": p95_ms, "priced_as": _priced_as(cfg, variant),
                        "delta_vs_paired_fullframe_ms": price.get("delta_ms")},
            "clips": len(clip_ids), "judges_e2e": judges, "delta_vs_fullframe": delta,
            "oracle": None if orc is None else {"miou": orc, "delta_vs_fullframe": orc_delta,
                                                "share_kept": (delta / orc_delta) if delta is not None and orc_delta
                                                and orc_delta > 0 else None}})
    uniform = {float(c["config"]["scale"]): c for c in configs if c["config"]["saliency"] == "uniform"}
    for c in configs:
        u = uniform.get(float(c["config"]["scale"]))
        c["delta_vs_uniform"] = None if u is None else c["miou"] - u["miou"]
    check = {}
    for s, u in sorted(uniform.items()):
        base = baseline.get(s)
        gap = None if base is None else u["miou"] - base
        checked = variant[:2] in CHECKED_RESAMPLES
        check[f"{s:g}"] = {"scale": s, "uniform_miou": u["miou"], "baseline_miou": base, "gap": gap,
                           "checked": checked, "tolerance": CHECK_TOLERANCE,
                           "ok": (gap is not None and abs(gap) <= CHECK_TOLERANCE + 1e-12) if checked else None}
    return {"variant": variant, "file": str(path), "clips": sorted(clip_ids), "positions": pos,
            "pipeline_check": check, "configs": configs}


def gate_b(configs: Sequence[Dict[str, Any]], checks: Dict[str, Dict[str, Any]], complete: bool,
           clips: int) -> Dict[str, Any]:
    """The spec's gate B for one model: ``continue`` when a warped configuration lies above the
    measured full-frame curve at equal median latency (mean judge) on every clip of the split,
    ``stop`` otherwise; not read (None) on a partial set or when a checked pipeline check fails.
    The best configuration is reported with its uniform row's own reading, so that a gain that a
    cheaper resampling alone would give (r3) is visible."""
    failed = sorted(f"{v}/c{k}" for v, ch in checks.items() for k, x in ch.items() if x["checked"] and not x["ok"])
    warped = [c for c in configs if c["config"]["saliency"] != "uniform"
              and not math.isnan(c["judges_e2e"]["mean"]["delta_miou_at_equal_latency"])]
    best = max(warped, key=lambda c: c["judges_e2e"]["mean"]["delta_miou_at_equal_latency"], default=None)
    above = [c["name"] for c in warped if c["judges_e2e"]["mean"]["delta_miou_at_equal_latency"] > 0]
    if not complete:
        verdict, state = None, f"not read (partial: {clips} clips)"
    elif failed:
        verdict, state = None, f"not read (pipeline check failed at {', '.join(failed)})"
    else:
        verdict = "continue" if above else "stop"
        state = (f"CONTINUE: {len(above)} configurations above the measured full-frame curve" if above else
                 "STOP: no configuration above the measured full-frame curve at equal latency")
    shown = None
    if best is not None:
        uni = next((c for c in configs if c["config"]["saliency"] == "uniform" and c["variant"] == best["variant"]
                    and c["config"]["scale"] == best["config"]["scale"]), None)
        shown = {"name": best["name"], "miou": best["miou"], "latency_ms": best["latency"]["mean_ms"],
                 "delta_miou_at_equal_latency": best["judges_e2e"]["mean"]["delta_miou_at_equal_latency"],
                 "delta_vs_fullframe": best["delta_vs_fullframe"], "delta_vs_uniform": best["delta_vs_uniform"],
                 "uniform_delta_miou_at_equal_latency": None if uni is None else
                 uni["judges_e2e"]["mean"]["delta_miou_at_equal_latency"]}
    return {"rule": "gate B (spec, stage B): continue when, for at least one model, a configuration lies above the "
                    "measured full-frame curve at equal mean latency on train (all clips; pipeline check of r1/r2 "
                    "passed)", "clips": clips, "complete": complete, "verdict": verdict, "state": state,
            "above": above, "best": shown}


def finalists_of(configs: Sequence[Dict[str, Any]], e2e: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
    """As Methods 2 and 3, per judge (mean: median latency, p95): the Pareto front over (latency,
    mIoU) of the warped configurations, and the best mIoU within each full-frame scale's measured
    latency; the union. The uniform rows are references, not finalists."""
    picked: Dict[str, Dict[str, Any]] = {}
    cands = [c for c in configs if c["config"]["saliency"] != "uniform"]
    for judge, key, ff_ms in (("mean", "mean_ms", e2e["median_ms"]), ("p95", "p95_ms", e2e["p95_ms"])):
        def add(c: Dict[str, Any], reason: str) -> None:
            entry = picked.setdefault(c["name"], {
                "model": model, "name": c["name"], "variant": c["variant"], "config": c["config"],
                "train": {"miou": c["miou"], "mean_ms": c["latency"]["mean_ms"], "p95_ms": c["latency"]["p95_ms"]},
                "reasons": []})
            entry["reasons"].append(reason)

        best = float("-inf")
        for c in sorted(cands, key=lambda c: (c["latency"][key], -c["miou"])):
            if c["miou"] > best:
                best = c["miou"]
                add(c, f"pareto_{judge}")
        for scale, lat in zip(e2e["scales"], ff_ms):
            within = [c for c in cands if c["latency"][key] <= lat]
            if within:
                add(max(within, key=lambda c: c["miou"]), f"best_within_fullframe_s{scale:g}_{judge}")
    return list(picked.values())


def _oracle_mious(spec: str, model: str) -> Dict[str, float]:
    """Phase 1 configuration name -> mIoU on train_all, from the registered and extension scores."""
    out: Dict[str, float] = {}
    for folder in spec.split(","):
        d = Path(folder) if Path(folder).is_absolute() else REPO_ROOT / folder
        for name in (f"score_train_all_{model}.json.gz", f"score_train_all_{model}.json"):
            if (d / name).is_file():
                for c in _read_json(d / name)["configs"]:
                    out.setdefault(c["name"], c["miou"])
                break
    return out


def _rel(path: Any) -> str:
    """A path as written in a result file: relative to the repository and with forward slashes
    when it is inside it, so that a file scored on Windows reads the same everywhere."""
    p = Path(path).resolve()
    return p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else p.as_posix()


def cmd_score2(args: argparse.Namespace) -> int:
    """Stage B's reading of the merged stream files: per configuration (:func:`score_stream`),
    gate B per model and overall (:func:`gate_b`), and the finalists (:func:`finalists_of`)."""
    tables = latency_tables(args.latency, args.fixture)
    out_dir = _stream_dir(args)
    gate = _read_json(Path(args.gate))
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split) if not args.no_expect else None
    names = list(dataset.class_names) if dataset is not None else [str(c) for c in range(19)]
    expected = {dataset.clip_id(i) for i in range(len(dataset))} if dataset is not None else None
    fin_dir = Path(args.finalists_dir) if args.finalists_dir else PHASE2
    gates: Dict[str, Any] = {}
    for model in _strs(args.models):
        variants = sorted({f.name[len(f"stream_{args.split}_{model}_"):-len(".jsonl.gz")]
                           for f in out_dir.glob(f"stream_{args.split}_{model}_*.jsonl.gz")} & set(STREAM_VARIANTS))
        if not variants:
            raise SystemExit(f"no merged stream file of {model} in {out_dir}: run `merge --kind stream`")
        clip_sets = []
        scored = []
        for v in variants:
            path = out_dir / f"stream_{args.split}_{model}_{v}.jsonl.gz"
            clips = sorted({r["clip_id"] for r in read_lines(path)})
            ref = fullframe_ref(Path(args.fullframe), model, args.split, tables, clips)
            e2e = e2e_curves(args.e2e, model, ref)
            if e2e is None:
                raise SystemExit(f"no latency_floor final session in {args.e2e} covers {model}'s scales")
            scored.append(score_stream(path, model, v, gate, ref, e2e, _oracle_mious(args.oracle_scores, model), names))
            clip_sets.append(set(clips))
        if any(s != clip_sets[0] for s in clip_sets):
            raise SystemExit(f"{model}: the variants' files hold different clips")
        configs = [c for s in scored for c in s["configs"]]
        checks = {s["variant"]: s["pipeline_check"] for s in scored}
        complete = expected is not None and clip_sets[0] == expected
        g = gates[model] = gate_b(configs, checks, complete, len(clip_sets[0]))
        payload = {"split": args.split, "model": model, "gate_a": _rel(args.gate), "fullframe": _public(ref),
                   "fullframe_e2e": {k: v for k, v in e2e.items() if k != "curves"}, "gate_b": g,
                   "pipeline_check": checks, "files": [_rel(s["file"]) for s in scored],
                   "positions": scored[0]["positions"],
                   "configs": configs}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"score2_{args.split}_{model}.json").write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
        fin = finalists_of(configs, e2e, model)
        fin_dir.mkdir(parents=True, exist_ok=True)
        (fin_dir / f"finalists_{model}.json").write_text(json.dumps(
            {"split": args.split, "model": model, "gate_b": g["verdict"],
             "rule": "per judge (mean: median latency, p95): the Pareto front over (latency, mIoU) of the warped "
                     "configurations and the best mIoU within each full-frame scale's measured latency; the union",
             "finalists": fin}, indent=1) + "\n")
        b = g["best"]
        shown = "no readable warped config" if b is None else (
            f"best {b['name']}: {b['delta_miou_at_equal_latency'] * 100:+.2f} pts at equal latency "
            f"({b['latency_ms']:.2f} ms), {b['delta_vs_fullframe'] * 100:+.2f} vs the full frame at its scale")
        checks_shown = "; ".join(f"{v} " + ", ".join(
            f"c{k} {'ok' if x['ok'] else ('reported' if not x['checked'] else 'NOT ok')}"
            + ("" if x["gap"] is None else f" ({x['gap'] * 100:+.2f})") for k, x in ch.items()) for v, ch in checks.items())
        print(f"[method5 score2] {model}: {len(configs)} configs over {len(clip_sets[0])} clips; pipeline check: "
              f"{checks_shown}; {shown}; gate B {g['state']}; {len(fin)} finalists")
    if gates and all(g["complete"] for g in gates.values()):
        states = [g["verdict"] for g in gates.values()]
        verdict = None if any(v is None for v in states) else ("continue" if "continue" in states else "stop")
        (out_dir / f"gate_b_{args.split}.json").write_text(json.dumps(
            {"split": args.split, "verdict": verdict, "models": gates}, indent=1, allow_nan=True) + "\n")
        print(f"[method5 score2] gate B: {verdict or 'NOT READ'} (models: {', '.join(gates)})")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from warp.bound import RESAMPLE, RESAMPLES

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="train_all")
        sp.add_argument("--models", default="pidnet_s,bisenetv2")
        sp.add_argument("--clip-ids", default=None, help="Comma-separated clip ids (default: all, or --chunk).")
        sp.add_argument("--chunk", default=None, help="i/N: the i-th contiguous block of the sorted clip ids.")
        sp.add_argument("--tag", default=None, help="Output file tag (default: from --chunk / --clip-ids).")
        sp.add_argument("--out-dir", default=None, help="Folder of the jsonl files (default experiments/method5/<split>).")
        sp.add_argument("--fixture", action="store_true", help="Synthetic fixture + dummy backbone (smoke test).")
        sp.add_argument("--device", default="auto")

    b = sub.add_parser("bound", help="Every configuration of the grid on the labelled frames, per clip (resumable).")
    common(b)
    b.add_argument("--scales", default="0.25,0.375,0.5,0.75", help="Input scales c (the warped image is c x the frame).")
    b.add_argument("--saliencies", default="E,B,S", help="E: error, B: boundaries, S: small classes.")
    b.add_argument("--alphas", default="0.25,0.5,0.75", help="Warp strengths (the uniform pass, 0, is always run).")
    b.add_argument("--sigmas", default="1,4", help="Saliency smoothing, in cells.")
    b.add_argument("--timings", default="now,prev", help="now: teacher of frame t; prev: of frame t - 1.")
    b.add_argument("--d-max", type=float, default=None,
                   help="Upper clip of the density in place of 1/scale: above it the warp magnifies beyond the "
                        "source pixels (extension of 2026-11-16, with --scales 1; keep it in its own --out-dir).")
    b.add_argument("--cell", type=int, default=16, help="Saliency cell, px (16: a 64 x 128 grid).")
    b.add_argument("--batch", type=int, default=4, help="Warped images of one scale per forward pass.")
    b.add_argument("--resample", default=RESAMPLE, choices=RESAMPLES,
                   help="antialias (the spec's, amended 2026-11-13): the baseline's filter, exact at alpha 0; "
                        "supersample: the plan's 2x supersampling + 2 x 2 mean, not run.")
    b.set_defaults(fn=cmd_bound)

    m = sub.add_parser("merge", help="Chunk files -> one bound file per model.")
    common(m)
    m.add_argument("--no-expect", action="store_true", help="Do not compare against the dataset's clips.")
    m.add_argument("--kind", default="bound", choices=("bound", "stream"),
                   help="bound: phase 1 files; stream: phase 2 stage B files, one merged file per variant "
                        "(default folder experiments/method5/phase2/<split>).")
    m.set_defaults(fn=cmd_merge)

    s = sub.add_parser("score", help="Per config: mIoU, latency, both judges, both references; the pipeline check "
                                     "and the verdict.")
    common(s)
    s.add_argument("--bound", default=None, help="Score this bound file (default: the merged one; one model only).")
    s.add_argument("--fullframe", default=str(REPO_ROOT / "experiments" / "method2" / "fullframe"))
    s.add_argument("--latency", default=DEFAULT_LATENCY, help="Latency-floor JSONs (glob, comma-separated).")
    s.add_argument("--e2e", default=DEFAULT_E2E, help="Latency-floor `final` sessions: the measured full-frame curve.")
    s.add_argument("--no-expect", action="store_true",
                   help="Do not load the dataset (clip list, class names): the verdict is then not read.")
    s.set_defaults(fn=cmd_score)

    pr = sub.add_parser("prior", help="Phase 2: E counts per cell and density rounds per clip, for the static prior "
                                      "P (train_all; resumable).")
    common(pr)
    pr.add_argument("--scales", default=None, help="Default: the stage B scales of the model (spec).")
    pr.add_argument("--cell", type=int, default=16, help="Saliency cell, px (16: a 64 x 128 grid).")
    pr.set_defaults(fn=cmd_prior)

    pm = sub.add_parser("prior-merge", help="Phase 2: chunk files -> results/method5/phase2/prior_<model>.pt.")
    common(pm)
    pm.add_argument("--dest", default=None, help="Folder of the prior files (default results/method5/phase2).")
    pm.add_argument("--no-expect", action="store_true", help="Do not compare against the dataset's clips.")
    pm.set_defaults(fn=cmd_prior_merge)

    st = sub.add_parser("stream", help="Phase 2, stage B: every configuration in its own closed loop over each clip, "
                                       "one confusion per labelled position (resumable).")
    common(st)
    st.add_argument("--variant", required=True, choices=STREAM_VARIANTS,
                    help="Resampling and unwarp, as gate A kept them (one variant per file).")
    st.add_argument("--scales", default=None, help="Default: the stage B scales of the model (spec).")
    st.add_argument("--saliencies", default="P",
                    help="Default P: gate A kept only the static prior (2026-11-17); B, S, U are refused by gate A.")
    st.add_argument("--alphas", default="0.1,0.15,0.25", help="Warp strengths (the uniform row is always run).")
    st.add_argument("--sigmas", default="1,4", help="Saliency smoothing, in cells.")
    st.add_argument("--rounds", type=int, default=50, help="Density rounds (gate A timed 50).")
    st.add_argument("--cell", type=int, default=16, help="Saliency cell, px (16: a 64 x 128 grid).")
    st.add_argument("--batch", type=int, default=4, help="Warped images of one scale per forward pass.")
    st.add_argument("--priors", default=str(PHASE2), help="Folder of prior_<model>.pt (prior-merge).")
    st.add_argument("--gate", default=None, help="gate_a.json (default results/method5/phase2/gate_a.json; "
                                                 "not read with --fixture unless given).")
    st.add_argument("--finalists", default=None, help="Run only these configurations (finalists_<model>.json); "
                                                      "required on val.")
    st.add_argument("--every-frame", action="store_true",
                    help="Run every frame even when every configuration is fixed (same confusions, 5 times slower).")
    st.set_defaults(fn=cmd_stream)

    s2 = sub.add_parser("score2", help="Phase 2, stage B: per configuration mIoU and measured price, the pipeline "
                                       "check, gate B and the finalists.")
    common(s2)
    s2.add_argument("--gate", default=str(PHASE2 / "gate_a.json"), help="gate_a.json: the measured price W(c).")
    s2.add_argument("--fullframe", default=str(REPO_ROOT / "experiments" / "method2" / "fullframe"))
    s2.add_argument("--latency", default=DEFAULT_LATENCY, help="Latency-floor JSONs (glob, comma-separated).")
    s2.add_argument("--e2e", default=DEFAULT_E2E, help="Latency-floor `final` sessions: the measured full-frame curve.")
    s2.add_argument("--oracle-scores", default="results/method5,results/method5/ext",
                    help="Folders of the phase 1 scores (the oracle's gain, alpha 0.25 and 0.1/0.15).")
    s2.add_argument("--finalists-dir", default=None, help="Where finalists_<model>.json go (default "
                                                          "results/method5/phase2).")
    s2.add_argument("--no-expect", action="store_true",
                    help="Do not load the dataset (clip list, class names): gate B is then not read.")
    s2.set_defaults(fn=cmd_score2)

    c = sub.add_parser("cost", help="Phase 2, stage A: the warp step's overhead over the full frame on a T4, paired "
                                    "per session, against the phase 1 budget (gate A).")
    c.add_argument("--sessions", default=str((PHASE2 / "latency_floor_m5w-s*.json").relative_to(REPO_ROOT)),
                   help="Latency-floor sessions of the `warp` part (glob, comma-separated).")
    c.add_argument("--scores", default=str(REPO_ROOT / "results" / "method5"),
                   help="Folder of the phase 1 registered scores (the budget).")
    c.add_argument("--e2e", default=DEFAULT_E2E, help="Latency-floor `final` sessions: the measured full-frame curve.")
    c.add_argument("--out", default=str(PHASE2 / "gate_a.json"))
    c.set_defaults(fn=cmd_cost)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
