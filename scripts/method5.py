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
saliency at 0), reads it against the full-frame curve of the same model under both judges (mean
and p95), against the full frame's measured end-to-end latency too, checks the pipeline (each
uniform row within 0.1 mIoU point of the full-frame baseline at its scale), and applies the
spec's pass rule.

Train only: the bound is a go/no-go check, val is never read here. ``--fixture`` runs on the
synthetic fixture with the dummy backbone (a smoke test, not results).
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
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
PASS_MARGIN = 0.010  # the spec's rule: +1.0 mIoU point above the curve at equal mean latency
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
                             batch=args.batch, resize_refs=True, resample=args.resample)
            base = {"split": args.split, "model": model, "clip_id": clip_id, "resample": args.resample,
                    "frames": frames}
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
def cmd_merge(args: argparse.Namespace) -> int:
    """Chunk files -> ``bound_<split>_<model>.jsonl.gz``. Refuses a (config, clip) seen twice, a
    clip without every config of the others and files written with different resamplings; lists
    the clips missing against the dataset's."""
    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    out_dir = _out_dir(args)
    for model in _strs(args.models):
        merged = out_dir / f"bound_{args.split}_{model}.jsonl.gz"
        files = sorted(f for f in out_dir.glob(f"bound_{args.split}_{model}_*.jsonl.gz") if f != merged)
        if not files:
            print(f"[method5 merge] {model}: no chunk file in {out_dir}")
            continue
        seen: Dict[Tuple[str, str], str] = {}
        per_clip: Counter = Counter()
        resamples = set()

        def rows():  # one line in memory at a time
            for f in files:
                for r in read_lines(f):
                    key = (r["name"], r["clip_id"])
                    if key in seen:
                        raise SystemExit(f"duplicate line {key} in {f.name} and {seen[key]}: a chunk merged twice?")
                    seen[key] = f.name
                    per_clip[r["clip_id"]] += 1
                    resamples.add(r.get("resample"))
                    yield r

        n = _write_lines(merged, rows())
        if len(set(per_clip.values())) > 1 or len(resamples) > 1:
            merged.unlink()
            what = (f"clips with different numbers of configs ({sorted(set(per_clip.values()))}): a cut-off chunk? "
                    "Run `bound` again on it" if len(set(per_clip.values())) > 1 else
                    f"files written with different resamplings {sorted(map(str, resamples))}")
            raise SystemExit(f"[method5 merge] {model}: {what}")
        print(f"[method5 merge] {model}: {len(files)} files, {n} lines, {len(per_clip)} clips -> {merged}")
        missing = sorted((expected or set()) - set(per_clip))
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


def summarise(name: str, config: Dict[str, Any], total, clips: int, frames: int, tables, model: str,
              class_names: Sequence[str], ref: Dict[str, Any], e2e: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One configuration over its clips: mIoU, per-class IoU, latency (the same on every frame:
    the mean judge reads the median, the p95 judge the p95), both judges against both references."""
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
    # back on the curve (mean judge)
    out["budget_ms"] = out["judges"]["mean"]["latency_saved_at_equal_miou"]
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
             and c.get("delta_vs_fullframe") is not None]
    if not cands:
        return None
    c = max(cands, key=lambda c: c["delta_vs_fullframe"])
    return {"name": c["name"], "miou": c["miou"], "delta_vs_fullframe": c["delta_vs_fullframe"],
            "delta_vs_uniform": c.get("delta_vs_uniform"), "readable": len(cands)}


def verdict_of(configs: Sequence[Dict[str, Any]], complete: bool, clips: int,
               check: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The spec's pass rule for one model: the best warped config with timing ``now`` at equal mean
    latency against the primary curve, read only on the whole clip set and a passed pipeline check.

    ``same_size`` is information only, not the rule: the same margin read on the mIoU above the
    full frame at the configuration's own scale, which is the configuration priced as that full
    frame on any reference (the primary curve does not count the full frame's post-processing,
    while a configuration's latency does)."""
    best = _best(configs, "judges", "mean", timing=RULE_TIMING)
    prev = _best(configs, "judges", "mean", timing="prev")
    delta = best["delta_miou_at_equal_latency"] if best else None
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
    if passes and (prev is None or prev["delta_miou_at_equal_latency"] < 0):
        note = "passes, but the memory version starts under water (best prev config below the curve)"
    same = _best_same_size(configs, RULE_TIMING)
    return {"rule": f"best warped config with timing {RULE_TIMING} >= +{PASS_MARGIN * 100:g} mIoU points above "
                    "the full-frame curve at equal mean latency, on every clip of train_all",
            "margin": PASS_MARGIN, "clips": clips, "complete": complete, "pipeline_check_ok": not failed and bool(check),
            "best": best, "best_prev": prev, "passes": passes, "state": why, "note": note,
            "same_size": {"reading": "mIoU minus the full frame's at the same scale (information only, not the rule)",
                          "best": same, "best_prev": _best_same_size(configs, "prev"),
                          "at_margin": None if same is None else same["delta_vs_fullframe"] >= PASS_MARGIN}}


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
        n_frames = sum(frames.values())
        from eval.tradeoff import miou_of

        resize = {float(configs_of[n]["scale"]): miou_of(t) for n, t in totals.items() if "ref" in configs_of[n]}
        configs = [summarise(n, configs_of[n], t, len(clip_ids), n_frames, tables, model, names, ref, e2e)
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
                for k in ("E", "B", "S") for tm in ("now", "prev") for judges in ("judges", "judges_e2e")
                for judge in ("mean", "p95")}
        payload = {"split": args.split, "model": model, "bound": str(path), "resample": resamples.pop(),
                   "fullframe": _public(ref),
                   "fullframe_e2e": None if e2e is None else {k: v for k, v in e2e.items() if k != "curves"},
                   "pipeline_check": check, "verdict": verdict, "best": best, "configs": configs}
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"score_{args.split}_{model}.json"
        out.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
        b = verdict["best"]
        shown = "no readable warped config" if b is None else (
            f"best now {b['name']}: {b['delta_miou_at_equal_latency'] * 100:+.2f} pts at {b['latency_ms']:.2f} ms")
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
    b.add_argument("--cell", type=int, default=16, help="Saliency cell, px (16: a 64 x 128 grid).")
    b.add_argument("--batch", type=int, default=4, help="Warped images of one scale per forward pass.")
    b.add_argument("--resample", default=RESAMPLE, choices=RESAMPLES,
                   help="supersample: the spec's 2x supersampling + 2 x 2 mean; antialias: the baseline's filter.")
    b.set_defaults(fn=cmd_bound)

    m = sub.add_parser("merge", help="Chunk files -> one bound file per model.")
    common(m)
    m.add_argument("--no-expect", action="store_true", help="Do not compare against the dataset's clips.")
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
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
