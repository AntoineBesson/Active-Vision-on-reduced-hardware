#!/usr/bin/env python3
"""Method 6: the feasibility bound of one pass of a frozen model on an oracle, locally (2-D) warped frame.

    python -m scripts.method6 bound --split train_all --models pidnet_s --chunk 0/4   # network passes, per clip
    python -m scripts.method6 merge --split train_all                                # chunk files -> one per model
    python -m scripts.method6 score --split train_all                                # checks, comparison, verdict

Spec (grid, latency parts and pass rule, fixed before any run):
docs/superpowers/specs/2026-11-17-method6-local-warp-bound-design.md

``bound`` runs, on every labelled frame of each clip, the teacher and the uniform passes, one pass
of the frozen model per local configuration on the warped frame (:func:`warp.bound2d.bound_clip2d`),
the uniform row through the 2-D path (the pipeline check) and, per scale, Method 5's best separable
configuration through Method 5's code (:data:`SEP_BAR`; the reproduction check). ``score`` prices
every configuration as the full frame at its scale, as Method 5 phase 1, checks both, compares
each local configuration with the full frame and with Method 5's best separable configuration at
its scale, and applies the pass rule.

Train only. ``--fixture`` runs on the synthetic fixture with the dummy backbone (a smoke test).
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # `python scripts/method6.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _floats, _strs, latency_tables, load_dataset, model_config  # noqa: E402
from scripts.method3 import DEFAULT_LATENCY, _dataset_and_clips, _public, _tag, fullframe_ref, read_lines  # noqa: E402
from scripts.method4 import DEFAULT_E2E, _append, e2e_curves, resume  # noqa: E402
from scripts.method5 import CHECK_TOLERANCE, _merge_one, pipeline_check, same_parts_curve, summarise  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method6"
M5_SCORES = "results/method5,results/method5/ext"
PASS_MARGIN = 0.010  # (a): +1.0 mIoU point above the full frame at the same scale
SEP_MARGIN = 0.005  # (b): +0.5 mIoU point above Method 5's best separable now configuration at that scale
REPRO_TOLERANCE = 0.001  # each separable row within 0.1 mIoU point of Method 5's

# Method 5's best separable configuration with timing now, per model and scale, on the 400 clips of
# train_all (registered grid and the alpha 0.1 / 0.15 extension; spec table). ``score`` checks these
# against the files; ``bound`` runs them as the reproduction rows: (saliency, alpha, sigma).
SEP_BAR = {
    "pidnet_s": {0.25: ("E", 0.25, 1.0), 0.375: ("E", 0.15, 1.0), 0.5: ("E", 0.15, 4.0)},
    "bisenetv2": {0.25: ("E", 0.25, 4.0), 0.375: ("E", 0.15, 4.0), 0.5: ("B", 0.15, 4.0)},
}
SEP_DEFAULT = ("E", 0.25, 1.0)  # other models (the fixture's dummy)


def _out_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else EXP / args.split


def grid_configs(args: argparse.Namespace, model: str) -> List[Any]:
    """The spec's grid (scale x saliency x strength x smoothing x kernel), the uniform row and the
    separable reproduction row at each scale."""
    from warp.bound2d import LocalConfig

    scales = _floats(args.scales)
    local = [LocalConfig(c, k, a, s, kern) for c, k, a, s, kern in itertools.product(
        scales, _strs(args.saliencies), _floats(args.alphas), _floats(args.sigmas), _floats(args.kernels))]
    bars = SEP_BAR.get(model, {})
    sep = [] if args.no_sep else [LocalConfig(c, *bars.get(c, SEP_DEFAULT), form="sep") for c in scales]
    return local + [LocalConfig(c) for c in scales] + sep


# --------------------------------------------------------------------------- #
# bound
# --------------------------------------------------------------------------- #
def cmd_bound(args: argparse.Namespace) -> int:
    if args.split == "val" and not args.fixture:
        raise SystemExit("the bound is a train-only check (spec): val is not read")
    from models.registry import build_backbone
    from warp.bound import resize_ref_name
    from warp.bound2d import bound_clip2d

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    for model in _strs(args.models):
        cfgs = grid_configs(args, model)
        scales = sorted({c.scale for c in cfgs}, reverse=True)
        expected = {c.name() for c in cfgs} | {resize_ref_name(s) for s in scales}
        path = _out_dir(args) / f"bound_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        done = resume(path, expected, label="method6 bound")
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method6 bound] {model}: {len(cfgs)} configs + {len(scales)} resize refs, {len(todo)} clips to run, "
              f"{len(clips) - len(todo)} already in {path.name}", flush=True)
        if not todo:
            continue
        backbone = build_backbone(model_config(model, args.fixture)["backbone"],
                                  context={"palette": dataset.palette}).to(device).eval()
        t0 = time.time()
        for n, i in enumerate(todo):
            clip_id = dataset.clip_id(i)
            frames = int(sum(bool(v) for v in dataset.labeled_mask(i)))
            out, dist = bound_clip2d(dataset, i, backbone, cfgs, device, allow_any_frame_size=args.fixture,
                                     cell=args.cell, batch=args.batch)
            base = {"split": args.split, "model": model, "clip_id": clip_id, "frames": frames}
            rows = [{**base, "name": c.name(), "config": asdict(c), "conf": out[c.name()].tolist(),
                     **({"dist": dist[c.name()]} if c.name() in dist else {})} for c in cfgs]
            rows += [{**base, "name": resize_ref_name(s), "ref": "resize", "config": {"scale": s},
                      "conf": out[resize_ref_name(s)].tolist()} for s in scales]
            _append(path, rows)
            dt = time.time() - t0
            print(f"[method6 bound] {model} clip {n + 1}/{len(todo)} {clip_id} ({dt:.0f} s, "
                  f"{dt / (n + 1):.1f} s/clip)", flush=True)
        print(f"[method6 bound] {model}: {len(clips)} clips in {path}")
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args: argparse.Namespace) -> int:
    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    out_dir = _out_dir(args)
    for model in _strs(args.models):
        merged = out_dir / f"bound_{args.split}_{model}.jsonl.gz"
        fs = sorted(f for f in out_dir.glob(f"bound_{args.split}_{model}_*.jsonl.gz") if f != merged)
        if not fs:
            print(f"[method6 merge] {model}: no chunk file in {out_dir}")
            continue
        clips = _merge_one(fs, merged, "split", f"method6 merge {model}")
        missing = sorted((expected or set()) - clips)
        if missing:
            print(f"[method6 merge] {model}: missing clips {missing}")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def method5_rows(dirs: str, split: str, model: str) -> Dict[str, Dict[str, Any]]:
    """Method 5's scored configurations (name -> row) from ``score_<split>_<model>.json[.gz]`` in
    each of the comma-separated ``dirs`` (registered grid, extension)."""
    rows: Dict[str, Dict[str, Any]] = {}
    for d in _strs(dirs):
        base = Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        for name in (f"score_{split}_{model}.json.gz", f"score_{split}_{model}.json"):
            if (base / name).is_file():
                for c in _load_json(base / name)["configs"]:
                    rows.setdefault(c["name"], c)
                break
    return rows


def separable_bars(m5: Dict[str, Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    """Per scale: Method 5's best warped configuration with timing now, by mIoU above the full frame."""
    out: Dict[float, Dict[str, Any]] = {}
    for c in m5.values():
        cf = c["config"]
        if cf.get("alpha", 0.0) <= 0 or cf.get("timing") != "now" or c.get("delta_vs_fullframe") is None:
            continue
        s = float(cf["scale"])
        if s not in out or c["delta_vs_fullframe"] > out[s]["delta_vs_fullframe"]:
            out[s] = {"name": c["name"], "miou": c["miou"], "delta_vs_fullframe": c["delta_vs_fullframe"]}
    return out


def reproduction_check(configs: Sequence[Dict[str, Any]], m5: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Each separable row against the same configuration in Method 5's files."""
    from warp.bound2d import LocalConfig

    out: Dict[str, Dict[str, Any]] = {}
    for c in configs:
        if c["config"].get("form") != "sep":
            continue
        twin = LocalConfig(**c["config"]).method5_name()
        ref = m5.get(twin)
        gap = None if ref is None else c["miou"] - ref["miou"]
        out[c["name"]] = {"method5": twin, "miou": c["miou"], "method5_miou": None if ref is None else ref["miou"],
                          "gap": gap, "ok": gap is not None and abs(gap) <= REPRO_TOLERANCE + 1e-12,
                          "tolerance": REPRO_TOLERANCE}
    return out


def verdict_of(configs: Sequence[Dict[str, Any]], bars: Dict[float, Dict[str, Any]], complete: bool, clips: int,
               check: Dict[str, Dict[str, Any]], repro: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The spec's pass rule for one model: at some scale, the best local configuration is (a) at
    least +1.0 point above the full frame and (b) at least +0.5 point above Method 5's best
    separable now configuration at that scale."""
    per_scale: Dict[str, Any] = {}
    local = [c for c in configs if c["config"].get("form") == "2d" and c["config"].get("alpha", 0.0) > 0
             and c.get("delta_vs_fullframe") is not None and not math.isnan(c["delta_vs_fullframe"])]
    for s in sorted({float(c["config"]["scale"]) for c in local}):
        best = max((c for c in local if float(c["config"]["scale"]) == s), key=lambda c: c["delta_vs_fullframe"])
        bar = bars.get(s)
        over = None if bar is None else best["delta_vs_fullframe"] - bar["delta_vs_fullframe"]
        a = best["delta_vs_fullframe"] >= PASS_MARGIN
        b = over is not None and over >= SEP_MARGIN
        per_scale[f"{s:g}"] = {"best": best["name"], "miou": best["miou"],
                               "delta_vs_fullframe": best["delta_vs_fullframe"],
                               "delta_vs_uniform": best.get("delta_vs_uniform"),
                               "separable_bar": bar, "delta_vs_separable": over, "a": a, "b": b, "both": a and b,
                               "budget_ms": best.get("budget_ms"), "distortion": best.get("distortion")}
    failed = sorted((k for k, v in check.items() if not v["ok"]), key=float)
    bad_repro = sorted(k for k, v in repro.items() if not v["ok"])
    if not complete:
        passes, why = None, f"not read (partial: {clips} clips)"
    elif failed or not check:
        passes, why = None, ("not read (pipeline check failed at scales " + ", ".join(failed) + ")"
                             if failed else "not read (no uniform row: no pipeline check)")
    elif bad_repro or not repro:
        passes, why = None, ("not read (reproduction check failed: " + ", ".join(bad_repro) + ")"
                             if bad_repro else "not read (no separable row: no reproduction check)")
    elif any(v["separable_bar"] is None for v in per_scale.values()):
        passes, why = None, "not read (no Method 5 separable configuration at some scale)"
    else:
        passes = any(v["both"] for v in per_scale.values())
        why = "PASSES" if passes else "FAILS"
    return {"rule": f"at some scale, the best local configuration is (a) >= +{PASS_MARGIN * 100:g} mIoU points above "
                    f"the full frame at that scale and (b) >= +{SEP_MARGIN * 100:g} points above Method 5's best "
                    "separable now configuration at that scale, on every clip of train_all (spec 2026-11-17)",
            "margins": {"a": PASS_MARGIN, "b": SEP_MARGIN}, "clips": clips, "complete": complete,
            "pipeline_check_ok": not failed and bool(check), "reproduction_ok": not bad_repro and bool(repro),
            "per_scale": per_scale, "passes": passes, "state": why}


def cmd_score(args: argparse.Namespace) -> int:
    import numpy as np
    from eval.tradeoff import miou_of

    tables = latency_tables(args.latency, args.fixture)
    out_dir = _out_dir(args)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split) if not args.no_expect else None
    names = list(dataset.class_names) if dataset is not None else [str(c) for c in range(19)]
    expected = {dataset.clip_id(i) for i in range(len(dataset))} if dataset is not None else None
    verdicts: Dict[str, Any] = {}
    for model in _strs(args.models):
        path = Path(args.bound) if args.bound else out_dir / f"bound_{args.split}_{model}.jsonl.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run `python -m scripts.method6 merge --split {args.split}`")
        totals: Dict[str, Any] = {}
        dists: Dict[str, Dict[str, float]] = {}
        clips_of: Dict[str, set] = {}
        configs_of: Dict[str, Dict[str, Any]] = {}
        frames: Dict[str, int] = {}
        for r in read_lines(path):
            name = r["name"]
            conf = np.asarray(r["conf"], dtype=np.int64)
            totals[name] = conf if name not in totals else totals[name] + conf
            clips_of.setdefault(name, set()).add(r["clip_id"])
            configs_of.setdefault(name, {**r["config"], **({"ref": r["ref"]} if "ref" in r else {})})
            frames[r["clip_id"]] = r.get("frames", 0)
            if "dist" in r:
                acc = dists.setdefault(name, {})
                for k, v in r["dist"].items():
                    acc[k] = acc.get(k, 0.0) + v
        clip_ids = set().union(*clips_of.values())
        short = [n for n, c in clips_of.items() if c != clip_ids]
        if short:
            raise SystemExit(f"{path}: configs {short[:3]} lack clips: merge again from complete chunks")
        ref = fullframe_ref(Path(args.fullframe), model, args.split, tables, sorted(clip_ids))
        e2e = e2e_curves(args.e2e, model, ref)
        same = same_parts_curve(ref, tables, model)
        n_frames = sum(frames.values())
        resize = {float(configs_of[n]["scale"]): miou_of(t) for n, t in totals.items() if "ref" in configs_of[n]}
        configs = [summarise(n, configs_of[n], t, len(clip_ids), n_frames, tables, model, names, ref, e2e, same)
                   for n, t in totals.items() if "ref" not in configs_of[n]]
        uniform = {float(c["config"]["scale"]): c["miou"] for c in configs if c["config"].get("alpha", 0.0) == 0.0}
        baseline = dict(zip(ref["scales"], ref["miou"]))
        m5 = method5_rows(args.method5_scores, args.split, model)
        bars = separable_bars(m5)
        for c in configs:
            s = float(c["config"]["scale"])
            u, b, bar = uniform.get(s), baseline.get(s), bars.get(s)
            c["delta_vs_uniform"] = None if u is None else c["miou"] - u
            c["delta_vs_fullframe"] = None if b is None else c["miou"] - b
            c["delta_vs_separable"] = None if bar is None else c["miou"] - bar["miou"]
            if c["name"] in dists:
                c["distortion"] = {k: v / max(n_frames, 1) for k, v in dists[c["name"]].items()}
        check = pipeline_check(configs, resize, ref)
        repro = reproduction_check(configs, m5)
        complete = expected is not None and clip_ids == expected
        verdict = verdicts[model] = verdict_of(configs, bars, complete, len(clip_ids), check, repro)
        payload = {"split": args.split, "model": model, "bound": str(path), "fullframe": _public(ref),
                   "fullframe_e2e": None if e2e is None else {k: v for k, v in e2e.items() if k != "curves"},
                   "fullframe_same_parts": {k: v for k, v in same.items() if k != "curves"},
                   "separable_bars": {f"{k:g}": v for k, v in sorted(bars.items())},
                   "pipeline_check": check, "reproduction_check": repro, "verdict": verdict, "configs": configs}
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"score_{args.split}_{model}.json"
        out.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
        gaps = ", ".join(f"c{k} {'ok' if v['ok'] else 'NOT ok'}" for k, v in check.items())
        reps = ", ".join(f"{k} {'ok' if v['ok'] else 'NOT ok'}" for k, v in repro.items())
        shown = "; ".join(f"c{k}: {v['best']} {v['delta_vs_fullframe'] * 100:+.2f} vs FF, "
                          + ("no separable bar" if v["delta_vs_separable"] is None
                             else f"{v['delta_vs_separable'] * 100:+.2f} vs separable")
                          for k, v in verdict["per_scale"].items())
        print(f"[method6 score] {model}: {len(configs)} configs over {len(clip_ids)} clips; pipeline check: {gaps}; "
              f"reproduction: {reps or 'none'}; {shown}; rule {verdict['state']} -> {out}")
    if verdicts and all(v["complete"] for v in verdicts.values()):
        states = [v["passes"] for v in verdicts.values()]
        passes = None if any(p is None for p in states) else any(states)
        (out_dir / f"verdict_{args.split}.json").write_text(
            json.dumps({"split": args.split, "passes": passes, "models": verdicts}, indent=1, allow_nan=True) + "\n")
        state = {True: "PASSES", False: "FAILS", None: "is NOT READ (a check failed)"}[passes]
        print(f"[method6 score] feasibility check {state} (models: {', '.join(verdicts)})")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="train_all")
        sp.add_argument("--models", default="pidnet_s,bisenetv2")
        sp.add_argument("--clip-ids", default=None, help="Comma-separated clip ids (default: all, or --chunk).")
        sp.add_argument("--chunk", default=None, help="i/N: the i-th contiguous block of the sorted clip ids.")
        sp.add_argument("--tag", default=None, help="Output file tag (default: from --chunk / --clip-ids).")
        sp.add_argument("--out-dir", default=None, help="Folder of the jsonl files (default experiments/method6/<split>).")
        sp.add_argument("--fixture", action="store_true", help="Synthetic fixture + dummy backbone (smoke test).")
        sp.add_argument("--device", default="auto")

    b = sub.add_parser("bound", help="Every configuration of the grid on the labelled frames, per clip (resumable).")
    common(b)
    b.add_argument("--scales", default="0.25,0.375,0.5", help="Input scales c (the warped image is c x the frame).")
    b.add_argument("--saliencies", default="E,B,S", help="E: error, B: boundaries, S: small classes.")
    b.add_argument("--alphas", default="0.1,0.25,0.5", help="Warp strengths (the uniform row, 0, is always run).")
    b.add_argument("--sigmas", default="1,4", help="Saliency smoothing, in cells.")
    b.add_argument("--kernels", default="4,12", help="Sampling kernel sigma_k, in cells (locality of the zoom).")
    b.add_argument("--no-sep", action="store_true", help="Leave out the separable reproduction rows.")
    b.add_argument("--cell", type=int, default=16, help="Saliency cell, px (16: a 64 x 128 grid).")
    b.add_argument("--batch", type=int, default=4, help="Warped images of one scale per forward pass.")
    b.set_defaults(fn=cmd_bound)

    m = sub.add_parser("merge", help="Chunk files -> one bound file per model.")
    common(m)
    m.add_argument("--no-expect", action="store_true", help="Do not compare against the dataset's clips.")
    m.set_defaults(fn=cmd_merge)

    s = sub.add_parser("score", help="Per config: mIoU, latency, judges, distortion; both checks; the verdict.")
    common(s)
    s.add_argument("--bound", default=None, help="Score this bound file (default: the merged one; one model only).")
    s.add_argument("--fullframe", default=str(REPO_ROOT / "experiments" / "method2" / "fullframe"))
    s.add_argument("--latency", default=DEFAULT_LATENCY, help="Latency-floor JSONs (glob, comma-separated).")
    s.add_argument("--e2e", default=DEFAULT_E2E, help="Latency-floor `final` sessions: the measured full-frame curve.")
    s.add_argument("--method5-scores", default=M5_SCORES,
                   help="Folders of Method 5's score_<split>_<model>.json[.gz] (comma-separated): the separable bar "
                        "and the reproduction check.")
    s.add_argument("--no-expect", action="store_true",
                   help="Do not load the dataset (the rule is then not read: completeness unknown).")
    s.set_defaults(fn=cmd_score)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
