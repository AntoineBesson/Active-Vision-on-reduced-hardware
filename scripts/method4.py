#!/usr/bin/env python3
"""Method 4, phase 1: the feasibility bound of a persistent canvas with a glimpse policy.

    python -m scripts.method4 bound --split train_all --models pidnet_s --chunk 0/4   # oracle replay, per clip
    python -m scripts.method4 merge --split train_all                                # chunk files -> one per model
    python -m scripts.method4 score --split train_all                                # mIoU, latency, judges, verdict

Spec (grid, latency parts and pass rule, fixed before any run):
docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md

``bound`` replays every configuration of the grid over the Method 3 per-clip caches
(``python -m scripts.method3 cache``; no network pass) with a per-frame greedy oracle
selector (:mod:`canvas.replay`). It appends each finished clip to a ``.jsonl.gz`` file
and, started again, skips the clips already there. ``score`` predicts each frame's T4
latency from the latency-floor tables with the policy at 0
(``LatencyTables.predict_canvas_frame``), reads every configuration against the
full-frame curve of the same model under both judges (mean and p95 per frame), against
the full frame's measured end-to-end latency too, and applies the spec's pass rule.

Train only: the bound is a go/no-go check, val is never read here. ``--fixture`` runs on
the synthetic fixture with the dummy backbone (a smoke test, not results).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import statistics
import sys
import time
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method4.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _floats, _ints, _strs, latency_tables, load_dataset, model_config  # noqa: E402
from scripts.method3 import (DEFAULT_LATENCY, _caches, _dataset_and_clips, _public, _tag, _write_lines,  # noqa: E402
                             fullframe_ref, judge_readings, read_lines)

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method4"
M3_CACHE = REPO_ROOT / "experiments" / "method3" / "cache"
DEFAULT_E2E = "results/latency_floor/latency_floor_final-s*.json"
PASS_MARGIN = 0.010  # the spec's rule: +1.0 mIoU point above the curve at equal mean latency
CANVAS_VARIANTS = ("iclk", "none")  # A and B: the variants the pass rule is read on
LETTER = {"off": "N", "iclk": "A", "none": "B"}


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _out_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else EXP / args.split


def _ages(text: str) -> List[Optional[int]]:
    return [None if v == "inf" else int(v) for v in _strs(text)]


def grid_configs(args: argparse.Namespace) -> List[Any]:
    """The spec's grid: context scale x glimpse size x (N, A x a_max, B x a_max) x selector."""
    from canvas.replay import CanvasConfig

    variants: List[Tuple[str, Optional[int]]] = []
    for v in _strs(args.variants):
        variants += [(v, None)] if v == "off" else [(v, a) for a in _ages(args.a_max)]
    selectors = [{"k": k} for k in _ints(args.ks)] + [{"gamma": g} for g in _floats(args.gammas)]
    return [CanvasConfig(ctx_scale=c, glimpse_size=g, variant=v, a_max=a, margin=args.margin, **sel)
            for c, g, (v, a), sel in itertools.product(_floats(args.ctx_scales), _ints(args.glimpse_sizes),
                                                       variants, selectors)]


def _scan(path: Path) -> Tuple[List[str], Dict[str, set], bool]:
    """A bound file's clips in order, the config names of each, and whether it ends cleanly
    (a run cut off mid-write leaves a truncated gzip member)."""
    order: List[str] = []
    names: Dict[str, set] = {}
    clean = True
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["clip_id"] not in names:
                        order.append(r["clip_id"])
                    names.setdefault(r["clip_id"], set()).add(r["name"])
    except (EOFError, OSError, zlib.error, json.JSONDecodeError):
        clean = False
    return order, names, clean


def resume(path: Path, expected: set) -> set:
    """The clips ``path`` already holds in full. A cut-off last clip is dropped (the file is
    rewritten without it); anything else that does not match the grid is refused."""
    if not path.exists():
        return set()
    order, names, clean = _scan(path)
    done = {c for c in order if names[c] == expected}
    bad = [c for c in order if c not in done]
    if bad and bad != order[-1:]:
        raise SystemExit(f"{path} holds clips {bad[:5]} written with another grid: use another --tag or delete it")
    if bad or not clean:
        def rows():
            try:
                for r in read_lines(path):
                    if r["clip_id"] in done:
                        yield r
            except (EOFError, OSError, zlib.error, json.JSONDecodeError):
                return

        kept = path.with_name(path.name + ".keep")
        _write_lines(kept, rows())
        kept.replace(path)
        print(f"[method4 bound] {path.name}: dropped a cut-off tail, kept {len(done)} clips", flush=True)
    return done


def _append(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """One clip's lines as one gzip member at the end of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, allow_nan=False) + "\n")


# --------------------------------------------------------------------------- #
# bound
# --------------------------------------------------------------------------- #
def cmd_bound(args: argparse.Namespace) -> int:
    if args.split == "val" and not args.fixture:
        raise SystemExit("the bound is a train-only check (spec): val is not read in phase 1")
    from canvas.replay import replay_canvas_many

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    cfgs = grid_configs(args)
    expected = {c.name() for c in cfgs}
    groups: Dict[Tuple[int, str], List[Any]] = {}
    for c in cfgs:
        groups.setdefault((c.glimpse_size, c.variant), []).append(c)
    for model in _strs(args.models):
        path = _out_dir(args) / f"bound_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        done = resume(path, expected)
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method4 bound] {model}: {len(cfgs)} configs, {len(todo)} clips to run, {len(clips) - len(todo)} "
              f"already in {path.name}", flush=True)
        t0 = time.time()
        for n, (clip_id, cache) in enumerate(_caches(args, model, todo, dataset)):
            rows = []
            for group in groups.values():
                for start in range(0, len(group), args.batch):
                    part = group[start:start + args.batch]
                    for cfg, res in zip(part, replay_canvas_many(cache, part, device, homography=args.homography)):
                        rows.append({"split": args.split, "model": model, "clip_id": clip_id, "name": cfg.name(),
                                     "config": asdict(cfg), "num_cells": res.num_cells, "conf": res.conf.tolist(),
                                     "k": res.k.tolist(),
                                     "canvas_share": round(float(res.canvas_share[1:].mean()), 6)})
            _append(path, rows)
            dt = time.time() - t0
            print(f"[method4 bound] {model} clip {n + 1}/{len(todo)} {clip_id} ({dt:.0f} s, "
                  f"{dt / (n + 1):.1f} s/clip)", flush=True)
        print(f"[method4 bound] {model}: {len(clips)} clips in {path}")
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args: argparse.Namespace) -> int:
    """Chunk files -> ``bound_<split>_<model>.jsonl.gz``. Refuses a (config, clip) seen twice and a
    clip without every config of the others; lists the clips missing against the dataset's."""
    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    out_dir = _out_dir(args)
    for model in _strs(args.models):
        files = sorted(out_dir.glob(f"bound_{args.split}_{model}_*.jsonl.gz"))
        if not files:
            print(f"[method4 merge] {model}: no chunk file in {out_dir}")
            continue
        seen: Dict[Tuple[str, str], str] = {}
        per_clip: Counter = Counter()

        def rows():  # one line in memory at a time
            for f in files:
                for r in read_lines(f):
                    key = (r["name"], r["clip_id"])
                    if key in seen:
                        raise SystemExit(f"duplicate line {key} in {f.name} and {seen[key]}: a chunk merged twice?")
                    seen[key] = f.name
                    per_clip[r["clip_id"]] += 1
                    yield r

        path = out_dir / f"bound_{args.split}_{model}.jsonl.gz"
        n = _write_lines(path, rows())
        if len(set(per_clip.values())) > 1:
            path.unlink()
            raise SystemExit(f"[method4 merge] {model}: clips with different numbers of configs "
                             f"({sorted(set(per_clip.values()))}): a cut-off chunk? Run `bound` again on it")
        print(f"[method4 merge] {model}: {len(files)} files, {n} lines, {len(per_clip)} clips -> {path}")
        missing = sorted((expected or set()) - set(per_clip))
        if missing:
            print(f"[method4 merge] {model}: missing clips {missing}")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def e2e_curves(spec: str, model: str, ref: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The full frame's measured end-to-end latency (latency-floor ``final`` sessions, graph level,
    median over sessions) at ``ref``'s scales, with ``ref``'s mIoU. None when no session matches."""
    from eval.tradeoff import FullFrameCurve

    paths = sorted(p for part in spec.split(",")
                   for p in glob.glob(part if Path(part).is_absolute() else str(REPO_ROOT / part)))
    if not paths:
        return None
    by_scale: Dict[float, List[Dict[str, float]]] = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["records"].get("final", []):
            if r.get("kind") == "fullframe" and r.get("model") == model and r.get("level") == "graph":
                by_scale.setdefault(float(r["scale"]), []).append(r["cuda_event"])
    if any(s not in by_scale for s in ref["scales"]):
        return None
    med = [statistics.median(e["median_ms"] for e in by_scale[s]) for s in ref["scales"]]
    p95 = [statistics.median(e["p95_ms"] for e in by_scale[s]) for s in ref["scales"]]
    return {"scales": ref["scales"], "miou": ref["miou"], "median_ms": med, "p95_ms": p95, "sessions": len(paths),
            "curves": {"mean": FullFrameCurve(med, ref["miou"]), "p95": FullFrameCurve(p95, ref["miou"])}}


class _Acc:
    """Running totals of one config over clips. A frame's latency depends on (k, first frame)
    only, so the frames are kept as a histogram."""

    def __init__(self):
        self.conf = None
        self.frames: Counter = Counter()
        self.share: List[float] = []
        self.num_cells = 0

    def add(self, row: Dict[str, Any]) -> None:
        import numpy as np

        conf = np.asarray(row["conf"], dtype=np.int64)
        self.conf = conf if self.conf is None else self.conf + conf
        self.frames.update((int(k), t == 0) for t, k in enumerate(row["k"]))
        self.share.append(row["canvas_share"])
        self.num_cells = row["num_cells"]

    def latency(self, tables, model: str, cfg, with_motion: bool = False) -> Dict[str, float]:
        import numpy as np

        keys = list(self.frames)
        count = np.array([self.frames[key] for key in keys])
        pred = [tables.predict_canvas_frame(model, cfg, k, first, with_motion=with_motion) for k, first in keys]
        median = np.repeat([p["median_ms"] for p in pred], count)
        p95 = np.repeat([p["p95_ms"] for p in pred], count)
        gflops = np.repeat([p["gflops"] for p in pred], count)
        return {"mean_ms": float(median.mean()), "p95_ms": float(np.percentile(p95, 95)),
                "gflops_mean": float(gflops.mean()), "frames": int(count.sum())}

    def summary(self, tables, model: str, cfg, class_names, ref, e2e) -> Dict[str, Any]:
        import numpy as np

        from eval.tradeoff import miou_of

        total = self.conf.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.diag(total) / (total.sum(0) + total.sum(1) - np.diag(total))
        miou = miou_of(total)
        lat = self.latency(tables, model, cfg)
        gflops = lat.pop("gflops_mean")
        ks = np.repeat([k for k, _ in self.frames], [self.frames[key] for key in self.frames])
        out = {"miou": miou, "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(class_names, iou)},
               "miou_by_position": [miou_of(c) for c in self.conf], "latency": lat, "gflops_mean": gflops,
               "k_mean": float(ks.mean()), "k_max": int(ks.max()), "glimpsed_share": float(ks.mean() / self.num_cells),
               "canvas_share": float(np.mean(self.share)), "clips": len(self.share),
               "judges": judge_readings(miou, lat["mean_ms"], lat["p95_ms"], ref)}
        # what a real policy may cost per frame before the config is back on the curve (mean judge)
        out["policy_budget_ms"] = out["judges"]["mean"]["latency_saved_at_equal_miou"]
        if e2e is not None:
            out["judges_e2e"] = judge_readings(miou, lat["mean_ms"], lat["p95_ms"], e2e)
        if cfg.variant == "iclk":
            m = self.latency(tables, model, cfg, with_motion=True)
            out["with_motion"] = {"latency": {k: m[k] for k in ("mean_ms", "p95_ms")},
                                  "judges": judge_readings(miou, m["mean_ms"], m["p95_ms"], ref)}
        return out


def _best(configs: Sequence[Dict[str, Any]], variants: Sequence[str], judges: str, judge: str) -> Optional[Dict[str, Any]]:
    """The config of ``variants`` with the highest ΔmIoU at equal latency that has a reading."""
    import math

    cands = [c for c in configs if c["config"]["variant"] in variants and judges in c
             and not math.isnan(c[judges][judge]["delta_miou_at_equal_latency"])]
    if not cands:
        return None
    c = max(cands, key=lambda c: c[judges][judge]["delta_miou_at_equal_latency"])
    return {"name": c["name"], "miou": c["miou"], "latency_ms": c[judges][judge]["latency_ms"],
            "delta_miou_at_equal_latency": c[judges][judge]["delta_miou_at_equal_latency"],
            "latency_saved_at_equal_miou": c[judges][judge]["latency_saved_at_equal_miou"], "readable": len(cands)}


def verdict_of(configs: Sequence[Dict[str, Any]], complete: bool, clips: int) -> Dict[str, Any]:
    """The spec's pass rule for one model: the best variant-A-or-B config at equal mean latency
    against the primary curve, read only on the whole clip set."""
    best = _best(configs, CANVAS_VARIANTS, "judges", "mean")
    delta = best["delta_miou_at_equal_latency"] if best else None
    return {"rule": f"best config of variant A or B >= +{PASS_MARGIN * 100:g} mIoU points above the full-frame "
                    "curve at equal mean latency, on every clip of train_all",
            "margin": PASS_MARGIN, "clips": clips, "complete": complete, "best": best,
            "passes": (delta is not None and delta >= PASS_MARGIN) if complete else None}


def cmd_score(args: argparse.Namespace) -> int:
    from canvas.replay import CanvasConfig

    tables = latency_tables(args.latency, args.fixture)
    out_dir = _out_dir(args)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split) if not args.no_expect else None
    names = list(dataset.class_names) if dataset is not None else [str(c) for c in range(19)]
    expected = {dataset.clip_id(i) for i in range(len(dataset))} if dataset is not None else None
    verdicts: Dict[str, Any] = {}
    for model in _strs(args.models):
        path = Path(args.bound) if args.bound else out_dir / f"bound_{args.split}_{model}.jsonl.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run `python -m scripts.method4 merge --split {args.split}`")
        accs: Dict[str, _Acc] = {}
        cfgs: Dict[str, Any] = {}
        clip_ids = set()
        for r in read_lines(path):
            cfgs.setdefault(r["name"], CanvasConfig(**r["config"]))
            clip_ids.add(r["clip_id"])
            accs.setdefault(r["name"], _Acc()).add(r)
        short = [n for n, a in accs.items() if len(a.share) != len(clip_ids)]
        if short:
            raise SystemExit(f"{path}: configs {short[:3]} lack clips: merge again from complete chunks")
        ref = fullframe_ref(Path(args.fullframe), model, args.split, tables, sorted(clip_ids))
        e2e = e2e_curves(args.e2e, model, ref)
        configs = [{"name": n, "model": model, "config": asdict(cfgs[n]),
                    **a.summary(tables, model, cfgs[n], names, ref, e2e)} for n, a in accs.items()]
        complete = expected is not None and clip_ids == expected
        verdict = verdicts[model] = verdict_of(configs, complete, len(clip_ids))
        best = {f"{LETTER[v]}/{judges}/{judge}": _best(configs, [v], judges, judge)
                for v in LETTER for judges in ("judges", "judges_e2e") for judge in ("mean", "p95")}
        payload = {"split": args.split, "model": model, "bound": str(path), "fullframe": _public(ref),
                   "fullframe_e2e": None if e2e is None else {k: v for k, v in e2e.items() if k != "curves"},
                   "verdict": verdict, "best": best, "configs": configs}
        out = out_dir / f"score_{args.split}_{model}.json"
        out.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
        b = verdict["best"]
        shown = "no readable A or B config" if b is None else (
            f"best A/B {b['name']}: {b['delta_miou_at_equal_latency'] * 100:+.2f} pts at {b['latency_ms']:.2f} ms")
        state = {True: "PASSES", False: "FAILS", None: f"not read (partial: {len(clip_ids)} clips)"}[verdict["passes"]]
        print(f"[method4 score] {model}: {len(configs)} configs over {len(clip_ids)} clips; {shown}; rule {state} -> {out}")
    if all(v["complete"] for v in verdicts.values()):
        passes = any(v["passes"] for v in verdicts.values())
        summary = {"split": args.split, "passes": passes, "models": verdicts}
        (out_dir / f"verdict_{args.split}.json").write_text(json.dumps(summary, indent=1, allow_nan=True) + "\n")
        print(f"[method4 score] feasibility check {'PASSES' if passes else 'FAILS'} "
              f"(models: {', '.join(verdicts)})")
    return 0


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
        sp.add_argument("--root", default=str(M3_CACHE), help="Method 3 cache root: <root>/<split>/<model>/<clip>.npz")
        sp.add_argument("--out-dir", default=None, help="Folder of the jsonl files (default experiments/method4/<split>).")
        sp.add_argument("--fixture", action="store_true", help="Synthetic fixture + dummy backbone (smoke test).")
        sp.add_argument("--device", default="auto")

    b = sub.add_parser("bound", help="Oracle replay of the grid over the Method 3 caches, per clip (resumable).")
    common(b)
    b.add_argument("--ctx-scales", default="0.25,0.375,0.5")
    b.add_argument("--glimpse-sizes", default="256,384,512")
    b.add_argument("--variants", default="off,iclk,none", help="off: no canvas (N); iclk: warped (A); none: not warped (B).")
    b.add_argument("--a-max", default="1,2,4,8,inf", help="Canvas age limits in frames (inf: none).")
    b.add_argument("--ks", default="1,2,4,8", help="Fixed numbers of glimpses per frame.")
    b.add_argument("--gammas", default="0.005,0.02,0.05", help="Adaptive selector: gain thresholds (share of a cell).")
    b.add_argument("--margin", type=int, default=32)
    b.add_argument("--homography", default="iclk", choices=["orb", "iclk"],
                   help="The estimator variant A's caches must have been built with (the T4 cost counted is IC-LK's).")
    b.add_argument("--batch", type=int, default=32, help="Configs replayed together (GPU memory: ~60 MB each).")
    b.set_defaults(fn=cmd_bound)

    m = sub.add_parser("merge", help="Chunk files -> one bound file per model.")
    common(m)
    m.add_argument("--no-expect", action="store_true", help="Do not compare against the dataset's clips.")
    m.set_defaults(fn=cmd_merge)

    s = sub.add_parser("score", help="Per config: mIoU, latency, both judges, both references; the verdict.")
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
