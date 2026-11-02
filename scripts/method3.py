#!/usr/bin/env python3
"""Method 3 (motion-gated previous-frame reuse): stats, caches, bounds, sweep, merge, score, finalists, val report.

    python -m scripts.method3 stats --split train_all --chunk 0/4          # motion statistics; fixes theta
    python -m scripts.method3 cache --split train_all --models pidnet_s     # per-frame clip caches (GPU)
    python -m scripts.method3 bounds --split train_all --models pidnet_s    # reuse bound, zone oracle, raw ablation
    python -m scripts.method3 sweep --split train_all --models pidnet_s     # every config of the grid, per clip
    python -m scripts.method3 merge --split train_all                       # chunk files -> one per model
    python -m scripts.method3 score --split train_all                       # mIoU, latency, both judges
    python -m scripts.method3 finalists --split train_all                   # Pareto fronts + equal-latency picks
    python -m scripts.method3 report --split val                            # the finalists, once, on val

Accuracy comes from the per-frame caches (``reuse.replay``: an exact replay of
:class:`reuse.method.MotionReuseMethod`), latency from the T4 latency-floor tables
(``foveation.latency_model.predict_reuse_frame``, per frame), and the reference is the
full-frame curve of the same model (``scripts.method2_fullframe``). Two judges: the
mean per-frame latency against the full frame's median, and the p95 per-frame latency
against the full frame's p95. Configs and finalists are chosen on train; val is
evaluated by ``report`` only. ``--fixture`` runs on the synthetic fixture with the
dummy backbone (a smoke test, not results).

Outputs are gzip-compressed JSON lines (``.jsonl.gz``), one line per (config, clip).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method3.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _floats, _strs, latency_tables, load_dataset, model_config  # noqa: E402

ensure_src_on_path()

DEFAULT_LATENCY = "results/latency_floor/latency_floor_t4-s*.json,results/latency_floor/latency_floor_m3-s*.json"
PERCENTILES = (95, 98, 99)  # theta candidates: these percentiles of the aligned |diff| on train
SHARE_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
EXP = REPO_ROOT / "experiments" / "method3"
RES = REPO_ROOT / "results" / "method3"


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _ff_split(split: str) -> str:
    """The full-frame reference of ``train_all`` is train's (same clips, same labels)."""
    return "train" if split == "train_all" else split


def select_clips(clip_ids: Sequence[str], wanted: Optional[str], chunk: Optional[str]) -> List[int]:
    """Dataset indices: ``wanted`` (comma-separated ids), chunk ``i/N`` (the i-th contiguous block
    of ``ceil(len/N)`` sorted ids, as ``make_train_img_all.zip_chunks``), or all clips."""
    by_id = {c: i for i, c in enumerate(clip_ids)}
    if wanted:
        ids = _strs(wanted)
        unknown = [c for c in ids if c not in by_id]
        if unknown:
            raise SystemExit(f"unknown clip ids {unknown}")
        return sorted(by_id[c] for c in ids)
    ids = sorted(clip_ids)
    if chunk:
        i, n = (int(v) for v in chunk.split("/"))
        size = math.ceil(len(ids) / n)
        ids = ids[i * size:(i + 1) * size]
    return [by_id[c] for c in ids]


def _tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    if args.clip_ids:
        return "ids"
    return "c" + args.chunk.replace("/", "of") if args.chunk else "all"


def _dataset_and_clips(args: argparse.Namespace):
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
    ids = [dataset.clip_id(i) for i in range(len(dataset))]
    return dataset, select_clips(ids, args.clip_ids, args.chunk)


def _cache_dir(args: argparse.Namespace, model: str) -> Path:
    return Path(args.root) / args.split / model


def _out_dir(args: argparse.Namespace) -> Path:
    return Path(args.out_dir) if args.out_dir else EXP / args.split


def _write_lines(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """gzip JSON lines, written to ``.tmp`` and renamed (a cut-off run leaves no half file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, allow_nan=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_lines(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _thetas(args: argparse.Namespace) -> List[int]:
    if args.thetas:
        return [int(v) for v in _strs(args.thetas)]
    path = Path(args.stats) if args.stats else RES / "motion_stats_train_all.json"
    if not path.is_file():
        raise SystemExit(f"no theta: pass --thetas or run `python -m scripts.method3 stats` ({path} is missing)")
    return [int(v) for v in json.loads(path.read_text())["theta_candidates"]]


def config_groups(args: argparse.Namespace) -> List[Tuple[Any, List[float]]]:
    """The grid as (config at tau = 1, every tau): configs that differ only in tau share a group."""
    from reuse.method import MotionReuseConfig

    variants: List[Optional[float]] = [None, *_floats(args.ctx_scales)]
    groups = [(MotionReuseConfig(full_scale=f, glimpse_size=g, theta=t, rho=r, tau=1.0, ctx_scale=c,
                                 margin=args.margin, homography=args.homography), _floats(args.taus))
              for c, f, g, t, r in itertools.product(variants, _floats(args.full_scales),
                                                     [int(v) for v in _strs(args.glimpse_sizes)], _thetas(args),
                                                     _floats(args.rhos))]
    i, n = (int(v) for v in args.part.split("/"))
    return groups[i::n]


def _moving_counts(cache, grid, theta: int, rho: float, device, key: str = "motion") -> List[int]:
    """Moving cells per frame (frame 0: 0). They depend on the frames only, never on the memory."""
    import torch

    from reuse.motion import moving_cells

    return [0] + [int(moving_cells(torch.from_numpy(cache[key][t]).to(device), grid, theta, rho).sum())
                  for t in range(1, cache.num_frames)]


def _fallbacks(counts: Sequence[int], num_cells: int, tau: float) -> Tuple[bool, ...]:
    return tuple(k / num_cells > tau for k in counts[1:])


def _caches(args: argparse.Namespace, model: str, clips: Sequence[int], dataset):
    from reuse.replay import ClipCache, cache_file

    for i in clips:
        path = cache_file(_cache_dir(args, model), dataset.clip_id(i))
        if not path.is_file():
            raise SystemExit(f"missing cache {path}: run `python -m scripts.method3 cache --split {args.split}`")
        yield dataset.clip_id(i), ClipCache.load(path)


def _inliers(cache) -> Dict[str, Any]:
    v = cache["inliers"][1:]
    return {"min": int(v.min()), "mean": float(v.mean()), "identity_frames": int((v == 0).sum())}


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def cmd_stats(args: argparse.Namespace) -> int:
    """Aligned and raw |diff| histograms, theta candidates and per-cell share quantiles, from the frames."""
    import numpy as np
    import torch

    from eval.harness import frame_to_tensor
    from foveation.ops import GlimpseGrid
    from reuse.motion import IDENTITY, MOTION_FACTOR, cell_share, estimate_homography, motion_map, to_gray_small

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    hist = {"aligned": np.zeros(256, np.int64), "raw": np.zeros(256, np.int64)}
    maps: List[np.ndarray] = []
    identity = 0
    frame_size = None
    t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(clips):
            prev = None
            for t in range(dataset.num_frames(i)):
                frame = frame_to_tensor(dataset.load_frame(i, t), device)
                frame_size = tuple(frame.shape[-2:])
                gray = to_gray_small(frame)
                if prev is not None:
                    H, support = estimate_homography(prev, gray, method=args.homography)
                    identity += int(support == 0)
                    m = motion_map(prev, gray, H).cpu().numpy()
                    hist["aligned"] += np.bincount(m.ravel(), minlength=256)
                    hist["raw"] += np.bincount(motion_map(prev, gray, IDENTITY).cpu().numpy().ravel(), minlength=256)
                    maps.append(m)
                prev = gray
            print(f"[method3 stats] clip {n + 1}/{len(clips)} ({time.time() - t0:.0f} s)", flush=True)
    cdf = np.cumsum(hist["aligned"]) / max(int(hist["aligned"].sum()), 1)
    pct = {str(q): int(np.searchsorted(cdf, q / 100 - 1e-12)) for q in PERCENTILES}
    thetas = sorted(set(pct.values()))
    cells: Dict[str, Any] = {}
    for g in (int(v) for v in _strs(args.glimpse_sizes)):
        grid = GlimpseGrid(frame_size, g, args.margin)
        cells[str(g)] = {}
        for theta in thetas:
            share = np.stack([cell_share(torch.from_numpy(m), grid, theta).numpy() for m in maps])  # (frames, cells)
            moving = {f"{r:g}": float((share > r).mean()) for r in _floats(args.rhos)}
            cells[str(g)][str(theta)] = {
                "share_quantiles": {f"{q:g}": float(np.quantile(share, q)) for q in SHARE_QUANTILES},
                "moving_cell_share": moving,
                "frames_moving_share_above_tau": {
                    f"{r:g}": {f"{u:g}": float(((share > r).mean(1) > u).mean()) for u in _floats(args.taus)}
                    for r in _floats(args.rhos)},
            }
    out = Path(args.out) if args.out else RES / f"motion_stats_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "clip_ids": [dataset.clip_id(i) for i in clips], "frames": len(maps),
               "homography": args.homography, "identity_fallbacks": identity, "motion_factor": MOTION_FACTOR,
               "hist_aligned": hist["aligned"].tolist(), "hist_raw": hist["raw"].tolist(), "percentiles": pct,
               "theta_candidates": thetas, "cells": cells}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[method3 stats] {len(maps)} frames, {identity} identity homographies, theta {thetas} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cmd_cache(args: argparse.Namespace) -> int:
    from eval.results import git_info
    from models.registry import build_backbone
    from reuse.replay import build_clip_cache, cache_file

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    built = skipped = 0
    t0 = time.time()
    for model in _strs(args.models):
        out = _cache_dir(args, model)
        todo = []
        for i in clips:
            path = cache_file(out, dataset.clip_id(i))
            tmp = path.with_name(path.name[:-len(".npz")] + ".tmp.npz")
            if tmp.exists():
                tmp.unlink()  # a cut-off write: never trusted
            if path.exists():
                skipped += 1
            else:
                todo.append(i)
        if not todo:
            continue
        cfg = model_config(model, args.fixture)
        backbone = build_backbone(cfg["backbone"], context={"palette": dataset.palette}).to(device).eval()
        meta = {"split": args.split, "model": model, "git": git_info(REPO_ROOT),
                "dataset_fingerprint": dataset.fingerprint()}
        for n, i in enumerate(todo):
            build_clip_cache(dataset, i, backbone, out, _floats(args.full_scales),
                             [int(v) for v in _strs(args.glimpse_sizes)], _floats(args.ctx_scales), device,
                             precision=args.precision, homography=args.homography, margin=args.margin,
                             allow_any_frame_size=args.fixture, extra_meta=meta)
            built += 1
            print(f"[method3 cache] {model} {n + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"[method3 cache] built {built}, skipped {skipped} existing, in {Path(args.root) / args.split}")
    return 0


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #
def cmd_bounds(args: argparse.Namespace) -> int:
    """Per clip: the reuse bound by lag per full scale; per motion rule (3a, tau = 1): the rule,
    the zone oracle at the rule's k, and the rule on the raw difference."""
    from reuse.method import MotionReuseConfig
    from reuse.replay import replay_clip, reuse_bound

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    rules = [MotionReuseConfig(full_scale=f, glimpse_size=g, theta=t, rho=r, tau=1.0, margin=args.margin,
                               homography=args.homography)
             for f, g, t, r in itertools.product(_floats(args.full_scales), [int(v) for v in _strs(args.glimpse_sizes)],
                                                 _thetas(args), _floats(args.rhos))]
    for model in _strs(args.models):
        def rows():
            for clip_id, cache in _caches(args, model, clips, dataset):
                out = {"split": args.split, "model": model, "clip_id": clip_id, "inliers": _inliers(cache),
                       "reuse_bound": {f"{s:g}": reuse_bound(cache, s, device).tolist()
                                       for s in _floats(args.full_scales)}, "rules": []}
                for cfg in rules:
                    motion = replay_clip(cache, cfg, device)
                    oracle = replay_clip(cache, cfg, device, selector="oracle", oracle_k=motion.moved_cells)
                    raw = replay_clip(cache, cfg, device, motion_key="motion_raw")
                    out["rules"].append({"name": cfg.name(), "config": asdict(cfg), "num_cells": motion.num_cells,
                                         **{k: {"conf": r.conf.tolist(), "moved_cells": r.moved_cells.tolist()}
                                            for k, r in (("motion", motion), ("oracle", oracle), ("raw", raw))}})
                print(f"[method3 bounds] {model} {clip_id}", flush=True)
                yield out

        path = _out_dir(args) / f"bounds_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        print(f"[method3 bounds] {_write_lines(path, rows())} clips -> {path}")
    return 0


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def cmd_sweep(args: argparse.Namespace) -> int:
    """Every config of the grid (or only the finalists) on every clip; configs whose fallback frames
    coincide (they differ only in tau) give the same replay, which runs once. On val only the
    finalists run: nothing is tuned there."""
    if args.split == "val" and not args.finalists and not args.fixture:
        raise SystemExit("val is the reporting split: sweep it with --finalists only")
    from reuse.method import MotionReuseConfig
    from reuse.replay import replay_clip

    device = resolve_device(args.device)
    dataset, clips = _dataset_and_clips(args)
    finalists = json.loads(Path(args.finalists).read_text())["finalists"] if args.finalists else None
    part = "" if args.part == "0/1" else "_p" + args.part.replace("/", "of")
    for model in _strs(args.models):
        if finalists is None:
            groups = config_groups(args)
        else:
            groups = [(MotionReuseConfig(**f["config"]), [f["config"]["tau"]]) for f in finalists if f["model"] == model]

        def rows():
            t0 = time.time()
            for n, (clip_id, cache) in enumerate(_caches(args, model, clips, dataset)):
                counts: Dict[Tuple[int, int, float], List[int]] = {}
                replays = 0
                for base, taus in groups:
                    grid = cache.grid(base.glimpse_size)
                    key = (base.glimpse_size, base.theta, base.rho)
                    if key not in counts:
                        counts[key] = _moving_counts(cache, grid, base.theta, base.rho, device)
                    done: Dict[Tuple[bool, ...], Any] = {}
                    for tau in taus:
                        cfg = replace(base, tau=tau)
                        fb = _fallbacks(counts[key], grid.num_cells, tau)
                        if fb not in done:
                            done[fb] = replay_clip(cache, cfg, device)
                            replays += 1
                        res = done[fb]
                        yield {"split": args.split, "model": model, "clip_id": clip_id, "name": cfg.name(),
                               "config": asdict(cfg), "num_cells": res.num_cells, "conf": res.conf.tolist(),
                               "moved_cells": res.moved_cells.tolist(), "fallback": res.fallback.tolist(),
                               "inliers": _inliers(cache)}
                print(f"[method3 sweep] {model} clip {n + 1}/{len(clips)}: {replays} replays "
                      f"({time.time() - t0:.0f} s)", flush=True)

        path = _out_dir(args) / f"sweep_{args.split}_{model}_{_tag(args)}{part}.jsonl.gz"
        print(f"[method3 sweep] {_write_lines(path, rows())} lines -> {path}")
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args: argparse.Namespace) -> int:
    """Chunk files -> ``sweep_<split>_<model>.jsonl.gz`` and ``bounds_<split>_<model>.jsonl.gz``.
    Refuses a (config, clip) seen twice; lists the clips missing against the dataset's."""
    expected = None
    if not args.no_expect:
        dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
        expected = {dataset.clip_id(i) for i in range(len(dataset))}
    out_dir = _out_dir(args)
    for model in _strs(args.models):
        for kind in ("sweep", "bounds"):
            files = sorted(p for p in out_dir.glob(f"{kind}_{args.split}_{model}_*.jsonl.gz"))
            if not files:
                continue
            seen: Dict[Tuple[str, str], str] = {}
            clips_of: Dict[str, set] = {}
            rows = []
            for f in files:
                for r in read_lines(f):
                    key = (r.get("name", ""), r["clip_id"])
                    if key in seen:
                        raise SystemExit(f"duplicate {kind} line {key} in {f.name} and {seen[key]}: "
                                         "a chunk merged twice?")
                    seen[key] = f.name
                    clips_of.setdefault(key[0], set()).add(r["clip_id"])
                    rows.append(r)
            path = out_dir / f"{kind}_{args.split}_{model}.jsonl.gz"
            n = _write_lines(path, rows)
            present = set.union(*clips_of.values())
            missing = set()
            for clips in clips_of.values():
                missing |= (expected if expected is not None else present) - clips
            print(f"[method3 merge] {kind} {model}: {len(files)} files, {n} lines, {len(present)} clips -> {path}")
            if missing:
                print(f"[method3 merge] {kind} {model}: missing clips {sorted(missing)}")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def fullframe_ref(ff_dir: Path, model: str, split: str, tables, clip_ids: Sequence[str]) -> Dict[str, Any]:
    """The model's full-frame curve on ``clip_ids``: mIoU per scale and the T4 median / p95."""
    import numpy as np

    from eval.tradeoff import FullFrameCurve, miou_of

    path = Path(ff_dir) / f"fullframe_{model}_{_ff_split(split)}.npz"
    if not path.is_file():
        raise SystemExit(f"missing {path}: run scripts.method2_fullframe for {model} on {_ff_split(split)}")
    d = np.load(path)
    col = {str(c): j for j, c in enumerate(d["clip_ids"])}
    missing = [c for c in clip_ids if c not in col]
    if missing:
        raise SystemExit(f"{path} has no clips {missing[:5]}")
    clips = d["confusion"][:, [col[c] for c in clip_ids]]
    scales = [float(s) for s in d["scales"]]
    lat = [tables.fullframe(model, s) for s in scales]
    miou = [miou_of(c.sum(0)) for c in clips]
    return {"scales": scales, "miou": miou, "median_ms": [x["median_ms"] for x in lat],
            "p95_ms": [x["p95_ms"] for x in lat], "gflops": [x["gflops"] for x in lat], "clips": clips,
            "curves": {"mean": FullFrameCurve([x["median_ms"] for x in lat], miou),
                       "p95": FullFrameCurve([x["p95_ms"] for x in lat], miou)}}


def _public(ref: Dict[str, Any]) -> Dict[str, Any]:
    return {k: ref[k] for k in ("scales", "miou", "median_ms", "p95_ms", "gflops")}


class FrameCost:
    """Per-frame T4 predictions of a config (memoised on moved cells / fallback / first)."""

    def __init__(self, tables, model: str):
        self.tables, self.model, self._memo = tables, model, {}

    def frames(self, cfg, moved: Sequence[int], fallback: Sequence[bool]) -> List[Dict[str, float]]:
        out = []
        for t, (k, fb) in enumerate(zip(moved, fallback)):
            key = (cfg, 0 if fb or t == 0 else int(k), bool(fb), t == 0)
            if key not in self._memo:
                self._memo[key] = self.tables.predict_reuse_frame(self.model, cfg, key[1], key[2], key[3])
            out.append(self._memo[key])
        return out


def judge_readings(miou: float, mean_ms: float, p95_ms: float, ref: Dict[str, Any]) -> Dict[str, Any]:
    """Both trade-off readings against both judges."""
    from eval.tradeoff import delta_miou_at_equal_latency, latency_saved_at_equal_miou

    out = {}
    for judge, ms in (("mean", mean_ms), ("p95", p95_ms)):
        curve = ref["curves"][judge]
        out[judge] = {"latency_ms": ms, "delta_miou_at_equal_latency": delta_miou_at_equal_latency(miou, ms, curve),
                      "latency_saved_at_equal_miou": latency_saved_at_equal_miou(miou, ms, curve)}
    return out


class _Acc:
    """Running totals of one config over clips."""

    def __init__(self):
        self.conf = None
        self.mean_ms: List[float] = []
        self.p95_ms: List[float] = []
        self.gflops: List[float] = []
        self.recomputed: List[float] = []
        self.fallback: List[bool] = []
        self.clips = 0

    def add(self, conf, frames, moved, fallback, num_cells) -> None:
        import numpy as np

        conf = np.asarray(conf, dtype=np.int64)
        self.conf = conf if self.conf is None else self.conf + conf
        self.mean_ms += [f["median_ms"] for f in frames]
        self.p95_ms += [f["p95_ms"] for f in frames]
        self.gflops += [f["gflops"] for f in frames]
        self.recomputed += [1.0 if t == 0 or fb else k / num_cells for t, (k, fb) in enumerate(zip(moved, fallback))]
        self.fallback += list(fallback[1:])
        self.clips += 1

    def summary(self, class_names, ref) -> Dict[str, Any]:
        import numpy as np

        from eval.tradeoff import miou_of
        from foveation.latency_model import frame_latency_summary

        total = self.conf.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.diag(total) / (total.sum(0) + total.sum(1) - np.diag(total))
        miou = miou_of(total)
        lat = {"mean_ms": frame_latency_summary(self.mean_ms)["mean_ms"],
               "p95_ms": frame_latency_summary(self.p95_ms)["p95_ms"], "frames": len(self.mean_ms)}
        return {"miou": miou, "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(class_names, iou)},
                "miou_by_position": [miou_of(c) for c in self.conf], "latency": lat,
                "gflops_mean": float(np.mean(self.gflops)),
                "fallback_share": float(np.mean(self.fallback)) if self.fallback else 0.0,
                "recomputed_share": float(np.mean(self.recomputed)), "clips": self.clips,
                "judges": judge_readings(miou, lat["mean_ms"], lat["p95_ms"], ref)}


def cmd_score(args: argparse.Namespace) -> int:
    from reuse.method import MotionReuseConfig

    tables = latency_tables(args.latency, args.fixture)
    out_dir = _out_dir(args)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split) if not args.no_expect else None
    names = list(dataset.class_names) if dataset is not None else [str(c) for c in range(19)]
    for model in _strs(args.models):
        path = out_dir / f"sweep_{args.split}_{model}.jsonl.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run `python -m scripts.method3 merge --split {args.split}`")
        cost = FrameCost(tables, model)
        accs: Dict[str, _Acc] = {}
        cfgs: Dict[str, Any] = {}
        clip_ids: List[str] = []
        for r in read_lines(path):
            cfg = cfgs.setdefault(r["name"], MotionReuseConfig(**r["config"]))
            if r["clip_id"] not in clip_ids:
                clip_ids.append(r["clip_id"])
            accs.setdefault(r["name"], _Acc()).add(r["conf"], cost.frames(cfg, r["moved_cells"], r["fallback"]),
                                                   r["moved_cells"], r["fallback"], r["num_cells"])
        ref = fullframe_ref(Path(args.fullframe), model, args.split, tables, sorted(clip_ids))
        configs = [{"name": n, "model": model, "config": asdict(cfgs[n]), **a.summary(names, ref)}
                   for n, a in accs.items()]
        payload = {"split": args.split, "model": model, "sweep": str(path), "fullframe": _public(ref),
                   "configs": configs}
        bpath = out_dir / f"bounds_{args.split}_{model}.jsonl.gz"
        if bpath.is_file():
            payload["bounds"] = _score_bounds(bpath, cost, names, ref)
        out = out_dir / f"score_{args.split}_{model}.json"
        out.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
        print(f"[method3 score] {model}: {len(configs)} configs over {len(clip_ids)} clips -> {out}")
    return 0


def _score_bounds(path: Path, cost: FrameCost, names, ref) -> Dict[str, Any]:
    import numpy as np

    from eval.tradeoff import miou_of
    from reuse.method import MotionReuseConfig

    lags: Dict[str, Any] = {}
    rules: Dict[str, Dict[str, _Acc]] = {}
    cfgs: Dict[str, Any] = {}
    for r in read_lines(path):
        for s, conf in r["reuse_bound"].items():
            conf = np.asarray(conf, dtype=np.int64)
            lags[s] = conf if s not in lags else lags[s] + conf
        for rule in r["rules"]:
            cfg = cfgs.setdefault(rule["name"], MotionReuseConfig(**rule["config"]))
            for kind in ("motion", "oracle", "raw"):
                moved = rule[kind]["moved_cells"]
                fallback = [False] * len(moved)
                rules.setdefault(rule["name"], {}).setdefault(kind, _Acc()).add(
                    rule[kind]["conf"], cost.frames(cfg, moved, fallback), moved, fallback, rule["num_cells"])
    return {"reuse_bound": {s: [miou_of(c) for c in conf] for s, conf in lags.items()},
            "rules": [{"name": n, "config": asdict(cfgs[n]), **{k: a.summary(names, ref) for k, a in kinds.items()}}
                      for n, kinds in rules.items()]}


# --------------------------------------------------------------------------- #
# finalists
# --------------------------------------------------------------------------- #
def cmd_finalists(args: argparse.Namespace) -> int:
    """Per model and per judge: the Pareto front over (judge latency, mIoU) and the best mIoU
    within each full-frame scale's latency under that judge; the union over judges."""
    out_dir = _out_dir(args)
    finalists: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for model in _strs(args.models):
        score = json.loads((out_dir / f"score_{args.split}_{model}.json").read_text())
        cands = score["configs"]
        ff = score["fullframe"]
        for judge, ff_ms in (("mean", ff["median_ms"]), ("p95", ff["p95_ms"])):
            def add(c: Dict[str, Any], reason: str) -> None:
                entry = finalists.setdefault((model, c["name"]), {
                    "model": model, "name": c["name"], "config": c["config"], "split": args.split,
                    "train": {"miou": c["miou"], "mean_ms": c["latency"]["mean_ms"], "p95_ms": c["latency"]["p95_ms"]},
                    "reasons": []})
                entry["reasons"].append(reason)

            ordered = sorted(cands, key=lambda c: (c["judges"][judge]["latency_ms"], -c["miou"]))
            best = float("-inf")
            for c in ordered:
                if c["miou"] > best:
                    best = c["miou"]
                    add(c, f"pareto_{judge}")
            for scale, lat in zip(ff["scales"], ff_ms):
                within = [c for c in cands if c["judges"][judge]["latency_ms"] <= lat]
                if within:
                    add(max(within, key=lambda c: c["miou"]), f"best_within_fullframe_s{scale:g}_{judge}")
    out = Path(args.out) if args.out else RES / "finalists.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "rule": "per model and judge (mean vs the full frame's median, "
                               "p95 vs its p95): Pareto front over (latency, mIoU) and the best mIoU within each "
                               "full-frame scale's latency; the union", "finalists": list(finalists.values())},
                              indent=1) + "\n")
    print(f"[method3 finalists] {len(finalists)} configurations -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace) -> int:
    """The finalists on val from the merged val sweep (``sweep --split val --finalists``), with
    clip-bootstrap 95% intervals for both judges (latencies fixed, as in Method 2)."""
    if args.split != "val":
        raise SystemExit("report evaluates the finalists on val, the reporting split; tune on train_all with sweep")
    import numpy as np

    from eval.tradeoff import clip_bootstrap
    from reuse.method import MotionReuseConfig

    tables = latency_tables(args.latency, args.fixture)
    finalists = json.loads(Path(args.finalists).read_text())["finalists"]
    names = list(load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split).class_names)
    results = []
    refs: Dict[str, Any] = {}
    clips_seen: List[str] = []
    for model in sorted({f["model"] for f in finalists}):
        path = _out_dir(args) / f"sweep_{args.split}_{model}.jsonl.gz"
        if not path.is_file():
            raise SystemExit(f"missing {path}: run `sweep --split val --finalists` on Colab, then `merge --split val`")
        lines: Dict[str, Dict[str, Any]] = {}
        for r in read_lines(path):
            lines.setdefault(r["name"], {})[r["clip_id"]] = r
        clip_ids = sorted(set().union(*lines.values())) if lines else []
        clips_seen = clip_ids
        ref = refs[model] = fullframe_ref(Path(args.fullframe), model, args.split, tables, clip_ids)
        cost = FrameCost(tables, model)
        for f in (f for f in finalists if f["model"] == model):
            have = lines.get(f["name"], {})
            if set(have) != set(clip_ids):
                raise SystemExit(f"{path}: finalist {f['name']} lacks clips {sorted(set(clip_ids) - set(have))}")
            cfg = MotionReuseConfig(**f["config"])
            acc = _Acc()
            for c in clip_ids:
                r = have[c]
                acc.add(r["conf"], cost.frames(cfg, r["moved_cells"], r["fallback"]), r["moved_cells"], r["fallback"],
                        r["num_cells"])
            s = acc.summary(names, ref)
            per_clip = np.stack([np.asarray(have[c]["conf"], dtype=np.int64).sum(0) for c in clip_ids])
            boot = {"mean": clip_bootstrap(per_clip, ref["clips"], ref["median_ms"], s["latency"]["mean_ms"],
                                           n=args.bootstrap, seed=0),
                    "p95": clip_bootstrap(per_clip, ref["clips"], ref["p95_ms"], s["latency"]["p95_ms"],
                                          n=args.bootstrap, seed=0)}
            results.append({"model": model, "name": f["name"], "config": f["config"], "reasons": f.get("reasons"),
                            "train": f.get("train"), **s, "bootstrap": boot})
            print(f"[method3 report] {model} {f['name']}: mIoU {s['miou']:.4f}", flush=True)
    out = Path(args.out) if args.out else RES / "val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "finalists_file": str(args.finalists), "clips": len(clips_seen),
                               "bootstrap_resamples": args.bootstrap,
                               "fullframe": {m: _public(r) for m, r in refs.items()}, "finalists": results},
                              indent=1, allow_nan=True) + "\n")
    print(f"[method3 report] {len(results)} finalists on {args.split} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, split=None):
        sp.add_argument("--split", default=split, required=split is None)
        sp.add_argument("--models", default="pidnet_s,bisenetv2")
        sp.add_argument("--clip-ids", default=None, help="Comma-separated clip ids (default: all, or --chunk).")
        sp.add_argument("--chunk", default=None, help="i/N: the i-th contiguous block of the sorted clip ids.")
        sp.add_argument("--tag", default=None, help="Output file tag (default: from --chunk / --clip-ids).")
        sp.add_argument("--root", default=str(EXP / "cache"), help="Cache root: <root>/<split>/<model>/<clip>.npz")
        sp.add_argument("--out-dir", default=None, help="Folder of the jsonl files (default experiments/method3/<split>).")
        sp.add_argument("--fixture", action="store_true", help="Synthetic fixture + dummy backbone (smoke test).")
        sp.add_argument("--device", default="auto")

    def grid(sp):
        sp.add_argument("--full-scales", default="1,0.75,0.5")
        sp.add_argument("--glimpse-sizes", default="256,384,512")
        sp.add_argument("--ctx-scales", default="0.25,0.375,0.5", help="3b context scales (3a is always included).")
        sp.add_argument("--margin", type=int, default=32)
        sp.add_argument("--homography", default="iclk", choices=["orb", "iclk"])
        sp.add_argument("--thetas", default=None, help="Motion thresholds (default: the stats file's candidates).")
        sp.add_argument("--stats", default=None, help="motion_stats JSON (default results/method3/motion_stats_train_all.json).")
        sp.add_argument("--rhos", default="0.02,0.05,0.1")
        sp.add_argument("--taus", default="0.2,0.4,0.6,1")

    def evaluation(sp):
        sp.add_argument("--fullframe", default=str(REPO_ROOT / "experiments" / "method2" / "fullframe"))
        sp.add_argument("--latency", default=DEFAULT_LATENCY, help="Latency-floor JSONs (glob, comma-separated).")
        sp.add_argument("--no-expect", action="store_true", help="Do not load the dataset (clip list, class names).")

    s = sub.add_parser("stats", help="Motion statistics from the frames (fixes theta).")
    common(s)
    grid(s)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_stats)

    c = sub.add_parser("cache", help="Per-frame clip caches (GPU).")
    common(c)
    grid(c)
    c.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    c.set_defaults(fn=cmd_cache)

    b = sub.add_parser("bounds", help="Reuse bound, zone oracle and raw-difference ablation per clip.")
    common(b)
    grid(b)
    b.set_defaults(fn=cmd_bounds)

    w = sub.add_parser("sweep", help="Every config of the grid on every clip.")
    common(w)
    grid(w)
    w.add_argument("--part", default="0/1", help="i/M: every M-th tau group of the grid, from the i-th.")
    w.add_argument("--finalists", default=None, help="Run only these configs (finalists.json); required on val.")
    w.set_defaults(fn=cmd_sweep)

    m = sub.add_parser("merge", help="Chunk files -> one sweep and one bounds file per model.")
    common(m)
    m.add_argument("--no-expect", action="store_true", help="Do not compare against the dataset's clips.")
    m.set_defaults(fn=cmd_merge)

    sc = sub.add_parser("score", help="Per config: mIoU, drift, latency, both judges.")
    common(sc)
    evaluation(sc)
    sc.set_defaults(fn=cmd_score)

    f = sub.add_parser("finalists", help="Pareto fronts + equal-latency picks, per judge, from the scores.")
    common(f, split="train_all")
    f.add_argument("--out", default=None)
    f.set_defaults(fn=cmd_finalists)

    r = sub.add_parser("report", help="The finalists, once, on val, with clip-bootstrap intervals.")
    common(r, split="val")
    evaluation(r)
    r.add_argument("--finalists", default=str(RES / "finalists.json"))
    r.add_argument("--out", default=None)
    r.add_argument("--bootstrap", type=int, default=1000)
    r.set_defaults(fn=cmd_report)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
