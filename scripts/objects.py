"""Per-object recall by size and camera-corrected motion, re-scored from the Method 3 caches.

Design: the "Moving-object metric: design" doc (2026-11-12) and ``src/eval/objects.py``.
No backbone pass: predictions come from the Method 3 per-clip caches, replayed with
:func:`reuse.replay.replay_clip` (Method 3) and :func:`canvas.replay.replay_canvas`
(Method 4), or read directly (full frame).

    python -m scripts.objects table  --split train_all                       # object table (CPU)
    python -m scripts.objects score  --split train_all --models pidnet_s \\
        --score results/method3/score_train_all_pidnet_s.json.gz \\
        --score experiments/method4/train_all/score_train_all_pidnet_s.json --edge 12   # IoU per object
    python -m scripts.objects report --split train_all --models pidnet_s --score ... # recall tables

``table`` writes ``results/objects/objects_<split>.json.gz`` and prints the motion and size
quantiles: the bin edges are frozen in ``results/objects/bins.json`` from them, before
``score``. ``score`` appends one line per (config, clip) to
``experiments/objects/<split>/iou_<model>_<tag>.jsonl.gz`` (resumable). ``report`` reads
them all, and writes ``results/objects/report_<split>_<model>.json``.

Method 2 cannot be re-scored per object (its caches hold per-cell confusions only). Its
stand-in here, Method 4's variant N (``4o-N/...``: context + glimpses, no canvas, oracle
selector), is a *screen*: ``report`` keeps it in a separate ``screen`` block that never
enters the main table. If it shows Method 2 matters for small moving objects, its finalists
are rerun exactly instead.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/objects.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _strs, load_dataset, model_config  # noqa: E402
from scripts.method3 import EXP as EXP3, select_clips  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "objects"
RES = REPO_ROOT / "results" / "objects"
SCREEN_PREFIX = "4o-N/"  # Method 4 variant N: the Method 2 screen, never reported as Method 2
TABLE_COLUMNS = ("clip", "t", "id", "cls", "area", "cx", "cy", "motion", "status", "matched", "raw_shift")


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _json_load(path: Path) -> Any:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _json_dump(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh, allow_nan=True)


def read_lines(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _append(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, allow_nan=True) + "\n")


def table_path(split: str) -> Path:
    return RES / f"objects_{split}.json.gz"


def load_table(path: Path):
    """The object table as numpy columns (``clip`` as str)."""
    import numpy as np

    d = _json_load(path)
    cols = {k: np.asarray(d["columns"][k]) for k in TABLE_COLUMNS}
    cols["clip"] = cols["clip"].astype(str)
    cols["motion"] = cols["motion"].astype(np.float64)
    cols["raw_shift"] = cols["raw_shift"].astype(np.float64)
    return cols, d.get("meta", {})


def load_bins(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path:
        return _json_load(Path(path))
    default = RES / "bins.json"
    return _json_load(default) if default.is_file() else None


# --------------------------------------------------------------------------- #
# Ground truth of a clip
# --------------------------------------------------------------------------- #
def clip_ground_truth(dataset, index: int, labeled_t: Sequence[int]):
    """(instance maps, FrameObjects) of the clip's labelled frames."""
    from eval.objects import frame_objects

    maps, objs = [], []
    for t in labeled_t:
        inst = dataset.annotation_ids(index, int(t))
        if inst is None:
            raise SystemExit(f"clip {dataset.clip_id(index)} frame {t}: no instance map")
        maps.append(inst)
        objs.append(frame_objects(inst))
    return maps, objs


def _caches(root: Path, split: str, model: str, dataset, clips: Sequence[int]):
    from reuse.replay import ClipCache, cache_file

    for i in clips:
        path = cache_file(Path(root) / split / model, dataset.clip_id(i))
        if not path.is_file():
            raise SystemExit(f"missing cache {path}: run `python -m scripts.method3 cache` first")
        yield i, ClipCache(path)


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def build_table(dataset, clips: Sequence[int], caches: Iterable[Tuple[int, Any]]) -> Dict[str, Any]:
    import numpy as np

    from eval.objects import clip_objects

    parts = []
    for i, cache in caches:
        _, objs = clip_ground_truth(dataset, i, cache.labeled_t)
        parts.append(clip_objects(dataset.clip_id(i), cache.labeled_t, objs, cache["H"], cache["inliers"]))
    cols = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in TABLE_COLUMNS}
    return cols


def table_summary(cols: Dict[str, Any], bins: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the bin edges are chosen from: quantiles, counts per cell, and the id-persistence check."""
    import numpy as np

    from eval.objects import MOTION_STATUSES, motion_bins, size_bins

    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    measured = cols["status"] == 0
    sb, mb = size_bins(cols["area"], bins), motion_bins(cols["motion"], cols["status"], bins)
    cars = measured & (cols["cls"] == 13)

    def quant(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[~np.isnan(x)]
        return {str(p): float(np.quantile(x, p)) for p in q} if len(x) else {}

    grid: Dict[str, Dict[str, int]] = {}
    for s, m in zip(sb, mb):
        grid.setdefault(str(s), {}).setdefault(str(m), 0)
        grid[str(s)][str(m)] += 1
    return {"objects": int(len(cols["t"])), "clips": int(len(set(cols["clip"].tolist()))),
            "status": {name: int((cols["status"] == k).sum()) for k, name in enumerate(MOTION_STATUSES)},
            "area_quantiles": quant(cols["area"]), "motion_quantiles": quant(cols["motion"][measured]),
            "car_motion_quantiles": quant(cols["motion"][cars]),
            "raw_shift_quantiles": quant(cols["raw_shift"]),
            "grid": grid}


def cmd_table(args: argparse.Namespace) -> int:
    dataset = load_dataset(model_config(args.models.split(",")[0], args.fixture), args.split)
    clips = select_clips([dataset.clip_id(i) for i in range(len(dataset))], args.clip_ids, args.chunk)
    model = _strs(args.models)[0]
    cols = build_table(dataset, clips, _caches(Path(args.root), args.split, model, dataset, clips))
    bins = load_bins(args.bins)
    summary = table_summary(cols, bins)
    out = Path(args.out) if args.out else table_path(args.split)
    _json_dump({"meta": {"split": args.split, "cache_model": model, "dataset_fingerprint": dataset.fingerprint(),
                         "summary": summary},
                "columns": {k: cols[k].tolist() for k in TABLE_COLUMNS}}, out)
    print(json.dumps(summary, indent=1))
    print(f"[objects table] {summary['objects']} objects in {summary['clips']} clips -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def score_paths(values: Optional[Sequence[str]]) -> List[str]:
    """``--score`` values (repeatable, each possibly comma-separated)."""
    return [p for v in (values or []) for p in _strs(v)]


def load_scores(paths: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Configs by name from Method 3 / Method 4 score files, and the first file's full-frame block."""
    configs: Dict[str, Dict[str, Any]] = {}
    fullframe: Dict[str, Any] = {}
    for p in paths:
        d = _json_load(Path(p))
        fullframe = fullframe or d.get("fullframe", {})
        for c in d["configs"]:
            configs[c["name"]] = c
    return configs, fullframe


def upper_edge(configs: Sequence[Dict[str, Any]], n: int) -> List[str]:
    """Up to ``n`` configs of the upper edge of the mIoU / mean-latency cloud, spread over latency,
    plus the config of highest ΔmIoU at equal latency."""
    import numpy as np

    pts = sorted(((c["judges"]["mean"]["latency_ms"], c["miou"], c["name"]) for c in configs), key=lambda p: (p[0], -p[1]))
    front, best = [], -math.inf
    for lat, miou, name in pts:
        if miou > best:
            front.append(name)
            best = miou
    picks = [front[int(j)] for j in np.unique(np.linspace(0, len(front) - 1, min(n, len(front))).round())] if front else []
    readable = [c for c in configs if not math.isnan(c["judges"]["mean"]["delta_miou_at_equal_latency"])]
    if readable:
        top = max(readable, key=lambda c: c["judges"]["mean"]["delta_miou_at_equal_latency"])["name"]
        if top not in picks:
            picks.append(top)
    return picks


def _family(name: str) -> str:
    if name.startswith("ff/"):
        return "fullframe"
    if name.startswith("3"):
        return "method3"
    if name.startswith(SCREEN_PREFIX):
        return "method2_screen"
    if name.startswith("4o-"):
        return "method4"
    raise SystemExit(f"unknown config family of {name!r}")


def cached_scales(cache) -> set:
    """Full-pass scales a cache can serve: ``full_{s}``, or ``ctx_{s}`` (the same full pass, see
    :func:`reuse.replay.build_clip_cache`)."""
    return {float(s) for s in cache.meta.get("full_scales", []) + cache.meta.get("ctx_scales", [])}


def replayable(cfg: Dict[str, Any], scales: Optional[set]) -> bool:
    """Whether every full pass the config needs is in the caches (``None``: assume so)."""
    if scales is None:
        return True
    need = [cfg.get("full_scale"), cfg.get("ctx_scale")]
    return all(float(s) in scales for s in need if s is not None)


def choose_configs(args: argparse.Namespace, scores: Dict[str, Dict[str, Any]], fullframe: Dict[str, Any],
                   scales_cached: Optional[set] = None) -> List[str]:
    """Full-frame scales, explicit configs, and the upper edge of each family. With
    ``scales_cached``, the edge is taken among the configs the caches can replay only."""
    scales = [float(v) for v in _strs(args.ff_scales)] if args.ff_scales else fullframe.get("scales", [])
    names = [f"ff/{s:g}" for s in scales]
    names += _strs(args.configs) if args.configs else []
    if args.edge:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for c in scores.values():
            if replayable(c["config"], scales_cached):
                groups.setdefault(_family(c["name"]), []).append(c)
        for fam in ("method3", "method4", "method2_screen"):
            names += upper_edge(groups.get(fam, []), args.edge)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def predictions(cache, name: str, scores: Dict[str, Dict[str, Any]], device) -> List[Any]:
    """The (H, W) predictions of ``name`` at the clip's labelled frames."""
    import torch

    labeled = [int(t) for t in cache.labeled_t]
    fam = _family(name)
    if fam == "fullframe":
        s = float(name[3:])
        keys = [k for k in (f"full_{s:g}", f"ctx_{s:g}") if k in cache._npz.files]
        if not keys:
            raise SystemExit(f"{cache.path}: no full pass at scale {s:g}; pass --ff-scales with the cached ones")
        arr = cache[keys[0]]
        return [torch.from_numpy(arr[t]) for t in labeled]
    if name not in scores:
        raise SystemExit(f"config {name!r} is in none of the --score files")
    cfg = scores[name]["config"]
    if fam == "method3":
        from reuse.method import MotionReuseConfig
        from reuse.replay import _key, replay_clip

        for prefix in ("full", "full_conf"):  # a full pass at s < 1 may be cached as the context pass
            key, alias = _key(prefix, cfg["full_scale"]), _key(prefix.replace("full", "ctx"), cfg["full_scale"])
            if key not in cache._npz.files and alias in cache._npz.files:
                cache._arrays[key] = cache[alias]
        _, preds = replay_clip(cache, MotionReuseConfig(**cfg), device, return_predictions=True)
    else:
        from canvas.replay import CanvasConfig, replay_canvas

        _, preds = replay_canvas(cache, CanvasConfig(**cfg), device, return_predictions=True)
    return [preds[t] for t in labeled]


def clip_ious(cache, maps, objs, name: str, scores, device) -> List[float]:
    """Local IoU of every object of the clip, in table order ((t, id))."""
    from eval.objects import local_ious

    out: List[float] = []
    for inst, fo, pred in zip(maps, objs, predictions(cache, name, scores, device)):
        out += local_ious(inst, pred.to("cpu").numpy(), fo).tolist()
    return out


def _tag(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    if args.chunk:
        return "chunk" + args.chunk.replace("/", "of")
    return "all"


def wait_for_gpu(max_temp: Optional[int], pause_s: float = 20.0) -> None:
    """Block while the GPU is hotter than ``max_temp`` C (laptop guard; no-op without nvidia-smi)."""
    import subprocess
    import time

    if not max_temp:
        return
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            temp = max(int(v) for v in out.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        if temp <= max_temp:
            return
        print(f"[objects score] GPU {temp} C > {max_temp} C, pausing {pause_s:g} s", flush=True)
        time.sleep(pause_s)


def cmd_score(args: argparse.Namespace) -> int:
    scores, fullframe = load_scores(score_paths(args.score))
    device = resolve_device(args.device)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
    clips = select_clips([dataset.clip_id(i) for i in range(len(dataset))], args.clip_ids, args.chunk)
    _, first = next(_caches(Path(args.root), args.split, _strs(args.models)[0], dataset, clips[:1]))
    names = choose_configs(args, scores, fullframe, cached_scales(first))
    for model in _strs(args.models):
        path = EXP / args.split / f"iou_{model}_{_tag(args)}.jsonl.gz"
        done = {(r["name"], r["clip_id"]) for r in read_lines(path)} if path.is_file() else set()
        print(f"[objects score] {model}: {len(names)} configs x {len(clips)} clips; {len(done)} lines already in {path}")
        for i, cache in _caches(Path(args.root), args.split, model, dataset, clips):
            clip_id = dataset.clip_id(i)
            todo = [n for n in names if (n, clip_id) not in done]
            if not todo:
                continue
            maps, objs = clip_ground_truth(dataset, i, cache.labeled_t)
            _append(path, ({"name": n, "clip_id": clip_id, "model": model,
                            "iou": (wait_for_gpu(args.max_temp), clip_ious(cache, maps, objs, n, scores, device))[1]}
                           for n in todo))
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def collect_ious(files: Sequence[Path], cols) -> Dict[str, Any]:
    """IoU arrays aligned with the table, per config; refuses duplicates and partial configs."""
    import numpy as np

    starts: Dict[str, Tuple[int, int]] = {}
    clip = cols["clip"]
    for c in dict.fromkeys(clip.tolist()):
        idx = np.flatnonzero(clip == c)
        starts[c] = (int(idx[0]), int(idx[-1]) + 1)
    out: Dict[str, Any] = {}
    seen = set()
    for f in files:
        for r in read_lines(f):
            key = (r["name"], r["clip_id"])
            if key in seen:
                raise SystemExit(f"{f}: ({key[0]}, {key[1]}) seen twice")
            seen.add(key)
            if r["clip_id"] not in starts:
                if not r["iou"]:  # a clip without any thing object (6 of the 400 on train_all)
                    continue
                raise SystemExit(f"{f}: clip {r['clip_id']} is not in the object table")
            a, b = starts[r["clip_id"]]
            if len(r["iou"]) != b - a:
                raise SystemExit(f"{f}: {key}: {len(r['iou'])} IoUs for {b - a} objects (table out of date?)")
            arr = out.setdefault(r["name"], np.full(len(clip), np.nan))
            arr[a:b] = r["iou"]
    partial = [n for n, a in out.items() if np.isnan(a).any()]
    if partial:
        raise SystemExit(f"configs {partial[:3]} lack clips: score the missing chunks first")
    return out


def groups_of(cols, bins: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    from eval.objects import _bins, motion_bins, size_bins

    b = _bins(bins)
    sb, mb = size_bins(cols["area"], bins), motion_bins(cols["motion"], cols["status"], bins)
    groups: Dict[str, Any] = {"all": np.ones(len(sb), bool)}
    small = np.isin(sb, b["size_names"][:2])
    moving = np.isin(mb, b["motion_names"][1:] + ["new"])
    groups["headline: " + "+".join(b["size_names"][:2]) + " x " + "+".join(b["motion_names"][1:] + ["new"])] = small & moving
    for s in b["size_names"]:
        groups[f"size={s}"] = sb == s
    for m in b["motion_names"] + ["new", "unknown"]:
        groups[f"motion={m}"] = mb == m
    for s in b["size_names"]:
        for m in b["motion_names"] + ["new"]:
            groups[f"{s} x {m}"] = (sb == s) & (mb == m)
    return groups


def _latency(name: str, scores, fullframe) -> float:
    if name.startswith("ff/"):
        s = float(name[3:])
        return float(fullframe["median_ms"][[float(x) for x in fullframe["scales"]].index(s)])
    return float(scores[name]["judges"]["mean"]["latency_ms"])


def build_report(cols, ious: Dict[str, Any], scores, fullframe, bins, n_boot: int = 1000,
                 found_iou: float = 0.5, tiny_iou: float = 0.25) -> Dict[str, Any]:
    import numpy as np

    from eval.objects import _bins, recall_at_equal_latency, recall_table, size_bins

    groups = groups_of(cols, bins)
    head = next(k for k in groups if k.startswith("headline"))
    ff = sorted((n for n in ious if n.startswith("ff/")), key=lambda n: _latency(n, scores, fullframe))
    if len(ff) < 2:
        raise SystemExit("the full-frame curve needs at least two scales scored")
    ff_lat = [_latency(n, scores, fullframe) for n in ff]
    tiny = size_bins(cols["area"], bins) == _bins(bins)["size_names"][0]
    out: Dict[str, Any] = {"objects": int(len(cols["t"])), "found_iou": found_iou, "headline": head,
                           "fullframe": {"configs": ff, "latency_ms": ff_lat}, "configs": [], "screen": []}
    for name, iou in ious.items():
        found = iou > found_iou
        lat = _latency(name, scores, fullframe)
        entry = {"name": name, "family": _family(name), "latency_ms": lat,
                 "miou": None if name.startswith("ff/") else scores[name]["miou"],
                 "recall": recall_table(found, cols["clip"], groups, n_boot=n_boot),
                 "tiny_recall_iou_%g" % tiny_iou: recall_table(iou > tiny_iou, cols["clip"], {"tiny": tiny},
                                                                n_boot=n_boot)["tiny"]}
        for key in (head, "all"):
            entry.setdefault("vs_fullframe", {})[key] = recall_at_equal_latency(
                found, lat, [ious[n] > found_iou for n in ff], ff_lat, cols["clip"], groups[key], n_boot=n_boot)
        (out["screen"] if entry["family"] == "method2_screen" else out["configs"]).append(entry)
    return out


def cmd_report(args: argparse.Namespace) -> int:
    cols, _ = load_table(Path(args.table) if args.table else table_path(args.split))
    bins = load_bins(args.bins)
    scores, fullframe = load_scores(score_paths(args.score))
    for model in _strs(args.models):
        files = sorted((EXP / args.split).glob(f"iou_{model}_*.jsonl.gz"))
        if not files:
            raise SystemExit(f"no IoU files for {model} in {EXP / args.split}")
        rep = build_report(cols, collect_ious(files, cols), scores, fullframe, bins, n_boot=args.n_boot)
        rep.update(split=args.split, model=model, bins=bins)
        out = RES / f"report_{args.split}_{model}.json"
        _json_dump(rep, out)
        head = rep["headline"]
        print(f"[objects report] {model}: {head}")
        for e in sorted(rep["configs"], key=lambda e: e["latency_ms"]):
            r, v = e["recall"][head], e["vs_fullframe"][head]
            print(f"  {e['name']:<40} {e['latency_ms']:7.2f} ms  recall {r['recall']:.3f} "
                  f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  vs curve {v['delta_recall_at_equal_latency'] * 100:+.2f} pts")
        if rep["screen"]:
            print("  Method 2 screen (not a Method 2 result):")
            for e in rep["screen"]:
                v = e["vs_fullframe"][head]
                print(f"    {e['name']:<38} {e['latency_ms']:7.2f} ms  vs curve {v['delta_recall_at_equal_latency'] * 100:+.2f} pts")
        print(f"  -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="train_all")
        sp.add_argument("--models", default="pidnet_s")
        sp.add_argument("--root", default=str(EXP3 / "cache"), help="Method 3 cache root: <root>/<split>/<model>/<clip>.npz")
        sp.add_argument("--clip-ids", default=None)
        sp.add_argument("--chunk", default=None, help="i/N: the i-th block of the sorted clip ids.")
        sp.add_argument("--bins", default=None, help="Bin edges JSON (default results/objects/bins.json, else built-in).")
        sp.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)

    t = sub.add_parser("table", help="object table from the instance maps and the cached homographies")
    common(t)
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_table)

    s = sub.add_parser("score", help="local IoU of every object, per config and clip")
    common(s)
    s.add_argument("--score", action="append", help="Method 3 / Method 4 score file (repeatable).")
    s.add_argument("--configs", default=None, help="Comma-separated config names (ff/<scale> for the full frame).")
    s.add_argument("--ff-scales", default=None, help="Full-frame scales to score (default: the score file's).")
    s.add_argument("--edge", type=int, default=0, help="Also take up to N upper-edge configs per family.")
    s.add_argument("--tag", default=None)
    s.add_argument("--device", default="auto")
    s.add_argument("--max-temp", type=int, default=None, help="Pause before each config while the GPU is above this (C).")
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("report", help="recall tables and the reading against the full-frame curve")
    common(r)
    r.add_argument("--score", action="append", help="The same score files as for `score`.")
    r.add_argument("--table", default=None)
    r.add_argument("--n-boot", type=int, default=1000)
    r.set_defaults(func=cmd_report)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
