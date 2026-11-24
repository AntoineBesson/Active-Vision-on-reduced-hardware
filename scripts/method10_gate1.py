#!/usr/bin/env python3
"""Method 10, gate 1: how much accuracy does stale context cost? (train_all, school cluster)

Spec: docs/superpowers/specs/2026-11-23-method10-slow-branch-reuse-design.md (gate 1).

    python -m scripts.method10_gate1 run --split train_all --models pidnet_s --tag c0of4   # GPU, per clip
    python -m scripts.method10_gate1 merge --split train_all                              # chunks -> one file per model
    python -m scripts.method10_gate1 score --split train_all                              # the gate 1 verdict

``run``, on every labelled frame t of each clip:

* the model's full frame at the 11 envelope scales (``ff/<s>``), so the envelope and its
  moving-object recall come from the same run as the method;
* at the gate scale s (0.5): the fresh branch on frame t with the slow tensors of frame t - k,
  k = 1..4, as they are (variant ``N``) and warped onto frame t by the chain of IC-LK
  homographies of Method 3 (variant ``H0``, priced at 0 ms: a bound only). k = 0 is ``ff/<s>``
  (the split reproduces the full frame exactly: ``tests/test_method10_split.py``).

One 19 x 19 confusion per (frame, configuration) and, with ``--obj``, the local IoU of every
thing object (``eval.objects``), in the object table's order.

``score``: a schedule with interval K has its labelled frame at staleness 0..K-1 with equal
probability (uniform key-frame phase), so its accuracy is the mIoU of the confusions pooled over
k = 0..K-1, and its moving-object recall the mean over the same k. Its mean latency is
``(T_key + (K-1) T_fresh) / K`` from the gate 0 timing, and the envelope (the higher of both
models' full-frame curves) is priced by the same gate 0 session. Gate 1 (variant N only): a model
continues if some K has Delta mIoU >= +1.0 point vs the envelope at its mean latency AND its
headline moving-object recall vs the envelope's at the same latency has a 95 % clip-bootstrap
interval not entirely below 0. At most 2 finalists per model.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/method10_gate1.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _strs, load_dataset, model_config  # noqa: E402
from scripts.method3 import _tag, select_clips  # noqa: E402
from scripts.method10 import ENVELOPE_SCALES, KS, MARGIN, curve_at, load_timing  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method10"
RES = REPO_ROOT / "results" / "method10"
NUM_CLASSES, IGNORE = 19, 255
SCALE = 0.5
KMAX = 4
VARIANTS = ("N", "H0")
MAX_FINALISTS = 2


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="network passes, per clip (GPU)")
    r.add_argument("--split", default="train_all")
    r.add_argument("--models", default="pidnet_s,bisenetv2")
    r.add_argument("--scale", type=float, default=SCALE)
    r.add_argument("--kmax", type=int, default=KMAX)
    r.add_argument("--variants", default=",".join(VARIANTS))
    r.add_argument("--obj", action="store_true", help="also record object IoUs (the moving-object metric)")
    r.add_argument("--chunk", default=None, help="i/N block of the sorted clip ids")
    r.add_argument("--clip-ids", default=None)
    r.add_argument("--tag", default=None)
    r.add_argument("--out-dir", default=str(EXP / "train_all"))
    r.add_argument("--device", default="auto")
    r.add_argument("--max-temp", type=int, default=None, help="laptop rule: pause at this GPU temperature")
    r.add_argument("--fixture", action="store_true", help="synthetic fixture, random weights (smoke test)")
    m = sub.add_parser("merge", help="chunk files -> run_<split>_<model>.jsonl.gz")
    m.add_argument("--split", default="train_all")
    m.add_argument("--models", default="pidnet_s,bisenetv2")
    m.add_argument("--out-dir", default=str(EXP / "train_all"))
    s = sub.add_parser("score", help="the gate 1 verdict")
    s.add_argument("--split", default="train_all")
    s.add_argument("--models", default="pidnet_s,bisenetv2")
    s.add_argument("--out-dir", default=str(EXP / "train_all"))
    s.add_argument("--timing", nargs="+", default=[str(RES / "gate0_s1.json")], help="gate 0 timing JSONs")
    s.add_argument("--table", default=None, help="object table (default results/objects/objects_<split>.json.gz)")
    s.add_argument("--bins", default=None)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--report", default=None, help="default results/method10/gate1_<split>.json")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- files

def read_lines(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _append(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def ff_name(s: float) -> str:
    return f"ff/{s:g}"


def m_name(variant: str, k: int) -> str:
    return f"{variant}/k{k}"


# --------------------------------------------------------------------------- run (GPU)

def _confusion(label, pred):
    import torch

    ok = label != IGNORE
    return torch.bincount(label[ok].long() * NUM_CLASSES + pred[ok].long(),
                          minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def run_clip(dataset, index: int, backbone, scale: float, kmax: int, variants: Sequence[str], device,
             obj: bool) -> Dict[str, Any]:
    import numpy as np
    import torch

    from eval.harness import FullFrameMethod, frame_to_tensor
    from models.utils import upsample_logits
    from reuse.motion import estimate_homography, to_gray_small, warp
    from reuse.slowfast import slowfast_split

    split = slowfast_split(backbone)
    labelled = [int(t) for t in np.flatnonzero(np.asarray(dataset.labeled_mask(index), dtype=bool))]
    resize = FullFrameMethod(backbone, scale=scale)._resize
    gt = None
    if obj:
        from scripts.objects import clip_ground_truth

        gt = clip_ground_truth(dataset, index, labelled)

    ff = {ff_name(s): [] for s in ENVELOPE_SCALES}
    meth = {m_name(v, k): [] for v in variants for k in range(1, kmax + 1)}
    ious: Dict[str, List[float]] = {}
    frames: Dict[int, Any] = {}

    def frame(t: int):
        if t not in frames:
            frames[t] = frame_to_tensor(dataset.load_frame(index, t), device)
        return frames[t]

    caches: Dict[int, Any] = {}
    gray: Dict[int, Any] = {}
    step_h: Dict[int, Any] = {}  # H of t -> t+1
    tiny = [0]

    def labels(logits, size, x):
        return upsample_logits(logits.float(), size, backbone.output_stride, backbone.align_corners,
                               input_size=tuple(x.shape[-2:])).argmax(1)[0]

    def homography(a: int, b: int):
        """a -> b (a < b), chained over consecutive frames."""
        H = torch.eye(3, dtype=torch.float64)
        for t in range(a, b):
            if t not in step_h:
                for u in (t, t + 1):
                    if u not in gray:
                        gray[u] = to_gray_small(frame(u))
                if min(gray[t].shape) < 16:  # only the synthetic fixture's 16 x 32 frames: too small for IC-LK
                    step_h[t] = torch.eye(3, dtype=torch.float64)
                    tiny[0] += 1
                else:
                    step_h[t] = estimate_homography(gray[t], gray[t + 1], method="iclk")[0]
            H = step_h[t] @ H
        return H / H[2, 2]

    with torch.no_grad():
        for n, t in enumerate(labelled):
            if t - kmax < 0:
                raise ValueError(f"clip {dataset.clip_id(index)}: labelled frame {t} has fewer than {kmax} frames before it")
            cur = frame(t)
            size = tuple(cur.shape[-2:])
            label = torch.from_numpy(np.asarray(dataset.load_label(index, t), dtype=np.int64)).to(device)

            def score(name: str, pred, store: Dict[str, List[Any]]) -> None:
                store[name].append(_confusion(label, pred).cpu().numpy().tolist())
                if gt is not None:
                    from eval.objects import local_ious

                    ious.setdefault(name, []).extend(
                        local_ious(gt[0][n], pred.to(torch.uint8).cpu().numpy(), gt[1][n]).tolist())

            for s in ENVELOPE_SCALES:
                x = FullFrameMethod(backbone, scale=s)._resize(cur)
                score(ff_name(s), labels(backbone(x), size, x), ff)
            x = resize(cur)
            for k in range(1, kmax + 1):
                src = t - k
                if src not in caches:
                    caches[src] = split.slow(resize(frame(src)))
                for v in variants:
                    cache = caches[src]
                    if v == "H0":
                        H = homography(src, t)
                        cache = tuple(warp(c, H, "bilinear", frame_size=size)[0] for c in cache)
                    elif v != "N":
                        raise ValueError(f"unknown variant {v!r}")
                    score(m_name(v, k), labels(split.fresh(x, cache), size, x), meth)
            nxt = labelled[n + 1] if n + 1 < len(labelled) else None
            keep = (nxt - kmax) if nxt is not None else None  # the next labelled frame needs frames keep..nxt
            for store in (frames, caches, gray, step_h):
                for u in [u for u in store if keep is None or u < keep]:
                    store.pop(u)
    return {"frames": labelled, "ff": ff, "m10": meth, "iou": ious, "h0_identity_steps": tiny[0]}


def cmd_run(args: argparse.Namespace) -> int:
    from models.registry import build_backbone

    device = resolve_device(args.device)
    dataset = load_dataset(model_config(_strs(args.models)[0], args.fixture), args.split)
    clips = select_clips([dataset.clip_id(i) for i in range(len(dataset))], args.clip_ids, args.chunk)
    variants = _strs(args.variants)
    for model in _strs(args.models):
        path = Path(args.out_dir) / f"run_{args.split}_{model}_{_tag(args)}.jsonl.gz"
        done = {r["clip_id"] for r in read_lines(path)} if path.is_file() else set()
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method10 gate1] {model}: s={args.scale:g}, k=1..{args.kmax}, variants {variants}, obj={args.obj}; "
              f"{len(todo)} clips to run, {len(clips) - len(todo)} already in {path.name}", flush=True)
        if not todo:
            continue
        cfg = ({"type": model, "num_classes": 19, "allow_random_init": True} if args.fixture
               else model_config(model, False)["backbone"])
        backbone = build_backbone(cfg, context={"palette": dataset.palette}).to(device).eval()
        t0 = time.time()
        for n, i in enumerate(todo):
            if args.max_temp:
                from scripts.objects import wait_for_gpu

                wait_for_gpu(args.max_temp)
            out = run_clip(dataset, i, backbone, args.scale, args.kmax, variants, device, args.obj)
            _append(path, [{"split": args.split, "model": model, "clip_id": dataset.clip_id(i),
                            "scale": args.scale, "kmax": args.kmax, **out}])
            dt = time.time() - t0
            print(f"[method10 gate1] {model} clip {n + 1}/{len(todo)} ({dt:.0f} s, {dt / (n + 1):.1f} s/clip)",
                  flush=True)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    for model in _strs(args.models):
        merged = out_dir / f"run_{args.split}_{model}.jsonl.gz"
        files = sorted(f for f in out_dir.glob(f"run_{args.split}_{model}_*.jsonl.gz") if f != merged)
        clips: Dict[str, Dict[str, Any]] = {}
        for f in files:
            for r in read_lines(f):
                if r["clip_id"] in clips:
                    raise SystemExit(f"{f}: clip {r['clip_id']} also in another chunk file")
                clips[r["clip_id"]] = r
        tmp = merged.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        _append(tmp, (clips[k] for k in sorted(clips)))
        tmp.replace(merged)
        print(f"[method10 gate1 merge] {model}: {len(clips)} clips from {len(files)} files -> {merged}")
    return 0


# --------------------------------------------------------------------------- score (numpy)

def miou(conf) -> Any:
    """mIoU of (..., C, C) confusions over the classes present (union > 0)."""
    import numpy as np

    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    union = conf.sum(-1) + conf.sum(-2) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / np.where(union > 0, union, 1), np.nan)
    return np.nanmean(iou, axis=-1)


def load_run(path: Path) -> Dict[str, Any]:
    """-> clip ids, per-clip summed confusions {name: (clips, C, C)}, per-clip IoU lists."""
    import numpy as np

    rows = sorted(read_lines(path), key=lambda r: r["clip_id"])
    if not rows:
        raise SystemExit(f"{path}: empty")
    names = sorted(rows[0]["ff"]) + sorted(rows[0]["m10"])
    conf = {n: np.zeros((len(rows), NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for n in names}
    for j, r in enumerate(rows):
        for n in names:
            per = r["ff"].get(n) if n.startswith("ff/") else r["m10"].get(n)
            if per is None:
                raise SystemExit(f"{path}: clip {r['clip_id']} lacks {n}")
            conf[n][j] = np.asarray(per, dtype=np.int64).sum(0)
    return {"clips": [r["clip_id"] for r in rows], "conf": conf, "iou": {r["clip_id"]: r.get("iou", {}) for r in rows},
            "frames": {r["clip_id"]: r["frames"] for r in rows}, "scale": rows[0]["scale"], "kmax": rows[0]["kmax"]}


def schedule_conf(run: Dict[str, Any], variant: str, K: int):
    """Per-clip confusions pooled over staleness 0..K-1 (k = 0 is the full frame at the gate scale)."""
    total = run["conf"][ff_name(run["scale"])].copy()
    for k in range(1, K):
        total = total + run["conf"][m_name(variant, k)]
    return total


def envelope_curve(per_model: Dict[str, Tuple[Sequence[float], Any]], w, at: float) -> float:
    """Max over models of their full-frame curve at ``at``; per_model: model -> (latencies, (clips, S, C, C))."""
    import numpy as np

    best = np.nan
    for lat, confs in per_model.values():
        vals = miou(np.tensordot(w, confs, axes=(0, 0)))  # (S,)
        v = curve_at(list(zip(lat, (100 * vals).tolist())), at)
        if v is not None and not (best >= v):
            best = v
    return best


def boot_weights(n: int, n_boot: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]).astype(np.float64)


def ci(values) -> List[float]:
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def moving_recall(runs: Dict[str, Dict[str, Any]], cols, head_mask, prices: Dict[str, List[float]], model: str,
                  variant: str, K: int, mean_ms: float, n_boot: int) -> Dict[str, Any]:
    """Headline recall of the schedule (mean over k = 0..K-1) minus the envelope of both models' recall
    curves at its mean latency, with a clip bootstrap (the same resamples for every curve)."""
    import numpy as np

    starts: Dict[str, Tuple[int, int]] = {}
    for c in dict.fromkeys(cols["clip"].tolist()):
        idx = np.flatnonzero(cols["clip"] == c)
        starts[c] = (int(idx[0]), int(idx[-1]) + 1)

    def found(m: str, name: str):
        arr = np.full(len(cols["clip"]), np.nan)
        for clip, d in runs[m]["iou"].items():
            if clip in starts and name in d:
                a, b = starts[clip]
                if len(d[name]) != b - a:
                    raise SystemExit(f"{m} {clip} {name}: {len(d[name])} IoUs for {b - a} objects")
                arr[a:b] = d[name]
        if np.isnan(arr[head_mask]).any():
            raise SystemExit(f"{m} {name}: object IoUs missing (run with --obj)")
        return arr > 0.5

    r = runs[model]
    names = [ff_name(r["scale"])] + [m_name(variant, k) for k in range(1, K)]
    sched = np.mean([found(model, n) for n in names], axis=0)  # expected found over phases
    clips, cidx = np.unique(cols["clip"], return_inverse=True)
    nc = len(clips)
    mask = np.asarray(head_mask, bool)
    tot = np.bincount(cidx[mask], minlength=nc).astype(np.float64)
    hit = np.bincount(cidx[mask], weights=sched[mask], minlength=nc)
    curves = {m: (prices[m], np.stack([np.bincount(cidx[mask], weights=found(m, ff_name(s))[mask], minlength=nc)
                                       for s in ENVELOPE_SCALES])) for m in runs}

    def reading(w):
        d = float(w @ tot)
        rec = float(w @ hit) / d
        env = max((v for v in (curve_at(list(zip(lat, (h @ w / d).tolist())), mean_ms) for lat, h in curves.values())
                   if v is not None), default=float("nan"))
        return rec, rec - env

    point = reading(np.ones(nc))
    boots = [reading(w) for w in boot_weights(nc, n_boot, seed=1)]
    return {"objects": int(tot.sum()), "recall": point[0], "delta_vs_envelope": point[1],
            "delta_ci95": ci([b[1] for b in boots])}


def cmd_score(args: argparse.Namespace) -> int:
    import numpy as np

    from scripts.method10 import _expand
    from scripts.objects import groups_of, load_bins, load_table, table_path

    models = _strs(args.models)
    runs = {m: load_run(Path(args.out_dir) / f"run_{args.split}_{m}.jsonl.gz") for m in models}
    clip_sets = {m: r["clips"] for m, r in runs.items()}
    if len({tuple(v) for v in clip_sets.values()}) != 1:
        raise SystemExit(f"the models' run files cover different clips: {[len(v) for v in clip_sets.values()]}")
    timing = load_timing(_expand(args.timing))
    prices = {m: [timing[("ff", m, float(s))]["median_ms"] for s in ENVELOPE_SCALES] for m in models}
    per_model = {m: (prices[m], np.stack([runs[m]["conf"][ff_name(s)] for s in ENVELOPE_SCALES], axis=1))
                 for m in models}
    n = len(runs[models[0]]["clips"])
    W = boot_weights(n, args.n_boot)
    ones = np.ones(n)
    cols = head = None
    table = Path(args.table) if args.table else table_path(args.split)
    if table.is_file() and all(any(r["iou"].values()) for r in runs.values()):
        cols, _ = load_table(table)
        groups = groups_of(cols, load_bins(args.bins))
        head = next(k for k in groups if k.startswith("headline"))
        head_mask = groups[head]
    rep: Dict[str, Any] = {"split": args.split, "clips": n, "margin": MARGIN, "timing": _expand(args.timing),
                           "fullframe_check": {m: {f"{s:g}": float(100 * miou(per_model[m][1][:, j].sum(0)))
                                                   for j, s in enumerate(ENVELOPE_SCALES)} for m in models},
                           "moving_headline": head, "models": {}}
    ref = RES / "fullframe_train_all.json"
    if args.split == "train_all" and ref.is_file():  # the same frames as Methods 7/8: their curves must come back
        known = json.loads(ref.read_text())["models"]
        for m in models:
            if m in known:
                diff = max(abs(rep["fullframe_check"][m][f"{s:g}"] - 100 * v)
                           for s, v in zip(known[m]["scales"], known[m]["miou"]))
                rep["fullframe_check"][m + "_max_abs_diff_vs_method7"] = diff
                print(f"[method10 gate1] {m}: full-frame curve vs Methods 7/8, max |diff| {diff:.3f} mIoU points")
    for m in models:
        r = runs[m]
        s = r["scale"]
        t_key, t_fresh = timing[("key", m, s)]["median_ms"], timing[("fresh", m, s)]["median_ms"]
        rows = []
        variants = sorted({name.split("/")[0] for name in r["conf"] if not name.startswith("ff/")})
        for v in variants:
            for K in [k for k in KS if k - 1 <= r["kmax"]]:
                conf = schedule_conf(r, v, K)
                mean_ms = (t_key + (K - 1) * t_fresh) / K
                val = float(100 * miou(conf.sum(0)))
                env = envelope_curve(per_model, ones, mean_ms)
                boots = [float(100 * miou(np.tensordot(w, conf, axes=(0, 0)))) - envelope_curve(per_model, w, mean_ms)
                         for w in W]
                row = {"variant": v, "K": K, "scale": s, "mean_ms": mean_ms, "miou": val,
                       "ff_same_scale": float(100 * miou(r["conf"][ff_name(s)].sum(0))),
                       "envelope": env, "delta_envelope": val - env, "delta_ci95": ci(boots),
                       "stale_k_miou": {f"k{k}": float(100 * miou(r["conf"][m_name(v, k)].sum(0)))
                                        for k in range(1, r["kmax"] + 1)}}
                if cols is not None:
                    row["moving"] = moving_recall(runs, cols, head_mask, prices, m, v, K, mean_ms, args.n_boot)
                rows.append(row)
        passing = [x for x in rows if x["variant"] == "N" and x["delta_envelope"] >= MARGIN - 1e-9
                   and "moving" in x and x["moving"]["delta_ci95"][1] >= 0]
        finalists = sorted(passing, key=lambda x: -x["delta_envelope"])[:MAX_FINALISTS]
        rep["models"][m] = {"t_key_ms": t_key, "t_fresh_ms": t_fresh, "rows": rows,
                            "finalists": [f"N/K{x['K']}" for x in finalists], "continue": bool(finalists)}
        print(f"\n== {m} (s={s:g}; key {t_key:.2f} ms, fresh {t_fresh:.2f} ms): gate 1 "
              f"{'PASSES ' + str(rep['models'][m]['finalists']) if finalists else 'FAILS'}")
        for x in rows:
            mv = x.get("moving")
            mv_txt = "" if not mv else (f"; moving recall {100 * mv['recall']:.1f}, vs envelope "
                                        f"{100 * mv['delta_vs_envelope']:+.1f} [{100 * mv['delta_ci95'][0]:+.1f}, "
                                        f"{100 * mv['delta_ci95'][1]:+.1f}]")
            print(f"  {x['variant']:2s} K={x['K']}: {x['mean_ms']:.2f} ms, mIoU {x['miou']:.2f} (FF {x['ff_same_scale']:.2f}), "
                  f"envelope {x['envelope']:.2f}, delta {x['delta_envelope']:+.2f} [{x['delta_ci95'][0]:+.2f}, "
                  f"{x['delta_ci95'][1]:+.2f}]{mv_txt}")
        print("     stale-k mIoU:", {v: {k: round(val, 2) for k, val in x["stale_k_miou"].items()}
                                     for v, x in {x["variant"]: x for x in rows}.items()})
    if cols is None:
        print("\n(no object IoUs or no object table: moving-object recall not scored, so nothing can pass)")
    out = Path(args.report) if args.report else RES / f"gate1_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return {"run": cmd_run, "merge": cmd_merge, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
