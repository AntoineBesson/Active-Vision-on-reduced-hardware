#!/usr/bin/env python3
"""Method 9: a realisable per-frame scale selector (PIDNet-S), from the clip's own past outputs.

    python -m scripts.method9 features --split train_all --chunk 0/4      # GPU: every frame at scale 0.5, features
    python -m scripts.method9 merge --split train_all                     # chunk files -> one file
    python -m scripts.method9 select --split train_all                    # CPU: S1 / S2 cross-fitted, train gate
    python -m scripts.method9 closedloop --split val                      # GPU: val, closed loop, the fitted selectors
    python -m scripts.method9 score --split val                           # the pass rule (needs the val dense run)

Spec: docs/superpowers/specs/2026-11-22-method9-scale-selector-design.md. The labelled frames'
confusions at every scale come from the Method 7 run files (``scripts.method7 run``); the oracle's
choices at the budget from ``results/method7/oracle_train_all.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _strs, load_dataset, model_config  # noqa: E402
from scripts.method3 import _tag, select_clips  # noqa: E402
from scripts import method7 as m7  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method9"
RES = REPO_ROOT / "results" / "method9"
MODEL = "pidnet_s"
FEATURE_SCALE = 0.5  # the open-loop pass on train, and the budget
BUDGET_SCALE = 0.5
GATE_MARGIN = 0.005  # train gate: +0.5 mIoU point above the envelope
KINDS = ("S1", "S2")


def _out_dir(args) -> Path:
    return Path(args.out_dir) if args.out_dir else EXP / args.split


def _model(args) -> str:
    return "dummy" if args.fixture else MODEL


# --------------------------------------------------------------------------- #
def cmd_features(args: argparse.Namespace) -> int:
    import torch

    from eval.harness import FullFrameMethod, frame_to_tensor
    from models.registry import build_backbone
    from upsample.selector import output_features

    device = resolve_device(args.device)
    model = _model(args)
    dataset = load_dataset(model_config(model, args.fixture), args.split)
    clips = select_clips([dataset.clip_id(i) for i in range(len(dataset))], args.clip_ids, args.chunk)
    path = _out_dir(args) / f"features_{args.split}_{model}_{_tag(args)}.jsonl.gz"
    done = m7._done(path)
    todo = [i for i in clips if dataset.clip_id(i) not in done]
    print(f"[method9 features] {model}: {len(todo)} clips to run, {len(clips) - len(todo)} in {path.name}", flush=True)
    if not todo:
        return 0
    net = build_backbone(model_config(model, args.fixture)["backbone"], context={"palette": dataset.palette}).to(device).eval()
    t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(todo):
            m7.wait_for_gpu(args.max_temp)
            feats = []
            for t in range(dataset.num_frames(i)):
                frame = frame_to_tensor(dataset.load_frame(i, t), device)
                feats.append(output_features(net(FullFrameMethod(net, scale=FEATURE_SCALE)._resize(frame))).tolist())
            labelled = [int(t) for t, v in enumerate(dataset.labeled_mask(i)) if v]
            m7._append(path, [{"clip_id": dataset.clip_id(i), "model": model, "scale": FEATURE_SCALE,
                               "labelled": labelled, "features": feats}])
            dt = time.time() - t0
            print(f"[method9 features] clip {n + 1}/{len(todo)} ({dt / (n + 1):.1f} s/clip)", flush=True)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    model = _model(args)
    out = _out_dir(args) / f"features_{args.split}_{model}.jsonl.gz"
    rows: Dict[str, Any] = {}
    for f in sorted(p for p in _out_dir(args).glob(f"features_{args.split}_{model}_*.jsonl.gz") if p != out):
        for r in m7.read_lines(f):
            if r["clip_id"] in rows:
                raise SystemExit(f"{f}: clip {r['clip_id']} seen twice")
            rows[r["clip_id"]] = r
    tmp = out.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    m7._append(tmp, (rows[k] for k in sorted(rows)))
    tmp.replace(out)
    print(f"[method9 merge] {len(rows)} clips -> {out}")
    return 0


# --------------------------------------------------------------------------- #
def _read(choice, r, lat, curves, boot_curves, W) -> Dict[str, Any]:
    """mIoU, mean latency (labelled frames) and delta vs the envelope with the clip bootstrap."""
    import numpy as np

    from upsample.scale_oracle import miou_many

    n_clips, F = len(r["clips"]), len(r["frame_clip"])
    per_clip = np.zeros((n_clips, m7.NUM_CLASSES, m7.NUM_CLASSES), dtype=np.int64)
    np.add.at(per_clip, r["frame_clip"], r["ff_frames"][np.arange(F), choice])
    miou, mean_ms = float(miou_many(per_clip.sum(0))), float(np.asarray(lat)[choice].mean())
    env, _ = m7.envelope_at(curves, mean_ms)
    env_b, _ = m7.envelope_at(boot_curves, mean_ms)
    d = miou_many((W @ per_clip.reshape(n_clips, -1)).reshape(-1, m7.NUM_CLASSES, m7.NUM_CLASSES)) - env_b
    d = d[~np.isnan(d)]
    return {"miou": miou, "mean_ms": mean_ms, "delta_envelope": float(miou - env),
            "delta_envelope_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "scale_share": np.bincount(choice, minlength=len(lat)).tolist()}


def _runs(args) -> Dict[str, Any]:
    """The Method 7 run files of the envelope's models (``--run-dir``)."""
    ns = argparse.Namespace(**vars(args))
    ns.out_dir = args.run_dir or str(m7.EXP / args.split)
    return m7._load_runs(ns, [_model(args)] if args.fixture else _strs(args.models))


def _curves(args, runs):
    from upsample.scale_oracle import miou_many

    prices, _ = m7._prices(args, runs)
    models = list(runs)
    n = len(runs[models[0]]["clips"])
    W = m7._boot_weights(n, args.n_boot)
    curves = {m: (prices[m], miou_many(runs[m]["ff"].sum(0))) for m in models}
    boot = {m: (prices[m], miou_many((W @ runs[m]["ff"].reshape(n, -1)).reshape(
        args.n_boot, len(runs[m]["ff_names"]), m7.NUM_CLASSES, m7.NUM_CLASSES))) for m in models}
    return prices, curves, boot, W


def cmd_select(args: argparse.Namespace) -> int:
    import numpy as np

    from upsample.selector import crossfit, fit_selector, past_features

    model = _model(args)
    runs = _runs(args)
    r = runs[model]
    prices, curves, boot, W = _curves(args, runs)
    lat = np.asarray(prices[model])
    base = r["ff_scales"].index(BUDGET_SCALE)
    feats = {x["clip_id"]: x for x in m7.read_lines(Path(args.features) if args.features else
                                                       _out_dir(args) / f"features_{args.split}_{model}.jsonl.gz")}
    oracle = json.loads(Path(args.oracle).read_text())
    entry = next(e for e in oracle["budgets"][model] if abs(e["budget_scale"] - BUDGET_SCALE) < 1e-9)
    target = np.asarray(entry["choice"])
    order = [r["clips"][j] for j in r["frame_clip"]]
    if entry["frame_clip"] != order:
        raise SystemExit("the oracle file's frames are not the run file's: rerun method7 oracle on these run files")
    x = np.full((len(order), 22 * 2), np.nan)
    pos: Dict[str, int] = {}
    for f, clip in enumerate(order):
        j = pos.get(clip, 0)
        pos[clip] = j + 1
        fx = feats.get(clip)
        if fx is None:
            raise SystemExit(f"clip {clip}: no features (run `method9 features`)")
        t = r["frames"][clip][j]
        x[f] = past_features(np.asarray(fx["features"]))[t]
    has = ~np.isnan(x).any(1)
    groups = np.asarray(order)
    rep: Dict[str, Any] = {"split": args.split, "model": model, "budget_scale": BUDGET_SCALE,
                           "budget_ms": float(lat[base]), "frames": int(len(order)), "with_past": int(has.sum()),
                           "margin": GATE_MARGIN, "oracle": _read(target, r, lat, curves, boot, W), "selectors": {}}
    fitted = {}
    for kind in KINDS:
        choice = np.full(len(order), base)
        choice[has] = crossfit(kind, x[has], target[has], groups[has], len(lat), folds=args.folds)
        res = _read(choice, r, lat, curves, boot, W)
        res["agree_with_oracle"] = float(np.mean(choice == target))
        res["continue"] = res["delta_envelope"] >= GATE_MARGIN - 1e-12
        rep["selectors"][kind] = res
        fitted[kind] = fit_selector(kind, x[has], target[has], len(lat)).to_json()
        print(f"[method9 select] {kind} (cross-fitted): {100 * res['miou']:.2f} @ {res['mean_ms']:.2f} ms  vs envelope "
              f"{100 * res['delta_envelope']:+.2f} [{100 * res['delta_envelope_ci95'][0]:+.2f}, "
              f"{100 * res['delta_envelope_ci95'][1]:+.2f}]  same as oracle {100 * res['agree_with_oracle']:.0f}%  "
              f"-> {'CONTINUE' if res['continue'] else 'STOP'}")
    o = rep["oracle"]
    print(f"[method9 select] oracle: {100 * o['miou']:.2f} @ {o['mean_ms']:.2f} ms vs envelope {100 * o['delta_envelope']:+.2f}")
    rep["continue"] = [k for k in KINDS if rep["selectors"][k]["continue"]]
    out_dir = EXP if args.fixture else RES
    m7._json_dump(rep, out_dir / f"select_{args.split}{'_fixture' if args.fixture else ''}.json")
    m7._json_dump({"model": model, "budget_scale": BUDGET_SCALE, "scales": r["ff_scales"], "selectors": fitted},
                  out_dir / f"selectors_{model}{'_fixture' if args.fixture else ''}.json")
    print(f"[method9 select] continue: {rep['continue'] or 'none: Method 9 stops'} -> {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_closedloop(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from eval.harness import FullFrameMethod, frame_to_tensor
    from models.registry import build_backbone
    from upsample.guided import bilinear_labels
    from upsample.selector import Selector, output_features, past_features

    device = resolve_device(args.device)
    model = _model(args)
    spec = json.loads(Path(args.selectors.format(model=model)).read_text())
    scales = spec["scales"]
    sels = {k: Selector.from_json(v) for k, v in spec["selectors"].items() if k in _strs(args.kinds)}
    base = scales.index(spec["budget_scale"])
    dataset = load_dataset(model_config(model, args.fixture), args.split)
    clips = select_clips([dataset.clip_id(i) for i in range(len(dataset))], args.clip_ids, args.chunk)
    path = _out_dir(args) / f"closedloop_{args.split}_{model}_{_tag(args)}.jsonl.gz"
    done = m7._done(path)
    net = build_backbone(model_config(model, args.fixture)["backbone"], context={"palette": dataset.palette}).to(device).eval()
    with torch.no_grad():
        for n, i in enumerate(c for c in clips if dataset.clip_id(c) not in done):
            m7.wait_for_gpu(args.max_temp)
            rows = {}
            for kind, sel in sels.items():
                feats: List[List[float]] = []
                chosen, conf = [], np.zeros((m7.NUM_CLASSES, m7.NUM_CLASSES), dtype=np.int64)
                labelled_scale = []
                for t in range(dataset.num_frames(i)):
                    if t == 0:
                        k = base
                    else:
                        past = past_features(np.asarray(feats + [feats[-1]]))[-1]
                        k = int(sel.choose(past[None])[0])
                    frame = frame_to_tensor(dataset.load_frame(i, t), device)
                    x = FullFrameMethod(net, scale=scales[k])._resize(frame)
                    logits = net(x)
                    feats.append(output_features(logits).tolist())
                    chosen.append(k)
                    if dataset.labeled_mask(i)[t]:
                        label = torch.from_numpy(np.asarray(dataset.load_label(i, t), dtype=np.int64)).to(device)
                        pred = bilinear_labels(logits, tuple(frame.shape[-2:]), net.output_stride, net.align_corners,
                                               tuple(x.shape[-2:]))
                        conf += m7._confusion(label, pred).cpu().numpy()
                        labelled_scale.append(k)
                rows[kind] = {"scales": chosen, "labelled_scales": labelled_scale, "conf": conf.tolist()}
            m7._append(path, [{"clip_id": dataset.clip_id(i), "model": model, "selectors": rows}])
            print(f"[method9 closedloop] clip {n + 1}", flush=True)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Val pass rule: each selector's closed-loop mIoU vs the envelope at its mean latency over ALL
    frames (the selector runs on every frame), with the clip bootstrap."""
    import numpy as np

    from upsample.scale_oracle import miou_many

    model = _model(args)
    runs = _runs(args)
    prices, curves, boot, W = _curves(args, runs)
    lat = np.asarray(prices[model])
    rows = {r["clip_id"]: r for f in sorted(_out_dir(args).glob(f"closedloop_{args.split}_{model}_*.jsonl.gz"))
            for r in m7.read_lines(f)}
    clips = runs[model]["clips"]
    missing = [c for c in clips if c not in rows]
    if missing:
        raise SystemExit(f"closed loop missing {len(missing)} clips, e.g. {missing[:3]}")
    rep: Dict[str, Any] = {"split": args.split, "model": model, "selector_overhead_ms": args.overhead_ms, "selectors": {}}
    for kind in rows[clips[0]]["selectors"]:
        conf = np.asarray([rows[c]["selectors"][kind]["conf"] for c in clips], dtype=np.int64)
        scales = np.concatenate([rows[c]["selectors"][kind]["scales"] for c in clips])
        mean_ms = float(lat[scales].mean()) + args.overhead_ms
        miou = float(miou_many(conf.sum(0)))
        env, _ = m7.envelope_at(curves, mean_ms)
        env_b, _ = m7.envelope_at(boot, mean_ms)
        d = miou_many((W @ conf.reshape(len(clips), -1)).reshape(-1, m7.NUM_CLASSES, m7.NUM_CLASSES)) - env_b
        d = d[~np.isnan(d)]
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        rep["selectors"][kind] = {"miou": miou, "mean_ms": mean_ms, "delta_envelope": float(miou - env),
                                  "delta_envelope_ci95": ci, "pass": float(miou - env) > 0 and ci[0] > 0}
        print(f"[method9 score] {kind}: {100 * miou:.2f} @ {mean_ms:.2f} ms  vs envelope {100 * (miou - env):+.2f} "
              f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]  -> {'PASS' if rep['selectors'][kind]['pass'] else 'FAIL'}")
    out = (EXP if args.fixture else RES) / f"score_{args.split}{'_fixture' if args.fixture else ''}.json"
    m7._json_dump(rep, Path(args.out) if args.out else out)
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="train_all")
        sp.add_argument("--out-dir", default=None)
        sp.add_argument("--fixture", action="store_true", help=argparse.SUPPRESS)

    def gpu(sp):
        sp.add_argument("--clip-ids", default=None)
        sp.add_argument("--chunk", default=None)
        sp.add_argument("--tag", default=None)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--max-temp", type=int, default=None)

    def scoring(sp):
        sp.add_argument("--models", default="pidnet_s,bisenetv2", help="Models of the envelope (Method 7 run files).")
        sp.add_argument("--run-dir", default=None, help="Method 7 run files (default experiments/method7/<split>).")
        sp.add_argument("--run-file", default=None)
        sp.add_argument("--e2e", default=m7.DEFAULT_E2E)
        sp.add_argument("--timing", default=None)
        sp.add_argument("--n-boot", type=int, default=1000)
        sp.add_argument("--out", default=None)

    f = sub.add_parser("features")
    common(f)
    gpu(f)
    f.set_defaults(func=cmd_features)
    m = sub.add_parser("merge")
    common(m)
    m.set_defaults(func=cmd_merge)
    s = sub.add_parser("select")
    common(s)
    scoring(s)
    s.add_argument("--features", default=None)
    s.add_argument("--oracle", default=str(REPO_ROOT / "results" / "method7" / "oracle_train_all.json"))
    s.add_argument("--folds", type=int, default=5)
    s.set_defaults(func=cmd_select)
    c = sub.add_parser("closedloop")
    common(c)
    gpu(c)
    c.add_argument("--selectors", default=str(RES / "selectors_{model}.json"))
    c.add_argument("--kinds", default=",".join(KINDS), help="The selectors that passed the train gate.")
    c.set_defaults(func=cmd_closedloop)
    v = sub.add_parser("score")
    common(v)
    scoring(v)
    v.add_argument("--overhead-ms", type=float, default=0.0, help="The selector's T4 overhead per frame.")
    v.set_defaults(func=cmd_score)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
