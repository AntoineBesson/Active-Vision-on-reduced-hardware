#!/usr/bin/env python3
"""Method 11: fine-tune the segmenter on warped frames.

Spec: docs/superpowers/specs/2026-11-25-method11-warp-finetune-design.md.

    # once: the boundary prior B-bar from the Cityscapes train labels (CPU, a few minutes)
    python -m scripts.method11 boundary-prior --cityscapes /data/cityscapes

    # one run (cluster: scripts/cluster/m11_train.sh)
    python -m scripts.method11 train --model bisenetv2 --kind boundary --scale 0.25 \\
        --cityscapes /data/cityscapes --run-dir experiments/method11/runs/bisenetv2_B_c0.25

    # its confusions on VPS train_all, per clip (cluster: scripts/cluster/m11_eval.sh)
    python -m scripts.method11 eval --run-dir experiments/method11/runs/bisenetv2_B_c0.25 --chunk 0/4
    # the frozen public weights through the same path (the envelope's frozen curves)
    python -m scripts.method11 eval --public --models bisenetv2 --kind multi --chunk 0/4

Scoring against the envelope (gate 1) is a separate step, written once the runs exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device  # noqa: E402
from scripts.method2 import _floats, _strs, model_config  # noqa: E402
from scripts.method3 import _dataset_and_clips, _tag  # noqa: E402
from scripts.method4 import _append, resume  # noqa: E402

ensure_src_on_path()

EXP = REPO_ROOT / "experiments" / "method11"
RESULTS = REPO_ROOT / "results" / "method11"
BOUNDARY_PRIOR = RESULTS / "prior_boundary.pt"
ERROR_PRIOR = REPO_ROOT / "results" / "method5" / "phase2" / "prior_scored_{model}.pt"
MODELS = ("pidnet_s", "bisenetv2")


# --------------------------------------------------------------------------- #
# boundary-prior
# --------------------------------------------------------------------------- #
def cmd_boundary_prior(args: argparse.Namespace) -> int:
    """B-bar: boundary pixels per 16-px cell, summed over the train labels, as prior records
    (one per model and scale, all the same map) that :func:`warp.policy.load_prior` reads."""
    import torch

    from finetune.cityscapes import CityscapesFine
    from finetune.warped import boundary_counts
    from warp.policy import prior_record

    ds = CityscapesFine(args.cityscapes, split="train")
    total: Optional[torch.Tensor] = None
    t0 = time.time()
    for i in range(len(ds)):
        _, label = ds[i]
        c = boundary_counts(torch.from_numpy(label), cell=args.cell)
        total = c if total is None else total + c
        if i % 500 == 0:
            print(f"[method11 boundary-prior] {i}/{len(ds)} ({time.time() - t0:.0f} s)", flush=True)
    # sum of per-frame maps, each count / cell^2: what load_prior divides by frames
    per_frame_sum = total.to(torch.float64) / float(args.cell * args.cell)
    scales = sorted(set(_floats(args.scales)))
    records = [prior_record(m, s, per_frame_sum, len(ds), source="boundary prior, Cityscapes train fine")
               for m in MODELS for s in scales]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out)
    share = float(per_frame_sum.sum() / len(ds) / per_frame_sum.numel())
    meta = {"frames": len(ds), "cell": args.cell, "grid": list(per_frame_sum.shape), "scales": scales,
            "mean_boundary_share": share, "source": "Cityscapes train fine labels (gtFine/train)"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[method11 boundary-prior] {len(ds)} frames, mean boundary share {share:.4f} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _backbone(model: str, fixture: bool, weights: Optional[str] = None):
    from models.registry import build_backbone

    # the fixture: the real architecture, randomly initialised (a dummy has nothing to train)
    spec = {"type": model, "allow_random_init": True} if fixture else dict(model_config(model, False)["backbone"])
    if weights is not None:
        spec.update(weights=str(weights), sha256=None, strict=True)
    return build_backbone(spec)


def _prior_for(kind: str, model: str, scale: float, prior_file: Optional[str]):
    import torch  # noqa: F401

    from warp.policy import load_prior

    if kind in ("uniform", "multi"):
        return None
    path = Path(prior_file) if prior_file else (BOUNDARY_PRIOR if kind == "boundary"
                                                else Path(str(ERROR_PRIOR).format(model=model)))
    if not path.exists():
        raise SystemExit(f"no prior file {path} (boundary: run boundary-prior first)")
    return load_prior(path, model, scale)


def cmd_train(args: argparse.Namespace) -> int:
    from finetune.cityscapes import CityscapesFine
    from finetune.train import DENSE_SCALES, RunConfig, run_training

    cfg = RunConfig(model=args.model, kind=args.kind, scale=args.scale,
                    scales=tuple(_floats(args.scales)) if args.scales else DENSE_SCALES,
                    alpha=args.alpha, sigma=args.sigma, d_max=args.d_max, epochs=args.epochs, batch=args.batch,
                    accum=args.accum, lr=args.lr, seed=args.seed, max_steps=args.max_steps,
                    amp=not args.no_amp, jitter=args.jitter)
    if args.fixture:
        dataset = _FixtureTrain(args.fixture_size)
    else:
        dataset = CityscapesFine(args.cityscapes, split="train")
    model = _backbone(args.model, args.fixture)
    prior = _prior_for(args.kind, args.model, args.scale, args.prior_file)
    run_dir = Path(args.run_dir) if args.run_dir else EXP / "runs" / cfg.name().replace("/", "_")
    print(f"[method11 train] {cfg.name()}: {len(dataset)} samples, run dir {run_dir}", flush=True)
    out = run_training(cfg, model, dataset, run_dir, resolve_device(args.device), prior=prior,
                       workers=args.workers)
    print(f"[method11 train] {cfg.name()}: {out}")
    return 0


class _FixtureTrain:
    """Smoke tests: random uint8 frames whose label is the brightest channel of 8-px blocks."""

    def __init__(self, size):
        import numpy as np

        h, w = size
        rng = np.random.default_rng(0)
        self.items = []
        for _ in range(8):
            lab = np.kron(rng.integers(0, 19, size=(h // 8, w // 8)), np.ones((8, 8), dtype=np.int64))
            frame = np.stack([(lab * 13) % 256, (lab * 7) % 256, (lab * 29) % 256]).astype(np.uint8)
            self.items.append((frame, lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def eval_configs(kind: str, scale: float, scales, alpha: float, sigma: float, d_max: Optional[float]):
    from finetune.train import RunConfig

    return RunConfig(model="x", kind=kind, scale=scale, scales=scales, alpha=alpha, sigma=sigma,
                     d_max=d_max).step_configs()


def cmd_eval(args: argparse.Namespace) -> int:
    """Per clip of ``--split``, one confusion per labelled position for each configuration the
    run was trained for (``multi``: all 13 scales), appended to ``eval_<split>_<run>_<tag>.jsonl.gz``
    (resumable). ``--public``: the frozen public weights through the same path."""
    import torch

    from finetune.train import DENSE_SCALES
    from warp.stream import build_steps, stream_clip

    if args.split == "val" and not (args.allow_val or args.fixture):
        raise SystemExit("val is read once, after gate 1: pass --allow-val only for the finalists")
    device = resolve_device(args.device)
    if args.public:
        if not args.kind:
            raise SystemExit("--public needs --kind (uniform, multi, boundary, error) and its scale(s)")
        runs = []
        for m in _strs(args.models):
            runs.append({"model": m, "kind": args.kind, "scale": args.scale, "scales": tuple(_floats(args.scales))
                         if args.scales else DENSE_SCALES, "alpha": args.alpha, "sigma": args.sigma,
                         "d_max": args.d_max, "weights": None,
                         "name": f"public/{m}/{args.kind}" + ("" if args.kind == "multi" else f"/c{args.scale:g}")})
    else:
        if not args.run_dir:
            raise SystemExit("eval needs --run-dir (a finished run) or --public")
        rd = Path(args.run_dir)
        meta = json.loads((rd / "run.json").read_text())
        if not (rd / "weights.pt").exists():
            raise SystemExit(f"{rd}: the run has not finished (no weights.pt)")
        c = meta["config"]
        runs = [{"model": c["model"], "kind": c["kind"], "scale": c["scale"], "scales": tuple(c["scales"]),
                 "alpha": c["alpha"], "sigma": c["sigma"], "d_max": c["d_max"], "weights": str(rd / "weights.pt"),
                 "name": meta["name"]}]
        args.models = c["model"]
    dataset, clips = _dataset_and_clips(args)
    out_dir = Path(args.out_dir) if args.out_dir else EXP / "eval" / args.split
    for run in runs:
        model = run["model"]
        cfgs = eval_configs(run["kind"], run["scale"], run["scales"], run["alpha"], run["sigma"], run["d_max"])
        prior = _prior_for(run["kind"], model, run["scale"], args.prior_file)
        priors = {} if prior is None else {float(run["scale"]): prior}
        path = out_dir / f"eval_{args.split}_{run['name'].replace('/', '_')}_{_tag(args)}.jsonl.gz"
        done = resume(path, {c.name() for c in cfgs}, label="method11 eval")
        todo = [i for i in clips if dataset.clip_id(i) not in done]
        print(f"[method11 eval] {run['name']}: {len(cfgs)} configs, {len(todo)} clips to run", flush=True)
        if not todo:
            continue
        backbone = _backbone(model, args.fixture, run["weights"]).to(device).eval()
        steps = build_steps(backbone, cfgs, {s: p.to(torch.float32) for s, p in priors.items()}, cell=args.cell)
        t0 = time.time()
        for n, i in enumerate(todo):
            clip_id = dataset.clip_id(i)
            positions, out = stream_clip(dataset, i, backbone, steps, device, batch=args.batch,
                                         allow_any_frame_size=args.fixture)
            base = {"split": args.split, "model": model, "run": run["name"], "kind": run["kind"],
                    "weights": "public" if run["weights"] is None else "finetuned", "clip_id": clip_id,
                    "frames": len(positions), "positions": positions, "cell": args.cell}
            _append(path, [{**base, "name": c.name(), "config": asdict(c), "conf": out[c.name()].tolist()}
                           for c in cfgs])
            dt = time.time() - t0
            print(f"[method11 eval] {run['name']} clip {n + 1}/{len(todo)} {clip_id} ({dt / (n + 1):.1f} s/clip)",
                  flush=True)
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("boundary-prior", help="B-bar from the Cityscapes train labels.")
    bp.add_argument("--cityscapes", required=True, help="Folder with leftImg8bit/train and gtFine/train.")
    bp.add_argument("--cell", type=int, default=16)
    bp.add_argument("--scales", default="0.25,0.375,0.5,0.625,0.75,0.875,1")
    bp.add_argument("--out", default=str(BOUNDARY_PRIOR))
    bp.set_defaults(func=cmd_boundary_prior)

    def warp_args(sp):
        sp.add_argument("--kind", default=None, choices=("uniform", "multi", "boundary", "error"))
        sp.add_argument("--scale", type=float, default=1.0)
        sp.add_argument("--scales", default=None, help="multi: comma-separated (default 0.25 to 1 by 1/16).")
        sp.add_argument("--alpha", type=float, default=0.25)
        sp.add_argument("--sigma", type=float, default=4.0)
        sp.add_argument("--d-max", type=float, default=None, help="Upper density clip (2 at scale 1: magnify).")
        sp.add_argument("--prior-file", default=None, help="Default: results/method11/prior_boundary.pt (boundary) "
                        "or results/method5/phase2/prior_scored_<model>.pt (error).")
        sp.add_argument("--fixture", action="store_true", help="Synthetic data + random-init model (smoke test).")
        sp.add_argument("--device", default="auto")

    tr = sub.add_parser("train", help="One fine-tuning run.")
    tr.add_argument("--model", required=True, choices=MODELS)
    warp_args(tr)
    tr.add_argument("--cityscapes", default=None, help="Folder with leftImg8bit/train and gtFine/train.")
    tr.add_argument("--run-dir", default=None)
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--accum", type=int, default=1)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--jitter", type=float, default=0.25)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--max-steps", type=int, default=None, help="Stop early (gate 0 checks).")
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--no-amp", action="store_true")
    tr.add_argument("--fixture-size", type=int, nargs=2, default=(32, 64))
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Confusions of a run (or of the public weights) on VPS clips.")
    warp_args(ev)
    ev.add_argument("--run-dir", default=None)
    ev.add_argument("--public", action="store_true")
    ev.add_argument("--models", default="pidnet_s,bisenetv2")
    ev.add_argument("--split", default="train_all")
    ev.add_argument("--clip-ids", default=None)
    ev.add_argument("--chunk", default=None, help="i/N: the i-th contiguous block of the sorted clip ids.")
    ev.add_argument("--tag", default=None)
    ev.add_argument("--out-dir", default=None)
    ev.add_argument("--cell", type=int, default=16)
    ev.add_argument("--batch", type=int, default=4)
    ev.add_argument("--allow-val", action="store_true")
    ev.set_defaults(func=cmd_eval)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
