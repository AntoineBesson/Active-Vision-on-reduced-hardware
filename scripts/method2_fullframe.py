#!/usr/bin/env python3
"""Per-clip confusion matrices of the full-frame baseline at every scale (method 2's reference).

    python -m scripts.method2_fullframe --config configs/baseline_fullframe_pidnet_s.yaml --split train
    python -m scripts.method2_fullframe --config configs/baseline_fixture.yaml --split val   # smoke test

Same model, resize and scoring as ``scripts.run_baseline``, but the confusion
is kept per clip, so the equal-latency comparison of method 2 can be
bootstrapped over clips. Each frame is loaded once and run at every scale.
Writes ``<out>/fullframe_<backbone.type>_<split>.npz``: ``scales`` (S,),
``confusion`` (S, N_clips, C, C) int64, ``clip_ids`` (N_clips,) and ``meta``
(JSON: config, git commit, dataset description). No latency here: the
latencies of method 2 come from the T4 tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):  # `python scripts/method2_fullframe.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, ensure_src_on_path, resolve_device, seed_everything  # noqa: E402
from scripts.run_baseline import resolve_config  # noqa: E402

ensure_src_on_path()

FRAME = (1024, 2048)  # the frame the T4 latency tables were measured at


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Baseline YAML (configs/baseline_*.yaml).")
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--fixture", action="store_true", help="Run on the synthetic fixture.")
    p.add_argument("--out", default=str(REPO_ROOT / "experiments" / "method2" / "fullframe"))
    p.add_argument("--device", default="auto")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--require-1024x2048", action="store_true",
                   help="Refuse other frame sizes even on the fixture (always on for real data).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(args)
    cfg["dataset"]["split"] = args.split
    device = resolve_device(args.device)

    import numpy as np
    import torch

    from data.registry import build_dataset
    from eval.harness import FullFrameMethod, frame_to_tensor
    from eval.metrics import ConfusionMatrix
    from eval.results import git_info
    from models.registry import build_backbone

    seed_everything(int((cfg.get("run") or {}).get("seed", 0)))
    dataset = build_dataset(cfg["dataset"], base_dir=REPO_ROOT)
    if len(dataset) == 0 or dataset.num_labeled_frames() == 0:
        raise SystemExit(f"split {args.split}: {len(dataset)} clips, {dataset.num_labeled_frames()} labelled frames")
    backbone = build_backbone(cfg["backbone"], context={"palette": dataset.palette}).to(device)
    eval_cfg = cfg.get("eval") or {}
    scales = [float(s) for s in eval_cfg.get("scales", [1.0])]
    methods = [FullFrameMethod(backbone, scale=s, precision=str(eval_cfg.get("precision", "fp32"))) for s in scales]
    n = len(dataset) if args.max_clips is None else min(args.max_clips, len(dataset))
    check_size = args.require_1024x2048 or not cfg.get("fixture")

    conf = np.zeros((len(scales), n, dataset.num_classes, dataset.num_classes), dtype=np.int64)
    for ci in range(n):
        cms = [ConfusionMatrix(dataset.num_classes, dataset.ignore_index) for _ in scales]
        for t in np.flatnonzero(np.asarray(dataset.labeled_mask(ci), dtype=bool)):
            image = dataset.load_frame(ci, int(t))
            if check_size and tuple(image.shape[:2]) != FRAME:
                raise SystemExit(f"clip {dataset.clip_id(ci)} frame {t} is {image.shape[0]}x{image.shape[1]}: "
                                 f"the T4 latency tables are for 1024x2048 frames")
            label = dataset.load_label(ci, int(t))
            frame = frame_to_tensor(image, device)
            with torch.no_grad():
                for cm, method in zip(cms, methods):
                    cm.update(method._segment(frame, method.backbone).cpu().numpy(), label)
        for s, cm in enumerate(cms):
            conf[s, ci] = cm.matrix
        print(f"\r[method2_fullframe] {args.split}: clip {ci + 1}/{n}", end="", flush=True)
    print()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fullframe_{cfg['backbone']['type']}_{args.split}.npz"
    meta = {"split": args.split, "config": cfg, "git": git_info(REPO_ROOT), "dataset": dataset.describe()}
    np.savez(path, scales=np.array(scales), confusion=conf,
             clip_ids=np.array([dataset.clip_id(ci) for ci in range(n)]), meta=json.dumps(meta, default=str))
    for s, scale in enumerate(scales):
        cm = ConfusionMatrix(dataset.num_classes, dataset.ignore_index)
        cm.matrix = conf[s].sum(0)
        print(f"  scale {scale:<5g} mIoU {cm.compute()['miou']:.4f}")
    print(f"[method2_fullframe] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
