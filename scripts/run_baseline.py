#!/usr/bin/env python3
"""Full-frame baseline (method 1): the backbone on whole frames at several input scales.

    # Real data (needs $CITYSCAPES_VPS_ROOT and the PIDNet-S Cityscapes weights):
    python -m scripts.run_baseline --config configs/baseline_fullframe_pidnet_s.yaml \\
        --set backbone.weights=/path/to/PIDNet_S_Cityscapes_val.pt

    # Smoke test on the synthetic fixture with the dummy backbone (no data, CPU):
    python -m scripts.run_baseline --config configs/baseline_fixture.yaml

Each scale is one run and writes one self-describing JSON (resolved config,
git commit, seed, dataset fingerprint, metrics, cost) to
``<output_dir>/<YYYY-MM-DD>_<run.name>/<run.name>_s<scale>.json``, next to
the resolved ``config.yaml``. Reduced scales (< 1) are the equal-cost
full-frame baselines for the foveated methods.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # `python scripts/run_baseline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import (  # noqa: E402
    REPO_ROOT,
    apply_overrides,
    ensure_fixture,
    ensure_src_on_path,
    expand_env,
    load_yaml,
    resolve_device,
    seed_everything,
)

ensure_src_on_path()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Baseline YAML (see configs/baseline_*.yaml).")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set backbone.weights=/w.pt --set eval.scales=[1.0,0.5]")
    p.add_argument("--fixture", action="store_true",
                   help="Run on the synthetic official-layout fixture (generated on demand) instead of real data.")
    p.add_argument("--out", default=None, help="Output directory (default: run.output_dir of the config).")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda[:N] (default: auto).")
    p.add_argument("--max-clips", type=int, default=None, help="Evaluate only the first N clips (quick checks).")
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = expand_env(apply_overrides(load_yaml(args.config), args.set))
    for key in ("dataset", "backbone"):
        if not isinstance(cfg.get(key), dict):
            raise SystemExit(f"config {args.config} needs a '{key}' section")
    if args.fixture or cfg.get("fixture"):
        cfg["fixture"] = True
        cfg["dataset"]["root"] = str(ensure_fixture())
    if args.max_clips is not None:
        cfg.setdefault("eval", {})["max_clips"] = args.max_clips
    return cfg


def run(cfg: Dict[str, Any], device: str, out_dir: Optional[Path] = None) -> List[Path]:
    """Evaluate every configured scale; return the written JSON paths."""
    from data.registry import build_dataset
    from eval.cost import skipped
    from eval.harness import FullFrameMethod, evaluate, measure_latency
    from eval.results import make_run_record, write_json
    from models.registry import build_backbone, describe_backbone
    from models.weights import WeightsError

    run_cfg = cfg.get("run", {}) or {}
    name = str(run_cfg.get("name", "fullframe"))
    seed = int(run_cfg.get("seed", 0))
    seed_everything(seed)

    dataset = build_dataset(cfg["dataset"], base_dir=REPO_ROOT)
    if len(dataset) == 0 or dataset.num_labeled_frames() == 0:
        raise SystemExit(
            f"dataset has {len(dataset)} clips and {dataset.num_labeled_frames()} labelled frames: "
            "check the paths/config with scripts/diagnose_real_data.py"
        )
    try:
        backbone = build_backbone(cfg["backbone"], context={"palette": dataset.palette}).to(device)
    except (WeightsError, ValueError) as exc:
        raise SystemExit(f"backbone: {exc}") from exc
    backbone_info = describe_backbone(backbone)

    eval_cfg = cfg.get("eval", {}) or {}
    scales = [float(s) for s in eval_cfg.get("scales", [1.0])]
    precision = str(eval_cfg.get("precision", "fp32"))
    frames = str(eval_cfg.get("frames", "labeled"))
    max_clips = eval_cfg.get("max_clips")
    clip_indices = list(range(len(dataset) if max_clips is None else min(int(max_clips), len(dataset))))
    cost_cfg = cfg.get("cost", {}) or {}
    lat_cfg = cost_cfg.get("latency", {}) or {}

    base = Path(out_dir) if out_dir else REPO_ROOT / str(run_cfg.get("output_dir", "experiments"))
    run_dir = base / f"{_dt.date.today().isoformat()}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    dataset_info = dict(dataset.describe())
    dataset_info.update(
        {"split": cfg["dataset"].get("split"), "synthetic_fixture": bool(cfg.get("fixture")),
         "clips_evaluated": len(clip_indices)}
    )
    print(f"[run_baseline] {dataset_info['name']} split={dataset_info['split']}: {len(clip_indices)} clips, "
          f"backbone={backbone_info.get('type')}, device={device}")
    if cfg.get("fixture"):
        print("[run_baseline] SYNTHETIC FIXTURE: numbers are a pipeline smoke test, not results.")

    written = []
    for scale in scales:
        method = FullFrameMethod(backbone, scale=scale, precision=precision)
        result = evaluate(dataset, method, device=device, clip_indices=clip_indices, frames=frames)
        cost = dict(result["cost"])
        frame_size = result["frame_sizes"][0]  # most frequent frame size (latency is measured there)
        cost["frame_size"] = list(frame_size)
        cost["input_size"] = list(method.input_size(*frame_size))
        if lat_cfg.get("enabled", True):
            lat_method = FullFrameMethod(backbone, scale=scale, precision=str(lat_cfg.get("precision", "fp16")))
            latency, memory = measure_latency(
                lat_method, frame_size, device,
                warmup=int(lat_cfg.get("warmup", 20)), iters=int(lat_cfg.get("iters", 100)),
                allow_cpu=bool(lat_cfg.get("allow_cpu", False)),
            )
        else:
            latency = memory = skipped("disabled in config")
        cost["latency"] = latency
        cost["memory"] = memory if cost_cfg.get("memory", True) else skipped("disabled in config")

        run_name = f"{name}_s{scale:.3f}"  # == the JSON file name
        method_info = dict(method.describe(), frames=frames, frame_sizes=[list(s) for s in result["frame_sizes"]])
        record = make_run_record(
            run_name=run_name, method=method_info, config=cfg, seed=seed, dataset=dataset_info,
            backbone=backbone_info, metrics=result["metrics"], cost=cost, notes=cfg.get("notes"),
        )
        path = write_json(run_dir / f"{run_name}.json", record)
        written.append(path)
        if latency.get("status") == "ok":
            lat_text = f"{latency['end_to_end']['median_ms']:.2f} ms median end-to-end"
        else:
            lat_text = f"latency skipped ({latency.get('reason')})"
        print(
            f"  scale {scale:<5g} input {cost['input_size'][0]}x{cost['input_size'][1]}  "
            f"mIoU {result['metrics']['miou']:.4f}  {cost['total']['gflops_per_frame']:.4g} GFLOPs/frame  "
            f"pixel fraction {cost['pixel_fraction']:.3f}  {lat_text}  -> {path.name}"
        )
    return written


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(args)
    device = resolve_device(args.device)
    paths = run(cfg, device=device, out_dir=Path(args.out) if args.out else None)
    print(f"[run_baseline] wrote {len(paths)} result file(s) to {paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
