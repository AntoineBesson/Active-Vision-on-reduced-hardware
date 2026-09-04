#!/usr/bin/env python3
"""Visual sanity check for the Cityscapes-VPS loader.

Point this at a local copy of the dataset; it loads ONE clip and writes a PNG
grid of a few frames with their annotation overlaid (where labeled) to
``experiments/data_check/``. Unlabeled frames are shown as-is and marked.

Examples
--------
    # Use the real dataset via env var / CLI:
    export CITYSCAPES_VPS_ROOT=/data/cityscapes_vps
    python scripts/verify_data.py --split val --clip-index 0

    # Smoke-test the pipeline with the bundled synthetic fixture (no real data):
    python scripts/verify_data.py --fixture

If neither --root nor $CITYSCAPES_VPS_ROOT is set, the script falls back to the
tiny test fixture and says so, so it always runs cleanly end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make ./src importable when running from a source checkout without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.cityscapes_vps import (  # noqa: E402  (after sys.path tweak)
    ROOT_ENV_VAR,
    CityscapesVPS,
    CityscapesVPSConfig,
)

FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "cityscapes_vps"
DEFAULT_OUT_DIR = _REPO_ROOT / "experiments" / "data_check"


def _ensure_fixture() -> Path:
    """Generate the synthetic fixture on demand (it is not committed)."""
    sys.path.insert(0, str(_REPO_ROOT / "tests" / "fixtures"))
    import generate_fixture

    return generate_fixture.ensure_fixture()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help=f"Dataset root (default: ${ROOT_ENV_VAR}).")
    p.add_argument("--config", default=None, help="Optional YAML config (see configs/cityscapes_vps.yaml).")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--annotation-subdir", default=None, help="Override the annotation subdir (e.g. labelmap).")
    p.add_argument("--clip-index", type=int, default=0, help="Which clip to visualize.")
    p.add_argument("--num-frames", type=int, default=6, help="Max frames to show in the grid.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Where to save the grid PNG.")
    p.add_argument("--alpha", type=float, default=0.5, help="Annotation overlay opacity.")
    p.add_argument(
        "--fixture",
        action="store_true",
        help="Force use of the bundled synthetic fixture (ignores --root / env var).",
    )
    return p.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> tuple[CityscapesVPS, bool]:
    """Return (dataset, used_fixture)."""
    import os

    if args.config:
        cfg = CityscapesVPSConfig.from_yaml(args.config, split=args.split)
    else:
        cfg = CityscapesVPSConfig(split=args.split)
    if args.annotation_subdir:
        cfg = cfg.with_overrides(annotation_subdir=args.annotation_subdir)

    used_fixture = False
    if args.fixture:
        cfg = cfg.with_overrides(root=str(_ensure_fixture()))
        used_fixture = True
    elif args.root:
        cfg = cfg.with_overrides(root=args.root)
    elif not os.environ.get(ROOT_ENV_VAR):
        print(
            f"[verify_data] No --root and no ${ROOT_ENV_VAR}; falling back to the "
            f"synthetic fixture at {FIXTURE_ROOT} (smoke-test mode).",
            file=sys.stderr,
        )
        cfg = cfg.with_overrides(root=str(_ensure_fixture()))
        used_fixture = True

    return CityscapesVPS(cfg), used_fixture


def colorize_labels(label_map: np.ndarray) -> np.ndarray:
    """Map an integer (H,W) label/id map to a deterministic (H,W,3) uint8 RGB."""
    import matplotlib

    ids = np.unique(label_map)
    cmap = matplotlib.colormaps["tab20"]
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for i, seg_id in enumerate(ids):
        if seg_id == 0:
            continue  # treat 0 as background / void -> leave transparent-ish
        color = np.array(cmap(i % cmap.N)[:3]) * 255
        out[label_map == seg_id] = color.astype(np.uint8)
    return out


def overlay(frame_rgb: np.ndarray, label_map: np.ndarray, alpha: float) -> np.ndarray:
    color = colorize_labels(label_map)
    mask = (label_map != 0)[..., None]
    blended = frame_rgb.astype(np.float32).copy()
    blended = np.where(
        mask,
        (1 - alpha) * frame_rgb + alpha * color,
        blended,
    )
    return blended.clip(0, 255).astype(np.uint8)


def select_frame_orders(num_frames_in_clip: int, want: int) -> list[int]:
    """Pick up to `want` frame orders, evenly spaced across the clip."""
    want = max(1, min(want, num_frames_in_clip))
    if want == num_frames_in_clip:
        return list(range(num_frames_in_clip))
    return list(np.linspace(0, num_frames_in_clip - 1, want).round().astype(int))


def main(argv=None) -> int:
    args = parse_args(argv)

    # Headless-safe: pick the Agg backend before importing pyplot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset, used_fixture = build_dataset(args)
    if len(dataset) == 0:
        print("[verify_data] No clips found — check your paths/config.", file=sys.stderr)
        return 1

    idx = max(0, min(args.clip_index, len(dataset) - 1))
    clip = dataset.load_clip(idx)
    images = clip["images"]
    if isinstance(images, np.ndarray):
        images = list(images)

    orders = select_frame_orders(clip["num_frames"], args.num_frames)
    ncols = min(len(orders), 6)
    nrows = int(np.ceil(len(orders) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")
    for i, order in enumerate(orders):
        ax = axes.flat[i]
        frame = images[order]
        ann = clip["annotations"][order]
        labeled = bool(clip["labeled_mask"][order])
        img = overlay(frame, ann, args.alpha) if (labeled and ann is not None) else frame
        ax.imshow(img)
        tag = "labeled" if labeled else "unlabeled"
        ax.set_title(f"f{int(clip['frame_indices'][order])} ({tag})", fontsize=9)

    src = "synthetic fixture" if used_fixture else str(dataset.config.resolve_root())
    fig.suptitle(
        f"Cityscapes-VPS {clip['split']} clip {clip['clip_id']} — "
        f"{int(clip['labeled_mask'].sum())}/{clip['num_frames']} labeled\nsource: {src}",
        fontsize=10,
    )
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clip_{clip['split']}_{clip['clip_id']}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[verify_data] Saved sanity-check grid to {out_path}")
    if used_fixture:
        print("[verify_data] (This was smoke-test mode on the synthetic fixture.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
