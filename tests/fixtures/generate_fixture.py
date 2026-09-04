"""Generate the tiny synthetic Cityscapes-VPS fixture used by the unit tests.

This writes a handful of small PNGs that mimic the directory layout and naming
documented at the top of ``src/data/cityscapes_vps.py`` — a *toy* stand-in so
the loader can be tested with no real (license-gated) dataset present.

The generated PNGs are tiny and fully deterministic, so they are NOT committed
to the repo (see .gitignore); instead the test suite regenerates them on demand
via tests/conftest.py -> ``ensure_fixture()``. Run this directly only if you
want the files on disk to inspect them, or to (re)generate them by hand:

    python tests/fixtures/generate_fixture.py

Toy layout produced (root = tests/fixtures/cityscapes_vps):

    val/
        img_all/                 clip 0000: frames 0..5   (6 frames)
                                 clip 0001: frames 0..2   (3 frames)
        panoptic_video/          panoptic PNGs for labeled frames only
                                 (frame_index % 5 == 0)  -> 0000:{0,5}, 0001:{0}
        labelmap/                single-channel semantic trainId PNGs, same frames

Real clips have 30 frames labeled at {0,5,10,15,20,25}; this fixture is
compressed to keep the committed files tiny while still exercising sparse
labeling (labeled vs unlabeled) and multi-clip discovery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FIXTURE_ROOT = Path(__file__).resolve().parent / "cityscapes_vps"
SPLIT = "val"
CITY = "frankfurt"
SEQ = 0  # original Cityscapes sequence id
HEIGHT, WIDTH = 8, 16
ANNOTATED_EVERY = 5

# clip id -> number of frames in that clip
CLIPS = {0: 6, 1: 3}


def _frame_name(clip: int, frame: int, suffix: str, ext: str = ".png") -> str:
    # e.g. 0000_0003_frankfurt_000000_000003_leftImg8bit.png
    return f"{clip:04d}_{frame:04d}_{CITY}_{SEQ:06d}_{frame:06d}{suffix}{ext}"


def _make_frame_rgb(clip: int, frame: int) -> np.ndarray:
    """A small, deterministic, visually-distinct RGB frame."""
    rng = np.random.default_rng(clip * 100 + frame)
    img = rng.integers(0, 60, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    # A bright moving square so successive frames look different in the overlay.
    x = (frame * 2) % (WIDTH - 3)
    img[2:5, x : x + 3] = np.array([220, 40 + 30 * clip, 40], dtype=np.uint8)
    return img


def _make_panoptic_rgb() -> np.ndarray:
    """COCO-panoptic-encoded RGB: id = R + 256*G + 256*256*B. Two segments."""
    ann = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    ann[:, : WIDTH // 2, 0] = 1  # left half  -> id 1
    ann[:, WIDTH // 2 :, 0] = 2  # right half -> id 2
    return ann


def _make_semantic_labelmap() -> np.ndarray:
    """Single-channel semantic trainId map (toy: road=0 / building=2)."""
    lbl = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    lbl[HEIGHT // 2 :, :] = 2
    return lbl


def generate(root: Path = FIXTURE_ROOT) -> Path:
    """(Re)write the fixture under ``root`` and return the split directory."""
    img_dir = root / SPLIT / "img_all"
    pan_dir = root / SPLIT / "panoptic_video"
    lbl_dir = root / SPLIT / "labelmap"
    for d in (img_dir, pan_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_img = n_ann = 0
    for clip, n_frames in CLIPS.items():
        for frame in range(n_frames):
            Image.fromarray(_make_frame_rgb(clip, frame), mode="RGB").save(
                img_dir / _frame_name(clip, frame, "_leftImg8bit")
            )
            n_img += 1
            if frame % ANNOTATED_EVERY == 0:  # labeled frame
                Image.fromarray(_make_panoptic_rgb(), mode="RGB").save(
                    pan_dir / _frame_name(clip, frame, "_gtFine_panoptic")
                )
                Image.fromarray(_make_semantic_labelmap(), mode="L").save(
                    lbl_dir / _frame_name(clip, frame, "_gtFine_labelTrainIds")
                )
                n_ann += 1

    return root / SPLIT


def ensure_fixture(root: Path = FIXTURE_ROOT) -> Path:
    """Generate the fixture only if it is missing; return the dataset root.

    Idempotent and safe to call from conftest / scripts on every run.
    """
    if not (root / SPLIT / "img_all").is_dir():
        generate(root)
    return root


def main() -> None:
    generate()
    n_files = sum(1 for _ in FIXTURE_ROOT.rglob("*.png"))
    print(f"Wrote {n_files} PNG files under {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
