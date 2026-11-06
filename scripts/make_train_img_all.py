"""Build <root>/train/img_all (every frame of each VPS train clip) and chunked zips for Colab.

The official Cityscapes-VPS preparation keeps only the 6 labelled frames of each
train clip (``train/img``). Method 3 runs on every frame, so the in-between
frames are copied from the raw Cityscapes ``leftImg8bit_sequence/val`` download
(every VPS clip is a Cityscapes val sequence), with the same names as
``val/img_all``. The dataset split ``train_all`` (configs/cityscapes_vps.yaml)
then reads them with the train labels.

    python scripts/make_train_img_all.py --root C:/Users/trash/cityscapes_vps --raw C:/Users/trash/cityscapes \\
        --zip-chunks 4 --zip-out C:/tmp/m3zips [--only-chunk 0]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ANNOTATED_EVERY = 5
KEYFRAME_OFFSET = 4  # clip position of the first labelled frame
IMG_RE = re.compile(r"^(?P<vid>\d{4})_\d{4}_(?P<city>[^_]+)_(?P<seq>\d{6})_(?P<frame>\d{6})_(?:newImg8bit|leftImg8bit)\.png$")

Clip = Tuple[str, int, List[int]]  # city, sequence, frame numbers of every frame


def frame_name(city: str, seq: int, frame: int) -> str:
    return f"{city}_{seq:06d}_{frame:06d}_leftImg8bit.png"


def train_clips(root: Path) -> Dict[str, Clip]:
    """Clip id -> (city, sequence, every frame number), from the labelled frames in ``train/img``."""
    labelled: Dict[str, List[Tuple[str, int, int]]] = {}
    for p in sorted((Path(root) / "train" / "img").glob("*.png")):
        m = IMG_RE.match(p.name)
        if m:
            labelled.setdefault(m["vid"], []).append((m["city"], int(m["seq"]), int(m["frame"])))
    clips: Dict[str, Clip] = {}
    for vid, frames in sorted(labelled.items()):
        if len({(c, s) for c, s, _ in frames}) != 1:
            raise ValueError(f"clip {vid}: labelled frames from several sequences")
        nums = sorted(f for _, _, f in frames)
        if any(b - a != ANNOTATED_EVERY for a, b in zip(nums, nums[1:])):
            raise ValueError(f"clip {vid}: labelled frames {nums} are not {ANNOTATED_EVERY} apart")
        city, seq = frames[0][0], frames[0][1]
        clips[vid] = (city, seq, list(range(nums[0] - KEYFRAME_OFFSET, nums[-1] + 1)))
    return clips


def build_img_all(root: Path, raw: Path) -> List[str]:
    """Hard-link (copy where linking fails, e.g. across drives) every frame of every train
    clip into ``<root>/train/img_all``; returns the clip ids.

    Every raw frame is checked before anything is written: a missing one raises,
    naming the clip and the frame."""
    root, raw = Path(root), Path(raw)
    clips = train_clips(root)
    src_dir = raw / "leftImg8bit_sequence" / "val"
    for vid, (city, seq, nums) in clips.items():
        for n in nums:
            if not (src_dir / city / frame_name(city, seq, n)).is_file():
                raise ValueError(f"clip {vid}: raw frame {frame_name(city, seq, n)} is missing from {src_dir / city}")
    out = root / "train" / "img_all"
    out.mkdir(parents=True, exist_ok=True)
    for vid, (city, seq, nums) in clips.items():
        for n in nums:
            src, dst = src_dir / city / frame_name(city, seq, n), out / frame_name(city, seq, n)
            if dst.is_file() and (os.path.samefile(src, dst) or dst.stat().st_size == src.stat().st_size):
                continue
            if dst.is_file():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    return list(clips)


def zip_chunks(root: Path, out: Path, n_chunks: int, only: Optional[Sequence[int]] = None) -> List[Path]:
    """``vps_train_all_chunk{i}.zip``: the i-th contiguous block of sorted clip ids, with their
    ``train/img_all`` frames, their ``train/panoptic_inst`` labels and the root train JSONs.
    ``only``: write just these chunks (a chunk is about 7 GB; disk space may allow one at a time)."""
    root, out = Path(root), Path(out)
    clips = train_clips(root)
    ids = sorted(clips)
    size = math.ceil(len(ids) / n_chunks)
    out.mkdir(parents=True, exist_ok=True)
    jsons = sorted(root.glob("*_train_*.json"))  # train_all reads none of the val/test JSONs
    paths = []
    for i in range(n_chunks):
        if only is not None and i not in only:
            continue
        block = ids[i * size:(i + 1) * size]
        path = out / f"vps_train_all_chunk{i}.zip"
        tmp = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
            for vid in block:
                city, seq, nums = clips[vid]
                for n in nums:
                    z.write(root / "train" / "img_all" / frame_name(city, seq, n), f"train/img_all/{frame_name(city, seq, n)}")
                for p in sorted((root / "train" / "panoptic_inst").glob(f"{vid}_*.png")):
                    z.write(p, f"train/panoptic_inst/{p.name}")
            for p in jsons:
                z.write(p, p.name)
        tmp.replace(path)
        paths.append(path)
    return paths


def main(argv: Sequence[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Cityscapes-VPS root (holds train/img, train/panoptic_inst).")
    p.add_argument("--raw", required=True, help="Raw Cityscapes root (holds leftImg8bit_sequence/val).")
    p.add_argument("--zip-chunks", type=int, default=0, help="Also write this many chunk zips (0: none).")
    p.add_argument("--zip-out", default=None, help="Folder for the chunk zips.")
    p.add_argument("--only-chunk", type=int, action="append", default=None,
                   help="Write only this chunk (repeatable); default: all.")
    args = p.parse_args(argv)
    ids = build_img_all(Path(args.root), Path(args.raw))
    frames = sum(len(c[2]) for c in train_clips(Path(args.root)).values())
    print(f"[train img_all] {len(ids)} clips, {frames} frames in {Path(args.root) / 'train' / 'img_all'}")
    if args.zip_chunks:
        if not args.zip_out:
            p.error("--zip-chunks needs --zip-out")
        for z in zip_chunks(Path(args.root), Path(args.zip_out), args.zip_chunks, only=args.only_chunk):
            print(f"[train img_all] {z} ({z.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
