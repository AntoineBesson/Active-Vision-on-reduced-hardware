"""Train in-between frames (train/img_all) from the raw Cityscapes sequences, and chunked zips (CPU)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.cityscapes_vps import CityscapesVPS, CityscapesVPSConfig
from scripts.make_train_img_all import build_img_all, zip_chunks

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "cityscapes_vps.yaml"
H, W = 8, 16
CLIPS = [("0001", "frankfurt", 0, 100), ("0002", "frankfurt", 0, 400), ("0003", "lindau", 1, 10),
         ("0004", "munster", 2, 50)]  # (vid, city, sequence, first frame number)


def _tree(tmp_path: Path, clips=CLIPS):
    """A raw leftImg8bit_sequence/val folder (30 frames per clip) and a VPS root with only the labelled frames."""
    raw, root = tmp_path / "raw", tmp_path / "vps"
    for vid, city, seq, first in clips:
        (raw / "leftImg8bit_sequence" / "val" / city).mkdir(parents=True, exist_ok=True)
        for pos in range(30):
            cs = f"{city}_{seq:06d}_{first + pos:06d}"
            rgb = np.full((H, W, 3), pos * 8, dtype=np.uint8)
            Image.fromarray(rgb).save(raw / "leftImg8bit_sequence" / "val" / city / f"{cs}_leftImg8bit.png")
            if pos % 5 == 4:
                fid = pos // 5 + 1
                for sub in ("img", "panoptic_inst"):
                    (root / "train" / sub).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(root / "train" / "img" / f"{vid}_{fid:04d}_{cs}_newImg8bit.png")
                sem = np.zeros((H, W), dtype=np.uint8)  # road; gray ids, divisor 1000 -> train id 0
                sem[:2] = 10  # sky
                Image.fromarray(sem).save(root / "train" / "panoptic_inst" / f"{vid}_{fid:04d}_{cs}_final_mask.png")
    (root / "im_all_info_train_city_vps.json").write_text(json.dumps({"images": []}))
    (root / "panoptic_gt_val_city_vps.json").write_text(json.dumps({"annotations": []}))  # not train: left out of zips
    return raw, root


def test_build_copies_all_30_frames_per_clip(tmp_path):
    raw, root = _tree(tmp_path, CLIPS[:2])
    assert build_img_all(root, raw) == ["0001", "0002"]
    names = sorted(p.name for p in (root / "train" / "img_all").iterdir())
    assert len(names) == 60
    assert names[0] == "frankfurt_000000_000100_leftImg8bit.png"
    assert "frankfurt_000000_000429_leftImg8bit.png" in names


def test_missing_raw_frame_names_the_clip_and_frame(tmp_path):
    raw, root = _tree(tmp_path, CLIPS[:2])
    (raw / "leftImg8bit_sequence" / "val" / "frankfurt" / "frankfurt_000000_000412_leftImg8bit.png").unlink()
    with pytest.raises(ValueError, match=r"0002.*000412"):
        build_img_all(root, raw)


def test_train_all_split_loads_30_frame_clips(tmp_path):
    raw, root = _tree(tmp_path, CLIPS[:2])
    build_img_all(root, raw)
    ds = CityscapesVPS(CityscapesVPSConfig.from_yaml(CONFIG, root=str(root), split="train_all"))
    assert [ds.clip_id(i) for i in range(len(ds))] == ["0001", "0002"]
    for i in range(len(ds)):
        assert ds.num_frames(i) == 30
        assert list(np.flatnonzero(ds.labeled_mask(i))) == [4, 9, 14, 19, 24, 29]
        label = ds.load_label(i, 4)
        assert label.shape == (H, W) and set(np.unique(label)) == {0, 10}


def test_zip_chunks_split_clips_by_sorted_id(tmp_path):
    raw, root = _tree(tmp_path)
    build_img_all(root, raw)
    zips = zip_chunks(root, tmp_path / "zips", 2)
    assert [z.name for z in zips] == ["vps_train_all_chunk0.zip", "vps_train_all_chunk1.zip"]
    seen = []
    for z, vids in zip(zips, (("0001", "0002"), ("0003", "0004"))):
        names = zipfile.ZipFile(z).namelist()
        frames = [n for n in names if n.startswith("train/img_all/")]
        inst = [n for n in names if n.startswith("train/panoptic_inst/")]
        assert len(frames) == 60 and len(inst) == 12
        assert {Path(n).name[:4] for n in inst} == set(vids)
        assert "im_all_info_train_city_vps.json" in names
        assert "panoptic_gt_val_city_vps.json" not in names
        seen += frames
    assert len(seen) == len(set(seen)) == 120


def test_build_links_instead_of_copying(tmp_path):
    raw, root = _tree(tmp_path, CLIPS[:1])
    build_img_all(root, raw)
    dst = root / "train" / "img_all" / "frankfurt_000000_000100_leftImg8bit.png"
    assert dst.stat().st_nlink == 2  # no second copy of 27 GB of frames
    build_img_all(root, raw)  # a rerun keeps the links
    assert dst.stat().st_nlink == 2


def test_only_one_chunk_is_written_on_request(tmp_path):
    raw, root = _tree(tmp_path)
    build_img_all(root, raw)
    zips = zip_chunks(root, tmp_path / "zips", 2, only=[1])
    assert [z.name for z in zips] == ["vps_train_all_chunk1.zip"]
    assert sorted(p.name for p in (tmp_path / "zips").iterdir()) == ["vps_train_all_chunk1.zip"]
    names = zipfile.ZipFile(zips[0]).namelist()
    assert {Path(n).name[:4] for n in names if n.startswith("train/panoptic_inst/")} == {"0003", "0004"}
