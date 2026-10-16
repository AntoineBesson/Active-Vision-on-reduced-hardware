"""configs/cityscapes_vps.yaml against a fixture mirroring the verified official layout.

The fixture (tests/fixtures/generate_fixture.py::generate_official) reproduces
the file names and the three label encodings written by the official VPSNet
preparation scripts. See the header of src/data/cityscapes_vps.py.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

import generate_fixture as gf
from data.base import VideoClip
from data.cityscapes_vps import (
    CITYSCAPES_PALETTE,
    CITYSCAPES_TRAIN_CLASSES,
    CityscapesVPS,
    CityscapesVPSConfig,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "cityscapes_vps.yaml"
ROOT = gf.OFFICIAL_ROOT


def _ds(split: str = "val", root: Path = ROOT, **overrides) -> CityscapesVPS:
    cfg = CityscapesVPSConfig.from_yaml(CONFIG, root=str(root), split=split)
    return CityscapesVPS(cfg, **overrides)


def _clip_no(clip_id: str) -> int:
    return next(i for i, c in enumerate(gf.OFFICIAL_CLIPS) if c[1] == clip_id)


def test_constants_match_official_category_table():
    # (city_default.py spells "vegitation"; the Cityscapes name is used here.)
    assert CITYSCAPES_TRAIN_CLASSES == tuple(n for _, n, _, _ in gf.OFFICIAL_CATEGORIES)
    np.testing.assert_array_equal(CITYSCAPES_PALETTE, [c for _, _, c, _ in gf.OFFICIAL_CATEGORIES])


def test_val_clips_rebuilt_from_plain_img_all_names():
    ds = _ds("val")
    # img_all names carry no clip id: clips come from sequence + consecutive frames,
    # named after the VPS video id of their annotations.
    assert ds.clip_ids == ["0001", "0002", "0003"]
    assert [ds.num_frames(i) for i in range(3)] == [10, 10, 10]
    np.testing.assert_array_equal(ds.frame_indices(0), np.arange(275, 285))
    np.testing.assert_array_equal(ds.frame_indices(1), np.arange(557, 567))  # same sequence, gap -> new clip
    assert [ds.clip_group(i) for i in range(3)] == ["frankfurt_000000", "frankfurt_000000", "lindau_000001"]
    for i in range(3):
        np.testing.assert_array_equal(np.flatnonzero(ds.labeled_mask(i)), [4, 9])
        np.testing.assert_array_equal(ds.is_keyframe(i), ds.labeled_mask(i))  # offset 4, every 5


def test_val_semantic_labels_from_panoptic_video_and_json():
    ds = _ds("val")
    for i, clip_id in enumerate(ds.clip_ids):
        clip = ds[i]
        assert isinstance(clip, VideoClip) and clip.frames.shape == (10, gf.OH, gf.OW, 3)
        for t in range(clip.num_frames):
            if clip.labeled_mask[t]:
                expected = gf.official_expected(_clip_no(clip_id), t, "panoptic")
                np.testing.assert_array_equal(clip.labels[t], expected)
            else:
                assert np.all(clip.labels[t] == 255)
    # Crowd (class 18 without object id) is void in the panoptic encodings.
    assert 18 not in np.unique(ds[0].labels)


def test_three_official_encodings_agree_up_to_crowd():
    video = _ds("val")
    inst = _ds("val", annotation_subdir="panoptic_inst", label_mapping=None)
    labelmap = _ds("val", annotation_subdir="labelmap", label_mapping=None)
    for i in range(len(video)):
        for t in np.flatnonzero(video.labeled_mask(i)):
            a, b, c = video.load_label(i, t), inst.load_label(i, t), labelmap.load_label(i, t)
            np.testing.assert_array_equal(a, b)
            differ = a != c
            # Only crowd pixels differ: labelmap keeps class 18 there.
            assert set(np.unique(c[differ]).tolist()) == {18}
            assert set(np.unique(a[differ]).tolist()) == {255}


def test_train_split_uses_prefixed_img_and_panoptic_inst():
    ds = _ds("train")
    assert ds.config.image_subdir == "img" and ds.config.annotation_subdir == "panoptic_inst"
    assert ds.clip_ids == ["0004", "0005"]
    for i, clip_id in enumerate(ds.clip_ids):
        assert ds.labeled_mask(i).all() and ds.num_frames(i) == 2
        np.testing.assert_array_equal(ds.frame_indices(i), [14, 19])  # ordered by Cityscapes frame number
        np.testing.assert_array_equal(ds.is_keyframe(i), [True, True])
        for t, pos in enumerate((4, 9)):
            np.testing.assert_array_equal(ds.load_label(i, t), gf.official_expected(_clip_no(clip_id), pos, "panoptic"))


def test_explicit_overrides_beat_split_overrides():
    ds = _ds("train", annotation_subdir="labelmap", label_mapping=None)
    assert ds.config.image_subdir == "img"  # from split_overrides.train
    assert ds.config.annotation_subdir == "labelmap"  # explicit override wins
    assert ds.label_mapping_config.id_rule == "identity"
    np.testing.assert_array_equal(ds.load_label(0, 0), gf.official_expected(_clip_no("0004"), 4, "labelmap"))
    # The base config keeps the YAML values; only the effective config changes.
    assert ds.base_config.split_overrides["train"]["annotation_subdir"] == "panoptic_inst"


def test_test_split_has_frames_but_no_labels():
    ds = _ds("test")
    assert len(ds) == 1 and ds.num_frames(0) == 10
    assert not ds.labeled_mask(0).any()
    assert ds.clip_id(0) == "lindau_000004_000010"  # no annotation -> first frame key
    assert np.all(ds[0].labels == 255)


def test_label_space_and_palette():
    ds = _ds("val")
    assert ds.class_names == CITYSCAPES_TRAIN_CLASSES and ds.ignore_index == 255
    np.testing.assert_array_equal(ds.palette, CITYSCAPES_PALETTE)


def test_explain_annotation_ids_maps_colour_ids_through_the_json():
    ds = _ds("val")
    t = int(np.flatnonzero(ds.labeled_mask(0))[0])
    ids = ds.annotation_ids(0, t)
    rows = {r["id"]: r["class"] for r in ds.explain_annotation_ids(0, t, np.unique(ids))}
    assert rows[0] == "<ignore>"
    assert rows[128 + 256 * 64 + 65536 * 128] == "road"  # stuff id == its class colour
    assert rows[142 * 65536] == "car"  # first car instance gets the base colour
    assert set(rows.values()) >= {"road", "sky", "building", "person", "car"}


def test_legacy_defaults_find_no_clip_in_official_img_all():
    # The pre-verification guess (clip-prefixed img_all names) cannot index the
    # official layout -- the reason configs/cityscapes_vps.yaml changed.
    with pytest.warns(UserWarning, match="did not match"):
        ds = CityscapesVPS(CityscapesVPSConfig(root=str(ROOT), split="val"))
    assert len(ds) == 0


def test_missing_segments_json_raises_clear_error(tmp_path):
    root = tmp_path / "vps"
    shutil.copytree(ROOT, root)
    (root / "panoptic_gt_val_city_vps.json").unlink()
    ds = _ds("val", root=root)
    with pytest.raises(FileNotFoundError, match="panoptic_gt_val_city_vps.json"):
        ds.load_label(0, 4)


def test_unknown_annotation_folder_requires_explicit_mapping(tmp_path):
    root = tmp_path / "vps"
    shutil.copytree(ROOT, root)
    (root / "val" / "labelmap").rename(root / "val" / "my_labels")
    ds = _ds("val", root=root, annotation_subdir="my_labels", label_mapping=None)
    assert ds.labeled_mask(0).sum() == 2
    with pytest.raises(ValueError, match="No label mapping"):
        ds.load_label(0, 4)
    ok = _ds("val", root=root, annotation_subdir="my_labels",
             label_mapping={"encoding": "gray", "id_rule": "identity"})
    assert ok.load_label(0, 4).shape == (gf.OH, gf.OW)


def test_diagnostics_report_regex_matches_and_joins():
    d = _ds("val").diagnostics()
    idx = d["index"]
    assert idx["image_files"] == 30 and idx["image_regex_matched"] == 30
    assert idx["annotation_files"] == 6 and idx["annotation_regex_matched"] == 6
    assert idx["frames_with_annotation"] == 6 and idx["orphan_annotations"] == 0
    assert d["label_mapping"]["id_rule"] == "segments_json"
    jsons = {r["file"]: r for r in d["side_car_jsons"]}
    assert jsons["im_all_info_val_city_vps.json"]["json_order_matches_clips"] is True
    assert jsons["panoptic_gt_val_city_vps.json"]["annotation_files_found"] == 6


def test_split_overrides_validation():
    with pytest.raises(ValueError, match="forbidden"):
        CityscapesVPSConfig(split_overrides={"train": {"root": "/x"}})
    with pytest.raises(ValueError, match="mapping"):
        CityscapesVPSConfig(split_overrides={"train": ["img"]})  # type: ignore[dict-item]


def test_contiguous_clips_in_one_sequence_are_split(tmp_path):
    """Assumption A1 violated: two videos on consecutive frames of one sequence."""
    root = tmp_path / "vps"
    img_all = root / "val" / "img_all"
    ann = root / "val" / "labelmap"
    img_all.mkdir(parents=True)
    ann.mkdir(parents=True)
    from PIL import Image

    blank = np.zeros((4, 4, 3), np.uint8)
    for frame in range(100, 120):  # 20 consecutive frames
        Image.fromarray(blank).save(img_all / f"munster_000009_{frame:06d}_leftImg8bit.png")
    # video 0010 = frames 100..109 (labelled 104, 109), video 0011 = 110..119 (114, 119)
    for vid, frames in (("0010", (104, 109)), ("0011", (114, 119))):
        for f in frames:
            Image.fromarray(np.zeros((4, 4), np.uint8)).save(ann / f"{vid}_0000_munster_000009_{f:06d}_final_mask.png")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _ds("val", root=root, annotation_subdir="labelmap", label_mapping=None)
    assert any("assumption A1" in str(w.message) for w in caught)
    assert ds.clip_ids == ["0010", "0011"]
    np.testing.assert_array_equal(ds.frame_indices(0), np.arange(100, 110))
    np.testing.assert_array_equal(ds.frame_indices(1), np.arange(110, 120))
