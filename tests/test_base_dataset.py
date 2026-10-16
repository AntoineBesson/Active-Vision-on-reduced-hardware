"""Tests for the dataset-agnostic VideoSegDataset interface (data.base)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.base import InMemoryVideoSeg, VideoClip, VideoSegDataset, default_palette, label_dtype
from data.cityscapes_vps import CITYSCAPES_TRAIN_CLASSES, CityscapesVPS, CityscapesVPSConfig

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "cityscapes_vps"
CLASSES = ("background", "thing", "stuff")


def _toy_clips():
    rng = np.random.default_rng(0)
    frames_a = rng.integers(0, 255, size=(4, 5, 6, 3), dtype=np.uint8)
    lbl = np.zeros((5, 6), dtype=np.uint8)
    lbl[:, 3:] = 2
    frames_b = rng.integers(0, 255, size=(2, 5, 6, 3), dtype=np.uint8)
    return [
        {"clip_id": "a", "frames": frames_a, "labels": [lbl, None, None, lbl + 0], "group": "seq0"},
        {"clip_id": "b", "frames": frames_b, "labels": [None, None], "frame_indices": [10, 11]},
    ]


@pytest.fixture
def toy() -> InMemoryVideoSeg:
    return InMemoryVideoSeg(_toy_clips(), class_names=CLASSES, ignore_index=255)


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        VideoSegDataset()  # type: ignore[abstract]

    class Incomplete(VideoSegDataset):  # missing the pixel methods
        class_names = ("x",)
        ignore_index = 255

        def __len__(self):
            return 0

        def clip_id(self, index):
            return "x"

        def labeled_mask(self, index):
            return np.zeros(1, bool)

    with pytest.raises(TypeError):
        Incomplete()


def test_label_dtype_is_smallest_fit():
    assert label_dtype(19, 255) == np.uint8
    assert label_dtype(300, 255) == np.int16
    assert label_dtype(19, -1) == np.int16
    assert label_dtype(70000, 255) == np.int32


def test_default_palette_shape_and_determinism():
    p = default_palette(19)
    assert p.shape == (19, 3) and p.dtype == np.uint8
    np.testing.assert_array_equal(p, default_palette(19))
    assert len({tuple(c) for c in p}) == 19


def test_in_memory_metadata(toy: InMemoryVideoSeg):
    assert len(toy) == 2
    assert toy.clip_ids == ["a", "b"]
    assert toy.num_classes == 3 and toy.class_names == CLASSES and toy.ignore_index == 255
    np.testing.assert_array_equal(toy.labeled_mask(0), [True, False, False, True])
    assert toy.num_frames(1) == 2
    np.testing.assert_array_equal(toy.frame_indices(1), [10, 11])
    assert toy.clip_group(0) == "seq0" and toy.clip_group(1) == "b"
    assert toy.num_labeled_frames() == 2


def test_get_clip_full(toy: InMemoryVideoSeg):
    clip = toy.get_clip(0)
    assert isinstance(clip, VideoClip)
    assert clip.num_frames == 4 and clip.frame_size == (5, 6)
    assert clip.frames.shape == (4, 5, 6, 3) and clip.frames.dtype == np.uint8
    assert clip.labels.shape == (4, 5, 6) and clip.labels.dtype == np.uint8
    np.testing.assert_array_equal(clip.labeled_indices, [0, 3])
    # Unlabelled frames are entirely ignore_index.
    assert np.all(clip.labels[1] == 255) and np.all(clip.labels[2] == 255)
    assert set(np.unique(clip.labels[0]).tolist()) == {0, 2}
    assert clip.class_names == CLASSES and clip.ignore_index == 255


def test_get_clip_subset_and_negative_index(toy: InMemoryVideoSeg):
    sub = toy.get_clip(0, frames=[2, 3])
    assert sub.num_frames == 2
    np.testing.assert_array_equal(sub.labeled_mask, [False, True])
    np.testing.assert_array_equal(sub.frame_indices, [2, 3])
    assert sub.meta["positions"] == [2, 3]
    last = toy[-1]
    assert last.clip_id == "b"
    np.testing.assert_array_equal(last.frame_indices, [10, 11])


def test_get_clip_without_labels_keeps_mask(toy: InMemoryVideoSeg):
    clip = toy.get_clip(0, load_labels=False)
    assert np.all(clip.labels == 255)
    np.testing.assert_array_equal(clip.labeled_mask, [True, False, False, True])


def test_get_clip_errors(toy: InMemoryVideoSeg):
    with pytest.raises(IndexError):
        toy.get_clip(5)
    with pytest.raises(IndexError):
        toy.get_clip(0, frames=[4])
    with pytest.raises(ValueError):
        toy.get_clip(0, frames=[])


def test_mismatched_label_shape_raises():
    frames = np.zeros((1, 4, 4, 3), np.uint8)
    ds = InMemoryVideoSeg(
        [{"clip_id": "x", "frames": frames, "labels": [np.zeros((3, 4), np.uint8)]}], class_names=CLASSES
    )
    with pytest.raises(ValueError, match="label shape"):
        ds.get_clip(0)


def test_duplicate_clip_ids_rejected():
    clips = _toy_clips()
    clips[1]["clip_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        InMemoryVideoSeg(clips, class_names=CLASSES)


def test_iteration_and_describe(toy: InMemoryVideoSeg):
    assert [c.clip_id for c in toy] == ["a", "b"]
    d = toy.describe()
    assert d == {
        "name": "in_memory",
        "num_clips": 2,
        "num_frames": 6,
        "num_labeled_frames": 2,
        "num_classes": 3,
        "ignore_index": 255,
        "class_names": list(CLASSES),
    }


# --- The Cityscapes-VPS loader is an adapter of the interface ---------------- #
def test_cityscapes_vps_is_a_video_seg_dataset():
    ds = CityscapesVPS(CityscapesVPSConfig(root=str(FIXTURE_ROOT), split="val", annotation_subdir="labelmap"))
    assert isinstance(ds, VideoSegDataset)
    assert ds.class_names == CITYSCAPES_TRAIN_CLASSES and ds.num_classes == 19
    assert ds.ignore_index == 255
    assert ds.palette.shape == (19, 3)
    assert ds.clip_ids == ["0000", "0001"]
    np.testing.assert_array_equal(ds.labeled_mask(0), [True, False, False, False, False, True])


def test_cityscapes_vps_get_clip_semantic_labels():
    ds = CityscapesVPS(CityscapesVPSConfig(root=str(FIXTURE_ROOT), split="val", annotation_subdir="labelmap"))
    clip = ds[0]
    assert isinstance(clip, VideoClip)
    assert clip.frames.shape == (6, 8, 16, 3)
    assert clip.labels.shape == (6, 8, 16)
    for t in range(clip.num_frames):
        if clip.labeled_mask[t]:
            # Fixture labelmap: top half road (0), bottom half building (2).
            assert set(np.unique(clip.labels[t]).tolist()) == {0, 2}
        else:
            assert np.all(clip.labels[t] == 255)
    # The pixels match the legacy loader exactly.
    np.testing.assert_array_equal(clip.frames, ds.load_clip(0)["images"])
