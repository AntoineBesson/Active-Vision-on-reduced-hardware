"""Unit tests for the Cityscapes-VPS loader, run against the synthetic fixture.

No real dataset is required: everything here uses the tiny toy files under
``tests/fixtures/cityscapes_vps`` (see tests/fixtures/generate_fixture.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.cityscapes_vps import (
    ROOT_ENV_VAR,
    CityscapesVPS,
    CityscapesVPSConfig,
    clip_to_tensors,
    decode_panoptic,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "cityscapes_vps"

# Expected fixture facts (must match generate_fixture.py).
EXPECTED_CLIP_LENGTHS = {"0000": 6, "0001": 3}
# frame_index % 5 == 0 -> labeled
EXPECTED_LABELED_ORDERS = {"0000": [0, 5], "0001": [0]}
FRAME_H, FRAME_W = 8, 16


@pytest.fixture
def dataset() -> CityscapesVPS:
    cfg = CityscapesVPSConfig(root=str(FIXTURE_ROOT), split="val")
    return CityscapesVPS(cfg)


def test_fixture_exists():
    assert FIXTURE_ROOT.is_dir(), (
        f"Missing fixture at {FIXTURE_ROOT}. Run: python tests/fixtures/generate_fixture.py"
    )


def test_discovers_all_clips(dataset: CityscapesVPS):
    assert len(dataset) == len(EXPECTED_CLIP_LENGTHS)
    assert {c.clip_id for c in dataset.clips} == set(EXPECTED_CLIP_LENGTHS)


def test_frames_per_clip(dataset: CityscapesVPS):
    for clip in dataset.clips:
        assert clip.num_frames == EXPECTED_CLIP_LENGTHS[clip.clip_id]


def test_frames_are_ordered_within_clip(dataset: CityscapesVPS):
    for clip in dataset.clips:
        indices = [f.frame_index for f in clip.frames]
        assert indices == sorted(indices)
        assert [f.order for f in clip.frames] == list(range(clip.num_frames))


def test_labeled_vs_unlabeled_identification(dataset: CityscapesVPS):
    for clip in dataset.clips:
        assert clip.labeled_orders == EXPECTED_LABELED_ORDERS[clip.clip_id]
        # Every labeled frame has an annotation path; every unlabeled one does not.
        for frame in clip.frames:
            has_ann = frame.annotation_path is not None
            assert has_ann == frame.is_labeled
            assert has_ann == (frame.order in EXPECTED_LABELED_ORDERS[clip.clip_id])


def test_load_clip_shapes(dataset: CityscapesVPS):
    clip = dataset.load_clip(0)  # clip 0000, 6 frames
    t = EXPECTED_CLIP_LENGTHS["0000"]

    assert clip["clip_id"] == "0000"
    assert clip["num_frames"] == t

    # Images stack into (T, H, W, 3) uint8.
    assert isinstance(clip["images"], np.ndarray)
    assert clip["images"].shape == (t, FRAME_H, FRAME_W, 3)
    assert clip["images"].dtype == np.uint8

    # Per-frame metadata arrays are all length T.
    assert clip["labeled_mask"].shape == (t,)
    assert clip["labeled_mask"].dtype == bool
    assert clip["is_keyframe"].shape == (t,)
    assert clip["frame_indices"].shape == (t,)
    assert len(clip["annotations"]) == t
    assert len(clip["image_paths"]) == t
    assert len(clip["annotation_paths"]) == t


def test_labeled_mask_matches_annotations(dataset: CityscapesVPS):
    clip = dataset.load_clip(0)
    expected = np.zeros(clip["num_frames"], dtype=bool)
    for order in EXPECTED_LABELED_ORDERS["0000"]:
        expected[order] = True
    np.testing.assert_array_equal(clip["labeled_mask"], expected)

    # Annotation entries are present exactly where labeled, and correctly shaped.
    for is_labeled, ann in zip(clip["labeled_mask"], clip["annotations"]):
        if is_labeled:
            assert isinstance(ann, np.ndarray)
            assert ann.shape == (FRAME_H, FRAME_W)  # decoded panoptic id map
        else:
            assert ann is None


def test_keyframe_flag_matches_labeled_here(dataset: CityscapesVPS):
    # In this fixture the "every 5th" heuristic and real labels agree.
    for i in range(len(dataset)):
        clip = dataset.load_clip(i)
        np.testing.assert_array_equal(clip["is_keyframe"], clip["labeled_mask"])


def test_panoptic_decoding_values(dataset: CityscapesVPS):
    clip = dataset.load_clip(0)
    ann = clip["annotations"][0]  # first (labeled) frame
    # Fixture encodes left half -> id 1, right half -> id 2.
    assert set(np.unique(ann).tolist()) == {1, 2}
    assert np.all(ann[:, : FRAME_W // 2] == 1)
    assert np.all(ann[:, FRAME_W // 2 :] == 2)


def test_decode_panoptic_formula():
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [5, 0, 0]  # id 5
    rgb[0, 1] = [0, 1, 0]  # id 256
    rgb[0, 2] = [0, 0, 1]  # id 65536
    decoded = decode_panoptic(rgb)
    np.testing.assert_array_equal(decoded[0], [5, 256, 65536])


def test_semantic_labelmap_branch():
    # Point the loader at the single-channel labelmap dir instead of panoptic.
    cfg = CityscapesVPSConfig(
        root=str(FIXTURE_ROOT), split="val", annotation_subdir="labelmap"
    )
    ds = CityscapesVPS(cfg)
    clip = ds.load_clip(0)
    ann = clip["annotations"][0]
    assert ann.ndim == 2  # single-channel semantic map stays (H, W)
    assert ann.shape == (FRAME_H, FRAME_W)
    assert set(np.unique(ann).tolist()) <= {0, 2}


def test_raw_annotation_without_decode(dataset: CityscapesVPS):
    clip = dataset.clips[0]
    labeled = next(f for f in clip.frames if f.is_labeled)
    raw = dataset.load_annotation(labeled.annotation_path, decode=False)
    assert raw.shape == (FRAME_H, FRAME_W, 3)


def test_annotation_suffix_exact_match():
    cfg = CityscapesVPSConfig(
        root=str(FIXTURE_ROOT),
        split="val",
        annotation_subdir="panoptic_video",
        annotation_suffix="_gtFine_panoptic",
    )
    ds = CityscapesVPS(cfg)
    assert ds.clips[0].labeled_orders == EXPECTED_LABELED_ORDERS["0000"]


def test_missing_annotation_dir_means_all_unlabeled():
    cfg = CityscapesVPSConfig(
        root=str(FIXTURE_ROOT), split="val", annotation_subdir="does_not_exist"
    )
    ds = CityscapesVPS(cfg)
    for clip in ds.clips:
        assert clip.labeled_orders == []
        assert not clip.labeled_mask.any()


def test_from_yaml_and_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(FIXTURE_ROOT))
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("split: val\nannotation_subdir: panoptic_video\n")
    cfg = CityscapesVPSConfig.from_yaml(yaml_path)  # root=None -> uses env var
    ds = CityscapesVPS(cfg)
    assert len(ds) == len(EXPECTED_CLIP_LENGTHS)


def test_missing_root_raises(monkeypatch):
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="root not set"):
        CityscapesVPS(CityscapesVPSConfig(root=None)).clips  # resolve happens in __init__


def test_from_yaml_rejects_unknown_keys(tmp_path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("not_a_real_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        CityscapesVPSConfig.from_yaml(yaml_path)


def test_clip_to_tensors_optional_torch(dataset: CityscapesVPS):
    torch = pytest.importorskip("torch")
    clip = dataset.load_clip(0)
    t = clip["num_frames"]
    tensors = clip_to_tensors(clip)
    assert tensors["images"].shape == (t, 3, FRAME_H, FRAME_W)
    assert tensors["images"].dtype == torch.float32
    assert float(tensors["images"].max()) <= 1.0
    for is_labeled, ann in zip(clip["labeled_mask"], tensors["annotations"]):
        if is_labeled:
            assert ann.shape == (FRAME_H, FRAME_W)
            assert ann.dtype == torch.int64
        else:
            assert ann is None
