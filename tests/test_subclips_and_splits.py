"""Sub-clip windows and clip-level splits (numpy only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import generate_fixture as gf
from data.base import InMemoryVideoSeg
from data.cityscapes_vps import CityscapesVPS, CityscapesVPSConfig
from data.splits import split_by_clip, split_indices_by_group
from data.subclips import SubClipSpec, enumerate_windows

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cityscapes_vps.yaml"


# --- windows -------------------------------------------------------------- #
def _starts(windows):
    return [int(w[0]) for w in windows]


def test_non_overlapping_default_windows():
    w = enumerate_windows(10, np.zeros(10, bool), SubClipSpec(length=3))
    assert _starts(w) == [0, 3, 6]
    np.testing.assert_array_equal(w[1], [3, 4, 5])


def test_window_and_frame_strides():
    spec = SubClipSpec(length=3, frame_stride=2, window_stride=1)
    assert spec.span == 5
    w = enumerate_windows(7, np.zeros(7, bool), spec)
    assert _starts(w) == [0, 1, 2]
    np.testing.assert_array_equal(w[2], [2, 4, 6])


def test_label_filters():
    mask = np.zeros(10, bool)
    mask[[4, 9]] = True
    spec = SubClipSpec(length=5, window_stride=1, require_last_labeled=True)
    assert _starts(enumerate_windows(10, mask, spec)) == [0, 5]  # windows ending at 4 and 9
    spec = SubClipSpec(length=3, window_stride=1, min_labeled=1)
    assert _starts(enumerate_windows(10, mask, spec)) == [2, 3, 4, 7]
    # VPS-style: only labelled frames, 5 apart
    spec = SubClipSpec(length=2, frame_stride=5, window_stride=1, min_labeled=2)
    w = enumerate_windows(10, mask, spec)
    assert len(w) == 1
    np.testing.assert_array_equal(w[0], [4, 9])


def test_short_clip_yields_nothing_and_spec_validation():
    assert enumerate_windows(2, np.zeros(2, bool), SubClipSpec(length=3)) == []
    with pytest.raises(ValueError):
        SubClipSpec(length=0)
    with pytest.raises(ValueError):
        SubClipSpec(length=2, frame_stride=0)
    with pytest.raises(ValueError):
        SubClipSpec(length=2, window_stride=0)
    with pytest.raises(ValueError):
        SubClipSpec(length=2, min_labeled=3)
    with pytest.raises(ValueError, match="shape"):
        enumerate_windows(3, np.zeros(4, bool), SubClipSpec(length=1))


# --- splits --------------------------------------------------------------- #
def test_split_is_a_partition_and_keeps_groups_together():
    groups = [f"g{i // 3}" for i in range(30)]  # 10 groups of 3 clips
    out = split_indices_by_group(groups, {"train": 0.7, "val": 0.3}, seed=1)
    assert sorted(out["train"] + out["val"]) == list(range(30))
    assert not set(out["train"]) & set(out["val"])
    for split in out.values():
        for i in split:  # whole groups only
            assert {j for j in range(30) if groups[j] == groups[i]} <= set(split)
    assert (len(out["train"]), len(out["val"])) == (21, 9)


def test_split_is_deterministic_and_order_independent():
    groups = [f"c{i}" for i in range(20)]
    a = split_indices_by_group(groups, {"train": 0.8, "val": 0.2}, seed=7)
    b = split_indices_by_group(groups, {"train": 0.8, "val": 0.2}, seed=7)
    assert a == b
    c = split_indices_by_group(groups, {"train": 0.8, "val": 0.2}, seed=8)
    assert a != c
    # Same set of groups listed in another order -> same assignment of groups.
    perm = list(reversed(groups))
    d = split_indices_by_group(perm, {"train": 0.8, "val": 0.2}, seed=7)
    assert {perm[i] for i in d["val"]} == {groups[i] for i in a["val"]}


def test_split_fraction_validation_and_zero_fraction():
    with pytest.raises(ValueError, match="sum to 1"):
        split_indices_by_group(["a", "b"], {"train": 0.5, "val": 0.3})
    with pytest.raises(ValueError):
        split_indices_by_group(["a"], {})
    out = split_indices_by_group(["a", "b", "c"], {"train": 1.0, "val": 0.0})
    assert out == {"train": [0, 1, 2], "val": []}


def test_split_by_clip_groups_sequences_of_the_real_layout():
    ds = CityscapesVPS(CityscapesVPSConfig.from_yaml(CONFIG, root=str(gf.OFFICIAL_ROOT), split="val"))
    out = split_by_clip(ds, {"a": 0.5, "b": 0.5}, seed=0)
    # Clips 0001 and 0002 come from the same Cityscapes sequence: same side.
    side = {ds.clip_id(i): name for name, idx in out.items() for i in idx}
    assert side["0001"] == side["0002"]
    by_id = split_by_clip(ds, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, seed=0, by_group=False)
    assert sorted(len(v) for v in by_id.values()) == [1, 1, 1]


def test_split_by_clip_on_in_memory_dataset():
    frames = np.zeros((2, 2, 2, 3), np.uint8)
    clips = [{"clip_id": f"c{i}", "frames": frames, "labels": [None, None], "group": f"g{i % 2}"} for i in range(6)]
    ds = InMemoryVideoSeg(clips, class_names=["x"])
    out = split_by_clip(ds, {"train": 0.5, "val": 0.5}, seed=3)
    groups = {name: {ds.clip_group(i) for i in idx} for name, idx in out.items()}
    assert groups["train"].isdisjoint(groups["val"])
