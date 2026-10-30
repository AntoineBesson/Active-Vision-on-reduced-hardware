"""Torch Dataset/DataLoader wrapper (skipped when torch is not installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import generate_fixture as gf  # noqa: E402
from data.base import InMemoryVideoSeg  # noqa: E402
from data.cityscapes_vps import CityscapesVPS, CityscapesVPSConfig  # noqa: E402
from data.splits import split_by_clip  # noqa: E402
from data.subclips import SubClipSpec  # noqa: E402
from data.torch_dataset import SubClipDataset, make_dataloader  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cityscapes_vps.yaml"


@pytest.fixture(scope="module")
def vps() -> CityscapesVPS:
    return CityscapesVPS(CityscapesVPSConfig.from_yaml(CONFIG, root=str(gf.OFFICIAL_ROOT), split="val"))


def test_sliding_windows_and_item_layout(vps):
    ds = SubClipDataset(vps, SubClipSpec(length=4, window_stride=3))
    # each 10-frame clip -> starts 0, 3, 6
    assert len(ds) == 9
    item = ds[1]
    assert item["clip_id"] == "0001" and item["clip_index"] == 0
    assert item["frames"].shape == (4, 3, gf.OH, gf.OW) and item["frames"].dtype == torch.float32
    assert 0.0 <= float(item["frames"].min()) and float(item["frames"].max()) <= 1.0
    assert item["labels"].shape == (4, gf.OH, gf.OW) and item["labels"].dtype == torch.int64
    assert item["positions"].tolist() == [3, 4, 5, 6]
    assert item["frame_indices"].tolist() == [278, 279, 280, 281]
    assert item["labeled_mask"].tolist() == [False, True, False, False]
    assert torch.all(item["labels"][0] == 255)
    expected = gf.official_expected(0, 4, "panoptic")
    np.testing.assert_array_equal(item["labels"][1].numpy(), expected)


def test_uint8_frames_option(vps):
    ds = SubClipDataset(vps, SubClipSpec(length=2), to_float=False)
    assert ds[0]["frames"].dtype == torch.uint8


def test_label_filters_and_skipped_clips(vps):
    ds = SubClipDataset(vps, SubClipSpec(length=5, window_stride=1, require_last_labeled=True))
    assert len(ds) == 6  # two windows (ending at positions 4 and 9) per clip
    assert all(bool(ds[i]["labeled_mask"][-1]) for i in range(len(ds)))
    too_long = SubClipDataset(vps, SubClipSpec(length=11))
    assert len(too_long) == 0 and too_long.skipped_clips == [0, 1, 2]


def test_split_by_clip_never_mixes_clips(vps):
    splits = split_by_clip(vps, {"train": 2 / 3, "val": 1 / 3}, seed=0)
    train = SubClipDataset(vps, SubClipSpec(length=2, window_stride=1), clip_indices=splits["train"])
    val = SubClipDataset(vps, SubClipSpec(length=2, window_stride=1), clip_indices=splits["val"])
    train_clips = {train.window(i)[0] for i in range(len(train))}
    val_clips = {val.window(i)[0] for i in range(len(val))}
    assert train_clips == set(splits["train"]) and val_clips == set(splits["val"])
    assert train_clips.isdisjoint(val_clips)
    with pytest.raises(ValueError, match="duplicates"):
        SubClipDataset(vps, SubClipSpec(length=2), clip_indices=[0, 0])


def test_random_mode_is_seeded_and_epoch_dependent(vps):
    spec = SubClipSpec(length=3, window_stride=1)
    a = SubClipDataset(vps, spec, mode="random", samples_per_clip=4, seed=5)
    b = SubClipDataset(vps, spec, mode="random", samples_per_clip=4, seed=5)
    assert len(a) == 12
    draws = [a.window(i)[1].tolist() for i in range(len(a))]
    assert draws == [b.window(i)[1].tolist() for i in range(len(b))]
    a.set_epoch(1)
    assert draws != [a.window(i)[1].tolist() for i in range(len(a))]
    c = SubClipDataset(vps, spec, mode="random", samples_per_clip=4, seed=6)
    assert draws != [c.window(i)[1].tolist() for i in range(len(c))]
    # every draw is a valid window of the right clip
    for i in range(len(a)):
        ci, pos = a.window(i)
        assert ci == a.usable_clips[i // 4]
        assert any(np.array_equal(pos, w) for w in a.windows[ci])


def _batches(loader):
    return [(b["clip_id"], b["positions"].tolist(), b["frames"].sum().item()) for b in loader]


# Python >= 3.12 warns when forking a process that already runs torch threads.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_dataloader_determinism_and_worker_independence(vps):
    spec = SubClipSpec(length=2, window_stride=1)
    ds = SubClipDataset(vps, spec, mode="random", samples_per_clip=2, seed=0)
    first = _batches(make_dataloader(ds, batch_size=2, shuffle=True, seed=123))
    again = _batches(make_dataloader(ds, batch_size=2, shuffle=True, seed=123))
    assert first == again
    other = _batches(make_dataloader(ds, batch_size=2, shuffle=True, seed=124))
    assert first != other
    workers = _batches(make_dataloader(ds, batch_size=2, shuffle=True, seed=123, num_workers=2))
    assert workers == first
    batch = next(iter(make_dataloader(ds, batch_size=3)))
    assert batch["frames"].shape == (3, 2, 3, gf.OH, gf.OW)
    assert batch["labels"].shape == (3, 2, gf.OH, gf.OW)
    assert isinstance(batch["clip_id"], list) and len(batch["clip_id"]) == 3


def test_transform_hook_and_mixed_size_error():
    small = np.zeros((2, 4, 4, 3), np.uint8)
    large = np.zeros((2, 6, 6, 3), np.uint8)
    ds = InMemoryVideoSeg(
        [
            {"clip_id": "s", "frames": small, "labels": [np.zeros((4, 4), np.uint8), None]},
            {"clip_id": "l", "frames": large, "labels": [None, None]},
        ],
        class_names=["x"],
    )

    def flip(sample):
        sample["frames"] = sample["frames"].flip(-1)
        sample["flipped"] = True
        return sample

    sub = SubClipDataset(ds, SubClipSpec(length=2), transform=flip)
    assert sub[0]["flipped"] is True
    with pytest.raises(RuntimeError, match="different sizes"):
        next(iter(make_dataloader(sub, batch_size=2)))
    assert sub.describe()["num_items"] == 2
