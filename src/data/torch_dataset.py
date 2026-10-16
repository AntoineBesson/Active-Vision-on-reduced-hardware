"""PyTorch ``Dataset`` / ``DataLoader`` wrappers around any :class:`VideoSegDataset`.

Requires torch (``pip install -e ".[torch]"``). Nothing else in :mod:`data`
imports this module, so the numpy-only base install keeps working.

Sampling
--------
* ``mode="sliding"``: every valid window of every selected clip, in order
  (evaluation; deterministic by construction).
* ``mode="random"``: ``samples_per_clip`` windows per clip and epoch, drawn
  from the valid windows with an RNG seeded by ``(seed, epoch, item index)``.
  The draw is therefore identical whatever the number of workers; call
  :meth:`SubClipDataset.set_epoch` each epoch to get fresh windows.

Clips are selected by index (e.g. from :func:`data.splits.split_by_clip`) and
windows never cross clip boundaries, so splits are by clip, never by frame.

Item layout (one sub-clip of T frames)::

    frames        (T, 3, H, W) float32 in [0, 1]  (uint8 if to_float=False)
    labels        (T, H, W)    int64 train ids, ignore_index where unlabelled
    labeled_mask  (T,)         bool
    frame_indices (T,)         int64 dataset-native frame numbers
    positions     (T,)         int64 positions inside the clip
    clip_index    int, clip_id str

Batches add a leading B dimension (``clip_id`` becomes a list).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, default_collate
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "data.torch_dataset requires torch. Install it with `pip install -e \".[torch]\"` "
        "(or reuse the CUDA torch preinstalled on Colab/Kaggle)."
    ) from exc

from .base import VideoSegDataset
from .subclips import SubClipSpec, enumerate_windows

__all__ = ["SubClipDataset", "collate_subclips", "make_dataloader", "seed_worker"]

MODES = ("sliding", "random")


class SubClipDataset(Dataset):
    """Sub-clips of the selected clips of a :class:`VideoSegDataset`."""

    def __init__(
        self,
        dataset: VideoSegDataset,
        spec: SubClipSpec,
        clip_indices: Optional[Sequence[int]] = None,
        mode: str = "sliding",
        samples_per_clip: int = 1,
        seed: int = 0,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        to_float: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if samples_per_clip < 1:
            raise ValueError(f"samples_per_clip must be >= 1, got {samples_per_clip}")
        self.dataset = dataset
        self.spec = spec
        self.mode = mode
        self.samples_per_clip = int(samples_per_clip)
        self.seed = int(seed)
        self.transform = transform
        self.to_float = to_float
        self.epoch = 0

        indices = list(range(len(dataset))) if clip_indices is None else [int(i) for i in clip_indices]
        if len(set(indices)) != len(indices):
            raise ValueError("clip_indices contains duplicates")
        for i in indices:
            if not 0 <= i < len(dataset):
                raise IndexError(f"clip index {i} out of range for {len(dataset)} clips")
        self.clip_indices: List[int] = indices
        self.windows: Dict[int, List[np.ndarray]] = {
            ci: enumerate_windows(dataset.num_frames(ci), dataset.labeled_mask(ci), spec) for ci in indices
        }
        self.usable_clips = [ci for ci in indices if self.windows[ci]]
        #: clips with no valid window (too short, or too few labelled frames)
        self.skipped_clips = [ci for ci in indices if not self.windows[ci]]
        self._sliding = [(ci, w) for ci in self.usable_clips for w in self.windows[ci]]

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the random window draw (``mode="random"`` only)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.mode == "sliding":
            return len(self._sliding)
        return len(self.usable_clips) * self.samples_per_clip

    def window(self, item: int):
        """(clip index, positions) of item ``item`` -- no pixel access."""
        if not 0 <= item < len(self):
            raise IndexError(f"item {item} out of range for {len(self)} sub-clips")
        if self.mode == "sliding":
            return self._sliding[item]
        ci = self.usable_clips[item // self.samples_per_clip]
        rng = np.random.default_rng([self.seed, self.epoch, item])
        windows = self.windows[ci]
        return ci, windows[int(rng.integers(len(windows)))]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        ci, positions = self.window(item)
        clip = self.dataset.get_clip(ci, frames=positions.tolist())
        frames = torch.from_numpy(np.ascontiguousarray(clip.frames)).permute(0, 3, 1, 2).contiguous()
        if self.to_float:
            frames = frames.float().div_(255.0)
        sample = {
            "frames": frames,
            "labels": torch.from_numpy(clip.labels.astype(np.int64)),
            "labeled_mask": torch.from_numpy(clip.labeled_mask.copy()),
            "frame_indices": torch.from_numpy(clip.frame_indices.astype(np.int64)),
            "positions": torch.as_tensor(positions, dtype=torch.int64),
            "clip_index": ci,
            "clip_id": clip.clip_id,
        }
        return self.transform(sample) if self.transform is not None else sample

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "spec": {
                "length": self.spec.length,
                "frame_stride": self.spec.frame_stride,
                "window_stride": self.spec.step,
                "min_labeled": self.spec.min_labeled,
                "require_last_labeled": self.spec.require_last_labeled,
            },
            "num_clips": len(self.clip_indices),
            "num_usable_clips": len(self.usable_clips),
            "num_windows": sum(len(w) for w in self.windows.values()),
            "num_items": len(self),
            "seed": self.seed,
        }


def collate_subclips(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """``default_collate`` with a clearer error for mixed frame sizes."""
    try:
        return default_collate(batch)
    except RuntimeError as exc:
        shapes = sorted({tuple(b["frames"].shape) for b in batch})
        if len(shapes) > 1:
            raise RuntimeError(
                f"cannot batch sub-clips of different sizes {shapes}; use batch_size=1 "
                "or a transform that crops/resizes to a common size"
            ) from exc
        raise


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs inside worker processes
    """Seed numpy/random in a DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(
    dataset: SubClipDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    multiprocessing_context: Optional[str] = None,
) -> DataLoader:
    """A reproducible DataLoader: seeded shuffling + seeded workers.

    ``multiprocessing_context`` (e.g. ``"spawn"``) is passed through; the
    default (fork on Linux) is fastest but Python >= 3.12 warns about forking
    a multi-threaded process.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_subclips,
        multiprocessing_context=multiprocessing_context if num_workers > 0 else None,
    )
