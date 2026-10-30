"""Deterministic train/val(/test) splits made at the CLIP level -- numpy only.

Frames are never split: a clip lands in exactly one split, and so does every
clip of the same *group* (see :meth:`data.base.VideoSegDataset.clip_group`,
e.g. clips cut from one continuous recording), so near-duplicate frames
cannot leak across splits.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

from .base import VideoSegDataset

__all__ = ["split_by_clip", "split_indices_by_group"]


def split_indices_by_group(
    groups: Sequence[str], fractions: Mapping[str, float], seed: int = 0
) -> Dict[str, List[int]]:
    """Assign item ``i`` (whose group is ``groups[i]``) to a named split.

    Groups are visited in a seeded random order and each goes, whole, to the
    split furthest below its target item count (``fraction * len(groups)``).
    The result depends only on the *set* of groups, the fractions and the
    seed -- not on the order of ``groups``. Returns sorted index lists.
    """
    if not fractions:
        raise ValueError("fractions must name at least one split")
    names = list(fractions)
    fr = np.array([float(fractions[n]) for n in names])
    if np.any(fr < 0) or not np.isclose(fr.sum(), 1.0, atol=1e-6):
        raise ValueError(f"fractions must be >= 0 and sum to 1, got {dict(fractions)}")

    members: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(str(g), []).append(i)
    unique = sorted(members)
    order = np.random.default_rng(seed).permutation(len(unique))

    targets = fr * len(groups)
    counts = np.zeros(len(names))
    out: Dict[str, List[int]] = {n: [] for n in names}
    for gi in order:
        g = unique[gi]
        deficit = targets - counts
        deficit[fr == 0] = -np.inf  # never assign to a zero-fraction split
        k = int(np.argmax(deficit))
        out[names[k]].extend(members[g])
        counts[k] += len(members[g])
    return {n: sorted(v) for n, v in out.items()}


def split_by_clip(
    dataset: VideoSegDataset,
    fractions: Mapping[str, float],
    seed: int = 0,
    by_group: bool = True,
) -> Dict[str, List[int]]:
    """Split the clips of ``dataset`` into named subsets of clip indices.

    ``by_group=True`` keeps clips of the same :meth:`clip_group` together;
    ``False`` splits by clip id only.
    """
    keys = [dataset.clip_group(i) if by_group else dataset.clip_id(i) for i in range(len(dataset))]
    return split_indices_by_group(keys, fractions, seed)
