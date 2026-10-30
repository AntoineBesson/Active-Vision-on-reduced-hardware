"""Evaluate glimpse-selection rules exactly from the per-cell caches (:mod:`foveation.cache`).

A selection is a boolean (frames, cells) mask. Its confusion is, per frame, the
context matrices of the unselected cells plus the glimpse matrices of the
selected ones, then summed per clip (for the clip bootstrap). The uncertainty
rules use the same pooling and selection ops as :class:`FoveatedSegmenter`,
on the features the context cache stored, so a rule picks the same cells here
as in the pipeline.

Reference rules: ``oracle`` (the k cells whose glimpse adds the most correct
pixels over the context, chosen with the labels: no rule gets more correct
pixels at a given geometry and k; mIoU weighs classes equally, so a rule can
still beat it in mIoU by favouring rare classes), ``random`` (seeded per frame), ``prior`` (the same k
cells in every frame, ranked by their mean oracle gain on train) and ``all``
(every cell glimpsed: the tiling bound).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from eval.tradeoff import miou_of

from .ops import SCORE_KINDS, GlimpseGrid, pool_scores, select_greedy_batch

__all__ = ["CachedPair", "Rule", "RULE_KINDS", "select_cells", "prior_order", "compose", "miou_of"]

RULE_KINDS = ("uncertainty", "oracle", "random", "prior", "all")


@dataclass
class CachedPair:
    """A context cache and a glimpse cache over the same frames and grid."""

    ctx_conf: np.ndarray  # (N, cells, C, C) int32
    gl_conf: np.ndarray  # (N, cells, C, C) int32
    features: Dict[str, Tuple[torch.Tensor, torch.Tensor]]  # kind -> (class_sums (N,cells,C), top10 (N,cells))
    q: int
    grid: GlimpseGrid
    clip_index: np.ndarray  # (N,) dataset clip index of each frame
    t: np.ndarray  # (N,) frame position in its clip
    meta: Dict[str, Any]  # {"context": ..., "glimpse": ...}

    @classmethod
    def load(cls, ctx_path: Path, gl_path: Path) -> "CachedPair":
        ctx, gl = np.load(ctx_path), np.load(gl_path)
        mc, mg = json.loads(str(ctx["meta"])), json.loads(str(gl["meta"]))
        if mc["kind"] != "context" or mg["kind"] != "glimpse":
            raise ValueError(f"expected a context and a glimpse cache, got {mc['kind']} and {mg['kind']}")
        for key in ("size", "margin", "rows", "cols", "frame_size", "num_classes"):
            if mc[key] != mg[key]:
                raise ValueError(f"caches disagree on {key}: {mc[key]} vs {mg[key]}")
        if not (np.array_equal(ctx["clip_index"], gl["clip_index"]) and np.array_equal(ctx["t"], gl["t"])):
            raise ValueError("caches cover different frames (or a different order)")
        feats = {k: (torch.from_numpy(ctx[f"class_sums_{k}"]), torch.from_numpy(ctx[f"top10_{k}"])) for k in SCORE_KINDS}
        grid = GlimpseGrid(tuple(mc["frame_size"]), mc["size"], mc["margin"])
        return cls(ctx_conf=ctx["conf"], gl_conf=gl["conf"], features=feats, q=int(ctx["q"]), grid=grid,
                   clip_index=ctx["clip_index"], t=ctx["t"], meta={"context": mc, "glimpse": mg})

    @property
    def num_frames(self) -> int:
        return int(self.ctx_conf.shape[0])

    @property
    def clip_ids(self) -> np.ndarray:
        """Dataset clip indices in the order of :func:`compose`'s first axis."""
        return np.unique(self.clip_index)

    def gain(self) -> np.ndarray:
        """(N, cells) correct pixels added by glimpsing each cell (may be negative)."""
        return (np.einsum("ncii->nc", self.gl_conf.astype(np.int64))
                - np.einsum("ncii->nc", self.ctx_conf.astype(np.int64)))


@dataclass(frozen=True)
class Rule:
    kind: str
    score: str = "entropy"
    pool: str = "mean"
    class_weights: Optional[Tuple[float, ...]] = None
    selection: str = "topk"
    suppress: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in RULE_KINDS:
            raise ValueError(f"unknown rule {self.kind!r}, expected one of {RULE_KINDS}")

    def name(self) -> str:
        if self.kind == "uncertainty":
            parts = [self.kind, self.score, self.pool, self.selection]
            if self.class_weights is not None:
                parts.append("weighted")
            return "/".join(parts)
        return f"random/seed{self.seed}" if self.kind == "random" else self.kind


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, order, True)


def prior_order(pair: CachedPair) -> torch.Tensor:
    """Cells sorted by mean oracle gain over the pair's frames (best first; compute it on train)."""
    return torch.sort(torch.from_numpy(pair.gain().mean(0)), descending=True, stable=True).indices


def select_cells(pair: CachedPair, rule: Rule, k: int, prior_order: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(N, cells) bool mask of the cells ``rule`` glimpses in each frame (``k`` ignored for ``all``)."""
    n, cells = pair.num_frames, pair.grid.num_cells
    if rule.kind == "all":
        return torch.ones(n, cells, dtype=torch.bool)
    if rule.kind == "uncertainty":
        sums, top10 = pair.features[rule.score]
        w = None if rule.class_weights is None else torch.tensor(rule.class_weights, dtype=torch.float32)
        scores = pool_scores(sums, top10, pair.q, rule.pool, w)
        if rule.selection == "topk":
            return _topk_mask(scores, k)
        idx = select_greedy_batch(scores, pair.grid, k, rule.suppress)
        return torch.zeros(n, cells, dtype=torch.bool).scatter_(1, idx, True)
    if rule.kind == "oracle":
        return _topk_mask(torch.from_numpy(pair.gain()).double(), k)
    if rule.kind == "random":
        mask = torch.zeros(n, cells, dtype=torch.bool)
        for i in range(n):
            g = torch.Generator().manual_seed(rule.seed * 1_000_003 + i)
            mask[i, torch.randperm(cells, generator=g)[:k]] = True
        return mask
    if prior_order is None:
        raise ValueError("the prior rule needs prior_order (from the train caches)")
    mask = torch.zeros(n, cells, dtype=torch.bool)
    mask[:, prior_order[:k]] = True
    return mask


def compose(pair: CachedPair, mask: torch.Tensor) -> np.ndarray:
    """(n_clips, C, C) int64 confusion of a selection, per clip in :attr:`CachedPair.clip_ids` order."""
    if not hasattr(pair, "_terms"):  # computed once per pair: sweeps compose thousands of selections
        # int32 is enough: a frame has ~2M pixels; it halves the memory of train-sized caches
        pair._terms = (pair.ctx_conf.sum(1, dtype=np.int64), pair.gl_conf - pair.ctx_conf)
    ctx_total, diff = pair._terms
    frames = ctx_total + np.einsum("nc,ncij->nij", mask.numpy().astype(np.int32), diff).astype(np.int64)
    clips = pair.clip_ids
    out = np.zeros((len(clips),) + frames.shape[1:], dtype=np.int64)
    np.add.at(out, np.searchsorted(clips, pair.clip_index), frames)
    return out


