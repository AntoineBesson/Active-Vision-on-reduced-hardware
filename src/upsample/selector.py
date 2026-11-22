"""Method 9: a realisable per-frame scale selector from the clip's own past outputs.

Spec: docs/superpowers/specs/2026-11-22-method9-scale-selector-design.md.

Per frame, :func:`output_features` reads the model's logits (no extra pass): mean pixel entropy,
mean max-probability, the 19-class pixel fractions of the prediction and the share of logit-grid
cells on a class boundary. :func:`past_features` turns a clip's per-frame features into, for each
frame t, two exponential moving averages over frames 0..t-1 only. Two selectors map them to a score
(S1: the short-EMA entropy, its sign set on train; S2: ridge regression to the oracle's scale
index), and :func:`assign` maps scores to scales with cut points set on train so that the scale
shares equal the oracle's. numpy only, except :func:`output_features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["N_CLASSES", "FEATURE_NAMES", "HALF_LIVES", "output_features", "past_features", "cut_points",
           "assign", "Selector", "fit_selector", "crossfit"]

N_CLASSES = 19
FEATURE_NAMES = (["entropy", "maxprob"] + [f"frac_{c}" for c in range(N_CLASSES)] + ["boundary"])
HALF_LIVES = (2.0, 10.0)  # frames
ENTROPY = 0  # index of the mean entropy (S1 reads its short EMA)


def output_features(logits) -> np.ndarray:
    """(1, C, gh, gw) logits -> (22,) float64 features (torch in, numpy out)."""
    import torch

    p = torch.softmax(logits.float(), 1)[0]
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(0)
    lab = p.argmax(0)
    frac = torch.bincount(lab.flatten(), minlength=p.shape[0]).float() / lab.numel()
    edge = torch.zeros_like(lab, dtype=torch.bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    out = torch.cat([ent.mean()[None], p.amax(0).mean()[None], frac[:N_CLASSES], edge.float().mean()[None]])
    return out.double().cpu().numpy()


def past_features(per_frame: np.ndarray, half_lives: Sequence[float] = HALF_LIVES) -> np.ndarray:
    """(T, D) features of a clip's frames -> (T, D x len(half_lives)) EMAs over frames before t
    (row 0, which has no past, is NaN)."""
    per_frame = np.asarray(per_frame, dtype=np.float64)
    T, D = per_frame.shape
    out = np.full((T, D * len(half_lives)), np.nan)
    for j, h in enumerate(half_lives):
        a = 1.0 - 0.5 ** (1.0 / h)
        ema = None
        for t in range(T):
            if ema is not None:
                out[t, j * D:(j + 1) * D] = ema
            ema = per_frame[t].copy() if ema is None else (1 - a) * ema + a * per_frame[t]
    return out


def cut_points(scores: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Cut points between consecutive scales so that ``scores`` split by them follow ``shares``
    (shares over scales sorted by cost, cheapest first)."""
    shares = np.asarray(shares, dtype=np.float64)
    cum = np.clip(np.cumsum(shares / shares.sum())[:-1], 0.0, 1.0)
    return np.quantile(np.asarray(scores, dtype=np.float64), cum)


def assign(scores: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Scale index (cheapest = 0) of each score: the number of cut points below it."""
    return np.searchsorted(np.asarray(cuts), np.asarray(scores), side="right")


@dataclass
class Selector:
    kind: str  # "S1" or "S2"
    cuts: np.ndarray
    sign: float = 1.0  # S1
    mean: Optional[np.ndarray] = None  # S2: standardisation
    std: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    b: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self.kind == "S1":
            return self.sign * x[:, ENTROPY]
        z = (x - self.mean) / self.std
        return z @ self.w + self.b

    def choose(self, x: np.ndarray) -> np.ndarray:
        return assign(self.score(x), self.cuts)

    def to_json(self) -> Dict[str, object]:
        d = {"kind": self.kind, "cuts": self.cuts.tolist(), "sign": self.sign, "b": self.b, **self.extra}
        for k in ("mean", "std", "w"):
            v = getattr(self, k)
            d[k] = None if v is None else v.tolist()
        return d

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "Selector":
        arr = {k: None if d.get(k) is None else np.asarray(d[k], dtype=np.float64) for k in ("mean", "std", "w")}
        return cls(d["kind"], np.asarray(d["cuts"], dtype=np.float64), float(d.get("sign", 1.0)), b=float(d.get("b", 0.0)),
                   **arr)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_selector(kind: str, x: np.ndarray, target: np.ndarray, n_scales: int, ridge: float = 1.0) -> Selector:
    """Fit on frames with a past (rows of ``x`` without NaN): ``target`` is the oracle's scale index.
    Cut points reproduce the oracle's scale shares on these frames."""
    x, target = np.asarray(x, dtype=np.float64), np.asarray(target)
    shares = np.bincount(target, minlength=n_scales) / len(target)
    if kind == "S1":
        sign = 1.0 if _spearman(x[:, ENTROPY], target) >= 0 else -1.0
        sel = Selector("S1", np.zeros(0), sign=sign)
    elif kind == "S2":
        mean, std = x.mean(0), x.std(0)
        std = np.where(std > 0, std, 1.0)
        z = (x - mean) / std
        y = target.astype(np.float64)
        b = float(y.mean())
        w = np.linalg.solve(z.T @ z + ridge * len(z) * np.eye(z.shape[1]), z.T @ (y - b))
        sel = Selector("S2", np.zeros(0), mean=mean, std=std, w=w, b=b)
    else:
        raise ValueError(f"kind must be S1 or S2, got {kind!r}")
    sel.cuts = cut_points(sel.score(x), shares)
    sel.extra = {"spearman_train": _spearman(sel.score(x), target)}
    return sel


def crossfit(kind: str, x: np.ndarray, target: np.ndarray, groups: np.ndarray, n_scales: int, folds: int = 5,
             seed: int = 0, ridge: float = 1.0) -> np.ndarray:
    """Out-of-fold scale choices: clips (``groups``) split into ``folds``; each fold's frames are
    chosen by a selector fitted, cut points included, on the other folds only."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % folds))
    f = np.asarray([fold_of[g] for g in groups])
    out = np.zeros(len(x), dtype=np.int64)
    for k in range(folds):
        test = f == k
        if not test.any():
            continue
        sel = fit_selector(kind, x[~test], target[~test], n_scales, ridge)
        out[test] = sel.choose(x[test])
    return out
