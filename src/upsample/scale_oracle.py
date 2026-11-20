"""Method 8: the per-frame scale oracle, an upper bound on choosing the full frame's scale per frame.

Spec: docs/superpowers/specs/2026-11-20-method7-8-guided-upsampling-scale-oracle-design.md.

Given the (19, 19) confusion of every labelled frame at every scale of a menu, and each scale's
latency, the oracle picks one scale per frame to maximise the dataset mIoU (of the summed
confusions) under a budget on the mean latency. mIoU does not split over frames, so this is a
Lagrangian relaxation solved by coordinate ascent (each frame in turn takes the scale that
maximises mIoU - lambda x mean latency given the others), a bisection on lambda for the budget,
and a greedy fill of the slack left. It uses the labels: it is a bound, not a method. No
realisable selector can do better than the exact optimum; this solver may fall slightly short of
it, which only makes the bound weaker by that much (the spec reads it as a bound for that reason
and checks it against random restarts). numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["miou_many", "OracleResult", "scale_oracle", "fixed_mix"]


def miou_many(conf: np.ndarray) -> np.ndarray:
    """mIoU of each (..., C, C) confusion (rows = ground truth) over its classes with a non-empty union
    (:func:`eval.tradeoff.miou_of`, vectorised)."""
    m = np.asarray(conf, dtype=np.float64)
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    union = m.sum(-2) + m.sum(-1) - tp
    ok = union > 0
    iou = np.where(ok, tp / np.where(ok, union, 1.0), 0.0)
    n = ok.sum(-1)
    return np.where(n > 0, iou.sum(-1) / np.maximum(n, 1), np.nan)


@dataclass
class OracleResult:
    choice: np.ndarray  # (F,) scale index per frame
    miou: float
    mean_ms: float
    lam: float
    sweeps: int


def _ascent(conf: np.ndarray, lat: np.ndarray, choice: np.ndarray, lam: float, max_sweeps: int,
            rng: np.random.Generator) -> tuple:
    n = conf.shape[0]
    total = conf[np.arange(n), choice].sum(0)
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        changed = 0
        for f in rng.permutation(n):
            cand = total[None] - conf[f, choice[f]][None] + conf[f]  # (S, C, C)
            gain = miou_many(cand) - lam * lat / n
            best = int(np.nanargmax(gain))
            if best != choice[f] and gain[best] > gain[choice[f]] + 1e-12:
                total = cand[best]
                choice[f] = best
                changed += 1
        if not changed:
            break
    return choice, total, sweeps


def _fill(conf: np.ndarray, lat: np.ndarray, choice: np.ndarray, total: np.ndarray, budget: float) -> tuple:
    """Greedy: while some single-frame change fits the budget and raises mIoU, take the best one."""
    n = conf.shape[0]
    cur = float(miou_many(total))
    spent = float(lat[choice].sum())
    while True:
        best = (0.0, -1, -1, None)
        for f in range(n):
            extra = lat - lat[choice[f]]
            fits = spent + extra <= budget * n + 1e-9
            if not fits.any():
                continue
            cand = total[None] - conf[f, choice[f]][None] + conf[f]
            gain = np.where(fits, miou_many(cand) - cur, -np.inf)
            s = int(np.argmax(gain))
            if gain[s] > best[0] + 1e-12:
                best = (float(gain[s]), f, s, cand[s])
        if best[1] < 0:
            return choice, total
        _, f, s, t = best
        spent += lat[s] - lat[choice[f]]
        choice[f], total, cur = s, t, cur + best[0]


def scale_oracle(conf: np.ndarray, lat: Sequence[float], budget_ms: float, iters: int = 30, max_sweeps: int = 20,
                 seed: int = 0, init: Optional[np.ndarray] = None, fill: bool = True) -> OracleResult:
    """``conf``: (F, S, C, C) per-frame confusions at each of S scales; ``lat``: (S,) latency of each
    scale; the mean latency over frames must not exceed ``budget_ms``. Returns the best feasible
    selection found."""
    conf = np.asarray(conf, dtype=np.int64)
    lat = np.asarray(lat, dtype=np.float64)
    if conf.ndim != 4 or conf.shape[1] != len(lat):
        raise ValueError(f"conf must be (F, S, C, C) with S = len(lat); got {conf.shape} and {len(lat)}")
    if budget_ms < lat.min() - 1e-12:
        raise ValueError(f"budget {budget_ms} ms is below the cheapest scale ({lat.min()} ms)")
    rng = np.random.default_rng(seed)
    n = conf.shape[0]
    feasible = np.flatnonzero(lat <= budget_ms + 1e-12)
    base = int(feasible[np.argmax(lat[feasible])])  # the costliest single scale within the budget
    start = np.full(n, base) if init is None else np.asarray(init).copy()

    def run(lam: float):
        choice, total, sweeps = _ascent(conf, lat, start.copy(), lam, max_sweeps, rng)
        return choice, total, sweeps, float(lat[choice].mean())

    best: Optional[OracleResult] = None

    def keep(choice, total, sweeps, mean, lam):
        nonlocal best
        if mean <= budget_ms + 1e-9:
            m = float(miou_many(total))
            if best is None or m > best.miou:
                best = OracleResult(choice.copy(), m, mean, lam, sweeps)

    lo, hi = 0.0, 1.0  # lambda: mIoU (fraction) per ms of mean latency
    r = run(lo)
    keep(*r, lo)
    if r[3] > budget_ms + 1e-9:  # unconstrained optimum over budget: bisect lambda
        while run(hi)[3] > budget_ms + 1e-9:
            hi *= 2.0
            if hi > 1e6:
                raise RuntimeError("scale_oracle: no lambda meets the budget")
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r = run(mid)
            keep(*r, mid)
            if r[3] > budget_ms + 1e-9:
                lo = mid
            else:
                hi = mid
        keep(*run(hi), hi)
    total = conf[np.arange(n), start].sum(0)
    keep(start.copy(), total, 0, float(lat[start].mean()), float("nan"))  # never worse than the single scale
    assert best is not None
    if fill:  # spend the slack the best relaxed solution leaves
        choice, total = _fill(conf, lat, best.choice.copy(), conf[np.arange(n), best.choice].sum(0), budget_ms)
        best = OracleResult(choice, float(miou_many(total)), float(lat[choice].mean()), best.lam, best.sweeps)
    return best


def fixed_mix(conf: np.ndarray, lat: Sequence[float], budget_ms: float) -> dict:
    """The best mix of two adjacent scales at a fixed frame share meeting the budget exactly in
    expectation, with frames assigned in order (no oracle): the label-free reference a selector
    must beat. Returns the mIoU and the pair."""
    conf = np.asarray(conf, dtype=np.int64)
    lat = np.asarray(lat, dtype=np.float64)
    n = conf.shape[0]
    order = np.argsort(lat)
    best = {"miou": -np.inf}
    for lo, hi in zip(order[:-1], order[1:]):
        if not lat[lo] <= budget_ms <= lat[hi]:
            continue
        share = 0.0 if lat[hi] == lat[lo] else (budget_ms - lat[lo]) / (lat[hi] - lat[lo])
        k = int(np.floor(share * n + 1e-9))
        choice = np.full(n, lo)
        choice[np.linspace(0, n - 1, k).round().astype(int) if k else []] = hi
        m = float(miou_many(conf[np.arange(n), choice].sum(0)))
        if m > best["miou"]:
            best = {"miou": m, "scales": [int(lo), int(hi)], "share_high": k / n, "mean_ms": float(lat[choice].mean())}
    return best
