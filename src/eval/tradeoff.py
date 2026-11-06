"""Where a method stands against the full-frame curve: the two trade-off readings, with a clip bootstrap.

The full-frame baseline at several input scales gives a curve of mIoU against
latency. A method at (latency, mIoU) is read against it in two ways, since the
study has not fixed which trade-off matters:

* **ΔmIoU at equal latency**: its mIoU minus the curve's mIoU at its latency;
* **latency saved at equal mIoU**: the curve's latency at its mIoU minus its latency.

The curve is piecewise linear between the measured scales; outside them (or,
for the second reading, where mIoU does not rise with latency) the reading is
undefined (NaN) rather than extrapolated. Uncertainty comes from resampling
clips with replacement, the same clips for the method and every reference
scale, so their shared difficulty cancels. numpy only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

__all__ = ["miou_of", "FullFrameCurve", "delta_miou_at_equal_latency", "latency_saved_at_equal_miou", "clip_bootstrap"]


def miou_of(matrix: np.ndarray) -> float:
    """mIoU of a (C, C) confusion (rows = ground truth) over the classes with a non-empty union.

    Same rule as :meth:`eval.metrics.ConfusionMatrix.compute` when no pixel is unpredicted.
    """
    m = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(m)
    union = m.sum(0) + m.sum(1) - tp
    defined = union > 0
    return float(np.mean(tp[defined] / union[defined])) if defined.any() else float("nan")


class FullFrameCurve:
    def __init__(self, latency_ms: Sequence[float], miou: Sequence[float]):
        order = np.argsort(np.asarray(latency_ms, dtype=np.float64))
        self.latency = np.asarray(latency_ms, dtype=np.float64)[order]
        self.miou = np.asarray(miou, dtype=np.float64)[order]

    def miou_at(self, latency_ms: float) -> float:
        if not self.latency[0] <= latency_ms <= self.latency[-1]:
            return float("nan")
        return float(np.interp(latency_ms, self.latency, self.miou))

    def latency_at(self, miou: float) -> float:
        if not np.all(np.diff(self.miou) > 0) or not self.miou[0] <= miou <= self.miou[-1]:
            return float("nan")
        return float(np.interp(miou, self.miou, self.latency))


def delta_miou_at_equal_latency(miou: float, latency_ms: float, curve: FullFrameCurve) -> float:
    return miou - curve.miou_at(latency_ms)


def latency_saved_at_equal_miou(miou: float, latency_ms: float, curve: FullFrameCurve) -> float:
    return curve.latency_at(miou) - latency_ms


def clip_bootstrap(config_clips: np.ndarray, ff_clips: np.ndarray, ff_latency_ms: Sequence[float],
                   latency_ms: float, n: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """95% percentile intervals of ``miou``, ``delta_miou`` and ``latency_saved`` over clip resamples.

    ``config_clips``: (N, C, C) per-clip confusion of the method; ``ff_clips``:
    (S, N, C, C) of the full frame at the S scales whose latencies are
    ``ff_latency_ms``. Latencies are fixed (they come from the T4, not from
    the clips). Resamples whose reading is undefined are dropped and counted
    in ``nan_resamples``; an interval with no defined resample is (NaN, NaN).
    """
    rng = np.random.default_rng(seed)
    num_clips = config_clips.shape[0]
    values: Dict[str, list] = {"miou": [], "delta_miou": [], "latency_saved": []}
    for _ in range(n):
        pick = rng.integers(0, num_clips, num_clips)
        miou = miou_of(config_clips[pick].sum(0))
        curve = FullFrameCurve(ff_latency_ms, [miou_of(ff[pick].sum(0)) for ff in ff_clips])
        values["miou"].append(miou)
        values["delta_miou"].append(delta_miou_at_equal_latency(miou, latency_ms, curve))
        values["latency_saved"].append(latency_saved_at_equal_miou(miou, latency_ms, curve))
    out: Dict[str, Any] = {"n": n, "seed": seed, "nan_resamples": {}}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        ok = arr[~np.isnan(arr)]
        out["nan_resamples"][key] = int(len(arr) - len(ok))
        out[key] = (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))) if len(ok) else (math.nan, math.nan)
    return out
