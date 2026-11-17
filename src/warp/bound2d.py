"""Method 6: one clip of the local-warp bound, every configuration of the grid on its labelled frames.

Spec: docs/superpowers/specs/2026-11-17-method6-local-warp-bound-design.md. Per labelled frame t:
the teacher (scale-1 full pass) and, for the error saliency E, the uniform pass at each scale
(:class:`warp.bound._Frame`, as Method 5, timing **now** only); per configuration the oracle
saliency, the local map (:func:`warp.grid2d.make_map2d`) and one pass of the frozen model on the
warped frame (:func:`local_passes`), scored against the labels.

Rows of three kinds: the local configurations; the uniform row per scale (alpha 0, through the 2-D
code path: the pipeline check); and the separable rows (Method 5's warp through Method 5's own
code, :func:`warp.bound.warped_passes`: the reproduction check). Each clip also records the
baseline's own resize per scale and, per local configuration, the distortion figures summed over
its labelled frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from eval.harness import frame_to_tensor
from reuse.method import FRAME
from reuse.replay import NUM_CLASSES, _confusion_t

from .bound import _Frame, _out_size, resize_ref_name, warped_passes
from .grid import make_grid
from .grid2d import Map2D, distortion, identity_map, make_map2d, unwarp_logits2d, warp_frame2d
from .oracle import KINDS, SAL_CELL

__all__ = ["LocalConfig", "FORMS", "DIST_KEYS", "local_passes", "bound_clip2d"]

FORMS = ("2d", "sep")
DIST_KEYS = ("aniso_mean", "aniso_p95", "mag_p5", "mag_p95", "t")


@dataclass(frozen=True)
class LocalConfig:
    scale: float
    saliency: Optional[str] = None  # "E", "B" or "S"; None for the uniform row
    alpha: float = 0.0
    sigma: float = 1.0  # smoothing of the saliency, in cells
    kernel: float = 4.0  # sigma_k of the sampling kernel, in cells (form "2d")
    form: str = "2d"  # "2d": the local map; "sep": Method 5's separable warp (reproduction rows)

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.saliency is not None and self.saliency not in KINDS:
            raise ValueError(f"saliency must be one of {KINDS} or None, got {self.saliency!r}")
        if (self.saliency is None) != (self.alpha == 0.0):
            raise ValueError("a saliency goes with alpha > 0, and alpha 0 (the uniform row) with none")
        if self.form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, got {self.form!r}")
        if self.form == "sep" and self.uniform:
            raise ValueError("the uniform row goes through the 2-D path (form '2d')")
        if self.sigma < 0 or self.kernel <= 0:
            raise ValueError(f"sigma must be >= 0 and kernel > 0, got {self.sigma}, {self.kernel}")

    @property
    def uniform(self) -> bool:
        return self.alpha == 0.0

    def name(self) -> str:
        if self.uniform:
            return f"6o/c{self.scale:g}/uniform"
        if self.form == "sep":
            return f"6o/c{self.scale:g}/sep/{self.saliency}/a{self.alpha:g}/s{self.sigma:g}"
        return f"6o/c{self.scale:g}/{self.saliency}/a{self.alpha:g}/s{self.sigma:g}/k{self.kernel:g}"

    def method5_name(self) -> str:
        """The Method 5 row a separable configuration reproduces (timing now)."""
        if self.form != "sep":
            raise ValueError("only a separable row has a Method 5 twin")
        return f"5o/c{self.scale:g}/{self.saliency}/a{self.alpha:g}/s{self.sigma:g}/now"


def local_passes(model: torch.nn.Module, frame: torch.Tensor, maps: Sequence[Map2D], scale: float,
                 dtype: torch.dtype = torch.float32) -> List[torch.Tensor]:
    """(1, 3, H, W) frame -> (H, W) labels per map: warp to round(scale H) x round(scale W)
    (:func:`warp.grid2d.warp_frame2d`), the network once on the batch, unwarp each
    (:func:`warp.grid2d.unwarp_logits2d`), argmax."""
    h, w = frame.shape[-2:]
    size = _out_size(frame, scale)
    x = torch.cat([warp_frame2d(frame, m, size) for m in maps]).to(dtype)
    logits = model(x).float()
    return [unwarp_logits2d(logits[i:i + 1], m, (h, w), model.align_corners, model.output_stride,
                            input_size=size).argmax(1)[0] for i, m in enumerate(maps)]


def bound_clip2d(dataset, index: int, model: torch.nn.Module, cfgs: Sequence[LocalConfig], device,
                 allow_any_frame_size: bool = False, cell: int = SAL_CELL, batch: int = 1,
                 resize_refs: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """(name -> (19, 19) int64 confusion summed over the clip's labelled frames, name -> the
    distortion figures summed over those frames, for the local configurations)."""
    names = [c.name() for c in cfgs]
    if len(set(names)) != len(names):
        raise ValueError("bound_clip2d: a configuration is given twice")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    labelled = np.flatnonzero(np.asarray(dataset.labeled_mask(index), dtype=bool))
    by_scale: Dict[float, List[LocalConfig]] = {}
    for c in cfgs:
        by_scale.setdefault(c.scale, []).append(c)
    conf = {n: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for n in names}
    if resize_refs:
        conf.update({resize_ref_name(s): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
                     for s in by_scale})
    dist = {c.name(): dict.fromkeys(DIST_KEYS, 0.0) for c in cfgs if c.form == "2d" and not c.uniform}
    model = model.eval()
    with torch.no_grad():
        for t in (int(v) for v in labelled):
            now = frame_to_tensor(dataset.load_frame(index, t), device)
            size = tuple(now.shape[-2:])
            if not allow_any_frame_size and size != FRAME:
                raise ValueError(f"frame is {size[0]}x{size[1]}; the T4 latency tables are for {FRAME[0]}x{FRAME[1]}")
            fr = _Frame(model, now, cell)
            label = torch.from_numpy(np.asarray(dataset.load_label(index, t), dtype=np.int64)).to(device)
            if tuple(label.shape) != size:
                raise ValueError(f"clip {dataset.clip_id(index)} frame {t}: label {tuple(label.shape)} and frame "
                                 f"{size} differ")
            gh, gw = size[0] // cell, size[1] // cell
            for scale, group in by_scale.items():
                if resize_refs:
                    conf[resize_ref_name(scale)] += _confusion_t(label, fr.labels(scale))
                local = [c for c in group if c.form == "2d"]
                sep = [c for c in group if c.form == "sep"]
                maps = []
                for c in local:
                    if c.uniform:
                        maps.append(identity_map(gh, gw, device))
                        continue
                    m = make_map2d(fr.saliency(c.saliency, c.sigma, c.scale), c.alpha, c.scale, c.kernel)
                    for k, v in distortion(m).items():
                        dist[c.name()][k] += v
                    maps.append(m)
                for start in range(0, len(local), batch):
                    preds = local_passes(model, now, maps[start:start + batch], scale)
                    for c, pred in zip(local[start:start + batch], preds):
                        conf[c.name()] += _confusion_t(label, pred)
                grids = [make_grid(fr.saliency(c.saliency, c.sigma, c.scale), c.alpha, c.scale) for c in sep]
                for start in range(0, len(sep), batch):
                    preds = warped_passes(model, now, grids[start:start + batch], scale)
                    for c, pred in zip(sep[start:start + batch], preds):
                        conf[c.name()] += _confusion_t(label, pred)
    return {n: m.cpu().numpy() for n, m in conf.items()}, dist
