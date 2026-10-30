"""Per-cell caches that make every glimpse selection cheap to evaluate exactly.

The glimpse grid's cells do not overlap: in a foveated frame, a selected cell's
inner square comes entirely from its glimpse and every other pixel from the
upsampled context. A cell's glimpse also does not depend on which other cells
are selected. So, for one (backbone, context scale, grid), two passes per frame
-- the context pass and *every* cell glimpsed -- determine the confusion matrix
of any selection: per frame and cell we store the confusion of the context
prediction and of the glimpse prediction over the cell's in-frame pixels, and
a selection's confusion is the sum of the glimpse matrices of the selected
cells and the context matrices of the others.

Context and glimpse caches are separate files (one per context model + scale +
grid, one per glimpse model + grid), so mixed pairs need no extra pass. Labels
are produced by the same functions as :class:`FoveatedSegmenter`
(:func:`context_pass`, :func:`glimpse_labels`, :func:`paste_labels`), and the
context cache stores the selection features of every score kind
(:func:`cell_features`), so selection from the cache matches the pipeline's.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from eval.harness import PRECISIONS, frame_to_tensor

from .ops import SCORE_KINDS, GlimpseGrid, cell_features, gather_crops, paste_labels, samples_per_cell
from .pipeline import context_pass, glimpse_labels

__all__ = ["FRAME", "per_cell_confusion", "ContextCells", "context_cells", "glimpse_cells", "build_cache", "cache_path"]

FRAME = (1024, 2048)  # the frame the T4 latency tables were measured at


def per_cell_confusion(pred: torch.Tensor, label: torch.Tensor, grid: GlimpseGrid, num_classes: int,
                       ignore_index: int) -> torch.Tensor:
    """(cells, C, C) int32 confusion (rows = ground truth) of each cell's in-frame inner square.

    Pixels labelled ``ignore_index`` and predictions outside [0, C) are not counted.
    """
    h, w = grid.frame_size
    c = num_classes
    cell = ((torch.arange(h, device=pred.device) // grid.inner)[:, None] * grid.cols
            + (torch.arange(w, device=pred.device) // grid.inner)[None, :])
    t, p = label.long(), pred.long()
    valid = (t != ignore_index) & (t >= 0) & (t < c) & (p >= 0) & (p < c)
    code = (cell * c + t) * c + p
    counts = torch.bincount(code[valid], minlength=grid.num_cells * c * c)
    return counts.view(grid.num_cells, c, c).to(torch.int32)


@dataclass
class ContextCells:
    conf: torch.Tensor  # (cells, C, C) int32
    class_sums: Dict[str, torch.Tensor]  # kind -> (cells, C) float32
    top10: Dict[str, torch.Tensor]  # kind -> (cells,) float32
    q: int


def context_cells(backbone: torch.nn.Module, frame: torch.Tensor, label: torch.Tensor, scale: float,
                  grid: GlimpseGrid, num_classes: int, ignore_index: int,
                  kinds: Sequence[str] = SCORE_KINDS) -> ContextCells:
    """Per-cell context confusion and selection features of one (1, 3, H, W) frame."""
    with torch.no_grad():
        logits, labels = context_pass(backbone, frame, scale)
        feats = cell_features(logits, grid, kinds)
    return ContextCells(conf=per_cell_confusion(labels, label, grid, num_classes, ignore_index),
                        class_sums={k: v[0].float() for k, v in feats.items()},
                        top10={k: v[1].float() for k, v in feats.items()},
                        q=samples_per_cell(logits.shape[-2], grid))


def glimpse_cells(backbone: torch.nn.Module, frame: torch.Tensor, label: torch.Tensor, grid: GlimpseGrid,
                  num_classes: int, ignore_index: int, chunk: int = 16) -> torch.Tensor:
    """(cells, C, C) int32 confusion of every cell's glimpse (inner square only, as pasted)."""
    padded = grid.pad_frame(frame)
    canvas = grid.to_canvas(torch.zeros(grid.frame_size, dtype=torch.long, device=frame.device))
    with torch.no_grad():
        for start in range(0, grid.num_cells, chunk):
            idx = torch.arange(start, min(start + chunk, grid.num_cells), device=frame.device)
            paste_labels(canvas, glimpse_labels(backbone, gather_crops(padded, grid, idx)), grid, idx)
    h, w = grid.frame_size
    return per_cell_confusion(canvas[:h, :w], label, grid, num_classes, ignore_index)


def cache_path(root: Path, split: str, kind: str, model: str, grid: GlimpseGrid, scale: Optional[float] = None) -> Path:
    g = f"g{grid.size}m{grid.margin}"
    if kind == "context":
        return Path(root) / split / f"ctx_{model}_s{scale:g}_{g}.npz"
    if kind == "glimpse":
        return Path(root) / split / f"gl_{model}_{g}.npz"
    raise ValueError(f"kind must be 'context' or 'glimpse', got {kind!r}")


def build_cache(dataset: Any, kind: str, backbone: torch.nn.Module, model_name: str, grid: GlimpseGrid,
                out_path: Path, device: Any, scale: Optional[float] = None,
                clip_indices: Optional[Sequence[int]] = None, precision: str = "fp32",
                allow_any_frame_size: bool = False, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """Run one pass over every labelled frame of ``clip_indices`` and write the cache npz.

    The file is written as ``<name>.tmp.npz`` and renamed when complete, so an
    interrupted build never leaves a file that looks finished.
    """
    if kind not in ("context", "glimpse"):
        raise ValueError(f"kind must be 'context' or 'glimpse', got {kind!r}")
    if kind == "context" and scale is None:
        raise ValueError("a context cache needs its scale")
    if not allow_any_frame_size and tuple(grid.frame_size) != FRAME:
        raise ValueError(f"grid frame {grid.frame_size[0]}x{grid.frame_size[1]}: the T4 latency tables are for 1024x2048")
    dtype = PRECISIONS[precision]
    if dtype != torch.float32:
        backbone = copy.deepcopy(backbone)  # don't convert the caller's model in place
    backbone = backbone.to(device=device, dtype=dtype).eval()
    c, ignore = dataset.num_classes, dataset.ignore_index
    indices = range(len(dataset)) if clip_indices is None else clip_indices
    conf, clip_index, ts, q = [], [], [], 0
    sums: Dict[str, list] = {k: [] for k in SCORE_KINDS}
    tops: Dict[str, list] = {k: [] for k in SCORE_KINDS}
    for ci in indices:
        for t in np.flatnonzero(np.asarray(dataset.labeled_mask(ci), dtype=bool)):
            image = dataset.load_frame(ci, int(t))
            if tuple(image.shape[:2]) != tuple(grid.frame_size):
                raise ValueError(f"clip {dataset.clip_id(ci)} frame {t} is {image.shape[0]}x{image.shape[1]}, "
                                 f"the grid is for {grid.frame_size[0]}x{grid.frame_size[1]}"
                                 + ("" if allow_any_frame_size else " (the T4 latency tables are for 1024x2048)"))
            frame = frame_to_tensor(image, device).to(dtype)
            label = torch.from_numpy(np.asarray(dataset.load_label(ci, int(t)), dtype=np.int64)).to(device)
            if kind == "context":
                cells = context_cells(backbone, frame, label, scale, grid, c, ignore)
                conf.append(cells.conf.cpu())
                for k in SCORE_KINDS:
                    sums[k].append(cells.class_sums[k].cpu())
                    tops[k].append(cells.top10[k].cpu())
                q = cells.q
            else:
                conf.append(glimpse_cells(backbone, frame, label, grid, c, ignore).cpu())
            clip_index.append(ci)
            ts.append(int(t))
    meta = {"kind": kind, "model": model_name, "scale": scale, "size": grid.size, "margin": grid.margin,
            "rows": grid.rows, "cols": grid.cols, "frame_size": list(grid.frame_size), "precision": precision,
            "num_classes": c, "ignore_index": ignore, **(extra_meta or {})}
    arrays: Dict[str, Any] = {"conf": torch.stack(conf).numpy(), "clip_index": np.array(clip_index, dtype=np.int32),
                              "t": np.array(ts, dtype=np.int32), "meta": json.dumps(meta, default=str)}
    if kind == "context":
        arrays["q"] = np.int32(q)
        for k in SCORE_KINDS:
            arrays[f"class_sums_{k}"] = torch.stack(sums[k]).numpy()
            arrays[f"top10_{k}"] = torch.stack(tops[k]).numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name[:-len(".npz")] + ".tmp.npz") if out_path.name.endswith(".npz") \
        else out_path.with_name(out_path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out_path)
    return out_path
