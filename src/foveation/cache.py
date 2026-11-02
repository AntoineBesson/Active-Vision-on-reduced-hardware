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
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from eval.harness import PRECISIONS, frame_to_tensor

from .ops import SCORE_KINDS, GlimpseGrid, cell_features, gather_crops, paste_labels, samples_per_cell
from .pipeline import (FUSIONS, context_pass, gather_inner, glimpse_labels, glimpse_logits,
                       labels_from_glimpse)

__all__ = ["FRAME", "per_cell_confusion", "ContextCells", "context_cells", "glimpse_cells", "CacheJob", "build_cache", "build_model_caches", "build_caches", "cache_path"]

FRAME = (1024, 2048)  # the frame the T4 latency tables were measured at


def per_cell_confusion(pred: torch.Tensor, label: torch.Tensor, grid: GlimpseGrid, num_classes: int,
                       ignore_index: int) -> torch.Tensor:
    """(cells, C, C) int32 confusion (rows = ground truth) of each cell's in-frame inner square.

    Pixels labelled ``ignore_index`` and predictions outside [0, C) are not
    counted: they go to an extra bin that is dropped, rather than being
    filtered out first, which would synchronise with the host on every call.
    """
    h, w = grid.frame_size
    c = num_classes
    cell = ((torch.arange(h, device=pred.device) // grid.inner)[:, None] * grid.cols
            + (torch.arange(w, device=pred.device) // grid.inner)[None, :])
    t, p = label.long(), pred.long()
    valid = (t != ignore_index) & (t >= 0) & (t < c) & (p >= 0) & (p < c)
    dump = grid.num_cells * c * c
    code = torch.where(valid, (cell * c + t.clamp(0, c - 1)) * c + p.clamp(0, c - 1), dump)
    counts = torch.bincount(code.flatten(), minlength=dump + 1)[:dump]
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
        logits, labels, _ = context_pass(backbone, frame, scale)
    return _context_cells_from(logits, labels, label, grid, num_classes, ignore_index, kinds)


def _context_cells_from(logits: torch.Tensor, labels: torch.Tensor, label: torch.Tensor, grid: GlimpseGrid,
                        num_classes: int, ignore_index: int, kinds: Sequence[str]) -> ContextCells:
    with torch.no_grad():
        feats = cell_features(logits, grid, kinds)
    return ContextCells(conf=per_cell_confusion(labels, label, grid, num_classes, ignore_index),
                        class_sums={k: v[0].float() for k, v in feats.items()},
                        top10={k: v[1].float() for k, v in feats.items()},
                        q=samples_per_cell(logits.shape[-2], grid))


def glimpse_cells(backbone: torch.nn.Module, frame: torch.Tensor, label: torch.Tensor, grid: GlimpseGrid,
                  num_classes: int, ignore_index: int, chunk: int = 16, input_size: Optional[int] = None) -> torch.Tensor:
    """(cells, C, C) int32 confusion of every cell's glimpse (inner square only, as pasted)."""
    padded = grid.pad_frame(frame)
    canvas = grid.to_canvas(torch.zeros(grid.frame_size, dtype=torch.long, device=frame.device))
    with torch.no_grad():
        for start in range(0, grid.num_cells, chunk):
            idx = torch.arange(start, min(start + chunk, grid.num_cells), device=frame.device)
            paste_labels(canvas, glimpse_labels(backbone, gather_crops(padded, grid, idx), input_size), grid, idx)
    h, w = grid.frame_size
    return per_cell_confusion(canvas[:h, :w], label, grid, num_classes, ignore_index)


def _tmp(path: Path, what: str) -> Path:
    """A disk-backed array (``conf``, ``sums``, ``tops``) of a cache being built (deleted when the file is written)."""
    path = Path(path)
    return path.with_name(path.name[:-len(".npz")] + f".{what}.tmp.npy")


def _conf_tmp(path: Path) -> Path:
    return _tmp(path, "conf")


def _state_path(jobs: Sequence[CacheJob]) -> Path:
    """The checkpoint of an interrupted :func:`build_caches`, one per set of output files."""
    key = hashlib.sha1("|".join(str(Path(j.path)) for j in jobs).encode("utf-8")).hexdigest()[:12]
    return Path(jobs[0].path).parent / f"_build_state_{key}.json"


def _signature(jobs: Sequence[CacheJob], indices: Sequence[int], precision: str, num_frames: int) -> str:
    return json.dumps({"jobs": [[j.kind, j.model, str(Path(j.path)), j.scale, j.grid.size, j.grid.margin,
                                 list(j.grid.frame_size), j.input_size, j.fusion, j.ctx_model] for j in jobs],
                       "clips": [int(i) for i in indices], "precision": precision, "frames": num_frames})


def _shapes(job: CacheJob, num_frames: int, c: int) -> Dict[str, Tuple[int, ...]]:
    cells, k = job.grid.num_cells, len(SCORE_KINDS)
    shapes = {"conf": (num_frames, cells, c, c)}
    if job.kind == "context":
        shapes.update(sums=(k, num_frames, cells, c), tops=(k, num_frames, cells))
    return shapes


def _open_accumulators(jobs: Sequence[CacheJob], num_frames: int, c: int, signature: str) -> Tuple[List, Dict]:
    """Disk-backed accumulators, reopened with the checkpoint's progress when it is of these very jobs."""
    state_path = _state_path(jobs)
    state = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except ValueError:
            state = None
    if state is not None and state.get("signature") != signature:
        state = None
    if state is not None:
        for j in jobs:
            for what, shape in _shapes(j, num_frames, c).items():
                f = _tmp(j.path, what)
                if not f.exists() or tuple(np.load(f, mmap_mode="r").shape) != shape:
                    state = None
    mode = "r+" if state is not None else "w+"
    acc = []
    for j, q in zip(jobs, (state or {}).get("q", [0] * len(jobs))):
        Path(j.path).parent.mkdir(parents=True, exist_ok=True)
        a = {what: np.lib.format.open_memmap(_tmp(j.path, what), mode=mode,
                                             dtype=np.int32 if what == "conf" else np.float32, shape=shape)
             for what, shape in _shapes(j, num_frames, c).items()}
        a["q"] = q
        acc.append(a)
    if state is None:
        state = {"signature": signature, "clips_done": 0, "clip_index": [], "t": []}
    return acc, state


def _checkpoint(jobs: Sequence[CacheJob], acc: List, state: Dict) -> None:
    for a in acc:
        for what in ("conf", "sums", "tops"):
            if what in a:
                a[what].flush()
    state["q"] = [int(a["q"]) for a in acc]
    path = _state_path(jobs)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class CacheJob:
    """One cache file.

    ``context``: per-cell confusion + selection features of ``model``'s context
    pass at ``scale``. ``glimpse``: per-cell confusion of ``model``'s glimpse of
    every cell (network input ``input_size``, default the crop size), either on
    its own (``fusion="replace"``) or fused with the context of ``ctx_model``
    at ``scale``.
    """

    kind: str
    model: str
    grid: GlimpseGrid
    path: Path
    scale: Optional[float] = None
    input_size: Optional[int] = None
    fusion: str = "replace"
    ctx_model: Optional[str] = None

    @property
    def ctx_key(self) -> Optional[Tuple[str, float]]:
        if self.kind == "context":
            return (self.model, self.scale)
        return None if self.fusion == "replace" else (self.ctx_model, self.scale)


def cache_path(root: Path, split: str, kind: str, model: str, grid: GlimpseGrid, scale: Optional[float] = None,
               input_size: Optional[int] = None, fusion: str = "replace", ctx_model: Optional[str] = None) -> Path:
    g = f"g{grid.size}m{grid.margin}"
    if kind == "context":
        return Path(root) / split / f"ctx_{model}_s{scale:g}_{g}.npz"
    if kind != "glimpse":
        raise ValueError(f"kind must be 'context' or 'glimpse', got {kind!r}")
    if input_size is not None and input_size != grid.size:
        g += f"_i{input_size}"
    if fusion == "replace":
        return Path(root) / split / f"gl_{model}_{g}.npz"
    return Path(root) / split / f"fu_{ctx_model}_s{scale:g}_{model}_{g}_{fusion}.npz"


Job = Tuple[str, GlimpseGrid, Optional[float], Path]  # (kind, grid, context scale or None, output path)


def build_cache(dataset: Any, kind: str, backbone: torch.nn.Module, model_name: str, grid: GlimpseGrid,
                out_path: Path, device: Any, scale: Optional[float] = None,
                clip_indices: Optional[Sequence[int]] = None, precision: str = "fp32",
                allow_any_frame_size: bool = False, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """One cache file: see :func:`build_caches`."""
    return build_model_caches(dataset, backbone, model_name, [(kind, grid, scale, Path(out_path))], device,
                              clip_indices=clip_indices, precision=precision,
                              allow_any_frame_size=allow_any_frame_size, extra_meta=extra_meta)[0]


def build_model_caches(dataset: Any, backbone: torch.nn.Module, model_name: str, jobs: Sequence[Job], device: Any,
                       **kwargs: Any) -> List[Path]:
    """Context and plain glimpse caches of one backbone: see :func:`build_caches`."""
    return build_caches(dataset, {model_name: backbone},
                        [CacheJob(kind, model_name, grid, Path(path), scale) for kind, grid, scale, path in jobs],
                        device, **kwargs)


def build_caches(dataset: Any, backbones: Dict[str, torch.nn.Module], jobs: Sequence[CacheJob], device: Any,
                 clip_indices: Optional[Sequence[int]] = None, precision: str = "fp32",
                 allow_any_frame_size: bool = False, extra_meta: Optional[Dict[str, Any]] = None,
                 progress: bool = False, chunk: int = 16) -> List[Path]:
    """Every cache of ``jobs`` in a single pass over the labelled frames of ``clip_indices``.

    Each frame is loaded once; each (context model, scale) runs once and serves
    every grid and fusion; each (glimpse model, grid, input size) runs once and
    serves every fusion. Labels come from the pipeline's own functions. Each
    file is written as ``<name>.tmp.npz`` and renamed when complete, so an
    interrupted build never leaves a file that looks finished. Progress is
    checkpointed after every clip: called again with the same jobs and clips,
    an interrupted build resumes after its last finished clip.
    """
    for job in jobs:
        if job.kind not in ("context", "glimpse"):
            raise ValueError(f"kind must be 'context' or 'glimpse', got {job.kind!r}")
        if job.kind == "context" and job.scale is None:
            raise ValueError("a context cache needs its scale")
        if job.fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {job.fusion!r}")
        if job.kind == "glimpse" and job.fusion != "replace" and (job.ctx_model is None or job.scale is None):
            raise ValueError("a fused glimpse cache needs ctx_model and scale")
        if not allow_any_frame_size and tuple(job.grid.frame_size) != FRAME:
            raise ValueError(f"grid frame {job.grid.frame_size[0]}x{job.grid.frame_size[1]}: "
                             f"the T4 latency tables are for 1024x2048")
    dtype = PRECISIONS[precision]
    nets = {}
    for name in {j.model for j in jobs} | {j.ctx_model for j in jobs if j.ctx_model}:
        net = backbones[name] if dtype == torch.float32 else copy.deepcopy(backbones[name])  # caller's model untouched
        nets[name] = net.to(device=device, dtype=dtype).eval()
    c, ignore = dataset.num_classes, dataset.ignore_index
    indices = list(range(len(dataset)) if clip_indices is None else clip_indices)
    frame_size = tuple(jobs[0].grid.frame_size)
    num_frames = sum(int(np.asarray(dataset.labeled_mask(ci), dtype=bool).sum()) for ci in indices)
    # per-cell arrays go to disk as they are computed: all jobs of a train split do not fit in RAM
    acc, state = _open_accumulators(jobs, num_frames, c, _signature(jobs, indices, precision, num_frames))
    ctx_keys = sorted({j.ctx_key for j in jobs if j.ctx_key is not None})
    fused_keys = {j.ctx_key for j in jobs if j.kind == "glimpse" and j.fusion != "replace"}
    groups: Dict[Tuple, List[int]] = {}
    for i, j in enumerate(jobs):
        if j.kind == "glimpse":
            groups.setdefault((j.model, j.grid.size, j.grid.margin, j.input_size), []).append(i)
    clip_index, ts = state["clip_index"], state["t"]
    for n, ci in enumerate(indices):
        if n < state["clips_done"]:
            continue
        for t in np.flatnonzero(np.asarray(dataset.labeled_mask(ci), dtype=bool)):
            image = dataset.load_frame(ci, int(t))
            if tuple(image.shape[:2]) != frame_size:
                raise ValueError(f"clip {dataset.clip_id(ci)} frame {t} is {image.shape[0]}x{image.shape[1]}, "
                                 f"the grid is for {frame_size[0]}x{frame_size[1]}"
                                 + ("" if allow_any_frame_size else " (the T4 latency tables are for 1024x2048)"))
            frame = frame_to_tensor(image, device).to(dtype)
            label = torch.from_numpy(np.asarray(dataset.load_label(ci, int(t)), dtype=np.int64)).to(device)
            with torch.no_grad():
                ctx, ctx_probs = {}, {}
                for key in ctx_keys:
                    logits, labels, up = context_pass(nets[key[0]], frame, key[1])
                    ctx[key] = (logits, labels)
                    if key in fused_keys:  # softmax once per frame, shared by every grid and fusion
                        probs = torch.softmax(up.float(), 1)
                        top, arg = probs.max(1, keepdim=True)
                        ctx_probs[key] = (probs, top, arg)
                    del up
                for i, j in enumerate(jobs):
                    if j.kind != "context":
                        continue
                    cells = _context_cells_from(*ctx[j.ctx_key][:2], label, j.grid, c, ignore, SCORE_KINDS)
                    acc[i]["conf"][len(ts)] = cells.conf.cpu().numpy()
                    for ki, k in enumerate(SCORE_KINDS):
                        acc[i]["sums"][ki, len(ts)] = cells.class_sums[k].float().cpu().numpy()
                        acc[i]["tops"][ki, len(ts)] = cells.top10[k].float().cpu().numpy()
                    acc[i]["q"] = cells.q
                for (model, _, _, input_size), members in groups.items():
                    grid = jobs[members[0]].grid
                    padded = grid.pad_frame(frame)
                    canvases = {i: grid.to_canvas(torch.zeros(frame_size, dtype=torch.long, device=frame.device))
                                for i in members}
                    for start in range(0, grid.num_cells, chunk):
                        idx = torch.arange(start, min(start + chunk, grid.num_cells), device=frame.device)
                        up = glimpse_logits(nets[model], gather_crops(padded, grid, idx), input_size)
                        base = labels_from_glimpse(up)
                        m, inner = grid.margin, grid.inner
                        if any(jobs[i].fusion != "replace" for i in members):
                            # the glimpse's softmax once per chunk, shared by every fused job of the group
                            pg = torch.softmax(up[:, :, m:m + inner, m:m + inner].float(), 1)
                            pg_top, pg_arg = pg.max(1)
                        for i in members:
                            j = jobs[i]
                            labels = base
                            if j.fusion != "replace":
                                probs, top, arg = ctx_probs[j.ctx_key]
                                labels = base.clone()
                                if j.fusion == "mean":  # == fuse_labels(..., "mean") on the gathered logits
                                    fused = (pg + gather_inner(probs, grid, idx)).argmax(1)
                                else:  # "conf": compare the top probabilities only
                                    fused = torch.where(pg_top >= gather_inner(top, grid, idx)[:, 0], pg_arg,
                                                        gather_inner(arg, grid, idx)[:, 0])
                                labels[:, m:m + inner, m:m + inner] = fused
                            paste_labels(canvases[i], labels, grid, idx)
                    h, w = frame_size
                    for i in members:
                        acc[i]["conf"][len(ts)] = per_cell_confusion(canvases[i][:h, :w], label, grid, c,
                                                                     ignore).cpu().numpy()
            clip_index.append(int(ci))
            ts.append(int(t))
        state["clips_done"] = n + 1
        _checkpoint(jobs, acc, state)
        if progress:
            print(f"\r[cache] clip {n + 1}/{len(indices)}", end="", flush=True)
    if progress:
        print()
    written = []
    for j, a in zip(jobs, acc):
        grid = j.grid
        meta = {"kind": j.kind, "model": j.model, "scale": j.scale, "size": grid.size, "margin": grid.margin,
                "rows": grid.rows, "cols": grid.cols, "frame_size": list(grid.frame_size), "precision": precision,
                "num_classes": c, "ignore_index": ignore, **(extra_meta or {})}
        if j.kind == "glimpse":
            meta.update(input_size=j.input_size or grid.size, fusion=j.fusion, ctx_model=j.ctx_model)
        arrays: Dict[str, Any] = {"conf": a["conf"],
                                  "clip_index": np.array(clip_index, dtype=np.int32),
                                  "t": np.array(ts, dtype=np.int32), "meta": json.dumps(meta, default=str)}
        if j.kind == "context":
            arrays["q"] = np.int32(a["q"])
            for ki, k in enumerate(SCORE_KINDS):
                arrays[f"class_sums_{k}"] = a["sums"][ki]
                arrays[f"top10_{k}"] = a["tops"][ki]
        out_path = Path(j.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name[:-len(".npz")] + ".tmp.npz") if out_path.name.endswith(".npz") \
            else out_path.with_name(out_path.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)  # per-cell confusions are mostly zeros
        os.replace(tmp, out_path)
        del arrays
        for what in ("conf", "sums", "tops"):
            if what in a:
                arr, a[what] = a[what], None
                arr._mmap.close()  # Windows cannot delete a mapped file; nothing else views this array
                del arr
                os.remove(_tmp(j.path, what))
        written.append(out_path)
    os.remove(_state_path(jobs))
    return written
