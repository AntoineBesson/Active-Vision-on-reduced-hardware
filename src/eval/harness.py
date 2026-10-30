"""Generic evaluation loop: any :class:`VideoSegDataset` x any :class:`FrameMethod`.

The loop is dataset- and method-agnostic:

1. for every selected clip, ``method.reset(...)``, then frames in temporal
   order (only the labelled ones for per-frame methods, every frame for
   temporal methods that declare ``requires_all_frames``);
2. ``method.predict(frame, t, ledger)`` returns an (H, W) map of train ids
   at full resolution and reports its compute to the :class:`CostLedger`;
3. labelled frames are accumulated into one :class:`ConfusionMatrix`.

Method (1) of the study -- the full-frame baseline at a given input scale --
is :class:`FullFrameMethod`. Foveated methods (context pass + glimpses,
with or without scene memory) implement the same protocol, so metrics and
cost come out in the same format.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # Python >= 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from data.base import VideoSegDataset
from models.utils import upsample_logits

from .cost import CostLedger, benchmark_latency, count_flops, measure_peak_memory, skipped
from .metrics import ConfusionMatrix

__all__ = ["FrameMethod", "FullFrameMethod", "evaluate", "measure_latency", "frame_to_tensor"]

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16}


@runtime_checkable
class FrameMethod(Protocol):
    """A segmentation method evaluated frame by frame.

    ``predict(frame, t, ledger)``: ``frame`` is (1, 3, H, W) float in [0, 1]
    on the evaluation device, ``t`` its position in the clip. Returns an
    (H, W) integer tensor of train ids; values outside [0, C) mean
    "unpredicted" and count as errors. Compute is reported to ``ledger``.
    ``latency_fns(frame_size, device)`` returns named zero-argument callables
    that run one frame of the method (for :func:`measure_latency`).
    """

    name: str
    requires_all_frames: bool

    def reset(self, clip: Mapping[str, Any]) -> None: ...

    def predict(self, frame: torch.Tensor, t: int, ledger: CostLedger) -> torch.Tensor: ...

    def describe(self) -> Dict[str, Any]: ...

    def latency_fns(self, frame_size: Tuple[int, int], device: torch.device) -> Dict[str, Callable[[], Any]]: ...


def frame_to_tensor(frame: np.ndarray, device: Any) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1] on ``device``."""
    t = torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True)).to(device)  # PIL arrays are read-only
    return t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


class FullFrameMethod:
    """Method (1): the backbone on the whole frame, resized by ``scale``.

    ``scale < 1`` gives the reduced-resolution full-frame runs used as
    equal-cost baselines: the frame is resized to ``round(scale * H) x
    round(scale * W)`` (bilinear, antialiased when shrinking), segmented, and
    the logits are upsampled back to (H, W) for evaluation.
    """

    name = "full_frame"
    requires_all_frames = False

    def __init__(self, backbone: torch.nn.Module, scale: float = 1.0, precision: str = "fp32", antialias: bool = True):
        if not scale > 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        self.scale = float(scale)
        self.precision = precision
        self.antialias = antialias
        self.dtype = PRECISIONS[precision]
        self.backbone = backbone if self.dtype == torch.float32 else copy.deepcopy(backbone).to(self.dtype)
        self.backbone.eval()
        self._flops: Dict[Tuple[int, int], Optional[int]] = {}

    def input_size(self, height: int, width: int) -> Tuple[int, int]:
        return max(1, int(round(height * self.scale))), max(1, int(round(width * self.scale)))

    def reset(self, clip: Mapping[str, Any]) -> None:
        pass  # stateless

    def _resize(self, frame: torch.Tensor) -> torch.Tensor:
        height, width = frame.shape[-2:]
        size = self.input_size(height, width)
        if size == (height, width):
            return frame
        shrink = size[0] < height or size[1] < width
        return F.interpolate(frame, size=size, mode="bilinear", align_corners=False, antialias=self.antialias and shrink)

    def _segment(self, frame: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        height, width = frame.shape[-2:]
        x = self._resize(frame)
        logits = model(x.to(next(model.parameters()).dtype))
        logits = upsample_logits(
            logits.float(), (height, width), model.output_stride, model.align_corners, input_size=x.shape[-2:]
        )
        return logits.argmax(1)[0]

    def predict(self, frame: torch.Tensor, t: int, ledger: CostLedger) -> torch.Tensor:
        h, w = self.input_size(*frame.shape[-2:])
        if (h, w) not in self._flops:
            x = self._resize(frame).to(self.dtype)
            self._flops[(h, w)] = count_flops(lambda: self.backbone(x))
        with torch.no_grad():
            pred = self._segment(frame, self.backbone)
        ledger.add("backbone", flops=self._flops[(h, w)] or 0, pixels=h * w)
        return pred

    def latency_fns(self, frame_size: Tuple[int, int], device: torch.device) -> Dict[str, Callable[[], Any]]:
        """``backbone``: the network alone at its input size; ``end_to_end``:
        resize + network + upsampling + argmax from a full-resolution frame."""
        height, width = frame_size
        h, w = self.input_size(height, width)
        model = copy.deepcopy(self.backbone).to(device).eval()
        gen = torch.Generator().manual_seed(0)
        x = torch.rand(1, 3, h, w, generator=gen).to(device=device, dtype=self.dtype)
        frame = torch.rand(1, 3, height, width, generator=gen).to(device)
        return {"backbone": lambda: model(x), "end_to_end": lambda: self._segment(frame, model)}

    def resident_bytes(self) -> int:
        """Bytes of weights (parameters + buffers) the method keeps on the device."""
        tensors = list(self.backbone.parameters()) + list(self.backbone.buffers())
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "scale": self.scale, "precision": self.precision, "antialias": self.antialias}


def evaluate(
    dataset: VideoSegDataset,
    method: FrameMethod,
    device: Any = "cpu",
    clip_indices: Optional[Sequence[int]] = None,
    frames: str = "labeled",
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Run ``method`` over the clips and score the labelled frames.

    ``frames="labeled"`` processes only annotated frames (enough for
    per-frame methods); ``"all"`` processes every frame in order (required
    when ``method.requires_all_frames``). Costs are per processed frame.
    Returns ``{"confusion", "ledger", "metrics", "cost", "frame_sizes"
    (most frequent first), "frame_size_counts", ...}``.
    """
    if frames not in ("labeled", "all"):
        raise ValueError(f"frames must be 'labeled' or 'all', got {frames!r}")
    if getattr(method, "requires_all_frames", False) and frames != "all":
        raise ValueError(f"{method.name} is temporal: evaluate it with frames='all'")
    device = torch.device(device)
    indices = list(range(len(dataset))) if clip_indices is None else [int(i) for i in clip_indices]
    confusion = ConfusionMatrix(dataset.num_classes, dataset.ignore_index, dataset.class_names)
    ledger = CostLedger()
    sizes: Counter = Counter()
    frames_processed = 0
    for k, ci in enumerate(indices):
        mask = np.asarray(dataset.labeled_mask(ci), dtype=bool)
        steps = np.flatnonzero(mask) if frames == "labeled" else np.arange(len(mask))
        if len(steps):
            method.reset({"index": ci, "clip_id": dataset.clip_id(ci), "num_frames": int(len(mask))})
        for t in steps:
            t = int(t)
            image = dataset.load_frame(ci, t)
            sizes[tuple(int(s) for s in image.shape[:2])] += 1
            ledger.begin_frame(image.shape[0] * image.shape[1])
            with torch.no_grad():
                pred = method.predict(frame_to_tensor(image, device), t, ledger)
            frames_processed += 1
            if mask[t]:
                label = dataset.load_label(ci, t)
                confusion.update(pred.cpu().numpy(), label)
        if progress is not None:
            progress(k + 1, len(indices))
    return {
        "confusion": confusion,
        "ledger": ledger,
        "metrics": confusion.compute(),
        "cost": ledger.summary(),
        "frame_sizes": [size for size, _ in sizes.most_common()],  # most frequent first
        "frame_size_counts": {f"{h}x{w}": n for (h, w), n in sizes.most_common()},
        "num_clips": len(indices),
        "frames_processed": frames_processed,
    }


def measure_latency(
    method: FrameMethod,
    frame_size: Tuple[int, int],
    device: Any,
    warmup: int = 20,
    iters: int = 100,
    allow_cpu: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(latency, peak memory) of one frame of ``method`` at batch size 1.

    GPU only unless ``allow_cpu`` (CPU numbers are for smoke tests and are
    labelled with their device). Inputs are synthetic tensors of the real
    sizes; cuDNN autotuning is enabled during the measurement. Memory adds
    the method's own weights to the peak increment, so a second resident
    copy of the model (kept for the accuracy pass) does not inflate it.
    """
    device = torch.device(device)
    if device.type != "cuda" and not allow_cpu:
        reason = "CUDA not available" if not torch.cuda.is_available() else f"device is {device}"
        return skipped(f"{reason}: latency is only reported on GPU"), skipped(f"{reason}: peak memory is CUDA-only")
    fns = method.latency_fns(tuple(frame_size), device)
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = device.type == "cuda"
    try:
        latency: Dict[str, Any] = {
            "status": "ok",
            "batch_size": 1,
            "precision": getattr(method, "precision", None),
            "frame_size": list(frame_size),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        for name, fn in fns.items():
            latency[name] = benchmark_latency(fn, device, warmup=warmup, iters=iters)
        memory = measure_peak_memory(fns["end_to_end"], device) if "end_to_end" in fns else skipped("no end_to_end fn")
        resident = getattr(method, "resident_bytes", None)
        if memory.get("status") == "ok" and callable(resident):
            weights_mib = resident() / 2**20
            memory["model_weights_mib"] = weights_mib
            memory["estimated_footprint_mib"] = weights_mib + memory["peak_increment_mib"]
    finally:
        torch.backends.cudnn.benchmark = previous
    return latency, memory
