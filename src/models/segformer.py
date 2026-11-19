"""SegFormer (Xie et al., NeurIPS 2021) behind :class:`SegmentationBackbone`.

A larger, frozen model of a different family (a hierarchical transformer
encoder with an all-MLP decoder) than PIDNet-S and BiSeNetV2. It is here to
test whether a warp has more room on a model that costs more per pixel and
whose accuracy may depend more on resolution, still judged at equal measured
latency against every model's full-frame curve.

The network is the Hugging Face ``transformers`` implementation
(``SegformerForSemanticSegmentation``), built from the variant table below so
that constructing it never downloads anything. Weights are the public
Cityscapes checkpoints of NVIDIA on the Hugging Face hub
(``nvidia/segformer-b{0..5}-finetuned-cityscapes-1024-1024``), or a local
directory / state-dict file of the same model; nothing is trained here.

Inference conventions of those checkpoints (``SegformerImageProcessor``): RGB
normalised with the ImageNet mean / std; logits come out at a quarter of the
input (``output_stride = 4``) and are upsampled bilinearly with
``align_corners=False``. Any input size works (no positional embedding); the
wrapper still pads to a multiple of 32 so every stage sees whole cells, and
crops the logits back to ``ceil(H / 4) x ceil(W / 4)``.

Note: NVIDIA evaluated these checkpoints with a sliding 1024 x 1024 window;
this harness runs every model on the whole (resized) frame, so the published
mIoU is not expected exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from .utils import IMAGENET_MEAN, IMAGENET_STD, normalize, pad_to_multiple
from .weights import WeightsError, extract_state_dict, load_checkpoint, load_into, sha256_of

__all__ = ["SegFormerBackbone", "SEGFORMER_VARIANTS", "CITYSCAPES_TRAIN_NAMES", "segformer_hub_id"]

# Mix Transformer encoders mit-b0..b5 (paper, Table 6) and their decoder width.
SEGFORMER_VARIANTS: Dict[str, Dict[str, Any]] = {
    "b0": {"depths": [2, 2, 2, 2], "hidden_sizes": [32, 64, 160, 256], "decoder_hidden_size": 256},
    "b1": {"depths": [2, 2, 2, 2], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 256},
    "b2": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
    "b3": {"depths": [3, 4, 18, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
    "b4": {"depths": [3, 8, 27, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
    "b5": {"depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768},
}

# Cityscapes train ids 0..18, the label order of the NVIDIA checkpoints and of this project.
CITYSCAPES_TRAIN_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
)


def segformer_hub_id(variant: str) -> str:
    """The public Cityscapes checkpoint of a variant on the Hugging Face hub."""
    return f"nvidia/segformer-{variant}-finetuned-cityscapes-1024-1024"


def _hf():
    try:
        import transformers  # noqa: F401
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("SegFormer needs the `transformers` package: pip install transformers") from exc
    return SegformerConfig, SegformerForSemanticSegmentation


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


class SegFormerBackbone(nn.Module):
    """SegFormer-B{0..5} for inference behind the ``SegmentationBackbone`` contract."""

    output_stride = 4
    align_corners = False
    size_divisor = 32

    def __init__(self, variant: str = "b2", num_classes: int = 19,
                 mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
        super().__init__()
        variant = str(variant).lower()
        if variant not in SEGFORMER_VARIANTS:
            raise ValueError(f"unknown SegFormer variant {variant!r}; known: {sorted(SEGFORMER_VARIANTS)}")
        self.variant = variant
        self.num_classes = int(num_classes)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.register_buffer("_mean", torch.tensor(self.mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self.std).view(1, -1, 1, 1), persistent=False)
        SegformerConfig, SegformerForSemanticSegmentation = _hf()
        config = SegformerConfig(num_labels=self.num_classes, **SEGFORMER_VARIANTS[variant])
        self.net = SegformerForSemanticSegmentation(config)
        self.weights_info: Dict[str, Any] = {"source": None}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        x = pad_to_multiple(normalize(images, self._mean, self._std), self.size_divisor)
        logits = self.net(pixel_values=x).logits
        s = self.output_stride
        return logits[..., : -(-h // s), : -(-w // s)]

    def load_weights(
        self,
        source: str,
        cache_dir: Optional[str] = None,
        sha256: Optional[str] = None,
        strict: bool = True,
        trust_pickle: bool = False,
    ) -> Dict[str, Any]:
        """``source``: a hub id (``nvidia/segformer-b2-finetuned-cityscapes-1024-1024``), a local
        ``save_pretrained`` directory, or a state-dict file (.pt / .pth / .bin / .safetensors)."""
        source = str(source)
        path = Path(source).expanduser()
        info: Dict[str, Any] = {"source": source}
        if path.is_file():
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(str(path))
            else:
                state = extract_state_dict(load_checkpoint(path, trust_pickle=trust_pickle), strip_prefixes=("module.",))
            info.update({"file": path.name, "sha256": sha256_of(path)})
            if sha256 and info["sha256"] != sha256.lower():
                raise WeightsError(f"sha256 mismatch for {path}: expected {sha256}, got {info['sha256']}")
        else:
            if sha256:
                raise WeightsError("sha256 pins a file; for a hub id pin the revision instead (`source@revision`)")
            _, SegformerForSemanticSegmentation = _hf()
            repo, _, revision = source.partition("@")
            try:
                hf = SegformerForSemanticSegmentation.from_pretrained(
                    repo, revision=revision or None, cache_dir=cache_dir)
            except OSError as exc:
                raise WeightsError(f"could not load SegFormer weights {source!r}: {exc}") from exc
            names = [hf.config.id2label[i] for i in range(hf.config.num_labels)]
            if [_norm_name(n) for n in names] != [_norm_name(n) for n in CITYSCAPES_TRAIN_NAMES[: len(names)]] \
                    or len(names) != self.num_classes:
                raise WeightsError(f"{source}: labels {names} are not the {self.num_classes} Cityscapes train ids "
                                   "in order; the predictions would be scored against the wrong classes")
            state = hf.state_dict()
            info.update({"revision": getattr(hf.config, "_commit_hash", None) or revision or None,
                         "labels": "cityscapes train ids 0..18 (checked)"})
            del hf
        report = load_into(self.net, state, strict=strict)
        self.weights_info = {**info, "load": report}
        return report

    def describe(self) -> Dict[str, Any]:
        return {
            "type": f"segformer_{self.variant}",
            "num_classes": self.num_classes,
            "output_stride": self.output_stride,
            "align_corners": self.align_corners,
            "normalization": {"mean": list(self.mean), "std": list(self.std)},
            "params": sum(p.numel() for p in self.parameters()),
            "weights": self.weights_info,
        }
