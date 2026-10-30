"""A tiny deterministic backbone for tests and smoke runs -- NOT a segmentation model.

``avg_pool(output_stride) -> 1x1 conv`` obeys the ``SegmentationBackbone``
contract with no download, on CPU, in milliseconds.

* Default: the 1x1 conv has seeded random weights (arbitrary predictions).
* ``class_colors`` given ((C, 3) RGB in [0, 1]): the conv implements
  nearest-colour classification, ``logit_c = -||x - colour_c||^2 / T`` up to a
  per-pixel constant (``2 x.c/T - ||c||^2/T``). On synthetic frames painted
  with the class colours (tests/fixtures) it is near-perfect at full
  resolution, which makes it a probe for resize/upsampling bugs in the
  evaluation pipeline: any misalignment shows up as lost IoU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pad_to_multiple

__all__ = ["DummyBackbone"]


class DummyBackbone(nn.Module):
    align_corners = False

    def __init__(
        self,
        num_classes: int,
        output_stride: int = 2,
        class_colors: Optional[Sequence[Sequence[float]]] = None,
        temperature: float = 0.01,
        seed: int = 0,
    ):
        super().__init__()
        if output_stride < 1:
            raise ValueError(f"output_stride must be >= 1, got {output_stride}")
        self.num_classes = int(num_classes)
        self.output_stride = int(output_stride)
        self.proj = nn.Conv2d(3, self.num_classes, kernel_size=1)
        self.mode = "random" if class_colors is None else "palette"
        with torch.no_grad():
            if class_colors is None:
                gen = torch.Generator().manual_seed(int(seed))
                self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=gen))
                self.proj.bias.copy_(torch.randn(self.proj.bias.shape, generator=gen))
            else:
                colors = torch.as_tensor(class_colors, dtype=torch.float32)
                if colors.shape != (self.num_classes, 3):
                    raise ValueError(f"class_colors must be ({self.num_classes}, 3), got {tuple(colors.shape)}")
                if colors.max() > 1.0:
                    colors = colors / 255.0
                self.proj.weight.copy_((2.0 / temperature) * colors.view(self.num_classes, 3, 1, 1))
                self.proj.bias.copy_(-(colors**2).sum(1) / temperature)
        self.weights_info: Dict[str, Any] = {"source": f"dummy:{self.mode}", "seed": int(seed)}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = pad_to_multiple(images, self.output_stride)
        if self.output_stride > 1:
            x = F.avg_pool2d(x, self.output_stride)
        return self.proj(x)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "dummy",
            "mode": self.mode,
            "num_classes": self.num_classes,
            "output_stride": self.output_stride,
            "align_corners": self.align_corners,
            "params": sum(p.numel() for p in self.parameters()),
            "weights": self.weights_info,
        }
