"""BiSeNetV2 (Yu et al., IJCV 2021) behind :class:`SegmentationBackbone`.

Ported from the widely used implementation by CoinCheung
(https://github.com/CoinCheung/BiSeNet, ``lib/models/bisenetv2.py``), whose
Cityscapes checkpoint ``model_final_v2_city.pth`` is published on that repo's
GitHub releases. Module names are kept identical so the checkpoint loads
unchanged; the auxiliary training heads (``aux*``) are dropped on load. Unlike
the original, constructing the model never downloads anything (the original
``init_weights`` fetches an ImageNet backbone).

Inference conventions of the original (tools/demo.py): RGB in [0, 1]
normalised with the mean/std of the training config (Cityscapes:
``configs/bisenetv2_city.py`` mean (0.3257, 0.3690, 0.3223), std (0.2112,
0.2148, 0.2115) -- set them in the backbone config); H, W must be multiples
of 32 (the wrapper pads instead of resizing); the head already upsamples x8,
so logits come out at input resolution (``output_stride = 1``).

Original code license (MIT):

    Copyright (c) 2018 CoinCheung

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from .utils import IMAGENET_MEAN, IMAGENET_STD, normalize, pad_to_multiple
from .weights import extract_state_dict, load_checkpoint, load_into, resolve_weights, sha256_of

__all__ = ["BiSeNetV2", "BiSeNetV2Backbone"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(ConvBNReLU(3, 64, 3, stride=2), ConvBNReLU(64, 64, 3, stride=1))
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2), ConvBNReLU(64, 64, 3, stride=1), ConvBNReLU(64, 64, 3, stride=1)
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2), ConvBNReLU(128, 128, 3, stride=1), ConvBNReLU(128, 128, 3, stride=1)
        )

    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(ConvBNReLU(16, 8, 1, stride=1, padding=0), ConvBNReLU(8, 16, 3, stride=2))
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        return self.fuse(torch.cat([self.left(feat), self.right(feat)], dim=1))


class CEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = self.conv_gap(self.bn(torch.mean(x, dim=(2, 3), keepdim=True)))
        return self.conv_last(feat + x)


class GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_chan)
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv2(self.dwconv(self.conv1(x)))
        return self.relu(feat + x)


class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_chan)
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv2(self.dwconv2(self.dwconv1(self.conv1(x))))
        return self.relu(feat + self.shortcut(x))


class SegmentBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
        self.S4 = nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64))
        self.S5_4 = nn.Sequential(GELayerS2(64, 128), GELayerS1(128, 128), GELayerS1(128, 128), GELayerS1(128, 128))
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.right1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128))
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.up1(self.right1(x_s))
        right2 = self.right2(x_s)
        left = left1 * torch.sigmoid(right1)
        right = self.up2(left2 * torch.sigmoid(right2))
        return self.conv(left + right)


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(nn.Upsample(scale_factor=2), ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, n_classes, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        return self.conv_out(self.drop(self.conv(x)))


class BiSeNetV2(nn.Module):
    def __init__(self, n_classes, aux_mode="train"):
        super().__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == "train":
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        self.init_weights()

    def forward(self, x):
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        logits = self.head(self.bga(feat_d, feat_s))
        if self.aux_mode == "train":
            return logits, self.aux2(feat2), self.aux3(feat3), self.aux4(feat4), self.aux5_4(feat5_4)
        if self.aux_mode == "eval":
            return (logits,)
        if self.aux_mode == "pred":
            return logits.argmax(dim=1)
        raise NotImplementedError(self.aux_mode)

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if getattr(module, "last_bn", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # (the original also downloads ImageNet backbone weights here; we never
        # touch the network in a constructor -- load weights explicitly instead)


class BiSeNetV2Backbone(nn.Module):
    """BiSeNetV2 for inference behind the ``SegmentationBackbone`` contract."""

    output_stride = 1  # the head upsamples x8 back to the (padded) input size
    align_corners = False
    size_divisor = 32
    aux_prefixes = ("aux2.", "aux3.", "aux4.", "aux5_4.")

    def __init__(self, num_classes: int = 19, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
        super().__init__()
        self.num_classes = int(num_classes)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.net = BiSeNetV2(self.num_classes, aux_mode="eval")
        self.weights_info: Dict[str, Any] = {"source": None}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        x = pad_to_multiple(normalize(images, self.mean, self.std), self.size_divisor)
        return self.net(x)[0][..., :h, :w]

    def load_weights(
        self,
        source: str,
        cache_dir: Optional[str] = None,
        sha256: Optional[str] = None,
        strict: bool = True,
        trust_pickle: bool = False,
    ) -> Dict[str, Any]:
        path = resolve_weights(source, cache_dir=cache_dir, sha256=sha256)
        state = extract_state_dict(load_checkpoint(path, trust_pickle=trust_pickle))
        report = load_into(self.net, state, strict=strict, ignore_prefixes=self.aux_prefixes)
        self.weights_info = {"source": str(source), "file": path.name, "sha256": sha256_of(path), "load": report}
        return report

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "bisenetv2",
            "num_classes": self.num_classes,
            "output_stride": self.output_stride,
            "align_corners": self.align_corners,
            "normalization": {"mean": list(self.mean), "std": list(self.std)},
            "params": sum(p.numel() for p in self.parameters()),
            "weights": self.weights_info,
        }
