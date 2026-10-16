"""PIDNet (Xu et al., "PIDNet: A Real-time Semantic Segmentation Network
Inspired by PID Controllers", CVPR 2023) behind :class:`SegmentationBackbone`.

The network below is ported from the official implementation
(https://github.com/XuJiacong/PIDNet, ``models/pidnet.py`` and
``models/model_utils.py``) with module/attribute names kept identical, so the
official checkpoints (``PIDNet_S_Cityscapes_val.pt``, ...) load unchanged.
Only inference is needed here, so the wrapper builds the network with
``augment=False`` and drops the checkpoints' auxiliary heads
(``seghead_p``/``seghead_d``) on load.

Official inference conventions reproduced by :class:`PIDNetBackbone`:
RGB input scaled to [0, 1] then ImageNet mean/std (datasets/base_dataset.py,
``input_transform``); logits at stride 8; upsampled with
``align_corners=True`` (configs/default.py ``MODEL.ALIGN_CORNERS``) while the
network internals use ``align_corners=False``. The forward pass needs H, W
divisible by 8, so the wrapper pads (replicate) at the bottom/right.

Original code license (MIT):

    Copyright (c) 2022 Jiacong Xu

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
import torch.nn.functional as F

from .utils import IMAGENET_MEAN, IMAGENET_STD, normalize, pad_to_multiple
from .weights import extract_state_dict, load_checkpoint, load_into, resolve_weights, sha256_of

__all__ = ["PIDNet", "PIDNetBackbone", "PIDNET_VARIANTS"]

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False  # align_corners inside the network

#: (m, n, planes, ppm_planes, head_planes) of models/pidnet.py::get_pred_model
PIDNET_VARIANTS: Dict[str, Dict[str, int]] = {
    "s": dict(m=2, n=3, planes=32, ppm_planes=96, head_planes=128),
    "m": dict(m=2, n=3, planes=64, ppm_planes=96, head_planes=128),
    "l": dict(m=3, n=4, planes=64, ppm_planes=112, head_planes=256),
}


# --------------------------------------------------------------------------- #
# Building blocks (models/model_utils.py)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)


class segmenthead(nn.Module):  # noqa: N801  (official name, keeps checkpoint keys)
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode="bilinear", align_corners=algc)
        return out


def _bn_relu_conv(cin, cout, **conv_kw):
    return nn.Sequential(BatchNorm2d(cin, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(cin, cout, bias=False, **conv_kw))


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale0 = _bn_relu_conv(inplanes, branch_planes, kernel_size=1)
        self.process1 = _bn_relu_conv(branch_planes, branch_planes, kernel_size=3, padding=1)
        self.process2 = _bn_relu_conv(branch_planes, branch_planes, kernel_size=3, padding=1)
        self.process3 = _bn_relu_conv(branch_planes, branch_planes, kernel_size=3, padding=1)
        self.process4 = _bn_relu_conv(branch_planes, branch_planes, kernel_size=3, padding=1)
        self.compression = _bn_relu_conv(branch_planes * 5, outplanes, kernel_size=1)
        self.shortcut = _bn_relu_conv(inplanes, outplanes, kernel_size=1)

    def forward(self, x):
        size = [x.shape[-2], x.shape[-1]]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=algc)  # noqa: E731
        x_list = [self.scale0(x)]
        x_list.append(self.process1(up(self.scale1(x)) + x_list[0]))
        x_list.append(self.process2(up(self.scale2(x)) + x_list[1]))
        x_list.append(self.process3(up(self.scale3(x)) + x_list[2]))
        x_list.append(self.process4(up(self.scale4(x)) + x_list[3]))
        return self.compression(torch.cat(x_list, 1)) + self.shortcut(x)


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), *_bn_relu_conv(inplanes, branch_planes, kernel_size=1))
        self.scale0 = _bn_relu_conv(inplanes, branch_planes, kernel_size=1)
        self.scale_process = _bn_relu_conv(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4)
        self.compression = _bn_relu_conv(branch_planes * 5, outplanes, kernel_size=1)
        self.shortcut = _bn_relu_conv(inplanes, outplanes, kernel_size=1)

    def forward(self, x):
        size = [x.shape[-2], x.shape[-1]]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=algc)  # noqa: E731
        x_ = self.scale0(x)
        scale_list = [up(self.scale1(x)) + x_, up(self.scale2(x)) + x_, up(self.scale3(x)) + x_, up(self.scale4(x)) + x_]
        scale_out = self.scale_process(torch.cat(scale_list, 1))
        return self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), BatchNorm(mid_channels))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), BatchNorm(mid_channels))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False), BatchNorm(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = F.interpolate(self.f_y(y), size=[input_size[2], input_size[3]], mode="bilinear", align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode="bilinear", align_corners=False)
        return (1 - sim_map) * x + sim_map * y


class Light_Bag(nn.Module):  # noqa: N801
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv_p((1 - edge_att) * i + p) + self.conv_i(i + edge_att * p)


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(
            BatchNorm(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


# --------------------------------------------------------------------------- #
# Network (models/pidnet.py)
# --------------------------------------------------------------------------- #
class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super().__init__()
        self.augment = augment
        # I branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        # P branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom)
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom)
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        # D branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes, momentum=bn_mom)
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom)
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom)
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False), BatchNorm2d(planes * 2, momentum=bn_mom)
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        # Prediction heads
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)
        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        size = [height_output, width_output]

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x), size=size, mode="bilinear", align_corners=algc)
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x), size=size, mode="bilinear", align_corners=algc)
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)), size=size, mode="bilinear", align_corners=algc)

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            return [self.seghead_p(temp_p), x_, self.seghead_d(temp_d)]
        return x_


# --------------------------------------------------------------------------- #
# SegmentationBackbone wrapper
# --------------------------------------------------------------------------- #
class PIDNetBackbone(nn.Module):
    """PIDNet-{S,M,L} for inference behind the ``SegmentationBackbone`` contract."""

    output_stride = 8
    align_corners = True  # official eval: MODEL.ALIGN_CORNERS = True
    size_divisor = 8  # forward() needs H, W divisible by 8
    #: training-only heads present in official checkpoints, dropped on load
    aux_prefixes = ("seghead_p.", "seghead_d.")

    def __init__(
        self,
        variant: str = "s",
        num_classes: int = 19,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        super().__init__()
        if variant not in PIDNET_VARIANTS:
            raise ValueError(f"unknown PIDNet variant {variant!r}; choose from {sorted(PIDNET_VARIANTS)}")
        self.variant = variant
        self.num_classes = int(num_classes)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.net = PIDNet(num_classes=self.num_classes, augment=False, **PIDNET_VARIANTS[variant])
        self.weights_info: Dict[str, Any] = {"source": None}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = pad_to_multiple(normalize(images, self.mean, self.std), self.size_divisor)
        return self.net(x)

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
        report = load_into(self.net, state, strict=strict, ignore_prefixes=self.aux_prefixes + ("sem_loss.", "bd_loss."))
        self.weights_info = {"source": str(source), "file": path.name, "sha256": sha256_of(path), "load": report}
        return report

    def describe(self) -> Dict[str, Any]:
        return {
            "type": f"pidnet_{self.variant}",
            "num_classes": self.num_classes,
            "output_stride": self.output_stride,
            "align_corners": self.align_corners,
            "normalization": {"mean": list(self.mean), "std": list(self.std)},
            "params": sum(p.numel() for p in self.parameters()),
            "weights": self.weights_info,
        }
