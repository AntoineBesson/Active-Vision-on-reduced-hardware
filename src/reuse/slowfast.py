"""Method 9: split a frozen two-branch model into a slow part (reused across frames) and a fresh part.

Not black-box: this reads the model's internal tensors and only exists for models whose
architecture has a separable low-resolution context branch (BiSeNetV2, PIDNet). It changes no
weight and trains nothing; with the slow tensors of the same frame it reproduces the model's own
forward exactly (``tests/test_method9_split.py``).

    split = slowfast_split(backbone)       # a PIDNetBackbone or BiSeNetV2Backbone
    logits, cache = split.key(images)      # key frame: the full model, also returns the cache
    logits = split.fresh(images, cache)    # other frames: fresh branch + cached slow tensors

``key(x)`` returns the same logits as ``backbone(x)``. ``slow(x)`` returns only the cache. The cache
is a tuple of tensors at low resolution (see each class); it may come from another frame of the
same size, which is the whole method.

* **BiSeNetV2.** slow = the semantic branch ``segment`` (its last output ``feat_s``, 1/32);
  fresh = ``detail`` (1/8) -> ``bga(feat_d, feat_s)`` -> ``head``.
* **PIDNet.** The shared stem (``conv1``, ``layer1``, ``layer2``) stays fresh. slow = the I branch
  (``layer3``, ``layer4``, ``layer5``, ``spp``) and the four tensors it feeds the other branches
  (``compression3/4``, ``diff3/4``); fresh = stem, P branch with ``pag3/4``, D branch, ``dfm``,
  ``final_layer``. The cache is ``(c3, d3, c4, d4, s5)`` at 1/16, 1/16, 1/32, 1/32, 1/64; the
  upsampling of ``d3``, ``d4``, ``s5`` to 1/8 is done in the fresh pass, as in the original.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from models.utils import normalize, pad_to_multiple

__all__ = ["slowfast_split", "BiSeNetV2Split", "PIDNetSplit"]

Cache = Tuple[torch.Tensor, ...]


class BiSeNetV2Split:
    def __init__(self, backbone):
        self.b = backbone
        self.net = backbone.net

    def _prep(self, images):
        return pad_to_multiple(normalize(images, self.b._mean, self.b._std), self.b.size_divisor)

    def slow(self, images) -> Cache:
        return (self.net.segment(self._prep(images))[-1],)

    def _fresh_x(self, x, cache: Cache, hw):
        feat_d = self.net.detail(x)
        return self.net.head(self.net.bga(feat_d, cache[0]))[..., : hw[0], : hw[1]]

    def fresh(self, images, cache: Cache):
        return self._fresh_x(self._prep(images), cache, images.shape[-2:])

    def key(self, images):
        x = self._prep(images)
        cache = (self.net.segment(x)[-1],)
        return self._fresh_x(x, cache, images.shape[-2:]), cache


class PIDNetSplit:
    def __init__(self, backbone):
        self.b = backbone
        self.net = backbone.net

    def _prep(self, images):
        return pad_to_multiple(normalize(images, self.b._mean, self.b._std), self.b.size_divisor)

    def _stem(self, x):
        n = self.net
        x = n.layer1(n.conv1(x))
        return n.relu(n.layer2(n.relu(x)))

    def _slow_from_stem(self, x) -> Cache:
        n = self.net
        x = n.relu(n.layer3(x))
        c3, d3 = n.compression3(x), n.diff3(x)
        x = n.relu(n.layer4(x))
        c4, d4 = n.compression4(x), n.diff4(x)
        s5 = n.spp(n.layer5(x))
        return c3, d3, c4, d4, s5

    def _fresh_from_stem(self, x, cache: Cache, size):
        from models.pidnet import algc

        n = self.net
        c3, d3, c4, d4, s5 = cache
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=algc)  # noqa: E731
        x_ = n.layer3_(x)
        x_d = n.layer3_d(x)
        x_ = n.pag3(x_, c3)
        x_d = x_d + up(d3)
        x_ = n.layer4_(n.relu(x_))
        x_d = n.layer4_d(n.relu(x_d))
        x_ = n.pag4(x_, c4)
        x_d = x_d + up(d4)
        x_ = n.layer5_(n.relu(x_))
        x_d = n.layer5_d(n.relu(x_d))
        return n.final_layer(n.dfm(x_, up(s5), x_d))

    @staticmethod
    def _size(x):
        return [x.shape[-2] // 8, x.shape[-1] // 8]

    def slow(self, images) -> Cache:
        return self._slow_from_stem(self._stem(self._prep(images)))

    def fresh(self, images, cache: Cache):
        x = self._prep(images)
        return self._fresh_from_stem(self._stem(x), cache, self._size(x))

    def key(self, images):
        x = self._prep(images)
        stem = self._stem(x)
        cache = self._slow_from_stem(stem)
        return self._fresh_from_stem(stem, cache, self._size(x)), cache


def slowfast_split(backbone):
    """The slow/fast split of a supported backbone (raises for any other model: not black-box)."""
    from models.bisenetv2 import BiSeNetV2Backbone
    from models.pidnet import PIDNetBackbone

    if isinstance(backbone, PIDNetBackbone):
        if backbone.net.augment:
            raise ValueError("PIDNet split expects the inference network (augment=False)")
        return PIDNetSplit(backbone)
    if isinstance(backbone, BiSeNetV2Backbone):
        return BiSeNetV2Split(backbone)
    raise TypeError(f"no slow/fast split for {type(backbone).__name__}: Method 9 needs a two-branch model")
