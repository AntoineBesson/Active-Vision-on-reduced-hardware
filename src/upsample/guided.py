"""Method 7: guided upsampling of a frozen model's low-scale logits with the full-resolution frame.

Spec: docs/superpowers/specs/2026-11-20-method7-8-guided-upsampling-scale-oracle-design.md.

The full frame at scale c runs the frozen model on a c-times smaller image and brings its logits
back to the frame with a bilinear upsampling (:func:`models.utils.upsample_logits`). Here the
bilinear step is replaced by a fast guided filter (He and Sun, 2015) whose guide is the
full-resolution frame, so that class boundaries snap to image edges. Training-free and
model-agnostic: it reads the logits and the frame only.

Per logit channel p (a class logit, or its softmax probability), on the logit grid:

    mean_I, mean_p, cov_Ip, var_I  box means of radius r of the guide I sampled on the grid
    a = cov_Ip / (var_I + eps),    b = mean_p - a mean_I                       (gray guide)
    a = (Sigma_I + eps U)^-1 cov_Ip, b = mean_p - a . mean_I                   (rgb guide)
    A, B = box means of a, b;  upsampled to the frame like the logits;  q = A . I + B

and the prediction is argmax over classes of q. The grid-to-frame geometry is the logits' own
(:func:`models.utils.upsample_logits`: the padded input extent, the model's ``align_corners``),
and the guide is sampled on the grid with that same geometry, so a guided filter that returns p
(r = 0, eps -> inf) gives back the bilinear baseline exactly (tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["GUIDES", "TARGETS", "GuidedConfig", "covered_size", "guide_on_grid", "box", "guided_coeffs",
           "guided_logits", "guided_labels", "bilinear_labels"]

GUIDES = ("gray", "rgb")
TARGETS = ("logit", "prob")


@dataclass(frozen=True)
class GuidedConfig:
    scale: float  # the frozen model runs on the frame resized by this, like the full frame at that scale
    guide: str = "gray"  # "gray": mean of R, G, B; "rgb": the three channels (3 x 3 solve per grid cell)
    target: str = "logit"  # what is filtered: the class logits, or their softmax probabilities
    radius: int = 1  # box radius on the logit grid
    eps: float = 1e-3  # regulariser; guide values are in [0, 1]

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 1.0:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if self.guide not in GUIDES:
            raise ValueError(f"guide must be one of {GUIDES}, got {self.guide!r}")
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")
        if self.radius < 0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")
        if not self.eps > 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def name(self) -> str:
        return f"7g/c{self.scale:g}/{self.guide}/{self.target}/r{self.radius}/e{self.eps:g}"

    @classmethod
    def from_name(cls, name: str) -> "GuidedConfig":
        tag, c, guide, target, r, e = name.split("/")
        if tag != "7g":
            raise ValueError(f"not a Method 7 name: {name!r}")
        return cls(float(c[1:]), guide, target, int(r[1:]), float(e[1:]))


def covered_size(logit_hw: Tuple[int, int], frame_hw: Tuple[int, int], output_stride: int,
                 input_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Frame-pixel size of the extent the logit grid covers (the padded input mapped to the frame),
    as :func:`models.utils.upsample_logits` computes it; at least the frame."""
    (gh, gw), (h, w), (ih, iw), s = logit_hw, frame_hw, input_hw, int(output_stride)
    if gh * s < ih or gw * s < iw:
        raise ValueError(f"logits {(gh, gw)} at stride {s} cannot cover input {(ih, iw)}")
    return int(round(gh * s * h / ih)), int(round(gw * s * w / iw))


def _to_frame(x: torch.Tensor, covered: Tuple[int, int], frame_hw: Tuple[int, int], align_corners: bool) -> torch.Tensor:
    """Grid -> frame, the logits' own way (interpolate to the covered extent, crop)."""
    if tuple(x.shape[-2:]) != tuple(covered):
        x = F.interpolate(x, size=covered, mode="bilinear", align_corners=align_corners)
    return x[..., :frame_hw[0], :frame_hw[1]]


def guide_on_grid(guide: torch.Tensor, logit_hw: Tuple[int, int], covered: Tuple[int, int],
                  align_corners: bool) -> torch.Tensor:
    """The (1, k, H, W) full-resolution guide sampled on the logit grid: padded (replicate) to the
    covered extent, then an antialiased bilinear resize with the model's ``align_corners``."""
    h, w = guide.shape[-2:]
    ph, pw = covered[0] - h, covered[1] - w
    if ph or pw:
        guide = F.pad(guide, (0, pw, 0, ph), mode="replicate")
    if tuple(guide.shape[-2:]) == tuple(logit_hw):
        return guide
    shrink = logit_hw[0] < guide.shape[-2] or logit_hw[1] < guide.shape[-1]
    return F.interpolate(guide, size=logit_hw, mode="bilinear", align_corners=align_corners, antialias=shrink)


def box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Box mean of radius ``r`` over the last two dims, normalised by the in-image count at the borders."""
    if r == 0:
        return x
    lead = x.shape[:-2]
    y = F.avg_pool2d(x.reshape(1, -1, *x.shape[-2:]), 2 * r + 1, stride=1, padding=r, count_include_pad=False)
    return y.view(*lead, *y.shape[-2:])


def _guide_full(frame: torch.Tensor, guide: str) -> torch.Tensor:
    return frame.mean(1, keepdim=True) if guide == "gray" else frame


def guided_coeffs(p: torch.Tensor, g: torch.Tensor, radius: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Box-averaged coefficients on the grid. ``p``: (1, C, gh, gw) target, ``g``: (1, k, gh, gw)
    guide (k = 1 or 3). Returns A (1, C, k, gh, gw) and B (1, C, gh, gw)."""
    k = g.shape[1]
    mean_i = box(g, radius)  # (1, k, gh, gw)
    mean_p = box(p, radius)  # (1, C, gh, gw)
    cov_ip = box(p[:, :, None] * g[:, None], radius) - mean_p[:, :, None] * mean_i[:, None]  # (1, C, k, gh, gw)
    if k == 1:
        var = box(g * g, radius) - mean_i * mean_i  # (1, 1, gh, gw)
        a = cov_ip / (var[:, None] + eps)
    else:
        gg = g[:, :, None] * g[:, None]  # (1, 3, 3, gh, gw)
        sigma = box(gg, radius) - mean_i[:, :, None] * mean_i[:, None]
        sigma = sigma + eps * torch.eye(3, dtype=g.dtype, device=g.device)[None, :, :, None, None]
        # per grid cell: a_c = Sigma^-1 cov_c, for every class c at once
        s = sigma.permute(0, 3, 4, 1, 2)  # (1, gh, gw, 3, 3)
        rhs = cov_ip.permute(0, 3, 4, 2, 1)  # (1, gh, gw, 3, C)
        a = torch.linalg.solve(s, rhs).permute(0, 4, 3, 1, 2)  # (1, C, 3, gh, gw)
    b = mean_p - (a * mean_i[:, None]).sum(2)
    return box(a, radius), box(b, radius)


def guided_logits(logits: torch.Tensor, frame: torch.Tensor, cfg: GuidedConfig, output_stride: int,
                  align_corners: bool, input_size: Tuple[int, int],
                  full_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(1, C, gh, gw) logits of the model's pass on the resized frame and the (1, 3, H, W) frame in
    [0, 1] -> (1, C, H, W) guided scores (argmax them for the labels). The coefficients are
    computed in float32 on the grid; the full-resolution steps run in ``full_dtype`` (float16 on
    the T4, as the full frame's own upsampling)."""
    h, w = frame.shape[-2:]
    p = logits.float()
    if cfg.target == "prob":
        p = torch.softmax(p, 1)
    covered = covered_size(tuple(p.shape[-2:]), (h, w), output_stride, input_size)
    guide = _guide_full(frame.float(), cfg.guide)
    g = guide_on_grid(guide, tuple(p.shape[-2:]), covered, align_corners)
    a, b = guided_coeffs(p, g, cfg.radius, cfg.eps)
    a, b, guide = a.to(full_dtype), b.to(full_dtype), guide.to(full_dtype)
    q = _to_frame(b, covered, (h, w), align_corners)
    for j in range(a.shape[2]):  # one guide channel at a time: one (1, C, H, W) temporary
        q = q + _to_frame(a[:, :, j], covered, (h, w), align_corners) * guide[:, j:j + 1]
    return q


def guided_labels(logits: torch.Tensor, frame: torch.Tensor, cfg: GuidedConfig, output_stride: int,
                  align_corners: bool, input_size: Tuple[int, int]) -> torch.Tensor:
    """(H, W) labels of :func:`guided_logits`."""
    return guided_logits(logits, frame, cfg, output_stride, align_corners, input_size).argmax(1)[0]


def bilinear_labels(logits: torch.Tensor, frame_hw: Tuple[int, int], output_stride: int, align_corners: bool,
                    input_size: Tuple[int, int]) -> torch.Tensor:
    """The full frame's own step (:func:`models.utils.upsample_logits`, argmax): the reference."""
    from models.utils import upsample_logits

    return upsample_logits(logits.float(), frame_hw, output_stride, align_corners, input_size=input_size).argmax(1)[0]
