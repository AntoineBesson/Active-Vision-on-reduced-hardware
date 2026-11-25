"""Method 11: the fine-tuning loop.

Spec: docs/superpowers/specs/2026-11-25-method11-warp-finetune-design.md ("What is trained").
One setting for every run, warp and control alike: SGD (momentum 0.9, weight decay 5e-4),
poly decay (power 0.9) from ``lr``, AMP, BatchNorm statistics frozen at the public values
(affine parameters trained), OHEM cross-entropy on the unwarped full-size logits, whole frames
with a horizontal flip (per batch, prior flipped with it) and photometric jitter (per sample).
The multi-scale control draws its scale per batch.

Resumable at epoch granularity: ``ckpt_last.pt`` in the run folder holds the weights, the
optimiser, the scaler and the epoch; a job that is cut off is simply submitted again.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from warp.policy import StepConfig

from .warped import WarpedForward

__all__ = ["RunConfig", "ohem_ce", "jitter", "freeze_bn", "run_training", "DENSE_SCALES"]

IGNORE = 255
#: the multi-scale control's scales: 0.25 to 1 in steps of 1/16
DENSE_SCALES = tuple(0.25 + k / 16 for k in range(13))


@dataclass
class RunConfig:
    """One fine-tuning run. ``kind``: ``uniform`` (specialist at ``scale``), ``multi`` (uniform,
    scale drawn per batch from ``scales``), ``boundary`` or ``error`` (warp with that prior)."""

    model: str
    kind: str
    scale: float = 1.0
    scales: Sequence[float] = DENSE_SCALES
    alpha: float = 0.25
    sigma: float = 4.0
    d_max: Optional[float] = None
    epochs: int = 40
    batch: int = 8
    accum: int = 1
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    poly_power: float = 0.9
    ohem_thresh: float = 0.7
    ohem_min_kept: int = 100_000
    jitter: float = 0.25
    flip: bool = True
    amp: bool = True
    seed: int = 0
    max_steps: Optional[int] = None  # gate 0 / smoke tests: stop after this many optimiser steps
    notes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "multi", "boundary", "error"):
            raise ValueError(f"unknown run kind {self.kind!r}")
        if self.kind == "multi" and not self.scales:
            raise ValueError("the multi-scale control needs scales")
        if self.batch < 1 or self.accum < 1 or self.epochs < 1:
            raise ValueError("batch, accum and epochs must be >= 1")

    def step_configs(self) -> List[StepConfig]:
        """The fixed warp configuration(s) this run trains (one, or one per scale for ``multi``)."""
        if self.kind == "multi":
            return [StepConfig(float(s), resample="r2", unwarp="u1") for s in self.scales]
        if self.kind == "uniform":
            return [StepConfig(float(self.scale), resample="r2", unwarp="u1")]
        return [StepConfig(float(self.scale), "P", self.alpha, self.sigma, resample="r2", unwarp="u1",
                           d_max=self.d_max)]

    def name(self) -> str:
        if self.kind == "multi":
            return f"11/{self.model}/M/seed{self.seed}"
        tag = {"uniform": "U", "boundary": "B", "error": "P"}[self.kind]
        mag = "" if self.d_max is None else f"/m{self.d_max:g}"
        return f"11/{self.model}/{tag}/c{self.scale:g}{mag}/seed{self.seed}"


def ohem_ce(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.7, min_kept: int = 100_000,
            ignore: int = IGNORE) -> torch.Tensor:
    """Online hard example mining cross-entropy: the mean loss of the pixels whose target
    probability is at most ``max(thresh, p_k)``, p_k the ``min_kept``-th smallest (at least
    ``min_kept`` pixels, all of them if fewer are valid). Ignored pixels never count."""
    ce = F.cross_entropy(logits, target, ignore_index=ignore, reduction="none")
    valid = target != ignore
    ce_v = ce[valid]
    n = int(ce_v.numel())
    if n == 0:
        return logits.sum() * 0.0
    p = torch.exp(-ce_v.detach())
    k = min(int(min_kept), n)
    kth = torch.kthvalue(p, k).values if k > 0 else p.min()
    keep = p <= torch.clamp(kth, min=float(thresh))
    return ce_v[keep].mean()


def jitter(x: torch.Tensor, strength: float, gen: torch.Generator) -> torch.Tensor:
    """Per-sample brightness, contrast and saturation factors in [1 - s, 1 + s] on a (B, 3, H, W)
    batch in [0, 1], clamped back to [0, 1]. ``strength`` 0: unchanged."""
    if strength <= 0:
        return x
    b = x.shape[0]

    def f():
        return (1 + (torch.rand(b, 1, 1, 1, generator=gen) * 2 - 1) * strength).to(x.device)

    x = x * f()
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * f() + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * f() + gray
    return x.clamp_(0, 1)


def freeze_bn(model: nn.Module) -> int:
    """Put every BatchNorm in eval mode (running statistics kept); returns how many there are."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            n += 1
    return n


def _poly(step: int, total: int, power: float) -> float:
    return (1 - min(step, total) / max(total, 1)) ** power


def run_training(cfg: RunConfig, model: nn.Module, dataset, out_dir: Path, device: str,
                 prior: Optional[torch.Tensor] = None, workers: int = 4,
                 log: Callable[[str], None] = print) -> Dict[str, object]:
    """Fine-tune ``model`` (its ``.net`` is saved) on ``dataset`` ((3, H, W) uint8, (H, W) int64
    pairs). Writes ``run.json``, ``log.jsonl``, ``ckpt_last.pt`` (each epoch) and, at the end,
    ``weights.pt`` (``{"state_dict": net state}``, loadable by the backbone registry)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_json = out_dir / "run.json"
    if run_json.exists():
        old = json.loads(run_json.read_text())["config"]
        new = json.loads(json.dumps(asdict(cfg)))
        if {k: v for k, v in old.items() if k != "max_steps"} != {k: v for k, v in new.items() if k != "max_steps"}:
            raise SystemExit(f"{out_dir} holds another run ({old.get('kind')}/{old.get('scale')}): use another folder")
    run_json.write_text(json.dumps({"name": cfg.name(), "config": asdict(cfg)}, indent=1))
    if (out_dir / "weights.pt").exists():
        log(f"[method11 train] {cfg.name()}: already finished ({out_dir / 'weights.pt'})")
        return {"done": True}

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = model.to(device)
    steps_fwd = [WarpedForward(model, c, prior=prior if c.saliency == "P" else None) for c in cfg.step_configs()]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    use_amp = bool(cfg.amp and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    per_epoch = len(dataset) // (cfg.batch * cfg.accum)
    if per_epoch < 1:
        raise SystemExit(f"{len(dataset)} samples cannot fill one step of {cfg.batch} x {cfg.accum}")
    total = per_epoch * cfg.epochs
    start_epoch = 0
    ckpt = out_dir / "ckpt_last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        log(f"[method11 train] {cfg.name()}: resuming after epoch {start_epoch}")

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    step = start_epoch * per_epoch
    t0 = time.time()
    with open(out_dir / "log.jsonl", "a", encoding="utf-8") as logf:
        for epoch in range(start_epoch, cfg.epochs):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.batch, shuffle=True, drop_last=True, num_workers=workers,
                generator=torch.Generator().manual_seed(cfg.seed * 1000 + epoch), pin_memory=use_amp,
                persistent_workers=False)
            model.train()
            n_bn = freeze_bn(model)
            it = iter(loader)
            for _ in range(per_epoch):
                lr = cfg.lr * _poly(step, total, cfg.poly_power)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                loss_sum = 0.0
                fwd = steps_fwd[int(torch.randint(len(steps_fwd), (1,), generator=gen))]
                flip = bool(cfg.flip and torch.rand(1, generator=gen).item() < 0.5)
                for _a in range(cfg.accum):
                    frames, labels = next(it)
                    x = frames.to(device, non_blocking=True).float().div_(255.0)
                    y = labels.to(device, non_blocking=True)
                    if flip:
                        x, y = x.flip(-1), y.flip(-1)
                    x = jitter(x, cfg.jitter, gen)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        logits = fwd(x, flip=flip)
                    loss = ohem_ce(logits.float(), y, cfg.ohem_thresh, cfg.ohem_min_kept) / cfg.accum
                    scaler.scale(loss).backward()
                    loss_sum += float(loss.detach())
                scaler.step(opt)  # lr 0 (gate 0) leaves the weights as they are
                scaler.update()
                step += 1
                if not math.isfinite(loss_sum):
                    raise SystemExit(f"{cfg.name()}: loss {loss_sum} at step {step}")
                if step % 20 == 0 or step == 1:
                    rec = {"step": step, "epoch": epoch, "loss": loss_sum, "lr": lr, "scale": fwd.cfg.scale,
                           "flip": flip, "elapsed_s": time.time() - t0}
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break
            del it, loader
            torch.save({"net": model.net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch + 1}, ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
            log(f"[method11 train] {cfg.name()}: epoch {epoch + 1}/{cfg.epochs}, step {step}/{total}, "
                f"{time.time() - t0:.0f} s, {n_bn} BN frozen")
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
    torch.save({"state_dict": model.net.state_dict()}, out_dir / "weights.pt.tmp")
    (out_dir / "weights.pt.tmp").replace(out_dir / "weights.pt")
    return {"done": True, "steps": step, "seconds": time.time() - t0}
