# Method 11: fine-tune the segmenter on warped frames

Status: **draft, 2026-11-25, not yet accepted.** The user asked for it on
2026-11-25 ("okay for fine tuning for warps to start"), after the compute
report "Compute cost and literature: glimpses vs warps". The user set the
goal: once the model is trained, this is no longer a harness over a frozen
model; the aim is a **new point on the cost/accuracy plane**, ideally more
accurate than the base model, and scale 1 with magnification must be tried.
Points marked *(my call)* are Claude's proposals. Nothing here may change
after the first training run, except in a dated "Amendments" section at the
end.

**Not model-agnostic, and the write-up must say so.** Each warp is learned
into one model's weights by fine-tuning that model. The warp itself (fixed
prior, separable grid, unwarp of the logits) is Method 5's and does not look
inside the network.

## Intent

Methods 5 and 6 showed that a frozen model on a warped frame gains at most
about +1.25 / +1.80 mIoU at scale 0.25 with an oracle, and that a realisable
warp gives nothing on val at equal latency. The warp's cost is not the
problem (0.056-0.061 ms on a T4 at scales 0.25-0.375); the frozen model's
accuracy on stretched objects is. Fine-tuning attacks exactly that:

> After the same fine-tuning budget, does a model trained and run on warped
> frames beat every full-frame curve at equal measured T4 latency, including
> the same model fine-tuned on uniform frames at the same scale?

## Closest prior work, and what is new here

Learning to Zoom and Unzoom (Thavamani et al., CVPR 2023; RELATED.md, read in
full 2026-11-25) already trained a segmenter (PSPNet-R50) with a fixed
boundary-prior separable warp on Cityscapes: +0.3 / +2.4 / +1.5 / +0.4 mIoU
over uniform at 64 / 128 / 256 / 512 px, on centred square crops, judged in
FLOPs. A fine-tuned warp alone would repeat that paper. What is new, and must
stay in the design:

1. public **real-time** segmenters (PIDNet-S, BiSeNetV2), fine-tuned from
   their Cityscapes weights, on the full 1024 × 2048 frame;
2. judged at **equal measured T4 latency** against the envelope of every
   model's full-frame curve, **including the uniform fine-tunes** below;
3. on **video** (Cityscapes-VPS) with **tiny/small moving-object recall**
   (LZU's per-class table loses riders −6.7 and poles −1.8 at 512 px);
4. **scale 1 with magnification** (LZU never magnifies for segmentation);
5. our error prior P next to their boundary prior.

## Data (decided by the user on 2026-11-25: Cityscapes train fine)

The public weights were trained on Cityscapes **train** (2,975 fine images).
Every Cityscapes-VPS clip is a Cityscapes **val** sequence, so VPS
`train_all` (400 clips) and VPS `val` (50 clips) are both unseen by the
public weights.

- **Chosen: fine-tune on Cityscapes train fine (2,975 images).** Gates
  run on VPS `train_all` exactly as in Methods 2-10, val is read once.
  Needs `leftImg8bit/train` (about 6 GB of the 11 GB
  `leftImg8bit_trainvaltest.zip`, packageID 3 from memory, to confirm) and `gtFine/train`
  (packageID 1, already used), fetched by `scripts/get_val.py` extended with
  a `--train` option (range requests, the user's Cityscapes login).
- Not chosen: fine-tune on VPS `train_all`'s labelled frames (2,400, already
  on the cluster). Then the gate needs a held-out part of `train_all` (for
  example chunk 0, 100 clips), and the fine-tune sees fewer, more correlated
  images (6 per clip).

## What is trained

Per model and scale s, starting from the public weights *(my call for all
hyperparameters; one setting for every run, warp and control alike)*:

- SGD, momentum 0.9, weight decay 5e-4, lr 1e-3 (a tenth of the original),
  poly decay (power 0.9), **40 epochs** of the 2,975 images (119 k samples),
  AMP. Batch 8; at s ≥ 0.75, batch 4 with 2 accumulation steps. BatchNorm
  statistics frozen at the public values, affine parameters trained (batch 4
  is too small for BN statistics, and every run is treated the same).
- Loss: OHEM cross-entropy (threshold 0.7, at least 100 k pixels per batch;
  both public recipes use OHEM, these values are my call), main head only (the released auxiliary heads are not
  loaded by our ports), computed on the logits **after the unwarp**, at the
  label resolution: the same path as the evaluation, made differentiable
  (`grid_sample`).
- Input: the **whole frame** (the warp is defined on the whole frame, so no
  random crop), horizontal flip (per batch; the prior is flipped with the frame) and
  photometric jitter. No scale jitter except in the multi-scale control.
- Checkpoint: the **last** one. No selection on `train_all`, which is the
  gate's data.

### The runs (gate 1)

| Run | Scales | Per model |
| --- | --- | --- |
| `U-s`: uniform specialist (control) | 0.25, 0.5, 0.75, 1.0 | 4 |
| `M`: uniform multi-scale (control), s drawn per batch from the 13 scales 0.25-1 in steps of 1/16 | evaluated at all 13 | 1 |
| `B-s`: warp with the boundary prior B̄ | 0.25, 0.5, 0.75, 1.0 | 4 |
| `P-0.25`: warp with Method 5's scored prior P (the frozen BiSeNetV2 finalist's settings) | 0.25 | 1 |

Both models: 20 runs, one seed. The warp is Method 5's separable grid with
the full frame's own filter (`r2u1`), α = 0.25, σ = 4 cells, d_min = 0.5, the
upper clip 1/s; at **s = 1, d_max = 2** (magnifies the salient band, shrinks
the rest; the frozen version gained at most +0.36 for PIDNet-S).
**B̄** = the mean ground-truth semantic-boundary map of the Cityscapes train
labels on the 16 px cell grid, computed once, model-free (LZU's prior; it is
also the only prior defined at s = 1, where P is empty). P exists only at
0.25 and 0.375 and was fitted on `train_all` itself (in-sample, as in Method 5).

Estimated cost (compute report, 6 × inference per sample): about 50 GPU-hours
of training on the cluster's P100s plus about 20 of scoring; four jobs run at
a time, so about two days of queue.

## Prices

The fine-tuned networks keep their architecture, so a forward pass costs what
it cost frozen: the dense full-frame T4 prices (latency floor, final
sessions, linear in pixels between measured scales, as in Methods 7-8) plus
the warp's measured overhead. That overhead is measured at 0.25-0.375
(Method 5 stage A); **at 0.5-1.0 it is not**, so one Colab T4 session times
`B-s` warp + unwarp at those scales before any val read. Until then gate 1
uses the 0.25-0.375 overhead scaled linearly in output pixels *(my call)*.

## Gate 0: the pipeline

Before the 20 runs, per model on 5 clips of `train_all`:

- a "fine-tune" with lr 0 at s = 1 reproduces the frozen mIoU to 0.1 point
  (data, loss and evaluation paths agree);
- one epoch at lr 1e-3, s = 0.5, uniform, changes the 5-clip mIoU by less
  than 2 points either way (no collapse).

## Gate 1: train_all, 400 clips

The **envelope** E(t) is the best mIoU at T4 latency ≤ t over every
full-frame curve: frozen PIDNet-S and BiSeNetV2 (the same 13 scales, run through the same eval path), the
`U-s` specialists and `M` (13 scales) of both models. For each warp run w:

    Δ(w) = mIoU(w) − E(latency(w))

with a 95 % clip-bootstrap interval (seed 0, 2,000 resamples, as before).
A run is a **finalist** if Δ ≥ +1.0, its interval is above 0, and its
tiny/small moving-object recall (the approved metric) has an interval not
entirely below the envelope point's. At most two finalists per model, the
largest Δ first. No finalist: Method 11 stops, negative, and the write-up
says by how much the uniform fine-tune alone moved the curve.

Reported but not gated: Δ at the same scale against `U-s` (the warp's own
effect, the comparison LZU makes) and, at s = 1, mIoU(B-1) against the best
full-frame mIoU of any model (the "more accurate than the base model" record).

## Val: read once

Each finalist and its envelope runs are retrained with two more seeds (1, 2).
Then val (50 clips) is read once. **Pass**: the mean Δ over the three seeds
is above 0 with its interval above 0, at the T4-measured price, and moving
recall is not significantly below the envelope.

## What this does not test

A per-frame (adaptive or temporal) saliency, a 2-D warp, glimpses, and any
model change beyond the weights. If gate 1 passes, the next step is LZU's own
open point: a saliency from the previous frame's output, priced on the T4.
