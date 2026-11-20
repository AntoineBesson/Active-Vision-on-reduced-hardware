# Method 6: a single pass on a locally (2-D) warped frame, oracle bound: negative

**Result (2026-11-20, 400 clips of `train_all`): the pre-registered check
fails for both backbones.** The rule needs a local 2-D warp that beats the
full frame by at least +1.0 mIoU point **and** Method 5's best separable warp
by at least +0.5 point, both at the same scale. At no scale, for neither
model, does the best 2-D configuration beat the separable warp, even with an
oracle saliency that costs nothing. It is 0.03 to 0.62 points below it
everywhere. Part (a) alone holds at scale 0.25 only (PIDNet-S +1.01,
BiSeNetV2 +1.19), which is below what the separable warp already gives there
(+1.25 and +1.80).

The grid, the latency parts and the pass rule were fixed in the
[spec](../../docs/superpowers/specs/2026-11-17-method6-local-warp-bound-design.md)
(accepted by the user on 2026-11-17, scale 0.75 kept) before any run. Tries
are logged in [TRIES.md](TRIES.md).

## The question

Method 5 closed negative: a separable warp (one density per row, one per
column) passed its oracle bound narrowly on train, but no warp that a real
policy can compute within the T4 budget beat the full frame on val. A
separable warp can only magnify whole rows and columns. Method 6 asks whether
a **free 2-D warp** does better: one that magnifies a region without
stretching its row and its column across the whole frame:

> At the same input size, does a frozen model on a locally warped frame beat
> both the uniformly downscaled frame and Method 5's best separable warp,
> when the warp is chosen by an oracle that costs nothing?

If it does not, a 2-D warp has nothing to add to Method 5, and its phase 2
(a real saliency, the T4 cost of a 2-D resampling) is not built.

## What is run

Per labelled frame of `train_all` (2400 frames, 400 clips), per model
(PIDNet-S, BiSeNetV2), frozen, fp32, on the school cluster's Tesla P100s:

- the teacher (scale-1 pass) and the uniform passes at each scale;
- the saliencies of Method 5 (**E** error vs the teacher, **B** boundaries,
  **S** small classes) on a 64 × 128 grid;
- a 2-D map built by kernel sampling from the saliency, inverted by Newton,
  then an antialiased 2-D resampling, one pass of the model on the warped
  image, and the output unwarped back to the frame;
- the grid: 4 scales (0.25, 0.375, 0.5, 0.75) × 3 saliencies × 3 strengths
  α (0.1, 0.25, 0.5) × 2 smoothings σ (1, 4 cells) × 2 kernels σ_k (4, 12
  cells) = 144 configurations, plus the uniform row (α = 0) and Method 5's
  best separable configuration through the same run, at each scale: 152 rows
  per model;
- latency priced as the full frame at the configuration's scale (T4 `area` +
  `reuse` `post`), with the warp, unwarp and saliency at 0 ms.

8 jobs (4 chunks × 2 models), 2 h each, 69 to 75 s per clip. Bound files in
`experiments/method6/train_all/` (not committed, 44 MB); score files here.

## Checks (before any reading)

| Check | PIDNet-S | BiSeNetV2 |
|---|---|---|
| Pipeline: uniform 2-D row vs the full frame at its scale (tolerance 0.1 pt) | ok at 4 scales, gaps ≤ 3e-5 pt | ok at 4 scales, gaps ≤ 1e-5 pt |
| Reproduction: separable row vs Method 5's score file (tolerance 0.1 pt) | ok at 4 scales, gaps < 1e-4 pt | ok at 4 scales, gaps < 1e-4 pt |

On chunk 0 alone the script's reproduction check read "NOT ok". That was
expected: it compares 100 clips with Method 5's 400-clip mIoU. On the same
100 clips, against Method 5's chunk-0 bound files, the four separable rows
matched within 0.001 point (see [TRIES.md](TRIES.md)).

## Results

Best 2-D configuration per scale, mIoU points against the full frame at the
same scale and against Method 5's best separable configuration at that scale
(the bar of part (b)). The rule needs ≥ +1.0 and ≥ +0.5.

| Model | Scale | Best 2-D configuration | vs full frame | Separable bar | vs separable | (a) | (b) |
|---|---|---|---|---|---|---|---|
| PIDNet-S | 0.25 | `E/a0.25/s1/k12` | **+1.01** | +1.25 | −0.24 | ✓ | ✗ |
| PIDNet-S | 0.375 | `E/a0.25/s4/k4` | +0.62 | +0.92 | −0.30 | ✗ | ✗ |
| PIDNet-S | 0.5 | `B/a0.25/s4/k4` | +0.53 | +1.00 | −0.46 | ✗ | ✗ |
| PIDNet-S | 0.75 | `B/a0.1/s4/k12` | +0.11 | +0.21 | −0.10 | ✗ | ✗ |
| BiSeNetV2 | 0.25 | `E/a0.25/s4/k12` | **+1.19** | +1.80 | −0.62 | ✓ | ✗ |
| BiSeNetV2 | 0.375 | `E/a0.1/s1/k12` | +0.78 | +1.17 | −0.39 | ✗ | ✗ |
| BiSeNetV2 | 0.5 | `B/a0.1/s1/k12` | +0.31 | +0.33 | −0.03 | ✗ | ✗ |
| BiSeNetV2 | 0.75 | `B/a0.1/s4/k4` | +0.04 | +0.07 | −0.03 | ✗ | ✗ |

Verdict: [verdict_train_all.json](verdict_train_all.json), `passes: false`.
Per-configuration rows (mIoU, per-class IoU, latency, the three full-frame
judges, budget, distortion): `score_train_all_<model>.json.gz`.

### What the grid shows

- **The 2-D warp is never better than the separable one.** In all 8
  model × scale cells the best 2-D configuration is below Method 5's best
  separable configuration. The gap is large at low scales and nearly zero at
  0.5 and 0.75 for BiSeNetV2, where neither warp gains much.
- **Strong local warps hurt.** α = 0.5 loses on average 0.3 to 2.8 points
  against the full frame, depending on model and scale. The worst
  configuration at each scale has α = 0.5, small-class saliency below 0.75,
  and loses 0.6 to 5.4 points. The best
  configurations use α = 0.1 or 0.25, with small distortion: local
  magnification at the 95th percentile ≤ 1.17, mean anisotropy ≤ 1.17.
- **The error oracle stays the best saliency at low scales** (E wins at 0.25
  and 0.375 for both models). Boundaries win at 0.5 and 0.75, where every
  warp is within a few tenths of the full frame. Small classes are worst on
  average.
- **The kernel width does not decide it.** σ_k = 12 and σ_k = 4 each give
  the best configuration at some scales, and their best rows are within 0.3
  points of each other.
- The budget left by the best configurations (latency saved at equal mIoU on
  the full frame priced with the same parts) is 0.13 to 1.19 ms. That is no
  reason to go on, since part (b) fails.

### Reading

A 2-D warp can put pixels where the error is without stretching whole rows
and columns. That does not translate into accuracy for these frozen
backbones. The more freely and strongly the frame is reshaped, the more the
model sees shapes it was not trained on. Within the grid's tolerable
distortion, the 2-D warp ends up no better than a separable one, and usually
worse. Together with Method 5, this says that for a frozen, model-agnostic
harness the gain from non-uniform resampling is at most about +1 to +2
points at scale 0.25 with an oracle, and does not survive a real policy and
its cost.

Not tested: a backbone trained or fine-tuned on warped frames (outside the
frozen-model harness), a warp learned end to end, and other kernels or
grids than the spec's.

## Next

Per the spec: the result is written up, the top-level README is updated,
and the method stops here. **What follows is the user's call.**

## Reproducing

```bash
# per chunk and model (cluster: scripts/cluster/m6_bound.sh, docs/cluster.md)
python -m scripts.method6 bound --split train_all --models pidnet_s --chunk 0/4
python -m scripts.method6 merge --split train_all
python -m scripts.method6 score --split train_all     # CPU, about a minute per model
```
