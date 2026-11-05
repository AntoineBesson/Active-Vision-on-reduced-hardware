# Method 2: foveation without memory — results

**Outcome: negative, confirmed by end-to-end T4 timing.** On a T4, no
configuration of Method 2 beats the full frame. With both sides measured end to
end, 46 of the 55 finalists chosen on VPS train land below the full-frame
mIoU-vs-latency curve on VPS val; the other 9 are slower than the full frame at
scale 1 and less accurate, so they are dominated. The best one, BiSeNetV2 with a
0.5 context, 4 fused 384 px glimpses and the fixed prior, is **-0.54 mIoU** at
equal latency and about **7.7 ms slower** than a full frame of the same mIoU.
Its accuracy interval from the predicted latencies, [-1.7, +0.3], makes it a
statistical tie on accuracy. See [Measured on the T4](#measured-on-the-t4).
The tries that led here, failures included, are in [TRIES.md](TRIES.md).

## What Method 2 does

Per frame, with no memory of earlier frames:

1. The **context** model segments the whole frame at a reduced scale (0.25,
   0.375 or 0.5).
2. The **policy** scores the cells of a fixed grid from the context
   (entropy, margin or max-probability, with mean or top-10% pooling and
   optional class weights) and picks k cells (top-k or greedy), or uses a
   fixed prior order.
3. The **glimpse** model segments a crop around each selected cell. A crop
   covers the cell plus a margin and runs at full resolution, or at reduced
   resolution for "wide" glimpses.
4. **Recomposition** writes each glimpse into its cell. It either
   overwrites the context there (`replace`), takes the argmax of the mean of
   the two softmaxes (`mean`), or keeps the more confident of the two
   (`conf`).

Code: `src/foveation/` (`pipeline.py`, `method.py`). Evaluation uses
per-cell confusion caches (`cache.py`, `cached_eval.py`), which reproduce
the real pipeline exactly. A test on 2 val clips gives identical selections
and pixel differences of at most 1e-4.

## Protocol

- **Data.** Selection and tuning use Cityscapes-VPS **train**: 400 clips,
  2400 labelled frames. Reporting uses VPS **val**: 50 clips, 300 labelled
  frames, evaluated **once**, on finalists frozen beforehand
  (`finalists.json`, committed before `val.json`).
  - Every VPS clip, train and val, comes from a Cityscapes val sequence
    (Frankfurt, Lindau, Münster), so neither backbone trained on these frames.
    The BiSeNetV2 checkpoint (CoinCheung) was trained on Cityscapes `train.txt`
    only, and PIDNet-S uses the official `_val` checkpoint, trained on
    Cityscapes train.
  - PIDNet-S full frame scores 76.1 on VPS train against 71.5 on VPS val. That
    gap isn't overfitting; the two splits hold different clips. Val is held out
    from tuning.
- **Metric.** 19-class mIoU, with per-class IoUs kept in `val.json`.
  Intervals are 95% clip bootstraps (n = 1000, seed 0). ΔmIoU and latency
  saved are paired with the full frame on the same resampled clips, so their
  intervals are much narrower than those of the mIoU itself.
- **Latency: the judge is the T4.** Latencies are T4 medians at batch 1,
  fp16, with CUDA graphs. They are *predicted* from the latency-floor
  tables (3 Colab sessions, see
  [../latency_floor](../latency_floor/README.md)): area, batch and overhead
  tables, using the median of the session medians. That model matches the
  12 measured end-to-end points within 5%.
  - p95 is the sum of the parts' p95s. This is conservative, since tails
    don't all coincide.
  - The overhead table was measured with entropy/mean scoring. Other
    scores weren't timed separately.
  - **Fused (`mean`, `conf`) and wide glimpses are lower bounds** (marked
    ≥): the tables don't time the extra softmaxes or the resize.
  - The full-frame reference is the backbone alone (no resize, upsample or
    argmax). This slightly favours the full frame. The Colab `final` part
    of the latency notebook times both sides end to end; see
    [Measured on the T4](#measured-on-the-t4). The tables below keep the
    predicted latencies.
  - Mixed pairs are read against the full-frame curve of their glimpse
    model.
- **Search space (train).**
  - 3 pairs (context + glimpse): PIDNet-S + PIDNet-S, BiSeNetV2 +
    BiSeNetV2, BiSeNetV2 + PIDNet-S.
  - Context scales 0.25 / 0.375 / 0.5; k = 1, 2, 4, 8.
  - 6 glimpse geometries (frame px : margin : resolution): 256:32:1,
    384:32:1, 512:32:1, 768:64:0.5, 1024:64:0.5, 512:32:0.75 (CLI notation:
    `input_size:input_margin:glimpse_scale`).
  - 3 fusions.
  - Every score / pooling / selection rule, the fixed prior, random
    selection (5 seeds), plus class weights tuned per configuration by
    coordinate ascent (`class_weights.json`).
  - References: the **pixel oracle**, which picks the k cells adding the
    most correct pixels (labels used). It maximises pixels, not mIoU, so it
    is *not* an mIoU ceiling. Also the **every-cell** tiling bound.
- **Finalists.** For each pair, the train Pareto front over (latency, mIoU),
  plus the best configuration under each full-frame latency. That gives 55
  in all, with oracle, random and every-cell excluded.

## Train: why it fails

Plots: `train_<pair>.png`. Pareto points and bounds:
`sweep_train_summary.json`.

- **Glimpses lack context.** Tiling the whole frame with 256–512 px
  full-resolution crops, overwriting the context, scores 53.5 / 60.3 / 63.0
  mIoU (PIDNet-S) against 76.1 for the full frame.
- **Fusion lifts the bound, but not far enough.** Every cell fused with a
  0.5 context reaches at most 70.7 (PIDNet-S; context alone 67.9) and 72.7
  (BiSeNetV2; context alone 70.5, full frame 73.4).
- **Wide glimpses** at half resolution carry no more detail than a 0.5
  context, and fusing them lowers accuracy.
- **On a T4 the full frame is cheap to scale up.** Latency grows more
  slowly than pixel count, so going from scale 0.5 to 0.75 costs little:
  PIDNet-S goes from 5.2 to 10.7 ms for +6.9 mIoU. Each glimpse, by
  contrast, pays a launch, a resize and the recomposition.
- **Selection.** The fixed prior ranks first or close to it at the best
  points. Tuned class weights move the best ΔmIoU by at most 0.5 either way. The
  pixel oracle is above the curve only for BiSeNetV2 with full-resolution
  glimpses (+0.9 to +2.0 on train), and below it for the PIDNet-S glimpse
  pairs.

## Val: the finalists

Full frame on val (scale 1 / 0.75 / 0.5 / 0.375 / 0.25):

- PIDNet-S: 71.5 / 70.1 / 64.8 / 58.6 / 50.1 at 19.3 / 10.7 / 5.2 / 3.6 /
  2.7 ms.
- BiSeNetV2: 70.4 / 69.9 / 67.6 / 64.3 / 59.1 at 31.2 / 17.8 / 8.1 / 4.7 /
  2.7 ms.

Plots: `val_<pair>.png`. Every finalist: `val.json`, which also holds the
oracle, prior and every-cell references at each finalist's geometry. The
tables below show the 4 best finalists of each pair. "Speedup" is against
the full frame at scale 1, realised (T4) / implied by FLOPs. "Rule" is
either the fixed prior or score/pooling/selection, with "/w" for tuned
class weights.

**bisenetv2+bisenetv2** (18 finalists; the 4 best by ΔmIoU)

| ctx scale | glimpse | fusion | k | rule | val mIoU [95%] | T4 median / p95 (ms) | GFLOPs | pixels | speedup realised / FLOPs | ΔmIoU at equal latency [95%] | latency saved at equal mIoU (ms) [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 384:32:1 | mean | 4 | prior | 69.6 [64.4, 71.6] | ≥22.0 / ≥22.5 | 104.6 | 53% | 1.4x / 1.9x | -0.5 [-1.7, +0.3] | -5.5 [-8.1, +4.3] |
| 0.5 | 384:32:1 | mean | 8 | prior | 69.8 [64.7, 71.8] | ≥30.6 / ≥31.4 | 159.9 | 81% | 1.0x / 1.2x | -0.6 [-2.1, +0.6] | -13.3 [-15.5, -2.4] |
| 0.5 | 384:32:1 | mean | 2 | prior | 68.8 [63.4, 71.0] | ≥17.8 / ≥18.2 | 76.9 | 39% | 1.8x / 2.6x | -1.1 [-2.6, -0.2] | -4.6 [-7.4, -1.3] |
| 0.5 | 512:32:1 | mean | 2 | prior | 69.0 [63.8, 70.9] | ≥21.8 / ≥22.3 | 98.4 | 50% | 1.4x / 2.0x | -1.1 [-2.8, -0.0] | -8.0 [-10.6, -2.1] |

Positive ΔmIoU: 0/17 (ΔmIoU undefined for 1: slower than the full frame at scale 1 and less accurate, so dominated); interval above 0: 0; interval containing 0: 2.

**bisenetv2+pidnet_s** (20 finalists; the 4 best by ΔmIoU)

| ctx scale | glimpse | fusion | k | rule | val mIoU [95%] | T4 median / p95 (ms) | GFLOPs | pixels | speedup realised / FLOPs | ΔmIoU at equal latency [95%] | latency saved at equal mIoU (ms) [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 384:32:1 | mean | 2 | prior | 69.0 [63.7, 71.0] | ≥16.9 / ≥17.3 | 62.5 | 39% | 1.1x / 1.5x | -2.1 [-4.0, -0.5] | -7.3 [-8.5, -6.1] |
| 0.5 | 384:32:1 | mean | 1 | prior | 68.7 [63.1, 70.7] | ≥15.7 / ≥16.3 | 55.8 | 32% | 1.2x / 1.7x | -2.2 [-4.2, -0.8] | -6.4 [-8.1, -5.3] |
| 0.5 | 256:32:1 | mean | 1 | maxprob/mean/topk/w | 67.5 [62.6, 69.6] | ≥14.9 / ≥15.3 | 52.2 | 28% | 1.3x / 1.8x | -3.2 [-5.0, -1.5] | -6.9 [-8.0, -5.6] |
| 0.5 | 256:32:1 | mean | 2 | maxprob/mean/greedy/w | 67.4 [62.4, 69.5] | ≥15.5 / ≥15.9 | 55.1 | 31% | 1.2x / 1.7x | -3.4 [-5.2, -1.6] | -7.6 [-8.7, -6.4] |

Positive ΔmIoU: 0/17 (ΔmIoU undefined for 3: slower than the full frame at scale 1 and less accurate, so dominated); interval above 0: 0; interval containing 0: 0.

**pidnet_s+pidnet_s** (17 finalists; the 4 best by ΔmIoU)

| ctx scale | glimpse | fusion | k | rule | val mIoU [95%] | T4 median / p95 (ms) | GFLOPs | pixels | speedup realised / FLOPs | ΔmIoU at equal latency [95%] | latency saved at equal mIoU (ms) [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 512:32:1 | mean | 4 | prior | 68.0 [63.1, 69.3] | ≥18.2 / ≥18.6 | 70.8 | 75% | 1.1x / 1.3x | -3.2 [-5.5, -1.6] | -9.5 [-10.7, -8.1] |
| 0.5 | 384:32:1 | mean | 4 | prior | 67.0 [62.2, 68.3] | ≥14.0 / ≥14.2 | 50.1 | 53% | 1.4x / 1.9x | -3.6 [-5.1, -2.4] | -6.5 [-7.3, -5.4] |
| 0.5 | 256:32:1 | mean | 1 | margin/mean/topk/w | 65.0 [60.3, 66.5] | ≥9.5 / ≥9.8 | 26.5 | 28% | 2.0x / 3.6x | -3.9 [-5.1, -2.6] | -4.1 [-4.3, -3.8] |
| 0.5 | 512:32:1 | mean | 2 | prior | 66.2 [61.7, 67.7] | ≥13.3 / ≥13.6 | 47.2 | 50% | 1.4x / 2.0x | -4.3 [-5.8, -2.6] | -6.7 [-7.0, -5.5] |

Positive ΔmIoU: 0/15 (ΔmIoU undefined for 2: slower than the full frame at scale 1 and less accurate, so dominated); interval above 0: 0; interval containing 0: 0.


**fp16 check** (`fp16_check.json`). The first finalist of each pair was
re-cached with the backbones in fp16 and gave |ΔmIoU| ≤ 0.016 points against
fp32 (threshold 0.1). So the fp32 caches stand for the fp16 deployment.

## Measured on the T4

Three Colab T4 sessions (`latency_floor_final-s{1,2,3}.json`, about 11 min each,
median clock 1200-1275 MHz) timed all 55 finalists interleaved with both models'
full frame, end to end from a full-resolution frame. Numbers are the median of
the three session medians, read against the val mIoU of `val.json`. Intervals
were not recomputed with these latencies.

Full frame, captured graphs, median ms (backbone-only table value in brackets):

| Scale | 1 | 0.75 | 0.5 | 0.375 | 0.25 |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S | 21.47 (19.35) | 13.02 (10.74) | 7.04 (5.22) | 5.25 (3.59) | 4.00 (2.73) |
| BiSeNetV2 | 33.30 (31.16) | 20.27 (17.78) | 10.05 (8.06) | 6.61 (4.72) | 4.27 (2.69) |

Resize, upsampling and argmax add about 1.3-2.5 ms.

Best finalist per pair, captured graphs (ΔmIoU against the measured curve):

| Pair | Finalist | Val mIoU | Predicted median | Measured median / p95 (ms) | ΔmIoU, median / p95 judge | Latency saved at equal mIoU |
| --- | --- | --- | --- | --- | --- | --- |
| bisenetv2+bisenetv2 | 0.5, 384, mean, k4, prior | 69.6 | 22.0 | 26.6 / 27.0 | -0.54 / -0.53 | -7.7 ms |
| bisenetv2+pidnet_s | 0.5, 384, mean, k1, prior | 68.7 | 15.7 | 17.4 / 17.9 | -2.09 / -2.14 | -5.9 ms |
| pidnet_s+pidnet_s | 0.5, 256, mean, k1, margin/mean/topk/w | 65.0 | 9.5 | 10.5 / 10.8 | -2.86 / -2.93 | -3.3 ms |

- **Verdict.** 0 of the 46 readable finalists are above the curve under the median
  judge, and 0 of 46 under p95. The other 9 are dominated. In eager mode, 0 of 16
  readable are above it; every PIDNet-S-glimpse finalist is slower than that
  model's scale 1 there.
- **Prediction.** Measured over predicted median is 1.15 (range 1.01-2.12), so the
  lower bounds were real. It is 1.10 for finalists timed with their own rule and
  1.28 for prior ones.
- **Prior finalists are upper bounds.** The 24 prior finalists are timed with entropy
  top-k scoring, which a fixed order doesn't need. Even at their predicted latency,
  against the measured curve, none is above it (best -0.37).

## What is left

Directions that would change the picture, none of them tried yet (see
[TRIES.md](TRIES.md)):

- glimpse models trained on crops;
- a context scale above 0.5;
- reuse across frames (Methods 3 and 4).
