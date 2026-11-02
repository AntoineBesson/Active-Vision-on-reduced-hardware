# Method 2: ledger of tries

Every attempt at Method 2 (foveation without memory), in order, including the
ones that failed, so the paper can report what was tried and why it was
abandoned or changed. Each entry: what ran, on which split, the numbers, the
conclusion and what changed next. Numbers are on VPS **train** unless stated;
val is evaluated once, on finalists only. Latencies are predicted T4 medians
from the latency-floor tables (batch 1, fp16, CUDA graphs), with the full
frame timed as the backbone alone (no resize, upsampling or argmax), which
slightly favours the full frame.

Raw per-configuration results stay in `experiments/method2/` (not committed);
each try has a compact summary in `tries/`.

---

## Try 1: overwrite the context with full-resolution glimpses (failed)

**Date:** 2026-10-30. **Summary:** `tries/try1_pidnet_replace_train.json`.

**Setup.** PIDNet-S for context and glimpses. Context at scale 0.25 / 0.375 /
0.5; glimpses of 256 / 384 / 512 px at full resolution, 32 px margin, k = 1,
2, 4, 8; the glimpse's argmax overwrites the context inside each selected
cell. Selection: entropy / margin / max-probability scores, mean or top-10%
pooling, top-k or greedy; references: pixel oracle (labels), fixed prior,
random (5 seeds), every cell (tiling bound). 549 configurations.

**Result.** No configuration reaches the full-frame curve. At equal T4
latency the best rule of each geometry is **5 to 19 mIoU points below** the
full frame (best: context 0.5 + one 256 px glimpse, 67.9 mIoU at 9.5 ms,
against the full frame interpolated at that latency, ~73).

Full frame on train: 76.1 (scale 1.0, 19.4 ms), 74.8 (0.75, 10.7 ms), 67.9
(0.5, 5.2 ms), 61.7 (0.375, 3.6 ms), 52.9 (0.25, 2.7 ms).

**Cause: glimpses lack context.** With every cell glimpsed (no selection at
all) the tiled prediction scores only **53.5 / 60.3 / 63.0 mIoU** for 256 /
384 / 512 px crops, against 76.1 for the full frame. A probe on 5 val clips
(PIDNet-S, mIoU / pixel accuracy) shows accuracy climbing back to the full
frame as crops and margins grow, so this is the network, not a bug.
*Protocol breach, disclosed:* this probe ran on val, not train, and its
768/64 and 1024/64 rows suggested the wide geometries of try 2 (384:32:0.5,
512:32:0.5, 384:24:0.75). It only shaped the search space, in foveation's
favour; tuning and the finalists stayed on train, and the result is
negative regardless. It should have run on train.

| Input | mIoU | Pixel acc. |
| --- | --- | --- |
| full frame | 57.5 | 95.6 |
| 256 px crops, 32 px margin | 44.3 | 86.1 |
| 384 px, 32 | 49.5 | 89.6 |
| 512 px, 32 | 47.8 | 90.1 |
| 512 px, 128 | 52.2 | 92.4 |
| 768 px, 64 | 54.7 | 93.8 |
| 1024 px, 64 | 55.3 | 93.5 |
| 1024 px, 256 | 56.4 | 95.0 |

(5 clips only: absolute mIoU is low because rare classes barely occur; the
trend is the point.) PIDNet-S is trained on 1024x1024 crops and has a global
pooling branch; a 256–512 px crop does not see enough of the scene to name
whole regions correctly.

**Other observations.**
- Among selection rules, max-probability or margin with top-10% pooling and
  top-k wins almost everywhere; the fixed prior often matches or beats it.
- The pixel oracle (k cells adding the most correct pixels) is far above every
  rule (e.g. 65.9 vs 58.1 at context 0.25, 4 x 384 px), so selection has room,
  but even the oracle stays below the full-frame curve because of the tiling
  bound.
- The oracle maximises correct pixels, not mIoU, so it is not an mIoU ceiling.

**Next (decided with the user):** keep the glimpse but fuse it with the
context inside the cell instead of overwriting it (try 2), and wide glimpses
that see more of the scene at reduced resolution (try 2). Larger margins at
full resolution were set aside for now (more pixels for the same cell).

---

## Try 2: fuse glimpses with the context; wide glimpses at reduced resolution (failed)

**Date:** 2026-10-30. **Summary:** `tries/try2_fusion_wide_train.json`.

**Setup.** Context and glimpse pairs PIDNet-S + PIDNet-S, BiSeNetV2 +
BiSeNetV2, BiSeNetV2 context + PIDNet-S glimpses. Fusion inside a selected
cell: `replace` (try 1), `mean` (argmax of the mean of the two softmaxes),
`conf` (the more confident of the two per pixel). Glimpse geometries
(frame px : margin : resolution) 256:32:1, 384:32:1, 512:32:1, 768:64:0.5,
1024:64:0.5, 512:32:0.75. Same scales, k, selection rules and references as
try 1.

**Run notes.** The first train cache pass (138 caches in one pass over 2400
frames, laptop RTX 2060) crashed after 1099 frames with a cuDNN execution
failure at ~87 °C; the builder could not resume, so it restarted from zero
(2026-10-30 19:39), now checkpointing every clip and retrying after a crash.
No result was lost to the crash, only time.

**Result.** Still no configuration reaches the full-frame curve, for any
pair. Best non-oracle rule at equal T4 latency (mIoU points; fused and wide
latencies are lower bounds, so the true gap is somewhat larger):

| Pair | Overwrite (try 1 rule) | Fused (mean / conf) | Wide (0.5 / 0.75) |
| --- | --- | --- | --- |
| PIDNet-S + PIDNet-S | -5.3 to -6.6 | -5.2 to -5.9 | -5.7 to -7.2 |
| BiSeNetV2 + BiSeNetV2 | -1.8 to -2.2 | **-0.6** to -1.7 | -1.8 to -3.2 |
| BiSeNetV2 ctx + PIDNet-S glimpses | -4.8 to -5.2 | -4.3 to -5.1 | -5.0 to -6.3 |

Every best rule uses the 0.5 context. The closest point is BiSeNetV2, context
0.5 + 8 fused 384 px glimpses: 72.8 mIoU at >= 30.6 ms, against 73.4 for the
BiSeNetV2 full frame at scale 0.75 in 17.8 ms. (The mixed pair is compared with
the PIDNet-S full-frame curve, its glimpse model's.)

Full frame on train, BiSeNetV2: 73.3 (scale 1.0, 31.2 ms), 73.4 (0.75,
17.8 ms), 70.5 (0.5, 8.1 ms), 66.8 (0.375, 4.7 ms), 59.2 (0.25, 2.7 ms).
PIDNet-S as in try 1.

**Fusion lifts the tiling bound but not enough.** Every cell glimpsed at full
resolution with a 0.5 context, mIoU (overwrite / mean / conf):

| Pair | 256 px | 384 px | 512 px | context 0.5 alone | full frame best |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S | 53.5 / 65.9 / 65.2 | 60.3 / 69.2 / 68.6 | 63.0 / 70.7 / 70.2 | 67.9 | 76.1 |
| BiSeNetV2 | 59.4 / 70.0 / 69.5 | 64.3 / 71.9 / 71.6 | 65.1 / 72.7 / 72.3 | 70.5 | 73.4 |

So even glimpsing everything adds at most ~2-3 points over the context alone,
and the full frame at a slightly higher scale gives more for less time. Mean
fusion beats conf by 0.3-1.0 points in this bound but by only ~0.1 at the best
operating points; at those points fusion beats overwriting by 0.1-1.3 points.

**Wide glimpses hurt.** A 768-1024 px region shrunk to half resolution has no
more detail than the 0.5 context, so fusing it lowers accuracy (PIDNet-S, every
cell 384:32:0.5 fused with context 0.5: 65.6 vs 67.9 for the context alone).
The 0.75 geometry sits between the two.

**Selection.** For BiSeNetV2 with full-resolution glimpses the pixel oracle is
now *above* the curve (+0.9 to +2.0 points), so a better rule would have some
headroom there (the oracle maximises correct pixels, so this is indicative, not
a ceiling); with wide glimpses it is +0.3 to -2.2, and for the PIDNet-S glimpse
pairs it stays below (-1.3 to -6.1).

**Why the full frame is hard to beat on a T4.** Full-frame latency grows
slower than its pixel count at these sizes, so the step from scale 0.5 to
0.75 is cheap (PIDNet-S: 5.2 -> 10.7 ms for +6.9 mIoU), while each glimpse
pays a launch, a resize and the recomposition, and each crop sees too
little of the scene to be accurate.

**Next (decided with the user):** finish the plan with what exists: tune
class weights, freeze finalists on train, report on val once.

---

## Class weights, finalists and val (2026-11-01)

**Class weights** were tuned on train for the best score kind of each of the
162 (pair, scale, geometry, fusion) combinations. Each run uses coordinate
ascent over the mean mIoU for k = 1, 2, 4, 8 (`class_weights.json`). In
sample, the tuned objective improves by +0.1 to +5.8 mIoU (median +1.1). At
the best operating points the effect is small: the best ΔmIoU moves by
+0.2 (PIDNet-S), -0.3 (mixed pair) and -0.5 (BiSeNetV2), because those
points use the fixed prior or other rules.

**Finalists:** 55 (`finalists.json`): the train Pareto front plus the best
configuration under each full-frame latency, frozen and committed before
val.

**Val (run once, `val.json`). Failed.** 0 of 55 finalists have a positive
ΔmIoU at equal latency. Only two have an interval that contains 0, both
BiSeNetV2 with a 0.5 context and fused 384 px glimpses chosen by the prior:

- k = 4: -0.5 [-1.7, +0.3];
- k = 8: -0.6 [-2.1, +0.6].

In latency saved at equal mIoU they are -5.5 and -13.3 ms (slower). The best
of the other pairs:

- BiSeNetV2 context + PIDNet-S glimpses: -2.1 [-4.0, -0.5];
- PIDNet-S + PIDNet-S: -3.2 [-5.5, -1.6].

The fp16 check passes: |ΔmIoU| ≤ 0.016 points (`fp16_check.json`).

**Conclusion for the paper.** Memoryless foveation with off-the-shelf
segmentation backbones doesn't beat simply running the full frame at a
lower scale on a T4. The two reasons:

- crops lack the scene context these networks need;
- on this GPU, full-frame latency scales sublinearly in pixels, while each
  glimpse pays a fixed overhead.

Write-up: [README.md](README.md).
