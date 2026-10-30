# Method 2 (foveation without memory): accuracy experiments

Status: draft for approval, 2026-10-30. Branch `foveation-no-memory`.

## Intent

The latency floor ([results/latency_floor](../../../results/latency_floor/README.md))
tells us what each foveated configuration costs on a T4. This design measures
what each one is worth. The aim is to find out whether a low-resolution context
pass followed by k full-resolution glimpses beats simply shrinking the whole
frame, at equal T4 latency, for PIDNet-S and BiSeNetV2. The same code must
serve both models, which also checks that the harness is model-agnostic.

The accuracy metric is the standard 19-class mIoU, and nothing else (decided
2026-10-30). No per-class subset is a target, although per-class IoUs are
still reported.

The trade-off target isn't decided yet (it waits on a discussion with a
collaborator). So the deliverable is full curves, from which either target can
be read:

- **ΔmIoU at equal latency**: the configuration's mIoU minus the full-frame
  mIoU interpolated at the same T4 latency;
- **latency saved at equal mIoU**: the full-frame latency interpolated at the
  same mIoU, minus the configuration's latency.

## Goals

1. Wrap `FoveatedSegmenter` as a `FrameMethod`, evaluated by the existing
   harness and run JSON like the full-frame baseline. Report `context`,
   `glimpses`, `policy` and `recomposition` to the cost ledger.
2. Bound the method before tuning it:
   - **tiling bound**: every cell glimpsed. This is what crops with margins
     lose against the full frame, and the ceiling for any selection;
   - **oracle selection**: the k cells that gain the most correct pixels over
     the context prediction, chosen with the labels. This is the ceiling for
     any selection policy at a given (context scale, glimpse size, k);
   - **floor selections**: random cells (fixed seeds) and a fixed prior
     (the most often useful cells on train).
3. Tune deterministic, uncertainty-driven selection on **VPS train**, then
   report the chosen configurations once on **VPS val**.
4. Report per configuration: mIoU, per-class IoU, predicted T4 latency (median
   and p95), GFLOPs, pixel fraction, realised / FLOPs-implied speedup, and
   both trade-off readings above. On val, add clip-bootstrap 95% intervals.
5. Measure the finalists end to end on the T4 (3 sessions, interleaved with
   the full-frame baselines), to confirm the predicted latencies.

## Non-goals

- No memory, no use of previous frames, no motion cue: that's Method 3
  (previous-frame reuse) and Method 4 (canvas).
- No RL and no learned policy. Every selection rule is deterministic.
- No retraining or fine-tuning of the backbones.
- No throughput or batch > 1 numbers.

## Anti-goals

- Tuning anything on val, or choosing finalists by their val numbers.
- Claiming a gain inside the noise. Val has 50 clips, and rare classes are
  decided by a handful of objects there.
- A latency number that didn't come from the T4 floor tables or a T4 run.
- Collapsing the result into one trade-off scalar.

## Constraints

- **Data.** VPS train has 400 clips × 6 labelled frames (2,400 frames) and VPS
  val has 50 × 6 (300). The 2,400 train labels are prepared locally
  (`train/labelmap/`). The baseline has only run on val so far, so check that
  the adapter loads the train split before building caches. Both splits come
  from Cityscapes val sequences, which the
  backbone checkpoints never saw in training: the PIDNet-S checkpoint is the
  `_Cityscapes_val` one, trained on Cityscapes train only. Check the same for
  the BiSeNetV2 checkpoint before reporting.
- **Where accuracy runs.** On the laptop RTX 2060, where the weights and data
  are. Accuracy doesn't depend on the GPU. Latency comes from the T4.
- **Precision.** The accuracy pass is fp32, like the baselines. Check once
  that fp16 changes the mIoU by less than 0.1 on one configuration. Latency
  stays fp16 with CUDA graphs.
- **Latency of a configuration** = context (area table) + glimpses (batch
  table, at that size and k) + overhead (overhead table), from the three T4
  sessions. The tables only cover measured points, so the search stays on
  those points:
  - context scales 0.25, 0.375, 0.5;
  - glimpse sizes 256, 384, 512;
  - k = 1, 2, 4, 8 (16 for crops of 256 and less);
  - margin 32;
  - top-k or greedy selection.

  Anything off the grid (another margin, k = 3, another scale) first needs a
  `latency_floor` run for that point. Mixed pairs (BiSeNetV2 context +
  PIDNet-S glimpses) use the stride-1 overhead table. Check that estimate
  against the four mixed end-to-end points already measured.
- **Full-frame reference curves.** Run the existing baseline at five scales
  on VPS train too, for tuning. The val curve already exists. Equal-latency
  interpolation is piecewise linear in latency between those five points.

## How it works

### Cached evaluation (the key decision)

The canvas is made of non-overlapping cells. A selected cell's inner square
comes entirely from its glimpse, and every other pixel from the upsampled
context. The glimpse of a cell also doesn't depend on which other cells are
selected. So for a given (backbone pair, context scale, glimpse size, margin),
every selection can be evaluated exactly from two passes per frame:

1. the context pass;
2. **every** cell glimpsed (the tiling bound itself).

Per frame and per cell, store two 19×19 confusion matrices, restricted to the
cell's in-frame pixels: context vs labels and glimpse vs labels. Also store
the context outputs that the selection scores need (see below). The confusion
matrix of any selection is then the sum, over cells, of the glimpse matrix for
selected cells and the context matrix for the others. Summing a few thousand
small matrices takes milliseconds, so a full train evaluation of a new
selection rule costs about a second instead of minutes.

Size, for example with 384 glimpses (4 × 7 grid): 28 cells × 2 × 19² int32 ≈ 80 kB per frame, about 200 MB per
(backbone pair, scale, size) on train. That's fine on disk.

**Validation.** On 2 val clips and 3 configurations (top-k, greedy, mixed
pair), the cached composition must give the same confusion matrix as running
`FoveatedSegmenter` frame by frame. The test is in the suite.

### Selection rules to compare

All rules read only the context pass of the current frame:

- cell score: entropy (current), margin (1 − (p1 − p2)), 1 − max prob;
- pooling within a cell: mean (current) or the mean of its top 10% pixels;
- optional class-prior term: weight the score by the predicted class of each
  pixel (for example up-weighting thin and small classes). The weights are
  tuned on train to maximise mIoU, never a class subset;
- top-k vs greedy with Chebyshev suppression 1;
- oracle, random and fixed-prior, as references.

### Search and selection of finalists

1. Build the caches on train for both models alone and for the mixed pair,
   at every point of the measured grid.
2. For each (pair, scale, size), sweep k and the selection rules, and put
   every result on the mIoU vs predicted-latency plane.
3. Finalists = the train Pareto front, plus the best point at each
   full-frame scale's latency (equal-latency picks), for each pair.
4. Evaluate the finalists once on val, with bootstrap intervals for both
   trade-off readings.
5. Time the finalists end to end on the T4, interleaved with the full-frame
   baselines, for 3 sessions. Report the measured median and p95 next to the
   prediction.

### Outputs

- `experiments/method2/` holds the caches (not committed) and one run JSON
  per evaluated configuration, in the baseline's format plus the selection
  fields.
- `results/method2/README.md` holds the curves (train and val), the bound
  gaps, the finalist table and the T4 confirmation, with the JSONs needed to
  reproduce the plots.
- A notebook or script that redraws every plot from the JSONs.

## What it will answer

- How much the tiling itself costs (tiling bound vs full frame).
- How much room selection has (oracle vs entropy vs random) at each k.
- Whether any foveated point lies above the full-frame curve at equal T4
  latency, for each model, and by how much on val, with intervals.
- Whether a BiSeNetV2 context with PIDNet-S glimpses is worth its 0.6 ms.

## Left to the builder

- File layout, CLI flags, cache format (npz, torch, memmap).
- How to store the context outputs the score variants need, at no coarser
  than the cell-score resolution.
- The bootstrap details (clip-level, ≥ 1,000 resamples, fixed seed).
- Plot style.

## Calls I made (to check)

- Accuracy in fp32 on the laptop; latency from the T4 tables.
- Search limited to the measured latency grid, margin fixed at 32.
- The cached per-cell evaluation instead of running the pipeline for every
  configuration.
- Equal-latency interpolation piecewise linear between the five full-frame
  scales.
- Class-prior weights are allowed as a selection feature, tuned on train for
  mIoU.
