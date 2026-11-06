# Method 3 (previous-frame reuse): recompute where things move

Status: draft for approval, 2026-11-02. No implementation at all (code,
caches, latency runs) starts until the Method 2 finalists' T4 `final` runs
are in (decided 2026-11-02). Points marked *(my call)* were settled by
Claude, not the user.

## Intent

Method 2 showed that, on a T4, a stateless frame can't be cut into a
low-res context plus glimpses and beat the full frame
([results/method2](../../../results/method2/README.md)). Most of a
street scene doesn't change much from one frame to the next, though. This
method keeps the previous frame's prediction and recomputes only the zones
where something moved. It is the first method with a temporal dependency.
It uses only the previous frame: the persistent canvas is Method 4.

The motion cue is the frame difference f_t − f_{t−1} (decided by the user).
Cityscapes is filmed from a moving car, so a raw difference lights up
almost the whole frame. The user hit this when trying frame differencing
before. The fix, decided 2026-11-02: align f_{t−1} to f_t with one global
homography before taking the difference. Then mostly the independently
moving objects remain.

The metric is the standard 19-class mIoU only. The trade-off target is
still open, so the deliverable is full curves. Two latency judges (decided
2026-11-02), because per-frame cost now varies:

- **mean** per-frame T4 latency;
- **p95** per-frame T4 latency, over all frames of all clips.

Both readings from Method 2 are reported against each judge: ΔmIoU at equal
latency, and latency saved at equal mIoU.

## The method, per clip

1. **Frame 0:** a full-frame pass at scale s_full. Its prediction becomes
   the memory: a label map plus a confidence map (max probability).
2. **Each later frame t:**
   1. **Homography:** estimate H mapping f_{t−1} onto f_t, on downscaled
      grayscale frames.
   2. **Motion map:** warp f_{t−1} with H, take |f_t − warped f_{t−1}|,
      smooth it, threshold it at θ. Pixels the warp leaves without a source
      (new content entering at the borders) always count as moving.
   3. **Moving cells:** on Method 2's cell grid, a cell is moving when its
      share of moving pixels is above ρ.
   4. **Fallback:** if the share of moving cells is above τ (a fast turn,
      a crowd crossing), run a full-frame pass at s_full instead and reset
      the memory. τ = 1 means never. The value is picked from the results,
      not in advance (decided 2026-11-02). A fallback frame and frame 0
      count as 100% of cells recomputed; 3b runs no context pass on them
      *(my call)*.
   5. **Otherwise:** warp the memory with the same H, nearest-neighbour for
      the labels. Recompute the moving cells as Method 2 glimpses: full
      resolution, margin 32, with Method 2's replace rule unchanged (the
      cell's inner square comes from its glimpse, seams with the warped
      memory included) *(my call)*. The glimpse model is the full pass's
      model. The result is the prediction for frame t and the new memory;
      a recomputed cell's confidence is its glimpse's max probability.
3. Everything is deterministic, with fixed seeds wherever a step is random
   (RANSAC).

### Two variants (decided 2026-11-02: try both, keep what the numbers support)

- **3a, reuse only:** steps above. Cheapest. Its risk is drift: nothing
  ever refreshes the static parts, so errors and warp smear build up along
  a clip.
- **3b, reuse plus a low-res pass:** each frame also gets a context pass at
  scale s_ctx, as in Method 2. In static cells, the warped memory and the
  low-res prediction are fused with Method 2's conf rule: per pixel, the
  more confident one wins, and the memory keeps that confidence. The mean
  rule is left out, because it would need the 19 class probabilities in
  memory *(my call)*. The user accepts a slightly higher latency if 3a's accuracy loss is too
  large.

## Goals

1. **Bounds before tuning** (on train):
   - **reuse bound:** warp the full-frame prediction forward, recompute
     nothing, and report mIoU against the number of frames since the full
     pass (1 to 29). This is the floor of reuse and shows how fast drift
     grows;
   - **no-homography ablation:** the same motion rule on the raw
     difference, to measure what the homography fix buys;
   - **zone oracle:** at the same number of recomputed cells, the cells
     where the warped memory is most wrong. Labels exist only every 5th
     frame, so the oracle picks cells against the scale-1 full-frame
     prediction as a teacher; it is still scored against the labels on the
     labelled frames. It is a ceiling relative to that teacher, not an
     absolute one *(my call)*.
2. **Sweep on VPS train**, then report chosen configurations once on VPS
   val with clip-bootstrap 95% intervals, as in Method 2. Finalists, as in
   Method 2: the Pareto front plus the best configuration under each
   full-frame latency, taken against both judges (the union) *(my call)*.
3. **Report per configuration:**
   - mIoU, per-class IoU, and mIoU by position in the clip (frames 4, 9,
     …, 29), so drift is visible;
   - mean and p95 latency, GFLOPs, the share of frames that fell back, and
     the mean share of cells recomputed;
   - both trade-off readings against both judges.
4. **Measure the finalists end to end on the T4** (3 sessions, interleaved
   with the full frame), streaming whole clips so the memory is real, on a
   fixed subset of val clips. The sweep's mean and p95 are predicted from
   the floor tables; these runs validate that prediction and replace it in
   the final tables, as in Method 2 *(my call)*.
5. **Ledger every try**, failures included, in `results/method3/TRIES.md`,
   as for Method 2.

## Non-goals

- **No frame skipping:** every frame produces a prediction from at least
  its moving zones. Key-frame-plus-warp schemes are out of scope (decided
  2026-11-02).
- No persistent canvas, no memory older than the previous frame (Method 4).
- No RL and no learned policy. No retraining or fine-tuning of the
  backbones.
- No learned optical flow. The motion cue is the aligned frame difference.
- Hardware other than the T4 is a later step. The design keeps it cheap:
  only the latency tables are device-specific.

## Anti-goals

- Tuning anything on val, or choosing finalists by their val numbers.
- A latency that leaves out the homography, the motion map or the warps,
  wherever they run (CPU or GPU).
- Claiming a gain inside the noise.
- Collapsing the result into one scalar, or reporting only one of the two
  judges.

## Constraints

- **Data.** Method 3 runs on all 30 frames of each clip. Labels are on
  frames 4, 9, 14, 19, 24, 29. Locally, VPS `train/` has only the labelled
  frames, while the raw Cityscapes download (`leftImg8bit_sequence/val`)
  has all 30 frames of every VPS clip, train included: all VPS clips are
  Cityscapes val sequences, and that download is already local
  (`C:/Users/trash/cityscapes`). Val already has `img_all/`. The
  adapter must load the in-between train frames from there, matched by
  Cityscapes sequence and frame number. Check that every train clip has 30
  consecutive frames before building anything.
- **Models:** PIDNet-S and BiSeNetV2, same checkpoints as Method 2, fp32
  for accuracy (check fp16 once, as in Method 2), fp16 with CUDA graphs for
  latency.
- **Search grid (initial):**
  - s_full: 1, 0.75, 0.5;
  - glimpse size 256, 384, 512, margin 32, same model as the full pass;
  - θ and ρ: a small grid, set from the motion-map statistics on train;
  - τ: 0.2, 0.4, 0.6, 1;
  - 3b: s_ctx 0.25, 0.375, 0.5, fusion mean or conf.

  The builder can widen it if the bounds show headroom.
- **Latency model.** The number of recomputed cells varies per frame. The
  latency of a frame is predicted from the floor tables: area for a full or
  context pass, batch for k glimpses, overhead, plus new motion parts. The
  per-frame values give the mean and the p95. The current tables stop at
  k = 8 and have no motion parts, so a `latency_floor` extension is needed:
  - batch and overhead for every k up to the largest grid's cell count;
  - the homography, the motion map and both warps, timed as deployed;
  - a 3b overhead with the fusion included.

  It runs on the T4 in 3 sessions before tuning starts.
- **Homography estimator.** It must be deterministic, run on downscaled
  grayscale frames, and be timed. Feature matching with RANSAC (OpenCV, on
  the CPU) is the default. The latency extension times it on the Colab T4
  host's CPU, which is the canonical number, and also times one GPU
  estimator, so the choice rests on data *(my call)*. Record the choice in
  the ledger.

## How it works: evaluation

The prediction at frame t depends on the previous memory, so Method 2's
per-cell confusion composition doesn't apply directly. The current frame's
inputs don't depend on history, though:

- the full-frame prediction at each s_full;
- the every-cell glimpse prediction at each glimpse size;
- the 3b context prediction and its max-probability map;
- the homography and the motion map.

So they can be computed once per frame on the laptop GPU and stored. A
cheap simulation then replays each configuration's memory, warps,
fallbacks and fusion over the clip, and scores the labelled frames. Only
label maps and max-probability maps (uint8-quantised) are stored, no full
probabilities *(my call)*. The storage format is left to the builder;
compressed maps of 12,000
train frames are a few GB per geometry.

**Validation.** On 2 val clips and 3 configurations (3a, 3b, one with
fallbacks), the replay must match running the real method frame by frame,
pixel for pixel. The test goes in the suite.

## Left to the builder

- Downscale factor and feature type for the homography, the smoothing of
  the motion map, and the exact θ and ρ grids.
- Cache format and layout.
- How the confidence map is warped (bilinear or nearest).
- Plot design.
