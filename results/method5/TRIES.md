# Method 5 (one pass on a warped frame): ledger of tries

Every try, failures included, for the paper. Phase 1 is a train-only
feasibility bound (`train_all`, 400 clips, 2400 labelled frames); val is not
read in it.

## 2026-11-12 — spec accepted before any run

Grid, latency parts and pass rule are in
`docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md`, accepted by
the user on 2026-11-12, before any code or run. Three details were made
precise while writing the plan, marked *(plan)* in the spec; none touches the
grid, the latency or the pass rule:

- **Antialiasing:** the baseline's downscale is antialiased and a plain
  `grid_sample` is not, so the warped image is sampled at twice the output
  size and averaged 2 × 2.
- **Unwarping:** the logits are sampled once, at the model's output stride and
  with its own corner convention, so that at α = 0 it is exactly the
  baseline's upsampling.
- **First frame:** a labelled frame at t = 0 has no previous frame; the
  **prev** timing then uses frame 0 itself. This does not occur on
  `train_all`.

## 2026-11-13 — the 2× supersampling does not reproduce the baseline at α = 0

Found while testing the warp, before any run. A 2 × 2 mean is a box filter;
the baseline's resize (`F.interpolate(..., antialias=True)`) is a triangle
filter whose width is the downscale factor. At α = 0:

- On a test photograph (matplotlib's `grace_hopper.jpg`, 600 × 512), mean
  absolute difference to the baseline's resize, per pixel: 2.19/255 at scale
  0.25, 1.71 at 0.375, 2.15 at 0.5, 0.18 at 0.75 (99th percentile 18.7, 14.0,
  18.4, 1.8). A plain `grid_sample` is worse: 5.44, 3.85, 2.15, 1.09. 4×
  supersampling lowers it at 0.375 and 0.5 (1.23, 1.04) but not at 0.25.
- On the synthetic fixture with the dummy backbone, mIoU of the uniform warp
  against the baseline's resize: 0.339 vs 0.300 at 0.25, 0.541 vs 0.459 at
  0.5, 0.613 vs 0.609 at 0.75. The 0.1-point pipeline check would fail.
  Fixture only: it says the gap can matter, not how much it does on
  Cityscapes.
- Where the warp compresses (down to 8× locally at c = 0.25, d = 0.5), a 2 × 2
  mean under-filters.

Added `--resample antialias`: per axis, a triangle filter whose half-width is
the number of source pixels each output pixel covers (the inverse map's
local span, at least 1), normalised per output pixel, as torch's own
antialiased kernel. At α = 0 it equals the baseline's resize to float
precision at every scale (tested). A bound file, `merge` and `score` refuse
rows of mixed resamplings. Settled the same day, see below.

## 2026-11-13 — the uniform rows start below the primary curve

Found while writing `score`, before any run. The spec prices a configuration
at `area` (frame at scale c) plus `reuse` `post` (upsample and argmax, 2.1 to
4.5 ms in the T4 tables). The primary full-frame curve (`fullframe_ref`)
counts `area` only. A uniform row is the full frame itself. Taking its mIoU
as the full frame's at that scale (Method 4's curve on the 400 clips) and
its latency from the tables, it sits at (ΔmIoU points at equal mean
latency):

| Model | c = 0.25 | 0.375 | 0.5 | 0.75 |
|---|---|---|---|---|
| PIDNet-S, primary curve | −13.72 | −7.14 | −2.82 | −0.41 |
| PIDNet-S, measured end to end | −6.10 | −2.50 | −0.51 | −0.07 |
| BiSeNetV2, primary curve | −8.08 | −3.15 | −0.94 | +0.01 |
| BiSeNetV2, measured end to end | −2.71 | −1.02 | −0.34 | +0.01 |

So a warp must first make up the full frame's post-processing before it
reaches the curve, mostly at small scales. The spec's question ("at the same
input size") reads mIoU against the full frame at the same scale. `score`
reported both readings while the user decided. Settled the same day, see
below.

## 2026-11-13 — the labelled+previous zip cannot be read as `train_all`

Plan task 5, step 1. `train_all` groups the frames of a sequence and cuts a
clip at every frame gap larger than `max_frame_gap` = 1. On a tree holding
only the labelled frames and the frame before each, a clip of 30 frames
becomes six 2-frame clips with new ids (`0001#2`, …): checked on a temporary
tree of 2 clips, which gave 12 clips. The clip ids would not match the
full-frame reference or the chunks. So the zip is not built (plan steps 2 and
3 skipped) and the Colab notebook uses the four chunk zips of the Method 3
notebook, already on Drive (about 7 GB each). A split with a larger
`max_frame_gap` could read such a zip, but is not needed.

## 2026-11-13 — the spec is amended before any run

The user settled both points above, before any `bound` run, in a dated
Amendments section of the spec:

- **Resampling:** the antialiased filter (`--resample antialias`) is the
  default. The uniform row must reproduce the baseline, otherwise a warped
  number mixes the warp with a worse downscale. Its inference cost, above a
  plain `grid_sample`, is counted at 0 ms in the bound and comes out of the
  phase 2 budget.
- **Pass rule:** the best **now** configuration's mIoU minus the full
  frame's at the same scale, ≥ +1.0 point. The rule as first written charged
  the post-processing to the warped pass but not to the curve. Margin, timing
  and pipeline check unchanged; the primary and end-to-end curves are
  reported in `verdict.curves`.

Following from the second (Claude's reading, noted in the spec): the budget
(`budget_ms`) is read on the full frame priced at `area` + `post`
(`judges_same_parts`). On that curve, a configuration's ΔmIoU at equal latency
is exactly the rule's reading, which a test checks.

## 2026-11-14 — pipeline check on 5 clips: passes

Plan task 6, step 1, approved by the user. PIDNet-S, laptop RTX 2060 (torch
2.14.1, cu126), fp32, at `ae29b4b` (resampling `antialias`). The first 5 clips
of `train_all` (`--chunk 0/80`: 0001, 0002, 0003, 0004, 0006), grid E, α 0.5,
σ 1, timing now, plus the uniform rows and the baseline's resize at each
scale, in `experiments/method5/pipecheck/`.

Each uniform row against the full-frame baseline on the same clips (mIoU):

| Scale | uniform | baseline | gap |
|---|---|---|---|
| 0.25 | 0.39505 | 0.39505 | +2.6e-7 |
| 0.375 | 0.44934 | 0.44934 | +7.0e-7 |
| 0.5 | 0.49421 | 0.49421 | −1.0e-7 |
| 0.75 | 0.61518 | 0.61518 | +5.1e-7 |

All four are within the 0.001 tolerance by four orders of magnitude, and the
run's own resize equals the baseline file exactly at every scale, so the
warp, the unwarp and the reference agree. The rule is not read (5 clips).

Time: 3.2 s per clip (4.4 s for the first), 24.5 s for `bound`, 4.6 s for
`score`; GPU at 70 °C at most. Extrapolated to the full grid (144 warped
passes per frame instead of 4): about 30 to 40 s per clip, so roughly 1 h per
model per chunk and 4 to 5 h per model on the laptop, above the spec's 2 to
3 h. Chunk 0 will measure it.

Not a reading, kept only as a smoke result: on these 5 clips the warped E
configurations are −1.10, +0.73, +1.13 and −2.81 points against the full
frame at scales 0.25, 0.375, 0.5 and 0.75. One α, one σ, 5 clips.

`budget_ms` is NaN for every row. The curve code (`FullFrameCurve.latency_at`,
unchanged) returns NaN unless the full-frame mIoU rises strictly with scale.
On these 5 clips it most likely does not between 0.75 and 1 (not checked:
the scale-1 mIoU was not in the report). The budget is reported only and
does not enter the rule.

