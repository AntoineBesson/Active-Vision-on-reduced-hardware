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

## 2026-11-14 to 16 — chunk 0 on the laptop: too hot, moved to Colab

Plan task 6, step 2, approved by the user. `bound --chunk 0/4`, full grid,
both models, laptop RTX 2060 Max-Q, at `52ceec3`. About 29 s per clip for
PIDNet-S, as extrapolated from the pipeline check.

- 2026-11-14: stopped after 14 clips, the GPU holding 88 °C (stop rule then:
  above 87 °C for 1 min).
- 2026-11-16: resumed after cooling. The GPU sat at 87 °C, throttled to about
  975 MHz, which is the state in which this laptop crashed during Method 2,
  and the 87 °C rule never fired. The guard was tightened to pause at 85 °C on
  two readings or on any thermal-slowdown flag, and to resume at 65 °C with no
  other job on the GPU. It held the run while the moving-object full pass used
  the GPU (08:41 to 08:53 local).
- Under that rule the laptop heats to 86 °C about 75 s after each restart and
  needs about 20 min to cool: about 2 clips per 22 min, so about a day for the
  rest of the chunk. Stopped at 66 PIDNet-S clips (0001 to 0082, 10 032 lines,
  read cleanly). The user chose Colab for the rest. The chunk zips were no
  longer on Drive (as in Method 4); chunk 0's was rebuilt on the laptop
  (7.41 GB, 100 clips) for upload.

The school cluster (P100s, used for Method 3) needs the Telecom Paris OpenVPN,
which asks for a password at each connection; one SSH test timed out.

## 2026-11-16 — first look on 66 PIDNet-S clips (not a reading)

`score` on the partial PIDNet-S file of chunk 0 (CPU only), to answer whether
the bound is worth finishing. The rule is not read (66 of 400 clips, one
model; the best of 72 configurations is picked on the same clips).

- Pipeline check: every uniform row within 2e-7 of the full frame.
- Best **now**: `5o/c0.25/E/a0.25/s1/now`, +1.92 points above the full frame
  at 0.25 (0.5089 against 0.4897). Best **prev**: `c0.25/E/a0.25/s4/prev`,
  +1.80.
- Best per scale (now / prev): 0.25 +1.92 / +1.80; 0.375 +0.67 / +0.51; 0.5
  +1.41 (`S/a0.25/s4`) / +1.11; 0.75 +0.72 / +0.74.
- Best per saliency (now): E +1.92, S +1.70, B +1.00. α = 0.25, the weakest
  warp of the grid, wins at every scale and for every saliency.
- Per class at 0.25 against the uniform pass: pole +3.1, bicycle +2.8,
  traffic sign +0.9, person +0.8, motorcycle −1.5, wall −6.8 points.
- On the curves (information only): primary +0.29 (`c0.75/S/a0.25/s1`), end
  to end +0.87 (`c0.5/S/a0.25/s4`).

Full-frame mIoU on these clips: 0.7378, 0.7243, 0.6506, 0.5951, 0.4897 at
scales 1, 0.75, 0.5, 0.375, 0.25.

## 2026-11-16 — chunk 0 finished on the laptop

The user chose to let the laptop cycle under the heat guard (pause at 85 °C
on two readings or on a thermal-slowdown flag, resume at 65 °C with no other
GPU job) rather than wait for the Colab upload. From the restart at 09:50 to
15:11 local (5.35 h), the guard paused 89 times, always on the software
thermal slowdown at 86 to 87 °C, and the run spent 3.54 h (66 %) paused.
While running: about 28 s per clip for PIDNet-S, 34 s for BiSeNetV2. Peak
87 °C. No traceback, no process death.

Both files read cleanly: `bound_train_all_pidnet_s_c0of4.jsonl.gz` and
`bound_train_all_bisenetv2_c0of4.jsonl.gz`, 15 200 lines each (100 clips ×
152, clips 0001 to 0124). On this laptop and this rule, a chunk takes about
5 h for both models. `merge` and `score` of chunk 0 are left to another
session at the user's request.


## 2026-11-16 — decisions taken before the reading on the 400 clips

Taken while chunk 0 was running (PIDNet-S done, BiSeNetV2 started), before
any `score` of a complete chunk. Nothing below changes `bound`; merged into
the branch after chunk 0 finished.

- **Budget on a curve that does not rise.** `budget_ms` was NaN whenever the
  full-frame mIoU does not rise strictly with scale, which is the case of
  BiSeNetV2 on the 400 clips (0.7339 at 0.75, 0.7335 at 1). It is now read on
  the frontier of the curve priced at `area` + `post`: the cheapest full frame
  that reaches the configuration's mIoU, a scale beaten by a cheaper one being
  dropped (`FullFrameCurve.latency_at(..., frontier=True)`). Same number as
  before on a rising curve; the judges' own readings and Methods 2 to 4 are
  not touched. The budget is not in the pass rule.
- **Moving-object recall (user: worth reading if cheap).** `bound` keeps one
  confusion per clip and configuration, so per-object recall cannot be read
  from its files, and chunk 0 was already running. Rule fixed now: once the
  mIoU reading is done, per model, the best **now** and the best **prev**
  configuration of each scale by mIoU (8 configurations, chosen on mIoU only)
  are run again on the labelled frames with a prediction pass scored by
  `eval.objects` (found at local IoU > 0.5, frozen bins), and read against the
  full frame's recall at the same scale, already scored on these clips. About
  8 passes per frame in place of 148. Reported only: it does not enter the
  pass rule. Not written yet (it needs the objects branch merged).
- The 16 GB of Method 3 per-clip caches were deleted (user's request). They
  can be rebuilt with `scripts.method3 cache`; the objects scores made from
  them are kept.
