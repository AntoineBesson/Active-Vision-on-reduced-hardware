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

## 2026-11-16 — chunk 0 merged and scored (100 clips, partial, rule not read)

`merge` then `score` on chunk 0 alone, CPU, at `4897742` (frontier budget).
100 of 400 clips, both models, 148 configurations each. The rule is not read:
a quarter of the clips, and the best configuration is picked on those clips.

- Pipeline check: passes for both models at the four scales (gaps below 0.01
  point).
- Best **now** against the full frame at the same scale, in mIoU points, with
  the budget in ms (cheapest full frame at the same mIoU, minus the pass):

  | Scale | PIDNet-S | BiSeNetV2 |
  |---|---|---|
  | 0.25 | +2.10 (`E/a0.25/s1`), 0.23 ms | +1.55 (`B/a0.25/s1`), 0.53 ms |
  | 0.375 | +1.12 (`S/a0.25/s1`), 0.29 ms | +1.22 (`B/a0.25/s4`), 1.02 ms |
  | 0.5 | +1.46 (`S/a0.25/s4`), 1.15 ms | +0.30 (`B/a0.25/s4`), 1.01 ms |
  | 0.75 | +0.40 (`S/a0.25/s1`), 2.31 ms | +0.34 (`S/a0.25/s1`), 12.09 ms |

- Best **prev**: PIDNet-S +1.45 / +1.00 / +1.24 / +0.35 at 0.25 / 0.375 / 0.5
  / 0.75; BiSeNetV2 +1.36 / +1.61 / +0.10 / +0.28. The saliency of the
  previous frame keeps most of the gain.
- α = 0.25 wins everywhere. Best at α = 0.5: +1.33 (PIDNet-S), +0.59
  (BiSeNetV2); at α = 0.75: +0.11 and +0.16. The grid's weakest warp is its
  best, so a weaker one is not covered.
- Configurations with timing now at +1.0 point or more: 11 of 72 (PIDNet-S),
  6 of 72 (BiSeNetV2).
- The budgets are small where the gain is: 0.2 to 1.2 ms below scale 0.75,
  because the full-frame curve is steep there (a slightly larger full frame
  buys the same mIoU for about 1 ms). BiSeNetV2's 12 ms at 0.75 comes from its
  flat top (0.7283 at 0.75, 0.7318 at 1 on these clips).
- `budget_ms` is NaN for 18 (PIDNet-S) and 24 (BiSeNetV2) of the 148
  configurations: those whose mIoU is below the full frame at 0.25, off the
  curve.

Full-frame mIoU on these clips, scales 1 / 0.75 / 0.5 / 0.375 / 0.25:
PIDNet-S 0.7577, 0.7439, 0.6672, 0.6095, 0.5073; BiSeNetV2 0.7318, 0.7283,
0.6963, 0.6516, 0.5789.

## 2026-11-16 — extension decided after chunk 0: α = 0.1 and 0.15 (post hoc)

The user's decision, after the chunk 0 scores above (α = 0.25 best everywhere,
on the edge of the grid). Two weaker strengths are added as an extension, run
in `experiments/method5/train_all_ext` with `--alphas 0.1,0.15`: 96
configurations per model on the 400 clips, about two thirds of the registered
grid's passes plus the teacher and uniform passes again. No code change; the
command line was tried on the synthetic fixture (`bound` and `merge`).

The registered rule stays on α ∈ {0.25, 0.5, 0.75}. The extension is reported
next to it as chosen after looking at a quarter of the data (spec amendment of
2026-11-16). Not run yet: it goes to the cluster with chunks 1 to 3.

## 2026-11-16 — second extension decided after chunk 0: scale 1 with magnification (post hoc)

The user's objection to the chunk 0 reading: a T4 runs the full frame at
scale 1, and there the registered warp is the identity, so nothing in the
grid can move the operating point. Extension: `--scales 1 --d-max 2` (density
up to 2, i.e. sampling up to twice finer than the source pixels), saliencies
B and S, α ∈ {0.1, 0.25, 0.5}, in `experiments/method5/train_all_mag` (24
configurations per model). `d_max` is a new optional argument of
`axis_density` / `make_grid` / `bound_clip`; the default path is unchanged
(tests). Tried on the synthetic fixture only. Not run yet.

The α = 0.1 / 0.15 extension is kept (the user first read them as image
scales; they are warp strengths at the same cost).

Also noted for the paper: the warp is separable, which the user had assumed
local. The next research points with frozen weights (local 2-D warp first,
fixed priors, related work) are in
`docs/research/2026-11-16-warp-next-research-points.md`.

## 2026-11-16 — chunks 1 to 3 of the registered grid sent to the school cluster

The user connected the VPN and asked for chunks 1 to 3. Commit `94d5618`
(`warp-bound`), job script `scripts/cluster/m5_bound.sh`, Tesla P100, one job
per chunk and model, registered grid only (148 configurations + 4 resize refs).

| Chunk | PIDNet-S job | BiSeNetV2 job | Submitted |
|---|---|---|---|
| 3 | 1107212 | 1107213 | 19:56 |
| 1 | 1107234 | 1107235 | 20:13 |
| 2 | 1107250 | 1107251 | 20:24, waiting on the 4-jobs-per-user limit |

- Data: the cluster only had the Method 3 zip of chunk 3. The zips of chunks 1
  and 2 were rebuilt on the laptop (7.27 and 7.12 GB) and uploaded at about
  11 MB/s; the first `scp` was cut by a network drop after 1.1 GB, the retry
  with `sftp reput` resumed it, and both md5 sums match the laptop's.
- The job's root holds one chunk only, so `bound` runs without `--chunk` and
  with `--tag c<i>of4`: the zip chunks and `--chunk i/4` are both the i-th
  contiguous block of the sorted clip ids (same 100 clips).
- Speed while running: about 18.5 s per clip for PIDNet-S and 21 s for
  BiSeNetV2 (the laptop: 28 and 34 s when not paused), so about 35 minutes per
  chunk and model.
- State when this was written: chunk 3 at 87 and 76 clips of 100, chunk 1 at
  35 and 29, chunk 2 queued. **Not finished, not copied back, not merged, not
  scored: the rule is not read.** The two extensions are not submitted (they
  also need the chunk 0 zip on the cluster).


## 2026-11-16 — chunks 1 to 3 done on the cluster; the reading on the 400 clips: the check PASSES

All six jobs ended `COMPLETED` with `JOB_OK` in their logs, at commit
`94d5618`, on Tesla P100-PCIE-16GB nodes. No failure, no resubmission.

| Job | Chunk | Model | Node | Elapsed | s per clip |
|---|---|---|---|---|---|
| 1107212 | 3 | PIDNet-S | node17 | 31:44 | 18.7 |
| 1107213 | 3 | BiSeNetV2 | node18 | 35:45 | 21.1 |
| 1107234 | 1 | PIDNet-S | node19 | 30:59 | 18.1 |
| 1107235 | 1 | BiSeNetV2 | node20 | 35:23 | 20.8 |
| 1107250 | 2 | PIDNet-S | node17 | 31:06 | 18.4 |
| 1107251 | 2 | BiSeNetV2 | node18 | 35:55 | 21.2 |

- Copied back to the laptop with `scp`: each of the 6 bound files has 15200
  lines (100 clips × 152 rows: 148 configurations and 4 resize refs), all with
  `resample = antialias`. Clip ids: chunk 1 0126 to 0249, chunk 2 0251 to
  0374, chunk 3 0376 to 0499; chunk 0 (laptop) 0001 to 0124. No overlap.
- `merge`: 4 files per model, 60800 lines, 400 clips. `score` on the laptop,
  then `results/method5/verdict_train_all.json` and the two
  `score_train_all_*.json.gz` committed (`18a6d2b`).
- **Pipeline check: passes** for both models at the four scales. The largest
  gap between a uniform row and the full-frame baseline is 2.2e-7 mIoU
  (tolerance 1e-3); the `ref/c*/resize` rows agree too.
- **Rule (amended 2026-11-13): PASSES for both backbones**, on the 400 clips:

  | | PIDNet-S | BiSeNetV2 |
  |---|---|---|
  | best **now** | `c0.25/E/a0.25/s1` | `c0.25/E/a0.25/s4` |
  | mIoU (full frame at c 0.25) | 0.5410 (0.5285) | 0.6099 (0.5919) |
  | Δ vs full frame, same scale | **+1.25** pts | **+1.80** pts |
  | predicted latency (area + post) | 4.86 ms | 5.10 ms |
  | budget (frontier, same parts) | 0.16 ms | 0.58 ms |
  | best **prev** (same config) | +0.98 pts, 0.12 ms | +1.59 pts, 0.51 ms |

  prev stays above the full frame for both, so no "under water" note.

- Best **now** per scale (Δ vs the full frame at that scale, budget):

  | Scale | PIDNet-S | BiSeNetV2 |
  |---|---|---|
  | 0.25 | +1.25 (`E/a0.25/s1`), 0.16 ms | +1.80 (`E/a0.25/s4`), 0.58 ms |
  | 0.375 | +0.91 (`E/a0.25/s4`), 0.22 ms | +0.73 (`E/a0.25/s4`), 0.73 ms |
  | 0.5 | +0.86 (`B/a0.25/s1`), 0.76 ms | +0.08 (`B/a0.25/s4`), 0.31 ms |
  | 0.75 | +0.15 (`S/a0.25/s1`), 0.91 ms | +0.03 (`B/a0.25/s1`), NaN |

  Best prev per scale: PIDNet-S +0.98 / +0.55 / +0.81 / +0.15; BiSeNetV2
  +1.59 / +0.63 / −0.00 / −0.10.
- Configurations with timing now at +1.0 point or more: 2 of 72 (PIDNet-S),
  4 of 72 (BiSeNetV2); prev: 0 and 2. Mean Δ of the now configurations per
  scale (0.25 to 0.75): PIDNet-S −2.75 / −2.37 / −1.66 / −0.36, BiSeNetV2
  −3.52 / −4.18 / −3.21 / −0.69. Most of the grid loses.
- Best now per saliency: E +1.25 / +1.80, B +0.86 / +1.29, S +0.41 / +0.48
  (PIDNet-S / BiSeNetV2). Per strength: α 0.25 +1.25 / +1.80, α 0.5 +0.00 /
  +0.04, α 0.75 −0.09 / −0.54.
- Per class, best now against the uniform pass at the same scale: PIDNet-S
  bus +5.4, motorcycle +4.3, bicycle +2.9, pole +2.7, traffic sign +1.9, truck
  +1.7 … train −1.6, wall −0.5, road −0.3. BiSeNetV2 truck +6.6, train +6.3,
  bus +5.0, fence +2.9, terrain +2.8, bicycle +2.4, pole +1.7 … rider −0.6.
- Curves (information only, `verdict.curves`): at equal mean latency on the
  primary curve the best is PIDNet-S −0.26 (`c0.75/S`), BiSeNetV2 +0.04; on
  the measured end-to-end curve +0.36 (`c0.5/B`) and +0.03. Once the full
  frame's scale is free to move, the gain is gone.
- Against chunk 0 alone, PIDNet-S's best fell from +2.10 to +1.25 and its
  count at +1 or more from 11 to 2; BiSeNetV2's best rose from +1.55 to
  +1.80. Chunk 0 was the easiest quarter for PIDNet-S.

What the pass does and does not say, for the decision on phase 2:

- It is narrow: only scale 0.25 and α 0.25 clear +1.0, and α 0.25 is the
  weakest warp of the grid, so the optimum may lie below it (the post hoc
  α = 0.1 / 0.15 extension, not run, would say).
- It is the E oracle, which reads the teacher's error on the current frame:
  the one saliency a real policy cannot see. B (boundaries) clears the margin
  for BiSeNetV2 only (+1.29), S for neither.
- The best is picked in-sample, out of 72 configurations, on the clips it is
  scored on.
- The budgets are 0.16 ms (PIDNet-S) and 0.58 ms (BiSeNetV2) for the
  antialiased resampling with a local width, the unwarping and the policy,
  all counted at 0 ms here.
- Scale 0.25 is far below the full frame's useful range on a T4 (PIDNet-S
  full frame at scale 1: 0.7608 in 19.4 ms).

## 2026-11-16 — α 0.1 / 0.15 extension sent to the cluster (post hoc)

The user's go, after the registered reading. Run `train_all_ext` of
`scripts/cluster/m5_bound.sh` (`--alphas 0.1,0.15`, the registered scales,
saliencies, smoothings and timings: 96 configurations per model plus the
uniform rows), all four chunks, Tesla P100, cluster repository at `f0f9c01`.
The scale 1 magnification extension (`train_all_mag`) is not submitted.

| Chunk | PIDNet-S job | BiSeNetV2 job |
|---|---|---|
| 0 | 1107366 | 1107367 |
| 1 | 1107368 | 1107369 |
| 2 | 1107370 | 1107371 |
| 3 | 1107372 | 1107373 |

- Chunk 0 had run on the laptop, so its zip was not on the cluster. The zip
  built on the laptop on 2026-11-16 (7,405,269,880 bytes, clips 0001 to 0124)
  was uploaded with `sftp put` in 675 s, about 11 MB/s, without a drop. Its md5
  matches on both sides (`544491c666be2a633d4a3ed871a45f50`).
- The first submission was refused by the laptop's permission check, because
  the user's go had reached it through the project chat; it went through once
  the user wrote it in the thread.
- Expected: about 25 minutes per job, 4 at a time, about 1 h in all. **Not
  finished, not copied back, not scored.** Its verdict is not the pass rule.

## 2026-11-16 — α 0.1 / 0.15 extension done and scored (post hoc, not the rule)

All eight jobs (1107366 to 1107373, nodes 16 to 21) ended `COMPLETED` with
`JOB_OK` on the first try, 21 to 25 minutes each. No resubmission. Copied back
to the laptop: 8 files of 10,400 lines (100 clips × 104 rows: 96
configurations, 4 uniform rows, 4 resize refs). `merge` in
`experiments/method5/train_all_ext`: 41,600 lines and 400 clips per model.
`score` there; the verdict and scores are committed in `results/method5/ext/`
(`a6da6f9`).

- **Pipeline check: passes** for both models at the four scales (largest gap
  2.1e-7 mIoU).
- The extension's own verdict file says BiSeNetV2 "passes" and PIDNet-S
  "fails" (+0.996, 0 of 48 at +1.0). **It is not the pass rule**, which is
  read on the registered grid only (spec amendment of 2026-11-16).
- Best **now** against the full frame at the same scale, all four strengths
  (α 0.25 and 0.5 from the registered grid), in mIoU points:

  | Scale | PIDNet-S α 0.1 / 0.15 / 0.25 / 0.5 | BiSeNetV2 α 0.1 / 0.15 / 0.25 / 0.5 |
  |---|---|---|
  | 0.25 | +0.86 / +0.96 / **+1.25** / −0.05 | +1.20 / +1.48 / **+1.80** / +0.04 |
  | 0.375 | +0.69 / **+0.92** / +0.91 / −0.39 | +0.98 / **+1.17** / +0.73 / −1.28 |
  | 0.5 | +0.78 / **+1.00** / +0.86 / −0.48 | +0.31 / **+0.33** / +0.08 / −1.67 |
  | 0.75 | **+0.21** / +0.10 / +0.15 / … | +0.07 / −0.09 / +0.03 / … |

- The best strength falls as the scale grows: α 0.25 at scale 0.25, α 0.15
  at 0.375 and 0.5. α 0.1 is below α 0.15 everywhere except at scale 0.75,
  where every gain is under 0.25 point. The registered grid's edge
  (α 0.25) was therefore the optimum only at scale 0.25.
- Budgets (frontier, same parts) grow with the scale: BiSeNetV2 at 0.375 with
  α 0.15 keeps +1.17 points for 1.18 ms (0.58 ms for its best at 0.25);
  PIDNet-S at 0.5 with α 0.15 keeps +1.00 for 0.87 ms (0.16 ms at 0.25).
- **prev**: PIDNet-S +1.00 at c0.5 (`E/a0.15/s4`, the same as now to 5e-6),
  +0.99 at c0.25; BiSeNetV2 +1.14 at c0.25 (`E/a0.1/s1`), +0.96 at c0.375.
- Best per saliency (now): PIDNet-S E +1.00, B +0.83, S +0.73; BiSeNetV2 E
  +1.48, S +0.96, B +0.81. The weaker warps help the small-class saliency S
  most (registered grid: +0.41 and +0.48).
- `budget_ms` is NaN for some BiSeNetV2 configurations at scale 0.75 and
  negative for one: its full-frame curve is flat at the top (0.7339 at 0.75,
  0.7335 at 1), so an mIoU above the full frame at 0.75 is reached by no full
  frame (NaN), and a negative Δ gives a negative budget. Same reading as in
  phase 1.

For phase 2 (open point 1 of its draft spec): α ∈ {0.15, 0.25, 0.35} covers
the best strength at every scale kept; α 0.1 adds nothing.

## 2026-11-16 — scale 1 magnification extension sent to the cluster (post hoc)

The user's go, before accepting the phase 2 spec: it is the only version that
can move the operating point a T4 uses, and if it gains, phase 2's scales
change. Run `train_all_mag` of `scripts/cluster/m5_bound.sh` (`--scales 1
--d-max 2 --saliencies B,S --alphas 0.1,0.25,0.5`, σ 1 and 4, both timings:
24 configurations per model plus the uniform row), all four chunks, Tesla
P100, cluster repository at `d843327`. Every pass is at full resolution.

| Chunk | PIDNet-S job | BiSeNetV2 job |
|---|---|---|
| 0 | 1107500 | 1107504 |
| 1 | 1107501 | 1107505 |
| 2 | 1107502 | 1107506 |
| 3 | 1107503 | 1107507 |

The chunk 0 zip was already on the cluster. **Not finished, not scored.**

## 2026-11-16 — scale 1 magnification extension done and scored (post hoc, not the rule)

All eight jobs (1107500 to 1107507) ended `COMPLETED` with `JOB_OK` on the
first try, 11 to 15 minutes each. No resubmission. Copied back: 8 files of
2,600 lines (100 clips × 26 rows: 24 configurations, the uniform row, the
resize ref). `merge` in `experiments/method5/train_all_mag`: 10,400 lines and
400 clips per model. Verdict and scores committed in `results/method5/mag/`
(`24a73e0`, on 2026-11-17 after the user's go).

- **Pipeline check: passes** at scale 1 (gaps 2.7e-8 and −1.5e-8).
- **No gain worth reading.** Best against the full frame at scale 1, in mIoU
  points:

  | | PIDNet-S (FF 0.7608) | BiSeNetV2 (FF 0.7335) |
  |---|---|---|
  | best now | +0.36 (`B/a0.1/s1`) | −0.02 (`B/a0.1/s4`) |
  | best prev | +0.31 (`B/a0.1/s1`) | −0.01 (`B/a0.1/s4`) |
  | B, α 0.1 / 0.25 / 0.5 (now) | +0.36 / +0.04 / −1.34 | −0.02 / −0.85 / −3.71 |
  | S, α 0.1 / 0.25 / 0.5 (now) | +0.06 / −1.11 / −4.78 | −0.15 / −2.53 / −7.90 |
  | configurations above 0 | 4 of 12 now, 4 of 12 prev | 0 of 12 now, 0 of 12 prev |

- Magnifying beyond the source pixels gains on large rigid classes and loses
  on people: PIDNet-S's best gains motorcycle +2.2, train +2.0, bus +1.1 and
  loses rider −1.4, person −0.5; BiSeNetV2's best loses rider −2.2,
  motorcycle −0.9, person −0.7.
- `budget_ms` is NaN for every positive configuration (no full frame on the
  curve reaches a higher mIoU than scale 1) and strongly negative otherwise.
  For BiSeNetV2, −13 ms: its full frame at 0.75 already matches scale 1.
- **Consequence for phase 2:** the operating point a T4 uses (scale 1) cannot
  be moved by this warp, with an oracle saliency. +0.36 point is below what 50
  val clips can resolve, the user's criterion for leaving BiSeNetV2 at 0.5
  out. The scales of the phase 2 spec stay as drafted.

## 2026-11-17 — phase 2, stage A: the warp step timed on a T4; gate A says continue, with the static prior P only

Spec `docs/superpowers/specs/2026-11-16-method5-phase2-design.md` and plan
`docs/superpowers/plans/2026-11-17-method5-phase2.md`, both accepted by the
user on 2026-11-17 before this run. Code: tasks 1 to 3 (`d567457`,
`2ed0e92`, `93ba77d`).

Three Colab T4 sessions run by the user (`m5w-s1` 08:06, `m5w-s2` 08:15,
`m5w-s3` 08:48, local time of the runtime; 426 to 450 s each; torch
2.14.1+cu130), `latency_floor --parts warp`, graph level, FP16, batch 1. The
CUDA graph test of the step passed on the T4 first. Per model and scale 0.25 /
0.375 / 0.5: the full frame and 26 variants of the step (r1/r2/r3 × u1/u2 ×
B/S/U/P at 50 density rounds, B and U at 10 rounds on r2/u2), α 0.25, σ 4.
163 records per session: BiSeNetV2 at 0.5 did not fit in one chunk and was
split in two, the full frame repeated (chunks 5 and 6). SM clock median 1185 /
1275 / 1200 MHz, maximum temperature 81 / 82 / 83 °C, `sw_power_cap`
throttling only, as in every T4 session so far. The full frame here is within
3 % of `latency_floor_final` at every scale.

`python -m scripts.method5 cost` → `results/method5/phase2/gate_a.json`
(sessions and gate committed in `c51cfe5`).
Δ = median over the sessions of (W − FF) paired in its chunk, ms, against
the phase 1 budget (the registered grid's best `now` configuration on the
measured end-to-end curve):

| | budget | P, r3u1 | P, r2u1 | cheapest B or S | cheapest U |
|---|---|---|---|---|---|
| PIDNet-S 0.25 | 0.176 | −0.08 fits | +0.06 fits | +1.70 (B r3u1) over | +1.58 (r3u1) over |
| PIDNet-S 0.375 | 0.262 | −0.16 fits | +0.13 fits | +1.62 (B r3u1) over | +1.57 (r3u1) over |
| PIDNet-S 0.5 | 0.754 | −0.14 fits | +0.28 fits | +1.57 (B r3u1) over | +1.65 (r3u1) over |
| BiSeNetV2 0.25 | 0.552 | −0.12 fits | +0.07 fits | +1.62 (B r3u1) over | +4.69 (r3u1) over |
| BiSeNetV2 0.375 | 0.676 | −0.20 fits | +0.17 fits | +1.45 (B r3u1) over | +8.50 (r3u1) over |
| BiSeNetV2 0.5 (not in stage B) | 0.284 | −0.17 fits | +0.31 straddles | +1.52 over | +14.5 over |

- **Verdict: continue**, by the spec's rule "if only P fits, stage B goes on
  with P alone". Every model and scale of stage B fits with P; no variant
  with a saliency computed per frame (B, S, U) fits anywhere, and the
  smallest of them is 1.45 ms over the full frame, 2 to 8 times the budget.
  Variants kept: `r3u1/P/n50` and `r2u1/P/n50` (the rounds do not matter for
  P, whose warp is computed once).
- **Why P is cheaper than the full frame with r3:** a precomputed bilinear
  `grid_sample` downscale costs less than the full frame's antialiased
  `F.interpolate`. r3 does not antialias, so stage B reads its accuracy
  (the spec: "its accuracy decides"); r2 is the baseline's own filter and
  costs 0.06 to 0.28 ms.
- **Where the per-frame cost goes** (PIDNet-S 0.25): r3u1 with B minus with P
  is 1.78 ms. 40 density rounds cost 0.58 ms (r2u2 B at 50 against 10
  rounds), about 15 µs a round, so 50 rounds are about 0.7 ms; the other
  1.0 ms is building the maps and filters each frame (many small float64
  operations, slow on a T4) and the B or S mask on the full-size labels. U is
  far dearer on BiSeNetV2 (stride 1: the top-2 softmax over 19 channels at
  the warped image's full resolution).
- **Implementation readings:** u2 (two lerps) is about 1.7 ms slower than u1
  (`grid_sample`), against the spec's expectation, because it writes the
  full-size logits several times; r1 (tap-by-tap gather) is about 3 ms slower
  than r2. Neither is kept.
- Room under the budget is left only for a fixed warp: with r3u1 the whole
  budget, with r2u1 0.12 / 0.13 / 0.47 ms (PIDNet-S) and 0.49 / 0.51 ms
  (BiSeNetV2). This is the spec's condition for the local 2-D warp's own
  oracle bound (decision 3), and only for a fixed 2-D warp: a per-frame one
  has no room.

## 2026-11-17 — after gate A: stage B with P only; a leaner per-frame step kept for later

The user chose, on gate A's result, to go on with the static prior P only, as
the spec's rule says, and asked that the cheaper per-frame test be written
down for later rather than run now. It is not in this phase's spec or plan,
and nothing of it has been built or timed. What it would be, from where
stage A's time went (PIDNet-S 0.25, B with r3u1, 1.78 ms over P):

- **Fewer density rounds:** 50 rounds are about 0.7 ms. The prior fit
  records, per frame, how many rounds phase 1's loop needs on the E, B and S
  maps at each α and σ (`prior_<model>.json`, `max_rounds`); a step would run
  that many and no more.
- **The warp maps and filters in float32 and at the saliency grid:** about
  1.0 ms goes into building the per-pixel maps and filter weights each
  frame, in many small float64 operations. Built in float32 from the
  64 × 128 cumulative density and interpolated, it would be a few kernels.
- **B and S on the network's own output:** the masks are made on the
  full-size unwarped labels; made on the warped logits' argmax (stride 8 for
  PIDNet-S) and pooled to the cells through the inverse map, they touch
  64 times fewer pixels. U on BiSeNetV2 (stride 1) the same way, after an
  8 × 8 pooling of the logits.
- **How it would be read:** a dated amendment to the phase 2 spec, the step
  timed with `latency_floor --parts warp` in three Colab T4 sessions, and the
  same gate A. The budgets stay phase 1's (0.18 to 0.75 ms); the smallest
  per-frame overhead measured is 1.45 ms, so PIDNet-S at 0.25 and 0.375 is
  the least likely to fit and BiSeNetV2 at 0.375 the most.
- **When:** after stage B with P. If no fixed warp gains on train, a
  per-frame one has to make up the whole gap with a saliency weaker than the
  oracle's, and the test is then worth less.

## 2026-11-17 — phase 2: the static prior P fitted on the 400 train_all clips

Code: plan Task 4 (`52dd98f`). Eight cluster jobs approved by the user
("go start the 8 cluster jobs"), Tesla P100, `scripts/cluster/m5_prior.sh`:
PIDNet-S chunks 0 to 3 jobs 1108031 to 1108034, BiSeNetV2 1108035 to 1108038.
All ended with `JOB_OK` on the first try, 2:43 to 3:17 per job, 1.0 to 1.5 s
per clip. Each chunk file holds 100 clips. Merged on the laptop
(`prior-merge`), committed in `0848552`: `phase2/prior_<model>.pt` and
`.json`.

Per labelled frame, the teacher (scale 1) and the uniform pass at each stage B
scale; P is the share of pixels where they differ, per 16-px cell, over the
2400 labelled frames:

| | c 0.25 | c 0.375 | c 0.5 |
|---|---|---|---|
| PIDNet-S, mean E share | 13.3 % | 10.3 % | 8.2 % |
| BiSeNetV2, mean E share | 8.1 % | 6.1 % | not in stage B |

- **Density rounds:** phase 1's loop needs at most 11 rounds on any frame's
  E, B or S map, at any α and σ (S: 10 to 11; E: 1 to 9; B: 1 to 7). Gate A
  timed 50, so stage B keeps 50. For P it does not matter: its warp is
  computed once.
- **Where P puts the resolution:** for both models, mostly at the horizon
  band (rows of cells 26 to 33, about 420 to 540 px from the top), and very
  little sideways (column densities 0.95 to 1.08 at α 0.25, σ 4). For
  PIDNet-S, 24 % of the prior's mass is also in the bottom 128 px (12 % for
  BiSeNetV2), most likely the ego vehicle, which the labels ignore. There
  the P warp of PIDNet-S samples at a density of 1.2 to 1.4, so part of its
  resolution goes to pixels that are not scored. This is the spec's P (the
  mean of phase 1's E, which counts every pixel, as the phase 1 oracle did),
  so it is not changed; it is noted to read stage B.

## 2026-11-17 — phase 2, stage B on the 400 train_all clips: gate B says continue, but most of the gain at equal latency is not the warp's

Code: plan Tasks 5 and 6 (`ce0a680`, `e2410ca`, `70f9248`). Grid as gate A
and the user left it: P only, α 0.1 / 0.15 / 0.25, σ 1 / 4, the uniform row
at each scale, variants `r2u1` (the baseline's antialiased filter as two
matrix products, `grid_sample` unwarp) and `r3u1` (bilinear `grid_sample`,
not antialiased). PIDNet-S 18 + 3 rows per variant, BiSeNetV2 12 + 2. A fixed
warp does not depend on earlier frames, so the runner read the 6 labelled
frames of each clip, not all 30 (same confusions, tested).

Runs approved by the user ("stage B go"; "if it passes commit the scores at
the end"), Tesla P100, cluster checkout `f02647f`:

- **Pipeline check:** jobs 1108154 (PIDNet-S) and 1108155 (BiSeNetV2), the
  first 5 clips of chunk 0. It passed.
- **400 clips:** jobs 1108173 to 1108180, 30-minute limit. All ended with
  `JOB_OK` on the first try, 7:09 to 9:12 per job, 1.9 to 2.6 s per clip and
  variant.
- Merged and scored on the laptop; scores, gate B and finalists committed in
  `dcbff92`.

**Pipeline check (400 clips):** every `r2u1` uniform row equals the full frame
(|gap| ≤ 2.3e-7 mIoU). The `r3u1` uniform rows, reported only: PIDNet-S
−0.21 / −0.22 / −0.12 points at 0.25 / 0.375 / 0.5, BiSeNetV2 +0.13 / +0.27
at 0.25 / 0.375.

**Gate B, as the spec reads it: continue, for both models.** 21 of PIDNet-S's
36 warped configurations and 11 of BiSeNetV2's 24 lie above the measured
full-frame curve at equal median latency. The best of each model is
`c0.375/P/a0.15/s1/r3u1`: +1.94 points for PIDNet-S and +1.78 for
BiSeNetV2.

**What that number is made of.** Three effects, read from the committed
scores and `gate_a.json`:

1. **The warp's own gain is small.** The warp is P minus the uniform row of
   the same variant and scale, which costs exactly the same:

   | | `r2u1` best warp | `r3u1` best warp | phase 1 oracle, same α and σ |
   |---|---|---|---|
   | PIDNet-S 0.25 | +0.25 (a0.25 s4) | +0.05 | +1.18 |
   | PIDNet-S 0.375 | +0.14 (a0.15 s1) | +0.07 | +0.92 |
   | PIDNet-S 0.5 | +0.49 (a0.15 s4) | +0.46 | +1.00 |
   | BiSeNetV2 0.25 | +0.69 (a0.15 s4) | +0.80 | +0.87 |
   | BiSeNetV2 0.375 | +0.36 (a0.25 s4) | +0.30 | +0.73 |

   This is in-sample: P was fitted on these clips.

2. **Most of the rest is a cheaper resize, not the warp.** The `r3u1` uniform
   row, with no warp at all, already lies above the curve: +1.87 points for
   PIDNet-S at 0.375 and +1.49 for BiSeNetV2. That row is a full frame
   resized with a bilinear `grid_sample` instead of the antialiased
   `F.interpolate`, 0.08 to 0.20 ms cheaper in stage A. A full-frame baseline
   could make the same swap.

3. **The pricing gives every configuration a discount.** The spec prices
   each configuration at W(c) from the stage A sessions and compares it with
   the full frame of the `final` sessions. But in the stage A sessions, the
   full frame itself ran 0.03 to 0.17 ms (1 to 3 %) faster than in the
   `final` sessions, at every scale:

   | | 0.25 | 0.375 | 0.5 |
   |---|---|---|---|
   | PIDNet-S, stage A / final | 3.967 / 3.996 | 5.113 / 5.251 | 6.913 / 7.044 |
   | BiSeNetV2, stage A / final | 4.162 / 4.270 | 6.446 / 6.615 | |

   That difference comes off every configuration's price.

**Priced paired instead.** Each configuration costs the `final` full frame at
its scale plus its paired stage A overhead. On this pricing, the `r2u1`
uniform rows, which output exactly the full frame, sit 0.18 to 0.46 points
below the curve: that is the cost of the matrix resampling. ΔmIoU at equal
latency (points):

| | `r2u1` uniform | `r2u1` best warp | `r3u1` uniform | `r3u1` best warp |
|---|---|---|---|---|
| PIDNet-S 0.25 | −0.41 | −0.16 | no reading | no reading |
| PIDNet-S 0.375 | −0.46 | −0.32 | +0.92 | +0.99 |
| PIDNet-S 0.5 | −0.32 | +0.17 | +0.35 | +0.82 |
| BiSeNetV2 0.25 | −0.21 | +0.47 | no reading | no reading |
| BiSeNetV2 0.375 | −0.18 | +0.18 | +0.93 | +1.23 |

- **No reading** means the configuration costs less than the full frame at
  0.25, the cheapest point of the curve. This is the case for every `r3u1`
  configuration at 0.25. On the spec's pricing, the `r2u1` configurations of
  BiSeNetV2 at 0.25 have no reading either.
- **What survives with the exact filter and a paired price:**
  - BiSeNetV2 at 0.25, +0.47 (a0.15 s4), with 10 configurations above the
    curve;
  - PIDNet-S at 0.5, +0.17 (a0.15 s4), with 3 configurations above.

**Other readings:**
- **Share of the oracle's gain kept,** best `r2u1` warp per scale:
  - PIDNet-S: 0.21 / 0.15 / 0.49;
  - BiSeNetV2: 0.79 / 0.49.
- **mIoU by labelled position** rises along the clip for every row, the
  uniform ones included. This comes from the clips' content: a fixed warp
  cannot drift.
- **Finalists, as the spec picks them:**
  - PIDNet-S, 6: c0.25 a0.25 s4, c0.375 a0.15 s1 and c0.5 a0.15 s4, each in
    both variants;
  - BiSeNetV2, 2, both `r3u1`: c0.25 a0.15 s4 and c0.375 a0.15 s1.

**Stage C (val) is not started.** The user decides first how it reads these
three effects.

Also: `score2` wrote absolute laptop paths into its `gate_a` and `files`
fields. The code now writes repository-relative paths, and the two committed
files were changed in those fields only; no number changed.

## 2026-11-17 — amendment: P refitted on the scored pixels, stage B again with the exact filter; gate B continues, BiSeNetV2 at 0.25 +0.94

**The user's decisions** (in the thread, after seeing the inputs, P and the
warp on real frames; dated amendment in the phase 2 spec, f1e7d4a):
- keep the warp;
- only `r2u1`, the full frame's own filter, may be a finalist;
- price each configuration paired;
- refit P on the pixels the score reads.

The first P's stage B, re-read under these rules (`score2 --from-scores
--pricing paired --rule-variants r2u1 --label amended`, CPU, ce0e9dd): gate
B continues, PIDNet-S +0.17 at 0.5 and BiSeNetV2 +0.47 at 0.25.

**The prior, refitted** (`prior --scored-only`; cluster jobs 1108538-45, P100,
1.6 to 2.0 s per clip, repository at f1e7d4a; merged on the laptop, cd7a54e):
E counts only where the label is not 255.
- **Share of P's mass in the bottom 128 px** (the ego vehicle), old → new:
  - PIDNet-S: 0.24 → 0.03 at 0.25, 0.29 → 0.03 at 0.375, 0.36 → 0.04 at 0.5;
  - BiSeNetV2: 0.12 → 0.04 at 0.25, 0.14 → 0.04 at 0.375.
- **Peak row:** the row profile now peaks at 432 px, the horizon, for every
  model and scale. For PIDNet-S, and for BiSeNetV2 at 0.375, it was at the
  hood (1008 px).
- **Density rounds** needed: still at most 11.

**Stage B with it** (`stream --prior scored`, `r2u1` only; jobs 1108558-65,
P100, 2.5 to 2.7 s per clip for PIDNet-S, 2.0 to 2.1 s for BiSeNetV2,
repository at cd7a54e; scored on the laptop, 74e4346):
- **Pipeline check:** the uniform rows equal the full frame to 2.3e-7 at
  every scale. They don't depend on P, and they are the same rows as in the
  first run.
- **Gate B (paired price, `r2u1`): continue for both models.**
  - PIDNet-S: 3 configurations above the curve, best +0.09.
  - BiSeNetV2: 9 configurations above the curve, best +0.94.

Best warp per scale, against the full frame at the same scale (also the
uniform row) and at equal latency, old P → new P (points):

| | best new config | Δ same scale, old → new | Δ equal latency, old → new | share of oracle kept, new |
|---|---|---|---|---|
| PIDNet-S 0.25 | a0.25 s1 | +0.25 → +0.43 | −0.16 → +0.02 | 0.35 |
| PIDNet-S 0.375 | a0.25 s4 | +0.14 → +0.36 | −0.32 → −0.10 | 0.40 |
| PIDNet-S 0.5 | a0.1 s4 | +0.49 → +0.41 | +0.17 → +0.09 | 0.74 |
| BiSeNetV2 0.25 | a0.25 s4 | +0.69 → +1.16 | +0.47 → +0.94 | 0.64 |
| BiSeNetV2 0.375 | a0.1 s1 | +0.36 → +0.29 | +0.18 → +0.11 | 0.30 |

- **The fix helps where the hood took most:** at 0.25 and 0.375 for
  PIDNet-S, and at 0.25 for BiSeNetV2, where the warp now keeps 64 % of the
  phase 1 oracle's gain at the same α and σ. At the scale each model gained
  most from before (0.5 for PIDNet-S, 0.375 for BiSeNetV2), the new P is
  0.07 to 0.08 points lower: no better, within what one P against another
  can move.
- **The cost of the warp step** is what separates the two columns. The
  uniform `r2u1` rows, exactly the full frame, sit 0.18 to 0.46 points below
  the curve at equal latency. A warp has to gain more than that to show
  above it.
- **Still in-sample:** P is fitted on these 400 clips. Val is its
  out-of-sample reading.

**Finalists** (paired, `r2u1`; `finalists_<model>_scored.json`):
- PIDNet-S: c0.25 a0.25 s1, c0.375 a0.25 s4 and c0.5 a0.1 s4. The finalist
  rule keeps the best under each full-frame scale's latency, so c0.375 is in
  although it lies below the curve.
- BiSeNetV2: c0.25 a0.25 s4 and c0.375 a0.1 s1.

**Next: stage C (val)**, with the user's go.

## 2026-11-17 — stage C: the finalists read once on val; no finalist clears the pass rule (provisional prices)

**Run** (the user's go in the thread; laptop RTX 2060, peak 72 °C, no
pause): the 5 finalists of `finalists_<model>_scored.json` streamed once on
the 50 val clips at 21d3069 (`stream --split val --prior scored --finalists`).
The uniform rows of their scales, the pipeline check, were added afterwards
with `stream --uniform-only` at c8d0bcd, without reading the finalists again
(they are the full frame, which Method 2 already read on val). Merged
confusions in `phase2/val/`, reading in `phase2/verdict_val_scored_provisional.json`
(00c3c31).

- **Pipeline check:** every uniform row equals the full frame to 3e-7.
- **Prices:** provisional. The `m5final` T4 sessions have not been run, so
  each finalist is priced as the final sessions' full frame at its scale plus
  the stage A overhead (paired). The file is not a verdict.

| | val mIoU (train) | Δ vs full frame, same scale | Δ at equal latency [95 %] | if the warp cost nothing [95 %] |
|---|---|---|---|---|
| PIDNet-S c0.25 a0.25 s1 | 50.94 (53.28) | +0.82 | +0.43 [−0.49, +1.20] | [−0.02, +1.53] |
| PIDNet-S c0.375 a0.25 s4 | 58.80 (62.10) | +0.18 | −0.28 [−3.16, +1.79] | [−2.93, +2.44] |
| PIDNet-S c0.5 a0.1 s4 | 65.06 (68.35) | +0.24 | −0.00 [−0.44, +0.64] | [−0.19, +0.87] |
| BiSeNetV2 c0.25 a0.25 s4 | 58.45 (60.34) | −0.69 | −0.83 [−1.51, +0.08] | [−1.38, +0.23] |
| BiSeNetV2 c0.375 a0.1 s1 | 64.87 (67.14) | +0.62 | +0.45 [−0.33, +0.99] | [−0.20, +1.19] |

Intervals: clip bootstrap, 1000 resamples, seed 0. The last column is the
interval of Δ against the uniform row of the same scale, read on the same
resamples (`same_scale_interval`, computed from the committed confusions;
`report` records it as `zero_cost` from now on).

- **The pass rule reads negative:** no finalist has its interval above 0.
- **BiSeNetV2 at 0.25, the train star** (+1.16 same scale, +0.94 at equal
  latency, in-sample), is −0.69 at the same scale on val. P was fitted on the
  train clips, and at this scale its gain does not carry over.
- **PIDNet-S at 0.25 is the warp's one consistent gain:** +0.82 at the same
  scale, 39 of the 50 clips better. If the warp cost nothing, its interval
  would sit on 0: −0.02 with seed 0, and −0.05 to +0.03 over seeds 0 to 7.
- **The cost is what decides it.** The warp step cost 0.056 to 0.061 ms over
  the full frame at 0.25 in each of the 3 stage A sessions. At that price the
  lower end is −0.49. For the `m5final` sessions to turn it positive, they
  would have to find an overhead of about 0.
- **The PIDNet-S c0.375 interval is wide** because truck and bus swing
  (per-class IoU +20.85 and −12.28 points) and few clips hold them: the two
  largest hold 75 % of val's truck pixels and 84 % of its bus pixels.

**Not decided yet:** close Method 5 as negative now, or run the 3 Colab T4
sessions (`m5f-s1` to `s3`, notebook ready) for the formal verdict first.
