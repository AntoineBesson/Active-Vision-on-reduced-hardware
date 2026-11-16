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
