# Method 6: feasibility bound for a single pass on a locally (2-D) warped frame

Status: **draft, written 2026-11-17, not yet accepted.** Nothing has been run.
The user asked for it on 2026-11-17 ("try something else still on the T4,
start with 2D warp"; no retraining, so the backbone stays frozen and the
method stays model-agnostic). The design is Claude's proposal; points marked
*(my call)* are open until the user accepts the spec. Once accepted, nothing
here may change after the first `bound` run, except in a dated
"Amendments" section at the end.

## Intent

Method 5 warped the frame with a **separable** map: one density for the rows,
one for the columns. To enlarge one object it enlarges the whole band of rows
and the whole band of columns the object sits in, so most of the pixels it
moves go to regions that did not ask for them. Its oracle bound passed
narrowly (+1.25 / +1.80 mIoU points at scale 0.25), and its deployable
version failed on val: a full frame given the same time does as well.

A local 2-D warp magnifies only the neighbourhood of what is salient. The
question is whether that shape, with the same frozen model and the same
oracles, has enough more headroom than the separable warp to be worth a
phase 2:

> At the same input size, does a frozen model on a locally warped frame beat
> both the uniformly downscaled frame and the best separable warp of Method 5,
> when the warp is chosen by an oracle that costs nothing?

If it does not, the shape of the warp is not what held Method 5 back, and no
memory-driven 2-D warp is built.

## The mechanism, per frame

Same as Method 5 phase 1, except step 2 and the two resampling steps.

1. **Saliency** s on the 64 × 128 grid (cells of 16 px), the oracles E, B and
   S of Method 5, unchanged (`src/warp/oracle.py`).
2. **Local warp: kernel sampling** ("Learning to Zoom", Recasens et al., ECCV
   2018; the formula only, nothing is trained). The attraction is
   a = (1 − α) + α · s / mean(s) (all ones when α = 0 or s is empty). Each
   control point u of the output (the 65 × 129 corners of the cell grid)
   samples the frame at

   x(u) = u + t · Σ_v a(v) k(u − v) (v − u) / Σ_v a(v) k(u − v),

   where v runs over the cell centres, k is a Gaussian of **σ_k cells** (the
   locality of the zoom; cells are square, 16 × 16 px, so the kernel is
   isotropic in pixels) and t ∈ [0, 1] is the backoff below. Output points
   are pulled toward salient cells, so salient regions get more output
   pixels. Between control points the map is bilinear.
   - **Borders** *(my call)*: the attraction is reflected across the frame's
     edges (edge cell included), so the sum's component normal to an edge is
     zero there; it is then set to exactly zero. The frame's border maps to
     the output's border: nothing is cropped, nothing outside is sampled.
   - **Clip, the 2-D version of Method 5's** *(my call)*: let J = ∂x/∂u in
     normalised coordinates (identity = 1). Along every direction an output
     pixel must cover at least one source pixel and at most twice what the
     uniform downscale gives it: the singular values of J stay in [c, 2]
     (Method 5: density in [d_min = 0.5, 1/c]). This also forbids folds.
     If the map breaks it anywhere, t is reduced by bisection (12 steps) to
     the largest value that keeps it, checked at the four corners of every
     control cell (a bilinear cell does not fold if its Jacobian is positive
     at its corners). t is recorded per frame.
   - α = 0 gives the identity map: the uniform downscale.
3. **Resample** the frame to H' × W' = c · (1024 × 2048) with the baseline's
   antialiased triangle filter, its half-width along each source axis being
   the extent of the output pixel's footprint on that axis (the bounding box
   of the images of its four corners), at least one source pixel. For a
   uniform map this is exactly the baseline's resize; for a separable map it
   is exactly Method 5's resampling (both are tests).
4. **One pass** of the frozen model on the warped image.
5. **Unwarp** the logits to 1024 × 2048: for every frame pixel centre x, its
   output coordinate u = x⁻¹(x) is found by Newton's method on the bilinear
   map (float64, to about 1e-6 of the frame's width), then the logits are sampled
   once at u with the model's own stride and corner convention, as in
   Method 5. For a uniform map this is exactly the baseline's upsampling.
6. Score against the labels on the labelled frames.

**Distortion is reported** per configuration (mean over frames): the mean and
95th percentile of the anisotropy σ_max/σ_min of J, the 5th and 95th
percentiles of the local linear magnification 1/√det J, and the mean
backoff t. This tests the research note's hypothesis that the frozen model
fails on distortion rather than on magnification.

Why kernel sampling and not a density-matching (optimal transport) map: it is
the published sampler for this purpose, it is local by construction with one
parameter for the locality (σ_k), it is cheap enough to run per frame later,
and the clip above gives it the same guarantees as Method 5's warp (no fold,
no crop, bounded magnification). It is a family, not every 2-D warp.

## The oracles

E, B and S as in Method 5, from the scale-1 teacher of the same model, with
timing **now** only *(my call)*. Method 5 showed that one frame of staleness
keeps most of the gain (+0.98 of +1.25, +1.59 of +1.80); if this bound passes,
**prev** is the first thing phase 2 runs. Dropping it here halves the
teacher passes.

## Grid (fixed once accepted)

Per model (PIDNet-S, BiSeNetV2):

| Dimension | Values |
| --- | --- |
| input scale c | 0.25, 0.375, 0.5 |
| saliency | E, B, S |
| strength α | 0.1, 0.25, 0.5 |
| saliency smoothing σ (cells) | 1, 4 |
| kernel σ_k (cells) | 4, 12 |

3 × 3 × 3 × 2 × 2 = 108 configurations per model, plus per scale:

- the **uniform** row (α = 0, through the 2-D code path): the pipeline check;
- one **separable** row, Method 5's best separable now configuration at
  that scale for that model (the bar of rule part (b), table below), through
  Method 5's own code: the reproduction check against Method 5's files.

(108 + 3 + 3 = 114 network passes per labelled frame.)

Scale 0.75 is left out *(my call)*: Method 5's best there was +0.15 / +0.03,
and scale 1 with magnification gave nothing worth reading. α 0.1 and 0.25
bracket Method 5's optimum (0.15 to 0.25); α 0.5 is there because a local
warp distorts less for the same strength and may tolerate more. σ_k 4 cells
(64 px) is an object-sized zoom, 12 cells (192 px) a region-sized one.

Names: `6o/c0.25/E/a0.25/s1/k4`, `6o/c0.25/uniform`,
`6o/c0.25/sep/E/a0.25/s1`.

## Latency

As Method 5 phase 1: a configuration is priced as the full frame at its scale
(T4 `area` + `reuse` `post`); building the warped image, unwarping and the
saliency cost **0**. The budget (latency saved at equal mIoU, on the full
frame priced with the same parts) is reported per configuration. It is what a
phase 2 would have to fit the 2-D resampling, the 2-D unwarp and a real
saliency into. For scale, Method 5's separable warp measured 0.056 to 0.061 ms
on a T4 (stage A); a 2-D filter will cost more.

## Reading

`scripts/method6.py score`, built on Method 5's reading (`summarise`,
`pipeline_check`, the three full-frame curves) on the same 400 clips of
`train_all`, fp32.

- Per configuration: ΔmIoU against the full frame at the same scale, against
  the uniform row, and against **the best separable now configuration of
  Method 5 at the same scale** (registered grid and the α 0.1 / 0.15
  extension, read from `results/method5/score_train_all_<model>.json.gz` and
  `results/method5/ext/`); the budget; the distortion figures; per-class IoU
  for the best ones.

Method 5's best separable now configuration per scale (the bar of part (b)):

| Scale | PIDNet-S | BiSeNetV2 |
|---|---|---|
| 0.25 | +1.25 (`E/a0.25/s1`) | +1.80 (`E/a0.25/s4`) |
| 0.375 | +0.92 (`E/a0.15/s1`) | +1.17 (`E/a0.15/s4`) |
| 0.5 | +1.00 (`E/a0.15/s4`) | +0.33 (`B/a0.15/s4`) |

(mIoU points above the full frame at that scale.)

**Checks, before any reading** (on chunk 0, then on the 400 clips):

1. **Pipeline:** each uniform row within 0.1 mIoU point of the full-frame
   baseline at its scale (as Method 5).
2. **Reproduction:** each separable row within 0.1 mIoU point of the same
   configuration in Method 5's score file. It shows that the 2-D run sees
   the same frames, teacher and oracle as the separable rows it is compared
   with.

If either fails, the run stops until it is understood.

## Pass rule (fixed once accepted, before any run)

> The check **passes** if, for at least one backbone and at least one scale,
> the best 2-D configuration at that scale is
> **(a)** at least **+1.0 mIoU point** (0.010) above the full frame at that
> scale, **and**
> **(b)** at least **+0.5 mIoU point** (0.005) above Method 5's best separable
> now configuration at that scale (table above),
> on all 400 clips of `train_all`. Otherwise it **fails**.

- (a) is Method 5 phase 1's rule, unchanged.
- (b) is new *(my call)*: the separable warp passed (a) and still failed on
  val, so a 2-D warp that only matches it has nothing to offer. +0.5 is half
  the oracle margin; both sides are the best of a grid picked on the same
  clips, so the selection bias is similar on both sides (the 2-D grid has
  108 configurations at three scales, the separable one 120 at four).
- The budget, the three curves and the distortion figures are reported and
  do not enter the rule.
- Chunk 0 (100 clips) is a smoke run and a first look; the rule is read on
  the 400 clips only.

**If it fails:** write it up next to Method 5 (`results/method6/README.md`),
update the top-level README, stop, ask the user.
**If it passes:** a phase 2 with its own spec, reviewed by the user first:
**prev** timing, a real saliency (the static prior P of Method 5 and the
previous output), the T4 cost of the 2-D resampling and unwarp against the
budget, val as in Method 5.

## Cost of the run

Per model and labelled frame: the teacher, the 3 uniform passes for E and
114 warped passes. Method 5 phase 1 ran 148 passes plus the t − 1 teacher in
18 to 21 s per clip on the cluster's P100 and 28 to 34 s on the laptop; the
2-D resampling (a gather of up to about 20 × 20 taps per output pixel) and the
Newton inverse (2 M points, about 10 iterations) add work that is not
measured yet. **Estimate:** 0.5 to 1 h per model and chunk of 100 clips on a
P100 or a T4; measured on chunk 0 first.

Where it runs (laptop, cluster or Colab T4) is the user's call; nothing runs
before the user says so in this thread. Laptop GPU runs follow the project's
thermal rule.

## Non-goals

- No training of anything; the backbones stay frozen (model-agnostic).
- No real saliency, no memory, no **prev** timing, no val, no new T4 timing.
- No change to Method 5's files or results.

## Anti-goals

- Changing the grid, the warp, the latency parts or the pass rule after
  seeing results.
- Reporting the bound as a result of a deployable method.

## How it is built

- `src/warp/grid2d.py`: the kernel-sampling map (`make_map2d`), its clip and
  backoff, dense forward evaluation, the Newton inverse, the antialiased 2-D
  resampling (`warp_frame2d`), the unwarp (`unwarp_logits2d`) and the
  distortion figures. Pure torch, tested on the CPU.
- `src/warp/bound2d.py`: one clip, every configuration (`bound_clip2d`).
- `scripts/method6.py`: `bound` (per clip, chunked, resumable), `merge`,
  `score` (both checks, the comparison with Method 5, the verdict).
- Tests: the identity map at α = 0; the warp at α = 0 is the baseline's
  resize; the unwarp at α = 0 is the baseline's upsampling; the 2-D
  resampling given a separable map's coordinates is Method 5's resampling; the
  inverse undoes the forward map; the clip holds and nothing folds on random
  saliencies; borders map to borders; a salient block is magnified in 2-D and
  its row and column far away are not.
- Ledger: `results/method6/TRIES.md`. Outcome: `results/method6/README.md`.

## Open points for the user

1. **The name.** "Method 6" keeps the numbering, since the map, the code and
   the rule differ from Method 5's.
2. **Rule part (b).** +0.5 point over the best separable warp. Lower (0) only
   asks "is it at least as good"; higher would decide the run before it ran.
3. **The grid.** Scale 0.75 out, **prev** out, two kernel sizes. Say so if
   either should come back before it is frozen.

## Amendments

(none yet)
