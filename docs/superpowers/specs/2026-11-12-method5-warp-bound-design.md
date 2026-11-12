# Method 5, phase 1: feasibility bound for a single pass on a warped frame

Status: **accepted by the user on 2026-11-12, before any run.** The idea is
the user's (2026-11-10: "one pass only but mixed quality based on memory",
then "our first test might be the oracle one pass warping"). The design is
Claude's proposal; points marked *(my call)* were settled by Claude and
accepted as written. Nothing in this file may change after the first `bound`
run, except in a dated "Amendments" section at the end.

Three details were made precise on 2026-11-12 while writing the plan, before
any code or run; they are marked *(plan)* below.

## Intent

Methods 2, 3 and 4 are negative on a T4. Method 4's bound says why: a frozen
model on crops is worse than the same model on the whole frame at a slightly
lower scale, and several passes cost more than their area. Method 5 removes
both causes. The frame is resampled non-uniformly, so that some regions keep
more pixels than others; the frozen model sees this one warped image, once;
its output is unwarped to the frame. No crop, no second pass, the whole scene
stays in view.

A warped image of H' × W' pixels costs exactly what the full frame at that
size costs. So the question is one of accuracy at a fixed input size:

> At the same input size, does a frozen model on a warped frame beat the same
> model on the uniformly downscaled frame, when the warp is chosen by an
> oracle that costs nothing?

If it does not, no memory-driven or learned warp will, and Method 5 is not
built. If it does, the margin is the budget of phase 2 (a warp chosen from
memory, with its real cost).

## The mechanism, per frame

1. **Saliency** s: a map on a 64 × 128 grid (1/16 of the frame) saying where
   resolution should go. Three oracle definitions, below.
2. **Separable warp** *(my call)*. The saliency is reduced to two 1-D
   profiles, its mean over rows and over columns. Each profile gives a
   sampling density along its axis:

   d = (1 − α) + α · s / mean(s), then clipped to [d_min, 1/c] and
   renormalised to mean 1 (repeat until stable).

   - α is the strength of the warp; α = 0 is the uniform downscale.
   - The upper clip 1/c means "never sample finer than the source pixels".
   - The lower clip d_min = 0.5 means "no region gets less than half the
     uniform density" *(my call)*.

   The cumulative sum of d gives, for each output column (row), the source
   column (row) it samples. Rows and columns are warped independently, so
   axis-aligned lines stay straight and the inverse map is two 1-D
   monotone lookups.
3. **Resample** the frame to H' × W' = c · (1024 × 2048) with this grid
   (bilinear, `grid_sample`). *(plan)* The image is sampled at twice the
   output size and averaged 2 × 2, because the baseline's downscale is
   antialiased and a plain `grid_sample` is not; without it the α = 0 pass
   would not reproduce the baseline.
4. **One pass** of the frozen model on the warped image: logits at H' × W'.
5. **Unwarp** the logits to 1024 × 2048 with the inverse grid (bilinear),
   then argmax. *(plan)* The logits are sampled at the model's output
   stride, in one step, with the model's own corner convention, so that at
   α = 0 this is exactly the baseline's upsampling.
6. Score against the labels on the labelled frames.

A saliency that is empty gives the uniform grid.

Why separable and not a free 2-D sampler: it is the form used in the
published work on detection (FOVEA), it cannot fold, its inverse is exact,
and it keeps straight road structure straight for a model that was never
trained on warped images. Its weakness is known: it magnifies whole rows and
columns, not regions. A free 2-D warp is left for phase 2 if this passes.

## The oracle saliencies

The teacher is the scale-1 full-frame prediction of the same model, as in
Methods 3 and 4. All three are computed at full resolution, average-pooled to
the 64 × 128 grid and smoothed by a Gaussian of σ cells.

- **E, error:** pixels where the uniform pass at the configuration's own
  scale c disagrees with the teacher. This is the direct oracle: put
  resolution where low resolution is wrong.
- **B, boundaries:** pixels of the teacher whose label differs from a
  4-neighbour.
- **S, small classes:** teacher pixels of pole, traffic light, traffic sign,
  person, rider, motorcycle, bicycle.

B and S are there because a real policy could plausibly predict them from the
previous output; E it could not *(my call to include all three)*.

**Timing of the oracle** *(my call)*:

- **now:** the saliency comes from the teacher of the current frame t. This
  is the ceiling.
- **prev:** the saliency comes from the teacher of frame t − 1, applied to
  frame t without alignment. Still an oracle (it uses a full pass), but it
  shows how much of the ceiling survives one frame of staleness, which is
  what "based on memory" will have to live with. For E, the uniform pass is
  also that of frame t − 1. *(plan)* A labelled frame at t = 0 has no
  previous frame; **prev** then uses frame 0 itself (this does not occur on
  `train_all`, whose labelled frames are 4, 9, …, 29).

**This is a heuristic ceiling, not a proof.** The three saliencies and the
separable form are a family, not every possible warp. A failed check says
"this family cannot do it with an oracle", not "no warp can".

## Grid (fixed once accepted)

Per model (PIDNet-S, BiSeNetV2):

| Dimension | Values |
| --- | --- |
| input scale c | 0.25, 0.375, 0.5, 0.75 |
| saliency | E, B, S |
| strength α | 0.25, 0.5, 0.75 |
| smoothing σ (cells of the 64 × 128 grid) | 1, 4 |
| oracle timing | now, prev |

4 × 3 × 3 × 2 × 2 = 144 configurations per model, plus the 4 uniform passes
(α = 0), which are the reference and a check of the pipeline. On the 2400
labelled frames of the 400 clips of `train_all`.

Scale 0.75 is in the grid although earlier methods stopped the context at
0.5: here it is the whole input, and 0.75 is where the full-frame curve is
strongest *(my call)*.

## Latency

Predicted from the existing T4 tables. No new timing run.

| Part | Table entry | Counted |
| --- | --- | --- |
| backbone | `area`, frame at scale c | yes |
| post-processing (upsample, argmax) | `reuse` `post` | yes |
| building the warped image | not measured | **0** |
| unwarping the logits | not measured | **0** |
| saliency and grid (the policy) | — | **0** |

The latency of a configuration is therefore the full frame's at scale c, the
same for every frame, so the mean and the p95 judge agree. The resampling
steps are set to 0 because they are not measured and because the bound must
be optimistic. The full frame pays a resize and an upsample too, so the true
extra cost is the difference between a `grid_sample` and an `interpolate`,
twice. **The latency saved at equal mIoU is the budget for all of it:**
resampling, unwarping, saliency and policy. If the check passes, phase 2
starts by timing the two resampling steps on a T4 against that budget.

## Reading

`src/eval/tradeoff.py`, unchanged, as in Method 4.

- **Primary reference:** the full-frame curve (`scripts.method3.fullframe_ref`)
  on the same clips, scales 0.25 to 1, T4 `area` median and p95.
- **Secondary reference:** the measured end-to-end full-frame latency
  (`latency_floor_final-s*.json`).
- For each configuration: ΔmIoU at equal latency, latency saved at equal
  mIoU, and the plain ΔmIoU against the uniform pass at the same c.
- Per-class IoU for the best configurations, since the gain is expected on
  small classes and the loss on large ones.

**Pipeline check, before any reading:** on chunk 0, each α = 0 pass must be
within 0.1 mIoU point of the full-frame baseline at the same scale. If it is
not, the resampling path differs from the baseline's and the run stops until
that is understood.

## Pass rule (fixed once accepted, before any run)

> The check **passes** if, for at least one backbone, the best configuration
> with oracle timing **now** is at least **+1.0 mIoU point** (0.010) above the
> primary full-frame curve at equal mean latency, on all 400 clips of
> `train_all`. Otherwise it **fails**.

- Same margin as Method 4, for the same reason: an oracle, free resampling, a
  free policy, the best of 72 configurations picked on the clips it is read
  on.
- The **prev** timing, the secondary reference and the p95 judge are reported
  but do not enter the rule. If **now** passes and **prev** does not reach the
  curve, that is reported as "passes, but the memory version starts under
  water", and the user decides.
- Chunk 0 (100 clips) is a smoke run and a first look. The rule is read on
  the 400 clips only.

**If it fails:** write it up, update the top-level README, stop, ask the user.
**If it passes:** phase 2, with its own spec and plan reviewed by the user
first (saliency from the previous output, real T4 cost of the resampling,
val protocol as Methods 2 and 3).

## Cost of the run

Unlike Method 4, nothing can be replayed from the caches: every configuration
is a network pass on a different image. Per model: 148 passes on each of 2400
frames, plus the teacher on 4800 frames (t and t − 1). Estimate, to be
replaced by a measurement on chunk 0: 2 to 3 h per model on the laptop RTX
2060 or a Colab T4, fp32.

Data: only the labelled frames and the frame before each are read (4800
frames, about 11 GB), from `train/img_all`. For Colab, one zip of those
frames replaces the four chunk zips.

## Non-goals

- No learned or memory-driven saliency, no training.
- No val, no new T4 timing.
- No fine-tuning of the backbones: the point is what a frozen model tolerates.
- No free 2-D warp, no temporal state (each labelled frame is independent,
  apart from the **prev** saliency).

## Anti-goals

- Changing the grid, the latency parts or the pass rule after seeing results.
- Reporting the bound as a result of a deployable method.
- Touching the results of Methods 2, 3 and 4.

## How it is built

- `src/warp/grid.py`: saliency to 1-D densities, clipping, forward and
  inverse grids; `warp_frame`, `unwarp_logits`. Pure torch, tested on the CPU.
- `src/warp/oracle.py`: the three saliencies from a teacher prediction.
- `scripts/method5.py`: `bound` (per clip, chunked, resumable per clip,
  `.jsonl.gz`, as `method4 bound`), `merge`, `score` (both references, the
  budget, the verdict).
- Tests on the synthetic fixture: α = 0 gives the identity grid; the inverse
  grid undoes the forward grid; the densities respect the clips and have mean
  1; a saliency concentrated in one block magnifies that block; the grids are
  monotone.
- Ledger: `results/method5/TRIES.md`. Outcome: `results/method5/README.md`.
- The same `score` can be read with the latency tables of another card (an
  embedded one): only the `area` and `post` entries are needed.

## Open points for the user

1. **The name.** "Method 5" keeps the numbering; Method 4's learned policy
   was never built.
2. **The teacher as oracle.** It could be the ground-truth labels on the
   labelled frames (a stronger oracle), but then **prev** has no source, since
   frame t − 1 is not labelled. The teacher keeps both timings comparable.
3. **The lower clip d_min = 0.5 and the family of saliencies.** These decide
   how aggressive the best warp can be. Say so if a harder setting (d_min
   0.25, α = 1) should be in the grid before it is frozen.
