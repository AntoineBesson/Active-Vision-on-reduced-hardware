# Method 5: a single pass on a warped frame — closed, negative

**Closed on 2026-11-17, negative.** Phase 1's oracle bound passed narrowly on
train. In phase 2 only a fixed warp (the static prior P) fit the T4 budget.
It passed gate B on train, but on val no finalist's 95 % interval at equal
latency is above 0. A fixed warp gains up to +0.8 points at the same input
scale, but a full frame given the same time does as well. See
[Phase 2, stage C](#phase-2-stage-c-the-finalists-once-on-val). Not tested:
a cheaper per-frame saliency, a free 2-D warp, a backbone trained on warped
frames.

## Phase 1: feasibility bound — passes, narrowly

**Result (2026-11-16, 400 clips of `train_all`): the pre-registered check
passes for both backbones, by a narrow margin.** With an oracle warp, one pass
at input scale 0.25 beats the full frame at the same scale by +1.25 mIoU
points (PIDNet-S) and +1.80 (BiSeNetV2), against a margin of +1.0. The pass
holds at scale 0.25 and the grid's weakest warp only, with the error oracle
that a real policy cannot see, and leaves 0.16 and 0.58 ms for everything the
bound counts at 0. See [Results](#results).

The grid, the latency parts and the pass rule were fixed in the
[spec](../../docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md)
(accepted 2026-11-12) before any run, and amended on 2026-11-13, still before
any run (see [Amended before the first run](#amended-before-the-first-run)).
Tries are logged in [TRIES.md](TRIES.md).

## The question

Methods 2, 3 and 4 are negative on a T4, and Method 4's bound says why: a
frozen model on crops is worse than the same model on the whole frame at a
slightly lower scale, and several passes cost more than their area. Method 5
keeps one pass on the whole scene, but resamples the frame non-uniformly so
that some regions keep more pixels than others. A warped image of H' × W'
pixels costs what the full frame at that size costs, so the question is one
of accuracy at a fixed input size:

> At the same input size, does a frozen model on a warped frame beat the same
> model on the uniformly downscaled frame, when the warp is chosen by an
> oracle that costs nothing?

If it does not, no memory-driven or learned warp will, and Method 5 is not
built.

## What is run

Per labelled frame of `train_all` (2400 frames, 400 clips), per model:

1. **Teacher:** the scale-1 full pass of the same model, on frame t and on
   frame t − 1.
2. **Saliency** on a 64 × 128 grid (cells of 16 px), average-pooled from a
   full-resolution mask and smoothed by a Gaussian of σ cells:
   - **E, error:** where the uniform pass at the configuration's scale c
     disagrees with the teacher;
   - **B, boundaries:** teacher pixels whose label differs from a 4-neighbour;
   - **S, small classes:** teacher pixels of pole, traffic light, traffic
     sign, person, rider, motorcycle, bicycle.
3. **Separable warp:** the saliency's row and column means give two 1-D
   densities d = (1 − α) + α · s / mean(s), clipped to [0.5, 1/c] and
   renormalised to mean 1. Their cumulative sums map each output row and
   column to the source; the inverse map is two 1-D monotone lookups. An
   empty saliency gives the uniform grid. The frame is resampled with the
   baseline's antialiased triangle filter, its width following the warp's
   local span, so that at α = 0 it is exactly the baseline's resize.
4. **One pass** of the frozen model on the warped image of c · (1024 × 2048).
5. **Unwarp** the logits to 1024 × 2048 with the inverse map, sampled once at
   the model's output stride with its own corner convention (at α = 0 this
   is exactly the baseline's upsampling), then argmax and score.

The **now** timing takes the saliency from frame t; **prev** takes it from
frame t − 1 and applies it to frame t without alignment (the frame itself
when t = 0, which does not occur on `train_all`).

Each clip also records the baseline's own resize at each scale
(`ref/c*/resize`), so that a pipeline-check gap can be traced to the
resampling or to the reference file.

The oracle is a heuristic ceiling, not a proof: the three saliencies and the
separable form are a family, not every possible warp.

## Grid

| Dimension | Values |
| --- | --- |
| input scale c | 0.25, 0.375, 0.5, 0.75 |
| saliency | E, B, S |
| strength α | 0.25, 0.5, 0.75 |
| smoothing σ (cells) | 1, 4 |
| oracle timing | now, prev |

144 configurations per model, plus the 4 uniform passes (α = 0), which are
the reference and the pipeline check. Names read
`5o/c0.5/E/a0.5/s4/now` and `5o/c0.5/uniform`.

## How it is read

- **Accuracy:** mIoU on the 2400 labelled frames of `train_all`, fp32. No val.
- **Latency:** predicted from the T4 tables (no new timing run): `area` of the
  full frame at scale c plus `reuse` `post`
  (`LatencyTables.predict_warp_frame`). Building the warped image, unwarping
  and the saliency cost 0. The latency is the same on every frame, so the
  mean and p95 judges agree.
- **Primary reference:** the full-frame curve (`scripts.method3.fullframe_ref`)
  on the same clips, scales 0.25 to 1. **Secondary:** the full frame's
  measured end-to-end latency (`latency_floor_final-s*.json`).
- Per configuration: ΔmIoU against the full frame at the same c
  (`delta_vs_fullframe`, the rule's reading) and against the uniform pass at
  the same c (`delta_vs_uniform`); ΔmIoU at equal latency and latency saved
  at equal mIoU against three curves (the primary one, the measured end to
  end one, and the full frame priced at `area` + `post` like a configuration);
  per-class IoU.
- **Budget:** the latency saved at equal mIoU on the full frame priced at
  `area` + `post` (`budget_ms`). It is what resampling, unwarping and a real
  policy may cost per frame, the antialiased filter included. Read on the
  frontier of that curve (the cheapest full frame that reaches the same
  mIoU), so that a flat top of the curve does not make it unreadable.

**Pipeline check, before any reading:** each uniform row must be within 0.1
mIoU point of the full-frame baseline at the same scale. Otherwise `score`
reports the rule as not read.

**Pass rule (amended 2026-11-13):** for at least one backbone, the best
configuration with timing **now** has an mIoU at least +1.0 point (0.010)
above the full frame's at the same scale, on all 400 clips. Both pay the same
backbone and post-processing there, so this is the comparison at equal
latency. Otherwise the check fails and Method 5 is not built. **prev**, the
curves and the p95 judge are reported only. If **now** passes and **prev**
stays below the full frame at its scale, `score` says "passes, but the memory
version starts under water".

## Amended before the first run

Both were found while writing the code and settled by the user on 2026-11-13,
before any run, in the spec's Amendments section. Neither changes the grid, the
latency parts or the margin.

1. **Resampling of the warped image.** The plan's 2× supersampling and 2 × 2
   mean is a box filter; the baseline's antialiased resize is a triangle
   filter. At α = 0 they differ by about 2/255 per pixel at scales 0.25 to
   0.5 on a test photograph, and by 4 to 8 mIoU points on the synthetic
   fixture, enough to fail the 0.1-point pipeline check. The bound now uses
   the baseline's filter with a local width (`--resample antialias`, the
   default), exact at α = 0. It costs more at inference than a plain
   `grid_sample`; the bound counts it at 0 ms, so it comes out of the budget.
2. **Which reference the rule reads.** A configuration pays the full frame's
   `post` (2.1 to 4.5 ms), but the primary curve counts `area` only, so the
   full frame itself would start below the primary curve: PIDNet-S −13.7,
   −7.1, −2.8, −0.4 points at scales 0.25, 0.375, 0.5, 0.75; BiSeNetV2 −8.1,
   −3.2, −0.9, 0.0. The rule now reads the mIoU above the full frame at the
   same scale; the primary curve is reported in `verdict.curves`.

## How to run

The full-frame reference of Method 2 (`experiments/method2/fullframe/`) and
the latency-floor JSONs are needed, as for Method 4. No cache: every
configuration is a network pass. PowerShell, from the repository root:

```powershell
$env:CITYSCAPES_VPS_ROOT = "C:/Users/trash/cityscapes_vps"
$env:PIDNET_S_WEIGHTS = "C:/Users/trash/weights/PIDNet_S_Cityscapes_val.pt"
$py = "C:/tmp/avrh-venv/Scripts/python.exe"

# 1. pipeline check: the first 5 clips, PIDNet-S, a small grid, in its own folder
#    (merge reads every bound file of the train_all folder)
& $py -m scripts.method5 bound --split train_all --chunk 0/80 --models pidnet_s `
    --alphas 0.5 --saliencies E --sigmas 1 --timings now --tag pipecheck --out-dir experiments/method5/pipecheck
& $py -m scripts.method5 score --split train_all --models pidnet_s --no-expect `
    --bound experiments/method5/pipecheck/bound_train_all_pidnet_s_pipecheck.jsonl.gz `
    --out-dir experiments/method5/pipecheck   # prints the check per scale; the rule is not read

# 2. chunk 0 (100 clips), both models, the full grid
& $py -m scripts.method5 bound --split train_all --chunk 0/4
& $py -m scripts.method5 merge --split train_all
& $py -m scripts.method5 score --split train_all      # partial: the rule is not read

# 3. chunks 1 to 3, then the reading on the 400 clips
foreach ($c in 1, 2, 3) { & $py -m scripts.method5 bound --split train_all --chunk "$c/4" }
& $py -m scripts.method5 merge --split train_all
& $py -m scripts.method5 score --split train_all      # writes verdict_train_all.json
```

**Extension, weaker warps (added 2026-11-16 after chunk 0, post hoc; see the
spec's amendment).** α = 0.1 and 0.15 on the same clips, in their own folder.
Its verdict file is not the pass rule, which is read on the registered grid
only.

```powershell
foreach ($c in 0, 1, 2, 3) {
    & $py -m scripts.method5 bound --split train_all --chunk "$c/4" --alphas 0.1,0.15 `
        --out-dir experiments/method5/train_all_ext
}
& $py -m scripts.method5 merge --split train_all --out-dir experiments/method5/train_all_ext
& $py -m scripts.method5 score --split train_all --out-dir experiments/method5/train_all_ext
```

**Extension, scale 1 with magnification (added 2026-11-16 after chunk 0, post
hoc).** At scale 1 the registered warp is the identity. With `--d-max 2` a
region may be sampled up to twice finer than the source pixels, at the full
frame's own cost. 24 full-resolution passes per frame, about two thirds of
the registered grid's compute.

```powershell
foreach ($c in 0, 1, 2, 3) {
    & $py -m scripts.method5 bound --split train_all --chunk "$c/4" --scales 1 --d-max 2 `
        --saliencies B,S --alphas 0.1,0.25,0.5 --out-dir experiments/method5/train_all_mag
}
& $py -m scripts.method5 merge --split train_all --out-dir experiments/method5/train_all_mag
& $py -m scripts.method5 score --split train_all --out-dir experiments/method5/train_all_mag
```

What comes after the bound (local warps, fixed priors) is listed in
[docs/research/2026-11-16-warp-next-research-points.md](../../docs/research/2026-11-16-warp-next-research-points.md).

A bound file refuses rows of another resampling, and so do `merge` and
`score`: keep the default `--resample antialias` for every run.

On Colab, `notebooks/method5_bound_colab.ipynb` runs one chunk (set `CHUNK`)
from the chunk zips of the Method 3 notebook and saves the bound files to
`MyDrive/avrh/method5/train_all/`; copy them into
`experiments/method5/train_all/` before `merge`.

`bound` appends each finished clip and, started again, skips the clips already
in the file. `--models pidnet_s` runs one model.

**Run time (measured):** on the laptop RTX 2060, 28 s per clip for PIDNet-S
and 34 s for BiSeNetV2 when not paused for heat (chunk 0 took 5.35 h, two
thirds of it paused). On the school cluster's P100 (chunks 1 to 3, one job per
chunk and model, `scripts/cluster/m5_bound.sh`, how-to in
[docs/cluster.md](../../docs/cluster.md)): 18.1 to 18.7 s per clip for
PIDNet-S and 20.8 to 21.2 s for BiSeNetV2, 31 to 36 minutes per job.

## Results

Read on 2026-11-16 on the 400 clips of `train_all` (2400 labelled frames),
fp32, registered grid only. Chunk 0 ran on the laptop, chunks 1 to 3 on the
cluster; job ids and timings are in [TRIES.md](TRIES.md). Files:
[verdict_train_all.json](verdict_train_all.json),
`score_train_all_{pidnet_s,bisenetv2}.json.gz`.

**Pipeline check: passes.** Every uniform row is within 2.2e-7 mIoU of the
full-frame baseline at its scale (tolerance 1e-3), for both models.

**Pass rule: passes for both backbones.**

| | PIDNet-S | BiSeNetV2 |
|---|---|---|
| best **now** | `5o/c0.25/E/a0.25/s1` | `5o/c0.25/E/a0.25/s4` |
| mIoU (full frame at scale 0.25) | 0.5410 (0.5285) | 0.6099 (0.5919) |
| Δ vs full frame, same scale | **+1.25** pts | **+1.80** pts |
| predicted T4 latency (area + post) | 4.86 ms | 5.10 ms |
| budget (frontier, same parts) | 0.16 ms | 0.58 ms |
| same config, **prev** | +0.98 pts, 0.12 ms | +1.59 pts, 0.51 ms |

Best **now** per scale, Δ against the full frame at that scale (budget):

| Scale | PIDNet-S | BiSeNetV2 |
|---|---|---|
| 0.25 | +1.25 (`E/a0.25/s1`), 0.16 ms | +1.80 (`E/a0.25/s4`), 0.58 ms |
| 0.375 | +0.91 (`E/a0.25/s4`), 0.22 ms | +0.73 (`E/a0.25/s4`), 0.73 ms |
| 0.5 | +0.86 (`B/a0.25/s1`), 0.76 ms | +0.08 (`B/a0.25/s4`), 0.31 ms |
| 0.75 | +0.15 (`S/a0.25/s1`), 0.91 ms | +0.03 (`B/a0.25/s1`), not readable |

Full-frame mIoU on the 400 clips, scales 1 / 0.75 / 0.5 / 0.375 / 0.25:
PIDNet-S 0.7608, 0.7480, 0.6794, 0.6174, 0.5285; BiSeNetV2 0.7335, 0.7339,
0.7055, 0.6685, 0.5919.

What else the grid shows:

- **The weakest warp wins.** Best at α 0.25: +1.25 / +1.80 (PIDNet-S /
  BiSeNetV2); at α 0.5: +0.00 / +0.04; at α 0.75: −0.09 / −0.54. The optimum
  may lie below the grid; the post hoc α 0.1 / 0.15 extension (not run) would
  tell.
- **Most of the grid loses.** Only 2 (PIDNet-S) and 4 (BiSeNetV2) of the 72
  now configurations clear +1.0. The mean now configuration is 0.4 to 4.2
  points below the full frame at its scale.
- **The saliency that wins is the one a policy cannot see.** Best per
  saliency: error E +1.25 / +1.80, boundaries B +0.86 / +1.29, small classes
  S +0.41 / +0.48. B, which a policy could approximate from the previous
  prediction, clears the margin for BiSeNetV2 only.
- **The previous frame keeps most of it:** prev +0.98 / +1.59.
- **The gain goes to large and thin objects:** for PIDNet-S bus +5.4,
  motorcycle +4.3, bicycle +2.9, pole +2.7 IoU points against the uniform
  pass (train −1.6); for BiSeNetV2 truck +6.6, train +6.3, bus +5.0, fence
  +2.9.
- **On the curves, nothing is left** (reported only): at equal mean latency,
  the best warped configuration is −0.26 (PIDNet-S) and +0.04 (BiSeNetV2) on
  the primary curve, +0.36 and +0.03 on the measured end-to-end curve. A
  slightly larger full frame buys the same accuracy.

**Reading.** The bound is met, so Method 5 is not ruled out, but only at the
bottom of the scale range and with almost no latency to spend. Phase 2 would
have to resample, unwarp and choose the warp in 0.16 ms (PIDNet-S) or 0.58 ms
(BiSeNetV2) per frame on a T4, with a saliency it can compute, at scale 0.25,
where the full frame is 23 (PIDNet-S) and 14 (BiSeNetV2) points below its own
scale 1. The best configuration is also picked in-sample among 72.

### Post hoc extension: weaker warps (α 0.1 and 0.15)

Decided after chunk 0, run on the same 400 clips with the same code, read
next to the rule and not as the rule (spec amendment of 2026-11-16). Files:
[ext/verdict_train_all.json](ext/verdict_train_all.json) and
`ext/score_train_all_*.json.gz`. Pipeline check: passes.

Best **now** against the full frame at the same scale, in mIoU points, for
α 0.1 / 0.15 / 0.25 / 0.5 (the last two from the registered grid):

| Scale | PIDNet-S | BiSeNetV2 |
|---|---|---|
| 0.25 | +0.86 / +0.96 / **+1.25** / −0.05 | +1.20 / +1.48 / **+1.80** / +0.04 |
| 0.375 | +0.69 / **+0.92** / +0.91 / −0.39 | +0.98 / **+1.17** / +0.73 / −1.28 |
| 0.5 | +0.78 / **+1.00** / +0.86 / −0.48 | +0.31 / **+0.33** / +0.08 / −1.67 |

- A weaker warp does not beat α 0.25 at scale 0.25, but at 0.375 and 0.5 the
  best strength is α 0.15: the optimum falls as the scale grows.
- There the budgets are larger: BiSeNetV2 keeps +1.17 points for 1.18 ms at
  scale 0.375, PIDNet-S +1.00 for 0.87 ms at scale 0.5 (against 0.58 and
  0.16 ms at scale 0.25).
- The best saliency is still the error oracle E; the weaker warps help the
  small-class saliency S most (BiSeNetV2 +0.96, PIDNet-S +0.73).

### Post hoc extension: scale 1 with magnification

At scale 1 the registered warp is the identity. With the density allowed up
to 2 (`--d-max 2`), a region can be sampled up to twice finer than the source
pixels, at the full frame's own cost. Saliencies B and S, α 0.1 / 0.25 / 0.5,
σ 1 and 4, both timings, on the same 400 clips; read next to the rule, not
as the rule. Files: [mag/verdict_train_all.json](mag/verdict_train_all.json)
and `mag/score_train_all_*.json.gz`. Pipeline check: passes.

- **No gain worth reading.** Best against the full frame at scale 1:
  PIDNet-S +0.36 points (`B/a0.1/s1`; +0.31 with the teacher at t − 1),
  BiSeNetV2 −0.02 (nothing above 0). Every α above 0.1 loses, up to −8 points
  at α 0.5.
- The magnified pass gains on large rigid classes (PIDNet-S: motorcycle +2.2,
  train +2.0, bus +1.1) and loses on riders and people (−1.4 and −0.5; −2.2
  and −0.7 for BiSeNetV2).
- So this warp cannot move the operating point a T4 actually uses, even with
  an oracle saliency.

## Phase 2, stage A: the cost on a T4

[Spec](../../docs/superpowers/specs/2026-11-16-method5-phase2-design.md) and
[plan](../../docs/superpowers/plans/2026-11-17-method5-phase2.md), accepted on
2026-11-17. One frame of the method (saliency from the previous output,
densities, resampling, network, unwarp, argmax: `warp.policy.WarpStep`) was
timed against the full frame on a T4, three Colab sessions, the two paired
within each session (`latency_floor --parts warp`; read by
`python -m scripts.method5 cost`, `phase2/gate_a.json`).

**Gate A: continue, with the static prior P only.** Overhead over the full
frame (median of the three paired session differences, ms), against the
phase 1 budget:

| | budget | P, bilinear (r3u1) | P, antialiased (r2u1) | best per-frame saliency |
|---|---|---|---|---|
| PIDNet-S 0.25 | 0.18 | −0.08 | +0.06 | +1.58 (U) |
| PIDNet-S 0.375 | 0.26 | −0.16 | +0.13 | +1.57 (U) |
| PIDNet-S 0.5 | 0.75 | −0.14 | +0.28 | +1.57 (B) |
| BiSeNetV2 0.25 | 0.55 | −0.12 | +0.07 | +1.62 (B) |
| BiSeNetV2 0.375 | 0.68 | −0.20 | +0.17 | +1.45 (B) |

- A warp chosen per frame from the method's own output (B, S, U) costs 1.45
  to 1.7 ms or more per frame in this implementation, 2 to 8 times the time
  the oracle's gain would pay for, at every model and scale. It does not go
  on to stage B.
- The fixed warp of the static prior costs nothing per frame (its weights
  are computed once). Stage B goes on with P alone, both resamplings: the
  bilinear one is cheaper than the full frame's antialiased resize but does
  not antialias, so its accuracy is read in stage B.
- Details, per-variant table and where the per-frame time goes:
  [TRIES](TRIES.md).
- Kept for later, at the user's request (2026-11-17): a leaner per-frame
  step (fewer density rounds, maps built in float32 at the saliency grid,
  B and S from the network's own low-resolution output), timed by the same
  gate. Not built; what it would be is in [TRIES](TRIES.md).

## Phase 2, stage B: the static prior P on train, in the closed loop

The 400 `train_all` clips, P only, both resamplings gate A kept (`r2u1`: the
baseline's antialiased filter; `r3u1`: bilinear, not antialiased), scored
against the measured end-to-end full-frame curve (`method5 score2`,
`phase2/score2_train_all_<model>.json`, `gate_b_train_all.json`,
`finalists_<model>.json`).

**Gate B, as the spec reads it: continue, for both models**, best +1.94
points at equal latency (PIDNet-S) and +1.78 (BiSeNetV2), both at 0.375 with
the bilinear resampling. But most of that is not the warp:

- **The warp's own gain** (P against the uniform row of the same variant,
  which costs the same) is +0.05 to +0.49 points for PIDNet-S and +0.30 to
  +0.80 for BiSeNetV2, in-sample.
- **The bilinear resize alone** (the `r3u1` uniform row, no warp) is already
  +1.87 / +1.49 points above the curve at 0.375: it is cheaper than the full
  frame's antialiased resize. A full frame could use it too.
- **The pricing** compares stage A's sessions with the `final` sessions,
  whose full frame ran 1 to 3 % slower: every configuration gets 0.03 to
  0.17 ms off. Priced paired, the exact filter (`r2u1`) leaves +0.47 points
  for BiSeNetV2 at 0.25 and +0.17 for PIDNet-S at 0.5.

**Amended (2026-11-17, the user's decisions, before any val read; spec
Amendments).** Only `r2u1` may be a finalist; configurations are priced
paired; and P is refitted counting E only on scored pixels. The first P put
24 to 36 % of its mass on the never-scored ego vehicle for PIDNet-S. Stage B
was run again with the new P (`prior_scored_<model>.pt`;
`score2_train_all_<model>_scored.json`, `gate_b_train_all_scored.json`,
`finalists_<model>_scored.json`).

**Gate B, amended: continue, for both models.**

| | best warp | Δ vs full frame, same scale | Δ at equal latency (paired) |
|---|---|---|---|
| PIDNet-S | c0.5 a0.1 s4 | +0.41 | +0.09 |
| BiSeNetV2 | c0.25 a0.25 s4 | +1.16 | +0.94 |

- **BiSeNetV2 at 0.25** keeps 64 % of the phase 1 oracle's gain, with a
  P a policy can use.
- **PIDNet-S** gains about 0.4 points at every scale, but the warp step's
  cost leaves it only just above the curve.
- **All in-sample:** P is fitted on these clips.

## Phase 2, stage C: the finalists once on val

The 5 finalists, streamed once on the 50 val clips, with the uniform rows of
their scales as the pipeline check (equal to the full frame to 3e-7);
`phase2/val/` and `phase2/verdict_val_scored_provisional.json`. Priced by
stage A, paired. The `m5final` T4 sessions were not run: the user closed the
method on this reading (spec amendment of 2026-11-17).

**The pass rule reads negative: no finalist's 95 % interval at equal latency
is above 0.**

| | Δ same scale | Δ at equal latency [95 %] |
|---|---|---|
| PIDNet-S c0.25 a0.25 s1 | +0.82 | +0.43 [−0.49, +1.20] |
| PIDNet-S c0.375 a0.25 s4 | +0.18 | −0.28 [−3.16, +1.79] |
| PIDNet-S c0.5 a0.1 s4 | +0.24 | −0.00 [−0.44, +0.64] |
| BiSeNetV2 c0.25 a0.25 s4 | −0.69 | −0.83 [−1.51, +0.08] |
| BiSeNetV2 c0.375 a0.1 s1 | +0.62 | +0.45 [−0.33, +0.99] |

- **BiSeNetV2 at 0.25**, +0.94 at equal latency on train, does not carry
  over to val.
- **PIDNet-S at 0.25** gains +0.82 at the same scale (39 of 50 clips
  better). Even at zero warp cost its interval would sit on 0 (−0.02 with
  the seed of the rule). The 0.06 ms the warp step costs at that scale is
  enough to put the lower end at −0.49.

Details: [TRIES](TRIES.md).

## Files

- Code: `src/warp/grid.py` (densities, forward and inverse maps, warp, unwarp),
  `src/warp/oracle.py` (E, B, S), `src/warp/bound.py` (per-clip runner),
  `scripts/method5.py` (`bound`, `merge`, `score`),
  `LatencyTables.predict_warp_frame` in `src/foveation/latency_model.py`.
- Phase 2: `src/warp/policy.py` (saliencies B, S, U, P and `WarpStep`), the
  `warp` part of `scripts/latency_floor.py`, `scripts/method5.py cost`;
  `src/warp/prior.py` and `method5 prior` / `prior-merge` (the static prior
  P); `src/warp/stream.py` and `method5 stream` / `merge --kind stream` /
  `score2` (stage B), `method5 report` and the `m5final` part of
  `scripts/latency_floor.py` (stage C); results in `phase2/` (the three T4 sessions,
  `gate_a.json`, `prior_<model>.pt` / `.json`, the prior fitted on the
  400 `train_all` clips, and `prior_scored_<model>.*`, fitted on their scored
  pixels; stage B's scores, gate B and finalists, plain, `amended` and
  `scored`; the val confusions in `phase2/val/` and the provisional val
  reading).
- Design: [spec](../../docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md),
  [plan](../../docs/superpowers/plans/2026-11-12-method5-warp-bound.md).
