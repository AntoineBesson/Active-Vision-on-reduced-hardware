# Method 5, phase 1: feasibility bound of a single pass on a warped frame — pending

**Result: pending. Nothing has been run.** The code is on the `warp-bound`
branch. The grid, the latency parts and the pass rule were fixed in the
[spec](../../docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md)
(accepted 2026-11-12) before any run. Two points are open with the user before
the first run (see [Open before the first run](#open-before-the-first-run)).
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
   empty saliency gives the uniform grid.
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
- Per configuration: ΔmIoU at equal latency, latency saved at equal mIoU (the
  budget for resampling, unwarping and a real policy), ΔmIoU against the
  uniform pass at the same c (`delta_vs_uniform`) and against the full frame
  at the same c (`delta_vs_fullframe`), and per-class IoU.

**Pipeline check, before any reading:** each uniform row must be within 0.1
mIoU point of the full-frame baseline at the same scale. Otherwise `score`
reports the rule as not read.

**Pass rule:** for at least one backbone, the best configuration with timing
**now** is at least +1.0 mIoU point (0.010) above the primary full-frame curve
at equal mean latency, on all 400 clips. Otherwise the check fails and Method
5 is not built. **prev**, the secondary reference and the p95 judge are
reported only. If **now** passes and **prev** stays below the curve, `score`
says "passes, but the memory version starts under water".

## Open before the first run

Both were found while writing the code, before any run. Either answer changes
one default and adds a dated note to the spec; neither changes the grid, the
latency parts or the margin.

1. **Resampling of the warped image.** The spec's 2× supersampling and 2 × 2
   mean is a box filter; the baseline's antialiased resize is a triangle
   filter. At α = 0 they differ by about 2/255 per pixel at scales 0.25 to
   0.5 on a test photograph, and by 4 to 8 mIoU points on the synthetic
   fixture, enough to fail the 0.1-point pipeline check. `--resample
   antialias` uses the baseline's triangle filter with a width that follows
   the warp, and is exact at α = 0. The default is still `supersample` (the
   spec) until the user decides.
2. **Which reference the rule reads.** A configuration pays the full frame's
   `post` (2.1 to 4.5 ms), but the primary curve counts `area` only. The
   uniform rows, which are the full frame itself, therefore start below the
   primary curve: PIDNet-S −13.7, −7.1, −2.8, −0.4 points at scales 0.25,
   0.375, 0.5, 0.75; BiSeNetV2 −8.1, −3.2, −0.9, 0.0. `score` also reports
   the same margin read against the full frame at the same scale
   (`verdict.same_size`), which is what the spec's question states.

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

Add `--resample antialias` to every `bound` if the user picks it (a bound file
refuses rows of the other resampling, and so do `merge` and `score`).

On Colab, `notebooks/method5_bound_colab.ipynb` runs one chunk (set `CHUNK`)
from the chunk zips of the Method 3 notebook and saves the bound files to
`MyDrive/avrh/method5/train_all/`; copy them into
`experiments/method5/train_all/` before `merge`.

`bound` appends each finished clip and, started again, skips the clips already
in the file. `--models pidnet_s` runs one model.

**Run time:** not measured yet. The spec's guess is 2 to 3 h per model for the
400 clips; the pipeline check gives the seconds per clip, and this line will
be corrected from it.

## Results

Pending.

## Files

- Code: `src/warp/grid.py` (densities, forward and inverse maps, warp, unwarp),
  `src/warp/oracle.py` (E, B, S), `src/warp/bound.py` (per-clip runner),
  `scripts/method5.py` (`bound`, `merge`, `score`),
  `LatencyTables.predict_warp_frame` in `src/foveation/latency_model.py`.
- Design: [spec](../../docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md),
  [plan](../../docs/superpowers/plans/2026-11-12-method5-warp-bound.md).
