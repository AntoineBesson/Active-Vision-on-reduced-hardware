# Method 5, phase 2: a warp a real policy can compute, at its real cost

Status: **draft, 2026-11-16. Nothing is built or run.** The user asked for
this spec on 2026-11-16, after phase 1 passed, and settled its open points
the same evening ([decisions](#the-users-decisions-2026-11-16)). Acceptance
waits for the scale 1 magnification extension, which is running: if it
gains, the scales of stage B change. The design is Claude's proposal; points
marked *(my call)* are Claude's choices. Once the user accepts it, nothing
here changes after the first stage B run, except in a dated "Amendments"
section at the end, as for phase 1.

## What phase 1 leaves

The [phase 1 bound](2026-11-12-method5-warp-bound-design.md) passed on the 400
clips of `train_all`, narrowly
([results](../../../results/method5/README.md)). An oracle warp at input scale
0.25 beats the full frame at the same scale:

| | PIDNet-S | BiSeNetV2 |
|---|---|---|
| best oracle, timing now | +1.25 pts (`c0.25/E/a0.25/s1`) | +1.80 pts (`c0.25/E/a0.25/s4`) |
| same, saliency of the teacher at t − 1 | +0.98 | +1.59 |
| best at scale 0.375 (now) | +0.91 | +0.73 |
| best boundary saliency B (now) | +0.86 (c0.5) | +1.29 (c0.25) |
| budget at c0.25, on the measured end-to-end curve | about 0.18 ms | about 0.55 ms |
| budget at c0.375, same | about 0.26 ms | about 0.68 ms |
| best at scale 0.5 (now) | +0.86 (`B/a0.25/s1`) | +0.08 |
| same saliency B at 0.5, teacher at t − 1 | +0.80 (`B/a0.25/s4`) | — |
| budget at c0.5, same | about 0.75 ms (0.70 ms at t − 1) | not kept |

The budget is the latency at which the full frame, measured end to end on the
T4 and interpolated between scales, reaches the oracle's mIoU, minus the full
frame's own latency at the same scale. Phase 1 counted three steps at 0 ms, so
this budget has to cover all of them:

- resampling the frame with a filter whose width follows the warp;
- unwarping the logits to the full frame;
- computing the saliency and the warp maps (the policy).

Two things eat into the margin before any cost is paid:

- A real policy cannot see the error oracle E, which won at both scales.
- The method's own previous output at scale 0.25 is a much worse guide than
  the scale-1 teacher that the **prev** timing used.

**Expected outcome:** negative is the most likely result. So the stages are
ordered cheapest first, each with a gate that stops the method as soon as one
fails *(my call)*.

## The question

> When the saliency is computed from what the method itself has already seen,
> and every step is timed on a T4, does one pass on a warped frame beat the
> full frame at equal measured latency, on val?

## Stage A: the cost of the warp on a T4 (first, with no accuracy run)

This is what the phase 1 spec asked for first: "phase 2 starts by timing the
two resampling steps on a T4 against that budget".

**Protocol:** that of the [latency floor](../../../results/latency_floor/README.md).
- Colab T4, batch 1, FP16, CUDA graphs, streaming.
- 3 sessions, interleaved with the full frame at the same scale.

**Pipelines timed**, at c ∈ {0.25, 0.375, 0.5}, for both models:

- **FF(c), the reference:** antialiased resize, backbone, upsample, argmax.
  This is the full frame as measured in `latency_floor_final`.
- **W(c), the method:** saliency from the previous prediction, densities and
  maps, warped resample, backbone, unwarp, argmax.

**Variants** *(my call)*:

| Step | Variant | Note |
|---|---|---|
| resample | r1: the current K-tap gather (`_filter_axis`) | what phase 1 ran; up to 18 taps per axis at c 0.25 |
| resample | r2: the same filter as two matrix products, Wy · frame · Wxᵀ | same weights, so same values up to fp16; tensor cores |
| resample | r3: plain bilinear `grid_sample`, no antialiasing | cheapest; not the baseline's filter, so its accuracy is read in stage B |
| unwarp | u1: `grid_sample` on the logits (phase 1) | reads a 2-channel full-resolution grid |
| unwarp | u2: separable, a 1-D row and column lookup per output pixel | same values as u1; memory traffic of a plain upsample |
| policy | B, S, U from the previous logits; P precomputed | defined in stage B |

- The densities are clipped and renormalised in a loop whose length depends on
  the data. Under a CUDA graph this becomes a fixed number of iterations,
  checked against the loop on the train saliencies *(my call)*.
- The static prior P is a fixed warp, so its resampling weights are
  precomputed and it has no policy step.

**Gate A, fixed before stage A runs:**
- Overhead, per session: Δs(c) is the median of W(c) minus the median of
  FF(c) in that session, for the cheapest variant. Both are timed interleaved
  in the session, so the difference is paired and the session's speed cancels.
- Δ(c) is the median of the three Δs(c). Its range over the sessions is
  reported next to it.
- Why this matters: between sessions a median moves by 3 to 5%
  ([latency floor](../../../results/latency_floor/README.md), Repeatability).
  At 4 ms that is 0.12 to 0.2 ms, the size of PIDNet-S's budget at scale 0.25
  (0.18 ms). Unpaired medians could not decide the gate.
- If the **smallest** of the three Δs(c) exceeds the phase 1 budget above for
  every model and scale kept in stage B, Method 5 stops: even a free oracle
  saliency cannot pay for its own resampling.
- If the sessions straddle the budget somewhere (smallest under, largest
  over), stage B goes on for that model and scale, and the README says the
  gate was not decided by the timing *(reviewer's call, to confirm)*.
- If only r3 fits, stage B goes on with r3 alone, and its accuracy decides.
- If only P fits, stage B goes on with P alone.

## Stage B: a saliency the method can compute, in a closed loop, on train

### Mechanism

**Closed loop:**
- Every frame of each 30-frame clip is processed in order. The warp of frame
  t comes from the method's own unwarped output at t − 1.
- Frame 0 of a clip takes the static prior P, or the uniform grid when the
  configuration has no P.
- Scoring is against the labels of the 6 labelled frames, as in phase 1.
- The chunk zips hold every frame of the train clips (`img_all`), as Method 3
  used them.

**Saliencies a policy can compute.** Each is pooled to the 64 × 128 grid and
smoothed by σ, as in phase 1.

- **B, boundaries** of the method's own previous prediction.
- **S, small classes** of the method's own previous prediction (the same
  seven classes).
- **U, uncertainty** of the method's own previous prediction: 1 minus the
  margin between the top two softmax scores, at the logit resolution. It
  stands in for the error oracle E, which a policy cannot compute
  *(my call)*.
- **P, static prior:** the mean of the phase 1 oracle saliency E at the
  configuration's scale, over the `train_all` frames, one map per model and
  scale. It costs nothing per frame. It is fitted on the same clips that
  stage B reads, so on train it is in-sample. Val is its out-of-sample
  reading.

### Grid

| Dimension | Values |
|---|---|
| input scale c | 0.25, 0.375, and 0.5 for PIDNet-S |
| saliency | B, S, U, P |
| strength α | 0.1, 0.15, 0.25 (the user's decision, from the α extension) |
| smoothing σ (cells) | 1, 4 |
| resampling | the variants that passed gate A |

**Size**, per resampling variant, plus the uniform pass at each scale:
- BiSeNetV2: 2 × 4 × 3 × 2 = 48 configurations;
- PIDNet-S: 3 × 4 × 3 × 2 = 72 configurations.

**Scale 0.5 is kept for PIDNet-S only** *(changed in review, 2026-11-16)*.
The draft left it out because +0.86 is under phase 1's +1.0 margin. But phase
2 is judged at equal latency, where the budget counts, not that margin: at
0.5 PIDNet-S has about 0.75 ms, four times its budget at 0.25, with the
boundary saliency B, which a policy can compute, and it keeps +0.80 from the
teacher at t − 1.

**Why the rest is left out:**
- **BiSeNetV2 at 0.5** has the largest budget of all, 0.7 to 1.3 ms on the
  measured curve, but its gain with a saliency a policy can compute
  (boundaries or small classes from the teacher at t − 1) is only +0.20 to
  +0.36 points. Fifty val clips are unlikely to resolve that *(the user's
  reason, 2026-11-16)*.
- **Both models at 0.75** gain +0.21 points or less, oracle included.

**References, not rerun:** the phase 1 oracle (now) and teacher (prev) on the
same clips.

**Cost:**
- Each configuration runs its own closed loop: 48 × 30 = 1440 passes per clip
  at scale 0.375 or below, batched across configurations frame by frame.
- From phase 1's speed on the P100 (about 900 passes per clip in 18 to 21 s),
  about 20 to 40 s per clip. That is at most about 1 h per chunk and model,
  as 8 cluster jobs.
- PIDNet-S's 24 configurations at scale 0.5 add 720 passes per clip at four
  times the pixels of scale 0.25, which is more than its two other scales
  together: count 2 to 2.5 h per chunk for PIDNet-S (an estimate, not
  measured).
- Fitting the static prior P is a run of its own, before stage B: phase 1
  kept confusion matrices, not saliency maps, so the teacher and the uniform
  pass at each scale are run again on the 2400 labelled train frames (about 4
  passes per frame and model, minutes per chunk).

### Pricing and reading

**Pricing:**
- **One latency quantity throughout, as in Methods 2 to 4:** a pipeline's
  latency is the median over the three sessions of its session median; its
  p95 is read the same way. The "mean" judge of the earlier methods is the
  mean over a clip's frames of that per-frame latency. Every frame of Method
  5 costs the same, so here it is W(c)'s median itself. Gate A, the pricing
  and the pass rule all read this number.
- Each configuration takes the measured W(c) of its variant from stage A,
  median and p95.
- It is compared with the full frame measured end to end
  (`latency_floor_final`). Both sides are measured, and no step is counted
  at 0.

**Reading, per configuration:**
- ΔmIoU at equal measured latency on the full-frame curve, and latency saved
  at equal mIoU.
- ΔmIoU against the full frame at the same scale.
- Share of the oracle's gain kept: Δ of the configuration over Δ of the
  phase 1 oracle (now) with the same scale, α and σ.
- Per-class IoU, and mIoU by position in the clip (frames 4, 9, …, 29), so
  that drift in the closed loop is visible.

**Gate B (train):** go on to val only if, for at least one model, a
configuration lies above the measured full-frame curve at equal mean latency
on train. Otherwise stop, write it up, update the top-level README, and ask
the user. Method 3 stopped this way.

## Stage C: val

- **Finalists, as in Methods 2 and 3:** the train Pareto front plus the best
  configuration under each full-frame scale's latency, for the mean and p95
  judges.
- **Read once** on VPS val (50 clips, 300 labelled frames), in the closed loop,
  with clip-bootstrap 95% intervals. P stays fitted on train.
- **Timed end to end** on the T4, streaming whole val clips, in 3 sessions
  interleaved with the full frame. These timings replace stage A's in the
  final table.

## Pass rule (fixed when this spec is accepted, before any stage B run)

> Method 5 is **positive** if, for at least one backbone, a finalist chosen on
> train has a ΔmIoU above 0 against the measured full-frame curve at equal
> mean latency on VPS val, and the lower end of its clip-bootstrap 95%
> interval is also above 0. Otherwise it is **negative**.

The p95 judge, the same-scale delta and the share of the oracle's gain kept
are reported but do not enter the rule. There is no margin (the user's
decision 2).

## Non-goals

- No training, neither of the backbone nor of a policy. A learned or RL
  policy comes only if this passes.
- No free 2-D warp (research point 1), unless gate A leaves room (decision 3).
- No scale 1 unless the magnification extension, now running, gains
  (decision 4).
- No embedded card. Only the latency tables would change.

## Anti-goals

- Tuning anything on val, or choosing finalists by their val numbers.
- Counting any step of the pipeline at 0 ms.
- Reporting a phase 1 oracle number as a result of the method.
- Changing the results of phase 1 or of Methods 2 to 4.

## How it would be built (after acceptance, with its own plan)

- **`src/warp/grid.py`:**
  - r2, the matrix form of the antialiased resampling, tested equal to r1 up
    to float precision.
  - u2, the separable unwarp, tested equal to `unwarp_logits`.
  - A fixed-iteration density for CUDA graphs.
- **`src/warp/policy.py`:**
  - B, S and U from logits at the model's output stride.
  - Fitting and loading the static prior P.
- **`src/warp/stream.py`:** the closed-loop clip runner, which batches the
  configurations frame by frame. At α = 0 it must reproduce the full frame:
  this is the pipeline check, as in phase 1.
- **`scripts/method5.py`:** `prior`, `stream` and `score2`, with the same
  rules on resuming, merging and partial reads as `bound`, `merge` and
  `score`.
- **Notebook and job script:** `notebooks/method5_cost_t4.ipynb` for stage A,
  and `scripts/cluster/m5_stream.sh` for stage B.
- **Results:** a phase 2 section in `results/method5/README.md`, and the
  ledger in `results/method5/TRIES.md`.

## The user's decisions (2026-11-16)

The open points of the draft, settled by the user after the α extension was
scored.

1. **α ∈ {0.1, 0.15, 0.25}.** The draft proposed {0.15, 0.25, 0.35}, on the
   error oracle E at scale 0.25, which a real policy cannot compute. For the
   two saliencies phase 2 can use, the gain peaks at α 0.25 or below and is
   negative at α 0.5 in every cell. With the teacher at t − 1 (timing
   **prev**), best Δ against the full frame at the same scale, in points:

   | Saliency | Model, scale | α 0.1 | α 0.15 | α 0.25 | α 0.5 |
   |---|---|---|---|---|---|
   | small classes S | PIDNet-S, 0.25 | +0.76 | +0.60 | −0.19 | −4.16 |
   | small classes S | BiSeNetV2, 0.375 | +0.70 | +0.77 | +0.01 | −6.73 |
   | boundaries B | PIDNet-S, 0.5 | +0.47 | +0.68 | +0.80 | −0.71 |
   | boundaries B | BiSeNetV2, 0.25 | +0.60 | +0.72 | +0.92 | −0.20 |

   S already loses at α 0.25, so dropping α 0.1 would drop its best setting.
   (The draft's "α 0.1 is below α 0.15 at every scale" held for E only.)
2. **Margin: none.** The rule stays "above 0, with the interval above 0".
   The oracle gains are about one point, so a +0.5 margin would decide the
   result before the run.
3. **Free 2-D warp: as drafted.** Bound it with an oracle only if gate A
   leaves room under the budget.
4. **Scale 1 with magnification: run now, before acceptance.** It is the only
   version that can move the operating point a T4 actually uses. If it gains,
   the scales of stage B change. Submitted on 2026-11-16 (8 cluster jobs).
5. **Both models.** With saliencies from the previous frame, the budgets on
   the measured curve are about 0.4 to 0.7 ms for PIDNet-S at scale 0.5 and
   about 0.65 to 0.75 ms for BiSeNetV2 at 0.375: each model has one scale
   with real room.

Still to confirm at acceptance: the rule when the gate A sessions straddle
the budget (stage B goes on, marked *to confirm* above).

## Changes after review (2026-11-16, before acceptance)

A second session checked the draft against the phase 1 result files (the
figures of the first table are reproduced) and made five changes, at the
user's request:

1. Scale 0.5 is kept for PIDNet-S, with its budget in the first table.
2. Gate A reads paired differences per session and says what happens when
   the sessions straddle the budget.
3. The budget table has the scale 0.5 row that gate A needs.
4. One latency quantity is named and used by the gate, the pricing and the
   rule.
5. The cost of fitting the static prior P is counted.

The reviewer's advice on the open points, since settled by the user (above):
run the α extension and the scale 1 extension before accepting this spec
(points 1 and 4); stage A can start meanwhile, since its timings are needed
either way; keep the draft margin (point 2); bound the 2-D warp only if gate A
leaves room (point 3); both models (point 5).
