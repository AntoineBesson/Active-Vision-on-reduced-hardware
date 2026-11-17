# Method 5, phase 2: a warp a real policy can compute — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, stage by stage and cheapest first, whether one pass on a separably warped frame, with a saliency the method computes from its own previous output and every step timed on a T4, beats the full frame at equal measured latency on VPS val.

**Architecture:** Phase 1's `src/warp/grid.py` gains the fast variants that stage A times (resampling as two matrix products, a separable unwarp, graph-safe density and filter). A new `src/warp/policy.py` makes the four computable saliencies, and a `WarpStep` module runs one frame of the method (state in, labels and next state out) so that the same code is timed in a CUDA graph (stage A) and run for accuracy (stages B and C). Stage A is a new part of `scripts/latency_floor.py` plus a `cost` reader. Stage B is a closed-loop clip runner (`src/warp/stream.py`) behind `scripts/method5.py stream / merge / score2`, run on the school cluster. Stage C reuses the same runner on val and Method 2's bootstrap.

**Tech Stack:** Python, PyTorch, numpy, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-method5-phase2-design.md` (accepted 2026-11-17; frozen after the first stage B run). Phase 1: `docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md` and its plan.

## Global Constraints

- Frozen backbones, no training of anything. The warp is **separable** (rows and columns). The local 2-D warp is not in this plan; it is bounded with an oracle only if gate A leaves room (spec, decision 3), in a plan of its own.
- Grid, fixed by the spec: c ∈ {0.25, 0.375} for both models plus 0.5 for PIDNet-S; saliency ∈ {B, S, U, P}; α ∈ {0.1, 0.15, 0.25}; σ ∈ {1, 4} cells; resampling = the variants that pass gate A. 48 (BiSeNetV2) and 72 (PIDNet-S) configurations per variant, plus one uniform (α = 0) row per scale.
- Saliency grid 64 × 128 (cells of 16 px). Small classes (train ids) 5, 6, 7, 11, 12, 17, 18. Density clip [0.5, 1/c].
- Accuracy runs (stages B, C) in fp32. Timing runs (stage A, stage C) in FP16, batch 1, CUDA graphs, with the latency floor's protocol.
- One latency quantity throughout: the median over 3 sessions of the session median (p95 the same way). Gate A, the pricing and the pass rule all read it.
- Gate A, gate B and the pass rule are the spec's, word for word. Phase 1 results and Methods 2 to 4 are not touched.
- Every GPU run is approved by the user before it starts (Colab sessions are run by the user; cluster jobs need his words in the thread). Laptop GPU runs follow the heat rule (pause at ≥ 85 °C on two readings or any thermal flag, resume ≤ 65 °C, one GPU job at a time).
- Small commits; `pytest -q` green before every push; pull before every commit (other sessions push). Every try, failures included, goes in `results/method5/TRIES.md`.
- Branch: `claude/project-thread-nmflag` (draft PR #8, stacked on `warp-bound`).

## Review Focus

1. **The matrix resampling r2 against the gather r1:** same weights, so equal to fp32 tolerance at α 0 and at a strong warp, and at α 0 equal to the baseline's antialiased `downsample`. A wrong tap count would silently cut the filter at c 0.25 (up to 18 taps). Test in Task 1.
2. **The separable unwarp u2 against `unwarp_logits` (u1)** for both conventions: PIDNet-S (stride 8, `align_corners=True`) and BiSeNetV2 (stride 1, `False`), including logits padded to whole strides and the border rows. Test in Task 1.
3. **Nothing in a timed step may sync with the host** (`.item()`, a data-dependent loop, a shape that depends on values): the capture would fail or, worse, time something else. The density runs a fixed number of rounds; the filter has a fixed tap count per scale. Test in Task 2 (capture on CUDA, skipped without it).
4. **The closed loop's first frame and its state:** frame 0 takes P (configurations with P) or the uniform grid; each configuration carries its own state, so two configurations of one batch can never read each other's previous output. Test in Task 5.
5. **FF(c) and W(c) of one model and scale timed in the same chunk of interleaved rounds:** the gate reads paired differences, and graphs are timed in memory-bounded chunks. Test in Task 3 (the case order and chunk boundaries, without a GPU).
6. **Val is read once:** `stream --split val` refuses to run without a finalists file (or `--fixture`). Test in Task 5.

---

### Task 1: Fast resampling and unwarping (`src/warp/grid.py`)

**Files:**
- Modify: `src/warp/grid.py`
- Test: `tests/test_warp_grid.py` (append)

**Interfaces:**
- Produces:
  - `axis_density(profile, alpha, scale, d_min=0.5, d_max=None, rounds: Optional[int] = None)`: with `rounds` set, exactly that many clip-and-renormalise rounds, no early stop, no host sync. `rounds=None` keeps phase 1's loop (unchanged results).
  - `make_grid(..., rounds=None)`: passes `rounds` through.
  - `filter_taps(scale: float, d_min: float = 0.5) -> int`: `ceil(2 / (scale · d_min)) + 2` (18 at 0.25, 13 at 0.375, 10 at 0.5), the tap count that never cuts the triangle.
  - `_axis_filter(density, n_src, n_out, taps: Optional[int] = None)`: with `taps`, no `.item()`.
  - `axis_matrix(density, n_src, n_out, taps) -> Tensor[n_out, n_src]`: the same weights, scattered into a dense matrix.
  - `warp_frame(frame, grid, out_size, ..., antialias=False, method="gather", taps=None)`: `method="matmul"` computes `Wy @ frame @ Wxᵀ` per channel (r2). r1 is `antialias=True, method="gather"`; r3 is `antialias=False, supersample=1` (plain bilinear `grid_sample`).
  - `unwarp_logits_separable(logits, grid, frame_size, align_corners, output_stride=1, input_size=None)` (u2): the positions of `_logit_positions` turned into two source indices and a weight per output row and per output column, then a gather-and-lerp along rows and then along columns, with border clamping.

- [ ] **Step 1: Write the failing tests**:
  - `test_fixed_rounds_match_the_loop`: on 2,000 random profiles and the hot-cell profile of phase 1, `axis_density(..., rounds=50)` equals `axis_density(...)` to 1e-6, at every grid scale and α.
  - `test_filter_taps_never_cut_the_triangle`: at α 0.25 on a hot cell, the weights of `_axis_filter(..., taps=filter_taps(c))` equal those with the data-dependent count, at each scale.
  - `test_matmul_resampling_equals_the_gather` (α 0 and α 0.25, c 0.25 and 0.5, fp32, atol 1e-5) and `test_matmul_at_alpha_zero_is_the_baseline_downscale`.
  - `test_separable_unwarp_equals_grid_sample[stride, align]` for (8, True) and (1, False), uniform and warped grids, with `input_size` smaller than the padded logits.
- [ ] **Step 2: Run** `pytest tests/test_warp_grid.py -q`. Expected: the new tests fail on missing names.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass, phase 1 tests unchanged.
- [ ] **Step 5: Commit** `Method 5 phase 2: graph-safe density and filter, matrix resampling, separable unwarp`.

### Task 2: Computable saliencies and the per-frame step (`src/warp/policy.py`)

**Files:**
- Create: `src/warp/policy.py`
- Test: `tests/test_warp_policy.py`

**Interfaces:**
- Produces:
  - `POLICY_KINDS = ("B", "S", "U", "P")`.
  - `raw_saliency(kind, labels: Tensor[H,W] | None, logits: Tensor[1,C,h,w] | None, grid: WarpGrid | None, cell=16) -> Tensor[64,128]`, unsmoothed:
    - B, S: `oracle.pixel_mask(kind, labels)` on the method's own output labels, average-pooled to cells (phase 1's definition, with the method's output as the teacher);
    - U: `1 − (p₁ − p₂)`, the top-2 softmax margin, on the warped logits at their own resolution, then sampled at the 64 × 128 cell centres of the frame through the inverse map (`_logit_positions` with the model's convention). Computing it on the unwarped full-resolution logits would read 19 × 2 M values per frame *(my call; the spec leaves where U is computed open)*;
    - P: not computed per frame (it is the stored prior).
  - `load_prior(path, model, scale) -> Tensor[64,128]` and the prior file format: `{"model", "scale", "sum": Tensor[64,128], "frames": int, "source": "E oracle, phase 1 teacher"}`, mean = sum / frames.
  - `@dataclass(frozen=True) StepConfig(scale, saliency, alpha, sigma, resample, unwarp, rounds, taps)` with `name()` = `5p/c0.25/B/a0.15/s1/r2u2` (uniform rows: `5p/c0.25/uniform/r2u2`).
  - `class WarpStep(nn.Module)`: `forward(frame: Tensor[1,3,H,W], state: Tensor[64,128]) -> (labels: Tensor[H,W], next_state: Tensor[64,128])`. In order: smooth `state` by σ, densities (fixed rounds), maps and filter (fixed taps), resample, network, unwarp, argmax; then the raw saliency of this frame's output becomes `next_state` (for P, `next_state` is the prior itself). `initial_state(kind)`: the prior for P, zeros (the uniform grid) otherwise.
- [ ] **Step 1: Write the failing tests**:
  - `test_b_and_s_match_the_phase1_oracle_on_the_same_labels` (σ applied after, as in phase 1).
  - `test_u_is_in_unit_range_and_peaks_where_two_classes_tie` (hand-made logits).
  - `test_uniform_step_reproduces_the_full_pass`: `WarpStep` at α 0 with r1 and r2 gives `reuse.method.full_pass` labels exactly, for the PIDNet-S and BiSeNetV2 conventions (dummy backbone with each stride and convention).
  - `test_p_ignores_the_previous_output`: two different frames' outputs, same `next_state`.
  - `test_step_captures_in_a_cuda_graph` (skipped without CUDA): capture and replay twice, outputs equal to eager.
- [ ] **Step 2: Run** the tests: they fail on the missing module.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`: all pass.
- [ ] **Step 5: Commit** `Method 5 phase 2: computable saliencies B, S, U, P and the per-frame warp step`.

### Task 3: Stage A, the cost on a T4 (`scripts/latency_floor.py` part `warp`, `scripts/method5.py cost`)

**Files:**
- Modify: `scripts/latency_floor.py`, `scripts/make_latency_notebook.py` (and regenerate `notebooks/latency_floor_colab.ipynb`), `scripts/method5.py`
- Test: `tests/test_method5_cost.py`

**Interfaces:**
- Part `warp` of `latency_floor.py`, graph level only, random weights and frames as in every part:
  - per model and scale c ∈ {0.25, 0.375, 0.5}: FF(c) (the same function as the `final` part's full frame), and W(c) = `WarpStep` for every resample ∈ {r1, r2, r3} × unwarp ∈ {u1, u2} × saliency ∈ {B, S, U, P} with 50 density rounds (phase 1's worst case), plus B and U with 10 rounds on r2/u2 (what the density step costs). α 0.25 and σ 4, the larger smoothing kernel (the cost does not depend on α). 26 W cases and FF per group;
  - a **variant** is (resample, unwarp, saliency, rounds); gate A reads each one;
  - each (model, scale) group is one chunk of interleaved rounds, so FF(c) and its W(c) are always timed together; a group that does not fit in the memory fraction is split, with FF(c) repeated in each part;
  - records carry `model`, `scale`, `kind` (`fullframe` / `warp`), `resample`, `unwarp`, `saliency`, `rounds`.
- `python -m scripts.method5 cost --sessions "results/method5/phase2/latency_floor_m5w-s*.json"` writes `results/method5/phase2/gate_a.json`:
  - per (model, scale, variant): the session medians of W and FF, Δs = W − FF per session, Δ = median of the Δs, their range, and the phase 1 budget;
  - budget per (model, scale): the registered grid's best `now` configuration at that scale, read on the measured end-to-end full-frame curve (`latency_floor_final`, as in `scripts.method4.e2e_curves`): latency at which the curve reaches its mIoU, minus the full frame's at the same scale (about 0.18 / 0.26 / 0.75 ms for PIDNet-S at 0.25 / 0.375 / 0.5, 0.55 / 0.68 ms for BiSeNetV2);
  - per (model, scale, variant): `fits` (largest Δs ≤ budget), `straddles` (smallest ≤ budget < largest), `over` (smallest > budget); the variants kept for stage B are those that fit or straddle (so P alone, or r3 alone, can go on, as the spec says);
  - verdict: `stop` when every (model, scale) kept by the spec is `over`; otherwise `continue`, with the list of what goes on. A straddle is reported as "not decided by the timing".
- [ ] **Step 1: Write the failing tests** (no GPU needed):
  - `test_warp_cases_group_ff_and_w_by_model_and_scale` (case order and chunk boundaries);
  - `test_gate_a_pairs_per_session`: hand-made sessions where unpaired medians would cross the budget but every paired Δs is under it → `fits`;
  - `test_gate_a_straddle_and_over_and_stop`;
  - `test_budget_reproduces_the_spec_table` from the committed phase 1 scores and `latency_floor_final` (to 0.01 ms).
- [ ] **Step 2–4:** implement, run `pytest -q`, and a `--quick` smoke run of the part on any CUDA GPU (the laptop, under the heat rule, with the user's OK; or skipped and done in the first Colab session).
- [ ] **Step 5: Commit** `Method 5 phase 2: stage A timing part and the gate A reader`. Update the notebook's header: `PARTS = "warp"`, `SESSION = "m5w-s1"` (then s2, s3), `BRANCH = "claude/project-thread-nmflag"`.

### Run A (the user runs it) and gate A

- [ ] The user runs 3 separate Colab T4 sessions of the notebook (about 15 minutes each) and saves the three JSONs to `results/method5/phase2/`.
- [ ] `cost`; commit the JSONs and `gate_a.json`; log in TRIES (sessions, GPU clocks, Δ per variant) and in the README's phase 2 section.
- [ ] **Stop:** write it up, update the top-level README, ask the user. **Continue:** say which models, scales and variants go on, then Task 4. If gate A leaves room well under the budget, tell the user: that is the condition for the local 2-D warp's own bound.

### Task 4: The static prior P and the density rounds (`scripts/method5.py prior`)

**Files:**
- Modify: `scripts/method5.py`, `scripts/cluster/m5_bound.sh` (or create `scripts/cluster/m5_prior.sh`)
- Test: `tests/test_method5_cli.py` (append)

**Interfaces:**
- `prior --split train_all [--chunk i/4] [--models ...]`: per labelled frame, the teacher (scale 1) and the uniform pass at each stage B scale of the model (`reuse.method.full_pass`), the E mask, average-pooled to 64 × 128 (unsmoothed); summed per (model, scale) into `experiments/method5/phase2/prior_<model>_<tag>.pt` with the frame count. Resumable per clip like `bound`.
- `prior-merge`: sums the chunks into `results/method5/phase2/prior_<model>.pt`; refuses a clip counted twice.
- While it runs, `prior` also records, for each frame's B, S and E profiles at each (scale, α), the number of rounds the phase 1 loop needs; `prior-merge` reports the maximum. If it is above the `rounds` that passed gate A, the stage B runs use the larger number and the README says so.
- [ ] Tests on the fixture: one clip → a 64 × 128 sum and its count; two chunks merge to the sum of both; a clip twice is refused.
- [ ] **Run** (user's go): 4 short cluster jobs per model (about 4 passes per labelled frame), then `prior-merge`; commit the two prior files; log in TRIES.

### Task 5: Stage B, the closed loop (`src/warp/stream.py`, `scripts/method5.py stream / score2`)

**Files:**
- Create: `src/warp/stream.py`
- Modify: `scripts/method5.py`
- Test: `tests/test_warp_stream.py`, `tests/test_method5_cli.py` (append)

**Interfaces:**
- `stream_clip(dataset, index, model, cfgs: Sequence[StepConfig], priors, device, batch) -> Dict[str, np.ndarray]`: every frame of the clip in order; per scale, the configurations' warped images go through the network as one batch; each configuration keeps its own state; returns per configuration a (6, 19, 19) confusion, one per labelled position (frames 4, 9, …, 29), so the total and the drift by position both come from it. fp32.
- `stream` subcommand, flags as `bound` plus `--resample r1|r2|r3 --unwarp u1|u2` (one variant per file, from `gate_a.json`) and `--priors results/method5/phase2`. Writes `experiments/method5/phase2/<split>/stream_<split>_<model>_<variant>_<tag>.jsonl.gz`; resumable per clip; drops a cut-off tail; refuses another grid or variant; refuses `--split val` without `--finalists FILE` or `--fixture`.
- `merge` handles stream files as it handles bound files (same rules).
- `score2`:
  - pricing: each configuration's latency is W(c) of its variant from `gate_a.json` (median and p95); the full frame is `latency_floor_final` (measured end to end); nothing is counted at 0;
  - per configuration: mIoU, ΔmIoU at equal measured latency on the full-frame curve, latency saved at equal mIoU, Δ against the full frame at the same scale, share of the oracle's gain kept (against the phase 1 or extension score of `E/now` at the same scale, α and σ), per-class IoU, mIoU per labelled position;
  - pipeline check: each uniform row equals the full-frame baseline at its scale within 0.001 mIoU (r1, r2); for r3 the gap is reported, not checked (it is not the baseline's filter);
  - gate B: `continue` when, for at least one model, a configuration lies above the measured full-frame curve at equal median latency on all 400 clips; otherwise `stop`; never read on a partial set;
  - finalists, as Methods 2 and 3: the train Pareto front plus the best configuration under each full-frame scale's latency, for the median and the p95 judges, written to `results/method5/phase2/finalists_<model>.json`.
- [ ] **Tests** (fixture): a uniform configuration streamed over a clip equals the full frame's confusion; two configurations in one batch do not share state (a B and a P configuration give their own results whatever the batch order); frame 0 starts from the prior or the uniform grid; resume, cut-off tail, wrong variant; val refused without finalists; `score2` on a partial set reads no gate; hand-made scores give the expected finalists.
- [ ] **Commit** after each green step: runner, CLI, `score2`.

### Task 6: Cluster job for stage B

**Files:**
- Create: `scripts/cluster/m5_stream.sh` (CHUNK, MODEL, VARIANT; same structure as `m5_bound.sh`)
- Modify: `docs/cluster.md` (stage B commands, expected time)

- [ ] The script runs `pytest -q tests/test_warp_grid.py tests/test_warp_policy.py` before the job, then `stream` on one chunk with `--tag c<i>of4`, and ends with `JOB_OK`. Its time limit is set from the pipeline check's measured seconds per clip (Run B, step 1), with a factor of 2.
- [ ] **Commit** `Method 5 phase 2: cluster job for the closed-loop runs`.

### Run B (each step approved by the user) and gate B

- [ ] **Step 1, pipeline check:** 5 clips, both models, the uniform rows and one configuration per saliency, on the cluster (or the laptop under the heat rule). Expected: every uniform row (r1 or r2) within 0.001 mIoU of the baseline. Log seconds per clip and replace the spec's time estimate in the README.
- [ ] **Step 2, the 400 clips:** 4 chunks × 2 models × the variants kept by gate A on the cluster. Copy back, `merge`, `score2`.
- [ ] **Step 3, gate B:** commit the scores and the finalists; log in TRIES; write the phase 2 section of `results/method5/README.md`. **Stop:** write it up, update the top-level README, ask the user. **Continue:** tell the user before stage C.

### Task 7: Stage C, val (only if gate B says continue)

**Files:**
- Modify: `scripts/latency_floor.py` (part `m5final`), `scripts/method5.py` (`report`), `docs/cluster.md`
- Test: `tests/test_method5_cli.py` (append)

- [ ] **Data:** VPS val (50 clips, every frame in `val/img_all`) is not on the cluster: zip and upload it (`sftp reput`, md5 checked), as for the train chunks, with the user's go.
- [ ] **Accuracy:** `stream --split val --finalists results/method5/phase2/finalists_<model>.json` on the cluster, once.
- [ ] **Timing:** part `m5final`: each finalist's `WarpStep` and the full frame at every scale, interleaved, graph level, 3 Colab sessions run by the user (`SESSION = "m5f-s1"`…). The cost does not depend on the frame's content, so random frames are used, as in every part.
- [ ] **Report:** `report --split val` reads the val confusions and the `m5final` sessions; per finalist, ΔmIoU at equal measured median latency against the full-frame curve on val, with `eval.tradeoff.clip_bootstrap` 95% intervals; the pass rule of the spec (Δ > 0 and the interval's lower end > 0, for at least one backbone); writes `results/method5/phase2/verdict_val.json`.
- [ ] Tests: the rule on hand-made bootstrap results (Δ > 0 with the interval crossing 0 → negative); `report` refuses a finalist missing from the val files.

### Gate (end of phase 2)

- [ ] Commit the verdict, write the result in `results/method5/README.md`, TRIES and the top-level README.
- [ ] **Negative:** stop and ask the user. **Positive:** tell the user; what follows (the local warp, a learned policy, an embedded card) is the user's call, with its own spec.
