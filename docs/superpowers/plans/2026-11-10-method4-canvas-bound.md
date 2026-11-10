# Method 4, phase 1 (canvas feasibility bound) Implementation Plan

**Goal:** Replay a persistent canvas of past glimpses with a per-frame greedy
oracle selector and a free policy over the Method 3 caches of `train_all`,
predict its T4 latency from the existing tables, and read it against the
full-frame curve under a pass rule fixed in advance.

**Architecture:** `src/canvas/replay.py` holds the canvas mechanics (a
reference replay and an exact batched one). `LatencyTables.predict_canvas_frame`
prices a frame. `scripts/method4.py` has `bound`, `merge` and `score`. No
network pass, no new timing run.

**Spec:** `docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md`

## Global Constraints

- Branch `canvas-bound`, cut from `main` after PR #3.
- Grid, latency parts and pass rule are the spec's; they do not change after the first run.
- Train only (`train_all`); val is not touched in this phase.
- Method 2 and Method 3 code paths keep their behaviour; their result files are not rewritten.
- The user runs the GPU jobs (laptop RTX 2060 or cluster P100). Accuracy only: no latency comes from those GPUs.
- Every try goes in `results/method4/TRIES.md`.
- `pytest -q` green before every push; small commits.

## Review Focus

1. **The batched replay drifts from the reference** (a mask off by a cell, a warp applied to the wrong subset): pinned by an exact-equality test on the fixture, predictions and canvases.
2. **A stale or empty canvas pixel wins the fusion**: empty and over-age pixels must count as confidence 0. Pinned in Task 1.
3. **A cut-off `bound` run leaves a half-written clip** that a resume or a merge then trusts: pinned in Task 3.
4. **Variant A on a cache built with ORB**: `bound` refuses. Pinned in Task 3.
5. **An unmeasured latency point**: raises naming `latency_floor`, never extrapolates. Pinned in Task 2.
6. **The verdict is computed from the wrong set** (variant N, the p95 judge, or a partial clip set): pinned in Task 4.

## Tasks

### Task 1: canvas mechanics (`src/canvas/replay.py`, `tests/test_canvas_replay.py`)

- [ ] `CanvasConfig` (ctx_scale, glimpse_size, variant `none|iclk|off`, a_max, k or gamma, margin) with a stable `name()`.
- [ ] `replay_canvas`: one configuration, written with `reuse` helpers; returns per-labelled-frame confusions, k per frame, share of pixels taken from the canvas.
- [ ] `replay_canvas_many`: configurations sharing (glimpse size, variant, margin) in one batch per frame.
- [ ] Tests: batched equals reference; k = every cell gives the every-cell glimpse; variant N never reads a canvas; on a static clip variant B with a_max ∞ keeps a glimpsed cell and a_max 1 drops it after one frame; the oracle's teacher agreement is not below any other choice of k cells on the first frame; adaptive k is 0 where no gain clears γ.
- [ ] Commit.

### Task 2: latency (`foveation/latency_model.py`, `tests/test_latency_model.py`)

- [ ] `predict_canvas_frame(model, cfg, k, first, with_motion=False)` per the spec's table.
- [ ] Tests: the sum of the named parts for N, A, B, first and later frames, k = 0; a missing point raises.
- [ ] Commit.

### Task 3: CLI `bound` and `merge` (`scripts/method4.py`, `tests/test_method4_cli.py`)

- [ ] `bound`: grid from flags (defaults = the spec's grid), per-clip lines appended as a gzip member, resume skips complete clips and drops a cut-off tail, refuses a variant-A cache that is not IC-LK.
- [ ] `merge`: chunk files to one per model; refuses duplicates, lists missing clips.
- [ ] Tests on the fixture (dummy backbone, CPU): grid coverage, lines equal a direct replay, resume, homography check.
- [ ] Commit.

### Task 4: `score` and the verdict

- [ ] Per configuration: mIoU, per-class IoU, mIoU by position, mean and p95 latency, both judges against the primary curve and the measured end-to-end curve, the with-motion reading for A, the policy budget.
- [ ] The verdict block: the spec's rule, read only when the clip set is the dataset's full set (else "not readable: partial").
- [ ] Tests: fields, both references, verdict true and false on doctored inputs, partial set.
- [ ] Commit.

### Task 5: docs

- [ ] `results/method4/README.md` (protocol, how to run, results pending) and `TRIES.md`.
- [ ] Commit, `pytest -q`, push the branch.

### Task 6 (user): runs

- [ ] Caches present for `train_all` (rebuild if not), chunk 0, then all 400; `merge`; `score`.

### Gate

- [ ] Read the score against the rule. Fail: write up, update the top-level README, stop and ask. Pass: phase 2 spec and plan for review.
