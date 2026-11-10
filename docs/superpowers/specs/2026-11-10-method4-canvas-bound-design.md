# Method 4, phase 1: feasibility bound for a persistent canvas with a glimpse policy

Status: pre-registered 2026-11-10, before any run. The design comes from the
user's brief of 2026-11-10. Points marked *(my call)* were settled by Claude,
not the user. Nothing in this file may change after the first `bound` run,
except in a dated "Amendments" section at the end.

## Intent

Method 2 (foveation without memory) and Method 3 (motion-gated reuse of the
previous frame) are both negative on a T4. Method 4 would be a learned glimpse
policy over a persistent canvas of past glimpses. Before building it, this
phase asks one question:

> With a near-perfect cell selector that costs nothing and a persistent
> canvas, can foveation beat the full-frame curve on VPS train?

If it can't, no learned policy will, and Method 4 is not built. If it can, the
margin and the latency budget left for a real policy are the inputs of phase 2.

## The mechanism, per clip

State: a **canvas** of past glimpse labels. Three maps:

- labels, (H, W) at full resolution;
- confidence, uint8 at 1/4 resolution (Method 3's motion resolution); 0 means
  "never written";
- age in frames since the pixel was written, uint8 at 1/4 resolution,
  saturating at 255.

The canvas holds glimpse predictions only. The context is never written into
it *(my call: this is the brief's "canvas of past glimpse labels"; Method 3's
memory, which kept the fused result, lost about 5 pts per frame of age)*.

Every frame t, frame 0 included (no full pass anywhere, so latency is even):

1. **Context pass** at scale c: labels and confidence.
2. **Canvas to the current frame** (t > 0): age + 1, then
   - variant **A**: labels, confidence and age are warped by the IC-LK
     homography of the Method 3 cache (nearest); pixels the warp leaves
     without a source become empty (confidence 0);
   - variant **B**: nothing (identity).
3. **Fusion**: Method 3's 3b rule, `reuse.method.fuse`. Per pixel, the more
   confident of the context and the canvas wins, the context on ties. Canvas
   pixels that are empty or older than a_max count as confidence 0. The result
   is the base prediction. The canvas itself is not changed by the fusion.
4. **Selection** (the oracle, below) picks k cells of Method 2's grid.
5. **Glimpses**: the selected cells' inner squares, from the cached every-cell
   glimpse, are pasted into the base prediction (Method 2's replace rule) and
   written into the canvas with their confidence and age 0.
6. The result is the frame's prediction, scored against the labels on the
   labelled frames (4, 9, …, 29).

A third variant, **N** (no canvas), skips steps 2, 3 and the canvas write. It
is Method 2 with an oracle selector, on the same frames, and shows what the
canvas adds *(my call)*.

## The selector: a per-frame greedy oracle

As Method 3's zone oracle. The teacher is the scale-1 full-frame prediction of
the same model at frame t. A cell's gain is the number of its pixels where the
glimpse agrees with the teacher, minus the number where the base prediction
(context fused with the canvas) does.

- **Fixed k**: the k cells of highest gain (stable order on ties), whatever
  the sign of the gain. k ∈ {1, 2, 4, 8}.
- **Adaptive k** *(my call for the rule and its values)*: every cell whose
  gain is above γ × (pixels of a cell's inner square). γ ∈ {0.005, 0.02,
  0.05}. k varies per frame, from 0 to every cell.

**This is a heuristic ceiling, not a proof.** It is myopic: it maximises the
agreement of the current frame only. A planning policy could glimpse a cell
now because the canvas will pay it back over the next frames, and could in
principle do better. It is also a ceiling relative to the teacher, not to the
labels. A failed check therefore says "a per-frame selector cannot do it,
even a perfect one", which is the family phase 2 would learn by behaviour
cloning; the RL part could only add the planning effect on top.

## Grid (fixed)

Per model (PIDNet-S, BiSeNetV2; the glimpse model is the context model):

| Dimension | Values |
| --- | --- |
| context scale c | 0.25, 0.375, 0.5 |
| glimpse size (margin 32) | 256, 384, 512 |
| variant and a_max | N; A and B, each with a_max ∈ {1, 2, 4, 8, ∞} |
| selector | k ∈ {1, 2, 4, 8}; adaptive γ ∈ {0.005, 0.02, 0.05} |

3 × 3 × 11 × 7 = 693 configurations per model, on all 400 clips of
`train_all` (30 frames each, 2400 labelled frames).

## Latency

Predicted per frame from the existing T4 tables (t4-s1..3 and m3-s1..3), as
Method 3's `score` does, with the frame's own k. No new timing run.

| Part | Table entry | Counted on |
| --- | --- | --- |
| context backbone | `area`, frame at scale c | every frame |
| context post-processing (upsample, argmax, confidence) | `reuse` `post` | every frame |
| k glimpses | `batch` at k | frames with k > 0 |
| glimpse post-processing (crop, upsample, argmax, confidence, paste) | `reuse` `glimpse_post` at k | frames with k > 0 |
| fusion | `reuse` `fusion` | A and B, t > 0 |
| gray downsample | `reuse` `gray` | A, every frame |
| IC-LK homography | `reuse` `homography` | A, t > 0 |
| warps | `reuse` `warp` | A, t > 0 |
| policy | 0 | — |

Calls of mine on this table:

- **Method 2's `overhead` table is not used.** It is measured only for
  k ∈ {1, 2, 4, 8}, so adaptive k has no entry, and it includes the scoring
  and selection, which are the policy and cost 0 here. Its other steps are the
  `post` and `glimpse_post` entries above, measured for every k. The brief
  asked for Method 2's overhead; this is the same work without the policy.
- **The motion map is not in the bound.** The mechanism does not use it. It is
  a policy input for phase 2 (variant A), so it belongs to the policy's cost,
  which the bound sets to 0. Counting it would make the bound pessimistic for
  a policy that does not use motion, and the bound must be optimistic to be a
  valid stop rule. The reading with the motion map added (1.3–1.6 ms) is
  reported next to it for variant A. The brief listed motion under variant A's
  costs; this is a deviation, flagged for the user.
- **The `warp` entry times two maps** (labels and confidence); the canvas
  warps three (age too). The prediction is slightly low for variant A.
- Backbone passes are graph-level timings, post-processing is eager, as in
  Method 3's `score`. Method 3 found the graph-level sum unconfirmed end to
  end; that also leans in the bound's favour.

Everything above errs in favour of the canvas. That is intended.

**Budget for a real policy.** For each configuration, the latency saved at
equal mIoU (mean judge) is the time a real policy may cost before the
configuration falls back onto the curve. It is reported for every
configuration and, in the README, for the best ones.

## Reading

`src/eval/tradeoff.py`, unchanged.

- **Primary reference:** the full-frame curve of Method 3's `score`
  (`scripts.method3.fullframe_ref`): mIoU of the same model on the same clips
  at scales 0.25, 0.375, 0.5, 0.75, 1, with the `area` table's median and p95.
- **Two judges:** mean per-frame latency against the full frame's median; p95
  per-frame latency over all frames of all clips against the full frame's p95.
- **Both readings** for each judge: ΔmIoU at equal latency, latency saved at
  equal mIoU. A configuration outside the curve's latency range has no
  reading.
- **Secondary reference:** the same mIoU values against the full frame's
  measured end-to-end latency in `latency_floor_final-s{1,2,3}.json` (graph
  level; median over sessions of the median and of the p95). It includes the
  full frame's own resize and post-processing, so it is the fairer comparison
  for a prediction that counts post-processing.

## Pass rule (fixed before any run)

> The check **passes** if, for at least one backbone, the best configuration
> of variant A or B is at least **+1.0 mIoU point** (0.010) above the primary
> full-frame curve at equal **mean** latency, on all 400 clips of `train_all`.
> Otherwise it **fails**.

- The 1.0 pt margin is there because the bound is optimistic: an oracle
  selector, a free policy, the best of 630 canvas configurations picked on the
  clips it is read on, and the latency choices above.
- The p95 judge, the secondary reference and the with-motion reading are
  reported but do not enter the rule.
- Variant N does not enter the rule either: it has no canvas *(my call)*. If N
  alone clears the margin, that is reported and the user decides what it
  means.
- Chunk 0 (100 clips) is a smoke run and a first look. The rule is read on the
  400 clips only.

**If it fails:** write it up, update the top-level README, stop, ask the user.
**If it passes:** phase 2 (its own spec and plan, reviewed by the user first).

## Non-goals

- No learned policy, no training, no new network pass: everything is replayed
  from the Method 3 per-clip caches.
- No val. The bound is a train-only go/no-go check.
- No new T4 timing.
- No frame skipping; no retraining of the backbones; no context above 0.5.

## Anti-goals

- Changing the grid, the latency parts or the pass rule after seeing results.
- Reporting the bound as a result of a deployable method.
- Touching Method 2's or Method 3's results.

## How it is built

- `src/canvas/replay.py`: `CanvasConfig`, a single-configuration reference
  replay written with Method 3's helpers (`warp`, `fuse`, `paste_cells`,
  `_blocks`), and a batched replay (configurations sharing glimpse size and
  variant run in lockstep) that must equal the reference exactly. A test pins
  that.
- `foveation.latency_model.LatencyTables.predict_canvas_frame`: the table
  above. An unmeasured point raises, as elsewhere.
- `scripts/method4.py`: `bound` (per clip, chunked, resumable per clip,
  `.jsonl.gz`), `merge`, `score` (both judges, both references, the budget,
  the pass rule's verdict).
- The cache of variant A must have been built with IC-LK; `bound` refuses a
  cache whose homography method differs.
- Tests on the synthetic fixture, on the CPU.
- Ledger: `results/method4/TRIES.md`. Outcome: `results/method4/README.md`.

## Left to the builder

- Batch sizes and the layout of the output lines.
- Extra diagnostics (share of pixels taken from the canvas, k per frame).
