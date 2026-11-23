# Method 9: reuse the slow branch across frames (frozen two-branch models)

Status: **draft for Antoine's acceptance, 2026-11-23. Nothing coded or run.** Points marked
*(my call)* were settled by Claude, not the user. Open points for the user are listed at the end.

## Why this and not Method 3 again

Method 3 reused the **prediction** of the previous frame and recomputed only the cells where
something moved ([results/method3](../../../results/method3/README.md)). It failed on three
measured facts, and the method below avoids each of them:

| Method 3 failure (train_all, T4) | What Method 9 does instead |
|---|---|
| Deciding what to recompute cost ~6.8 ms (gray, IC-LK homography, motion map) before any segmentation. | No decision step. The schedule is fixed (a key frame every K frames); variant N uses no motion estimate at all. |
| A one-frame-old **prediction**, even homography-warped, already loses ~5 mIoU points (PIDNet-S 0.761 → 0.708). | No prediction is reused. Every frame gets a fresh pass of the model's **fast, high-resolution branch**; only the **slow, low-resolution semantic features** are reused, and the model's own fusion layer combines them with fresh detail. |
| Motion is a poor proxy for error; moving objects were the worst case (moving-object recall −23 pts). | The fresh branch sees the current frame everywhere, so object positions and edges come from the current frame. Stale information is only coarse context (1/32-1/64 of the input). |

This is the "clockwork" idea (Shelhamer et al. 2016: update deep layers less often than shallow
ones) applied to a frozen network with no retraining. Both of our models are explicitly
two-branch, so the split is the architecture's own, not one we invent. The published versions
that work well (Deep Feature Flow, Accel, TDNet; see `docs/research/RELATED.md`) all train a flow
or fusion module. Nobody has shown whether a frozen real-time model tolerates stale context
features; that is the question.

It still predicts every frame from that frame's own pixels, so the earlier "no frame skipping"
decision (Method 3 non-goals) is respected *(my call)*. Antoine opened key-frame reuse on
2026-11-23.

## The method

Per clip, at a fixed input scale s and key-frame interval K:

1. **Key frame** (t = 0, K, 2K, ...): the full model at scale s. Cache the slow-branch tensors.
2. **Other frames**: run only the fast branch at scale s on the current frame, read the cached
   slow-branch tensors, and run the model's own fusion and head unchanged.

The split, per model (tensor names from `src/models/`):

- **BiSeNetV2.** Slow = the semantic branch (`segment`, its output `feat_s` at 1/32). Fresh = the
  detail branch (`detail`, 1/8), then `bga(feat_d, feat_s)` and `head`. This is the split the
  architecture was designed around.
- **PIDNet-S.** The three branches share a stem (`conv1`, `layer1`, `layer2`, 1/8), which stays
  fresh. Slow = the I branch: `layer3`, `layer4`, `layer5`, `spp`, plus `compression3/4` and
  `diff3/4` (functions of the I branch only). Fresh = stem, P branch (`layer3_`..`layer5_`,
  with `pag3/4` reading the cached compressions), D branch (`layer3_d`..`layer5_d`, adding the
  cached `diff3/4`), `dfm`, `final_layer`. *(my call: the stem could also be split, but it is
  shared and at 1/8, so reusing it would make the P and D branches stale too.)*

**Variants of the reused tensors:**

- **N (no alignment):** the cached tensors as they are. Costs nothing beyond the fresh branch.
- **H0 (free alignment, a bound only):** the cached tensors warped to the current frame with the
  homography chain of Method 3 (IC-LK, key frame → current frame; read from the Method 3 caches
  on the laptop, or recomputed with `src/reuse/motion.py`), **priced at 0 ms**. It says how much alignment could add; it cannot pass the rule on its own (see gate 1).

Weights are the same public checkpoints as every other method. Nothing is trained.

## Baseline and prices

The same baseline as Methods 7 and 8: the **envelope** of PIDNet-S's and BiSeNetV2's dense
full-frame curves (11 scales, 0.25 to 1) at equal T4 latency, end to end (resize, network,
upsampling, argmax; FP16, batch 1, CUDA graph). If SegFormer's accuracy run puts any of its
points above that envelope, it joins the envelope *(my call)*.

A schedule's **mean** per-frame latency is
`(T_full(s) + (K−1)·T_fresh(s)) / K`, with `T_fresh(s)` timed with the cached tensors as static
inputs (so the cache read is in the timing). **Mean latency is the judge**, as in Methods 3, 4
and 8. The p95 is reported too, and is stated plainly: a key frame costs `T_full(s)`, so this
method **cannot** improve worst-case latency, only throughput. A hard real-time budget would
need the slow pass spread over several frames, which is outside this spec.

## Gate 0: the cost split, T4 timing only (one Colab session, ~15 min)

Before any accuracy run. On the T4, interleaved, CUDA graphs: `T_full(s)` and `T_fresh(s)` for
both models at s ∈ {0.5, 0.625, 0.75, 0.875, 1}.

The most a schedule can gain is bounded by K → ∞ and by the assumption that stale context does
not beat fresh context (accuracy ≤ the same model's full frame at s):

    ceiling(model, s) = FF_model(s) − Envelope(T_fresh(s))     [mIoU points, train_all]

**Gate 0 rule:** a model continues only if `ceiling(model, s) ≥ +1.0` at some s. If neither model
does, Method 9 stops and no accuracy GPU time is spent. This needs no new accuracy number: the
full-frame mIoU per scale already exists on train_all.

What would make it fail: if the fast branch is most of the cost (BiSeNetV2's detail branch is
wide and at 1/8; PIDNet-S's stem is shared), `T_fresh` is close to `T_full` and nothing is left to
win. That is exactly what this gate measures.

## Gate 1: the staleness bound on train_all (cluster or Colab)

Only for a model that passed gate 0. Data: VPS `train_all`, 400 clips, all 30 frames (the same
chunk zips as Methods 3 to 5), labels on frames 4, 9, ..., 29.

- For each labelled frame t and each staleness k ∈ {0, 1, 2, 3, 4} (k = 0 is the full frame):
  slow tensors from frame t−k, fresh branch on frame t, at each s of gate 0, variants N and H0.
  Store one 19 × 19 confusion per (frame, model, s, k, variant), plus the predictions the
  moving-object metric needs (`src/eval/objects.py`).
- A schedule with interval K puts a labelled frame at staleness 0..K−1 with equal probability
  (key-frame phase uniform). Its accuracy is the mIoU of the confusions **pooled over
  k = 0..K−1**, which is the exact expectation over phases *(my call)*. K ∈ {2, 3, 4, 5}.
- Grid: 2 models × 5 scales × 4 K × 2 variants = 80 schedules, from 2400 labelled frames ×
  ~9 passes per (model, scale). Estimated ~2 to 3 h on one cluster P100 per model.

**Gate 1 rule (pre-registered):** a model continues if some schedule in **variant N** has
**ΔmIoU ≥ +1.0 point vs the envelope at its mean latency** (the margin of Methods 4 and 8), **and**
its headline moving-object recall (tiny + small objects that move, `results/objects`) against the
envelope at the same latency has a 95 % clip-bootstrap interval **not entirely below 0**. At most
2 finalists per model (the two highest Δ). *(my call: moving objects are a co-requirement here,
not only reported, because Antoine named them the priority; see open point 2.)*

If only H0 passes: the method stops as specified, and the write-up states the slack (how many ms
an alignment step could cost and still pass), so a cheaper motion source can be judged later.

## Phase 2: the real thing, only after gate 1

- **T4 timing:** 3 Colab sessions, the finalists end to end on streamed clips (key frames and
  fresh frames in their real order) interleaved with both models' full frame at the 11 scales.
- **Val, read once:** the finalists on VPS val with these prices. **Pass:** Δ mIoU vs the envelope
  at equal mean latency above 0, with its 95 % clip-bootstrap interval above 0 (Method 7's rule).
- **Check:** at K = 1 the method must reproduce the full frame to float precision (pipeline check,
  in the test suite before any run).

## Non-goals

No training or fine-tuning of anything, no learned flow, no INT8 / TensorRT or other inference
tricks (Antoine 2026-11-23: unfair comparison), no prediction reuse (Method 3), no learned or
adaptive key-frame policy in this spec (only if gate 1 passes, as a separate spec).

## Open points for Antoine

1. **Margins:** gate 0 and gate 1 at +1.0 point, as Methods 4 and 8. Keep?
2. **Moving objects as a co-requirement of gate 1** (proposed above), or reported only as in Method 7?
3. **Scales:** 0.5 to 1 only (where the fresh branch is big enough to matter), or the full 11?
4. **Where to run gate 1:** the school cluster (my default, as Method 6) or Colab. Gate 0 needs one Colab T4 session.
