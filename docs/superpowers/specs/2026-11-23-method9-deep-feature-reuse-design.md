# Method 9: reuse the slow branch across frames (frozen two-branch models)

Status: **accepted by Antoine 2026-11-24; gate 0 coded (`src/reuse/slowfast.py`,
`scripts/method9.py`, `notebooks/method9_gate0_colab.ipynb`), not yet run.** Points marked
*(my call)* were settled by Claude, not the user. Antoine's answers to the open points are at the end.

## Not black-box model-agnostic (Antoine 2026-11-24: must be written)

Method 9 is **frozen** (no weight changes, no training, the public checkpoints) but it is **not
model-agnostic** in the sense of Methods 2 to 8, which only touched the input or the output of
an unmodified network:

- it reads and injects **internal tensors** of the network (the semantic branch of BiSeNetV2, the
  I branch of PIDNet), so it needs the model's code, not just a callable;
- it only exists for models whose architecture has a **separable low-resolution context
  branch**. The split is hand-written per family (`src/reuse/slowfast.py` refuses any other
  model). A single-path network (SegFormer, a plain encoder-decoder) has no such split without
  choosing a cut point inside it, which is the clockwork design space and not free;
- how much it can save is set by the architecture (gate 0), not by the method.

Positioning in the project: the model-agnostic methods (2 to 8) are written up first as
documented negative results; Method 9 is the first **architecture-aware, still frozen** step,
and work on improving the base model (training) only comes after it. Results of Method 9 are
reported under that label and never as model-agnostic.

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
two-branch, so the split is the architecture's own, not one we invent.

### Related work: what exists, and what is not new here (checked 2026-11-24)

| Work | What is reused / recomputed | Trained for the reuse? | Closest to us in |
|---|---|---|---|
| **Clockwork Convnets** (Shelhamer et al., ECCV-W 2016, [arXiv 1608.03609](https://arxiv.org/abs/1608.03609)) | FCN-8s (VGG16) cut into 3 stages; shallow stage every frame, deep score stages cached and fused by the net's own skip fusion. Fixed schedules (alternating, exponential, skip) and an adaptive one (pool4 score change). | No schedule-specific training is described: the fixed schedules run the ordinary FCN. | **The mechanism.** Method 9's variant N is essentially a fixed clockwork schedule. |
| Deep Feature Flow (Zhu et al. 2017), Accel (Jain et al. 2019), DVSNet (Xu et al. 2018) | Key-frame features warped by optical flow, plus a correction branch or a learned key-frame policy. | Yes (flow net, fusion, policy). | Key frames. |
| Low-Latency VSS (Li et al., CVPR 2018, [arXiv 1804.00389](https://arxiv.org/abs/1804.00389)) | Low-level features every frame, high-level features propagated by learned spatially variant convolution; adaptive key frames. | Yes. | Fresh low-level path + stale high level. |
| TDNet (Hu et al., CVPR 2020) | Deep features split into sub-networks spread over consecutive frames, attention fusion. | Yes (a new architecture). | Spreading the slow part over frames (our "out of scope" p95 fix). |
| DCFM (Zheng et al., AAAI 2024, [arXiv 2403.02689](https://arxiv.org/abs/2403.02689)) | Key frame's deep "common" feature reused directly (no flow); a shallow encoder runs every frame and a fusion module merges them. | Yes, end to end with dedicated losses; own encoder-decoder, not a two-branch real-time net. | Reuse without propagation. |

**What is therefore not new:** caching deep / slow features and refreshing shallow ones with a
fixed schedule (Clockwork, 2016). We must cite it as the method and not present the mechanism
as ours.

**What we have not found in the literature** (searches on clockwork / slow-branch reuse with
BiSeNet, PIDNet, DDRNet, and training-free video acceleration of frozen image models): (1) the
clockwork schedule on **modern two-branch real-time models** whose context branch is a
designed, separable path, **frozen**, with the cut at the architecture's own branch boundary;
(2) judged against the **resolution-scaling envelope of several models at equal measured
latency** on a T4 (Clockwork compares to frame skipping and per-frame cost, not to simply
running at a lower resolution); (3) **moving, small-object recall** as a co-requirement. So the
contribution, if it passes, is an evaluation of a 2016 idea on 2020s real-time models under a
stricter baseline, not a new mechanism. If it fails, it is a documented negative result for
that idea under this baseline.

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
both models at s ∈ {0.5, 1} (Antoine 2026-11-24: more scales only if this looks promising). The
same session also times both models' full frame at the 11 envelope scales, so the envelope is
priced in the same session as the method.

The most a schedule can gain is bounded by K → ∞ and by the assumption that stale context does
not beat fresh context (accuracy ≤ the same model's full frame at s):

    ceiling(model, s) = FF_model(s) − Envelope(T_fresh(s))     [mIoU points, train_all]

**Gate 0 rule:** a model continues only if `ceiling(model, s) ≥ +1.0` at some s. If neither model
does, Method 9 stops and no accuracy GPU time is spent. This needs no new accuracy number: the
full-frame mIoU per scale already exists on train_all.

What would make it fail: if the fast branch is most of the cost (BiSeNetV2's detail branch is
wide and at 1/8; PIDNet-S's stem is shared), `T_fresh` is close to `T_full` and nothing is left to
win. That is exactly what this gate measures.

**Prediction from FLOPs, before the T4 run (2026-11-24, `eval.cost.count_flops`, 1024 x 2048 frame):**

| Model | s | full | fresh | slow alone | fresh / full |
|---|---|---|---|---|---|
| PIDNet-S | 0.5 | 23.6 G | 15.9 G | 13.1 G | 67 % |
| PIDNet-S | 1 | 94.4 G | 63.6 G | 52.4 G | 67 % |
| BiSeNetV2 | 0.5 | 49.2 G | 46.5 G | 2.7 G | 95 % |
| BiSeNetV2 | 1 | 196.8 G | 186.1 G | 10.7 G | 95 % |

(The PIDNet slow part counts the stem again, which the split does not recompute.) BiSeNetV2's
semantic branch is only ~5 % of its FLOPs: its head (1024 channels at 1/8) and detail branch
dominate, so it is expected to **fail** gate 0 unless the depthwise semantic branch is much
slower on the T4 than its FLOPs say. For PIDNet-S, if latency follows FLOPs (fresh ≈ 67 % of the
key frame), the ceiling against the train_all envelope is about **+1 point at s = 1** (borderline)
and **several points at s = 0.5**, where both curves are steep. The T4 decides.

Code: `src/reuse/slowfast.py` (the split; `tests/test_method9_split.py` checks that with the same
frame's slow tensors it equals the model's own forward, bit-exact for the key frame and to 1e-5
for the fresh pass, and that a stale cache changes the output and is never written to);
`scripts/method9.py gate0` (timing) and `ceiling` (verdict); the train_all full-frame curves are
copied to `results/method9/fullframe_train_all.json` from Methods 7/8.

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
- Grid: 2 models × 2 scales × 4 K × 2 variants = 32 schedules (only models that passed gate 0), from 2400 labelled frames ×
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

## Antoine's answers (2026-11-24)

1. **Margins:** +1.0 point at gates 0 and 1, kept.
2. **Moving objects:** a co-requirement of gate 1, as proposed.
3. **Scales:** 0.5 and 1 only for now; more scales only if it looks promising.
4. **Gate 1 on the school cluster.** Gate 0 on one Colab T4 session.
5. Not model-agnostic: written above. The model-agnostic failures are written up first, then
   work moves to improving the base model.
