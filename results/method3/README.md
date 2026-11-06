# Method 3: motion-gated reuse of the previous frame — results

**Outcome: negative on VPS train; stopped before val.** On a T4, none of the
1296 configurations lands above the full-frame mIoU-vs-latency curve, for
either model and under either latency judge (mean and p95 per frame). The zone
oracle, tested on the rules without fallback, stays below the curve too. That
meets the plan's stop condition, so val was not run: every number here is on
train. The tries that led here, failures included, are in [TRIES.md](TRIES.md).

## What Method 3 does

On frame 0, a full pass at scale s_full; its prediction becomes the memory.
On each later frame:

1. Estimate a homography from the previous frame (IC-LK on downscaled
   grayscale), warp the previous frame with it, and threshold the difference
   at θ. Pixels with no source in the warp count as moving.
2. A cell of Method 2's grid is moving when its share of moving pixels is
   above ρ.
3. If the share of moving cells is above τ, run a full pass and reset the
   memory (the fallback; τ = 1 means never).
4. Otherwise warp the memory and recompute the moving cells as full-resolution
   glimpses (margin 32).

Variant **3a** does only this. Variant **3b** also runs a low-resolution pass on
every non-fallback frame and fuses it with the warped memory in the static
cells, keeping the more confident prediction per pixel.

Code: `src/reuse/`, `scripts/method3.py`. Design:
[spec](../../docs/superpowers/specs/2026-11-02-method3-motion-reuse-design.md),
[plan](../../docs/superpowers/plans/2026-11-02-method3-motion-reuse.md).

## Protocol

- **Data.** VPS train, 400 clips, 2400 labelled frames. All VPS clips come from
  Cityscapes val sequences, so neither backbone trained on these frames (see
  [Method 2](../method2/README.md)). The comparison with the full-frame curve
  uses the same frames. The configs reported here are the best of the grid on
  these clips, so their readings lean in Method 3's favour.
- **Grid.** s_full 1, 0.75, 0.5; glimpse 256, 384, 512; θ 25, 38, 51; ρ 0.02,
  0.05, 0.1; τ 0.2, 0.4, 0.6, 1; 3b context 0.25, 0.375, 0.5. That gives 324
  configurations of 3a and 972 of 3b, for each model.
- **Accuracy.** mIoU in fp32, by exact replay from per-clip caches. Chunks 0-2
  ran on the laptop RTX 2060 and chunk 3 on a P100.
- **Latency.** Predicted per frame from T4 tables (t4-s1..3 and m3-s1..3; fp16,
  batch 1), using the moved cells and fallbacks of the sweep. Backbone passes
  use CUDA-graph timings, the same level as the full-frame curve.
- **Reading.** mIoU against the full-frame curve at the same latency, in points
  (1 pt = 0.01 mIoU). A configuration slower than the full frame at scale 1 has
  no reading. No trade-off scalar is chosen.

## Results on train

| | PIDNet-S | BiSeNetV2 |
| --- | --- | --- |
| full frame, scale 1 | 0.7608, 19.35 ms | 0.7335, 31.16 ms |
| full frame, scale 0.75 | 0.7480, 10.74 ms | 0.7339, 17.78 ms |
| full frame, scale 0.5 | 0.6794, 5.22 ms | 0.7055, 8.06 ms |
| configurations above the curve, either judge | 0 / 1296 | 0 / 1296 |
| best 3a, mean judge | −7.26 pts (0.6870, 18.5 ms) | −4.61 pts (0.6876, 24.9 ms) |
| best 3b, mean judge | −9.54 pts (0.6627, 17.5 ms) | −2.57 pts (0.7079, 28.9 ms) |
| best 3a, p95 judge | no reading: fastest p95 is 20.6 ms | −6.90 pts (0.6646, 28.5 ms) |
| best 3b, p95 judge | no reading: fastest p95 is 27.0 ms | −15.78 pts (0.5757, 31.5 ms) |
| readable configurations, mean / p95 judge | 603 / 0 | 1153 / 141 |

The best configuration under each judge is picked separately. Unreadable
configurations are slower than the full frame at scale 1 and, in every case, less
accurate than it, so they are dominated. Best mIoU over the whole grid: 0.7229 for
PIDNet-S and 0.7081 for BiSeNetV2.

The other reading, latency saved at equal mIoU, is negative wherever it exists:
at best −10.51 ms (3a) and −12.13 ms (3b) for PIDNet-S. BiSeNetV2 has none,
because its train curve does not rise from scale 0.75 to scale 1 (0.7339 to
0.7335), so `latency_at` is undefined by design. By hand, on the rising part of
the curve, its best 3b config (0.7079, 28.9 ms) matches a full frame of about
8.9 ms.

## Bounds on train

Rules without fallback (τ = 1), 81 per model, mean judge:

| | PIDNet-S | BiSeNetV2 |
| --- | --- | --- |
| best motion rule | −25.35 pts | −18.86 pts |
| best zone oracle at the rule's k | −11.88 pts | −5.56 pts |
| best rule on the raw difference (no homography) | −28.34 pts | −18.59 pts |
| oracle − motion at the same k, median | 11.88 pts | 10.74 pts |
| raw − aligned mIoU, median | +3.62 pts, for 12% more of the frame recomputed | +4.02 pts, for 12% more |

The oracle has no reading above the curve under p95 either. It picks cells
against the scale-1 full-frame prediction as a teacher, so it is a ceiling
relative to that teacher, not an absolute one. It was not run on rules with a
fallback.

Reuse bound (warp the prediction forward, recompute nothing), scale 1:

| Frames since the full pass | 1 | 2 | 5 | 10 | 29 |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S | 0.708 | 0.660 | 0.571 | 0.484 | 0.328 |
| BiSeNetV2 | 0.683 | 0.640 | 0.555 | 0.473 | 0.323 |

Drift by labelled position in the clip (frames 4, 9, …, 29), s_full 0.75, 384 px
glimpses, θ 25 or 38, ρ 0.02:

| Config | PIDNet-S | BiSeNetV2 |
| --- | --- | --- |
| 3a, τ = 1, θ 38 | 0.582 0.523 0.491 0.479 0.470 0.451 | 0.590 0.544 0.513 0.494 0.490 0.468 |
| 3a, τ = 0.2, θ 25 | 0.688 0.682 0.682 0.694 0.681 0.688 | 0.682 0.684 0.683 0.690 0.687 0.685 |

## Why it loses

- **Deciding costs too much.** On the T4, grayscale, homography and motion map
  take about 6.8 ms before any segmentation (median gray 0.50, IC-LK 4.88, motion
  1.26-1.60 ms), more than a whole PIDNet-S frame at scale 0.5 (5.22 ms). With the
  warps and fusion, a reused frame costs about 9.5 ms before its first glimpse.
- **Reuse is not free in accuracy.** One frame old already costs about 5 pts
  against the full frame.
- **Motion is a poor proxy for error.** At the same number of recomputed cells,
  the oracle is about 11 pts better than the motion rule, and the raw difference
  also beats the aligned one at about 12% more recomputation.
- **Keeping accuracy needs the full pass.** The best configurations fall back on
  about 65% of frames, and each fallback pays the decision on top of the pass.
  Without fallback, drift takes the rules far lower.

## Checks

- **Same clip on two GPUs.** Clip 0376 on the RTX 2060 and the P100, all 1296
  configurations: moved cells and fallbacks identical; lines not byte-identical
  (16 and 0 of 1296 match); at most 6.6e-6 of pixels differ; clip mIoU differs by
  at most 0.0007 pts. One clip only.
- **Sum of parts against end-to-end frames on the T4.** The plan's 5% limit fails
  for two of the four cases when graph timings are summed, as `score` does:

  | Case | Measured | Sum, graph passes | Sum, eager passes |
  | --- | --- | --- | --- |
  | PIDNet-S 384 k2 | 22.23 ms | 14.79 ms (−33.5%) | 22.48 ms (+1.1%) |
  | PIDNet-S 384 k8 | 24.55 ms | 24.31 ms (−1.0%) | not computed separately |
  | BiSeNetV2 384 k4 | 20.22 ms | 20.19 ms (−0.2%) | not computed separately |
  | PIDNet-S 256 k12, context 0.25 | 38.56 ms | 26.03 ms (−32.5%) | 37.64 ms (−2.4%, k12 interpolated) |

  The end-to-end frames run the backbone eagerly, and eager PIDNet-S has a floor of
  about 11.7 ms whatever the input size. The eager sum is confirmed on the two
  failing cases; the graph-level sum has no end-to-end check. Graph timing favours
  Method 3, because it assumes one captured graph per k. Real latency would be equal
  or worse, so this cannot turn the negative result positive.

## Limits

- **Train only.** No finalists, val run, bootstrap intervals, fp16 accuracy check
  or `reuse_final` T4 run: the plan stops before them. The fp32/fp16 mismatch
  between accuracy and latency stays stated, not measured, for this method.
- **Latencies are predicted**, from parts, not measured on whole clips.
- **The T4 sessions ran at a lower clock** than t4-s1..3 (median 960-1080 MHz
  against 1140-1215 MHz, under the 70 W cap), and large BiSeNetV2 batches read up
  to 15% slower in m3-s2 and m3-s3. The tables merge all sessions.
- **This tests one memory scheme**, where motion picks the cells and the memory is
  the last prediction. It does not rule out other uses of memory.

## Files

- [`score_train_all_pidnet_s.json.gz`](score_train_all_pidnet_s.json.gz),
  [`score_train_all_bisenetv2.json.gz`](score_train_all_bisenetv2.json.gz): every
  configuration with both judges, per-class IoU and drift, plus the bounds.
- [`../latency_floor/`](../latency_floor/): `latency_floor_m3-s{1,2,3}.json`.
- `motion_stats_train_all*.json`: the motion-map statistics behind the grid.

To re-read a file without the sweep:
`python -c "import gzip,json; d=json.load(gzip.open('results/method3/score_train_all_pidnet_s.json.gz')); print(d['fullframe'])"`
