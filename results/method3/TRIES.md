# Method 3 (previous-frame reuse): ledger of tries

Every try, failures included, for the paper. Tuning data is VPS train
(`train_all`, 30 frames per clip); val is used once, on frozen finalists.

## Early check of the motion cue (2026-11-02, Task 2, train only)

Six frame pairs (clips 0001, 0188, 0376; t = 1 and 15) of `train_all`, motion
at 256×512, grid 384:32 (28 cells), θ = 30, ρ = 0.05. Not a tuning run: a
sanity check that the homography fix does what the design assumes.

| clip, t | raw difference: moving cells / px > θ | ORB + RANSAC | GPU IC-LK |
| --- | --- | --- | --- |
| 0001, 1 | 14 / 12.2% | 11 / 8.7% | 10 / 6.8% |
| 0001, 15 | 11 / 8.6% | 8 / 5.8% | 6 / 4.5% |
| 0188, 1 | 4 / 3.0% | 0 / 0.9% | 0 / 0.8% |
| 0188, 15 | 3 / 2.5% | 1 / 1.3% | 1 / 1.0% |
| 0376, 1 | 5 / 2.5% | 6 / 3.8% | 2 / 0.9% |
| 0376, 15 | 5 / 2.9% | 3 / 1.3% | 2 / 0.8% |

- The aligned difference flags fewer cells than the raw one, as intended.
- IC-LK (dense, GPU-able) is better than ORB on every pair. ORB once fitted the
  wrong motion (0376, t = 1: translation 12.7 px against IC-LK's −4.5) and did
  worse than no alignment.
- **Cost warning:** ORB takes about 30 ms per pair on the laptop CPU, more
  than PIDNet-S's whole full frame on the T4 (19 ms). Unless the Colab host's
  CPU is much faster, the CPU estimator cannot pay for itself; the T4 timing in
  Task 5 decides. IC-LK took 157 ms on the laptop CPU and will run on the GPU.
- Open: the ego car's hood (bottom rows, static in the image) is moved by the
  warp and may show as motion; to check on the train statistics (Task 7).

## Cost of the motion cue, first measurement (2026-11-02, Task 5, laptop RTX 2060)

A `--quick` smoke run of the new latency parts on the laptop GPU (not the
judge; the T4 sessions `m3-s*` are) found the per-frame overhead far too high:
IC-LK 240 ms, the label + confidence warps 18 ms, ORB 32 ms (host CPU), against
19 ms for PIDNet-S's whole full frame on the T4. Causes and fixes:

| op | before | after | fix |
| --- | --- | --- | --- |
| warps (labels 1024×2048 + confidence 256×512) | 18.4 ms | 3.4 ms | sampling grid in float32, not float64 (consumer GPUs and the T4 run float64 at ~1/32 speed), H applied elementwise |
| IC-LK homography | 240 ms | 7.7 ms | no GPU waits in the loop (`solve_ex`, a 3×3 adjugate inverse, the mask as a weight), 2 pyramid levels (8 + 4 iterations), the whole fit replayed from a CUDA graph |
| motion map + moving cells | – | 1.4 ms | float32 warp |

Accuracy is unchanged on the tests (a known 12 × 20 px shift is still recovered
within 1 px; replay = method pixel for pixel). Even so, homography + motion +
warps cost about 13 ms per frame on the laptop GPU, so the room left for
glimpses under PIDNet-S's 10.7–19 ms full frames is small. The T4 numbers decide.

## 2026-11-02 — `method3 stats` on one real train clip (CPU, ORB)

Smoke run of the CLI on train_all clip 0001 (29 frame pairs): no identity fallback;
aligned |diff| percentiles 95/98/99 = 35/54/71; at g = 384 the share of moving cells is
1.5–37% across the 9 (theta, rho) pairs. One clip only: theta is fixed by chunk 0 on Colab.

## 2026-11-02 — motion statistics on chunk 0 (100 train clips, 2,900 frame pairs, CPU)

| homography | identity fallbacks | aligned \|diff\| p95 / p98 / p99 | moving cells at g = 384, ρ = 0.05 (θ = p95 / p98 / p99) |
|---|---|---|---|
| ORB | 0 | 30 / 50 / 75 | 28.2% / 13.5% / 7.7% |
| IC-LK | 0 | 25 / 38 / 51 | 28.3% / 12.1% / 6.2% |

IC-LK leaves smaller residuals after alignment, as on the 6 pairs checked earlier; it is
kept, and θ ∈ {25, 38, 51}. Files: `motion_stats_train_all{,_orb,_iclk}.json`.

## 2026-11-02 — replay on the GPU: exactness and speed

- **Exactness on real val clips (laptop GPU):** the replay and the streaming method give the
  same moving cells and fallbacks on every frame; predictions differ on at most 2.4e-6 of a
  frame's pixels. Cause, confirmed directly: the cache glimpses all 28 cells in one batch,
  the method only the moving ones, and cuDNN's float noise across batch sizes flips the
  argmax of 2 pixels in one cell. Exact on the CPU fixture.
- **Speed:** one replay took 0.32 s per clip (launch- and copy-bound: 428 ms CPU for 152 ms
  GPU), i.e. ~190 s per clip for the 1,296-config grid. Cache arrays kept on the GPU
  (0.24 s), then configs sharing (glimpse, θ, ρ) replayed in one batch (48 configs in 2.7 s,
  equal to the single replays bit for bit): the full grid now takes 25 s per clip, 480
  replays after merging the τ values with the same fallback frames. A sub-histogram trick
  for the batched confusion gave nothing (2.69 → 2.77 s) and was dropped.
- One real clip cache (3 full scales, 3 glimpse sizes, 3 context scales): 23 s, 24 MB.

## 2026-11-02 — chunk 0 × PIDNet-S on the laptop GPU (cache, sweep, bounds)

100 train clips (one cache already built by the smoke run). Cache 45 min (~18 s/clip,
2.5 GB); sweep 52 min (31 s/clip: 984 distinct replays per clip, not 480 — real clips have
more distinct fallback patterns across τ than the fixture); bounds 2 h 02 min (72 s/clip,
now the slowest stage). Outputs: `experiments/method3/train_all/{sweep,bounds}_train_all_pidnet_s_chunk0.jsonl.gz`
(129,600 sweep lines, 100 bounds lines). No errors.

## 2026-11-02 — chunk 0 × BiSeNetV2 on the laptop GPU (cache, sweep, bounds)

Same timings as PIDNet-S: cache 45 min (27 s/clip, 2.6 GB), sweep 52 min (984 replays per
clip), bounds 2 h 03 min. 129,600 sweep lines, 100 bounds lines. No errors.

## 2026-11-03 — chunk 1 × PIDNet-S and BiSeNetV2 on the laptop GPU

PIDNet-S: stopped overnight for GPU cool-down at cache 59/100, resumed (cache skipped the
59 built clips: 13 min for the other 41), sweep 51 min, bounds 2 h. BiSeNetV2: cache 53 min,
sweep 52 min, bounds 2 h. 840 distinct replays per clip on this chunk. 129,600 sweep lines
and 100 bounds lines per model. No errors.

## 2026-11-03 — chunk 2 × PIDNet-S and BiSeNetV2 on the laptop GPU

PIDNet-S: cache 36 min, sweep 43 min, bounds 2 h. BiSeNetV2: cache 60 min, sweep 44 min,
bounds 2 h. 129,600 sweep lines and 100 bounds lines per model (smaller files than chunks
0–1). No errors. GPU at 84 °C right after the run.

## 2026-11-05 — moving chunk 3 to the school cluster (Tesla P100)

- Compute nodes have no internet: the venv was installed from the gateway, weights uploaded.
- First replay check job (1088898) passed the test (worst differing share 4.77e-07 of a
  frame on the P100) but my job script's `grep " 1 passed"` failed, because `pytest -q`
  starts that line with no space; the dependent jobs never ran. The Method 3 notebook had
  the same check and would have refused every Colab run: fixed, with a test.

## 2026-11-05 — chunk 3 × PIDNet-S and BiSeNetV2 on the cluster P100

Jobs 1088932 (PIDNet-S, node16) and 1088933 (BiSeNetV2, node17) ran side by side in fp32 at
commit `adf4376`; later commits change only the notebook check, its test and this file.
Whole job: 1 h 59 min and 1 h 53 min, against about 3 h 40 min per model on the laptop. Cache
plus sweep took about 49 min, bounds 64–70 min. Both printed `JOB_OK`, with no error lines in
the logs. The four files came back with matching MD5s. 129,600 sweep lines and 100 bounds
lines per model, as on chunks 0–2. Across the four chunks: 400 distinct clips per model, none
twice, same fields in every file. These are accuracy runs only: no P100 time goes in a latency
figure.

Merge: 518,400 sweep lines and 400 bounds lines per model, no duplicates, no clip missing
against the dataset (18 min). `score` stops as expected: "no T4 measurement for reuse", until
the m3-s1..3 latency sessions come back. The Method 2 full-frame reference holds the same 400
clips.

## 2026-11-05 — code clean-up, checked against the saved results

Behaviour-preserving clean-up of the Method 3 code after the sweep. To check it, clip 0376 was
re-run from scratch on the cleaned code (cache, sweep, bounds; P100, job 1089442) and compared
with the lines saved by the chunk 3 run at `adf4376`: 1296 of 1296 sweep lines and the bounds
line identical, for both models. The real-data replay test passes. Three real-data tests of
Method 2's cache fail on the cluster only because its data holds no `val/` folder (they did not
run on the laptop either, where they skip without the environment variables). The latency script's `reuse` and
`reuse_e2e` parts run in `--quick` mode. They now time the method's own functions; no Method 3
latency had been measured before this change. The 400 clips were not re-run.

## 2026-11-05 — the same clip on two GPUs (RTX 2060 against P100)

Chunks 0–2 ran on the laptop and chunk 3 on the P100, so no clip had run on both. Clip 0376 was
re-run from scratch on the laptop (cache, sweep, bounds; 4 min 17 s for both models, GPU at 85 °C
right after) and compared with the P100's chunk 3 lines, all 1296 configs.

| | PIDNet-S | BiSeNetV2 |
| --- | --- | --- |
| moved cells and fallbacks identical | 1296 / 1296 | 1296 / 1296 |
| lines identical to the byte | 16 / 1296 | 0 / 1296 |
| pixels differing, median / max | 1.4e-6 / 5.3e-6 | 1.3e-6 / 6.6e-6 |
| clip mIoU difference, max | 0.0007 pts | 0.0007 pts |

Same size as the float noise already seen across batch sizes on one GPU. The motion decisions are
the same, so the latency predictions are too. One clip only.

## 2026-11-05 — T4 sessions m3-s1..3

Three separate Colab T4 sessions at commit `9c27522` (30–32 min each; all 458 configurations; the
files were pushed to `foveation-no-memory` by mistake and copied here). Clock median 960–1080 MHz
under the 70 W software cap, against 1140–1215 MHz in t4-s1..3. The points shared with t4-s1..3
agree closely except for large BiSeNetV2 batches in m3-s2 and m3-s3 (up to +15% at 384 k16); the
tables merge all sessions.

Cost of a frame's steps on the T4, median: gray 0.50 ms, IC-LK 4.88 ms (p95 9.22), motion map
1.26–1.60 ms, warps 2.62–2.67 ms, fusion 0.42 ms. ORB 33.8 ms on the Colab CPU, so IC-LK stays.
Before any glimpse, a reused frame already costs about 9.5 ms, and deciding alone (gray, homography,
motion) about 6.8 ms: more than PIDNet-S's whole 0.5-scale frame (5.22 ms).

**End-to-end check of the sum of parts** (plan limit 5%): two of four fail as planned.

| case | measured median | sum with graph passes (what `score` uses) | sum with eager passes |
| --- | --- | --- | --- |
| PIDNet-S 384 k2 | 22.23 ms | 14.79 ms (−33.5%) | 22.48 ms (+1.1%) |
| PIDNet-S 384 k8 | 24.55 ms | 24.31 ms (−1.0%) | about the same (eager and graph agree at k8) |
| BiSeNetV2 384 k4 | 20.22 ms | 20.19 ms (−0.2%) | about the same |
| PIDNet-S 256 k12, context 0.25 | 38.56 ms | 26.03 ms (−32.5%) | 37.64 ms (−2.4%; eager k12 interpolated between k8 and k16) |

The end-to-end frames run the backbone eagerly, and eager PIDNet-S has a floor of about 11.7 ms
whatever the input size. So the sum of parts is confirmed at eager level; the graph-level sum is
not confirmed end to end. `score` uses graph passes, the same level as the full-frame curve. This
favours Method 3: it assumes one captured graph per k, which are many graphs (each with its private
memory pool) that a real deployment would have to hold.

## 2026-11-05 — score on train_all (400 clips): Method 3 is below the curve everywhere

`score` on both models, 1296 configs, T4 tables t4-s1..3 + m3-s1..3. Points are mIoU against the
full-frame curve at the same latency; a config slower than the full frame at scale 1 has no reading.

| | PIDNet-S | BiSeNetV2 |
| --- | --- | --- |
| full frame, scale 1 | 0.7608, 19.35 ms | 0.7335, 31.16 ms |
| full frame, scale 0.75 | 0.7480, 10.74 ms | 0.7339, 17.78 ms |
| full frame, scale 0.5 | 0.6794, 5.22 ms | 0.7055, 8.06 ms |
| configs above the curve, either judge | 0 / 1296 | 0 / 1296 |
| best 3a, mean judge | −7.26 pts (0.6870, 18.5 ms) | −4.61 pts (0.6876, 24.9 ms) |
| best 3b, mean judge | −9.54 pts (0.6627, 17.5 ms) | −2.57 pts (0.7079, 28.9 ms) |
| best 3a, p95 judge | no reading: fastest p95 20.6 ms, slower than the full frame | −6.90 pts (0.6646, 28.5 ms) |
| best 3b, p95 judge | no reading: fastest p95 27.0 ms, slower than the full frame | −15.78 pts (0.5757, 31.5 ms) |
| best motion rule without fallback, mean judge | −25.35 pts | −18.86 pts |
| zone oracle at the rule's k, mean judge | −11.88 pts | −5.56 pts |
| oracle − motion rule at the same k, median | 11.88 pts | 10.74 pts |
| raw difference − aligned, median | +3.62 pts, for 12% more of the frame recomputed | +4.02 pts, for 12% more |

Reuse bound at scale 1, mIoU by lag 1 / 2 / 5 / 10 / 29: PIDNet-S 0.708 / 0.660 / 0.571 / 0.484 /
0.328; BiSeNetV2 0.683 / 0.640 / 0.555 / 0.473 / 0.323. One frame old already costs about 5 pts.

The best configs fall back to a full pass on about 65% of frames, and their full pass costs more
than the plain full frame because the decision was paid first. Even the oracle, which knows the
errors, stays below the curve under both judges. A preview with all reuse steps counted as free
also had 0 configs above the curve, except one zone-oracle reading on PIDNet-S (+0.67 pts at 2.9 ms).
Val not touched.

## 2026-11-05 — closed as negative, val not run

The plan's stop condition (Task 7 step 4) is met, and the user chose to close Method 3 rather than
run val anyway. So Task 8's finalists, val sweep, intervals, fp16 check and `reuse_final` run were
not done. Outcome in [README.md](README.md); the score files are kept next to it, gzipped.
