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
