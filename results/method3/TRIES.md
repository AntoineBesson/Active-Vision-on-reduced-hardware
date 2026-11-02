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
