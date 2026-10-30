# Latency floor on a T4

What a frame costs on the target GPU, before any accuracy is measured: backbone
latency against input area and against the number of batched crops, the
foveation pipeline's own overhead (identity backbone), and end-to-end checks of
the full pipeline against the sum of its parts. Random weights and frames, no
dataset.

- Script: [`scripts/latency_floor.py`](../../scripts/latency_floor.py)
- Notebook: [`notebooks/latency_floor_colab.ipynb`](../../notebooks/latency_floor_colab.ipynb)
- Raw results: `latency_floor_t4-s{1,2,3}.json`, one per Colab session

## Protocol

Tesla T4 (driver 580.82, torch 2.11 + CUDA 13.0, cuDNN 9.27), batch 1, FP16,
streaming. Every configuration is measured both eagerly and as a CUDA graph.
Per-frame CUDA events after `synchronize`, 20 warmup frames, 5 interleaved
rounds × 40 frames (A, B, C, A, B, C…), no data loading and no host sync in
the timed region. NVML telemetry is sampled every 100 ms. Three separate Colab
sessions (about 13 min each). Below, "median" is the median over the three
session medians.

Every session ran at the **70 W software power cap** (`sw_power_cap`, the
T4's normal state under load). The SM clock varied between 585 and 1,590 MHz
(session median 1,140–1,215 MHz), with temperatures up to 80–82 °C. These are
the conditions the numbers describe.

## Repeatability

Between sessions, a configuration's median varies by a median of 3–5% and at
most 8% (BiSeNetV2, 8 × 768² crops). Session 3 is about 4% faster on large
inputs than sessions 1–2, which is consistent with a cooler GPU (80 vs 82 °C).
Speedups vary by ±3% between sessions. No conclusion below depends on a
difference that size.

## Findings

**1. Eager mode is launch-bound; CUDA graphs are required.** Eagerly,
PIDNet-S takes about 11.5 ms and BiSeNetV2 about 7.5 ms for any input up to
1.5 MP. As graphs, the floor drops to 2.2 and 1.5 ms. In eager mode the
foveated PIDNet-S (24.5 ms) is slower than the full frame. All methods are
compared as CUDA graphs.

**2. Above ~0.25 MP, latency is linear in pixels** (graph, fit on inputs of
0.26–2.1 MP, max residual 0.4 ms):

| Model | Intercept | Per megapixel | Full 1024×2048 |
| --- | --- | --- | --- |
| PIDNet-S | 0.8–0.9 ms | 8.3–8.8 ms | 19.4 ms (18.6–19.6) |
| BiSeNetV2 | 0.4–0.5 ms | 14.0–14.7 ms | 31.2 ms (29.8–31.2) |

Full-frame curve (graph, median ms), with VPS-val mIoU from the
[full-frame baseline](../../README.md#real-data-results):

| Scale | 0.25 | 0.375 | 0.5 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S ms | 2.73 | 3.59 | 5.22 | 10.74 | 19.35 |
| PIDNet-S mIoU | 50.1 | 58.6 | 64.8 | 70.1 | 71.5 |
| BiSeNetV2 ms | 2.69 | 4.72 | 8.06 | 17.78 | 31.16 |
| BiSeNetV2 mIoU | 59.1 | 64.3 | 67.6 | 69.9 | 70.4 |

**3. Batching crops saves nothing beyond small crops.** k crops cost about k
times one crop once the batch exceeds ~0.25 MP (graph, median ms):

| Crop | k=1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S 128² | 2.28 | 2.33 | 2.59 | 3.23 | 4.80 |
| PIDNet-S 256² | 2.27 | 2.74 | 3.83 | 6.11 | 11.39 |
| PIDNet-S 384² | 2.71 | 3.82 | 6.13 | 11.44 | 22.65 |
| PIDNet-S 512² | 3.49 | 5.50 | 9.78 | 19.79 | 37.70 |
| BiSeNetV2 128² | 1.51 | 1.48 | 1.72 | 2.67 | 4.28 |
| BiSeNetV2 256² | 1.84 | 2.63 | 4.17 | 7.76 | 14.96 |
| BiSeNetV2 384² | 2.90 | 4.69 | 8.69 | 16.93 | 33.29 |
| BiSeNetV2 512² | 4.25 | 7.88 | 15.07 | 30.04 | 59.07 |

**4. Pipeline overhead is 2–7 ms** (resize, entropy, cell scores, selection,
crop, upsampling, argmax, paste back; graph, top-k):

| Logit stride | Context 0.25 | 0.375 | 0.5 |
| --- | --- | --- | --- |
| 8 (PIDNet-S) | 1.9–4.1 ms | 2.1–4.3 ms | 2.0–4.2 ms |
| 1 (BiSeNetV2) | 2.6–4.2 ms | 3.4–5.2 ms | 4.6–6.9 ms |

It grows with k and the crop size. With stride 1 it also grows with the
context scale, because the context logits are already at full resolution.
Greedy selection costs 0.03–0.35 ms more than top-k. Eager and graph costs are
about the same, so this is memory-bound work. It's 15–25% of a foveated frame,
which makes it a target for optimisation.

**5. The sum of the parts predicts the pipeline.** Context + glimpse batch +
overhead, each measured separately, gives the measured end-to-end median to
within 1–4% as graphs (measured / predicted 0.98–1.04). Any configuration's
latency can therefore be estimated from these tables, with only the finalists
timed end to end.

**6. Realised vs FLOPs-implied speedup** (graph, against the same model at
full frame; range over the three sessions):

| Context + glimpses | ms | Speedup | FLOPs-implied | Realised / implied |
| --- | --- | --- | --- | --- |
| PIDNet-S, 0.25 + 4×384 | 11.5 | 1.63–1.70× | 2.91× | 0.57 |
| PIDNet-S, 0.375 + 8×256 | 12.6 | 1.48–1.55× | 2.56× | 0.59 |
| PIDNet-S, 0.5 + 2×512 | 13.6 | 1.37–1.44× | 2.00× | 0.69 |
| PIDNet-S, 0.25 + 4×512 | 15.7 | 1.19–1.25× | 1.78× | 0.67 |
| BiSeNetV2, 0.25 + 4×384 | 14.4 | 2.07–2.18× | 2.91× | 0.72 |
| BiSeNetV2, 0.375 + 8×256 | 16.3 | 1.83–1.93× | 2.56× | 0.72 |
| BiSeNetV2, 0.5 + 2×512 | 21.4 | 1.39–1.46× | 2.00× | 0.70 |
| BiSeNetV2, 0.25 + 4×512 | 21.2 | 1.41–1.49× | 1.78× | 0.80 |
| BiSeNetV2 ctx + PIDNet-S glimpses, 0.25 + 4×384 | 12.1 | 1.54–1.62× (vs PIDNet-S) | 2.43× | 0.65 |

The realised speedup is 57–80% of what the FLOPs suggest. The gap is the
per-call floor (finding 2) plus the pipeline's own overhead (finding 4).
Pairing a BiSeNetV2 context with PIDNet-S glimpses is never faster than
PIDNet-S alone.

## What this means for Method 2

- **The equal-latency bar is high for PIDNet-S.** The full frame at scale 0.75
  takes 10.7 ms for 70.1 mIoU, only 1.4 points below full resolution.
  So the fastest foveated configuration tested (11.5 ms) has to exceed about
  70 mIoU to be worth anything.
- **BiSeNetV2 has more room.** Full resolution costs 31 ms, while scale 0.5
  gives 67.6 mIoU at 8.1 ms. A foveated BiSeNetV2 at 14–16 ms competes with
  full frames at scales 0.5–0.75 (67.6–69.9 mIoU).
- **Few, mid-sized crops.** Since batching saves nothing above ~0.25 MP, cost
  is set by the total number of crop pixels plus a fixed ~2–4 ms. Many small
  crops (≤ 256²) are the only case where batching helps.
- **Cut the overhead next.** The growth with context scale at stride 1 points
  at the full-resolution upsampling of the context logits. The ops haven't
  been profiled one by one yet.
