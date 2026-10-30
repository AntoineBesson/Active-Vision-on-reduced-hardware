# Method 2: ledger of tries

Every attempt at Method 2 (foveation without memory), in order, including the
ones that failed, so the paper can report what was tried and why it was
abandoned or changed. Each entry: what ran, on which split, the numbers, the
conclusion and what changed next. Numbers are on VPS **train** unless stated;
val is evaluated once, on finalists only. Latencies are predicted T4 medians
from the latency-floor tables (batch 1, fp16, CUDA graphs), with the full
frame timed as the backbone alone (no resize, upsampling or argmax), which
slightly favours the full frame.

Raw per-configuration results stay in `experiments/method2/` (not committed);
each try has a compact summary in `tries/`.

---

## Try 1: overwrite the context with full-resolution glimpses (failed)

**Date:** 2026-10-30. **Summary:** `tries/try1_pidnet_replace_train.json`.

**Setup.** PIDNet-S for context and glimpses. Context at scale 0.25 / 0.375 /
0.5; glimpses of 256 / 384 / 512 px at full resolution, 32 px margin, k = 1,
2, 4, 8; the glimpse's argmax overwrites the context inside each selected
cell. Selection: entropy / margin / max-probability scores, mean or top-10%
pooling, top-k or greedy; references: pixel oracle (labels), fixed prior,
random (5 seeds), every cell (tiling bound). 549 configurations.

**Result.** No configuration reaches the full-frame curve. At equal T4
latency the best rule of each geometry is **5 to 19 mIoU points below** the
full frame (best: context 0.5 + one 256 px glimpse, 67.9 mIoU at 9.5 ms,
against the full frame interpolated at that latency, ~73).

Full frame on train: 76.1 (scale 1.0, 19.4 ms), 74.8 (0.75, 10.7 ms), 67.9
(0.5, 5.2 ms), 61.7 (0.375, 3.6 ms), 52.9 (0.25, 2.7 ms).

**Cause: glimpses lack context.** With every cell glimpsed (no selection at
all) the tiled prediction scores only **53.5 / 60.3 / 63.0 mIoU** for 256 /
384 / 512 px crops, against 76.1 for the full frame. A probe on 5 val clips
(PIDNet-S, mIoU / pixel accuracy) shows accuracy climbing back to the full
frame as crops and margins grow, so this is the network, not a bug:

| Input | mIoU | Pixel acc. |
| --- | --- | --- |
| full frame | 57.5 | 95.6 |
| 256 px crops, 32 px margin | 44.3 | 86.1 |
| 384 px, 32 | 49.5 | 89.6 |
| 512 px, 32 | 47.8 | 90.1 |
| 512 px, 128 | 52.2 | 92.4 |
| 768 px, 64 | 54.7 | 93.8 |
| 1024 px, 64 | 55.3 | 93.5 |
| 1024 px, 256 | 56.4 | 95.0 |

(5 clips only: absolute mIoU is low because rare classes barely occur; the
trend is the point.) PIDNet-S is trained on 1024x1024 crops and has a global
pooling branch; a 256–512 px crop does not see enough of the scene to name
whole regions correctly.

**Other observations.**
- Among selection rules, max-probability or margin with top-10% pooling and
  top-k wins almost everywhere; the fixed prior often matches or beats it.
- The pixel oracle (k cells adding the most correct pixels) is far above every
  rule (e.g. 65.9 vs 58.1 at context 0.25, 4 x 384 px), so selection has room,
  but even the oracle stays below the full-frame curve because of the tiling
  bound.
- The oracle maximises correct pixels, not mIoU, so it is not an mIoU ceiling.

**Next (decided with the user):** keep the glimpse but fuse it with the
context inside the cell instead of overwriting it (try 2), and wide glimpses
that see more of the scene at reduced resolution (try 2). Larger margins at
full resolution were set aside for now (more pixels for the same cell).
