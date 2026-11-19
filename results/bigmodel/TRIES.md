# Larger frozen models (SegFormer): ledger of tries

Question: is the small models' flat full-frame curve the reason every warp
loses? A larger frozen model costs more per pixel, its attention grows faster
than the pixel count, and its accuracy may depend more on resolution, so a warp
may have more room on it. The rule stays **accuracy at equal measured T4
latency against every model's full-frame curve** (PIDNet-S, BiSeNetV2 and the
large model itself), not a slower, more accurate system.

## 2026-11-19 — screen written before any run

Code: `src/models/segformer.py` (Hugging Face `transformers` SegFormer,
NVIDIA's public Cityscapes checkpoints `nvidia/segformer-b{0,2,5}-finetuned-cityscapes-1024-1024`,
nothing trained), `scripts/bigmodel_curve.py`, notebook
`notebooks/bigmodel_curve_colab.ipynb` (3 Colab T4 sessions; accuracy in the
first only).

- **Latency:** PIDNet-S, BiSeNetV2, SegFormer B0 / B2 / B5 at scales 0.125,
  0.1875, 0.25, 0.375, 0.5, 0.75, 1; fp16, CUDA graphs, all models interleaved
  in the same rounds; `area` (backbone alone) and `e2e` (resize, backbone,
  upsampling to the frame, argmax). Median over the 3 sessions.
- **Accuracy:** VPS val mIoU, fp32, `run_baseline` at the same scales.
- **Screen (not a pass rule, a go / stop for building a bound):** GO if some
  SegFormer scale below 1 is within **2.0 points** of the best small model's
  mIoU at the same latency (linear between that model's two measured scales
  around it, never extrapolated; above its slowest point, its full-frame mIoU)
  **and** SegFormer's own curve gains at least **1.0 point** at the next larger
  scale. Read on `e2e` and on `area`; both reported. GO means: run the Method
  5/6 oracle bound on that model and scale. STOP on both readings means a
  bigger frozen model does not give a warp room either.
- Known difference to the published SegFormer numbers: NVIDIA evaluates with a
  sliding 1024 × 1024 window; this harness runs every model on the whole
  resized frame.
