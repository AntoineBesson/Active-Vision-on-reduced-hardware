# Methods 7 and 8: guided upsampling of low-scale logits, and the per-frame scale oracle

Status: **draft, 2026-11-20, awaiting Antoine's acceptance. Nothing has been run.** Once the
first real run starts, this spec is frozen; any change goes in a dated Amendments section.

Context: Methods 2 to 6 all fail against the full frame at equal latency on a T4. Warps are
exhausted (Method 5 separable, Method 6 local 2-D). The 2026-11-19 memo "Options left before
fine-tuning" ranks the two training-free checks below first. Its argument: the T4 full-frame curve
is flat above scale 0.75 and latency is linear in pixels plus a floor, so a frozen add-on only has
room where the curve is steep (scales 0.375 to 0.625) and must cost about 1 ms or less.

Constraints (unchanged): no retraining, model-agnostic, both PIDNet-S and BiSeNetV2, Cityscapes-VPS,
`train_all` (400 clips) for every choice, `val` (50 clips) read once at the end.

## The baseline: a dense full-frame curve and the envelope of both models

Earlier readings interpolated linearly between five full-frame scales, which flattered methods that
landed between them (the moving-object run turned a +1.1 into −2.7 once 0.75 was measured). Here the
full frame is measured at 11 scales, every 1/16 between 0.25 and 0.75, then 0.875 and 1:

    0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.875, 1

The reference at a latency L is the **envelope**: the higher of the two models' full-frame curves
at L (piecewise linear between the 11 points of each). A method on PIDNet-S must beat BiSeNetV2's
full frame where that one is better (around 4 to 6 ms), and the other way round. The same-model
curve and the same-scale full frame are reported too, but the rules below use the envelope.

**Prices (T4, end to end from a full-resolution frame: resize, network, upsampling, argmax; FP16,
batch 1, CUDA graph).** Until Method 7's own T4 sessions exist, a scale is priced from the measured
latency-floor `final` sessions, linear in input pixels between the five measured scales (latency is
linear in pixels above ~0.25 MP, max residual 0.4 ms):

| scale | 0.25 | 0.3125 | 0.375 | 0.4375 | 0.5 | 0.5625 | 0.625 | 0.6875 | 0.75 | 0.875 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PIDNet-S ms | 4.00 | 4.56 | 5.25 | 6.08 | 7.04 | 8.31 | 9.73 | 11.30 | 13.02 | 16.94 | 21.47 |
| BiSeNetV2 ms | 4.27 | 5.33 | 6.61 | 8.21 | 10.05 | 12.23 | 14.65 | 17.34 | 20.27 | 26.32 | 33.30 |

Measured: 0.25, 0.375, 0.5, 0.75 and 1; the others are interpolated. The Method 7 timing
sessions measure all 11 scales and replace these for the val reading.

## Method 7: guided upsampling of low-scale logits

**Idea.** At scale c the model sees a c-times smaller frame; its logits are bilinearly upsampled,
so boundaries are blurred at the logit grid's resolution (stride 8 for PIDNet-S, so a 16 px grid
at c = 0.5; stride 1 for BiSeNetV2, a 2 px grid). A fast guided filter (He and Sun 2015) with the
full-resolution frame as guide replaces the bilinear step: per class channel p, on the logit grid,
`a = cov(I, p) / (var(I) + eps)`, `b = mean(p) − a mean(I)` (box means of radius r), then A and B
(box means of a and b) are upsampled the logits' own way and `q = A I + B`; the label is argmax q.
For an RGB guide, `a = (Σ_I + eps U)^-1 cov(I, p)` per grid cell. The guide is sampled on the grid
with the logits' own geometry, so `r = 0` gives back the bilinear baseline exactly (tested). The
model's output is read, nothing else: model-agnostic, training-free.

Code: `src/upsample/guided.py`, CLI `scripts/method7.py`.

**Grid (96 configurations per model):**

| factor | values |
|---|---|
| scale c | 0.25, 0.375, 0.5, 0.625 |
| guide | gray (mean of R, G, B), rgb |
| target | logit, prob (softmax) |
| radius r (logit-grid cells) | 1, 2 |
| eps (guide in [0, 1]) | 1e-4, 1e-3, 1e-2 |

**Cost.** The guided step's T4 overhead is measured paired with the full frame at the same scale,
in the same sessions (guided e2e minus full-frame e2e, per session, then the median over 3
sessions). Full-resolution steps in FP16, coefficients in FP32 on the grid. Price of a configuration
= the full frame's price at its scale + that overhead.

**Stage 1, train_all, zero overhead (gate 1).** Every configuration is read against the envelope at
the price of its scale alone (overhead 0, the most favourable case). Per model, finalists are the
configurations with **Δ vs envelope ≥ +0.5 mIoU point** at zero overhead, at most 2 per model (the
two highest). No finalist for either model: Method 7 stops, negative, and no T4 time is spent.

**Stage 2, T4 timing (after the SegFormer notebook frees the Colab T4).** `time` in 3 separate Colab
sessions: both models' full frame at the 11 scales and the finalists, interleaved, CUDA graphs.

**Stage 3, val, read once (the pass rule).** Each finalist at its T4 price. **Method 7 passes if a
finalist's Δ mIoU vs the envelope at equal latency is above 0 with its 95% clip-bootstrap interval
above 0.** Secondary, on `train_all` (the object table is train-only): the headline moving-object
recall (tiny + small objects that move, `results/objects`) of each finalist against the same
model's full-frame recall curve at equal latency; a finalist whose interval lies entirely below 0
there is reported as failing on moving objects even if it passes on mIoU.

## Method 8: per-frame scale oracle (a bound)

**Idea.** Some frames need a high scale and others don't. If a selector could choose the scale
per frame, the same mean latency could buy more mIoU than one fixed scale. Before building any
selector, the oracle bound: with the labels, pick the scale of every frame to maximise the dataset
mIoU under a budget on the mean latency. If even the oracle does not clear the margin, no
realisable selector can.

**Menu:** the same model's 11 full-frame scales (the frames of the dense run, nothing more to run).

**Solver** (`src/upsample/scale_oracle.py`): mIoU does not split over frames, so the oracle is a
Lagrangian relaxation (coordinate ascent on mIoU − λ × mean latency, bisection on λ for the
budget) followed by a greedy fill of the slack; best of 3 random orders. It can fall slightly short
of the exact optimum (checked exhaustively on small cases in the tests); that only weakens the
bound by as much.

**Budgets:** the same model's full frame at 0.5, 0.625 and 0.75 (where the curve is steep enough
for a mix to matter, and where the memo says room is left).

**Pass rule (train_all):** the bound passes if, at some budget for either model, **the oracle's
mIoU is at least +1.0 point above the envelope at its mean latency** (the same margin as Method 4's
bound). Reported with it: the clip-bootstrap interval (the oracle's selection held fixed), the
p95 latency of the selection, the share of each scale, a label-free fixed mix of two adjacent
scales at the same budget, and the headline moving-object recall of the oracle's selection.

**If it passes,** a realisable selector is specified next (a separate spec): candidates are cheap
signals already available, such as the previous frame's prediction entropy or class mix, priced on
the T4, and the pass rule becomes Method 7's (val, envelope, CI above 0). **If it fails,** per-frame
scale selection is closed.

Mean latency is the judge, as in Methods 3 and 4. A per-frame scale makes latency vary per frame;
a hard real-time budget would need p95, so p95 is reported.

## Runs

One GPU pass does both methods: per labelled frame, the model at each of the 11 scales, scored
with the bilinear step (per-frame confusions, kept for Method 8) and, at the 4 Method 7 scales,
with the 24 guided configurations of that scale (clip sums). `--obj ff` also records every thing
object's local IoU for the full-frame rows (moving-object metric and Method 8's selection); the
finalists' object IoUs are a second, short run with `--configs`.

```bash
# train_all, per model and chunk (cluster: scripts/cluster/m7_run.sh; laptop: add --max-temp 85)
python -m scripts.method7 run --split train_all --models pidnet_s --chunk 0/4 --obj ff
python -m scripts.method7 merge --split train_all
python -m scripts.method7 score --split train_all --write-finalists      # gate 1, finalists
python -m scripts.method7 oracle --split train_all                       # Method 8
# finalists' object IoUs, then the moving-object readings
python -m scripts.method7 run --split train_all --models pidnet_s --chunk 0/4 --ff-scales none \
    --configs <finalists> --obj all --tag fin_c0of4
python -m scripts.method7 merge --split train_all
python -m scripts.method7 objects --split train_all
# T4, 3 Colab sessions (stage 2)
python -m scripts.method7 time --session s1
# val, once (stage 3)
python -m scripts.method7 run --split val --models pidnet_s,bisenetv2 --guided-scales none --configs <finalists>
python -m scripts.method7 merge --split val
python -m scripts.method7 score --split val --finalists results/method7/finalists_{model}.json \
    --timing "experiments/method7/timing/timing_s*.json"
```

Laptop rule: one GPU run at a time, pause at ≥ 85 °C (two readings) or any thermal-slowdown flag,
resume at ≤ 65 °C (`--max-temp 85`). Cluster submissions are typed by Antoine.

## What would make the results wrong

* **Prices between measured scales.** Linear in pixels is accurate on the T4 (max residual 0.4 ms
  on the area fit), and the val reading uses Method 7's own measurements at all 11 scales.
* **The guided step's FP16 full-resolution part.** Tested to agree with FP32 to 5e-3 in score; the
  accuracy runs use FP32 throughout, the timing FP16, as the full frame's own upsampling.
* **Selection on train.** Finalists come from train only; val is read once, on the finalists only.
* **The oracle uses labels**: it is only a bound, never a result.

## Amendments

### 2026-11-22: train_all results, and Method 8 step 2 (predictability screen), fixed before it runs

Antoine accepted the spec by submitting the cluster run on 2026-11-20 (jobs 1114454 to 1114461, all 8 chunks
complete, 400 clips per model). Results on train_all (files in `results/method7/`):

* **Method 7, gate 1.** PIDNet-S: STOP. Every guided configuration is below the bilinear step at its
  own scale; the best is −1.99 [−2.21, −1.79] vs the envelope. BiSeNetV2: CONTINUE on the rule,
  with finalists `7g/c0.25/rgb/logit/r2/e0.0001` (+0.53 [+0.50, +0.57] at zero overhead) and
  `7g/c0.25/rgb/prob/r2/e0.0001` (+0.52). The curve's slope there is about 3.8 points per ms, so the
  finalists pass the val rule only if the T4 overhead stays under about 0.14 ms. Stage 2 (T4 timing)
  runs when the Colab T4 is free; nothing else in Method 7 changes.
* **Method 8 bound: PASSES.** Oracle vs envelope at the budgets ff 0.5 / 0.625 / 0.75: PIDNet-S
  +4.85 / +3.97 / +3.21, BiSeNetV2 +2.08 / +1.10 / +0.98. The headline moving-object recall of the
  oracle's selections stays within ±0.7 points of the full frame at the same budget.

**Step 2, before any selector is built: is the oracle's gain predictable?** The oracle picks each
frame's scale from its own labels, so part of its gain can be per-frame noise that no selector can
anticipate. Without any new GPU pass, from the same per-frame confusions (`method7 screen`):

* **stale oracle**: each labelled frame takes the oracle's choice of the previous labelled frame of
  its clip (5 frames, about 0.3 s, earlier); the first labelled frame of a clip takes the budget
  scale. This is what a selector that sees only the past could do if it read the past perfectly.
* reported with it: the **clip oracle** (one scale per clip, chosen with labels), and two nulls, the
  oracle's choices **shuffled within each clip** and **shuffled over all frames**.

Each is read against the envelope at its own mean latency, with the same clip bootstrap.
**Rule:** a realisable selector is specified and built only if the **stale oracle keeps at least
+1.0 mIoU point above the envelope** at some budget for either model. Otherwise Method 8 is closed:
the oracle's gain is not predictable from the past. If the stale oracle passes, the selector spec
fixes its features (the previous frame's own output, such as entropy and class mix, at no extra
pass), its cost on the T4 and a val pass rule like Method 7's.
