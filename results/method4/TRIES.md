# Method 4 (persistent canvas + glimpse policy): ledger of tries

Every try, failures included, for the paper. Phase 1 is a train-only
feasibility bound (`train_all`, 30 frames per clip); val is not read in it.

## 2026-11-10 — spec fixed before any run

Grid, latency parts and pass rule are in
`docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md`, committed
(`1195da1`) before the first replay. Two points differ from the brief and are
flagged there: Method 2's `overhead` table is replaced by Method 3's `post` and
`glimpse_post` entries (measured for every k, without the scoring), and the
motion map is left out of the bound and reported as a side reading.

## 2026-11-10 — the Method 3 caches are not on the laptop any more

`experiments/method3/` holds the sweep and bounds outputs only; the per-clip
`.npz` caches (about 20 GB) are gone, and the cluster was not reachable from
this session to check for chunk 3's. They have to be rebuilt before the bound:
with `--full-scales 1` (the bound needs only the teacher, the three context
scales and the three glimpse sizes) one clip takes 16 s and 19.5 MB on the RTX
2060, so about 27 min and 2 GB per model per chunk.

## 2026-11-10 — a design effect seen on the test fixture

On a static synthetic clip, a correct full-resolution glimpse in the canvas is
overruled by the low-resolution context wherever the context is at least as
confident (the 3b rule gives ties to the context, and uint8 confidence
saturates at 255). The oracle then glimpses those cells again. This is the
fusion rule the brief asked for, kept as is; it means the canvas saves fewer
glimpses than its coverage suggests. Fixture only, dummy backbone.

## 2026-11-10 — smoke run on one real clip (not a reading)

Clip 0001 of `train_all`, PIDNet-S, laptop RTX 2060, fp32: cache 16 s; `bound`
on the 693 configurations 23.5 s (GPU memory fine at batch 32, 0.56 MB of
output); `score` 6 s. The pipeline runs end to end at 1024×2048 and the rule is
reported as "not read (partial: 1 clips)", as designed. One clip says nothing
about the curve (its full-frame mIoU is 0.59 against 0.76 on the 400 clips):
no number from it is kept. Estimate from it: about 40 min per model per chunk
for `bound`, about 2 h 40 min per model for the 400 clips.

## 2026-11-10 — chunk 0 (100 clips of 400): a partial reading, not the rule

Laptop RTX 2060, fp32, both models, default grid (693 configurations). Time:
caches 68 min for the two models (4.1 GB), `bound` 51 min (PIDNet-S) and 52 min
(BiSeNetV2, 31 s per clip), `merge` and `score` 3 min: 2 h 55 in all, so about
8 h 45 for chunks 1 to 3. `score` reports "rule not read (partial: 100 clips)".
The full-frame curve below is the one of the same 100 clips.

Best configuration of each variant, mean judge, primary reference (points of
mIoU against the curve at equal mean latency):

| model | N (no canvas) | B (no warp) | A (IC-LK) |
|---|---|---|---|
| PIDNet-S | −5.11 (`c0.5/g384/k4`, 15.6 ms) | −4.88 (`c0.5/g512/a1/y0.02`, 19.3 ms) | −6.30 (`c0.5/g512/a2/y0.05`, 18.9 ms) |
| BiSeNetV2 | −0.85 (`c0.5/g512/y0.02`, 29.3 ms) | −1.00 (`c0.5/g384/a1/y0.02`, 29.9 ms) | −1.47 (`c0.5/g384/a1/k4`, 30.2 ms) |

No configuration is above the curve, under either judge or either reference
(best against the measured end-to-end curve: −3.97 PIDNet-S, −0.89 BiSeNetV2,
both variant B). What the 100 clips show, to be confirmed on the 400:

- **The ceiling is the glimpses, not the selector.** The highest mIoU of any
  configuration is 0.717 (PIDNet-S, 33 ms) and 0.727 (BiSeNetV2, 48 ms); the
  full frame at scale 0.75 gives 0.744 in 10.7 ms and 0.728 in 17.8 ms. The
  oracle cannot reach the curve even when its latency is ignored.
- **The canvas adds almost nothing at equal k.** With a_max = 1, variant B is
  within ±0.2 point of N at the same context, glimpse size and k (best case
  +0.55 at `g512/k1`, PIDNet-S), for 0.4 ms of fusion. Variant A gains up to
  +0.7 point over N (`g512/k1`) but pays about 8 ms of gray, IC-LK and warps.
- **Old canvas pixels hurt.** With a_max = ∞ every configuration loses about
  1 point to a_max = 1, in A as in B: stale glimpses overrule a context that
  is right. The canvas share stays under 0.45.
- **The adaptive selector saves few glimpses.** With the canvas the oracle
  asks for 5 to 10 % fewer cells at the same γ (PIDNet-S), and sometimes more
  (BiSeNetV2, variant B).

## 2026-11-10 — chunk 1: reading on 200 clips of 400, still partial

Laptop RTX 2060, 2 h 52 (caches 69 min, `bound` 98 min for the two models).
The cluster was not reachable (SSH timed out), so chunks 2 and 3 go to Colab
with `notebooks/method4_bound_colab.ipynb` (written for this, not yet tested
there when committed). `score` still reports "rule not read (partial: 200
clips)". Curve of the same 200 clips; mean judge, primary reference:

| model | N (no canvas) | B (no warp) | A (IC-LK) |
|---|---|---|---|
| PIDNet-S | −4.56 (`c0.5/g512/y0.02`, 19.1 ms) | −4.65 (`c0.5/g512/a1/y0.02`, 18.7 ms) | −5.81 (`c0.5/g512/a2/y0.05`, 18.5 ms) |
| BiSeNetV2 | −1.00 (`c0.5/g384/y0.02`, 28.5 ms) | −0.94 (`c0.5/g384/a1/y0.02`, 29.3 ms) | −1.17 (`c0.5/g384/a1/k4`, 30.2 ms) |

Same picture as chunk 0: no configuration above the curve for either model
(best against the measured end-to-end curve: −3.39 and −0.93, variant B). The
highest mIoU of any configuration is 0.714 (PIDNet-S, 33 ms) and 0.723
(BiSeNetV2, 48 ms), under the full frame at scale 0.75 (0.743 and 0.728).

## 2026-11-10 — chunk 3: reading on 300 clips of 400, still partial

Laptop RTX 2060, 3 h 05 (caches 70 min, `bound` 109 min for the two models; the
GPU sat at 86 °C). Chunk 2 was meant for Colab, but its zip was no longer on
Drive (`FileNotFoundError` in the notebook's data cell); the zip was rebuilt
(7.12 GB) and chunk 2 was run on the laptop after chunk 3 instead, with a
watchdog that stops the run at 90 °C. On chunks 0, 1 and 3, mean judge, primary
reference: best canvas configuration −4.59 points (PIDNet-S,
`4o-B/c0.5/g512/a1/y0.02`, 18.2 ms) and −0.98 point (BiSeNetV2,
`4o-B/c0.5/g384/a1/y0.02`, 28.5 ms). The same two configurations as on 100 and
200 clips, within 0.3 point.

## 2026-11-10 — chunk 2, and the reading on the 400 clips: the check FAILS

Chunk 2 on the laptop, 3 h 05; the GPU peaked at 90 °C and the watchdog did not
have to stop it. `merge`: 4 files, 400 clips, 277 200 lines per model. `score`
on the complete set reads the pre-registered rule
(`experiments/method4/train_all/verdict_train_all.json`): **fails for both
models**. Best configuration of variant A or B at equal mean latency: PIDNet-S
−4.56 points (`4o-B/c0.5/g384/a1/y0.05`, 12.4 ms), BiSeNetV2 −0.85 point
(`4o-B/c0.5/g384/a1/y0.02`, 27.9 ms). Of the 693 configurations per model,
none is above the curve under the mean or the p95 judge, against the primary or
the measured end-to-end reference. The numbers are in the README.

Two points seen only on the full grid, which correct the chunk 0 entry (it
looked at context 0.5 only):

- The canvas does help a weak context. At equal k against variant N, mean over
  glimpse sizes and k: +1.4 (B, a_max 1) to +3.1 points (A, a_max 4) at
  context 0.25 for PIDNet-S, +0.8 to +1.9 for BiSeNetV2. At context 0.5 the
  gain is +0.0 (B) and +0.2 to +0.3 (A), and a_max = ∞ loses 0.6 to 1.1 point.
  The configurations nearest the curve all use context 0.5, where there is
  little left for the canvas to add.
- BiSeNetV2's full-frame curve is flat at the top on these clips: 0.7339 at
  scale 0.75 (17.8 ms) against 0.7335 at scale 1 (31.2 ms). A configuration at
  28 ms is compared with a plateau it does not reach (best mIoU of any
  configuration: 0.7266 at 43 ms).

Total cost of the bound: about 12 h of laptop GPU for the four chunks.
