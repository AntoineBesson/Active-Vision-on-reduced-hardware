# Method 4, phase 1: feasibility bound of a persistent canvas — negative

**Result (2026-11-10): the feasibility check fails for both backbones, so
Method 4 is not built.** With an oracle selector that costs nothing and a
persistent canvas, no configuration reaches the full-frame curve on VPS train
on a T4. The grid, the latency parts and the pass rule were fixed in the
[spec](../../docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md)
before any run. Tries are logged in [TRIES.md](TRIES.md).

## The question

Method 4 would be a learned glimpse policy over a persistent canvas of past
glimpses. Before building it: with a near-perfect cell selector that costs
nothing, can a canvas beat the full-frame curve on VPS train on a T4? If not,
no learned per-frame policy will.

## What is replayed

Per frame, frame 0 included (no full pass anywhere):

1. a context pass at scale c ∈ {0.25, 0.375, 0.5};
2. the canvas (labels, confidence and age of past glimpses) is aged and, in
   variant A, warped by the IC-LK homography; variant B does not warp;
3. context and canvas are fused by Method 3's 3b rule (the more confident
   wins, the context on ties); canvas pixels older than a_max are ignored;
4. a greedy oracle picks cells against the scale-1 full pass of the same
   frame: fixed k ∈ {1, 2, 4, 8}, or every cell whose gain is above
   γ ∈ {0.005, 0.02, 0.05} of a cell;
5. the picked cells' glimpses (256, 384 or 512 px, margin 32) are pasted and
   written into the canvas.

Variant N has no canvas (Method 2 with the oracle selector). 693
configurations per model; 630 of them have a canvas.

The oracle is a heuristic ceiling, not a proof: it is per-frame and greedy,
and relative to the teacher. A planning policy could in principle do better.

## How it is read

- **Accuracy:** mIoU on the 2400 labelled frames of `train_all`, fp32, by
  replay of the Method 3 caches. No network pass, no val.
- **Latency:** predicted per frame from the T4 tables (t4-s1..3, m3-s1..3):
  context pass and its post-processing, k glimpses and theirs, fusion, and for
  variant A gray, IC-LK and the warps. The policy costs 0. The motion map is
  not counted (it would be a policy input); the reading with it is given too.
- **Reference:** the full-frame curve of Method 3's `score`, under the mean
  and the p95 judge. Second reference: the full frame's measured end-to-end
  latency (`latency_floor_final-s*.json`).
- **Budget:** the latency saved at equal mIoU is what a real policy may cost.

**Pass rule:** for at least one backbone, the best configuration of variant A
or B is ≥ +1.0 mIoU point above the curve at equal mean latency, on the 400
clips. Otherwise the check fails and Method 4 is not built.

## How to run

The Method 3 caches are needed (`experiments/method3/cache/train_all/<model>/`,
about 20 MB per clip with `--full-scales 1`). They are skipped where they
exist. PowerShell, from the repository root:

```powershell
$env:CITYSCAPES_VPS_ROOT = "C:/Users/trash/cityscapes_vps"
$env:PIDNET_S_WEIGHTS = "C:/Users/trash/weights/PIDNet_S_Cityscapes_val.pt"
$py = "C:/tmp/avrh-venv/Scripts/python.exe"

# chunk 0 (100 clips): caches, then the bound
& $py -m scripts.method3 cache --split train_all --chunk 0/4 --full-scales 1
& $py -m scripts.method4 bound --split train_all --chunk 0/4
& $py -m scripts.method4 merge --split train_all
& $py -m scripts.method4 score --split train_all      # partial: the rule is not read

# chunks 1 to 3, then the reading on the 400 clips
foreach ($c in 1, 2, 3) {
  & $py -m scripts.method3 cache --split train_all --chunk "$c/4" --full-scales 1
  & $py -m scripts.method4 bound --split train_all --chunk "$c/4"
}
& $py -m scripts.method4 merge --split train_all
& $py -m scripts.method4 score --split train_all      # writes verdict_train_all.json
```

On Colab, `notebooks/method4_bound_colab.ipynb` runs one chunk (set `CHUNK`)
from the chunk zips of the Method 3 notebook and saves the two bound files to
`MyDrive/avrh/method4/train_all/`; copy them into `experiments/method4/train_all/`
before `merge`.

`bound` appends each finished clip and, started again, skips the clips already
in the file. Outputs go to `experiments/method4/train_all/`. `--models
pidnet_s` runs one model.

## Results

400 clips of `train_all` (2400 labelled frames), 693 configurations per model.
mIoU points against the full-frame curve at equal latency; best configuration
of each variant.

| Model | Judge, reference | N (no canvas) | B (no warp) | A (IC-LK) |
|---|---|---|---|---|
| PIDNet-S | mean, T4 tables | −4.39 (`c0.5/g384/y0.02`, 17.3 ms) | **−4.56** (`c0.5/g384/a1/y0.05`, 12.4 ms) | −6.06 (`c0.5/g512/a2/y0.05`, 18.0 ms) |
| PIDNet-S | p95, T4 tables | −4.39 | −4.57 | not readable |
| PIDNet-S | mean, measured end to end | −2.63 | −2.74 | −5.28 |
| BiSeNetV2 | mean, T4 tables | −0.90 (`c0.5/g384/y0.02`, 27.0 ms) | **−0.85** (`c0.5/g384/a1/y0.02`, 27.9 ms) | −1.05 (`c0.5/g384/a1/k4`, 30.2 ms) |
| BiSeNetV2 | p95, T4 tables | −1.24 | −1.19 | −1.54 |
| BiSeNetV2 | mean, measured end to end | −0.79 | −0.85 | −1.06 |

**Verdict (rule read on the bold cells): fails.** The rule asked for +1.0 point
above the curve for at least one backbone; the best is −0.85. No configuration
is above the curve under any judge or reference, so there is no latency budget
for a real policy. ("Not readable": the configurations' p95 latency is past the
end of the curve.)

Full-frame curve on the same clips (mIoU, T4 median):

| Scale | 1.0 | 0.75 | 0.5 | 0.375 | 0.25 |
|---|---|---|---|---|---|
| PIDNet-S | 0.761, 19.3 ms | 0.748, 10.7 ms | 0.679, 5.2 ms | 0.617, 3.6 ms | 0.529, 2.7 ms |
| BiSeNetV2 | 0.734, 31.2 ms | 0.734, 17.8 ms | 0.706, 8.1 ms | 0.669, 4.7 ms | 0.592, 2.7 ms |

### Why it fails

- **The glimpses are the ceiling, not the selector or the memory.** The highest
  mIoU of any configuration is 0.718 for PIDNet-S (33 ms) and 0.727 for
  BiSeNetV2 (43 ms). The full frame at scale 0.75 gives 0.748 in 10.7 ms and
  0.734 in 17.8 ms. The oracle does not reach the curve even with its latency
  ignored: a frozen model on crops is worse than the same model on the whole
  frame at a slightly lower scale.
- **The canvas helps only where the context is weak.** At equal k against
  variant N, it adds 1.4 to 3.1 points at context 0.25 for PIDNet-S (0.8 to 1.9
  for BiSeNetV2), but 0.0 to 0.3 point at context 0.5, where every
  configuration near the curve sits. At most 56 % of the output pixels come
  from the canvas.
- **Old canvas pixels hurt a good context.** At context 0.5, a canvas without
  age limit (a_max = ∞) is 0.6 to 1.1 point below no canvas at all.
- **Alignment does not pay for itself.** Variant A is about 0.2 point above B
  at context 0.5 and costs about 8 ms more per frame (gray, IC-LK, warps).
- **The canvas saves few glimpses.** At the same γ and context 0.5, the oracle
  asks for 6 % fewer cells (B) to 20 % fewer (A) for PIDNet-S; for BiSeNetV2,
  3 % more (B) to 12 % fewer (A).

### Limits of this reading

- The oracle is per-frame, greedy and relative to the teacher: a heuristic
  ceiling, not a proof. The grid stops at 8 fixed glimpses and γ = 0.005, so a
  selector allowed more cells is not covered; it would also cost more.
- Train only, fp32 accuracy with fp16 T4 latency, as in Method 3's sweep. Val
  was not read.
- BiSeNetV2's curve is flat between scales 0.75 and 1 on these clips, so its
  −0.85 is a distance to a plateau.

## Files

- Code: `src/canvas/replay.py`, `scripts/method4.py`,
  `LatencyTables.predict_canvas_frame` in `src/foveation/latency_model.py`.
- Numbers: `verdict_train_all.json` (the rule as read by `score`) and
  `score_train_all_<model>.json.gz` (every configuration, both judges, both
  references), copied from `experiments/method4/train_all/`.
- Design: [spec](../../docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md),
  [plan](../../docs/superpowers/plans/2026-11-10-method4-canvas-bound.md).
