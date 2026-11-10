# Method 4, phase 1: feasibility bound of a persistent canvas — protocol and status

**Status (2026-11-10): chunks 0 and 1 run (200 clips of 400); the rule is not read yet.** The grid, the
latency parts and the pass rule were fixed in the
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

Pending: the rule is read on the 400 clips only.

Chunks 0 and 1 (200 clips, partial): no configuration is above the full-frame
curve. The best canvas configuration is 4.65 points below it for PIDNet-S and
0.94 point below for BiSeNetV2 (variant B, mean judge). Details in
[TRIES.md](TRIES.md).

## Files

- Code: `src/canvas/replay.py`, `scripts/method4.py`,
  `LatencyTables.predict_canvas_frame` in `src/foveation/latency_model.py`.
- Design: [spec](../../docs/superpowers/specs/2026-11-10-method4-canvas-bound-design.md),
  [plan](../../docs/superpowers/plans/2026-11-10-method4-canvas-bound.md).
