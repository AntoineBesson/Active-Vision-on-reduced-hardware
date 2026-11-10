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
