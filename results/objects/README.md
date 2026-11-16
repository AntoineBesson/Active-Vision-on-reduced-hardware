# Per-object recall by size and motion — protocol (no results yet)

Every thing instance (person, rider, car, truck, bus, train, motorcycle, bicycle) of every
labelled frame of VPS `train_all` is scored on its own: **found** when its local IoU with the
prediction is above 0.5. Recall is split by object size and by its motion once camera motion
is removed (centroid residual after the cached IC-LK homographies, px per frame). The
headline is recall on tiny + small objects that move (slow, fast or new), read against the
full-frame curve at equal mean latency, with a 95% clip bootstrap. Definitions:
`src/eval/objects.py`; design: the "Moving-object metric: design" doc (2026-11-12).

No backbone pass: predictions come from the Method 3 per-clip caches (full frame read
directly, Method 3 and Method 4 replayed exactly).

**Method 2 rule.** Method 2's caches hold per-cell confusions only, so it cannot be re-scored
per object. Method 4's variant N (`4o-N/...`, context + glimpses, oracle selector) is used as a
*screen* only: `report` keeps it in a separate `screen` block, and it never appears in the
final tables. If the screen shows Method 2 could matter for small moving objects, its best
finalists are rerun with a prediction dump and only those exact numbers are reported.

## Order of work

1. `table` once, then freeze the bin edges in `bins.json` from its quantiles (aim: at least
   ~300 objects per cell of the main grid; parked cars give the motion noise floor). Edges are
   not moved after step 2 starts.
2. `score` per model and chunk (GPU; resumable).
3. `report`.

```powershell
$py = "C:/tmp/avrh-venv/Scripts/python.exe"
& $py -m scripts.objects table --split train_all --models pidnet_s
& $py -m scripts.objects score --split train_all --models pidnet_s --chunk 0/4 --edge 12 `
    --score results/method3/score_train_all_pidnet_s.json.gz `
    --score experiments/method4/train_all/score_train_all_pidnet_s.json
& $py -m scripts.objects report --split train_all --models pidnet_s `
    --score results/method3/score_train_all_pidnet_s.json.gz `
    --score experiments/method4/train_all/score_train_all_pidnet_s.json
```

If the caches no longer hold the 0.75 full pass, pass `--ff-scales 1,0.5,0.375,0.25`
(0.5 and lower are read from the `ctx_*` arrays).

## Results, VPS `train_all` (2026-11-13)

Bins: the built-in defaults, frozen in `bins.json` before scoring (size edges 1024 / 4096 /
16384 px, motion edges 2 / 8 px per frame; smallest main cell 1,217 objects). Table:
43,791 objects in 394 clips (6 of the 400 clips hold no thing object); 37,288 measured,
6,503 new. Camera correction halves the median centroid shift (5.9 to 3.2 px per frame).

Configs: the full frame at 1, 0.75, 0.5, 0.375, 0.25, and the upper edge (`--edge 12`) of
each family. The caches hold `full_1` and `ctx_{0.5,0.375,0.25}` only (`ctx_s` is the same
full pass as `full_s`), so the Method 3 edge is taken among configs at `f1` / `f0.5`, and
`ff/0.75` is run live on the labelled frames (`fullpass`; on two clips it reproduces the
cached scale-1 and 0.5 IoUs exactly).

Headline: recall of tiny + small objects that move (slow, fast or new). Δ is recall minus the
full-frame curve at equal mean latency (T4). Brackets are the 95% clip bootstrap of Δ.

| model | full frame | Method 4 B, best Δ | Method 3, best Δ | Method 2 screen (4o-N), best Δ |
|---|---|---|---|---|
| pidnet_s | 0.626 at 10.7 ms (0.75), 0.698 at 19.4 ms (1) | −9.0 pts [−9.6, −8.3] (0.591 at 17.3 ms) | −23.4 pts [−24.7, −22.0] (0.424 at 14.6 ms) | −8.8 pts [−9.5, −8.1] at 17.3 ms |
| bisenetv2 | 0.647 at 17.8 ms (0.75), 0.705 at 31.2 ms (1) | −1.2 pts [−1.5, −0.8] (0.680 at 27.9 ms) | −15.4 pts [−16.4, −14.5] (0.510 at 21.7 ms) | −0.7 pts [−1.1, −0.4] at 27.0 ms |

Reading: on small moving objects, no method reaches the full-frame curve at equal latency, for
either model; every best Δ has a CI entirely below zero. The 0.75 point sits well above the
straight line between 0.5 and 1, so it turns the earlier +1.1 pts of Method 4 B on bisenetv2
(read against that line) into −2.7 pts. Method 4 B comes closest on bisenetv2 (about −1 pt
at 28 ms) and is 9 pts short on pidnet_s; Method 3 is 15 to 23 pts short. The Method 2 screen
tracks Method 4 B and is not a Method 2 result.

Per-config tables: `report_train_all_<model>.json`. Scoring logs: `logs/`. Run notes: the
laptop RTX 2060 was held under ~83 C with `--max-temp 82` (checked before each config).
