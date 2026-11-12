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
