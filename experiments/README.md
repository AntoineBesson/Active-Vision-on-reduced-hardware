# experiments/

One subfolder per run. Everything in here is **gitignored except this README**
(see `.gitignore`), so run outputs never bloat the repo.

## Convention

Create one directory per experiment/run, named `YYYY-MM-DD_<short-slug>` (add a
`_<seed>` or `_<tag>` suffix when it helps), e.g.:

```
experiments/
  2026-09-04_baseline-t4/
    config.yaml          # exact resolved config used for the run
    metrics.json         # final / per-epoch metrics
    checkpoints/         # *.pth (gitignored)
    logs/                # stdout, tensorboard, etc.
    samples/             # qualitative outputs
```

Rules of thumb:
- The run folder should be **self-describing**: drop a copy of the resolved
  config in it so a run can be reproduced without guessing.
- Keep large artifacts (checkpoints, videos) here — they stay local.
- If a result matters, copy the summary number into your notes / the paper,
  not just the run folder.

## `data_check/`

`scripts/verify_data.py` writes its sanity-check grids to
`experiments/data_check/`. That folder is created on demand and its contents are
gitignored.

## Baseline runs

`python -m scripts.run_baseline --config configs/<baseline>.yaml` follows the
convention above: `experiments/<YYYY-MM-DD>_<run.name>/` holds the resolved
`config.yaml` and one JSON per input scale (`<run.name>_s<scale>.json`) with
the config, git commit, seed, dataset fingerprint, metrics and cost.

## `data_check/diagnose/`

`scripts/diagnose_real_data.py` writes `diagnose_report.json` and the
`overlay_*.png` images there.
