# Active Vision on Reduced Hardware

A research side-project on **lightweight foveated video segmentation**: a small
non-ViT backbone that maintains a persistent memory across "glimpses" (inspired
by CanViT), targeting a single T4-class GPU (16 GB) or a Colab/Kaggle free-tier
runtime. This repository currently contains the **infrastructure and data
plumbing only** — model, glimpse-policy and training code land in later work.

## Status

| Piece | State |
| --- | --- |
| Repo scaffolding (`src/`, `scripts/`, `configs/`, `tests/`) | ✅ |
| Cityscapes-VPS clip loader (`src/data/cityscapes_vps.py`) | ✅ |
| Synthetic test fixture + unit tests | ✅ |
| Data verification / visualization script | ✅ |
| Model backbone / glimpse memory | ⬜ later |
| Glimpse policy (uncertainty / motion) | ⬜ later |
| Training loop | ⬜ later |

## Layout

```
src/                 # library code (importable package root)
  data/              # datasets & loaders  (cityscapes_vps.py lives here)
  models/            # backbones + glimpse memory (empty for now)
  policies/          # foveation / glimpse policies (empty for now)
  eval/              # metrics & reporting (empty for now)
scripts/             # CLI entry points (verify_data.py)
configs/             # experiment/data configs (YAML -> dataclasses)
experiments/         # one folder per run; gitignored except its README
tests/               # unit tests + a tiny synthetic dataset fixture
```

## Setup

Requires Python ≥ 3.9. The base install is deliberately tiny (numpy + pillow +
pyyaml) so it runs anywhere; the deep-learning stack is an opt-in extra.

```bash
python -m venv .venv && source .venv/bin/activate

# Base loader + dev/test tooling (numpy, pillow, pyyaml, pytest, matplotlib):
pip install -e ".[dev]"

# Later, when you add model/training code, also install torch. On Colab/Kaggle
# the CUDA build is usually preinstalled; otherwise:
pip install -e ".[torch]"
```

> **Why `pyproject.toml`?** It makes `src/` an installable package (clean
> imports, no `sys.path` hacks), keeps pytest config in one place, and uses
> optional "extras" so the base footprint stays small for constrained hardware.
> `pip install -e ".[dev]"` is enough to run the test suite.

## Running the tests

The suite runs entirely against a **synthetic fixture** — no real dataset
required:

```bash
pytest              # or: pytest tests/ -v
```

The fixture is a handful of tiny, fully-deterministic PNGs generated on demand
by `tests/fixtures/generate_fixture.py` (auto-invoked from `tests/conftest.py`),
so it is not committed as binaries — a fresh clone regenerates it on first run.
The one torch-dependent test is auto-skipped if torch isn't installed.

## Getting the real Cityscapes-VPS data

This repo contains **no download logic** — Cityscapes-VPS is license-gated, so
you must obtain it yourself (register at cityscapes-dataset.com / the VPSNet
release) and place it locally.

Point the loader at your copy with an environment variable:

```bash
export CITYSCAPES_VPS_ROOT=/path/to/cityscapes_vps
```

...or set `root:` in a config (see `configs/cityscapes_vps.yaml`).

### ⚠️ Assumed directory structure (please verify against your download)

I did **not** have the real files when writing this, so the expected layout is a
best-effort reconstruction of the VPSNet release and is stated as an
**assumption**. The full, authoritative description lives in the header comment
of [`src/data/cityscapes_vps.py`](src/data/cityscapes_vps.py). In short:

```
$CITYSCAPES_VPS_ROOT/
  train/ | val/ | test/
    img_all/          # ALL frames of every clip (dense; most are unlabeled)
    panoptic_video/   # panoptic PNG labels, ONLY for the labeled frames
    labelmap/         # (optional) single-channel semantic trainId PNGs
```

Assumed frame naming (Cityscapes-VPS style):

```
0005_0025_frankfurt_000000_001736_leftImg8bit.png
└clip┘└frm┘ └── original Cityscapes id ──┘└ suffix ┘
```

Key assumptions:
- Clips are **grouped by the leading clip id** and **ordered by the frame
  index**; each clip is a 30-frame snippet.
- Annotations are **sparse**: ~every 5th frame is labeled (6 labeled frames per
  clip). A frame is treated as *labeled* iff a matching annotation file exists —
  the loader does **not** hard-code the every-5th rule (it's exposed only as an
  `is_keyframe` hint).
- Panoptic PNGs use the COCO convention `id = R + 256·G + 256²·B`.

**If your download differs, you don't need to edit the loader** — every path,
subdir name, filename regex and suffix is a field on `CityscapesVPSConfig`.
Retarget them via `configs/cityscapes_vps.yaml` (or kwargs) and, ideally, correct
the header comment so it matches reality.

## Visual sanity check

Once the real data is in place, render a grid of a clip's frames with their
annotations overlaid (saved under `experiments/data_check/`):

```bash
python scripts/verify_data.py --split val --clip-index 0
# or, without real data, smoke-test the pipeline on the fixture:
python scripts/verify_data.py --fixture
```

## License

MIT (see `pyproject.toml`). The Cityscapes / Cityscapes-VPS data has its own
license and is **not** included in this repo.
