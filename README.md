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

Point the loader at your copy (the folder that holds `train/ val/ test/` and
the `*_city_vps.json` files) with an environment variable:

```bash
export CITYSCAPES_VPS_ROOT=/path/to/cityscapes_vps
```

...or set `root:` in a config (see `configs/cityscapes_vps.yaml`).

### Cityscapes-VPS format: what is verified, what is assumed

The layout and label encodings were checked against the **official VPSNet
release** ([mcahny/vps](https://github.com/mcahny/vps): `docs/DATASET.md`,
`prepare_data/*.py`, `tools/eval_vpq.py`, `tools/dataset/cityscapes_vps.py`).
No real files were available yet, so nothing here has been checked against
real data; `scripts/diagnose_real_data.py` will do that. Full details, with
file-by-file sources, are in the header of
[`src/data/cityscapes_vps.py`](src/data/cityscapes_vps.py);
[`configs/cityscapes_vps.yaml`](configs/cityscapes_vps.yaml) encodes them.

**Verified from the official code:**

- The VPS zip ships annotations + JSON only. Frames are copied from Cityscapes
  `leftImg8bit_sequence/val` (all 500 VPS videos are Cityscapes *val*
  sequences), 400/100/100 videos for train/val/test, 30 frames each, labelled at
  clip positions 4, 9, 14, 19, 24, 29.
- `val/` and `test/` have `img_all/` (every frame) whose files keep their
  **original Cityscapes names** (`frankfurt_000000_000294_leftImg8bit.png`):
  there is **no clip prefix**, so clips are rebuilt from the `city_seq` part
  plus consecutive frame numbers. `train/` has no `img_all/`, only `img/`
  (the 6 labelled frames per clip, names `{vid}_{fid}_{city}_{seq}_{frame}_{newImg8bit|leftImg8bit}.png`).
- Annotations are named `{vid}_{fid}_{city}_{seq}_{frame}_{final_mask|gtFine_color}.png`
  and come in three encodings, all in the 19-class train-id space (VPS category
  ids *are* Cityscapes train ids):
  - `labelmap/` (train, val): train id per pixel, 255 = void;
  - `panoptic_inst/` (train, val): stuff = train id, thing = `train_id*1000 + obj`, 255 = void;
  - `val/panoptic_video/`: COCO-panoptic RGB (`id = R + 256 G + 256² B`) whose ids
    are random colours. The class of a segment is **only** in
    `panoptic_gt_val_city_vps.json`; it cannot be recovered from the colour.
- Crowd regions keep their class in `labelmap/` but are void (255) in the two
  panoptic encodings.

**What the config does with this:** val/test frames come from `img_all/`,
val labels from `panoptic_video/` + JSON, train frames and labels from
`img/` + `panoptic_inst/` (both panoptic encodings treat crowd the same way).
Everything maps to the 19 train classes with ignore index 255, and the
mapping lives entirely in config (`label_mapping`), so another dataset can
supply its own.

**Still assumed (each is checked by `scripts/diagnose_real_data.py`):**

1. Two VPS videos never use consecutive frame numbers of the same Cityscapes
   sequence, so a run of consecutive `img_all` frames is one clip. If a run
   holds annotations of several videos, it is split with a warning.
2. The second 4-digit field (`fid`) is undocumented; it is ignored, and
   frames are ordered by the Cityscapes frame number.
3. Annotation PNG bit depth (8/16/32-bit) is read as stored.

The earlier guess (clip-prefixed names inside `img_all/`) finds **zero clips**
on the official layout. It survives only as the `CityscapesVPSConfig`
defaults, because the original synthetic fixture and its tests use it.

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
