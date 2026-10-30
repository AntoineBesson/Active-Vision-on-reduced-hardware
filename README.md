# Active Vision on Reduced Hardware

A research side-project on **lightweight foveated video segmentation** for a
single T4-class GPU (16 GB), using public data only.

We do **not** build a new segmentation model. We add a layer on top of an
existing one: preprocessing (a low-resolution context view plus
high-resolution crops), an optional persistent scene memory (a canvas), and
a gaze policy that chooses which regions to inspect. Compared methods:

1. full frame (the backbone on the whole frame, at full or reduced resolution);
2. foveation without memory (uncertainty-driven policy);
3. foveation with a persistent canvas and a policy that can read it;
4. optionally, an RL-trained policy.

Nothing outside one dataset adapter is Cityscapes-specific: preprocessing,
policies and evaluation are written against a dataset-agnostic interface so
they can be retrained or reused on other video segmentation datasets.

## Where we are (2026-10-30)

**Method 1 (full frame) is done and validated on real data.** The data
layer, backbones, evaluation harness and baseline CLI run end to end on
Cityscapes-VPS val. The pipeline reproduces the published Cityscapes val
numbers of both backbones (PIDNet-S 78.76 vs 78.8, BiSeNetV2 74.95 vs 74.95),
so the full-frame curves below are the reference the foveated methods must
beat at equal cost. See [Real-data results](#real-data-results).

Reference points on VPS val (300 labelled frames, mIoU):

| Pixel fraction | 1.0 | 0.56 | 0.25 | 0.14 | 0.06 |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S | 71.5 | 70.1 | 64.8 | 58.6 | 50.1 |
| BiSeNetV2 | 70.4 | 69.9 | 67.6 | 64.3 | 59.1 |

PIDNet-S is better and cheaper at full resolution, but BiSeNetV2 degrades
much more gracefully at low resolution: the context view of a foveated
method will run at those low scales, so this matters for the choice below.

### Next steps

1. **Method 2: foveation without memory.** Implement a `FrameMethod` that
   runs the backbone on a low-resolution context view (e.g. scale 0.25),
   then on K high-resolution crops chosen by an uncertainty-driven
   `GazePolicy` (e.g. top-K entropy boxes), pastes the crop predictions back
   into the full-frame map, and reports `context`, `glimpses`, `policy` and
   `recomposition` to the cost ledger. Success = above the full-frame
   mIoU-vs-pixel-fraction curve above. Decide first which backbone serves as
   context (see the low-scale numbers).
2. **Method 3: persistent canvas.** Implement `SceneMemory` (canvas of
   logits/confidence with age) and a canvas-aware policy; this is where the
   unlabelled in-between frames of each 30-frame clip are used.
3. **Latency on a T4** (Colab): the latency floor is measured (3 sessions,
   CUDA graphs, FP16, batch 1), see
   [results/latency_floor](results/latency_floor/README.md). Full frame:
   PIDNet-S 19.4 ms, BiSeNetV2 31.2 ms; the best foveated configuration
   tested is 1.6–1.7× (PIDNet-S) and 2.1–2.2× (BiSeNetV2) faster, before any
   accuracy is measured. The laptop RTX 2060 numbers below are only
   indicative.
4. Optional: Method 4 (RL policy); a secondary evaluation on the 500 official
   val images (frame `_000019`) for less noisy rare-class IoUs.

## Quickstart

```bash
# 1. Environment (Windows: .venv\Scripts\activate)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch]"   # CUDA wheels: add --index-url https://download.pytorch.org/whl/cu126
pytest                          # 131 tests, synthetic data, seconds

# 2. Data (once, ~45 GB on disk, details in "Getting the real Cityscapes-VPS data")
pip install remotezip requests
python scripts/get_val.py --out /data/cityscapes            # Cityscapes login, ~3.5 h
#    + VPS zip from Dropbox, official VPSNet prep scripts (2 patches on Windows)
export CITYSCAPES_VPS_ROOT=/data/cityscapes_vps

# 3. Check the data, then run the full-frame baselines
python scripts/diagnose_real_data.py --config configs/diagnose_cityscapes_vps.yaml
export PIDNET_S_WEIGHTS=/path/to/PIDNet_S_Cityscapes_val.pt     # manual Google Drive download
python -m scripts.run_baseline --config configs/baseline_fullframe_pidnet_s.yaml
python -m scripts.run_baseline --config configs/baseline_fullframe_bisenetv2.yaml
# -> experiments/<date>_<run>/<run>_s<scale>.json
```

## Status

| Piece | State |
| --- | --- |
| Dataset-agnostic interface `VideoSegDataset` (`src/data/base.py`) | ✅ |
| Cityscapes-VPS adapter + 19-class mapping, layout verified against the official scripts | ✅ confirmed on real data |
| Torch sub-clip `Dataset`/`DataLoader`, clip-level splits | ✅ |
| Interfaces: `SegmentationBackbone`, `GazePolicy`, `SceneMemory` | ✅ contracts only |
| Backbones: PIDNet-S (+M/L), BiSeNetV2, dummy | ✅ ports bit-exact vs official code |
| Evaluation harness: mIoU, FLOPs, GPU latency, peak memory, run JSON | ✅ incl. GPU path (RTX 2060) |
| Full-frame baseline CLI, 5 input scales | ✅ fixture and real data |
| Real-data diagnostics script | ✅ run on real data, no problem |
| Method 2: foveation without memory | ⬜ next |
| Method 3: foveation + persistent canvas | ⬜ |
| Method 4: RL-trained policy | ⬜ optional |
| Full-frame results on real data | ✅ see [Real-data results](#real-data-results) |

## Layout

```
src/                       library code (importable package root)
  data/
    base.py                VideoSegDataset interface, VideoClip, InMemoryVideoSeg
    label_mapping.py       config-driven raw annotation -> train-id mapping
    cityscapes_vps.py      Cityscapes-VPS adapter (the only Cityscapes-specific module)
    registry.py            build_dataset(spec)
    subclips.py            sub-clip windows (length T, strides)
    splits.py              clip-level (never frame-level) splits
    torch_dataset.py       torch Dataset / DataLoader (needs torch)
  models/
    interfaces.py          SegmentationBackbone, SceneMemory, MemoryState
    pidnet.py              PIDNet-S/M/L (official architecture, ported, MIT)
    bisenetv2.py           BiSeNetV2 (CoinCheung implementation, ported, MIT)
    dummy.py               tiny deterministic backbone for tests and smoke runs
    weights.py             weights from a path or URL, explicit failures
    registry.py            build_backbone(spec)
    utils.py               padding, logits upsampling
  policies/
    interfaces.py          GazePolicy, GazeState, Box
  eval/
    metrics.py             confusion-matrix mIoU / per-class IoU (numpy)
    cost.py                cost ledger, FLOPs, latency, peak memory
    results.py             one JSON file per run
    harness.py             evaluation loop + full-frame method (torch)
scripts/
  run_baseline.py          full-frame baseline:  python -m scripts.run_baseline
  get_val.py               downloads only the Cityscapes val folders needed
  diagnose_real_data.py    checks a local copy of the data before trusting it
  verify_data.py           quick visual grid of one clip
  common.py                YAML / --set / seeding helpers for the CLIs
configs/
  cityscapes_vps.yaml                 data config (official layout)
  baseline_fixture.yaml               baseline smoke test (synthetic fixture + dummy)
  baseline_fullframe_pidnet_s.yaml    baseline on real data, PIDNet-S
  baseline_fullframe_bisenetv2.yaml   baseline on real data, BiSeNetV2
  diagnose_cityscapes_vps.yaml        diagnostics config
experiments/               one folder per run (gitignored except its README)
tests/                     unit tests; fixtures/generate_fixture.py builds two
                           synthetic datasets on demand (nothing binary is committed)
```

## Setup

Python ≥ 3.9. The base install (numpy + pillow + pyyaml) covers the data
layer, the label mapping, splits, metrics, result files and the diagnostics
script; torch is an optional extra for models, the torch data pipeline and
the evaluation loop.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # base + pytest + matplotlib
pip install -e ".[torch]"   # torch + torchvision (Colab/Kaggle usually preinstall them)
# CPU-only torch, if you do not want the CUDA wheels:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Running the tests

Everything runs on synthetic data, on CPU, in seconds:

```bash
pytest            # or: pytest -v
```

`tests/conftest.py` generates two tiny deterministic datasets on first use:
`tests/fixtures/cityscapes_vps/` (the original fixture, an older layout
guess) and `tests/fixtures/cityscapes_vps_official/` (mirrors the verified
official layout and all three label encodings). Without torch, the
torch-dependent tests are skipped; without CUDA, the GPU-only tests (latency,
peak memory) are skipped. A subprocess test checks that every numpy-only
module imports with torch blocked.

## Full-frame baseline

```bash
# Smoke test: synthetic fixture + dummy backbone, CPU, no data or download.
python -m scripts.run_baseline --config configs/baseline_fixture.yaml

# Real data (see "Running on the real data"):
python -m scripts.run_baseline --config configs/baseline_fullframe_pidnet_s.yaml \
    --set backbone.weights=/path/to/PIDNet_S_Cityscapes_val.pt
```

Options: `--set key=value` (any config value, YAML-parsed, e.g.
`--set eval.scales=[1.0,0.5]`), `--device auto|cpu|cuda`, `--max-clips N`,
`--fixture`, `--out DIR`.

Each input scale is one run and writes one JSON file to
`experiments/<date>_<run.name>/<run.name>_s<scale>.json`, next to the
resolved `config.yaml`. A run record contains the resolved config, the git
commit and whether the work tree was dirty, the seed, the dataset
description and fingerprint (a hash of the clip metadata), the backbone
description (including the weights' sha256), the metrics, the cost and the
software/hardware environment. Reduced scales (0.75, 0.5, 0.375, 0.25 of
1024×2048) are the equal-cost full-frame baselines for the foveated methods.

**Metrics.** mIoU and per-class IoU are computed only on labelled frames,
from one confusion matrix accumulated over all clips (dataset-level IoU, as
in the Cityscapes benchmark). Classes absent from both ground truth and
predictions are excluded from the mean. Pixels a method leaves unpredicted
count as errors, never as ignore.

**Cost.** One ledger sums named components per frame. The baseline only has
`backbone`; foveated methods will add context, glimpses, policy, memory and
recomposition to the same total.
- *FLOPs* use `torch.utils.flop_counter.FlopCounterMode`. It ships with torch,
  so it adds no dependency, and it counts at the dispatcher level, so it
  handles the data-dependent control flow of gaze loops, which tracing
  (fvcore) can miss and module hooks (ptflops) skip for functional ops. It
  counts 2 FLOPs per multiply-accumulate of convolutions and matrix products.
  Elementwise ops, normalisation, pooling, interpolation and biases are not
  counted. MACs = FLOPs / 2 are also reported.
- *Latency*: CUDA, batch 1, fp16 weights and inputs, cuDNN autotuning, 20
  warm-up calls, then 100 calls each timed between `torch.cuda.synchronize()`
  barriers. Median and p95 are reported for the backbone alone and end to
  end (full-resolution frame → resize → network → upsampling → argmax).
  Without CUDA the record says `"status": "skipped"`: no CPU number
  masquerades as GPU latency.
- *Peak memory*: `torch.cuda.max_memory_allocated` during the end-to-end
  runs. Reported as the absolute peak, the increment over the memory
  already resident, and `estimated_footprint_mib` = the measured model's
  weights + that increment. The estimate is not inflated by the fp32 copy of
  the model kept on the GPU for the accuracy pass.
- *Pixel fraction*: pixels fed to the backbone per frame over the
  full-resolution pixel count (`scale²` for the baseline).

The accuracy pass runs in fp32 (`eval.precision`), as PIDNet's own
evaluation does.

## Backbone weights

Weights come from a local path or a URL given in the config
(`backbone.weights`, optional `sha256` pin). URLs are downloaded once into
the torch hub cache. A missing file, an unreachable URL (no network) or a
Google Drive share page fails with a message saying what to do instead.
Real backbones refuse to run without weights unless `allow_random_init:
true` is set, for smoke tests only.

- **PIDNet-S**: download `PIDNet_S_Cityscapes_val.pt` from the links in the
  [official README](https://github.com/XuJiacong/PIDNet#models) (Google
  Drive, manual download) and set `PIDNET_S_WEIGHTS` or
  `--set backbone.weights=...`. Use the **`_val`** checkpoint, which was
  trained on Cityscapes train only. The `_test` one was trained on
  train+val, and **every Cityscapes-VPS clip is a Cityscapes val sequence**,
  so it would leak. The released file (sha256 `b51aa935…7c01`) loads
  strictly; its training-loss buffers (`sem_loss.*`, `sb_loss.*`) are dropped.
  The individual Drive links in that README may be dead; the author's
  backup folder linked at its top has the same files.
- **BiSeNetV2**: `configs/baseline_fullframe_bisenetv2.yaml` points at the
  public release asset `model_final_v2_city.pth` of
  [CoinCheung/BiSeNet](https://github.com/CoinCheung/BiSeNet), pinned by sha256.
  That file was checked to load strictly into the port.
- **Dummy**: no weights. Seeded random, or nearest-colour (`class_colors:
  dataset_palette`), which makes it an alignment probe on the fixture.

Both ports keep the official module names, so official checkpoints load
unchanged. Against the official code, with random weights and randomised BN
statistics, they give identical state-dict keys and parameter counts
(PIDNet-S: 7,625,715) and bit-identical outputs.

## Getting the real Cityscapes-VPS data

Cityscapes-VPS (Kim et al., CVPR 2020) is built from Cityscapes, which is
licence-gated (non-commercial use; registration and approval required). The
VPS annotations are CC BY-NC-SA 4.0 on top of the Cityscapes licence. The
steps below follow the official
[VPSNet DATASET.md](https://github.com/mcahny/vps/blob/master/docs/DATASET.md)
and were run end to end on Windows with Python 3.14 and Pillow 13 (see the
two script patches in step 4).

1. **Cityscapes account.** Register at <https://www.cityscapes-dataset.com/register/>
   and wait for approval.
2. **Cityscapes packages** (log in, <https://www.cityscapes-dataset.com/downloads/>):
   - the `gtFine` package (`gtFine_trainvaltest.zip`);
   - the `leftImg8bit_sequence` package (packageID 1 and 14; the latter is
     about 324 GB). Only the `val/` part (500 sequences × 30 frames) is needed.

   `scripts/get_val.py` downloads just those two `val/` folders (2,000 +
   15,000 files, ~34 GB) without fetching the whole archives: it reads the
   zips remotely with HTTP range requests. It is resumable and asks for the
   Cityscapes login (or reads `CITYSCAPES_USERNAME` / `CITYSCAPES_PASSWORD`).
   At ~2.7 MB/s the frames took about 3.5 h.

   ```bash
   pip install remotezip requests
   python scripts/get_val.py --out /path/to/cityscapes   # -> gtFine/val, leftImg8bit_sequence/val
   ```
3. **VPS annotations.** Download `cityscapes-vps-dataset-1.0.zip` from the
   link in the [VPSNet README](https://github.com/mcahny/vps) (Dropbox) and
   unzip it so you have `data/cityscapes_vps/` with the `*_city_vps.json`
   files and the `train/ val/ test/` folders.
4. **Merge and build the labels** with the official scripts, run from a
   clone of the VPSNet repository:

   ```bash
   git clone https://github.com/mcahny/vps && cd vps
   pip install numpy pillow tqdm "git+https://github.com/cocodataset/panopticapi.git"
   # data/ holds cityscapes_vps/, leftImg8bit_sequence/ and gtFine/
   bash ./prepare_data/merge_datasets.sh data/cityscapes_vps data/leftImg8bit_sequence/val/ data/gtFine/val/
   bash ./prepare_data/create_panoptic_labels.sh data/cityscapes_vps/
   ```

   On Windows, or with Pillow ≥ 10, two patches to the official scripts are
   needed. Without them the label step finishes with no output (the worker
   pool hides the errors), and on Windows it overwrites `cls/` in place:
   - `create_panoptic_labels.py` and `create_panoptic_video_labels.py`:
     replace `x.split('/')[-1]` with `os.path.basename(x)` (with `\` paths,
     the "file name" is the full input path, so outputs overwrite the inputs);
   - `create_panoptic_labels.py`: save `pan_map.astype(np.uint16)`. Pillow ≥ 10
     cannot write 32-bit PNGs; old Pillow wrote them as 16-bit, and ids
     (≤ 18999) fit.

   The `.sh` wrappers just call the Python scripts per split, so on Windows
   run those calls directly. Expected counts: `train/labelmap`,
   `train/panoptic_inst` 2400; `val/labelmap`, `val/panoptic_inst`,
   `val/panoptic_video` 300.

   Keep `val/labelmap/` and `val/panoptic_inst/`. DATASET.md says they can be
   deleted, but the diagnostics use them to cross-check the label mapping.
5. **Point this repo at it.** Use the folder that contains `train/ val/
   test/` and the `*_city_vps.json` files:

   ```bash
   export CITYSCAPES_VPS_ROOT=/absolute/path/to/vps/data/cityscapes_vps
   ```

   Or set `root:` in `configs/cityscapes_vps.yaml`, or pass `--root` to the
   diagnostics.

### Cityscapes-VPS format: what is verified, what is assumed

The layout and label encodings were checked against the **official VPSNet
release** ([mcahny/vps](https://github.com/mcahny/vps): `docs/DATASET.md`,
`prepare_data/*.py`, `tools/eval_vpq.py`, `tools/dataset/cityscapes_vps.py`).
All of it was then confirmed on the real data by
`scripts/diagnose_real_data.py` (2026-10-30, no problem reported). Full details, with
file-by-file sources, are in the header of
[`src/data/cityscapes_vps.py`](src/data/cityscapes_vps.py);
[`configs/cityscapes_vps.yaml`](configs/cityscapes_vps.yaml) encodes them.

**Verified from the official code:**

- The VPS zip ships annotations + JSON only. Frames are copied from Cityscapes
  `leftImg8bit_sequence/val` (all 500 VPS videos are Cityscapes *val*
  sequences), 400/50/50 videos for train/val/test, 30 frames each, labelled at
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
  panoptic encodings. Also, each panoptic instance takes the majority semantic
  class of its mask, so a few thing pixels differ from `labelmap` (e.g. a
  rider's pixels inside a bicycle instance). On real data `labelmap` agrees with
  the configured labels on 99.96% (val) / 99.83% (train) of pixels.

**What the config does with this:** val/test frames come from `img_all/`,
val labels from `panoptic_video/` + JSON, train frames and labels from
`img/` + `panoptic_inst/` (both panoptic encodings treat crowd the same way).
Everything maps to the 19 train classes with ignore index 255, and the
mapping lives entirely in config (`label_mapping`), so another dataset can
supply its own.

**Assumptions, all confirmed on the real data:**

1. Two VPS videos never use consecutive frame numbers of the same Cityscapes
   sequence, so a run of consecutive `img_all` frames is one clip (no split
   warning: val and test give exactly 50 clips of 30 frames).
2. The second 4-digit field (`fid`) is undocumented; it is ignored, and
   frames are ordered by the Cityscapes frame number.
3. Annotation PNG bit depth is read as stored (`panoptic_inst` is 16-bit with
   the patched script, `panoptic_video` RGB).

The earlier guess (clip-prefixed names inside `img_all/`) finds **zero clips**
on the official layout. It survives only as the `CityscapesVPSConfig`
defaults, because the original synthetic fixture and its tests use it.

## Running on the real data

All of this was run on 2026-10-30; results are in [Real-data results](#real-data-results).

1. **Diagnose the copy**, then read the summary and look at the overlays:

   ```bash
   python scripts/diagnose_real_data.py --config configs/diagnose_cityscapes_vps.yaml
   # -> experiments/data_check/diagnose/{diagnose_report.json, overlay_*.png}
   ```

   What the official layout should give: val 50 clips × 30 frames with 6
   labelled frames each at positions 4, 9, 14, 19, 24, 29; train 400 clips
   of 6 frames (only `img/` exists); test 50 clips × 30 frames, none
   labelled; every filename matching its regex; no orphan annotation;
   1024×2048 frames; `panoptic_inst` agreeing 100% with `panoptic_video` +
   JSON; `labelmap` differing only on crowd regions and instance
   majority-class pixels (~0.05–0.2%). Any other outcome means
   the config (or an assumption) needs fixing before running anything else.
2. **Eyeball a clip:** `python scripts/verify_data.py --split val --clip-index 0`.
3. **Full-frame baselines.** Five scales each, one JSON per scale:

   ```bash
   export PIDNET_S_WEIGHTS=/path/to/PIDNet_S_Cityscapes_val.pt
   python -m scripts.run_baseline --config configs/baseline_fullframe_pidnet_s.yaml
   python -m scripts.run_baseline --config configs/baseline_fullframe_bisenetv2.yaml
   ```

   As a plausibility check only, compare the scale-1.0 mIoU with the
   official PIDNet-S Cityscapes val figure (78.8 mIoU). That figure is over
   a different frame set (the 500 gtFine images, without VPS crowd
   handling), so equality is not expected, but a large gap points to a
   preprocessing or mapping problem: check the overlays and cross-checks first.

## Real-data results

Cityscapes-VPS val: 50 clips, the 300 labelled frames, fp32 accuracy pass.
Latency is the median end-to-end time on a laptop **RTX 2060 Max-Q (6 GB)**,
batch 1, fp16; a T4 will give different numbers.

| Backbone | Scale | Input | mIoU | GFLOPs/frame | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| PIDNet-S | 1.0 | 1024×2048 | 71.5 | 94.4 | 26.5 |
| PIDNet-S | 0.75 | 768×1536 | 70.1 | 53.1 | 26.4 |
| PIDNet-S | 0.5 | 512×1024 | 64.8 | 23.6 | 25.1 |
| PIDNet-S | 0.375 | 384×768 | 58.6 | 13.3 | 21.5 |
| PIDNet-S | 0.25 | 256×512 | 50.1 | 5.9 | 20.5 |
| BiSeNetV2 | 1.0 | 1024×2048 | 70.4 | 196.8 | 37.3 |
| BiSeNetV2 | 0.75 | 768×1536 | 69.9 | 110.7 | 26.1 |
| BiSeNetV2 | 0.5 | 512×1024 | 67.6 | 49.2 | 20.0 |
| BiSeNetV2 | 0.375 | 384×768 | 64.3 | 27.7 | 16.6 |
| BiSeNetV2 | 0.25 | 256×512 | 59.1 | 12.3 | 14.6 |

**Pipeline check against the official numbers.** On VPS val, PIDNet-S is
7 points below its official 78.8, almost all of it on rare classes (train
26.9, motorcycle 31.4, wall 48.0 IoU), which a 50-clip set decides with a
handful of objects. To rule out a pipeline bug, the same code (frame loading,
backbone wrapper, upsampling, confusion matrix) was run on the **official 500
Cityscapes val images** (frame `_000019` of each sequence, gtFine `labelIds`
→ train ids): PIDNet-S gives **78.76** (official 78.8) and BiSeNetV2
**74.95** (CoinCheung single-scale: 74.95). The gap on VPS val is therefore
the frame set, not the pipeline.

Note: `BiSeNetV2Backbone` defaults to ImageNet mean/std; the CoinCheung
checkpoint needs its Cityscapes mean/std (set in
`configs/baseline_fullframe_bisenetv2.yaml`). With ImageNet stats it drops
to 69.1 on official val.

## Using another dataset

1. Write an adapter: subclass `data.base.VideoSegDataset` (`class_names`,
   `ignore_index`, `__len__`, `clip_id`, `labeled_mask`, `load_frame`,
   `load_label`; optionally `frame_indices`, `clip_group`, `palette` and the
   diagnostic hooks) and add a `from_spec(spec, base_dir)` classmethod.
2. Describe its annotation encoding with a `label_mapping` block
   (`data.label_mapping`: rgb/gray ids, identity / divisor / COCO-panoptic
   JSON rule, optional category→train-id table). No code needed for common
   formats.
3. Register a short name in `data/registry.py` (or use `type:
   package.module:Class`) and point a config's `dataset:` section at it.
   The torch pipeline, splits, backbones, harness, baseline CLI and
   diagnostics work unchanged.

## Interfaces for the next methods

- `SegmentationBackbone.forward(images (B,3,H,W) in [0,1]) -> logits
  (B,C,ceil(H/s),ceil(W/s))`. The backbone normalises and pads internally.
  `models.utils.upsample_logits` maps logits back to any frame size.
- `GazePolicy.select(state: GazeState, memory: MemoryState | None) -> list[Box]`,
  with a `uses_memory` flag, so memory-free and canvas-aware policies share
  one contract. `Box` uses coordinates normalised to the full frame.
- `SceneMemory.reset(frame_size)`, `update(glimpse_output (C,h,w), box, *, t)`,
  `read() -> MemoryState` (canvas `(D,Hc,Wc)`, optional confidence and age).
- `eval.harness.FrameMethod`: what the evaluation loop runs. A foveated
  method implements `predict(frame, t, ledger)` and reports each component's
  cost to the ledger.

## License

MIT (see `pyproject.toml`). `src/models/pidnet.py` and
`src/models/bisenetv2.py` are ports of MIT-licensed code; their copyright
and permission notices are reproduced in the file headers. The Cityscapes
and Cityscapes-VPS data have their own licences and are **not** included.
