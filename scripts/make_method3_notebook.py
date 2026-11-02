"""Generate notebooks/method3_colab.ipynb (Method 3's stats, caches, sweeps and bounds on a Colab GPU).

    python scripts/make_method3_notebook.py notebooks/method3_colab.ipynb
"""

import json
import sys

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# Method 3 on a Colab GPU

Runs `python -m scripts.method3` from the `motion-reuse` branch, one stage at a time, and
copies each output to `MyDrive/avrh/method3/<split>/` as soon as it is written. Outputs
already on Drive are skipped, so a runtime that is cut off can simply be run again.

**Before the first run**, put in `MyDrive/avrh/` on Google Drive: `vps_train_all_chunk0.zip`
… `vps_train_all_chunk3.zip` (one per session is enough), `vps_val.zip` for `STAGE = "val"`,
and `PIDNet_S_Cityscapes_val.pt`. BiSeNetV2's checkpoint is downloaded from its release URL.

**Order** (one chunk per model per session):
1. `STAGE = "stats"`, `CHUNK = 0`: motion statistics, which fix θ. Commit
   `method3/train_all/motion_stats_train_all.json` to `results/method3/` before the sweeps.
2. `STAGE = "cache+sweep+bounds"` for `CHUNK` 0–3 × `MODEL` `pidnet_s`, `bisenetv2`. The
   caches stay on the runtime's local disk (`KEEP_CACHE = True` also copies them to Drive,
   several GB per chunk and model). If one session cannot finish a chunk, split the grid with
   `PART = "0/2"`, then `"1/2"` (needs `KEEP_CACHE = True` to skip rebuilding the caches).
3. `STAGE = "val"`, `SPLIT = "val"`, per model: the finalists only, from the committed
   `results/method3/finalists.json`.

Before the first sweep of a split, the notebook checks on real clips, on this GPU, that the
replay equals the streaming method pixel for pixel (`tests/test_reuse_replay.py`).
Accuracy runs in fp32; latency is not measured here (that is the latency-floor notebook).
"""),
    code("""
STAGE = "stats"        # "stats", "cache+sweep+bounds", "cache+sweep", "bounds", "val"
SPLIT = "train_all"    # "val" for STAGE = "val"
CHUNK = 0              # 0..3: vps_train_all_chunk{CHUNK}.zip (ignored on val)
MODEL = "pidnet_s"     # or "bisenetv2"
PART = "0/1"           # sweep: the i/M-th part of the grid
KEEP_CACHE = False     # True: also keep the clip caches on Drive
BRANCH = "motion-reuse"
DRIVE = "/content/drive/MyDrive/avrh"
"""),
    code("""
!nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu --format=csv
import torch
assert torch.cuda.is_available(), "Switch the runtime to a GPU (Runtime -> Change runtime type)."
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
"""),
    code("""
import os, shutil, time, zipfile
from google.colab import drive
drive.mount("/content/drive")
VPS = "/content/vps"
ZIP = "vps_val.zip" if SPLIT == "val" else f"vps_train_all_chunk{CHUNK}.zip"
marker = f"{VPS}/.{ZIP}_ok"
if not os.path.exists(marker):
    shutil.rmtree(VPS, ignore_errors=True)   # one chunk on disk at a time: the dataset is what is extracted
    t = time.time()
    shutil.copy(f"{DRIVE}/{ZIP}", "/content/vps.zip")   # local disk: much faster reads than Drive
    zipfile.ZipFile("/content/vps.zip").extractall(VPS)
    os.remove("/content/vps.zip")
    open(marker, "w").close()
    print(f"{ZIP} ready in {time.time() - t:.0f} s")
shutil.copy(f"{DRIVE}/PIDNet_S_Cityscapes_val.pt", "/content/PIDNet_S_Cityscapes_val.pt")
"""),
    code("""
!rm -rf /content/avrh && git clone -q --depth 1 -b $BRANCH https://github.com/AntoineBesson/Active-Vision-on-reduced-hardware.git /content/avrh
%cd /content/avrh
!git log --oneline -1
!pip install -q -e ".[reuse]"
"""),
    code("""
import glob, shlex, subprocess, sys
env = dict(os.environ, CITYSCAPES_VPS_ROOT=VPS, PIDNET_S_WEIGHTS="/content/PIDNet_S_Cityscapes_val.pt",
           M3_EQUIV_SPLIT=SPLIT)
OUT, ROOT, LOCAL = f"{DRIVE}/method3/{SPLIT}", "/content/m3cache", "/content/m3out"
for d in (OUT, ROOT, LOCAL):
    os.makedirs(d, exist_ok=True)
log = f"/content/method3_{SPLIT}.log"

def run(args):
    t = time.time()
    with open(log, "a") as f:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.method3", *shlex.split(args)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            f.write(line)
            print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"method3 {args.split()[0]} failed (exit {proc.returncode}); full log: {log}")
    print(f"== {args.split()[0]} done in {(time.time() - t) / 60:.1f} min ==")

def saved(name):
    return os.path.exists(f"{OUT}/{name}")

def keep(name):
    shutil.copy(f"{LOCAL}/{name}", OUT)
    print("saved to Drive:", f"{OUT}/{name}")
"""),
    code("""
steps = set(STAGE.split("+"))
TAG = "val" if SPLIT == "val" else f"chunk{CHUNK}"
PART_SUFFIX = "" if PART == "0/1" else "_p" + PART.replace("/", "of")
STATS = f"{DRIVE}/method3/train_all/motion_stats_train_all.json"
if "stats" in steps:
    name = f"motion_stats_{SPLIT}.json"
    if saved(name):
        print("already on Drive:", name)
    else:
        run(f"stats --split {SPLIT} --models {MODEL} --out {LOCAL}/{name}")
        keep(name)
else:
    cache_saved = f"{DRIVE}/method3/cache/{SPLIT}/{MODEL}"
    if KEEP_CACHE and os.path.isdir(cache_saved):   # caches of an earlier runtime: method3 cache skips them
        shutil.copytree(cache_saved, f"{ROOT}/{SPLIT}/{MODEL}", dirs_exist_ok=True)
    sweep_name = f"sweep_{SPLIT}_{MODEL}_{TAG}{PART_SUFFIX}.jsonl.gz"
    bounds_name = f"bounds_{SPLIT}_{MODEL}_{TAG}.jsonl.gz"
    todo_sweep = ("sweep" in steps or SPLIT == "val") and not saved(sweep_name)
    todo_bounds = "bounds" in steps and SPLIT != "val" and not saved(bounds_name)
    if SPLIT != "val":
        assert os.path.exists(STATS), f"run STAGE = 'stats' first ({STATS} is missing)"
    if todo_sweep or todo_bounds:
        if not saved(".equivalence_ok"):   # replay == streaming method on 2 real clips of this split, this GPU
            test = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_reuse_replay.py", "-k",
                                   "real_val_clips"], env=env, capture_output=True, text=True)
            print(test.stdout[-3000:])
            if " 1 passed" not in test.stdout or "skipped" in test.stdout:
                raise RuntimeError("the replay does not match the streaming method on real clips (or the test was skipped)")
            open(f"{OUT}/.equivalence_ok", "w").close()
        run(f"cache --split {SPLIT} --models {MODEL} --root {ROOT}")
        if KEEP_CACHE:
            shutil.copytree(f"{ROOT}/{SPLIT}/{MODEL}", cache_saved, dirs_exist_ok=True)
    if todo_sweep and SPLIT == "val":
        run(f"sweep --split val --models {MODEL} --root {ROOT} --finalists results/method3/finalists.json --tag {TAG} --out-dir {LOCAL}")
        keep(sweep_name)
    elif todo_sweep:
        run(f"sweep --split {SPLIT} --models {MODEL} --root {ROOT} --stats {STATS} --tag {TAG} --part {PART} --out-dir {LOCAL}")
        keep(sweep_name)
    if todo_bounds:
        run(f"bounds --split {SPLIT} --models {MODEL} --root {ROOT} --stats {STATS} --tag {TAG} --out-dir {LOCAL}")
        keep(bounds_name)
print(sorted(os.listdir(OUT)))
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1) + "\n")
