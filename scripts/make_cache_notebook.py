"""Generate notebooks/method2_cache_colab.ipynb (method 2's per-cell caches built on a Colab GPU).

    python scripts/make_cache_notebook.py notebooks/method2_cache_colab.ipynb
"""

import json
import sys

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# Method 2 caches on a Colab GPU

Runs `python -m scripts.method2 cache` from the `foveation-no-memory` branch, one glimpse
geometry at a time, and copies each finished geometry's caches to Google Drive at once, so a
runtime that is cut off loses at most the geometry in progress: run the notebook again and
the geometries already on Drive are skipped.

**Before the first run**, put in `MyDrive/avrh/` on Google Drive:
`vps_train.zip` (and `vps_val.zip` for `SPLIT = "val"`) and `PIDNet_S_Cityscapes_val.pt`.
The zips hold only the folders the caches read (`train/img`, `train/panoptic_inst`;
`val/img_all`, `val/panoptic_video`, `panoptic_gt_val_city_vps.json`). BiSeNetV2's
checkpoint is downloaded from its release URL.

The caches land in `MyDrive/avrh/cache/<split>/`; copy them to
`experiments/method2/cache/<split>/` of the repo. Accuracy caches run in fp32, so any GPU
gives the same numbers up to float noise; latency is not measured here.
"""),
    code("""
SPLIT = "train"   # "val" once the train sweep has fixed the finalists
BRANCH = "foveation-no-memory"
DRIVE = "/content/drive/MyDrive/avrh"
GEOMS = ["256:32:1", "384:32:1", "512:32:1", "384:32:0.5", "512:32:0.5", "384:24:0.75"]   # input:margin:scale
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
marker = f"{VPS}/.{SPLIT}_ok"
if not os.path.exists(marker):
    t = time.time()
    shutil.copy(f"{DRIVE}/vps_{SPLIT}.zip", "/content/vps.zip")   # local disk: much faster reads than Drive
    zipfile.ZipFile("/content/vps.zip").extractall(VPS)
    os.remove("/content/vps.zip")
    open(marker, "w").close()
    print(f"dataset ready in {time.time() - t:.0f} s")
shutil.copy(f"{DRIVE}/PIDNet_S_Cityscapes_val.pt", "/content/PIDNet_S_Cityscapes_val.pt")
"""),
    code("""
!rm -rf /content/avrh && git clone -q --depth 1 -b $BRANCH https://github.com/AntoineBesson/Active-Vision-on-reduced-hardware.git /content/avrh
%cd /content/avrh
!git log --oneline -1
!pip install -q -e .
"""),
    code("""
import glob, subprocess, sys
root, saved = "/content/cache", f"{DRIVE}/cache/{SPLIT}"
os.makedirs(f"{root}/{SPLIT}", exist_ok=True); os.makedirs(saved, exist_ok=True)
for f in glob.glob(f"{saved}/*.npz"):   # caches of an earlier runtime: method2 cache skips them
    if not os.path.exists(f"{root}/{SPLIT}/{os.path.basename(f)}"):
        shutil.copy(f, f"{root}/{SPLIT}/")
env = dict(os.environ, CITYSCAPES_VPS_ROOT=VPS, PIDNET_S_WEIGHTS="/content/PIDNet_S_Cityscapes_val.pt")
log = f"/content/method2_cache_{SPLIT}.log"
for geom in GEOMS:
    t = time.time()
    with open(log, "a") as f:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.method2", "cache", "--split", SPLIT,
                                 "--geoms", geom, "--root", root],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        while chunk := proc.stdout.read(200):   # the progress line ends in \\r, not \\n
            f.write(chunk)
            print(chunk, end="", flush=True)
    if proc.wait() != 0:
        print("".join(open(log).readlines()[-40:]))
        raise RuntimeError(f"method2 cache failed on {geom} (exit {proc.returncode}); full log: {log}")
    new = [p for p in glob.glob(f"{root}/{SPLIT}/*.npz") if not os.path.exists(f"{saved}/{os.path.basename(p)}")]
    for p in new:
        shutil.copy(p, saved)
    print(f"\\n== {geom}: {len(new)} caches saved to Drive in {(time.time() - t) / 60:.1f} min ==")
"""),
    code("""
print(len(glob.glob(f"{saved}/*.npz")), "caches on Drive in", saved)
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1) + "\n")
