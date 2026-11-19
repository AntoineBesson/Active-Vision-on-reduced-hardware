"""Generate notebooks/bigmodel_curve_colab.ipynb (large-model full-frame curve on a Colab T4).

    python scripts/make_bigmodel_notebook.py notebooks/bigmodel_curve_colab.ipynb
"""

import json
import sys

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# Larger frozen models: full-frame curve on a T4

First check before any warp on a larger model. It measures, for PIDNet-S, BiSeNetV2 and
SegFormer B0 / B2 / B5 (NVIDIA's public Cityscapes weights, nothing trained):

1. **latency** on the T4 at 7 scales (0.125 to 1), fp16, CUDA graphs, every model
   interleaved in the same rounds: the backbone alone (`area`) and from the full frame to
   the argmax (`e2e`). Random weights, no dataset.
2. **accuracy** (VPS val mIoU, fp32) of every model at the same scales, with
   `scripts/run_baseline.py`. Only needed once (`ACCURACY = True` in the first session).
3. **the screen** (`scripts/bigmodel_curve.py report`): for each SegFormer point, the best
   small model's mIoU at the same latency. GO only if a SegFormer scale below 1 is within
   2 points of it **and** its own curve still gains at least 1 point at the next scale.

**Runtime → Change runtime type → T4 GPU**, then *Runtime → Run all*. Do **3 sessions**
(`SESSION = "s1"`, then `"s2"`, `"s3"`, with *Disconnect and delete runtime* in between);
set `ACCURACY = False` for s2 and s3. Every result is saved to `MyDrive/avrh/bigmodel/` as it
is written, and the report at the end reads all sessions found there.

**Before the first run**, `MyDrive/avrh/` must hold `vps_val.zip` and
`PIDNet_S_Cityscapes_val.pt` (the same files as the Method 2 notebook). BiSeNetV2 and
SegFormer weights are downloaded from their public URLs.
"""),
    code("""
SESSION = "s1"        # s1, s2, s3: one per separate Colab session
ACCURACY = True       # True in s1 only (fp32 accuracy does not change between sessions)
BRANCH = "claude/project-thread-pbd01q"
DRIVE = "/content/drive/MyDrive/avrh"
LARGE = ["segformer_b0", "segformer_b2", "segformer_b5"]
SCALES = [1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125]
"""),
    code("""
!nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,temperature.gpu --format=csv
import torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("torch", torch.__version__, "| GPU:", name)
assert name and "T4" in name, "Switch the runtime to a T4 GPU (Runtime -> Change runtime type)."
"""),
    code("""
import os, shutil, time, zipfile
from google.colab import drive
drive.mount("/content/drive")
OUT = f"{DRIVE}/bigmodel"
os.makedirs(f"{OUT}/accuracy", exist_ok=True)
VPS = "/content/vps"
if ACCURACY and not os.path.exists(f"{VPS}/.val_ok"):
    t = time.time()
    shutil.copy(f"{DRIVE}/vps_val.zip", "/content/vps.zip")   # local disk: much faster reads than Drive
    zipfile.ZipFile("/content/vps.zip").extractall(VPS)
    os.remove("/content/vps.zip")
    open(f"{VPS}/.val_ok", "w").close()
    print(f"dataset ready in {time.time() - t:.0f} s")
if ACCURACY:
    shutil.copy(f"{DRIVE}/PIDNet_S_Cityscapes_val.pt", "/content/PIDNet_S_Cityscapes_val.pt")
"""),
    code("""
!rm -rf /content/avrh && git clone -q --depth 1 -b $BRANCH https://github.com/AntoineBesson/Active-Vision-on-reduced-hardware.git /content/avrh
%cd /content/avrh
!git log --oneline -1
!pip install -q -e . nvidia-ml-py transformers
"""),
    md("""
## 1. Latency (about 15 to 25 min)
"""),
    code("""
import subprocess, sys
def run(cmd, log, env=None):
    with open(log, "a") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        while chunk := proc.stdout.read(200):   # progress lines end in \\r, not \\n
            f.write(chunk)
            print(chunk, end="", flush=True)
    if proc.wait() != 0:
        print("".join(open(log).readlines()[-40:]))
        raise RuntimeError(f"{cmd[3:5]} failed (exit {proc.returncode}); full log: {log}")

models = ",".join(["pidnet_s", "bisenetv2"] + LARGE)
run([sys.executable, "-u", "-m", "scripts.bigmodel_curve", "latency", "--session", SESSION, "--models", models,
     "--scales", ",".join(map(str, SCALES)), "--out", "/content/latency"], f"/content/bigmodel_latency_{SESSION}.log")
shutil.copy(f"/content/latency/latency_{SESSION}.json", OUT)
print("saved", f"{OUT}/latency_{SESSION}.json")
"""),
    md("""
## 2. Accuracy on VPS val (first session only; about 1 h with B5)

One `run_baseline` per model. A model whose JSONs are already on Drive is skipped, so a
runtime that is cut off resumes where it stopped.
"""),
    code("""
import glob, json
CONFIGS = {"pidnet_s": "configs/baseline_fullframe_pidnet_s.yaml", "bisenetv2": "configs/baseline_fullframe_bisenetv2.yaml"}
env = dict(os.environ, CITYSCAPES_VPS_ROOT=VPS, PIDNET_S_WEIGHTS="/content/PIDNet_S_Cityscapes_val.pt")
if ACCURACY:
    for model in ["pidnet_s", "bisenetv2"] + LARGE:
        if len(glob.glob(f"{OUT}/accuracy/fullframe_{model}_s*.json")) >= len(SCALES):
            print(f"== {model}: already on Drive, skipped"); continue
        sets = ["--set", f"eval.scales={SCALES}", "--set", "cost.latency.enabled=false", "--set", "cost.memory=false",
                "--set", f"run.name=fullframe_{model}"]
        cfg = CONFIGS.get(model, "configs/baseline_fullframe_segformer.yaml")
        if model.startswith("segformer_"):
            v = model.split("_")[1]
            sets += ["--set", f"backbone.type={model}",
                     "--set", f"backbone.weights=nvidia/segformer-{v}-finetuned-cityscapes-1024-1024"]
        t = time.time()
        run([sys.executable, "-u", "-m", "scripts.run_baseline", "--config", cfg, "--out", "/content/acc", *sets],
            f"/content/bigmodel_accuracy.log", env=env)
        for p in glob.glob(f"/content/acc/*_fullframe_{model}/fullframe_{model}_s*.json"):
            shutil.copy(p, f"{OUT}/accuracy/")
        print(f"\\n== {model}: {(time.time() - t) / 60:.1f} min, saved to Drive ==")
"""),
    md("""
## 3. The screen (reads every session and accuracy file on Drive)
"""),
    code("""
run([sys.executable, "-u", "-m", "scripts.bigmodel_curve", "report", "--latency", f"{OUT}/latency_*.json",
     "--accuracy", f"{OUT}/accuracy/*.json", "--out", f"{OUT}/report.json"], "/content/bigmodel_report.log")
"""),
    code("""
import matplotlib.pyplot as plt
rep = json.load(open(f"{OUT}/report.json"))
fig, axes = plt.subplots(1, len(rep["parts"]), figsize=(7 * len(rep["parts"]), 5), squeeze=False)
for ax, (part, r) in zip(axes[0], rep["parts"].items()):
    for m, pts in r["curves"].items():
        if pts:
            ax.plot([p["latency_ms"] for p in pts], [p["miou"] for p in pts], "o-" if m in rep["small"] else "s--", label=m)
            for p in pts:
                ax.annotate(f"{p['scale']:g}", (p["latency_ms"], p["miou"]), fontsize=7)
    ax.set(xscale="log", xlabel="T4 median latency (ms, fp16, graph)", ylabel="VPS val mIoU",
           title=f"{part}: screen {r['screen']}")
    ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=150); plt.show()
print("sessions:", sorted({os.path.basename(p) for p in rep["latency_files"]}))
"""),
    md("""
**Result files.** Everything is in `MyDrive/avrh/bigmodel/`. The cell below also prints the
report and this session's latency compressed (gzip + base64) between markers, so they can
be pasted back into the thread.
"""),
    code("""
import base64, gzip
for path in (f"{OUT}/report.json", f"{OUT}/latency_{SESSION}.json"):
    raw = open(path, "rb").read()
    b64 = base64.b64encode(gzip.compress(raw, 9)).decode()
    tag = os.path.basename(path).replace(".json", "").upper()
    print(f"BIGMODEL_{tag}_BEGIN"); print(b64); print(f"BIGMODEL_{tag}_END")
    print(len(raw), "bytes ->", len(b64), "chars\\n")
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1) + "\n")
