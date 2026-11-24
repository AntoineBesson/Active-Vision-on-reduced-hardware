"""Generate notebooks/method10_gate0_colab.ipynb (Method 10 gate 0: T4 timing of the slow/fast split).

    python scripts/make_method10_notebook.py notebooks/method10_gate0_colab.ipynb
"""

import json
import sys

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# Method 10, gate 0: is the slow branch worth reusing? (T4, ~10 min)

Spec: `docs/superpowers/specs/2026-11-23-method10-slow-branch-reuse-design.md`.

Times on the T4, end to end from the 1024 x 2048 frame (fp16, CUDA graphs, everything
interleaved in the same rounds), for PIDNet-S and BiSeNetV2 at scales 0.5 and 1:

* **key**: the full model, also keeping the slow-branch tensors;
* **fresh**: a non-key frame (fresh branch + cached slow tensors);
* **ff**: both models' full frame at the 11 envelope scales, so the envelope is priced in this session.

Then the **ceiling**: `FF_model(s) - Envelope(T_fresh(s))` with the train_all full-frame mIoU of
Methods 7/8. Gate 0 passes for a model if it is at least +1.0 point at some scale. No dataset,
no weights (latency does not depend on them).

**Runtime → Change runtime type → T4 GPU**, then *Runtime → Run all*. One session is enough;
results go to `MyDrive/avrh/method10/` and are printed at the end for pasting back.
"""),
    code("""
SESSION = "s1"
BRANCH = "claude/project-thread-pbd01q"
DRIVE = "/content/drive/MyDrive/avrh"
SCALES = "0.5,1"
"""),
    code("""
!nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,temperature.gpu --format=csv
import torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("torch", torch.__version__, "| GPU:", name)
assert name and "T4" in name, "Switch the runtime to a T4 GPU (Runtime -> Change runtime type)."
"""),
    code("""
import os, shutil
from google.colab import drive
drive.mount("/content/drive")
OUT = f"{DRIVE}/method10"
os.makedirs(OUT, exist_ok=True)
"""),
    code("""
!rm -rf /content/avrh && git clone -q --depth 1 -b $BRANCH https://github.com/AntoineBesson/Active-Vision-on-reduced-hardware.git /content/avrh
%cd /content/avrh
!git log --oneline -1
!pip install -q -e . nvidia-ml-py
!python -m pytest -q tests/test_method10_split.py tests/test_method10_cli.py
"""),
    md("""
## 1. Timing
"""),
    code("""
import subprocess, sys
def run(cmd, log):
    with open(log, "a") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while chunk := proc.stdout.read(200):
            f.write(chunk)
            print(chunk, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd[3:5]} failed (exit {proc.returncode}); full log: {log}")

run([sys.executable, "-u", "-m", "scripts.method10", "gate0", "--session", SESSION, "--scales", SCALES,
     "--out", "/content/method10"], f"/content/method10_gate0_{SESSION}.log")
shutil.copy(f"/content/method10/gate0_{SESSION}.json", OUT)
print("saved", f"{OUT}/gate0_{SESSION}.json")
"""),
    md("""
## 2. The ceiling and the gate 0 verdict (reads every session on Drive)
"""),
    code("""
run([sys.executable, "-u", "-m", "scripts.method10", "ceiling", "--timing", f"{OUT}/gate0_*.json",
     "--out", f"{OUT}/gate0_report.json"], "/content/method10_ceiling.log")
"""),
    md("""
**Result files** are in `MyDrive/avrh/method10/`. The cell below prints them compressed
(gzip + base64) between markers, to paste back into the thread.
"""),
    code("""
import base64, gzip
for path in (f"{OUT}/gate0_report.json", f"{OUT}/gate0_{SESSION}.json"):
    raw = open(path, "rb").read()
    b64 = base64.b64encode(gzip.compress(raw, 9)).decode()
    tag = os.path.basename(path).replace(".json", "").upper()
    print(f"METHOD10_{tag}_BEGIN"); print(b64); print(f"METHOD10_{tag}_END")
    print(len(raw), "bytes ->", len(b64), "chars")
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1) + "\n")
