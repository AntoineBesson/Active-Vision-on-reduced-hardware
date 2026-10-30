"""Generate notebooks/latency_floor_colab.ipynb (the T4 latency runs on Colab).

    python scripts/make_latency_notebook.py notebooks/latency_floor_colab.ipynb
"""

import json
import sys

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
    md("""
# Latency floor on a T4 (batch 1, FP16, streaming)

Runs `scripts/latency_floor.py` from the `foveation-no-memory` branch: backbone latency
vs input area and vs number of batched glimpses (PIDNet-S, BiSeNetV2), the pipeline's own
overhead with a zero-compute backbone, and a few end-to-end points compared with the floor
predicted from the parts. No dataset needed (random weights and frames).

**Runtime → Change runtime type → T4 GPU**, then *Runtime → Run all*. One run = one session;
do 3 runs in separate Colab sessions (*Runtime → Disconnect and delete runtime* between them)
with a different `SESSION` each time.

**Method 2 finalists:** set `PARTS = "final"` (and `SESSION = "final-s1"`, ...). It times the
finalists of `results/method2/finalists.json` interleaved with both models' full frame at
five scales, end to end.
"""),
    code("""
SESSION = "t4-s1"   # change for each separate session: t4-s2, t4-s3, ... (final-s1, ... for the finalists)
BRANCH = "foveation-no-memory"
PARTS = "area,batch,overhead,e2e"   # "final": time the method 2 finalists (results/method2/finalists.json)
"""),
    code("""
!nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,temperature.gpu --format=csv
import torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print("torch", torch.__version__, "| GPU:", name)
assert name and "T4" in name, "Switch the runtime to a T4 GPU (Runtime -> Change runtime type)."
"""),
    code("""
!rm -rf /content/avrh && git clone -q --depth 1 -b $BRANCH https://github.com/AntoineBesson/Active-Vision-on-reduced-hardware.git /content/avrh
%cd /content/avrh
!git log --oneline -1
!pip install -q -e . nvidia-ml-py
"""),
    code("""
import subprocess, sys
log = f"/content/latency_floor_{SESSION}.log"
with open(log, "w") as f:
    proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.latency_floor", "--session", SESSION, "--parts", PARTS],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        f.write(line)
        if not line.startswith("W1"):  # torch's "triton not found" warnings
            print(line, end="")
if proc.wait() != 0:
    print("".join(open(log).readlines()[-40:]))
    raise RuntimeError(f"latency_floor failed (exit {proc.returncode}); full log: {log}")
"""),
    code("""
import json, matplotlib.pyplot as plt
res = json.load(open(f"experiments/latency_floor/latency_floor_{SESSION}.json"))
R = res["records"]
print(res["environment"]); print("session telemetry:", res["session_telemetry"])

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for model, mk in (("pidnet_s", "o"), ("bisenetv2", "s")):
    for level, ls in (("eager", "-"), ("graph", "--")):
        pts = sorted((r["pixels"] / 1e6, r["cuda_event"]["median_ms"]) for r in R.get("area", [])
                     if r["model"] == model and r["level"] == level)
        if pts:
            axes[0].plot(*zip(*pts), ls, marker=mk, label=f"{model} {level}")
        for q in sorted({r["size"] for r in R.get("batch", [])}):
            pts = sorted((r["k"], r["cuda_event"]["median_ms"]) for r in R.get("batch", [])
                         if r["model"] == model and r["level"] == level and r["size"] == q)
            if pts and level == "graph":
                axes[1].plot(*zip(*pts), ls, marker=mk, label=f"{model} {q}px")
axes[0].set(xlabel="input megapixels (k=1)", ylabel="median latency (ms)", title="backbone vs area"); axes[0].legend()
axes[1].set(xlabel="k glimpses in one batch", ylabel="median latency (ms)", title="backbone vs k (graph)"); axes[1].legend(fontsize=7)
ov = [r for r in R.get("overhead", []) if r["selection"] == "topk"]
for level, ls in (("eager", "-"), ("graph", "--")):
    for stride in (8, 1):
        pts = sorted((r["k"] * 1000 + r["size"], r["cuda_event"]["median_ms"]) for r in ov
                     if r["level"] == level and r["stride"] == stride and r["context_scale"] == 0.25)
        if pts:
            axes[2].plot(range(len(pts)), [p[1] for p in pts], ls, marker="o", label=f"stride {stride} {level}")
axes[2].set(xlabel="(k, glimpse size) configs, context 0.25", ylabel="median latency (ms)", title="pipeline overhead (topk)"); axes[2].legend()
plt.tight_layout(); plt.show()
for r in R.get("e2e", []):
    p = r.get("predicted_ms", {}).get("total")
    print(f"{r['level']:5s} {r['name']:40s} measured {r['cuda_event']['median_ms']:6.2f} ms  predicted {p if p is None else round(p, 2)}")
for r in R.get("final", []):
    p = r.get("predicted_median_ms")
    print(f"{r['level']:5s} {r['name']:70s} median {r['cuda_event']['median_ms']:6.2f}  p95 {r['cuda_event']['p95_ms']:6.2f} ms"
          + ("" if p is None else f"  predicted {p:6.2f}"))
"""),
    md("""
**Result file.** The cell below prints the result JSON compressed (gzip + base64) between
markers, so it can be copied from the page; `files.download` saves the plain JSON.
"""),
    code("""
import base64, gzip
raw = open(f"experiments/latency_floor/latency_floor_{SESSION}.json", "rb").read()
b64 = base64.b64encode(gzip.compress(raw, 9)).decode()
print(f"LATENCY_FLOOR_{SESSION}_BEGIN"); print(b64); print(f"LATENCY_FLOOR_{SESSION}_END")
print(len(raw), "bytes ->", len(b64), "chars")
"""),
    code("""
from google.colab import files
files.download(f"experiments/latency_floor/latency_floor_{SESSION}.json")
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1) + "\n")
