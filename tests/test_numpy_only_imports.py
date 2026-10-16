"""The base install (numpy + pillow + pyyaml) must not need torch.

Runs a fresh interpreter in which importing torch/torchvision raises, then
imports every module that is supposed to be torch-free.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# Modules that must import without torch (extended as packages are added).
NUMPY_ONLY_MODULES = [
    "data",
    "data.base",
    "data.label_mapping",
    "data.cityscapes_vps",
    "data.splits",
    "data.subclips",
]

_PROBE = r"""
import importlib, importlib.abc, sys

class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "torchvision"):
            raise ImportError("torch is blocked in this test: " + name)
        return None

sys.meta_path.insert(0, _BlockTorch())
for mod in sys.argv[1:]:
    importlib.import_module(mod)
try:
    importlib.import_module("data.torch_dataset")
except ImportError as exc:
    assert "pip install" in str(exc), exc
else:
    raise SystemExit("data.torch_dataset imported without torch?!")
print("OK")
"""


def test_numpy_only_modules_import_without_torch():
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *NUMPY_ONLY_MODULES],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().endswith("OK")
