"""One self-describing JSON file per evaluation run (standard library + numpy).

Every record carries the resolved config, the git commit (and whether the
work tree was dirty), the seed, the dataset/backbone descriptions, metrics,
cost and the software/hardware environment, so a number can always be traced
back to exactly what produced it. NaN/inf are written as ``null`` so the files
are strict JSON.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

__all__ = ["SCHEMA_VERSION", "git_info", "environment_info", "to_jsonable", "make_run_record", "write_json"]

SCHEMA_VERSION = 1


def _git(args, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Commit hash, branch and dirty flag of the repository (None if unavailable)."""
    cwd = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[2]
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": None if status is None else bool(status),
    }


def environment_info() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["cudnn"] = torch.backends.cudnn.version()
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
    return env


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/paths/tuples; NaN and inf become None."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def make_run_record(
    *,
    run_name: str,
    method: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    dataset: Mapping[str, Any],
    backbone: Mapping[str, Any],
    metrics: Mapping[str, Any],
    cost: Mapping[str, Any],
    notes: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_info(),
        "seed": int(seed),
        "method": dict(method),
        "dataset": dict(dataset),
        "backbone": dict(backbone),
        "metrics": dict(metrics),
        "cost": dict(cost),
        "environment": environment_info(),
        "config": dict(config),
    }
    if notes:
        record["notes"] = notes
    if extra:
        record.update(dict(extra))
    return to_jsonable(record)


def write_json(path: "str | os.PathLike", record: Mapping[str, Any]) -> Path:
    """Atomically write strict JSON (no NaN) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_jsonable(record), indent=2, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        # mkstemp creates 0600 files; give results normal permissions (umask applies).
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
