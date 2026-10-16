"""Shared CLI plumbing: config files, ``--set`` overrides, seeding, the fixture.

Library code takes plain dicts; YAML handling and command-line overrides live
here, at the edge.
"""

from __future__ import annotations

import copy
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"


def ensure_src_on_path() -> None:
    """Make ./src importable from a source checkout without installing."""
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))


def load_yaml(path: "str | os.PathLike") -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(obj: Any) -> Any:
    """Replace ``${VAR}`` in strings by the environment value (left as is if unset)."""
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    return obj


def apply_overrides(cfg: Mapping[str, Any], assignments: Iterable[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` assignments (values parsed as YAML: numbers, lists, null...)."""
    import yaml

    out = copy.deepcopy(dict(cfg))
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str = "auto") -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {name} requested but CUDA is not available")
    return name


def ensure_fixture() -> Path:
    """Generate (if needed) the synthetic official-layout fixture; return its root."""
    fixtures = REPO_ROOT / "tests" / "fixtures"
    if str(fixtures) not in sys.path:
        sys.path.insert(0, str(fixtures))
    import generate_fixture

    return generate_fixture.ensure_official_fixture()
