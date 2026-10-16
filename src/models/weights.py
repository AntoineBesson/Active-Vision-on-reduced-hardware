"""Load pretrained weights from a local path or a URL, with clear failure modes.

``backbone.weights`` in a config may be

* a local path (``~`` expanded), or
* an ``http(s)://`` / ``file://`` URL, downloaded once into ``cache_dir``
  (default ``$TORCH_HOME/hub/checkpoints``).

Failures are explicit: a missing file, an unreachable URL (no network) or a
Google Drive *share page* (which needs a browser confirmation, not a direct
download) raise :class:`WeightsError` explaining what to do instead. An
optional ``sha256`` pins the exact file.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

__all__ = ["WeightsError", "resolve_weights", "load_checkpoint", "extract_state_dict", "load_into", "sha256_of"]


class WeightsError(RuntimeError):
    """Weights could not be obtained or do not fit the model."""


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_cache_dir() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints"


def resolve_weights(source: str, cache_dir: Optional[str] = None, sha256: Optional[str] = None) -> Path:
    """Return a local file for ``source`` (path or URL), downloading if needed."""
    if not source:
        raise WeightsError("empty weights source")
    if "${" in str(source):
        raise WeightsError(f"weights source {source!r} contains an environment variable that is not set")
    parsed = urllib.parse.urlparse(str(source))
    if parsed.scheme in ("http", "https", "file"):
        if parsed.netloc.endswith("drive.google.com") or parsed.netloc.endswith("docs.google.com"):
            raise WeightsError(
                f"{source} is a Google Drive share page, which cannot be fetched as a direct download "
                "(it needs a browser confirmation). Download the file in a browser (or with `gdown`) "
                "and set `weights` to its local path."
            )
        name = os.path.basename(parsed.path) or "weights.pt"
        target_dir = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
        path = target_dir / name
        if not path.is_file():
            target_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".part")
            os.close(fd)
            try:
                torch.hub.download_url_to_file(str(source), tmp, progress=False)
                shutil.move(tmp, path)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                raise WeightsError(
                    f"Could not download weights from {source}: {exc}. If this machine has no network "
                    "access, download the file elsewhere and set `weights` to its local path."
                ) from exc
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    else:
        path = Path(str(source)).expanduser()
        if not path.is_file():
            raise WeightsError(f"weights file not found: {path}")
    if sha256 and sha256_of(path) != sha256.lower():
        raise WeightsError(f"sha256 mismatch for {path}: expected {sha256}, got {sha256_of(path)}")
    return path


def load_checkpoint(path: Path, trust_pickle: bool = False) -> Any:
    """``torch.load`` on CPU; restricted to tensors/containers unless ``trust_pickle``."""
    try:
        return torch.load(path, map_location="cpu", weights_only=not trust_pickle)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        if trust_pickle:
            raise
        raise WeightsError(
            f"{path} could not be loaded with weights_only=True ({type(exc).__name__}: {exc}). "
            "If you trust this file, set `trust_pickle: true` in the backbone config."
        ) from exc


def extract_state_dict(obj: Any, strip_prefixes: Iterable[str] = ("module.", "model.")) -> Dict[str, torch.Tensor]:
    """Find the parameter dict inside a checkpoint and drop wrapper prefixes.

    Handles ``{"state_dict": ...}`` / ``{"model": ...}`` nesting and DataParallel
    (``module.``) or training-wrapper (``model.``, as in PIDNet's FullModel) keys.
    """
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(obj, Mapping) and isinstance(obj.get(key), Mapping):
            obj = obj[key]
            break
    if not isinstance(obj, Mapping):
        raise WeightsError(f"checkpoint is a {type(obj).__name__}, not a state dict")
    state = {}
    for k, v in obj.items():
        if not torch.is_tensor(v):
            continue
        for prefix in strip_prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
        state[k] = v
    return state


def load_into(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    strict: bool = True,
    ignore_prefixes: Iterable[str] = (),
) -> Dict[str, Any]:
    """Copy ``state`` into ``module`` and report what happened.

    Keys starting with ``ignore_prefixes`` (e.g. training-only auxiliary heads)
    are dropped silently. With ``strict``, missing parameters, leftover
    unexpected keys or shape mismatches raise :class:`WeightsError`.
    """
    ignore = tuple(ignore_prefixes)
    own = module.state_dict()
    ignored = sorted(k for k in state if ignore and k.startswith(ignore))
    candidate = {k: v for k, v in state.items() if not (ignore and k.startswith(ignore))}
    mismatched = sorted(
        f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}"
        for k, v in candidate.items()
        if k in own and tuple(v.shape) != tuple(own[k].shape)
    )
    usable = {k: v for k, v in candidate.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    missing = sorted(k for k in own if k not in usable)
    unexpected = sorted(k for k in candidate if k not in own)
    report = {
        "loaded": len(usable),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": mismatched,
        "ignored": len(ignored),
    }
    if strict and (missing or unexpected or mismatched):
        raise WeightsError(
            "checkpoint does not fit the model: "
            f"{len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:3]}), shape mismatches {mismatched[:3]}. "
            "Check num_classes / variant, or set `strict: false` to load what fits."
        )
    module.load_state_dict(usable, strict=False)
    return report
