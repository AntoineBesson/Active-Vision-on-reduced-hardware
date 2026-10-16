"""Dataset registry and CLI config helpers (no torch)."""

from __future__ import annotations

from pathlib import Path

import pytest

import generate_fixture as gf
from data.base import VideoSegDataset
from data.registry import build_dataset, resolve_dataset_class
from scripts.common import apply_overrides, expand_env

REPO = Path(__file__).resolve().parents[1]


def test_build_dataset_from_spec_with_relative_paths():
    rel_root = str(gf.OFFICIAL_ROOT.relative_to(REPO))
    ds = build_dataset(
        {"type": "cityscapes_vps", "config": "configs/cityscapes_vps.yaml", "split": "train", "root": rel_root},
        base_dir=REPO,
    )
    assert isinstance(ds, VideoSegDataset) and ds.clip_ids == ["0004", "0005"]
    # dotted "module:Class" form and adapter overrides (highest precedence)
    ds2 = build_dataset(
        {"type": "data.cityscapes_vps:CityscapesVPS", "config": str(REPO / "configs" / "cityscapes_vps.yaml"),
         "split": "val", "root": str(gf.OFFICIAL_ROOT), "overrides": {"annotation_subdir": "labelmap", "label_mapping": None}},
    )
    assert ds2.config.annotation_subdir == "labelmap" and len(ds2) == 3


def test_registry_errors():
    with pytest.raises(ValueError, match="unknown dataset type"):
        build_dataset({"type": "kitti"})
    with pytest.raises(ValueError, match="needs a 'type'"):
        build_dataset({})
    with pytest.raises(TypeError, match="not a VideoSegDataset"):
        resolve_dataset_class("pathlib:Path")
    with pytest.raises(ValueError, match="unknown dataset spec keys"):
        build_dataset({"type": "cityscapes_vps", "colour": "red"})


def test_apply_overrides_and_env(monkeypatch):
    cfg = {"a": {"b": 1}, "c": "x"}
    out = apply_overrides(cfg, ["a.b=2", "a.d=[1, 2]", "e.f=null", "c=hello world"])
    assert out == {"a": {"b": 2, "d": [1, 2]}, "c": "hello world", "e": {"f": None}}
    assert cfg == {"a": {"b": 1}, "c": "x"}  # input untouched
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["novalue"])
    monkeypatch.setenv("AVRH_TEST_VAR", "/w.pt")
    monkeypatch.delenv("AVRH_UNSET_VAR", raising=False)
    assert expand_env({"w": "${AVRH_TEST_VAR}", "l": ["${AVRH_UNSET_VAR}"], "n": 3}) == {
        "w": "/w.pt", "l": ["${AVRH_UNSET_VAR}"], "n": 3,
    }
