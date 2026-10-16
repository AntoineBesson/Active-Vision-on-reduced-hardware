"""scripts/diagnose_real_data.py on the official-layout fixture (no torch needed)."""

from __future__ import annotations

import json
import shutil

import generate_fixture as gf
from scripts import diagnose_real_data as diag


def _report(out):
    return json.loads((out / "diagnose_report.json").read_text())


def test_fixture_report_covers_every_requested_item(tmp_path, capsys):
    assert diag.main(["--fixture", "--out", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    rep = _report(tmp_path)
    assert rep["problems"] == [] and "no problem detected" in text
    val = rep["splits"]["val"]
    # clips / frames per split and labelled frames per clip
    assert val["clips"]["num_clips"] == 3 and val["clips"]["num_frames"] == 30
    assert val["clips"]["labeled_per_clip"] == {"2": 3} and val["clips"]["labeled_positions"] == {"4": 3, "9": 3}
    assert rep["splits"]["train"]["clips"]["num_clips"] == 2
    assert rep["splits"]["test"]["clips"]["num_labeled_frames"] == 0
    # regex matches
    idx = val["diagnostics"]["index"]
    assert idx["image_regex_matched"] == 30 and idx["annotation_regex_matched"] == 6
    # unique raw ids and decoded classes on the sampled annotations
    assert len(val["samples"]) == 6
    rows = val["samples"][0]["ids"]
    assert {r["class"] for r in rows} >= {"road", "sky", "car", "<ignore>"}
    assert any(r["id"] == 0 and r["category"] is None for r in rows)
    # histogram of the 19 classes + ignore, resolution
    assert len(val["histogram"]) == 20 and val["histogram"]["<ignore>"] > 0
    assert val["frame_sizes"] == {f"{gf.OH}x{gf.OW}": 30}
    # cross-checks behave as documented: panoptic_inst identical, labelmap differs on crowd only
    assert val["crosscheck"]["panoptic_inst"]["pixel_agreement"] == 1.0
    assert val["crosscheck"]["labelmap"]["top_disagreements"][0][0] == "ignore->bicycle"
    # three overlays, from different clips
    pngs = sorted(tmp_path.glob("overlay_*.png"))
    assert len(pngs) == 3 and len({p.name.split("_")[2] for p in pngs}) == 3


def test_missing_root_is_reported_not_raised(tmp_path):
    assert diag.main(["--root", str(tmp_path / "nowhere"), "--splits", "val", "--out", str(tmp_path)]) == 0
    rep = _report(tmp_path)
    assert any("could not be built" in p for p in rep["problems"])
    assert rep["splits"]["val"]["layout"]["root"]["exists"] is False


def test_wrong_regex_is_reported(tmp_path):
    args = ["--fixture", "--splits", "val", "--out", str(tmp_path), "--overlays", "0",
            "--set", r'dataset.overrides={filename_regex: "^(?P<clip>\\d{4})_(?P<frame>\\d{4})_"}']
    assert diag.main(args) == 0
    problems = _report(tmp_path)["problems"]
    assert any("do not match the filename regex" in p for p in problems)
    assert any("no clip found" in p for p in problems)
    assert not list(tmp_path.glob("overlay_*.png"))


def test_broken_label_mapping_is_reported_per_sample(tmp_path):
    root = tmp_path / "vps"
    shutil.copytree(gf.OFFICIAL_ROOT, root)
    (root / "panoptic_gt_val_city_vps.json").unlink()
    assert diag.main(["--root", str(root), "--splits", "val", "--sample", "2", "--out", str(tmp_path / "o")]) == 0
    rep = _report(tmp_path / "o")
    assert sum("FileNotFoundError" in p for p in rep["problems"]) == 2
    assert all("error" in s for s in rep["splits"]["val"]["samples"])
