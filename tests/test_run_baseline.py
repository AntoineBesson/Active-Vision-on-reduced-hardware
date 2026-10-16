"""Full-frame baseline CLI end to end on the synthetic fixture (CPU, no data)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts import run_baseline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CFG = str(REPO / "configs" / "baseline_fixture.yaml")


def _records(out: Path):
    files = sorted(out.glob("*_fullframe_dummy_fixture/*.json"))
    return files, [json.loads(f.read_text()) for f in files]


def test_cli_writes_one_json_per_scale(tmp_path):
    assert run_baseline.main(["--config", FIXTURE_CFG, "--out", str(tmp_path)]) == 0
    files, recs = _records(tmp_path)
    assert [f.name for f in files] == [
        "fullframe_dummy_fixture_s0.250.json",
        "fullframe_dummy_fixture_s0.500.json",
        "fullframe_dummy_fixture_s1.000.json",
    ]
    assert (files[0].parent / "config.yaml").is_file()
    assert [r["run_name"] + ".json" for r in recs] == [f.name for f in files]
    by_scale = {r["method"]["scale"]: r for r in recs}
    full = by_scale[1.0]
    # provenance
    assert full["schema_version"] == 1 and full["seed"] == 0
    assert full["git"]["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", full["git"]["commit"])
    assert full["config"]["dataset"]["type"] == "cityscapes_vps" and full["config"]["fixture"] is True
    assert full["dataset"]["synthetic_fixture"] is True and len(full["dataset"]["fingerprint"]) == 40
    assert full["dataset"]["num_labeled_frames"] == 6 and full["backbone"]["type"] == "dummy"
    assert "SYNTHETIC" in full["notes"]
    # metrics: the nearest-colour dummy is exact at full resolution (alignment check)
    assert full["metrics"]["miou"] == pytest.approx(1.0) and full["metrics"]["num_frames"] == 6
    assert by_scale[0.25]["metrics"]["miou"] < by_scale[0.5]["metrics"]["miou"] < 1.0
    # cost
    for scale, rec in by_scale.items():
        assert rec["cost"]["pixel_fraction"] == pytest.approx(scale**2)
        assert rec["cost"]["input_size"] == [round(16 * scale), round(32 * scale)]
        assert "FlopCounterMode" in rec["cost"]["flop_counter"]
        assert rec["cost"]["components"]["backbone"]["flops_per_frame"] > 0
        status = rec["cost"]["latency"]["status"]
        assert status == ("ok" if rec["environment"]["cuda_available"] else "skipped")


def test_cli_overrides_and_max_clips(tmp_path):
    run_baseline.main(["--config", FIXTURE_CFG, "--out", str(tmp_path), "--set", "eval.scales=[0.5]",
                       "--set", "run.seed=7", "--max-clips", "1"])
    files, recs = _records(tmp_path)
    assert len(files) == 1
    assert recs[0]["seed"] == 7 and recs[0]["dataset"]["clips_evaluated"] == 1
    assert recs[0]["metrics"]["num_frames"] == 2


def test_missing_weights_fail_with_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.delenv("PIDNET_S_WEIGHTS", raising=False)
    with pytest.raises(SystemExit, match="environment variable that is not set"):
        run_baseline.main(["--config", str(REPO / "configs" / "baseline_fullframe_pidnet_s.yaml"),
                           "--fixture", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="needs pretrained weights"):
        run_baseline.main(["--config", str(REPO / "configs" / "baseline_fullframe_pidnet_s.yaml"),
                           "--fixture", "--out", str(tmp_path), "--set", "backbone.weights=null"])


def test_module_entry_point(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_baseline", "--config", FIXTURE_CFG, "--out", str(tmp_path),
         "--set", "eval.scales=[1.0]"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "wrote 1 result file" in proc.stdout
