"""Per-clip full-frame confusions (method 2's reference curves) on the synthetic fixture (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from eval.metrics import ConfusionMatrix  # noqa: E402
from scripts import method2_fullframe, run_baseline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CFG = str(REPO / "configs" / "baseline_fixture.yaml")


def test_per_clip_confusions_sum_to_the_baseline_miou(tmp_path):
    assert method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(tmp_path)]) == 0
    data = np.load(tmp_path / "fullframe_dummy_val.npz")
    assert data["scales"].tolist() == [1.0, 0.5, 0.25]
    assert data["confusion"].shape == (3, len(data["clip_ids"]), 19, 19)
    assert json.loads(str(data["meta"]))["split"] == "val"
    baseline = run_baseline.main(["--config", FIXTURE_CFG, "--out", str(tmp_path / "bl"),
                                  "--set", "cost.latency.enabled=false"])
    assert baseline == 0
    recs = {json.loads(f.read_text())["method"]["scale"]: json.loads(f.read_text())
            for f in (tmp_path / "bl").glob("*/*.json")}
    for s, scale in enumerate(data["scales"]):
        cm = ConfusionMatrix(19)
        cm.matrix = data["confusion"][s].sum(0)
        assert cm.compute()["miou"] == pytest.approx(recs[float(scale)]["metrics"]["miou"], abs=1e-9)


def test_refuses_frames_other_than_1024x2048_without_fixture(tmp_path):
    with pytest.raises(SystemExit, match="1024x2048"):
        method2_fullframe.main(["--config", FIXTURE_CFG, "--split", "val", "--out", str(tmp_path),
                                "--require-1024x2048"])
