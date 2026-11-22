"""Methods 7 and 8 CLI end to end on the synthetic fixture (CPU): run -> merge -> score -> oracle."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import method7  # noqa: E402


def _read(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("m7")
    common = ["--fixture", "--split", "val", "--out-dir", str(d)]
    assert method7.main(["run", *common, "--chunk", "0/2", "--device", "cpu"]) == 0
    assert method7.main(["run", *common, "--chunk", "1/2", "--device", "cpu"]) == 0
    assert method7.main(["merge", *common]) == 0
    return d, common


def test_run_files(work):
    d, _ = work
    rows = _read(d / "run_val_dummy.jsonl.gz")
    assert rows and all(set(r["ff"]) == {"ff/0.25", "ff/0.5", "ff/0.75", "ff/1"} for r in rows)
    n_guided = 1 * 2 * 2 * 2 * 3  # one scale x guides x targets x radii x eps
    for r in rows:
        assert len(r["guided"]) == n_guided
        assert all(len(v) == len(r["frames"]) for v in r["ff"].values())
        tot = np.asarray(r["ff"]["ff/0.5"]).sum()
        assert all(np.asarray(v).sum() == tot for v in r["guided"].values())  # same scored pixels


def test_rerun_resumes(work):
    d, common = work
    before = _read(d / "run_val_dummy_c0of2.jsonl.gz")
    assert method7.main(["run", *common, "--chunk", "0/2", "--device", "cpu"]) == 0
    assert _read(d / "run_val_dummy_c0of2.jsonl.gz") == before


def test_score_and_oracle(work):
    d, common = work
    out = d / "score.json"
    assert method7.main(["score", *common, "--out", str(out), "--n-boot", "50"]) == 0
    rep = json.loads(out.read_text())
    assert len(rep["configs"]) == 24 and "dummy" in rep["gate1"]
    c = rep["configs"][0]
    assert abs(c["delta_same_scale"] - (c["miou"] - c["ff_same_scale"])) < 1e-12
    # one model: the envelope is the model's own curve
    assert all(abs(c["delta_envelope"] - c["delta_same_curve"]) < 1e-12 for c in rep["configs"])
    out = d / "oracle.json"
    assert method7.main(["oracle", *common, "--out", str(out), "--n-boot", "50", "--budgets", "0.5,0.75",
                         "--seeds", "2"]) == 0
    rep = json.loads(out.read_text())
    for e in rep["budgets"]["dummy"]:
        assert e["oracle_mean_ms"] <= e["budget_ms"] + 1e-9
        assert e["oracle_miou"] >= e["ff_same_scale"] - 1e-12


def test_objects_report_with_oracle(tmp_path):
    """A hand-made object table and run file: the oracle's objects come from the scale it chose per frame."""
    rng = np.random.default_rng(0)
    clips, frames = ["a", "b", "c"], [0, 5]
    cols = {k: [] for k in ("clip", "t", "id", "cls", "area", "cx", "cy", "motion", "status", "matched", "raw_shift")}
    for c in clips:
        for t in frames:
            if c == "c" and t == 0:
                continue  # a labelled frame without objects
            for k in range(4):
                for key, v in (("clip", c), ("t", t), ("id", k), ("cls", 11), ("area", 500 + 100 * k), ("cx", 1.0),
                               ("cy", 1.0), ("motion", 5.0), ("status", 0), ("matched", 1), ("raw_shift", 1.0)):
                    cols[key].append(v)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"meta": {}, "columns": cols}))
    conf = np.eye(19, dtype=int).tolist()
    run = tmp_path / "run_train_all_dummy.jsonl.gz"
    rows = []
    for c in clips:
        n = sum(1 for x in cols["clip"] if x == c)
        rows.append({"clip_id": c, "frames": frames,
                     "ff": {f"ff/{s:g}": [conf, conf] for s in (0.25, 0.5, 1.0)},
                     "guided": {}, "iou": {f"ff/{s:g}": (rng.uniform(0, 1, n) * s + 0.3 * (s == 1.0)).tolist()
                                           for s in (0.25, 0.5, 1.0)}})
    method7._append(run, rows)
    # oracle: frame 0 of each clip at scale index 2 (ff/1), frame 5 at index 0 (ff/0.25)
    frame_clip = [c for c in clips for _ in frames]
    choice = [2, 0] * len(clips)
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"budgets": {"dummy": [{"budget_scale": 0.5, "oracle_mean_ms": 5.0,
                                                         "frame_clip": frame_clip, "choice": choice}]}}))
    out = tmp_path / "obj.json"
    assert method7.main(["objects", "--fixture", "--split", "train_all", "--out-dir", str(tmp_path), "--table", str(table),
                         "--oracle", str(oracle), "--out", str(out), "--n-boot", "20"]) == 0
    rep = json.loads(out.read_text())["models"]["dummy"]
    names = [e["name"] for e in rep["configs"]]
    assert "8o/budget0.5" in names and "ff/1" in names
    # the oracle's found objects: ff/1's on frame 0, ff/0.25's on frame 5
    clip = np.asarray(cols["clip"])
    t = np.asarray(cols["t"])
    found = 0
    for r in rows:
        m = clip == r["clip_id"]
        tt = t[m]
        found += int(((np.asarray(r["iou"]["ff/1"]) > 0.5) & (tt == 0)).sum())
        found += int(((np.asarray(r["iou"]["ff/0.25"]) > 0.5) & (tt == 5)).sum())
    e = next(e for e in rep["configs"] if e["name"] == "8o/budget0.5")
    assert e["recall"]["found"] == found


def test_screen_selections_and_cli(work):
    sel = method7.screen_selections([3, 1, 2, 0, 4], ["a", "a", "a", "b", "b"], 5, base=9)
    assert sel["stale"].tolist() == [9, 3, 1, 9, 0]
    assert sorted(sel["shuffle_all"].tolist()) == [0, 1, 2, 3, 4]
    assert sorted(sel["shuffle_clip"][:3].tolist()) == [1, 2, 3]
    d, common = work
    out = d / "screen.json"
    assert method7.main(["screen", *common, "--out", str(out), "--n-boot", "30", "--budgets", "0.5", "--seeds", "1"]) == 0
    e = json.loads(out.read_text())["models"]["dummy"][0]
    assert set(e) >= {"frame_oracle", "clip_oracle", "stale", "shuffle_clip", "shuffle_all", "pass"}
    assert e["frame_oracle"]["miou"] >= e["clip_oracle"]["miou"] - 1e-12


def test_merge_takes_a_finalists_rerun(tmp_path):
    base = {"split": "val", "model": "dummy", "clip_id": "x", "frames": [0]}
    conf = np.eye(19, dtype=int).tolist()
    method7._append(tmp_path / "run_val_dummy_c0of1.jsonl.gz",
                    [{**base, "ff": {"ff/1": [conf]}, "guided": {"7g/c0.5/gray/logit/r1/e0.001": conf}, "iou": {"ff/1": [0.7]}}])
    method7._append(tmp_path / "run_val_dummy_fin.jsonl.gz",
                    [{**base, "ff": {}, "guided": {"7g/c0.5/gray/logit/r1/e0.001": conf},
                      "iou": {"7g/c0.5/gray/logit/r1/e0.001": [0.6]}}])
    assert method7.main(["merge", "--fixture", "--split", "val", "--out-dir", str(tmp_path), "--no-expect"]) == 0
    (r,) = _read(tmp_path / "run_val_dummy.jsonl.gz")
    assert set(r["iou"]) == {"ff/1", "7g/c0.5/gray/logit/r1/e0.001"} and len(r["guided"]) == 1
    # a repeated IoU row is still refused
    method7._append(tmp_path / "run_val_dummy_fin2.jsonl.gz", [{**base, "iou": {"ff/1": [0.1]}}])
    with pytest.raises(SystemExit):
        method7.main(["merge", "--fixture", "--split", "val", "--out-dir", str(tmp_path), "--no-expect"])
