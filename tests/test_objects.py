"""Per-object recall (eval.objects) and its CLI (scripts.objects) on synthetic data."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval import objects as ob

CAR, PERSON = 13, 11


def _inst(h=40, w=60):
    return np.zeros((h, w), dtype=np.int64)  # road everywhere


# --------------------------------------------------------------------------- #
# Objects and local IoU
# --------------------------------------------------------------------------- #
def test_frame_objects_finds_things_only_and_drops_small():
    inst = _inst()
    inst[5:15, 5:15] = CAR * 1000 + 1  # 100 px
    inst[20:30, 40:50] = PERSON * 1000  # obj 0 is a valid id, 100 px
    inst[0:2, 0:2] = CAR * 1000 + 2  # 4 px: dropped
    inst[30:40, 0:10] = 8  # vegetation (stuff)
    inst[35:40, 50:60] = 255  # void / crowd
    fo = ob.frame_objects(inst, min_area=10)
    assert fo.ids.tolist() == [PERSON * 1000, CAR * 1000 + 1]
    assert fo.cls.tolist() == [PERSON, CAR]
    assert fo.area.tolist() == [100, 100]
    np.testing.assert_allclose(fo.centroid[1], [9.5, 9.5])
    assert fo.bbox[1].tolist() == [5, 5, 15, 15]


def test_local_iou_ignores_same_class_neighbour_but_counts_spill():
    inst = _inst()
    inst[10:20, 10:20] = CAR * 1000 + 1
    inst[10:20, 20:30] = CAR * 1000 + 2  # touching car to the right
    inst[25:28, 10:20] = 255  # void below
    fo = ob.frame_objects(inst, min_area=1)
    pred = np.zeros_like(inst)
    pred[10:20, 10:30] = CAR  # both cars perfectly, as one blob
    np.testing.assert_allclose(ob.local_ious(inst, pred, fo), [1.0, 1.0])
    pred[20:25, 10:20] = CAR  # 50 px of spill onto road below car 1 (inside its ring)
    pred[25:28, 10:20] = CAR  # on void: not counted
    iou = ob.local_ious(inst, pred, fo)
    assert iou[0] == pytest.approx(100 / 150)
    assert iou[1] == pytest.approx(100 / 140)  # only columns 12..19 of the spill are in car 2's ring


def test_local_iou_wrong_class_is_a_miss():
    inst = _inst()
    inst[10:20, 10:20] = PERSON * 1000 + 1
    fo = ob.frame_objects(inst, min_area=1)
    pred = np.zeros_like(inst)
    pred[10:20, 10:20] = 12  # rider
    assert ob.local_ious(inst, pred, fo)[0] == 0.0


def test_ring_width():
    assert ob.ring_width(np.array([1, 64 * 64, 400 * 400])).tolist() == [8, 16, 100]


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #
def _shift(dx, dy):
    H = np.eye(3)
    H[0, 2], H[1, 2] = dx, dy
    return H


def test_chain_and_warp():
    H = np.stack([np.eye(3)] + [_shift(1, 0)] * 5)
    np.testing.assert_allclose(ob.warp_point(ob.chain_homography(H, 0, 5), np.array([[10.0, 3.0]])), [[15.0, 3.0]])
    with pytest.raises(ValueError):
        ob.chain_homography(H, 3, 3)


def test_clip_objects_removes_camera_motion_and_flags_new_and_unknown():
    T = 11
    H = np.stack([np.eye(3)] + [_shift(2, 0)] * (T - 1))  # camera pans: the scene moves 2 px / frame
    labeled = [0, 5, 10]
    frames = []
    for j, t in enumerate(labeled):
        inst = _inst(60, 120)
        inst[10:20, 10 + 2 * t:20 + 2 * t] = CAR * 1000 + 1  # moves with the scene: static
        inst[30:40, 10 + 5 * t:20 + 5 * t] = PERSON * 1000 + 1  # 5 px / frame: 3 after correction
        if t == 10:
            inst[45:55, 100:110] = CAR * 1000 + 7  # appears on the last labelled frame
        frames.append(ob.frame_objects(inst, min_area=1))
    cols = ob.clip_objects("c", labeled, frames, H)
    car = (cols["id"] == CAR * 1000 + 1)
    person = (cols["id"] == PERSON * 1000 + 1)
    np.testing.assert_allclose(cols["motion"][car], 0.0, atol=1e-9)
    np.testing.assert_allclose(cols["motion"][person], 3.0)
    np.testing.assert_allclose(cols["raw_shift"][person], 5.0)
    assert cols["matched"][person].tolist() == [1, 2, 1]
    new = cols["id"] == CAR * 1000 + 7
    assert cols["status"][new].tolist() == [1] and math.isnan(cols["motion"][new][0])
    assert list(cols["clip"]) == ["c"] * len(cols["t"])
    assert cols["t"].tolist() == sorted(cols["t"].tolist())

    support = np.ones(T, int)
    support[3] = 0  # failed fit between labelled frames 0 and 5
    cols = ob.clip_objects("c", labeled, frames, H, support)
    first = (cols["t"] == 0) & person
    assert cols["status"][first].tolist() == [2]  # its only pair is unusable
    mid = (cols["t"] == 5) & person
    assert cols["status"][mid].tolist() == [0]  # the forward pair still measures it
    np.testing.assert_allclose(cols["motion"][mid], 3.0)


def test_bins():
    bins = ob.DEFAULT_BINS
    assert ob.size_bins(np.array([10, 1024, 5000, 20000]), bins).tolist() == ["tiny", "small", "medium", "large"]
    names = ob.motion_bins(np.array([0.5, 3.0, 9.0, np.nan, np.nan]), np.array([0, 0, 0, 1, 2]))
    assert names.tolist() == ["static", "slow", "fast", "new", "unknown"]
    with pytest.raises(ValueError):
        ob.size_bins(np.array([1]), {"size_edges": [1, 2]})


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #
def test_recall_table_and_bootstrap():
    clip = np.array(["a"] * 4 + ["b"] * 4)
    found = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    out = ob.recall_table(found, clip, {"all": np.ones(8, bool), "none": np.zeros(8, bool)}, n_boot=200)
    assert out["all"]["recall"] == 0.5 and out["all"]["objects"] == 8
    lo, hi = out["all"]["ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0 and lo < hi
    assert math.isnan(out["none"]["recall"])


def test_recall_at_equal_latency():
    clip = np.array(["a", "a", "b", "b"])
    mask = np.ones(4, bool)
    ff = [np.array([0, 0, 0, 0], bool), np.array([1, 1, 1, 1], bool)]  # recall 0 at 10 ms, 1 at 20 ms
    out = ob.recall_at_equal_latency(np.array([1, 1, 1, 0], bool), 15.0, ff, [10.0, 20.0], clip, mask, n_boot=50)
    assert out["recall"] == 0.75
    assert out["delta_recall_at_equal_latency"] == pytest.approx(0.25)
    out = ob.recall_at_equal_latency(np.ones(4, bool), 30.0, ff, [10.0, 20.0], clip, mask, n_boot=10)
    assert math.isnan(out["delta_recall_at_equal_latency"]) and out["nan_resamples"] == 10


# --------------------------------------------------------------------------- #
# CLI on the fixture: Method 3 caches built with the oracle backbone
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


class _WithInstances:
    """The val fixture with a ``panoptic_inst``-style map: one object per thing class."""

    def __init__(self, vps):
        self.vps = vps

    def __len__(self):
        return len(self.vps)

    def __getattr__(self, name):
        return getattr(self.vps, name)

    def annotation_ids(self, i, t):
        lab = self.vps.load_label(i, t).astype(np.int64)
        return np.where(np.isin(lab, ob.THING_CLASSES), lab * 1000 + 1, lab)


@pytest.fixture(scope="module")
def m3_caches(tmp_path_factory, vps, oracle_backbone):
    pytest.importorskip("cv2")
    from reuse.replay import build_clip_cache

    root = tmp_path_factory.mktemp("objcache")
    out = root / "val" / "dummy"
    for i in range(len(vps)):
        build_clip_cache(vps, i, oracle_backbone, out, full_scales=(1.0, 0.5), glimpse_sizes=(12,),
                         ctx_scales=(0.25,), device="cpu", margin=2, allow_any_frame_size=True)
    return root


def test_cli_table_score_report(vps, m3_caches, monkeypatch):
    from scripts import objects as so

    monkeypatch.setattr(ob, "MIN_AREA", 1)
    data = _WithInstances(vps)
    clips = list(range(len(data)))
    cols = so.build_table(data, clips, so._caches(m3_caches, "val", "dummy", data, clips))
    assert len(cols["t"]) > 0 and set(cols["cls"].tolist()) <= set(ob.THING_CLASSES)
    summary = so.table_summary(cols, None)
    assert summary["objects"] == len(cols["t"])

    m3 = {"name": "3a/f1/g12/t2/r0/u1", "miou": 0.9,
          "config": {"full_scale": 1.0, "glimpse_size": 12, "theta": 2, "rho": 0.0, "tau": 1.0, "ctx_scale": None,
                     "margin": 2, "homography": "orb"},
          "judges": {"mean": {"latency_ms": 7.0, "delta_miou_at_equal_latency": 0.0}}}
    m2 = {"name": "4o-N/c0.25/g12/k2", "miou": 0.5,
          "config": {"ctx_scale": 0.25, "glimpse_size": 12, "variant": "off", "a_max": None, "k": 2, "gamma": None,
                     "margin": 2},
          "judges": {"mean": {"latency_ms": 6.0, "delta_miou_at_equal_latency": float("nan")}}}
    scores = {m3["name"]: m3, m2["name"]: m2}
    fullframe = {"scales": [1.0, 0.5], "median_ms": [10.0, 5.0]}

    ious = {}
    for i, cache in so._caches(m3_caches, "val", "dummy", data, clips):
        maps, objs = so.clip_ground_truth(data, i, cache.labeled_t)
        for name in ("ff/1", "ff/0.5", m3["name"], m2["name"]):
            ious.setdefault(name, []).extend(so.clip_ious(cache, maps, objs, name, scores, "cpu"))
    ious = {k: np.asarray(v) for k, v in ious.items()}
    assert all(len(v) == len(cols["t"]) for v in ious.values())
    np.testing.assert_allclose(ious["ff/1"], 1.0)  # the oracle backbone is perfect at full resolution

    rep = so.build_report(cols, ious, scores, fullframe, None, n_boot=20)
    assert [e["name"] for e in rep["screen"]] == [m2["name"]]  # the Method 2 stand-in stays out of the table
    assert {e["name"] for e in rep["configs"]} == {"ff/1", "ff/0.5", m3["name"]}
    full = next(e for e in rep["configs"] if e["name"] == "ff/1")
    assert full["recall"]["all"]["recall"] == 1.0


def test_upper_edge_and_families():
    from scripts import objects as so

    cfgs = [{"name": f"3a/{i}", "miou": m, "judges": {"mean": {"latency_ms": l, "delta_miou_at_equal_latency": d}}}
            for i, (l, m, d) in enumerate([(5, 0.5, -0.1), (6, 0.4, -0.2), (7, 0.6, -0.05), (9, 0.7, float("nan"))])]
    assert so.upper_edge(cfgs, 10) == ["3a/0", "3a/2", "3a/3"]
    assert so.upper_edge(cfgs, 1) == ["3a/0", "3a/2"]  # + the best reading
    assert so._family("4o-N/c0.5/g384/k4") == "method2_screen"
    assert so._family("4o-A/c0.5/g384/ainf/k4") == "method4"
    assert so._family("ff/0.75") == "fullframe"
