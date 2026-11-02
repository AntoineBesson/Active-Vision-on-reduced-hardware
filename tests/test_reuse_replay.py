"""Per-frame caches and the exact replay of Method 3 (CPU fixture; one CUDA + real-data check)."""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from eval.cost import CostLedger  # noqa: E402
from eval.harness import frame_to_tensor  # noqa: E402
from reuse import replay as rp  # noqa: E402
from reuse.method import MotionReuseConfig, MotionReuseMethod  # noqa: E402
from reuse.replay import ClipCache, build_clip_cache, replay_clip, reuse_bound  # noqa: E402

GEOM = dict(glimpse_size=12, margin=2)
CACHE_KW = dict(full_scales=(1.0, 0.5), glimpse_sizes=(12,), ctx_scales=(0.25,), device="cpu", margin=2,
                allow_any_frame_size=True)
CONFIGS = [
    MotionReuseConfig(full_scale=1.0, theta=2, rho=0.0, tau=1.0, **GEOM),  # 3a
    MotionReuseConfig(full_scale=0.5, theta=2, rho=0.0, tau=1.0, ctx_scale=0.25, **GEOM),  # 3b
    MotionReuseConfig(full_scale=1.0, theta=2, rho=0.0, tau=0.5, **GEOM),  # falls back on some frames only
]


class _Static:
    """Every frame of clip 0 of ``vps`` is its frame 0; labels as in ``vps``."""

    def __init__(self, vps):
        self.vps = vps

    def __len__(self):
        return 1

    def clip_id(self, i):
        return "static"

    def num_frames(self, i):
        return self.vps.num_frames(0)

    def labeled_mask(self, i):
        return self.vps.labeled_mask(0)

    def load_frame(self, i, t):
        return self.vps.load_frame(0, 0)

    def load_label(self, i, t):
        return self.vps.load_label(0, t)


@pytest.fixture(scope="module")
def caches(tmp_path_factory, vps, oracle_backbone):
    root = tmp_path_factory.mktemp("m3cache")
    return [ClipCache.load(build_clip_cache(vps, i, oracle_backbone, root, **CACHE_KW)) for i in range(2)]


def _method_preds(vps, i, backbone, cfg):
    m = MotionReuseMethod(backbone, cfg, allow_any_frame_size=True)
    m.reset({})
    out, moved = [], []
    for t in range(vps.num_frames(i)):
        ledger = CostLedger()
        ledger.begin_frame(16 * 32)
        out.append(m.predict(frame_to_tensor(vps.load_frame(i, t), "cpu"), t, ledger))
        moved.append(m.last_step["moved_cells"])
    return out, moved


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name())
def test_replay_matches_the_method_pixel_for_pixel(caches, vps, oracle_backbone, cfg):
    for i, cache in enumerate(caches):
        res, preds = replay_clip(cache, cfg, "cpu", return_predictions=True)
        ref, moved = _method_preds(vps, i, oracle_backbone, cfg)
        assert len(preds) == len(ref)
        for t, (a, b) in enumerate(zip(preds, ref)):
            assert torch.equal(a, b), (cfg.name(), i, t)
        assert res.moved_cells.tolist() == moved
        if cfg.tau < 1:
            assert 0 < res.fallback.sum() < len(preds) - 1  # a mix of fallback and reuse frames
        else:
            assert any(0 < m < res.num_cells for m in moved[1:])  # some cells, not all, recomputed
        assert res.conf.shape == (len(cache.labeled_t), 19, 19) and res.conf.dtype == np.int64


def test_cache_holds_every_frame(caches, vps):
    c = caches[0]
    assert c.meta["clip_id"] == vps.clip_id(0) and c.num_frames == vps.num_frames(0)
    assert c["H"].shape == (c.num_frames, 3, 3) and np.array_equal(c["H"][0], np.eye(3))
    assert c["full_1"].shape == (c.num_frames, 16, 32) and c["full_conf_1"].shape == (c.num_frames, 4, 8)
    assert c["cells_12"].shape == (c.num_frames, 16, 32) and c["motion_raw"].dtype == np.uint8
    assert list(c.labeled_t) == list(np.flatnonzero(vps.labeled_mask(0)))


def test_cache_write_is_atomic(tmp_path, vps, oracle_backbone, monkeypatch):
    real, calls = rp.glimpse_pass, {"n": 0}

    def failing(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated cuDNN failure")
        return real(*a, **kw)

    monkeypatch.setattr(rp, "glimpse_pass", failing)
    with pytest.raises(RuntimeError, match="simulated"):
        build_clip_cache(vps, 0, oracle_backbone, tmp_path, **CACHE_KW)
    assert not list(tmp_path.glob("*.npz"))
    monkeypatch.setattr(rp, "glimpse_pass", real)
    path = build_clip_cache(vps, 0, oracle_backbone, tmp_path, **CACHE_KW)
    assert path.is_file() and sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_oracle_beats_the_motion_rule_at_equal_k_on_the_first_step(caches):
    cfg = CONFIGS[0]
    for cache in caches:
        rule, rule_preds = replay_clip(cache, cfg, "cpu", return_predictions=True)
        oracle, oracle_preds = replay_clip(cache, cfg, "cpu", selector="oracle", oracle_k=rule.moved_cells,
                                           return_predictions=True)
        teacher = torch.from_numpy(cache["full_1"][1].astype(np.int64))
        assert int((oracle_preds[1] == teacher).sum()) >= int((rule_preds[1] == teacher).sum())
        assert oracle.moved_cells[1] == rule.moved_cells[1]


def test_reuse_bound_of_a_static_clip_equals_the_full_pass(tmp_path, vps, oracle_backbone):
    cache = ClipCache.load(build_clip_cache(_Static(vps), 0, oracle_backbone, tmp_path, **CACHE_KW))
    bound = reuse_bound(cache, 1.0, "cpu")
    assert bound.shape == (cache.num_frames - 1, 19, 19)
    full0 = cache["full_1"][0].astype(np.int64)
    for lag in range(1, cache.num_frames):
        expected = np.zeros((19, 19), np.int64)
        for j, t in enumerate(cache.labeled_t):
            if t - lag >= 0:
                lab = cache["labels"][j].astype(np.int64)
                ok = lab != 255
                np.add.at(expected, (lab[ok], full0[ok]), 1)
        assert np.array_equal(bound[lag - 1], expected), lag


@pytest.mark.skipif(not (torch.cuda.is_available() and os.environ.get("CITYSCAPES_VPS_ROOT")
                         and os.environ.get("PIDNET_S_WEIGHTS")), reason="needs CUDA, $CITYSCAPES_VPS_ROOT and $PIDNET_S_WEIGHTS")
def test_replay_matches_the_method_on_real_val_clips(tmp_path):
    from pathlib import Path

    from data.cityscapes_vps import CityscapesVPS, CityscapesVPSConfig
    from models.registry import build_backbone

    repo = Path(__file__).resolve().parents[1]
    split = os.environ.get("M3_EQUIV_SPLIT", "val")  # the Colab notebook passes the split it has on disk
    ds = CityscapesVPS(CityscapesVPSConfig.from_yaml(repo / "configs" / "cityscapes_vps.yaml", split=split))
    import yaml

    bcfg = yaml.safe_load((repo / "configs" / "baseline_fullframe_pidnet_s.yaml").read_text())["backbone"]
    bcfg["weights"] = os.environ["PIDNET_S_WEIGHTS"]
    model = build_backbone(bcfg).cuda().eval()
    cfgs = [MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.05, tau=1.0, homography="iclk"),
            MotionReuseConfig(full_scale=0.75, glimpse_size=384, theta=20, rho=0.05, tau=1.0, ctx_scale=0.25,
                              homography="iclk"),
            MotionReuseConfig(full_scale=1.0, glimpse_size=384, theta=20, rho=0.02, tau=0.2, homography="iclk")]
    for i in range(2):
        cache = ClipCache.load(build_clip_cache(ds, i, model, tmp_path, full_scales=(1.0, 0.75), glimpse_sizes=(384,),
                                                ctx_scales=(0.25,), device="cuda", homography="iclk"))
        for cfg in cfgs:
            _, preds = replay_clip(cache, cfg, "cuda", return_predictions=True)
            m = MotionReuseMethod(model, cfg)
            m.reset({})
            for t in range(ds.num_frames(i)):
                ledger = CostLedger()
                ledger.begin_frame(1024 * 2048)
                ref = m.predict(frame_to_tensor(ds.load_frame(i, t), "cuda"), t, ledger)
                assert torch.equal(preds[t].cuda(), ref), (cfg.name(), i, t)
