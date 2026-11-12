"""Method 4 phase 1: the canvas replay under the oracle selector (CPU fixture)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from canvas.replay import CanvasConfig, replay_canvas, replay_canvas_many  # noqa: E402
from reuse.replay import ClipCache, build_clip_cache  # noqa: E402

GEOM = dict(glimpse_size=12, margin=2)  # GlimpseGrid((16, 32), 12, 2): inner 8 -> 2 x 4 cells
CELLS = 8
CACHE_KW = dict(full_scales=(1.0,), glimpse_sizes=(12,), ctx_scales=(0.25, 0.5), device="cpu", margin=2,
                allow_any_frame_size=True, homography="orb")  # IC-LK needs frames larger than the fixture's
ORB = dict(homography="orb")
SELECTORS = [dict(k=1), dict(k=3), dict(k=CELLS), dict(gamma=0.0), dict(gamma=0.2)]


def _group(variant):
    ages = [None] if variant == "off" else [1, 2, None]
    return [CanvasConfig(ctx_scale=c, variant=variant, a_max=a, **sel, **GEOM)
            for c, a, sel in itertools.product((0.25, 0.5), ages, SELECTORS)]


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
    root = tmp_path_factory.mktemp("m4cache")
    return [ClipCache(build_clip_cache(vps, i, oracle_backbone, root, **CACHE_KW)) for i in range(2)]


@pytest.fixture(scope="module")
def static_cache(tmp_path_factory, vps, oracle_backbone):
    root = tmp_path_factory.mktemp("m4static")
    return ClipCache(build_clip_cache(_Static(vps), 0, oracle_backbone, root, **CACHE_KW))


def test_names_are_unique_and_stable():
    cfgs = [c for v in ("off", "iclk", "none") for c in _group(v)]
    assert len({c.name() for c in cfgs}) == len(cfgs)
    assert CanvasConfig(0.25, 384, "iclk", a_max=4, k=2).name() == "4o-A/c0.25/g384/a4/k2"
    assert CanvasConfig(0.5, 256, "none", gamma=0.02).name() == "4o-B/c0.5/g256/ainf/y0.02"
    assert CanvasConfig(0.375, 512, "off", k=8).name() == "4o-N/c0.375/g512/k8"


@pytest.mark.parametrize("bad", [dict(variant="orb", k=1), dict(variant="off", k=1, gamma=0.1), dict(variant="off"),
                                 dict(variant="off", k=1, a_max=2), dict(variant="none", k=1, a_max=0)])
def test_config_refuses_nonsense(bad):
    with pytest.raises(ValueError):
        CanvasConfig(ctx_scale=0.25, **GEOM, **bad)


@pytest.mark.parametrize("variant", ["off", "iclk", "none"])
def test_batched_replay_equals_the_reference(caches, variant):
    group = _group(variant)
    for cache in caches:
        many, many_preds = replay_canvas_many(cache, group, "cpu", return_predictions=True, **ORB)
        for cfg, res, preds in zip(group, many, many_preds):
            ref, ref_preds = replay_canvas(cache, cfg, "cpu", return_predictions=True, **ORB)
            assert len(preds) == cache.num_frames
            for t, (a, b) in enumerate(zip(preds, ref_preds)):
                assert torch.equal(a, b), (cfg.name(), t)
            assert np.array_equal(res.conf, ref.conf) and np.array_equal(res.k, ref.k), cfg.name()
            assert res.canvas_share == pytest.approx(ref.canvas_share, abs=1e-6)
            assert res.num_cells == ref.num_cells == CELLS
            assert res.conf.shape == (len(cache.labeled_t), 19, 19) and res.conf.dtype == np.int64


def test_the_canvas_changes_some_prediction(caches):
    """Guards the equality test above against a canvas that is never read."""
    used = 0.0
    for cache in caches:
        for variant in ("iclk", "none"):
            used += sum(float(r.canvas_share.sum()) for r in replay_canvas_many(cache, _group(variant), "cpu", **ORB))
    assert used > 0


def test_batched_replay_refuses_a_mixed_group(caches):
    with pytest.raises(ValueError, match="share"):
        replay_canvas_many(caches[0], [_group("none")[0], _group("iclk")[0]], "cpu")


def test_every_cell_gives_the_every_cell_glimpse(caches):
    cache = caches[0]
    for variant in ("off", "none"):
        cfg = CanvasConfig(ctx_scale=0.25, variant=variant, k=CELLS, **GEOM)
        res, preds = replay_canvas(cache, cfg, "cpu", return_predictions=True)
        assert res.k.tolist() == [CELLS] * cache.num_frames
        for t, p in enumerate(preds):
            assert torch.equal(p, torch.from_numpy(cache["cells_12"][t].astype(np.int64)))


def test_k_zero_is_the_context_and_has_no_canvas(caches):
    cache = caches[0]
    res, preds = replay_canvas(cache, CanvasConfig(ctx_scale=0.25, variant="none", k=0, **GEOM), "cpu",
                               return_predictions=True)
    assert not res.k.any() and not res.canvas_share.any()
    for t, p in enumerate(preds):
        assert torch.equal(p, torch.from_numpy(cache["ctx_0.25"][t].astype(np.int64)))


def test_variant_n_never_reads_a_canvas(caches):
    res = replay_canvas(caches[0], CanvasConfig(ctx_scale=0.25, variant="off", k=2, **GEOM), "cpu")
    assert not res.canvas_share.any()


def test_fixed_k_picks_the_cells_of_highest_gain_on_the_first_frame(caches):
    """Frame 0 has an empty canvas, so the base is the context: check the choice by brute force."""
    for cache in caches:
        ctx, cells, teacher = (cache[k][0].astype(np.int64) for k in ("ctx_0.25", "cells_12", "full_1"))
        _, preds = replay_canvas(cache, CanvasConfig(ctx_scale=0.25, variant="none", k=2, **GEOM), "cpu",
                                 return_predictions=True)
        got = int((preds[0].numpy() == teacher).sum())
        best = 0
        for pair in itertools.combinations(range(CELLS), 2):
            out = ctx.copy()
            for c in pair:
                r, col = divmod(c, 4)
                out[r * 8:(r + 1) * 8, col * 8:(col + 1) * 8] = cells[r * 8:(r + 1) * 8, col * 8:(col + 1) * 8]
            best = max(best, int((out == teacher).sum()))
        assert got == best


def test_adaptive_k_only_glimpses_cells_above_the_threshold(caches):
    cache = caches[0]
    none = replay_canvas(cache, CanvasConfig(ctx_scale=0.25, variant="off", gamma=1.0, **GEOM), "cpu")
    assert not none.k.any()  # no cell can gain more than all of its pixels
    some = replay_canvas(cache, CanvasConfig(ctx_scale=0.25, variant="off", gamma=0.0, **GEOM), "cpu")
    ctx, cells, teacher = (cache[k][0].astype(np.int64) for k in ("ctx_0.25", "cells_12", "full_1"))
    gains = [int((cells[r * 8:r * 8 + 8, c * 8:c * 8 + 8] == teacher[r * 8:r * 8 + 8, c * 8:c * 8 + 8]).sum())
             - int((ctx[r * 8:r * 8 + 8, c * 8:c * 8 + 8] == teacher[r * 8:r * 8 + 8, c * 8:c * 8 + 8]).sum())
             for r in range(2) for c in range(4)]
    assert some.k[0] == sum(g > 0 for g in gains) and 0 < some.k[0] <= CELLS


def _canvas_shares(cache, a_max, variant="none"):
    """Adaptive selector at gamma 0: every cell with something to gain is glimpsed."""
    return replay_canvas(cache, CanvasConfig(ctx_scale=0.25, variant=variant, a_max=a_max, gamma=0.0, **GEOM), "cpu", **ORB)


def test_a_static_clip_reuses_its_glimpses_until_they_are_too_old(static_cache):
    """Static clip, every frame the same. Frame 0 has an empty canvas. Later the canvas serves the
    pixels where it is more confident than the context, so fewer cells have anything to gain; the
    context still wins where it is at least as confident, so some cells are glimpsed again. With
    a_max = 1, the cells not glimpsed again on frame 1 are 2 frames old on frame 2 and are ignored."""
    forever = _canvas_shares(static_cache, None)
    assert forever.canvas_share[0] == 0 and (forever.canvas_share[1:] > 0).all()
    assert 0 < forever.k[1] < forever.k[0]
    assert len(set(forever.k[1:].tolist())) == 1 and len(set(forever.canvas_share[1:].tolist())) == 1  # steady
    short = _canvas_shares(static_cache, 1)
    assert short.k[:2].tolist() == forever.k[:2].tolist()
    assert short.canvas_share[1] == forever.canvas_share[1]  # age 1: still used
    assert 0 < short.canvas_share[2] < forever.canvas_share[2]  # only frame 1's glimpses are young enough
    assert short.k[2] > forever.k[2]  # the expired cells have to be glimpsed again


def test_the_identity_homography_makes_a_and_b_agree(static_cache):
    """A static clip's homographies are (close to) the identity: A must not lose the canvas."""
    a, b = _canvas_shares(static_cache, None, "iclk"), _canvas_shares(static_cache, None, "none")
    H = static_cache["H"]
    if np.allclose(H, np.eye(3)[None], atol=1e-6):
        assert np.array_equal(a.k, b.k) and np.array_equal(a.conf, b.conf)
    assert (a.canvas_share[1:] > 0).all()


def test_variant_a_refuses_a_cache_built_with_another_homography(tmp_path, vps, oracle_backbone):
    cache = ClipCache(build_clip_cache(vps, 0, oracle_backbone, tmp_path, **CACHE_KW))  # built with ORB
    cfg = CanvasConfig(ctx_scale=0.25, variant="iclk", k=1, **GEOM)
    for fn, arg in ((replay_canvas, cfg), (replay_canvas_many, [cfg])):
        with pytest.raises(ValueError, match="homography 'iclk'"):
            fn(cache, arg, "cpu")  # the default: the estimator whose T4 cost is counted
    replay_canvas(cache, CanvasConfig(ctx_scale=0.25, variant="none", k=1, **GEOM), "cpu")  # B does not warp


def test_a_wrong_margin_or_k_is_refused(caches):
    with pytest.raises(ValueError, match="margin"):
        replay_canvas(caches[0], CanvasConfig(ctx_scale=0.25, glimpse_size=12, margin=1, variant="off", k=1), "cpu")
    with pytest.raises(ValueError, match="outside"):
        replay_canvas(caches[0], CanvasConfig(ctx_scale=0.25, variant="off", k=CELLS + 1, **GEOM), "cpu")
