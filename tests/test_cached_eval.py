"""Cached evaluation of glimpse selections: exact against the pipeline (fixture CPU; real data on GPU)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import generate_fixture as gf  # noqa: E402
from eval.harness import evaluate  # noqa: E402
from eval.metrics import ConfusionMatrix  # noqa: E402
from foveation.cache import build_cache  # noqa: E402
from foveation.cached_eval import CachedPair, Rule, compose, miou_of, prior_order, select_cells  # noqa: E402
from foveation.method import FoveatedMethod  # noqa: E402
from foveation.ops import GlimpseGrid  # noqa: E402
from foveation.pipeline import FoveationConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pair(tmp_path_factory, vps, oracle_backbone):
    root = tmp_path_factory.mktemp("cache")
    grid = GlimpseGrid((gf.OH, gf.OW), 12, 2)
    ctx = build_cache(vps, "context", oracle_backbone, "dummy", grid, root / "ctx.npz", "cpu", scale=0.25,
                      allow_any_frame_size=True)
    gl = build_cache(vps, "glimpse", oracle_backbone, "dummy", grid, root / "gl.npz", "cpu", allow_any_frame_size=True)
    return CachedPair.load(ctx, gl)


def _pipeline_total(vps, backbone, cfg):
    return evaluate(vps, FoveatedMethod(backbone, backbone, cfg))["confusion"].matrix


def test_compose_without_glimpses_is_the_context_prediction(pair, vps, oracle_backbone):
    none = torch.zeros(pair.num_frames, pair.grid.num_cells, dtype=torch.bool)
    per_clip = compose(pair, none)
    assert per_clip.shape == (len(pair.clip_ids), 19, 19)
    assert np.array_equal(per_clip.sum(0), pair.ctx_conf.sum((0, 1)))
    assert miou_of(per_clip.sum(0)) < 1.0


def test_every_cell_matches_the_tiled_pipeline(pair, vps, oracle_backbone):
    mask = select_cells(pair, Rule("all"), 0)
    assert mask.all()
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=12, glimpse_margin=2, k=pair.grid.num_cells)
    assert np.array_equal(compose(pair, mask).sum(0), _pipeline_total(vps, oracle_backbone, cfg))


@pytest.mark.parametrize("selection,score,pool", [("topk", "entropy", "mean"), ("greedy", "margin", "top10")])
def test_uncertainty_selection_matches_the_pipeline(pair, vps, oracle_backbone, selection, score, pool):
    rule = Rule("uncertainty", score=score, pool=pool, selection=selection)
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=12, glimpse_margin=2, k=2, selection=selection,
                          score=score, pool=pool)
    assert np.array_equal(compose(pair, select_cells(pair, rule, 2)).sum(0), _pipeline_total(vps, oracle_backbone, cfg))


def test_oracle_gains_the_most_correct_pixels(pair):
    def correct(rule, k):
        return np.trace(compose(pair, select_cells(pair, rule, k, prior_order=prior_order(pair))).sum(0))

    for k in (1, 2, 4):
        best = correct(Rule("oracle"), k)
        for rule in (Rule("uncertainty"), Rule("random", seed=3), Rule("prior")):
            assert best >= correct(rule, k)
        assert select_cells(pair, Rule("oracle"), k).sum(1).eq(k).all()


def test_random_selection_is_reproducible(pair):
    a = select_cells(pair, Rule("random", seed=1), 2)
    assert torch.equal(a, select_cells(pair, Rule("random", seed=1), 2))
    assert a.sum(1).eq(2).all()


def test_miou_of_matches_the_confusion_matrix(pair):
    m = compose(pair, select_cells(pair, Rule("uncertainty"), 1)).sum(0)
    cm = ConfusionMatrix(19)
    cm.matrix = m
    assert miou_of(m) == pytest.approx(cm.compute()["miou"])


def test_rule_names_are_stable():
    assert Rule("uncertainty", score="margin", pool="top10", selection="greedy").name() == "uncertainty/margin/top10/greedy"
    assert Rule("random", seed=3).name() == "random/seed3"
    assert Rule("oracle").name() == "oracle"


# --------------------------------------------------------------------------- #
# Real data: the cache against FoveatedSegmenter on the first 2 VPS val clips
# --------------------------------------------------------------------------- #
REAL = (torch.cuda.is_available() and os.environ.get("CITYSCAPES_VPS_ROOT") and os.environ.get("PIDNET_S_WEIGHTS"))


@pytest.mark.skipif(not REAL, reason="needs CUDA, $CITYSCAPES_VPS_ROOT and $PIDNET_S_WEIGHTS")
@pytest.mark.parametrize("ctx_name,selection", [("pidnet_s", "topk"), ("pidnet_s", "greedy"), ("bisenetv2", "topk")])
def test_cache_matches_the_pipeline_on_real_data(tmp_path, ctx_name, selection):
    from scripts.common import expand_env, load_yaml

    from data.registry import build_dataset
    from eval.harness import frame_to_tensor
    from models.registry import build_backbone

    def backbone(name):
        cfg = expand_env(load_yaml(REPO / "configs" / f"baseline_fullframe_{name}.yaml"))
        return build_backbone(cfg["backbone"]).cuda().eval(), cfg

    gl_bb, cfg = backbone("pidnet_s")
    ctx_bb = gl_bb if ctx_name == "pidnet_s" else backbone("bisenetv2")[0]
    cfg["dataset"]["split"] = "val"
    dataset = build_dataset(cfg["dataset"], base_dir=REPO)
    grid = GlimpseGrid((1024, 2048), 384, 32)
    clips = [0, 1]
    ctx = build_cache(dataset, "context", ctx_bb, ctx_name, grid, tmp_path / "ctx.npz", "cuda", scale=0.25,
                      clip_indices=clips)
    gl = build_cache(dataset, "glimpse", gl_bb, "pidnet_s", grid, tmp_path / "gl.npz", "cuda", clip_indices=clips)
    pair = CachedPair.load(ctx, gl)
    rule = Rule("uncertainty", selection=selection)
    mask = select_cells(pair, rule, 4)
    cached = compose(pair, mask)

    fcfg = FoveationConfig(context_scale=0.25, glimpse_size=384, glimpse_margin=32, k=4, selection=selection)
    method = FoveatedMethod(ctx_bb, gl_bb, fcfg)
    for n, (ci, t) in enumerate(zip(pair.clip_index, pair.t)):
        frame = frame_to_tensor(dataset.load_frame(int(ci), int(t)), "cuda")
        labels, idx = method.predict_with_idx(frame)
        assert set(idx.tolist()) == set(torch.nonzero(mask[n]).flatten().tolist()), f"frame {n}: selection differs"
        cm = ConfusionMatrix(19, 255)
        cm.update(labels.cpu().numpy(), dataset.load_label(int(ci), int(t)))
        frame_cached = pair.ctx_conf[n].sum(0) + ((pair.gl_conf[n] - pair.ctx_conf[n]) * mask[n].numpy()[:, None, None]).sum(0)
        diff = np.abs(frame_cached.astype(np.int64) - cm.matrix).sum()
        assert diff <= 1e-4 * cm.matrix.sum(), f"frame {n}: {diff} pixels differ"
    total = cached.sum(0)
    assert total.sum() > 0
