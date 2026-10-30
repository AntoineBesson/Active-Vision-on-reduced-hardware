"""FoveatedMethod: the context + glimpses segmenter as a FrameMethod of the harness (CPU, fixture)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import generate_fixture as gf  # noqa: E402
from eval.harness import FrameMethod, evaluate, frame_to_tensor  # noqa: E402
from foveation.method import FoveatedMethod  # noqa: E402
from foveation.ops import GlimpseGrid  # noqa: E402
from foveation.pipeline import FoveatedSegmenter, FoveationConfig  # noqa: E402


def test_every_cell_glimpsed_is_exact_and_costs_are_split(vps, oracle_backbone):
    grid = GlimpseGrid((gf.OH, gf.OW), 12, 2)
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=12, glimpse_margin=2, k=grid.num_cells)
    m = FoveatedMethod(oracle_backbone, oracle_backbone, cfg)
    assert isinstance(m, FrameMethod)
    res = evaluate(vps, m)
    assert res["metrics"]["miou"] == pytest.approx(1.0)  # tiling + paste lose nothing
    comps = res["cost"]["components"]
    assert set(comps) == {"context", "glimpses", "policy", "recomposition"}
    assert comps["glimpses"]["pixels_per_frame"] == grid.num_cells * 12 * 12
    assert comps["context"]["pixels_per_frame"] == 4 * 8
    assert m.describe()["k"] == grid.num_cells and m.describe()["glimpse_model"] == "DummyBackbone"


def test_context_only_cells_lose_accuracy(vps, oracle_backbone):
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=12, glimpse_margin=2, k=1)
    assert evaluate(vps, FoveatedMethod(oracle_backbone, oracle_backbone, cfg))["metrics"]["miou"] < 1.0


def test_foveated_method_matches_the_segmenter(vps, oracle_backbone):
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=12, glimpse_margin=2, k=2)
    frame = frame_to_tensor(vps.load_frame(0, int(np.flatnonzero(vps.labeled_mask(0))[0])), "cpu")
    labels, idx = FoveatedMethod(oracle_backbone, oracle_backbone, cfg).predict_with_idx(frame)
    with torch.no_grad():
        ref_labels, ref_idx = FoveatedSegmenter(oracle_backbone, oracle_backbone, cfg, (gf.OH, gf.OW))(frame, return_idx=True)
    assert torch.equal(labels, ref_labels) and torch.equal(idx, ref_idx)


def test_latency_fns_run_one_frame(oracle_backbone):
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=12, glimpse_margin=2, k=2)
    fns = FoveatedMethod(oracle_backbone, oracle_backbone, cfg).latency_fns((gf.OH, gf.OW), torch.device("cpu"))
    assert set(fns) == {"end_to_end"} and fns["end_to_end"]().shape == (gf.OH, gf.OW)
