"""Per-cell confusion caches of the context and glimpse passes (CPU, fixture)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import generate_fixture as gf  # noqa: E402
from eval.harness import frame_to_tensor  # noqa: E402
from eval.metrics import ConfusionMatrix  # noqa: E402
from foveation.cache import build_cache, cache_path, context_cells, glimpse_cells, per_cell_confusion  # noqa: E402
from foveation.ops import SCORE_KINDS, GlimpseGrid  # noqa: E402


def test_per_cell_confusion_sums_to_the_frame_confusion():
    g = torch.Generator().manual_seed(0)
    grid = GlimpseGrid((100, 230), 48, 8)  # rows/cols with padding at the bottom/right
    pred = torch.randint(0, 5, (100, 230), generator=g)
    label = torch.randint(0, 5, (100, 230), generator=g)
    label[:10] = 255
    cells = per_cell_confusion(pred, label, grid, 5, 255)
    assert cells.shape == (grid.num_cells, 5, 5) and cells.dtype == torch.int32
    cm = ConfusionMatrix(5, 255)
    cm.update(pred.numpy(), label.numpy())
    assert np.array_equal(cells.sum(0).numpy(), cm.matrix)
    assert (cells >= 0).all()


def test_cells_in_padding_or_all_ignore_are_zero():
    grid = GlimpseGrid((100, 230), 48, 8)  # inner 32
    label = torch.full((100, 230), 255)
    label[:32, :32] = 1
    cells = per_cell_confusion(torch.ones(100, 230, dtype=torch.long), label, grid, 5, 255)
    assert cells[0].sum() == 32 * 32 and cells[1:].sum() == 0


def test_context_and_glimpse_cells_of_a_perfect_backbone(vps, oracle_backbone):
    grid = GlimpseGrid((gf.OH, gf.OW), 12, 2)
    t = int(np.flatnonzero(vps.labeled_mask(0))[0])
    frame = frame_to_tensor(vps.load_frame(0, t), "cpu")
    label = torch.from_numpy(np.asarray(vps.load_label(0, t), dtype=np.int64))
    gl = glimpse_cells(oracle_backbone, frame, label, grid, 19, 255)
    total = gl.sum(0)
    assert total.sum() == int((label != 255).sum()) and total.sum() == torch.trace(total)
    ctx = context_cells(oracle_backbone, frame, label, 0.25, grid, 19, 255)
    assert ctx.conf.sum() == total.sum() and torch.trace(ctx.conf.sum(0)) < total.sum()
    assert set(ctx.class_sums) == set(SCORE_KINDS) and ctx.class_sums["entropy"].shape == (grid.num_cells, 19)
    assert ctx.top10["margin"].shape == (grid.num_cells,) and ctx.q >= 1


def test_cache_files_hold_every_labelled_frame(tmp_path, vps, oracle_backbone):
    grid = GlimpseGrid((gf.OH, gf.OW), 12, 2)
    p = build_cache(vps, "glimpse", oracle_backbone, "dummy", grid, tmp_path / "gl.npz", "cpu",
                    allow_any_frame_size=True)
    d = np.load(p)
    assert d["conf"].shape == (vps.num_labeled_frames(), grid.num_cells, 19, 19) and d["conf"].dtype == np.int32
    conf = d["conf"].sum((0, 1))
    assert conf.sum() > 0 and conf.sum() == np.trace(conf)
    meta = json.loads(str(d["meta"]))
    assert (meta["kind"], meta["size"], meta["margin"], meta["rows"], meta["cols"]) == ("glimpse", 12, 2, 2, 4)
    c = build_cache(vps, "context", oracle_backbone, "dummy", grid, tmp_path / "ctx.npz", "cpu", scale=0.25,
                    allow_any_frame_size=True)
    dc = np.load(c)
    assert dc["class_sums_entropy"].shape == (vps.num_labeled_frames(), grid.num_cells, 19)
    assert np.array_equal(dc["clip_index"], d["clip_index"]) and np.array_equal(dc["t"], d["t"])
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_cache_refuses_other_frame_sizes(tmp_path, vps, oracle_backbone):
    with pytest.raises(ValueError, match="1024x2048"):
        build_cache(vps, "glimpse", oracle_backbone, "dummy", GlimpseGrid((1024, 2048), 384, 32),
                    tmp_path / "x.npz", "cpu")


def test_cache_paths():
    grid = GlimpseGrid((1024, 2048), 384, 32)
    root = __import__("pathlib").Path("c")
    assert cache_path(root, "train", "context", "pidnet_s", grid, 0.25).as_posix() == "c/train/ctx_pidnet_s_s0.25_g384m32.npz"
    assert cache_path(root, "val", "glimpse", "bisenetv2", grid).as_posix() == "c/val/gl_bisenetv2_g384m32.npz"
