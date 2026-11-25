"""Method 11: the warped fine-tuning path (CPU, tiny tensors)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from finetune.cityscapes import IGNORE, to_train_ids  # noqa: E402
from finetune.train import RunConfig, freeze_bn, jitter, ohem_ce  # noqa: E402
from finetune.warped import WarpedForward, boundary_counts  # noqa: E402
from models.dummy import DummyBackbone  # noqa: E402
from models.utils import upsample_logits  # noqa: E402
from warp.policy import StepConfig, WarpStep  # noqa: E402


def _dummy(stride=1, align=False):
    m = DummyBackbone(19, output_stride=stride)
    m.align_corners = align
    return m.eval()


def test_label_ids_map_to_train_ids():
    a = np.array([[7, 8, 26], [0, 33, 255]], dtype=np.uint8)
    assert to_train_ids(a).tolist() == [[0, 1, 13], [IGNORE, 18, IGNORE]]


def test_boundary_counts_ignore_void_pairs():
    lab = torch.zeros(16, 16, dtype=torch.long)
    lab[:, 8:] = 1
    c = boundary_counts(lab, cell=8)
    assert c.tolist() == [[8, 8], [8, 8]]  # columns 7 and 8 are boundary pixels
    lab[:, 8:] = 255
    assert int(boundary_counts(lab, cell=8).sum()) == 0


@pytest.mark.parametrize("stride,align", [(1, False), (8, True)])
def test_uniform_forward_matches_evaluation_path(stride, align):
    """The training path's argmax is the evaluation's (WarpStep, r2u1) for the uniform grid and a prior."""
    model = _dummy(stride, align)
    frame = torch.rand(1, 3, 32, 64)
    prior = torch.rand(4, 8)
    for cfg, p in ((StepConfig(0.5, resample="r2", unwarp="u1"), None),
                   (StepConfig(0.5, "P", 0.25, 1.0, resample="r2", unwarp="u1"), prior)):
        fwd = WarpedForward(model, cfg, prior=p, cell=8)
        step = WarpStep(model, cfg, prior=p, cell=8)
        with torch.no_grad():
            ours = fwd(frame).argmax(1)[0]
            ref, _ = step(frame, step.initial_state((32, 64)))
        assert torch.equal(ours, ref)


def test_uniform_scale_one_is_the_full_frame():
    model = _dummy(1, False)
    frame = torch.rand(2, 3, 32, 64)
    fwd = WarpedForward(model, StepConfig(1.0, resample="r2", unwarp="u1"), cell=8)
    with torch.no_grad():
        ref = upsample_logits(model(frame), (32, 64), 1, False, input_size=(32, 64))
        assert torch.allclose(fwd(frame), ref, atol=1e-5)


def test_flipped_prior_mirrors_the_warp():
    model = _dummy(1, False)
    prior = torch.rand(4, 8)
    cfg = StepConfig(0.5, "P", 0.5, 0.0, resample="r2", unwarp="u1")
    fwd = WarpedForward(model, cfg, prior=prior, cell=8)
    g, gf = fwd.grid((32, 64)), fwd.grid((32, 64), flip=True)
    assert torch.allclose(gf.dx, g.dx.flip(0)) and torch.allclose(gf.dy, g.dy)


def test_magnification_at_scale_one_changes_the_grid():
    model = _dummy(1, False)
    prior = torch.zeros(4, 8)
    prior[:, 3] = 1.0
    cfg = StepConfig(1.0, "P", 0.5, 0.0, resample="r2", unwarp="u1", d_max=2.0)
    g = WarpedForward(model, cfg, prior=prior, cell=8).grid((32, 64))
    assert float(g.dx.max()) > 1.0 and float(g.dx.max()) <= 2.0 + 1e-6
    assert cfg.name().endswith("/m2")


def test_ohem_keeps_at_least_min_kept_and_ignores_void():
    logits = torch.randn(1, 19, 4, 4, requires_grad=True)
    target = torch.randint(0, 19, (1, 4, 4))
    target[0, 0] = IGNORE
    loss = ohem_ce(logits, target, thresh=0.0, min_kept=5)
    loss.backward()
    assert torch.isfinite(loss)
    assert int((logits.grad.abs().sum(1) > 0).sum()) >= 5
    assert float(logits.grad[0, :, 0].abs().sum()) == 0.0
    full = ohem_ce(logits, target, thresh=1.0, min_kept=1)
    valid = target != IGNORE
    ref = torch.nn.functional.cross_entropy(logits, target, ignore_index=IGNORE)
    assert torch.allclose(full, ref) and int(valid.sum()) == 12


def test_jitter_stays_in_range():
    x = torch.rand(3, 3, 8, 8)
    y = jitter(x, 0.25, torch.Generator().manual_seed(0))
    assert y.min() >= 0 and y.max() <= 1 and not torch.equal(x, y)
    assert torch.equal(jitter(x, 0.0, torch.Generator()), x)


def test_run_configs_and_names():
    assert RunConfig("bisenetv2", "multi").step_configs()[0].scale == 0.25
    assert len(RunConfig("bisenetv2", "multi").step_configs()) == 13
    b = RunConfig("pidnet_s", "boundary", scale=1.0, d_max=2.0)
    assert b.name() == "11/pidnet_s/B/c1/m2/seed0"
    assert b.step_configs()[0].saliency == "P" and b.step_configs()[0].d_max == 2.0
    with pytest.raises(ValueError):
        RunConfig("pidnet_s", "glimpse")


def test_freeze_bn_keeps_running_stats():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    net.train()
    assert freeze_bn(net) == 1
    before = net[1].running_mean.clone()
    net(torch.rand(2, 3, 4, 4) * 10)
    assert torch.equal(net[1].running_mean, before)


def test_train_cli_smoke(tmp_path):
    """Two steps of a warped run on the synthetic set, then resume is a no-op and the weights load."""
    from scripts import method11

    prior = tmp_path / "prior.pt"
    from warp.policy import prior_record
    torch.save([prior_record("bisenetv2", 0.5, torch.rand(2, 4) * 3, 3)], prior)
    run = tmp_path / "run"
    argv = ["train", "--model", "bisenetv2", "--kind", "boundary", "--scale", "0.5", "--fixture",
            "--prior-file", str(prior), "--run-dir", str(run), "--batch", "2", "--epochs", "1",
            "--max-steps", "2", "--workers", "0", "--device", "cpu"]
    assert method11.main(argv) == 0
    assert (run / "weights.pt").exists() and (run / "ckpt_last.pt").exists()
    meta = json.loads((run / "run.json").read_text())
    assert meta["name"] == "11/bisenetv2/B/c0.5/seed0"
    assert method11.main(argv) == 0  # already finished
    m = method11._backbone("bisenetv2", True, str(run / "weights.pt"))
    assert m.weights_info["source"] == str(run / "weights.pt")


def test_eval_cli_on_the_vps_fixture(tmp_path):
    """A uniform run is trained on the synthetic set, then scored per clip on the VPS fixture; resumable."""
    import gzip

    from scripts import method11

    run = tmp_path / "run"
    assert method11.main(["train", "--model", "bisenetv2", "--kind", "uniform", "--scale", "0.5", "--fixture",
                          "--run-dir", str(run), "--batch", "2", "--epochs", "1", "--max-steps", "1",
                          "--workers", "0", "--device", "cpu"]) == 0
    out = tmp_path / "eval"
    argv = ["eval", "--run-dir", str(run), "--fixture", "--split", "val", "--out-dir", str(out),
            "--device", "cpu", "--tag", "t"]
    assert method11.main(argv) == 0
    files = list(out.glob("eval_val_*_t.jsonl.gz"))
    assert len(files) == 1
    rows = [json.loads(line) for line in gzip.open(files[0], "rt")]
    assert rows and all(r["weights"] == "finetuned" and r["kind"] == "uniform" for r in rows)
    assert np.asarray(rows[0]["conf"]).shape[1:] == (19, 19)
    assert method11.main(argv) == 0
    assert len([json.loads(line) for line in gzip.open(files[0], "rt")]) == len(rows)
