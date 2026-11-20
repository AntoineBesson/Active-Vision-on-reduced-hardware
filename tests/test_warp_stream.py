"""Method 5, phase 2, stage B: the closed-loop clip runner (CPU, synthetic fixture)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.harness import frame_to_tensor  # noqa: E402
from models.dummy import DummyBackbone  # noqa: E402
from reuse.method import full_pass  # noqa: E402
from reuse.motion import MOTION_FACTOR  # noqa: E402
from reuse.replay import _confusion_t  # noqa: E402
from warp.policy import StepConfig, WarpStep  # noqa: E402
from warp.stream import build_steps, stream_clip  # noqa: E402

CELL = 8  # 16 x 32 fixture frames -> a 2 x 4 saliency grid


def _models(vps):
    """Nearest-colour dummies with BiSeNetV2's convention (stride 1, align_corners False) and
    PIDNet-S's (stride 8, align_corners True)."""
    bise = DummyBackbone(19, output_stride=1, class_colors=vps.palette / 255.0).eval()
    pid = DummyBackbone(19, output_stride=8, class_colors=vps.palette / 255.0).eval()
    pid.align_corners = True
    return {"bisenetv2": bise, "pidnet_s": pid}


def _prior(seed=3, shape=(2, 4)):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g)


class _Synthetic:
    """One 6-frame clip of 32 x 64 frames painted with the class colours in 4-px blocks that slide
    2 px a frame, labelled at frames 2 and 5. Unlike the 16 x 32 fixture, the predictions have
    boundaries and small classes at scale 0.5, so B, S and U warps differ from the uniform grid
    and from one another frame to frame."""

    def __init__(self, palette, n=6, seed=0):
        base = np.random.default_rng(seed).integers(0, 19, size=(8, 16))
        self.labels = [np.roll(np.kron(base, np.ones((4, 4), dtype=np.int64)), 2 * t, axis=1) for t in range(n)]
        self.palette, self.n = palette, n

    def clip_id(self, index):
        return "syn"

    def labeled_mask(self, index):
        return [t in (2, 5) for t in range(self.n)]

    def frame_indices(self, index):
        return list(range(100, 100 + self.n))

    def load_label(self, index, t):
        return self.labels[t]

    def load_frame(self, index, t):
        return self.palette[self.labels[t]].astype(np.uint8)


class _Counting:
    """The dataset, counting the frames loaded, with optional frame numbers (to make gaps)."""

    def __init__(self, base, numbers=None):
        self.base, self.numbers, self.loaded = base, numbers, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def frame_indices(self, index):
        return self.numbers if self.numbers is not None else self.base.frame_indices(index)

    def load_frame(self, index, t):
        self.loaded.append(int(t))
        return self.base.load_frame(index, t)


def _labelled(vps, index=0):
    return [int(t) for t in np.flatnonzero(np.asarray(vps.labeled_mask(index), dtype=bool))]


def _label(vps, t, index=0):
    return torch.from_numpy(np.asarray(vps.load_label(index, t), dtype=np.int64))


@pytest.mark.parametrize("model_kind", ["bisenetv2", "pidnet_s"])
def test_a_uniform_row_streamed_over_a_clip_is_the_full_frame(vps, model_kind):
    """The pipeline check at the runner's level: with r2 (the baseline's filter), the uniform row's
    confusion at every labelled position is the full frame's at that scale."""
    model = _models(vps)[model_kind]
    cfgs = [StepConfig(s, resample="r2", unwarp="u1") for s in (0.5, 0.25)]
    positions, out = stream_clip(vps, 0, model, build_steps(model, cfgs, {}, CELL), "cpu", batch=2,
                                 allow_any_frame_size=True)
    assert positions == _labelled(vps) and len(positions) > 0
    for c in cfgs:
        conf = out[c.name()]
        assert conf.shape == (len(positions), 19, 19) and conf.dtype == np.int64
        for j, t in enumerate(positions):
            frame = frame_to_tensor(vps.load_frame(0, t), "cpu")
            small = (max(1, frame.shape[-2] // MOTION_FACTOR), max(1, frame.shape[-1] // MOTION_FACTOR))
            ref = _confusion_t(_label(vps, t), full_pass(model, frame, c.scale, small)[0]).numpy()
            assert np.array_equal(conf[j], ref), (c.name(), t)


@pytest.mark.parametrize("model_kind", ["bisenetv2", "pidnet_s"])
def test_configurations_of_one_batch_keep_their_own_state(vps, model_kind):
    """B, U (closed loops) and P configurations give their own results alone, together, and in any
    order of the batch; their results differ, so a shared state would show."""
    model = _models(vps)[model_kind]
    data = _Synthetic(vps.palette)
    cfgs = [StepConfig(0.5, k, 0.25, 1, "r2", "u1") for k in ("B", "U", "P")]
    priors = {0.5: _prior(shape=(4, 8))}

    def run(group, batch):
        return stream_clip(data, 0, model, build_steps(model, group, priors, CELL), "cpu", batch=batch,
                           allow_any_frame_size=True)[1]

    alone = {k: v for c in cfgs for k, v in run([c], 1).items()}
    assert not np.array_equal(alone[cfgs[0].name()], alone[cfgs[1].name()])
    assert not np.array_equal(alone[cfgs[0].name()], alone[cfgs[2].name()])
    for group, batch in ((cfgs, 3), (cfgs[::-1], 3), (cfgs[1:] + cfgs[:1], 2)):
        together = run(group, batch)
        for c in cfgs:
            assert np.array_equal(together[c.name()], alone[c.name()]), (c.name(), [g.name() for g in group])


@pytest.mark.parametrize("model_kind", ["bisenetv2", "pidnet_s"])
@pytest.mark.parametrize("kind", ["B", "S", "U"])
def test_the_closed_loop_starts_from_the_uniform_grid_and_carries_its_own_output(vps, model_kind, kind):
    """Every frame in order, frame 0 from the uniform grid (no P), each next frame from the state
    the step returned on the previous one: the same as stepping the clip by hand, and not the same
    as starting every frame from the uniform grid."""
    model = _models(vps)[model_kind]
    syn = _Synthetic(vps.palette)
    data = _Counting(syn)
    cfg = StepConfig(0.5, kind, 0.25, 1, "r2", "u1")
    positions, out = stream_clip(data, 0, model, build_steps(model, [cfg], {}, CELL), "cpu",
                                 allow_any_frame_size=True)
    assert positions == [2, 5] and data.loaded == list(range(syn.n))
    step = WarpStep(model, cfg, cell=CELL)
    state = step.initial_state((32, 64))
    assert torch.equal(state, torch.zeros(4, 8))
    loop = np.zeros((2, 19, 19), dtype=np.int64)
    reset = np.zeros_like(loop)
    with torch.no_grad():
        for t in range(syn.n):
            frame = frame_to_tensor(syn.load_frame(0, t), "cpu")
            labels, state = step(frame, state)
            fresh, _ = step(frame, step.initial_state((32, 64)))
            if t in positions:
                label = torch.from_numpy(syn.load_label(0, t))
                loop[positions.index(t)] += _confusion_t(label, labels).numpy()
                reset[positions.index(t)] += _confusion_t(label, fresh).numpy()
    assert np.array_equal(out[cfg.name()], loop)
    assert not np.array_equal(loop, reset)


def test_p_starts_from_its_prior(vps):
    model = _models(vps)["bisenetv2"]
    prior = _prior()
    step = build_steps(model, [StepConfig(0.25, "P", 0.15, 4, "r3", "u1")], {0.25: prior}, CELL)[0]
    assert torch.equal(step.initial_state((16, 32)), prior)


def test_fixed_configurations_run_the_labelled_frames_only_with_the_same_confusions(vps):
    model = _models(vps)["pidnet_s"]
    cfgs = [StepConfig(0.5, "P", a, s, "r3", "u1") for a in (0.1, 0.25) for s in (1, 4)]
    cfgs += [StepConfig(0.5, resample="r3", unwarp="u1")]
    steps = build_steps(model, cfgs, {0.5: _prior()}, CELL)
    quick, full = _Counting(vps), _Counting(vps)
    pos_q, out_q = stream_clip(quick, 0, model, steps, "cpu", batch=3, allow_any_frame_size=True)
    pos_f, out_f = stream_clip(full, 0, model, steps, "cpu", batch=3, allow_any_frame_size=True, every_frame=True)
    assert quick.loaded == _labelled(vps) and full.loaded == list(range(len(vps.labeled_mask(0))))
    assert pos_q == pos_f and out_q.keys() == out_f.keys()
    assert all(np.array_equal(out_q[k], out_f[k]) for k in out_q)


def test_the_closed_loop_refuses_a_clip_with_missing_frames(vps):
    model = _models(vps)["bisenetv2"]
    n = len(vps.labeled_mask(0))
    gappy = _Counting(vps, numbers=list(range(0, 5 * n, 5)))  # only every fifth frame on disk
    with pytest.raises(ValueError, match="every frame"):
        stream_clip(gappy, 0, model, build_steps(model, [StepConfig(0.25, "B", 0.25, 1)], {}, CELL), "cpu",
                    allow_any_frame_size=True)
    # a fixed warp reads no previous frame: it runs
    stream_clip(gappy, 0, model, build_steps(model, [StepConfig(0.25)], {}, CELL), "cpu", allow_any_frame_size=True)


def test_the_runner_refuses_bad_inputs(vps):
    model = _models(vps)["bisenetv2"]
    with pytest.raises(ValueError, match="no prior"):
        build_steps(model, [StepConfig(0.25, "P", 0.25, 1)], {}, CELL)
    with pytest.raises(ValueError, match="twice"):
        build_steps(model, [StepConfig(0.25), StepConfig(0.25)], {}, CELL)
    steps = build_steps(model, [StepConfig(0.25)], {}, CELL)
    with pytest.raises(ValueError, match="1024"):
        stream_clip(vps, 0, model, steps, "cpu")
    with pytest.raises(ValueError, match="model"):
        stream_clip(vps, 0, _models(vps)["pidnet_s"], steps, "cpu", allow_any_frame_size=True)
    half = [WarpStep(model, StepConfig(0.25), cell=CELL, precision="fp16")]
    with pytest.raises(ValueError, match="fp32"):
        stream_clip(vps, 0, model, half, "cpu", allow_any_frame_size=True)
