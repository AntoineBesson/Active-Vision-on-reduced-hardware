"""Structural conformance of toy implementations to the interfaces (needs torch).

The toy policies/memory below exist only in this test to show that both a
memory-free and a canvas-aware gaze policy fit the same GazePolicy contract,
and that a canvas memory fits SceneMemory. No implementation lives in src/.
"""

from __future__ import annotations

import pytest

from models.interfaces import MemoryState, SceneMemory, SegmentationBackbone
from policies.interfaces import Box, GazePolicy, GazeState

# --- Structural conformance (needs torch for the toy implementations) ------- #
torch = pytest.importorskip("torch")
nn = torch.nn


class ToyBackbone(nn.Module):
    num_classes = 3
    output_stride = 2
    align_corners = False

    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 3, 1)

    def forward(self, images):
        return self.head(nn.functional.avg_pool2d(images, 2, ceil_mode=True))


class MemoryFreePolicy:
    """Look where the context pass is least confident."""

    uses_memory = False

    def reset(self):
        pass

    def select(self, state, memory=None):
        probs = state.context_logits.softmax(0)
        h, w = probs.shape[1:]
        idx = int(probs.max(0).values.flatten().argmin())
        i, j = divmod(idx, w)
        return [Box(j / w, i / h, (j + 1) / w, (i + 1) / h)][: state.budget]


class CanvasPolicy:
    """Look at the canvas cell that has not been observed for longest."""

    uses_memory = True

    def reset(self):
        pass

    def select(self, state, memory=None):
        age = memory.age[0]
        h, w = age.shape
        i, j = divmod(int(age.flatten().argmax()), w)
        return [Box(j / w, i / h, (j + 1) / w, (i + 1) / h)]


class ToyCanvas:
    def __init__(self, num_classes=3, canvas_size=(4, 8)):
        self.c, self.size = num_classes, canvas_size

    def reset(self, frame_size):
        hc, wc = self.size
        self.state = MemoryState(
            canvas=torch.zeros(self.c, hc, wc), confidence=torch.zeros(1, hc, wc), age=torch.zeros(1, hc, wc)
        )
        return self.state

    def update(self, glimpse_output, box, *, t=None):
        hc, wc = self.size
        top, left, bottom, right = box.to_pixels(hc, wc)
        patch = nn.functional.interpolate(glimpse_output[None], size=(bottom - top, right - left))[0]
        s = self.state
        if t is not None and t > s.t:
            s.age += 1
        s.canvas[:, top:bottom, left:right] = patch
        s.confidence[:, top:bottom, left:right] = 1.0
        s.age[:, top:bottom, left:right] = 0
        s.t = t if t is not None else s.t
        return s

    def read(self):
        return self.state


def test_toys_conform_to_the_protocols():
    assert isinstance(ToyBackbone(), SegmentationBackbone)
    assert isinstance(MemoryFreePolicy(), GazePolicy) and isinstance(CanvasPolicy(), GazePolicy)
    assert isinstance(ToyCanvas(), SceneMemory)

    class NotABackbone(nn.Module):
        num_classes = 3

        def forward(self, x):
            return x

    assert not isinstance(NotABackbone(), SegmentationBackbone)  # no output_stride/align_corners


def test_both_policy_families_run_in_the_same_loop():
    backbone = ToyBackbone().eval()
    frames = torch.rand(3, 3, 16, 32)  # a 3-frame clip, (T, 3, H, W)
    for policy in (MemoryFreePolicy(), CanvasPolicy()):
        memory = ToyCanvas()
        memory.reset((16, 32))
        policy.reset()
        for t, frame in enumerate(frames):
            with torch.no_grad():
                context = backbone(nn.functional.interpolate(frame[None], scale_factor=0.5))[0]
            assert context.shape == (3, 4, 8)  # (C, ceil(8/2), ceil(16/2))
            state = GazeState(t=t, frame_size=(16, 32), context_logits=context, budget=1)
            boxes = policy.select(state, memory=memory.read() if policy.uses_memory else None)
            assert len(boxes) == 1 and isinstance(boxes[0], Box)
            top, left, bottom, right = boxes[0].to_pixels(16, 32)
            with torch.no_grad():
                glimpse = backbone(frame[None, :, top:bottom, left:right])[0]
            memory.update(glimpse, boxes[0], t=t)
        out = memory.read()
        assert out.canvas.shape == (3, 4, 8) and out.t == 2
        assert float(out.confidence.max()) == 1.0
