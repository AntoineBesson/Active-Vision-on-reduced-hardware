"""Method 10 slow/fast split: with the same frame's slow tensors it is the model's own forward."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _net(kind):
    from models.registry import build_backbone

    torch.manual_seed(0)
    net = build_backbone({"type": kind, "num_classes": 19, "allow_random_init": True}).eval()
    # random BN statistics, so a wrong wiring cannot hide behind identity-like layers
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
    return net


@pytest.mark.parametrize("kind", ["pidnet_s", "bisenetv2"])
@pytest.mark.parametrize("hw", [(128, 256), (96, 200)])
def test_split_reproduces_the_forward(kind, hw):
    from reuse.slowfast import slowfast_split

    net = _net(kind)
    split = slowfast_split(net)
    x = torch.rand(1, 3, *hw)
    with torch.no_grad():
        ref = net(x)
        key, cache = split.key(x)
        fresh = split.fresh(x, split.slow(x))
        fresh_key_cache = split.fresh(x, cache)
    assert key.shape == ref.shape == fresh.shape
    torch.testing.assert_close(key, ref, rtol=0, atol=0)
    torch.testing.assert_close(fresh, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fresh_key_cache, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["pidnet_s", "bisenetv2"])
def test_stale_cache_changes_the_output_and_is_not_modified(kind):
    from reuse.slowfast import slowfast_split

    net = _net(kind)
    split = slowfast_split(net)
    a, b = torch.rand(1, 3, 128, 256), torch.rand(1, 3, 128, 256)
    with torch.no_grad():
        cache = split.slow(a)
        saved = [t.clone() for t in cache]
        stale = split.fresh(b, cache)
        ref_b = net(b)
    assert stale.shape == ref_b.shape
    assert not torch.allclose(stale, ref_b)  # the slow tensors matter
    for t, s in zip(cache, saved):
        assert torch.equal(t, s)  # the fresh pass never writes into the cache


def test_other_models_are_refused():
    from models.registry import build_backbone
    from reuse.slowfast import slowfast_split

    with pytest.raises(TypeError, match="two-branch"):
        slowfast_split(build_backbone({"type": "dummy", "num_classes": 19, "allow_random_init": True}))
