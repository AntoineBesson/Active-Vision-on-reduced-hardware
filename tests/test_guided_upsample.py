import pytest

torch = pytest.importorskip("torch")

from upsample.guided import GuidedConfig, bilinear_labels, covered_size, guided_coeffs, guided_logits  # noqa: E402
from models.utils import upsample_logits  # noqa: E402


def _case(stride, align, gh, gw, ih, iw, h=64, w=128, c=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c, gh, gw, generator=g), torch.rand(1, 3, h, w, generator=g), stride, align, (ih, iw))


@pytest.mark.parametrize("stride,align,gh,gw,ih,iw", [(8, True, 4, 8, 32, 64), (8, True, 5, 9, 33, 65),
                                                     (1, False, 32, 64, 32, 64), (8, False, 4, 8, 32, 64)])
@pytest.mark.parametrize("guide", ["gray", "rgb"])
def test_radius_zero_is_the_bilinear_baseline(stride, align, gh, gw, ih, iw, guide):
    logits, frame, s, a, inp = _case(stride, align, gh, gw, ih, iw)
    cfg = GuidedConfig(0.5, guide, "logit", 0, 1e-3)
    out = guided_logits(logits, frame, cfg, s, a, inp)
    ref = upsample_logits(logits, frame.shape[-2:], s, a, input_size=inp)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5)


def test_huge_eps_is_a_box_smoothed_bilinear():
    logits, frame, s, a, inp = _case(8, True, 4, 8, 32, 64)
    out = guided_logits(logits, frame, GuidedConfig(0.5, "gray", "logit", 1, 1e9), s, a, inp)
    from upsample.guided import box

    ref = upsample_logits(box(box(logits, 1), 1), frame.shape[-2:], s, a, input_size=inp)
    assert torch.allclose(out, ref, atol=1e-4)


def test_guided_follows_an_edge_the_bilinear_blurs():
    # two classes split by a vertical edge at x = 37 in the frame (not on the coarse grid)
    h, w, f = 32, 64, 8
    frame = torch.zeros(1, 3, h, w)
    frame[..., 37:] = 1.0
    truth = (torch.arange(w) >= 37).long().expand(h, w)
    coarse = torch.nn.functional.avg_pool2d(frame[:, :1], f)  # the model sees a blurred edge
    logits = torch.cat([1 - coarse, coarse], 1) * 8 - 4
    inp = (h // f * f, w // f * f)
    bil = bilinear_labels(logits, (h, w), f, False, inp)
    gui = guided_logits(logits, frame, GuidedConfig(0.25, "gray", "logit", 1, 1e-4), f, False, inp).argmax(1)[0]
    assert (gui == truth).float().mean() > (bil == truth).float().mean()
    assert (gui == truth).float().mean() > 0.99


def test_rgb_solve_matches_gray_for_a_gray_guide():
    logits, frame, s, a, inp = _case(8, True, 6, 10, 48, 80)
    gray = frame.mean(1, keepdim=True).expand(-1, 3, -1, -1).contiguous()
    g1 = guided_logits(logits, gray, GuidedConfig(0.5, "gray", "logit", 1, 1e-2), s, a, inp)
    # an rgb guide with three equal channels: the 3x3 system is singular but for eps; the result
    # spreads the gray coefficient over the channels: (var 11^T + 3 eps I)^-1 gives the gray result at eps
    g3 = guided_logits(logits, gray, GuidedConfig(0.5, "rgb", "logit", 1, 3e-2), s, a, inp)
    assert torch.allclose(g1, g3, atol=1e-3)


def test_names_round_trip_and_validation():
    c = GuidedConfig(0.375, "rgb", "prob", 2, 1e-4)
    assert c.name() == "7g/c0.375/rgb/prob/r2/e0.0001"
    assert GuidedConfig.from_name(c.name()) == c
    with pytest.raises(ValueError):
        GuidedConfig(0.5, "lab")
    with pytest.raises(ValueError):
        GuidedConfig(0.5, eps=0)
    assert covered_size((5, 9), (64, 128), 8, (33, 65)) == (78, 142)


def test_fp16_full_resolution_steps_agree():
    logits, frame, s, a, inp = _case(8, True, 4, 8, 32, 64)
    cfg = GuidedConfig(0.5, "gray", "prob", 1, 1e-3)
    a32 = guided_logits(logits, frame, cfg, s, a, inp)
    a16 = guided_logits(logits, frame, cfg, s, a, inp, full_dtype=torch.float16).float()
    assert torch.allclose(a32, a16, atol=5e-3)
