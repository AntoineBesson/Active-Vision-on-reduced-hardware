"""Backbone wrappers, weight loading and the logits contract (CPU, no network)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.bisenetv2 import BiSeNetV2Backbone  # noqa: E402
from models.dummy import DummyBackbone  # noqa: E402
from models.interfaces import SegmentationBackbone  # noqa: E402
from models.pidnet import PIDNetBackbone  # noqa: E402
from models.registry import build_backbone  # noqa: E402
from models.utils import pad_to_multiple, upsample_logits  # noqa: E402
from models.weights import WeightsError, extract_state_dict, resolve_weights  # noqa: E402


@pytest.fixture(scope="module")
def pidnet():
    torch.manual_seed(0)
    return PIDNetBackbone(variant="s", num_classes=19).eval()


@pytest.mark.parametrize("size", [(64, 96), (60, 90), (17, 33)])
def test_pidnet_output_contract(pidnet, size):
    h, w = size
    with torch.no_grad():
        out = pidnet(torch.rand(1, 3, h, w))
    assert out.shape == (1, 19, -(-h // 8), -(-w // 8))  # ceil(H/8), ceil(W/8)
    assert isinstance(pidnet, SegmentationBackbone)
    assert (pidnet.output_stride, pidnet.align_corners) == (8, True)


def test_pidnet_param_count_matches_official():
    # 7,625,715 = parameter count of the official get_pred_model('pidnet_s', 19),
    # checked bit-for-bit against github.com/XuJiacong/PIDNet when porting.
    assert sum(p.numel() for p in PIDNetBackbone(variant="s").parameters()) == 7_625_715
    with pytest.raises(ValueError, match="variant"):
        PIDNetBackbone(variant="xl")


def test_bisenetv2_output_contract():
    model = BiSeNetV2Backbone(num_classes=19).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 60, 90))  # padded to 64x96 internally, then cropped
    assert out.shape == (1, 19, 60, 90)
    assert isinstance(model, SegmentationBackbone) and model.output_stride == 1


def test_dummy_contract_and_determinism():
    a, b = DummyBackbone(num_classes=5, output_stride=2, seed=3), DummyBackbone(num_classes=5, output_stride=2, seed=3)
    x = torch.rand(2, 3, 7, 9)
    assert a(x).shape == (2, 5, 4, 5)
    assert torch.equal(a(x), b(x))
    assert not torch.equal(a(x), DummyBackbone(num_classes=5, output_stride=2, seed=4)(x))
    assert isinstance(a, SegmentationBackbone)


def test_dummy_palette_mode_is_nearest_colour():
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    model = DummyBackbone(num_classes=3, output_stride=1, class_colors=colors)
    labels = torch.tensor([[0, 1, 2, 1]])
    img = torch.as_tensor(colors[labels.numpy()] / 255.0, dtype=torch.float32).permute(2, 0, 1)[None]
    img = (img + 0.05 * torch.randn_like(img)).clamp(0, 1)
    assert torch.equal(model(img).argmax(1)[0], labels)  # (1,1,4) -> (1,4)
    with pytest.raises(ValueError, match="class_colors"):
        DummyBackbone(num_classes=4, class_colors=colors)


# --- logits contract helpers ------------------------------------------------- #
def test_pad_to_multiple():
    x = torch.rand(1, 3, 5, 7)
    assert pad_to_multiple(x, 4).shape == (1, 3, 8, 8)
    assert pad_to_multiple(x, 1) is x
    y = torch.rand(1, 3, 8, 8)
    assert pad_to_multiple(y, 4) is y


def test_upsample_logits_is_exact_for_unpadded_cells():
    # One-hot cells on a stride-4 grid: nearest-free check of alignment + crop.
    logits = torch.zeros(1, 2, 2, 3)
    logits[0, 1, :, 1] = 10.0  # column 1 of the grid covers x in [4, 8)
    for align in (False, True):
        out = upsample_logits(logits, (6, 10), output_stride=4, align_corners=align)
        assert out.shape == (1, 2, 6, 10)
    out = upsample_logits(logits, (8, 12), output_stride=4, align_corners=False).argmax(1)[0]
    assert out[:, 5:7].eq(1).all() and out[:, 0:2].eq(0).all() and out[:, 10:].eq(0).all()
    same = torch.rand(1, 2, 3, 5)
    assert torch.equal(upsample_logits(same, (3, 5), output_stride=1, align_corners=False), same)
    with pytest.raises(ValueError, match="cannot cover"):
        upsample_logits(same, (20, 20), output_stride=4, align_corners=False)


def test_upsample_logits_from_a_downscaled_input():
    # Backbone saw a (6, 10) input (frame downscaled 2x from (12, 20)), stride 4 ->
    # (2, 3) grid covering the padded (8, 12) input = (16, 24) frame pixels.
    logits = torch.zeros(1, 2, 2, 3)
    logits[0, 1] = -10.0
    logits[0, 1, :, 2] = 10.0  # last grid column: input x in [8, 12) -> frame x in [16, 24)
    out = upsample_logits(logits, (12, 20), output_stride=4, align_corners=False, input_size=(6, 10))
    assert out.shape == (1, 2, 12, 20)
    pred = out.argmax(1)[0]
    # +-10 logits put the decision boundary halfway between cell centres, i.e.
    # exactly at the cell edge mapped into frame coordinates (x = 16).
    assert pred[:, 16:].eq(1).all() and pred[:, :16].eq(0).all()
    # no padding: identical to a direct interpolation to the frame size
    g = torch.rand(1, 2, 4, 8)
    direct = torch.nn.functional.interpolate(g, size=(64, 128), mode="bilinear", align_corners=True)
    assert torch.allclose(upsample_logits(g, (64, 128), 8, True, input_size=(32, 64)), direct)


# --- weights ----------------------------------------------------------------- #
def _save_like_official(tmp_path, model, name="w.pt"):
    state = {"model." + k: v for k, v in model.net.state_dict().items()}
    path = tmp_path / name
    torch.save({"state_dict": state, "epoch": 1}, path)
    return path


def test_weights_from_path_and_file_url(tmp_path, pidnet):
    path = _save_like_official(tmp_path, pidnet)
    fresh = PIDNetBackbone(variant="s")
    report = fresh.load_weights(str(path))
    assert report["loaded"] == len(fresh.net.state_dict()) and not report["missing"]
    for k, v in pidnet.net.state_dict().items():
        assert torch.equal(fresh.net.state_dict()[k], v)
    assert fresh.weights_info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # file:// URL goes through the download path into the cache dir
    cache = tmp_path / "cache"
    via_url = PIDNetBackbone(variant="s")
    via_url.load_weights(path.as_uri(), cache_dir=str(cache))
    assert (cache / "w.pt").is_file()
    assert torch.equal(via_url.net.final_layer.conv2.weight, pidnet.net.final_layer.conv2.weight)


def test_weights_failures_are_explicit(tmp_path, pidnet):
    with pytest.raises(WeightsError, match="not found"):
        resolve_weights(str(tmp_path / "missing.pt"))
    with pytest.raises(WeightsError, match="no network"):
        # port 9 (discard) on localhost: connection refused, no real network needed
        resolve_weights("http://127.0.0.1:9/pidnet.pt", cache_dir=str(tmp_path))
    with pytest.raises(WeightsError, match="Google Drive"):
        resolve_weights("https://drive.google.com/file/d/abc/view?usp=sharing")
    path = _save_like_official(tmp_path, pidnet)
    with pytest.raises(WeightsError, match="sha256 mismatch"):
        resolve_weights(str(path), sha256="0" * 64)
    wrong = PIDNetBackbone(variant="s", num_classes=7)
    with pytest.raises(WeightsError, match="shape mismatches"):
        wrong.load_weights(str(path))
    partial = wrong.load_weights(str(path), strict=False)
    assert partial["shape_mismatch"] and partial["loaded"] > 400


def test_extract_state_dict_variants():
    t = torch.zeros(1)
    assert extract_state_dict({"module.a": t, "epoch": 3}) == {"a": t}
    assert set(extract_state_dict({"model": {"model.b": t}})) == {"b"}
    with pytest.raises(WeightsError):
        extract_state_dict([t])


def test_non_tensor_pickles_need_explicit_trust(tmp_path, pidnet):
    path = tmp_path / "pickled.pt"
    torch.save({"state_dict": pidnet.net.state_dict(), "meta": object()}, path)
    with pytest.raises(WeightsError, match="trust_pickle"):
        PIDNetBackbone(variant="s").load_weights(str(path))


# --- registry -------------------------------------------------------------- #
def test_build_backbone_registry(tmp_path, pidnet):
    with pytest.raises(ValueError, match="needs pretrained weights"):
        build_backbone({"type": "pidnet_s", "num_classes": 19})
    smoke = build_backbone({"type": "pidnet_s", "num_classes": 19, "allow_random_init": True})
    assert "random-init" in smoke.weights_info["source"] and not smoke.training
    loaded = build_backbone({"type": "pidnet_s", "weights": str(_save_like_official(tmp_path, pidnet))})
    assert loaded.describe()["weights"]["load"]["missing"] == []
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    dummy = build_backbone({"type": "dummy", "num_classes": 2, "class_colors": "dataset_palette"}, {"palette": palette})
    assert dummy.mode == "palette"
    with pytest.raises(ValueError, match="dataset palette"):
        build_backbone({"type": "dummy", "num_classes": 2, "class_colors": "dataset_palette"})
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone({"type": "segformer"})
    bise = build_backbone({"type": "bisenetv2", "allow_random_init": True})
    assert bise.describe()["type"] == "bisenetv2"
