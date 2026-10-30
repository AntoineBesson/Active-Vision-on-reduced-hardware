"""Pytest bootstrap: make sure the synthetic Cityscapes-VPS fixtures exist.

The fixture PNGs are tiny and fully deterministic, so they are generated on
demand (from tests/fixtures/generate_fixture.py) rather than committed as
binaries. This session-scoped autouse fixture regenerates them if missing, so
`pytest` works on a fresh clone with no real dataset present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing the generator module regardless of the working directory.
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))

import generate_fixture  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _ensure_cityscapes_vps_fixture():
    generate_fixture.ensure_fixture()
    generate_fixture.ensure_official_fixture()


@pytest.fixture(scope="session")
def vps():
    """The official-layout synthetic fixture (val split): 16x32 frames painted with the class colours."""
    from data.cityscapes_vps import CityscapesVPS, CityscapesVPSConfig

    config = Path(__file__).resolve().parents[1] / "configs" / "cityscapes_vps.yaml"
    return CityscapesVPS(CityscapesVPSConfig.from_yaml(config, root=str(generate_fixture.OFFICIAL_ROOT), split="val"))


@pytest.fixture(scope="session")
def oracle_backbone(vps):
    """Nearest-colour dummy at stride 1: perfect on ``vps`` at full resolution."""
    from models.dummy import DummyBackbone

    return DummyBackbone(num_classes=19, output_stride=1, class_colors=vps.palette / 255.0).eval()
