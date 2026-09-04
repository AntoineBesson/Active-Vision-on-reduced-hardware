"""Pytest bootstrap: make sure the synthetic Cityscapes-VPS fixture exists.

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
