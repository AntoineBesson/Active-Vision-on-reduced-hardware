"""scripts/latency_floor.py: every end-to-end point has its floor in the measured grids."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from scripts.latency_floor import FRAME, FULL, QUICK  # noqa: E402


@pytest.mark.parametrize("s", [FULL, QUICK], ids=["full", "quick"])
def test_e2e_points_are_covered_by_the_parts(s):
    for cs, q, k in s["e2e"]:
        assert cs in s["full_scales"]  # context backbone time (area part)
        assert q in s["batch_sizes"] and k in s["ks"] and k * q * q <= s["max_batch_pixels"]  # glimpse batch
        assert cs in s["ctx_scales"] and q in s["glimpses"] and k in s["overhead_ks"]  # pipeline overhead
        assert (FRAME[0] * cs) % 32 == 0 and (FRAME[1] * cs) % 32 == 0 and q % 32 == 0  # no padding in either model
