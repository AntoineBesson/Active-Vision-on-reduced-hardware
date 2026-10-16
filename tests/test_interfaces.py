"""Interfaces: Box geometry and GazeState (no torch needed).

Structural conformance of toy implementations is in
tests/test_interfaces_conformance.py (needs torch).
"""

from __future__ import annotations

import math

import pytest

from policies.interfaces import Box, GazePolicy, GazeState


# --- Box -------------------------------------------------------------------- #
def test_box_validation_and_geometry():
    b = Box(0.25, 0.5, 0.75, 1.0)
    assert (b.width, b.height) == (0.5, 0.5) and math.isclose(b.area, 0.25)
    for bad in [(0.5, 0, 0.5, 1), (0, 0.6, 1, 0.5), (-0.1, 0, 1, 1), (0, 0, 1.1, 1), (0, 0, float("nan"), 1)]:
        with pytest.raises(ValueError):
            Box(*bad)
    with pytest.raises(Exception):
        b.x0 = 0.1  # frozen


def test_box_to_pixels_covers_and_is_never_empty():
    assert Box(0.25, 0.5, 0.75, 1.0).to_pixels(16, 32) == (8, 8, 16, 24)
    # covering: fractional bounds round outwards
    assert Box(0.26, 0.1, 0.74, 0.9).to_pixels(10, 10) == (1, 2, 9, 8)
    # float noise does not grow the box (0.7 * 10 == 7.000000000000001)
    assert Box(0.3, 0.3, 0.7, 0.7).to_pixels(10, 10) == (3, 3, 7, 7)
    # a sliver still yields one pixel; a full box yields the full grid
    assert Box(0.5, 0.5, 0.500001, 0.500001).to_pixels(10, 10) == (5, 5, 6, 6)
    assert Box(0.0, 0.0, 1.0, 1.0).to_pixels(7, 9) == (0, 0, 7, 9)
    assert Box(0.999, 0.999, 1.0, 1.0).to_pixels(10, 10) == (9, 9, 10, 10)


def test_box_pixel_round_trip():
    b = Box.from_pixels(4, 8, 12, 24, height=16, width=32)
    assert b == Box(0.25, 0.25, 0.75, 0.75)
    assert b.to_pixels(16, 32) == (4, 8, 12, 24)
    assert b.pixel_area(16, 32) == 8 * 16


def test_gaze_state_defaults():
    s = GazeState(t=3, frame_size=(1024, 2048))
    assert s.budget == 1 and s.context_logits is None and list(s.previous_boxes) == [] and s.extras == {}
