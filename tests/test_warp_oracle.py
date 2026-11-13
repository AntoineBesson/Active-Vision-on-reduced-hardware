"""Method 5: the oracle saliencies E (error), B (boundaries) and S (small classes) (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from warp.oracle import SMALL_CLASSES, pixel_mask, saliency, smooth  # noqa: E402


def test_each_kind_marks_the_right_pixels():
    t = torch.zeros(32, 64, dtype=torch.long)
    t[:, 32:] = 1
    t[0:16, 0:16] = 5  # a pole block
    u = t.clone()
    u[16:32, 48:64] = 3  # the uniform pass is wrong there
    e, b, s = (saliency(k, t, u, sigma=0.0 + 1e-6, cell=16) for k in "EBS")
    assert e.shape == (2, 4) and e.argmax() == 7 and e.flatten()[:7].sum() < 1e-3
    assert s.argmax() == 0 and s.flatten()[1:].sum() < 1e-3
    assert b[0, 0] > 0 and b[1, 1] > 0 and b[1, 3] < 1e-3  # block edge, class edge, flat area


def test_boundaries_are_marked_on_both_sides():
    t = torch.zeros(4, 6, dtype=torch.long)
    t[:, 3:] = 2
    m = pixel_mask("B", t)
    assert m[:, 2].all() and m[:, 3].all() and int(m.sum()) == 8
    t2 = torch.zeros(4, 6, dtype=torch.long)
    t2[2:] = 1
    m2 = pixel_mask("B", t2)
    assert m2[1].all() and m2[2].all() and int(m2.sum()) == 12


def test_small_classes_are_the_specs():
    assert SMALL_CLASSES == (5, 6, 7, 11, 12, 17, 18)
    t = torch.arange(19).repeat(2, 1)
    assert pixel_mask("S", t)[0].nonzero().flatten().tolist() == list(SMALL_CLASSES)


def test_smoothing_spreads_and_keeps_the_mass():
    t = torch.zeros(64, 128, dtype=torch.long)
    t[16:32, 48:64] = 11
    a, b = saliency("S", t, None, 1e-6, 16), saliency("S", t, None, 1.0, 16)
    assert (b > 0).sum() > (a > 0).sum() and abs(float(a.sum() - b.sum())) < 0.05 * float(a.sum())


def test_smoothing_is_symmetric_and_keeps_the_mass_away_from_the_borders():
    g = torch.zeros(64, 128)
    g[30, 60] = 1.0
    for sigma in (1.0, 4.0):
        s = smooth(g, sigma)
        assert abs(float(s.sum()) - 1) < 1e-5 and s.argmax() == 30 * 128 + 60
        assert torch.allclose(s[30, 60 - 5], s[30, 60 + 5]) and torch.allclose(s[30 - 5, 60], s[30 + 5, 60])
    assert torch.equal(smooth(g, 0.0), g)


def test_smoothing_keeps_the_mass_at_the_borders_too():
    """Reflect padding with the edge cell repeated: a border cell's mass stays in the map (with a
    wide kernel it piles up against the border, so the peak may move onto the edge row)."""
    for cell in ((0, 0), (1, 127), (63, 5)):
        g = torch.zeros(64, 128)
        g[cell] = 1.0
        for sigma in (1.0, 4.0):
            s = smooth(g, sigma)
            assert abs(float(s.sum()) - 1) < 1e-5
            assert sigma > 1 or s.argmax() == cell[0] * 128 + cell[1]


def test_smoothing_a_map_smaller_than_the_kernel():
    s = smooth(torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 0]]), 4.0)  # radius 12 > the map: capped
    assert s.shape == (2, 4) and torch.isfinite(s).all() and s[0, 0] == s.max()
    assert smooth(torch.ones(1, 1), 1.0).item() == 1.0


def test_bad_inputs_raise():
    t = torch.zeros(16, 32, dtype=torch.long)
    with pytest.raises(ValueError):
        saliency("E", t, None, 1.0)
    with pytest.raises(ValueError):
        saliency("X", t, t, 1.0)
    with pytest.raises(ValueError, match="cells"):
        saliency("S", torch.zeros(20, 32, dtype=torch.long), None, 1.0)
    with pytest.raises(ValueError):
        smooth(torch.zeros(2, 2), -1.0)
