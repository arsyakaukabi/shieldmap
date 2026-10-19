"""Stage 2 core: frontier, price, constraints, Shapley, Pareto and TOPSIS on a synthetic 200-compound fixture."""
import math

import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis import core


@pytest.fixture
def fx():
    n = 200
    i = np.arange(n)
    els = [["Pb", "O"] if k < 5 else (["U", "O"] if k in (5, 6) else (["W"] if k % 2 else ["Fe", "O"]))
           for k in i]
    u = pd.DataFrame({
        "row": i,
        "elements": els,
        "contains_radioactive": [k in (5, 6) for k in i],
        "energy_above_hull": np.where(i % 10 == 7, 0.2, 0.0),     # rows 7, 17, ... unstable
        "theoretical": i == 9,                                    # row 9 theoretical
        "band_gap": np.where(i < 100, 0.0, 4.0),
        "min_crustal_abundance_ppm": np.where(i == 11, np.nan, 10.0 ** (i % 5 - 2)),  # 0.01…100
    })
    q = np.stack([i + 1.0, (n - i).astype(float)], axis=1)      # col0: row k → k+1; col1 reversed
    return u, q


def test_frontier_kinds(fx):
    _, q = fx
    m = np.ones(200, bool)
    np.testing.assert_allclose(core.frontier(q, m, "best"), [1, 1])
    np.testing.assert_allclose(core.frontier(q, m, "10th"), [10, 10])
    np.testing.assert_allclose(core.frontier(q, m, "median50"), [25.5, 25.5])
    assert core.frontier(q[:, 0], m, "best") == 1
    small = np.zeros(200, bool); small[:5] = True
    assert np.isnan(core.frontier(q[:, 0], small, "10th"))


def test_constraints_and_prices(fx):
    u, q = fx
    base = np.ones(200, bool)
    pb = core.constraint_mask("PB", u=u)
    assert pb.sum() == 195 and not pb[:5].any()
    rad = core.constraint_mask("RAD", u=u)
    # best without Pb and U (rows 0–6 removed) is row 7 → q = 8 → price 7
    np.testing.assert_allclose(core.price(q[:, 0], base, pb & rad), 7.0)
    stab = core.constraint_mask("STAB", 0.05, u=u)
    assert not stab[7] and stab[8]
    exp = core.constraint_mask("EXP", u=u)
    assert not exp[9]
    b4 = pb & rad & stab & exp                                # first allowed row: 8 → q = 9
    np.testing.assert_allclose(core.price(q[:, 0], base, b4), 8.0)
    gap = core.constraint_mask("GAP", 3.1, u=u)
    np.testing.assert_allclose(core.price(q[:, 0], base, gap), 100.0)    # first row with gap: 100 → q=101
    ab = core.constraint_mask("ABUND", 1.0, u=u)                         # i%5 ∈ {2,3,4}; row 11 NaN → fails
    assert not ab[11] and ab[2] and not ab[0]
    np.testing.assert_allclose(core.price(q[:, 0], base, ab), 2.0)       # first row with i%5≥2 is row 2
    ex = core.constraint_mask("EXCL", "W", u=u)
    np.testing.assert_allclose(core.price(q[:, 1], base, ex), 1.0)  # col1 best: row 199 (W, q=1) → row 198 (q=2)
    np.testing.assert_allclose(core.price(q, base, pb), [5.0, 0.0])


def test_bundles_cover_named_constraints(fx):
    u, _ = fx
    b4 = core.bundle_mask("B4", u=u)
    expect = (core.constraint_mask("RAD", u=u) & core.constraint_mask("PB", u=u)
              & core.constraint_mask("STAB", 0.05, u=u) & core.constraint_mask("EXP", u=u))
    assert (b4 == expect).all()
    assert core.bundle_mask("B0", u=u).all()
    assert (core.bundle_mask("B5", u=u) == (b4 & core.constraint_mask("GAP", 3.1, u=u))).all()


def test_shapley_efficiency_and_symmetry():
    players = ["A", "B", "C", "D"]
    val = {"A": 1.0, "B": 2.0, "C": 0.5, "D": 0.0}

    def v(s):                                   # additive game + an A–B interaction of 1
        return sum(val[p] for p in s) + (1.0 if {"A", "B"} <= s else 0.0)

    phi = core.shapley(players, v)
    assert phi["A"] == pytest.approx(1.5) and phi["B"] == pytest.approx(2.5)
    assert phi["C"] == pytest.approx(0.5) and phi["D"] == pytest.approx(0.0)
    assert sum(phi.values()) == pytest.approx(v(frozenset(players)) - v(frozenset()))


def test_pareto_layers():
    a = np.array([1, 2, 3, 4, 2, 5, np.nan])
    b = np.array([1, 3, 5, 4, 2, 6, 9])          # maximise b
    lay = core.pareto_layers(a, b)
    # front 1: (1,1), (2,3), (3,5), (5,6); then (2,2) and (4,4) on layer 2; NaN never
    assert list(lay) == [1, 1, 1, 2, 2, 1, 0]


def test_topsis_and_ranks():
    m = np.array([[1.0, 10.0], [2.0, 5.0], [3.0, 1.0]])
    c = core.topsis(m, np.array([1.0, 0.0]), np.array([False, True]))   # only the cost criterion
    assert np.argmax(c) == 0 and np.argmin(c) == 2
    g = core.simplex_grid(3, 0.1)
    assert len(g) == 66 and np.allclose(g.sum(axis=1), 1)
    q = np.array([3.0, 1.0, 2.0, 1.0]); m = np.array([True, True, True, False])
    np.testing.assert_array_equal(core.top_k(q, m, 2), [1, 2])
    r = core.ranks(q, m)
    assert list(r[:3]) == [3, 1, 2] and math.isnan(r[3])
