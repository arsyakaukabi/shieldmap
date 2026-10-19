"""Tests for analysis/candidates.py (A3). Pure helpers always run; data checks skip without stage 2 data."""
import numpy as np
import pytest

from shieldmap.analysis import candidates as C
from shieldmap.analysis import core


def test_textbook_rule():
    assert C.textbook(["W"]) == "yes"
    assert C.textbook(["Ca", "W", "O"]) == "yes"
    assert C.textbook(["Ba", "S", "O"]) == "yes"
    assert C.textbook(["Ca", "C", "O"]) == "yes"          # concrete-like
    assert C.textbook(["Si", "O"]) == "yes"
    assert C.textbook(["Zn", "W", "O"]) == "no"            # Zn outside the list
    assert C.textbook(["Pr", "Sb", "Pd"]) == "no"
    assert C.textbook(["Ir"]) == "no"
    assert C.textbook(["Li", "F"]) == "no"                 # light-only but not concrete-like


def test_lex_rank_orders_by_keys_then_row():
    idx = np.array([10, 11, 12, 13])
    k1 = np.array([2, 1, 2, 1])
    k2 = np.array([0.5, 0.9, 0.1, 0.9])
    r = C.lex_rank(idx, k1, k2)
    # k1=1: rows 11, 13 tie on k2 -> row order; then k1=2: row 12 (0.1) before row 10 (0.5)
    assert r.tolist() == [4, 1, 3, 2]


def test_lex_rank_single_key_matches_core_ranks():
    rng = np.random.default_rng(0)
    q = rng.random(50)
    mask = rng.random(50) < 0.6
    idx = np.flatnonzero(mask)
    np.testing.assert_array_equal(C.lex_rank(idx, q[idx]), core.ranks(q, mask)[idx].astype(int))


_have_data = (core.P2 / "universe.parquet").exists()


@pytest.mark.skipif(not _have_data, reason="Phase 2 universe not built")
def test_pb_cosine_distance_reference():
    d, info = C.pb_cosine_distance()
    assert abs(d[info["pb_row"]]) < 1e-12
    assert info["max_rel_dev_vs_xsec_grid"] < 1e-6
    assert info["n_nodes"] == 261


@pytest.mark.skipif(not _have_data, reason="Phase 2 universe not built")
def test_formula_ranks_r2_equals_core_ranks():
    q, vs = C.scen_matrix("CS137")
    bm = core.bundle_mask("B4")
    fr = C.formula_ranks(q[:, :1], "R2", bm)
    m = core.base_mask("R2") & bm
    r = core.ranks(q[:, 0], m)
    rows = np.flatnonzero(m)
    codes = C.attrs().fcode.to_numpy()[rows]
    np.testing.assert_array_equal(fr[codes, 0], r[rows])
