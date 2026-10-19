"""Unit tests for the A1 landscape helpers (synthetic data; no stage 1 tables needed)."""
import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis import landscape as L


def test_decompose_identity_and_shares():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.2, 5000)
    Y = np.column_stack([0.3 * x + rng.normal(0, 0.1, 5000), rng.normal(0, 0.4, 5000)])
    d = L.decompose(x, Y)
    for j in range(2):
        assert d["var_logmu"][j] == pytest.approx(np.var(x + Y[:, j]), rel=1e-12)
        assert d["share_density"][j] + d["share_composition"][j] + d["share_cov"][j] == pytest.approx(1.0)
        assert d["var_logrho"][j] == pytest.approx(np.var(x))
        assert d["var_logmurho"][j] == pytest.approx(np.var(Y[:, j]))
    assert d["corr"][0] > 0.4 and abs(d["corr"][1]) < 0.05


def test_top_mu_subset_size_and_negative_cov():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 10000); Y = rng.normal(0, 1, (10000, 1))
    d = L._decompose_top_mu(x, Y, 0.01)
    assert d["n"][0] == 100
    assert d["cov"][0] < 0                       # selection on x + y induces negative covariance


def test_find_crossings_log_interpolation():
    e = np.array([100.0, 200.0, 400.0])
    d = np.array([1.0, -1.0, -2.0])
    c = L.find_crossings(e, d)
    assert len(c) == 1 and c[0]["direction"] == "comp->dens"
    assert c[0]["E_cross_keV"] == pytest.approx(np.sqrt(100 * 200))
    edges = pd.DataFrame({"symbol": ["X"], "shell": ["K"], "energy_keV": [150.0]})
    assert L.find_crossings(e, d, edges)[0]["edge_between_nodes"] == "X K"
    assert L.find_crossings(e, np.array([-1.0, 1.0, 2.0]))[0]["direction"] == "dens->comp"


def test_energy_index_exact():
    e = np.array([10.0, 20.0, 661.657])
    assert list(L.energy_index([20, 661.657], e)) == [1, 2]
    with pytest.raises(AssertionError):
        L.energy_index([661.66], e)
