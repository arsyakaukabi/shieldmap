"""The hardening revision: crossover under population reweighting (analysis3/reweight.py)."""
import numpy as np
import pytest

from shieldmap.analysis import landscape as p2
from shieldmap.analysis3 import reweight


def test_uniform_weights_reproduce_phase2_decomposition():
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    Y = rng.normal(size=(500, 4)) + 0.3 * x[:, None]
    w = np.full(500, 1 / 500)
    a = reweight.weighted_decomposition(x, Y, w)
    b = p2.decompose(x, Y)
    for k in ("var_logmurho", "cov", "share_density", "share_composition", "share_cov"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-12)
    assert a["n_eff"] == pytest.approx(500)


def test_zero_weight_rows_are_excluded():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300); Y = rng.normal(size=(300, 2))
    w = np.r_[np.full(200, 1.0), np.zeros(100)]
    a = reweight.weighted_decomposition(x, Y, w / w.sum())
    b = p2.decompose(x[:200], Y[:200])
    np.testing.assert_allclose(a["var_logmurho"], b["var_logmurho"], rtol=1e-12)
    assert a["n"] == 200


def test_unweighted_crossing_matches_phase2():
    import pandas as pd
    from shieldmap.analysis3 import common
    f = common.P3 / "RW" / "crossings.csv"
    if not f.exists():
        pytest.skip("run python -m shieldmap.analysis3.reweight first")
    c = pd.read_csv(f)
    e = c[(c.population == "all") & c.coherent & (c.direction == "down") & (c.E_cross_keV > 100)].E_cross_keV.max()
    assert e == pytest.approx(235.2, abs=0.1)        # 235.15 (edge-aware interpolation), float64 235.20
