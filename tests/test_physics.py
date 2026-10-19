"""Physics-engine tests."""
import math

import numpy as np
import pandas as pd
import pytest

from shieldmap import paths
from shieldmap.physics import mixture, transmission

FIX = paths.FIXTURES / "srd126_mu_rho.csv"
XS = paths.INTERIM / "elements" / "xsec_grid.npz"
need_data = pytest.mark.skipif(not (FIX.exists() and XS.exists()), reason="fixtures or Table A missing")


@pytest.fixture(scope="module")
def els():
    from shieldmap.ingest import xsec
    return xsec.load_elements(paths.RAW / "xcom" / "extracted" / "XCOM")


@pytest.fixture(scope="module")
def srd():
    return pd.read_csv(FIX)


def _srd_value(df, material, e_mev):
    """SRD 126 μ/ρ at a tabulated energy, or log-log interpolated between neighbours."""
    d = df[(df.material == material) & df.edge_label.isna()].sort_values("energy_MeV")
    hit = d[np.isclose(d.energy_MeV, e_mev, rtol=1e-9)]
    if len(hit):
        return float(hit.mu_rho_cm2_g.iloc[0])
    return float(np.exp(np.interp(np.log(e_mev), np.log(d.energy_MeV), np.log(d.mu_rho_cm2_g))))


# 2.1 ------------------------------------------------------------------------------------------
@need_data
@pytest.mark.parametrize("Z,material", [(82, "Pb"), (74, "W"), (26, "Fe"), (13, "Al")])
@pytest.mark.parametrize("e_keV", [60.0, 100.0, 600.0, 661.657, 800.0, 1000.0])
def test_element_mu_rho_vs_srd126(els, srd, Z, material, e_keV):
    mat = srd[(srd.kind == "element") & (srd.Z == Z)].material.iloc[0]
    ref = _srd_value(srd, mat, e_keV / 1000)
    ours = els[Z].mu_rho(np.array([e_keV * 1000]))["total_wcoh"][0]
    assert abs(ours / ref - 1) <= 0.02, (material, e_keV, ours, ref)


# 2.2 ------------------------------------------------------------------------------------------
@need_data
def test_water_from_h_and_o(els, srd):
    water = srd[(srd.kind == "compound") & (srd.material == "H2O")]
    water = water[water.edge_label.isna() & (water.energy_MeV >= 0.01) & (water.energy_MeV <= 15)]
    assert len(water) >= 20
    w = mixture.fractions_from_formula("H2O")
    e = water.energy_MeV.to_numpy() * 1e6
    ours = w[0] * els[1].mu_rho(e)["total_wcoh"] + w[7] * els[8].mu_rho(e)["total_wcoh"]
    rel = np.abs(ours / water.mu_rho_cm2_g.to_numpy() - 1)
    assert rel.max() <= 0.005, rel.max()


# 2.5 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("T", [0.5, 0.1, 0.01, 0.001])
def test_monoenergetic_limit_exact(T):
    mu = np.array([[0.0123], [1.7], [55.0], [4321.0]])
    x = transmission.solve_thickness(mu, np.array([1.0]), T)
    np.testing.assert_allclose(x, math.log(1 / T) / mu[:, 0], rtol=1e-9)


def test_single_line_ranking_independent_of_T():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.01, 10, size=(500, 1))
    ranks = [np.argsort(transmission.solve_thickness(mu, np.array([1.0]), T)) for T in (0.5, 0.1, 0.01, 0.001)]
    for r in ranks[1:]:
        assert (r == ranks[0]).all()


def test_brent_matches_scipy_and_hits_target():
    from scipy.optimize import brentq
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.2, 80, size=(200, 150)); w = rng.uniform(0, 1, 150); w[::7] = 0
    x = transmission.solve_thickness(mu, w, 0.01)
    lt = transmission.log_transmission(mu[:, w > 0], w[w > 0], x)
    np.testing.assert_allclose(lt, math.log(0.01), atol=1e-8)
    for i in range(10):
        r = brentq(lambda xx: np.log(np.sum(w * np.exp(-mu[i] * xx)) / w.sum()) - math.log(0.01),
                   0, 100, xtol=1e-14, rtol=1e-13)
        assert x[i] == pytest.approx(r, rel=1e-9)


def test_polyenergetic_not_averaged():
    """Two-line source: the exact T(x) differs from exp(-<μ>x); the solver uses the exact sum."""
    mu = np.array([[1.0, 10.0]]); w = np.array([1.0, 1.0])
    x = transmission.solve_thickness(mu, w, 0.1)[0]
    assert 0.5 * (math.exp(-x) + math.exp(-10 * x)) == pytest.approx(0.1, rel=1e-9)
    assert abs(x - math.log(10) / 5.5) > 0.1


# 2.4 ------------------------------------------------------------------------------------------
HVL_FEW_PERCENT = 0.03          # "within a few percent", fixed before any result was seen


@need_data
def test_tube_hvl_reproduces_spekpy(els):
    """Air-kerma-weighted Al HVL1 from our engine (XCOM Al, SRD 126 air) vs SpekPy's, every RQR × angle."""
    from shieldmap import config
    from shieldmap.physics import spectra
    import pandas as pd
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet")
    rho_al = float(el.loc[el.symbol == "Al", "density_solid_exp"].iloc[0])
    for q in config.get("xray_tubes.qualities"):
        for ang in config.get("xray_tubes.anode_angle_deg"):
            sp = spectra.tube_spectrum(q, float(ang))
            assert abs(sp.meta["spekpy_hvl1_mm_al"] / q["hvl1_mm_al"] - 1) <= config.get("xray_tubes.hvl_match_tolerance_rel")
            mu_al = rho_al * els[13].mu_rho(sp.energy_keV * 1000)["total_wcoh"]
            x = transmission.solve_thickness(mu_al[None, :], sp.weights("air_kerma"), 0.5)[0] * 10
            assert abs(x / sp.meta["spekpy_hvl1_mm_al"] - 1) <= HVL_FEW_PERCENT, (sp.name, x)
            assert sp.meta["dropped_air_kerma_fraction"] < 1e-6


@need_data
def test_quadrature_pieces_never_straddle_an_edge():
    from shieldmap.physics import spectra
    edges = spectra.all_edges_keV()
    cell, lo, hi, mid = spectra.quadrature(120.0, 0.25)
    inside = (edges[None, :] > lo[:, None] + 1e-9) & (edges[None, :] < hi[:, None] - 1e-9)
    assert not inside.any()
    np.testing.assert_allclose(np.bincount(cell, weights=hi - lo), 0.25, rtol=1e-12)
