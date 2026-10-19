"""Validation against primary data (compound μ/ρ vs SRD 126; DFT density bias table)."""
import numpy as np
import pandas as pd
import pytest

from shieldmap import paths
from shieldmap.physics import mixture

FIX = paths.FIXTURES / "srd126_mu_rho.csv"
pytestmark = pytest.mark.skipif(not FIX.exists() or not (paths.INTERIM / "mp" / "compounds.parquet").exists(),
                                reason="fixtures or Table B missing")

# One published SRD 126 value is physically inconsistent: CsI at 34.5483 keV lists
# μ/ρ = 26.87 cm²/g between the I K edge (21.22 above) and the Cs K edge (17.19 below), where no edge
# of Cs or I exists and μ/ρ must decrease monotonically. It is kept in the fixture and tested as such.
ERRATA = {("CsI", 0.0345483)}


@pytest.fixture(scope="module")
def els():
    from shieldmap.ingest import xsec
    return xsec.load_elements(paths.RAW / "xcom" / "extracted" / "XCOM")


@pytest.fixture(scope="module")
def srd():
    return pd.read_csv(FIX)


def test_compound_mu_rho_off_edge_2pct(els, srd):
    comps = sorted(set(srd[srd.kind == "compound"].material) - {"H2O"})
    assert len(comps) == 10
    n = 0
    for mat in comps:
        d = srd[(srd.material == mat) & srd.edge_label.isna() & (srd.energy_MeV >= 0.01) & (srd.energy_MeV <= 15)]
        w = mixture.fractions_from_formula(mat)
        edges = np.array([en for z in np.nonzero(w)[0] for _, en in els[z + 1].edges_keV()])
        for e_mev, ref in zip(d.energy_MeV, d.mu_rho_cm2_g):
            near = edges.size and np.min(np.abs(e_mev * 1000 / edges - 1)) <= 0.01
            if near or (mat, round(e_mev, 7)) in ERRATA:
                continue
            ours = sum(w[z] * els[z + 1].mu_rho(np.array([e_mev * 1e6]))["total_wcoh"][0] for z in np.nonzero(w)[0])
            assert abs(ours / ref - 1) <= 0.02, (mat, e_mev, ours, ref)
            n += 1
    assert n >= 260


def test_documented_erratum_is_non_monotone_in_the_reference(srd):
    d = srd[srd.material == "CsI"].sort_values(["energy_MeV"], kind="stable")
    above_i = d[(d.edge_label == "53 K") & (d.edge_side == "above")].mu_rho_cm2_g.iloc[0]
    below_cs = d[(d.edge_label == "55 K") & (d.edge_side == "below")].mu_rho_cm2_g.iloc[0]
    mid = d[np.isclose(d.energy_MeV, 0.0345483)].mu_rho_cm2_g.iloc[0]
    assert mid > max(above_i, below_cs)                  # impossible without an edge in between


def test_density_bias_table():
    p = paths.REPORTS / "data" / "step4_density_bias.csv"
    if not p.exists():
        pytest.skip("run `python -m shieldmap.validation step4`")
    d = pd.read_csv(p)
    assert d.symbol.is_unique and len(d) >= 70
    assert (d.rel_bias == d.density / d.density_solid_exp - 1).all() or np.allclose(
        d.rel_bias, d.density / d.density_solid_exp - 1)
