"""The XCOM 3.1 port reproduces the tabulated nodes of every element file and behaves at edges."""
import numpy as np
import pytest

from shieldmap import paths
from shieldmap.ingest import xsec

XDIR = paths.RAW / "xcom" / "extracted" / "XCOM"
pytestmark = pytest.mark.skipif(not (XDIR / "MDATX3.100").exists(), reason="XCOM package not retrieved")


@pytest.fixture(scope="module")
def els():
    return xsec.load_elements(XDIR)


def _regular_nodes(el):
    below = {i - 1 for i in el.edge_index}
    m = np.array([i not in below for i in range(el.e.size)])
    return m & ~np.isclose(el.e, 1.022e6) & ~np.isclose(el.e, 2.044e6)   # XCOM shifts these by 7/14 eV


def test_all_100_elements_parse(els):
    assert sorted(els) == list(range(1, 101))


def test_atomic_weights_consistent(els):
    for el in els.values():
        assert abs(el.atwt_file / el.atwt_xcom - 1) < 3e-4


def test_conversion_mass_is_table_a_mass(els):
    """b/atom → cm²/g must use the masses behind the mass fractions and MP densities (rev.)."""
    import pandas as pd
    a = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet").set_index("Z").atomic_mass
    for z, el in els.items():
        assert el.mass == a[z]
        if z <= 83 and z not in (43, 61):                # stable elements: same physical mass
            assert abs(el.mass / el.atwt_xcom - 1) < 5e-4, z


def test_scattering_and_pair_reproduce_nodes_exactly(els):
    for el in els.values():
        m = _regular_nodes(el)
        cs = el.cross_sections_b(el.e[m])
        for k, ref in (("coh", el.coh), ("incoh", el.incoh), ("pair_nuc", el.pair_nuc), ("pair_el", el.pair_el)):
            ok = ref[m] > 0
            np.testing.assert_allclose(cs[k][ok], ref[m][ok], rtol=1e-12)
            assert np.all(cs[k][~ok] == 0)


def test_photoabsorption_reproduces_nodes(els):
    """At/above-edge and regular nodes: XCOM's main table and segment tables agree to 4 digits."""
    worst = 0.0
    for el in els.values():
        edges = dict(zip(el.edge_index, el.edge_energies))
        m = _regular_nodes(el)
        for i in np.where(m)[0]:
            e = el.e[i]
            if e < 10e3:                                  # below the project range
                continue
            ee = max(e, edges[i]) * (1 + 1e-9) if i in edges else e
            worst = max(worst, abs(el.cross_sections_b([ee])["pe"][0] / el.pe[i] - 1))
    assert worst < 2e-4, worst


def test_below_edge_nodes(els):
    for el in els.values():
        edges = dict(zip(el.edge_index, el.edge_energies))
        for i_above, ed in edges.items():
            i = i_above - 1
            if el.e[i] < 10e3:
                continue
            v = el.cross_sections_b([min(el.e[i], ed) * (1 - 1e-9)])["pe"][0]
            assert abs(v / el.pe[i] - 1) < 5e-3, (el.Z, el.e[i])


def test_edge_jump_and_side_convention(els):
    pb = els[82]
    k = dict(pb.edges_keV())["K"] * 1000
    below, at, above = pb.mu_rho(np.array([k * (1 - 1e-7), k, k * (1 + 1e-7)]))["total_wcoh"]
    assert above / below > 3.5                           # Pb K-edge jump ~ 4.0
    assert at == pytest.approx(above, rel=1e-6)          # an energy equal to an edge goes above it


def test_no_grid_node_sits_on_an_edge(els):
    """No node coincides with an edge (edges are given to 0.1 eV), and no node falls in the window
    between an above-edge table node and a slightly higher stored edge energy (Ra–U K edges, M5 of
    Re), where XCOM's side assignment would depend on rounding."""
    from shieldmap import grid
    g = grid.energy_grid_keV() * 1000.0                   # eV
    for el in els.values():
        for ed, i_above in zip(el.edge_energies, el.edge_index):
            assert np.min(np.abs(g - ed)) > 0.1, (el.Z, ed)
            lo, hi = sorted((el.e[i_above], ed))
            assert not np.any((g >= lo) & (g <= hi)), (el.Z, ed)


def test_pb_values_against_nist_reference_points(els):
    """Spot values printed by NIST for Pb (cm²/g, with coherent): 60 keV 5.021, 100 keV 5.549, 1 MeV 0.07102."""
    v = els[82].mu_rho(np.array([60e3, 100e3, 1e6]))["total_wcoh"]
    np.testing.assert_allclose(v, [5.021, 5.549, 0.07102], rtol=1e-3)
