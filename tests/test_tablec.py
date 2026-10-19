"""Table C consistency: stored arrays, key-energy table, profiles and envelope against first principles."""
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sps

from shieldmap import config, paths
from shieldmap.physics import mixture, tablec, transmission

OUT = paths.PROCESSED
pytestmark = pytest.mark.skipif(not (OUT / "MANIFEST.json").exists() or not (OUT / "profiles.parquet").exists()
                                or (OUT / "profiles.parquet").stat().st_size < 1000, reason="Table C not built")


@pytest.fixture(scope="module")
def tc():
    idx, e, mw = tablec.load(True)
    _, _, mwo = tablec.load(False)
    return idx, e, mw, mwo


@pytest.fixture(scope="module")
def variants():
    return pd.read_parquet(OUT / "profile_variants.parquet")


@pytest.fixture(scope="module")
def profiles():
    return pq.read_table(OUT / "profiles.parquet").to_pandas()


def test_shapes_and_counts(tc, variants, profiles):
    idx, e, mw, mwo = tc
    n = len(idx)
    assert mw.shape == mwo.shape == (n, e.size)
    assert len(profiles) == n * len(variants)
    assert len(variants) == (4 * 3 * 3 + 2 * 4) * 3 * 2          # 264
    assert profiles.groupby("variant_id").size().eq(n).all()


def test_mu_rho_matches_mixture_rule(tc):
    idx, e, mw, mwo = tc
    W = sps.load_npz(paths.INTERIM / "mp" / "mass_fractions.npz").tocsr()
    xs = mixture.load_element_xsec()
    rows = np.random.default_rng(7).choice(len(idx), 200, replace=False)
    np.testing.assert_allclose(mw[rows], (W[rows] @ xs["total_wcoh"]), rtol=1e-6)
    np.testing.assert_allclose(mwo[rows], (W[rows] @ xs["total_wocoh"]), rtol=1e-6)
    assert np.all(np.asarray(mwo[rows]) <= np.asarray(mw[rows]))


def test_key_energy_table(tc):
    idx, e, mw, _ = tc
    k = pd.read_parquet(OUT / "key_energies.parquet")
    assert set(np.round(k.energy_keV.unique(), 6)) == set(np.round(config.get("key_energies_keV"), 6))
    s = k.sample(500, random_state=3)
    ie = np.array([int(np.argmin(np.abs(e - x))) for x in s.energy_keV])
    np.testing.assert_allclose(s.mu_rho_wcoh, np.asarray(mw)[s.row.to_numpy(), ie], rtol=1e-6)
    np.testing.assert_allclose(s.hvl_cm_wcoh * s.mu_wcoh, math.log(2), rtol=1e-12)
    np.testing.assert_allclose(s.tvl_cm_wcoh * s.mu_wcoh, math.log(10), rtol=1e-12)
    parts = s[["coh_mu_rho", "incoh_mu_rho", "pe_mu_rho", "pair_nuc_mu_rho", "pair_el_mu_rho"]].sum(axis=1)
    np.testing.assert_allclose(parts, s.mu_rho_wcoh, rtol=1e-6)


def test_pure_lead_entries_are_pb_relative_unity(tc, variants, profiles):
    """For any MP Pb entry, μ/ρ equals that of the reference, so areal mass ratio = 1 and thickness ratio = ρ_ref/ρ_MP."""
    idx = tc[0]
    pb = idx[idx.formula_reduced == "Pb"]
    if len(pb) == 0:
        pytest.skip("no pure-Pb entry in this (limited) build")
    rho_ref = variants.pb_density_g_cm3.iloc[0]
    p = profiles[profiles.row.isin(pb.row)]
    np.testing.assert_allclose(p.pb_relative_areal_mass, 1.0, rtol=2e-6)
    dens = pb.set_index("row").density
    np.testing.assert_allclose(p.pb_relative_thickness, rho_ref / dens.loc[p.row].to_numpy(), rtol=2e-6)


def test_single_line_profiles_closed_form(tc, variants, profiles):
    idx, e, mw, mwo = tc
    i662 = int(np.argmin(np.abs(e - 661.657)))
    v = variants[(variants.spectrum == "Cs-137")]
    rows = np.random.default_rng(11).choice(len(idx), 300, replace=False)
    for vv in v.itertuples():
        m = (mw if vv.coherent else mwo)[rows, i662].astype(np.float64) * idx.density.to_numpy()[rows]
        p = profiles[(profiles.variant_id == vv.variant_id)].set_index("row").loc[rows]
        # μ/ρ stored as float32 → compare at float32 precision
        np.testing.assert_allclose(p.thickness_req_cm, math.log(1 / vv.T_target) / m, rtol=2e-6)
    # Pb-relative ratio independent of T for a single line
    for coh in (True, False):
        ids = v[(v.coherent == coh) & (v.response == "air_kerma")].variant_id
        r = profiles[profiles.variant_id.isin(ids)].pivot(index="row", columns="variant_id", values="pb_relative_thickness")
        assert np.allclose(r.to_numpy(), r.to_numpy()[:, :1], rtol=1e-5)


def test_tube_profiles_hit_target(tc, variants, profiles):
    """Recompute T(x_req) from the stored spectrum bins and element cross sections for random rows."""
    idx = tc[0]
    bins = pd.read_parquet(OUT / "spectrum_bins.parquet")
    W = sps.load_npz(paths.INTERIM / "mp" / "mass_fractions.npz").tocsr()
    from shieldmap.physics import response
    rng = np.random.default_rng(5)
    for vv in variants[variants.source_kind == "tube"].sample(6, random_state=1).itertuples():
        b = bins[bins.spectrum == vv.spectrum]
        xs = mixture.element_mu_rho_at(b.energy_keV.to_numpy())["total_wcoh" if vv.coherent else "total_wocoh"]
        w = b.fluence.to_numpy() * response.response(vv.response, b.energy_keV.to_numpy())
        rows = rng.choice(len(idx), 50, replace=False)
        mu = (W[rows] @ xs) * idx.density.to_numpy()[rows, None]
        p = profiles[profiles.variant_id == vv.variant_id].set_index("row").loc[rows]
        lt = transmission.log_transmission(mu, w, p.thickness_req_cm.to_numpy().astype(np.float64))
        np.testing.assert_allclose(lt, math.log(vv.T_target), atol=5e-6)   # float32 storage of x


def test_envelope_dominates_elements(tc):
    env = pd.read_parquet(OUT / "envelope.parquet")
    xs = mixture.load_element_xsec()
    np.testing.assert_allclose(env.max_mu_rho_wcoh, xs["total_wcoh"].max(axis=0), rtol=1e-12)
    idx, e, mw, _ = tc
    rows = np.random.default_rng(9).choice(len(idx), 2000, replace=False)
    assert np.all(np.asarray(mw[rows]) <= env.max_mu_rho_wcoh.to_numpy()[None, :] * (1 + 1e-6))
