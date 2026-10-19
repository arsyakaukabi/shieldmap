"""Stage 4a the analysis4/xs.py: the XCOM path of the cross-section sensitivity reproduces the map
(Table C DIAG profiles, refsets single-line prices) before any xraylib number is used; xraylib element tables
behave as documented (CS_Total ≈ XCOM away from the listed elements; Kissel used only for Z ≥ 26)."""
from __future__ import annotations

import numpy as np
import pytest

from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common, xs

pytest.importorskip("xraylib")
needs_s = pytest.mark.skipif(not (common.P4 / "S" / "bands.parquet").exists(),
                             reason="run python -m shieldmap.analyze4a first")
needs_c = pytest.mark.skipif(not (c3.core.P2 / "profiles_areal.npy").exists(), reason="Phase 2 profile cache absent")


@needs_c
def test_xcom_path_reproduces_table_c_diag_profiles():
    """A few R2 ∧ B4 lanthanide rows and two textbook rows, all 36 primary DIAG variants, float32 precision."""
    d = xs.diag_rows()
    rows = np.concatenate([d["ln_rows"][:: max(1, d["ln_rows"].size // 6)][:6], d["textbook"].row.to_numpy()[:2]])
    q = xs.diag_requirements("xcom", rows)
    v = xs.diag_variants()
    ref = np.asarray(c3.profiles("areal")[:, v.variant_id.to_numpy()], dtype=np.float64)[rows]
    assert np.max(np.abs(q / ref - 1.0)) <= xs.TOL_PROFILE_REL


@needs_c
def test_xcom_tstar_reproduces_table_c():
    v = xs.diag_variants()
    assert np.max(np.abs(xs.pb_tstar("xcom", v) / v.T_target.to_numpy(float) - 1.0)) <= xs.TOL_TSTAR_REL


@needs_s
def test_xcom_path_reproduces_step2_single_line_prices():
    vm = xs.validate_mono(xs.mono_prices("xcom"))
    assert vm["pass"], vm
    assert vm["n_frontier_compound_mismatch"] == 0


@needs_s
def test_pb_window_headline_value():
    """Narrow-beam PB|B4 areal in the 88.5–90.5 keV window: Pb → Tl, 2.77 % (headline H55)."""
    mp = xs.mono_prices("xcom")
    w = mp[(mp.label == "PB|B4") & (mp.basis == "areal") & mp.energy_keV.between(88.5, 90.5)]
    r = w.loc[w.price.idxmax()]
    assert (r.formula_den, r.formula_num) == ("Pb", "Tl")
    assert abs(r.price - 0.0277) < 5e-5


def test_xraylib_total_matches_xcom_for_pb_and_kissel_only_heavy():
    e = np.array([30.0, 60.0, 89.0, 150.0])
    gi = np.array([c3.energy_index(x) for x in e])
    xc, _ = xs.element_table("xcom", e, gi)
    tot, fb = xs.element_table("xrl_total", e, gi)
    kis, _ = xs.element_table("xrl_kissel", e, gi)
    assert np.max(np.abs(tot[81] / xc[81] - 1)) < 1e-3                    # Pb: same underlying data
    assert np.array_equal(kis[: xs.KISSEL_ZMIN - 1], tot[: xs.KISSEL_ZMIN - 1])   # light elements: CS_Total
    assert 99 in fb and 100 in fb                                          # Es, Fm keep XCOM
    assert np.array_equal(tot[98:], xc[98:])
