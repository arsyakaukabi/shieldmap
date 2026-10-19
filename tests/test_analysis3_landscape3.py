"""Stage 3 the landscape completion: synthetic fixtures with known answers; no stage 1 tables needed."""
import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis3 import landscape3 as L


def test_fit_power_law_recovers_exponent():
    Z = np.arange(6, 84, dtype=float)
    n, a, r2 = L.fit_power_law(Z, 3.7 * Z ** 4.25)
    assert n == pytest.approx(4.25, abs=1e-10)
    assert np.exp(a) == pytest.approx(3.7, rel=1e-10)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_fit_power_law_ignores_nonpositive_and_noise_reduces_r2():
    Z = np.arange(1, 101, dtype=float)
    y = 2.0 * Z ** -0.1
    y[:5] = 0.0                                          # e.g. pair below threshold
    n, _, r2 = L.fit_power_law(Z, y)
    assert n == pytest.approx(-0.1, abs=1e-10) and r2 == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    yn = Z ** 2 * np.exp(rng.normal(0, 0.3, Z.size))
    n2, _, r22 = L.fit_power_law(Z, yn)
    assert abs(n2 - 2) < 0.15 and r22 < 1.0
    assert np.isnan(L.fit_power_law(Z[:2], Z[:2])[0])   # fewer than 3 points


def test_share_quantiles_known_answer():
    # three compounds, two energies; pe shares 0.1, 0.5, 0.9 at energy 0, and 0.2 for all at energy 1
    pe = np.array([[1.0, 2.0], [5.0, 2.0], [9.0, 2.0]])
    incoh = np.array([[9.0, 8.0], [5.0, 8.0], [1.0, 8.0]])
    q = L.share_quantiles({"pe": pe, "incoh": incoh}, (0.25, 0.5, 0.75))
    np.testing.assert_allclose(q["pe"][:, 0], [0.3, 0.5, 0.7])
    np.testing.assert_allclose(q["pe"][:, 1], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(q["pe"][1] + q["incoh"][1], [1.0, 1.0])   # medians of complementary shares


def test_log_crossing_directions():
    e = np.array([10.0, 100.0, 1000.0])
    d = np.array([1.0, -1.0, 1.0])
    assert L.log_crossing(e, d, "down") == [pytest.approx(10 ** 1.5)]
    assert L.log_crossing(e, d, "up") == [pytest.approx(10 ** 2.5)]
    assert len(L.log_crossing(e, d, "any")) == 2


def test_spread_and_pair_concordance():
    assert L.spread([2.0, 2.2, 2.1]) == pytest.approx(0.1)
    assert L.pair_concordance([1, 2, 3], [10, 20, 30]) == (3, 0, 0)
    assert L.pair_concordance([1, 2, 3], [30, 20, 10]) == (0, 3, 0)
    assert L.pair_concordance([1, 2, 3], [10, 10, 30]) == (2, 0, 1)
    # only pairs whose second values differ by > 5 %: (10, 10.2) is dropped, the two others are kept
    assert L.pair_concordance([1, 2, 3], [10.2, 10.0, 30.0], 0.05) == (2, 0, 0)
    assert L.pair_concordance([1, 2, 3], [10.2, 10.0, 30.0]) == (2, 1, 0)


def _matches():
    return pd.DataFrame({
        "formula_reduced": ["A", "A", "A", "B", "B", "C"],
        "material_id": ["mp-1", "mp-2", "mp-3", "mp-4", "mp-5", "mp-6"],
        "cod_id": [100, 200, 100, 300, 300, 400],
        "cod_ids_equivalent": ["100", "200", "100,101", "300", "300", "400"],
        "rho_mp": [5.0, 5.5, 5.1, 3.0, 3.3, 7.0],
        "rho_cod": [5.2, 5.4, 5.25, 3.1, 3.1, 7.1],
        "delta": [0.0] * 6,
        "e_hull": [0.0, 0.02, 0.05, 0.0, 0.01, 0.0],
        "primary_class": ["oxide"] * 6})


def test_polymorph_table_disjoint_cod_sets():
    pt = L.polymorph_table(_matches()).set_index("formula_reduced")
    # A: mp-3 shares COD 100 with mp-1 (lower e_hull kept) → mp-1, mp-2; B: both map to COD 300 → dropped; C: one
    assert list(pt.index) == ["A"]
    a = pt.loc["A"]
    assert a.n_poly == 2 and a.material_ids == "mp-1,mp-2"
    assert a.spread_dft == pytest.approx(0.1)
    assert a.spread_cod == pytest.approx(5.4 / 5.2 - 1)
    assert (a.pairs_concordant, a.pairs_discordant) == (1, 0) and a.densest_same
    s = L.polymorph_summary(pt.reset_index())
    assert s["n_formulas"] == 1 and s["frac_pairs_concordant"] == 1.0


def test_envelope_comparison_sign():
    # compound needs 0.9 cm, best element 1.0 cm → 11.1 % advantage (survives); 1.1 cm → negative
    adv = L.survival_compare(np.array([0.9, 1.1]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(adv, [1 / 0.9 - 1, 1 / 1.1 - 1])
    assert adv[0] > 0 > adv[1]


def test_envelope_identity_mu_ratio():
    # μ ratio vs the envelope element = (μ/ρ ratio) × (ρ ratio): the decomposition used for the mechanism
    mr_c, rho_c, mr_env, rho_env = 3.1, 11.9, 2.0, 22.5
    assert (mr_c / mr_env) * (rho_c / rho_env) == pytest.approx(mr_c * rho_c / (mr_env * rho_env))


def test_mechanism_label():
    base = {"K_edge_just_below": True, "partner": "Ir", "partner_solid_density": 22.6, "edge_solid_density": 6.9}
    assert L.mechanism_label(base) == "K edge + dense partner"
    assert L.mechanism_label({**base, "partner": "N", "partner_solid_density": 1.0}) == \
        "K edge + compact packing (light partner)"
    assert L.mechanism_label({**base, "K_edge_just_below": False}) == "other"
    assert L.mechanism_label({**base, "partner": ""}).startswith("elemental entry")


def test_envelope_density_axis_counts(monkeypatch):
    """The F2: thickness scales by 1 + δ; exceedance μ/μ_env/(1 + δ) > 1, survival (1 + adv)/(1 + δ) > 1;
    elemental entries are counted apart from multi-element compounds."""
    import pandas as pd
    win = pd.DataFrame({"row": [1, 2, 3], "material_id": ["a", "b", "c"], "formula_pretty": ["A", "B", "C"],
                        "in_R2": [True, True, False], "theoretical": [False, True, False],
                        "mechanism": ["K edge + dense partner", "other", "elemental entry (DFT ρ > experimental ρ)"],
                        "max_ratio_mu_nonrad_env": [1.05, 1.01, 1.02]})
    surv = pd.DataFrame({"row": [1, 1, 2, 3], "T": [0.1, 0.1, 0.1, 0.1], "advantage_eff": [0.02, 0.04, 0.005, -0.1]})

    def fake(rows):
        n = len(rows)
        return pd.DataFrame({"row": rows, "primary_class": ["x"] * n, "nominal": 0.0, "class_median": -0.02,
                             "class_p95": 0.03, "measured_delta": [0.045, np.nan, np.nan],
                             "measured_or_nominal": [0.045, 0.0, 0.0], "measured_or_p95": [0.045, 0.03, 0.03]})
    monkeypatch.setattr(L, "envelope_deltas", fake)
    long, cnt = L.envelope_density_axis(win, surv)
    g = long.set_index(["case", "row"])
    assert g.loc[("nominal", 1), "exceeds_narrow"] and g.loc[("nominal", 2), "exceeds_narrow"]
    assert g.loc[("class_p95", 1), "exceeds_narrow"] and not g.loc[("class_p95", 2), "exceeds_narrow"]
    assert g.loc[("class_p95", 1), "survives_primary_T0.1"]             # 1.04/1.03 > 1
    assert not g.loc[("measured_or_p95", 1), "survives_primary_T0.1"]   # 1.04/1.045 < 1
    assert g.loc[("class_median", 2), "survives_primary_T0.1"]
    c = cnt.set_index(["case", "population", "scope"])
    assert c.loc[("nominal", "multi_element", "R2"), "n_entries"] == 2
    assert c.loc[("nominal", "elemental", "R3"), "n_entries"] == 1
    assert c.loc[("class_p95", "multi_element", "R2"), "exceeds_narrow"] == 1
