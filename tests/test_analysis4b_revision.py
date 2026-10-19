"""Stage 4b revision (analysis4b.revision): blend linearity, like-for-like interpolation, GAP threshold sweep."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis3 import common as c3
from shieldmap.analysis4b import revision as R

REV = R.OUT
needs_rev = pytest.mark.skipif(not (REV / "e7_gap_sweep.csv").exists(),
                               reason="run shieldmap.analysis4b.revision.run() first")


# ------------------------------------------------------------------------------------------- E2 blend line
def _v13():
    from shieldmap.physics import mixture
    v = c3.variants().set_index("variant_id").loc[13]
    e, w = c3.spectrum_weights(v.spectrum, "air_kerma")
    return v, w, mixture.element_mu_rho_at(e)["total_wcoh"]


def test_blend_mixture_rule_reproduces_table_c():
    """A compound's mass fractions × element μ/ρ, through the blend solver, give its Table C areal requirement:
    the blend line uses exactly the map's quadrature, weights and T*."""
    v, w, xs = _v13()
    u = c3.universe()
    W = c3.mass_fractions()
    prof = c3.profiles("areal")
    for mid in ("mp-1213880", "mp-23152", "mp-23262"):          # Ce5AgSn3 (B4 frontier), Bi, Bi2O3
        i = c3.row_of(mid)
        mu = np.asarray(W[i].toarray()).ravel() @ xs
        q = R._solve_areal(mu, w, float(v.T_target))[0]
        assert abs(q / float(prof[i, 13]) - 1) < 2e-6, u.formula_pretty.iat[i]


def test_blend_is_linear_in_mass_fractions():
    v, w, xs = _v13()
    X = xs[[R.ZNUM[s] - 1 for s in R.GARMENT]]
    f = np.array([0.3, 0.0, 0.0, 0.0, 0.35, 0.35])                # Sn/Ba/Gd
    mu = f @ X
    assert np.allclose(mu, 0.3 * xs[49] + 0.35 * xs[55] + 0.35 * xs[63], rtol=1e-12)
    q = R._solve_areal(mu, w, float(v.T_target))[0]
    T = (w * np.exp(-mu * q)).sum() / w.sum()
    assert abs(math.log(T) - math.log(v.T_target)) < 1e-8          # the solver hits T*
    # pure element at fraction 1 = the element's Table C requirement
    qbi = R._solve_areal(X[R.GARMENT.index("Bi")], w, float(v.T_target))[0]
    assert abs(qbi / float(c3.profiles("areal")[R._bi_row(), 13]) - 1) < 2e-6


@needs_rev
def test_blend_outputs_consistent():
    b = pd.read_csv(REV / "e2_blend_by_variant.csv")
    assert len(b) == 36
    pure = b[[f"q_pure_{s}" for s in R.GARMENT]].min(axis=1)
    assert (b.q_blend <= pure + 1e-12).all()                        # a blend is never worse than its best element
    assert (b.q_blend <= b.q_binary + 1e-12).all()
    assert ((b.blend_vs_B4_lo <= b.blend_vs_B4 + 1e-12) & (b.blend_vs_B4 <= b.blend_vs_B4_hi + 1e-12)).all()
    r = b.set_index("variant_id").loc[13]
    assert r.B4_frontier == "Ce5AgSn3" and abs(r.q_B4 - 0.2814) < 1e-4
    assert abs(r.q_Bi - 0.3808) < 1e-4
    assert r.q_blend <= 0.280267 + 1e-9                            # R2 check 4 (5 % grid): refinement can only help


# ------------------------------------------------------------------------------------------- E3 like-for-like
def test_interp_T_exact_for_quadratic_log_T():
    q = np.array([0.2, 0.25, 0.3])
    lt = -12.0 * q + 3.0 * q ** 2
    for q0 in (0.21, 0.2835, 0.29):
        assert abs(math.log(R._interp_T(q, np.exp(lt), q0)) - (-12.0 * q0 + 3.0 * q0 ** 2)) < 1e-12


def test_like_for_like_synthetic():
    """Pure exponentials: required q = μ_Pb q_Pb / μ_X exactly, so the advantage is μ_Ln/μ_Bi − 1."""
    mu = {"Pb": 12.0, "Bi": 12.5, "Ln": 14.0}
    pts = {}
    for m, k in mu.items():
        q = np.array([0.2, 0.25, 0.3, 0.35]) * 12.0 / k
        pts[m] = {"q": q, "fl": np.exp(-k * q), "fl_se": np.zeros(q.size)}
    t_pb, qa, qb, inside = R._lfl_point(pts, "Bi", "Ln", "fl", 0.2835)
    assert abs(t_pb - math.exp(-12.0 * 0.2835)) < 1e-12
    assert abs(qa - 12.0 * 0.2835 / 12.5) < 1e-9 and abs(qb - 12.0 * 0.2835 / 14.0) < 1e-9
    assert abs((qa / qb - 1) - (14.0 / 12.5 - 1)) < 1e-8 and inside


@needs_rev
def test_like_for_like_reproduces_reviewer_and_metrics():
    d = pd.read_csv(REV / "e3_like_for_like.csv").set_index(["case_id", "estimator"])
    # fixed-T values reproduce data/phase4a/mc/metrics.csv
    assert np.allclose(d.adv_fixedT, d.adv_fixedT_metrics_csv, atol=1e-9)
    # like-for-like: R2 check 3 (check3_advantages_reweighted.csv, air kerma)
    for key, val in {("c3_rqr5", "fl"): 6.753020, ("c3_rqr5", "broad"): 10.370608,
                     ("c4_rqr8", "fl"): 16.778467, ("c4_rqr8", "broad"): 22.899101}.items():
        assert abs(100 * d.loc[key, "adv_lfl"] - val) < 1e-4
        assert 0 < d.loc[key, "adv_lfl_se"] < 0.01


# ------------------------------------------------------------------------------------------- E7 threshold sweep
@needs_rev
def test_gap_sweep_monotone_and_biof_switch():
    g = pd.read_csv(REV / "e7_gap_sweep.csv")
    s = g[~g.cubic_only]
    for where, x in s.groupby("where"):
        x = x.sort_values("threshold_eV")
        assert (np.diff(x.price.to_numpy()) >= -1e-12).all(), where       # nested sets: price never falls
    c = s[s["where"] == "Cs-137 thickness"].set_index("threshold_eV")
    assert c.at[3.1, "leader"] == "BiOF" and c.at[2.5, "leader"] == "BiOF"
    assert c.at[3.2, "leader"] == "HfO2" and abs(100 * c.at[3.2, "price"] - 148.4) < 0.05   # R3 check 4
    assert abs(100 * c.at[3.1, "price"] - 134.58) < 0.01                                     # H09
    cub = g[g.cubic_only].set_index("where")
    assert cub.at["Cs-137 thickness", "leader"] == "Lu2O3" and abs(100 * cub.at["Cs-137 thickness", "price"] - 167.4) < 0.05
    assert cub.at["Co-60 thickness", "leader"] == "Lu2O3" and abs(100 * cub.at["Co-60 thickness", "price"] - 147.9) < 0.05
    g31 = s[s.threshold_eV == 3.1].set_index("where").price
    assert (cub.price >= g31.loc[cub.index] - 1e-12).all()                                  # cubic ⊂ GAP(3.1)
    bg = pd.read_csv(REV / "e7_biof_gap.csv").iloc[0]
    assert bg.material_id == "mp-23074" and abs(bg.band_gap_eV - 3.1303) < 1e-6


def test_biof_is_the_only_gap_leader_between_3_1_and_3_2():
    u = c3.universe()
    x = u[(u.formula_pretty == "BiOF") & u.in_R2].iloc[0]
    assert 3.1 <= x.band_gap < 3.2


# ------------------------------------------------------------------------------------------- E5 membership
def test_fully_oxidised_classifier():
    yes = ("Ba2SmSbO6", "BiOF", "HfO2", "Bi2O3", "BaSO4", "LaPO4", "Gd3Sb5O12", "Lu2O3", "Ba(NO3)2")
    no = ("Eu5Cd2Sb5O", "Pr3InO", "Eu4Sb2O", "LaAgSO", "Ce5AgSn3", "H5NO", "Cs3Er2I9", "BiOCl")
    assert all(R._fully_oxidised(f) for f in yes)
    assert not any(R._fully_oxidised(f) for f in no)
    assert R._fully_oxidised("BiOCl", oxyhalide=True)
