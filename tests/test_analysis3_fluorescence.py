"""K- and L-fluorescence escape correction — validations
(a)–(f), the E₁ primary kernel, L lines, the air gap and API checks."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from shieldmap.analysis3 import common, fluorescence as F


def _row(formula):
    from shieldmap.physics import mixture
    return mixture.fractions_from_formula(formula)[None, :]


# ---------------------------------------------------------------------------------------------
# escape kernel
# ---------------------------------------------------------------------------------------------
def test_a_opaque_fluorescence_gives_zero_escape():
    """(a) μ_fluor → ∞ ⇒ zero escape, for any primary attenuation and thickness."""
    for a in (0.1, 3.0, 50.0):
        for t in (0.01, 0.3, 2.0):
            y = F.slab_yield(1.0, a, 1e12, t, escape="E2")
            assert 0.0 <= y < 1e-12
            assert F.slab_yield(1.0, a, 1e12, t, escape="E1") < 1e-10
    # and monotone: more opaque → less escape
    for esc in ("E1", "E2"):
        ys = [F.slab_yield(1.0, 5.0, b, 0.3, escape=esc) for b in (0.1, 1.0, 10.0, 100.0, 1000.0)]
        assert np.all(np.diff(ys) < 0)


def test_b_thin_slab_limit():
    """(b) E₂ (planar current): Y ≈ ½ ω p τ_K t as t → 0 (relative deviation O(t)). The E₁ (fluence) kernel
    has no finite thin-slab limit (log divergence at grazing angles); its α = 0 closed form is tested below."""
    tau, a, b = 2.0, 8.0, 5.0
    for t in (1e-6, 1e-5):
        y = F.slab_yield(tau, a, b, t, omega_p=0.9, escape="E2")
        assert abs(y / (0.5 * 0.9 * tau * t) - 1) < 50 * (a + b) * t
    assert abs(F.escape_kernel(0.0, 0.0, "E2") - 0.5) < 1e-14


def test_e1_kernel_closed_form_at_zero_primary_attenuation():
    """G_E1(0, β) = ½ ∫₀¹ (1 − e^{−β/c}) / β dc = (1 − E₂(β)) / (2β)."""
    from scipy.special import expn
    for be in (0.01, 0.1, 1.0, 7.0, 60.0, 900.0):
        ref = (1.0 - expn(2, be)) / (2.0 * be)
        assert abs(F.escape_kernel(0.0, be, "E1") / ref - 1) < 1e-5


def test_e_kernel_matches_independent_double_integral_and_monte_carlo():
    """(e) slab integral vs a from-scratch double quadrature over (depth, direction) and a Monte Carlo."""
    a, b, t = 4.0, 6.5, 0.35                           # g/cm² regime of the DIAG solutions
    y = F.slab_yield(1.0, a, b, t, escape="E2")
    ref, _ = integrate.dblquad(lambda c, s: math.exp(-a * s) * 0.5 * math.exp(-b * (t - s) / c),
                               0.0, t, 0.0, 1.0, epsabs=0, epsrel=1e-11)
    assert abs(y / ref - 1) < 1e-7
    y1 = F.slab_yield(1.0, a, b, t, escape="E1")
    ref1, _ = integrate.dblquad(lambda c, s: math.exp(-a * s) * 0.5 * math.exp(-b * (t - s) / c) / c,
                                0.0, t, 0.0, 1.0, epsabs=0, epsrel=1e-10)
    assert abs(y1 / ref1 - 1) < 1e-5
    # Monte Carlo: photons interact at depth ~ Exp(a); a fraction τ_K/a of interactions is a K absorption
    # (here τ_K = a, i.e. every interaction counts); isotropic emission; forward escape with e^{−b(t−s)/c}
    rng = np.random.default_rng(20261012)
    n = 4_000_000
    s = rng.exponential(1.0 / a, n)
    inside = s < t
    cos = rng.uniform(-1.0, 1.0, n)
    fwd = inside & (cos > 0)
    p = np.zeros(n)
    p[fwd] = np.exp(-b * (t - s[fwd]) / cos[fwd])
    mc, se = p.mean(), p.std() / math.sqrt(n)
    y_full = F.slab_yield(a, a, b, t, escape="E2")     # τ_K = a: photons crossing the back face (current)
    assert abs(mc - y_full) < 4 * se
    assert se / mc < 0.01
    # fluence behind the back face: each crossing photon contributes 1/cos θ (E₁ kernel, the primary)
    p1 = np.zeros(n)
    p1[fwd] = p[fwd] / cos[fwd]
    mc1, se1 = p1.mean(), p1.std() / math.sqrt(n)
    y1_full = F.slab_yield(a, a, b, t, escape="E1")
    assert abs(mc1 - y1_full) < 4 * se1
    assert se1 / mc1 < 0.01


def test_kernel_quadrature_against_adaptive_reference():
    def ref(al, be):
        f = lambda c: (math.exp(-al) - math.exp(-be / c)) / (be / c - al) if abs(be / c - al) > 1e-12 else math.exp(-al)
        pts = [p for p in (be / al, be, 0.01, 0.1) if 0 < p < 1]
        return 0.5 * integrate.quad(f, 0, 1, points=pts, limit=400, epsabs=0, epsrel=1e-12)[0]
    for al in (1e-3, 0.5, 3.0, 20.0, 300.0):
        for be in (0.01, 0.3, 2.0, 15.0):
            assert abs(F.escape_kernel(al, be, "E2") / ref(al, be) - 1) < 1e-6


def test_e1_kernel_quadrature_against_adaptive_reference():
    """E₁ (primary): 32-node quadrature within 1e-5 of adaptive quadrature for β ≥ 0.01."""
    def ref(al, be):
        f = lambda c: ((math.exp(-al) - math.exp(-be / c)) / (be / c - al) if abs(be / c - al) > 1e-12
                       else math.exp(-al)) / c
        pts = sorted({p for p in (be / al, be, 0.01, 0.1, 1e-3, 1e-4) if 0 < p < 1})
        return 0.5 * integrate.quad(f, 0, 1, points=pts, limit=800, epsabs=0, epsrel=1e-12)[0]
    for al in (1e-3, 0.5, 3.0, 20.0, 300.0):
        for be in (0.01, 0.3, 2.0, 15.0, 100.0, 1000.0):
            assert abs(F.escape_kernel(al, be, "E1") / ref(al, be) - 1) < 1e-5


def test_fluence_kernel_exceeds_planar_current_and_is_the_default():
    for al, be in ((1.0, 0.5), (5.0, 2.0), (0.1, 8.0)):
        assert F.escape_kernel(al, be, "E1") > F.escape_kernel(al, be, "E2")
        assert F.escape_kernel(al, be) == F.escape_kernel(al, be, "E1")
    assert F.ESCAPE_PRIMARY == "E1"
    e, phi = common.spectrum("RQR6_th12")
    assert F.FluorescenceModel(_row("Gd2O3"), e, phi).escape == "E1"


def test_air_gap_factor_matches_double_integral_and_reduces_escape():
    a, b, t, air = 4.0, 6.5, 0.35, 0.3
    for esc, wfun in (("E1", lambda c: 1.0 / c), ("E2", lambda c: 1.0)):
        y = F.slab_yield(1.0, a, b, t, escape=esc, air=air)
        ref, _ = integrate.dblquad(
            lambda c, s: math.exp(-a * s) * 0.5 * math.exp(-(b * (t - s) + air) / c) * wfun(c),
            0.0, t, 0.0, 1.0, epsabs=0, epsrel=1e-11)
        assert abs(y / ref - 1) < 1e-6
        assert y < F.slab_yield(1.0, a, b, t, escape=esc)
        assert F.slab_yield(1.0, a, b, t, escape=esc, air=0.0) == F.slab_yield(1.0, a, b, t, escape=esc)


# ---------------------------------------------------------------------------------------------
# K-shell data
# ---------------------------------------------------------------------------------------------
def test_kshell_data_xcom_vs_xraylib():
    for Z in (50, 56, 63, 65, 68, 74, 82, 83):
        k = F.k_shell_data(Z)
        assert abs(k.edge_keV / k.edge_keV_xraylib - 1) < 2e-3
        assert abs(k.J_xcom / k.J_xraylib - 1) < 0.05
        assert 0.85 < k.omega_K < 0.97
        assert 0.98 < sum(k.group_prob) <= 1.0 + 1e-12
        assert all(e < k.edge_keV for e in k.group_energy_keV)
    assert F.k_shell_data(1) is None


# ---------------------------------------------------------------------------------------------
# effective transmission and the solver
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rqr6():
    return common.spectrum("RQR6_th12")


def test_c_thick_slab_transmission_vanishes(rqr6):
    e, phi = rqr6
    m = F.FluorescenceModel(np.vstack([_row("TbSb"), F.pb_row()]), e, phi)
    T = m.transmission(np.array([60.0, 60.0]))
    assert np.all(T < 1e-12)
    Tg = np.stack([m.transmission(np.full(2, t)) for t in np.geomspace(0.01, 5, 40)], axis=1)
    assert np.all(np.diff(Tg, axis=1) < 0)          # T_eff decreasing (Brent bracket is unique)
    _, Tp, Tf, _ = m.transmission(np.array([0.3, 0.3]), parts=True)
    assert 0 < Tf[0] < Tp[0]
    assert Tf[1] > 0.0                               # Pb L lines (9–15.8 keV)
    mk = F.FluorescenceModel(np.vstack([_row("TbSb"), F.pb_row()]), e, phi, l_shell=False)
    _, _, TfK, _ = mk.transmission(np.array([0.3, 0.3]), parts=True)
    assert TfK[1] == 0.0                             # Pb K edge (88 keV) is above the 80 kV endpoint
    assert 0 < TfK[0] < Tf[0]


def test_d_fluorescence_off_reproduces_table_c(rqr6):
    """(d) narrow-beam solve (fluorescence off) = Table C areal mass to ≤ 1e-6 relative."""
    rng = np.random.default_rng(7)
    base = np.flatnonzero(common.base_mask("R2"))
    rows = np.sort(rng.choice(base, 40, replace=False))
    W = common.mass_fractions()[rows]
    P = common.profiles("areal")
    v = F._diag_variants()
    for sp in ("RQR6_th12", "RQR9_th16"):
        e, phi = common.spectrum(sp)
        m = F.FluorescenceModel(W, e, phi, fluorescence=False)
        for r in v[v.spectrum == sp].itertuples():
            q = m.solve_effective(r.T_target)
            assert np.max(np.abs(q / np.asarray(P[rows, r.variant_id], float) - 1)) <= 1e-6


def test_f_no_k_edge_in_beam_gives_exactly_zero_correction():
    """(f) constituents without a K edge inside the spectrum and without L lines ≥ 3.2 keV ⇒ q_eff == q_narrow
    exactly; U has no K edge in a 70 kV beam but emits L lines (13.6–20 keV)."""
    e, phi = common.spectrum("RQR5_th12")             # 70 kV: U (K 115.6 keV) and W (69.5 keV) edges
    W = np.vstack([_row("SiO2"), _row("Fe2O3"), _row("UO2"), _row("CaCO3")])
    mk = F.FluorescenceModel(W, e, phi, l_shell=False)
    assert not mk.has_fluor[[0, 1, 2, 3]].any()
    m = F.FluorescenceModel(W, e, phi)
    assert list(m.has_fluor) == [False, False, True, False]
    r = F.solve_areal_mass_effective(W, 0.0143, spectrum="RQR5_th12")
    assert np.array_equal(r["q_eff"][[0, 1, 3]], r["q_narrow"][[0, 1, 3]])
    assert np.all(r["T_fl"][[0, 1, 3]] == 0.0)
    assert r["q_eff"][2] > r["q_narrow"][2]
    rk = F.solve_areal_mass_effective(W, 0.0143, spectrum="RQR5_th12", l_shell=False)
    assert np.array_equal(rk["q_eff"], rk["q_narrow"])


def test_pb_needs_more_than_nominal_at_rqr9_and_like_for_like_is_consistent():
    v = F._diag_variants()
    r = v[(v.spectrum == "RQR9_th12") & np.isclose(v.pb_mm, 0.35)].iloc[0]
    e, phi = common.spectrum("RQR9_th12")
    m = F.FluorescenceModel(F.pb_row(), e, phi)
    qn = m.solve_narrow(r.T_target)[0]
    assert abs(qn / r.pb_areal - 1) < 1e-8          # narrow beam: Pb needs its nominal areal mass
    qe = m.solve_effective(r.T_target)[0]
    assert qe > qn * 1.001
    T_lfl = m.transmission(np.array([r.pb_areal]))[0]
    assert T_lfl > r.T_target
    assert abs(m.solve_effective(T_lfl)[0] / r.pb_areal - 1) < 1e-8   # like-for-like: Pb = nominal


def test_mono_line_matches_closed_form_and_fluorescence_raises_requirement():
    W = np.vstack([_row("Er2O3"), _row("SiO2")])
    r = F.mono_effective(W, 60.0, 0.01)
    from shieldmap.physics import mixture
    mu = (W @ mixture.element_mu_rho_at(np.array([60.0]))["total_wcoh"])[:, 0]
    assert np.allclose(r["q_narrow"], math.log(100) / mu, rtol=1e-9)
    assert r["q_eff"][0] > r["q_narrow"][0] and r["q_eff"][1] == r["q_narrow"][1]
    assert r["mu_rho_eff"][0] < r["mu_rho_narrow"][0]


def test_kerma_fraction_bounds(rqr6):
    e, phi = rqr6
    m = F.FluorescenceModel(_row("TbSb"), e, phi)
    thick = m.kerma_fraction_K(None)
    at = m.kerma_fraction_K(np.array([0.3]))
    tot_thick = sum(v[1][0] for v in thick.values())
    assert set(thick) == {51, 65}
    assert 0.3 < tot_thick < 1.0
    assert all(0 < v[1][0] < 1 for v in at.values())


# ---------------------------------------------------------------------------------------------
# L lines (the hardening revision)
# ---------------------------------------------------------------------------------------------
def test_l_groups_are_single_subshell_and_narrow():
    for Z in (57, 64, 74, 82, 83, 92):
        gs = F.l_shell_groups(Z)
        assert gs
        for g in gs:
            assert g.span_keV[0] >= F.L_CUTOFF_KEV
            assert g.span_keV[1] <= g.span_keV[0] * (1 + F.L_GROUP_TOL) + 1e-12
            assert g.span_keV[0] <= g.energy_keV <= g.span_keV[1]
            assert all(n.startswith(g.subshell) for n in g.lines)
        for sub in ("L1", "L2", "L3"):
            assert sum(g.rate for g in gs if g.subshell == sub) <= 1.0 + 1e-9


def test_kissel_k_production_agrees_with_xcom_k_production():
    """Cross-check of the two production routes for K lines (xraylib Kissel cascade vs XCOM pe(1−1/J)ωp):
    agreement within 3 % between 1.02 and 2 E_K (the K share of photoabsorption is not constant)."""
    import xraylib as xrl
    for Z in (50, 57, 64, 74, 82):
        k = F.k_shell_data(Z)
        for f in (1.02, 1.3, 2.0):
            E = k.edge_keV * f
            pe = F._xs_at([E])["pe"][Z - 1, 0]
            mine = pe * (1 - 1 / k.J_xcom) * k.omega_K * sum(k.group_prob)
            kis = sum(xrl.CS_FluorLine_Kissel_Cascade(Z, getattr(xrl, f"{n}_LINE"), E) for n, _, _ in k.lines)
            assert abs(kis / mine - 1) < 0.03


def test_l_line_yield_matches_direct_depth_integral():
    """Pb, 50 keV line source: L-only escaping response from the model vs an independent depth integral of
    ½E₁(b(t − s)) per group (scipy exp1, adaptive quadrature in depth)."""
    from scipy.special import exp1
    from shieldmap.physics import response as R
    e0, t = 50.0, 0.12
    m = F.FluorescenceModel(F.pb_row(), [e0], [1.0])
    fl = m.fluorescence_parts(np.array([t]), shells=("L",))[82][0]
    groups, sig = F.l_production(82, [e0])
    a = float(m.a[0, 0])
    ref = 0.0
    for g, grp in enumerate(groups):
        b = float(F._xs_at([grp.energy_keV])["total_wcoh"][81, 0])
        val, _ = integrate.quad(lambda s: math.exp(-a * s) * 0.5 * exp1(b * (t - s)), 0, t, limit=400,
                                points=[t * (1 - 1e-3)], epsabs=0, epsrel=1e-10)
        ref += sig[g, 0] * val * R.response("air_kerma", np.array([grp.energy_keV]))[0]
    ref /= R.response("air_kerma", np.array([e0]))[0]
    assert abs(fl / ref - 1) < 1e-4


def test_l_lines_raise_pb_requirement_at_rqr5_and_air_gap_lowers_it():
    v = F._diag_variants()
    r = v[(v.spectrum == "RQR5_th12") & np.isclose(v.pb_mm, 0.25)].iloc[0]
    e, phi = common.spectrum("RQR5_th12")
    q = {}
    for lab, kw in (("K", {"l_shell": False}), ("KL", {}), ("KL10", {"air_cm": 10.0}), ("KL100", {"air_cm": 100.0})):
        q[lab] = F.FluorescenceModel(F.pb_row(), e, phi, **kw).solve_effective(r.T_target)[0]
    assert q["K"] == pytest.approx(r.pb_areal, rel=1e-8)          # no K edge below 70 kV
    assert q["KL"] > q["KL10"] > q["KL100"] > q["K"]
    with pytest.raises(ValueError):
        F.FluorescenceModel(F.pb_row(), e, phi, l_cutoff_keV=2.0)


def test_air_definition_is_srd126_dry_air():
    d = F.air_definition()
    assert d["density_g_cm3"] == pytest.approx(1.205e-3)
    assert abs(d["W"].sum() - 1) < 1e-12 and d["W"][6] > 0.75        # N


def test_node_rule_accuracy_and_row_kernel():
    """NODE_RULE bounds (vs 128 nodes, α ≤ 20) and escape_kernel_rows = per-row escape_kernel."""
    al = np.geomspace(1e-3, 20.0, 50)[:, None]
    bounds = {24: 1e-7, 16: 3e-8, 12: 1e-9}
    for thr, n in F.NODE_RULE:
        be = np.geomspace(thr, 1e4, 50)[None, :]
        for esc in ("E1", "E2"):
            ref = F.escape_kernel(al, be, esc, 128)
            assert np.max(np.abs(F.escape_kernel(al, be, esc, n) / ref - 1)) <= bounds[n]
    rng = np.random.default_rng(3)
    alpha = rng.uniform(0.01, 15.0, (7, 30))
    beta = np.array([0.05, 0.5, 2.0, 5.0, 40.0, 300.0, 0.31])
    G = F.escape_kernel_rows(alpha, beta, "E1", air=0.2)
    for i, b in enumerate(beta):
        ref = F.escape_kernel(alpha[i], b, "E1", n_nodes=F.nodes_for_beta(b), air=0.2)
        assert np.allclose(G[i], ref, rtol=1e-13, atol=0)      # same nodes; batched BLAS rounding only
