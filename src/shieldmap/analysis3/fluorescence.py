"""First-order K- and L-fluorescence escape correction (the hardening revision F2).

The hardening revision (reports/phase3/step2_fluorescence.md):
* the escape kernel is E₁ (fluence, 1/c weighting) for every response: air kerma, fluence and energy
  fluence are all fluence-weighted, and for a laterally infinite slab the uncollided fluence behind it is
  ½E₁, independent of distance. E₂ (planar current, not an air-kerma quantity) is a sensitivity only;
* L-shell fluorescence is added (all L1–L3 lines ≥ the Ar K edge, 3.2029 keV, where the stage 1 response
  spline starts; SRD 126 itself extends to 1 keV, see the omissions below), with xraylib Kissel cascade production cross sections (L photoionisation,
  Coster–Kronig, and vacancies transferred from the K shell). K lines keep the XCOM-based production;
* an optional air gap d between the back face and the kerma point (primary d = 0) attenuates every line
  by exp(−(μ/ρ)_air(E_l) ρ_air d / c) inside the direction integral.
The original text below describes the K model; the revision notes above take precedence.

A first-order correction, stated as such; not a Monte Carlo. Regenerate everything with
``python -m shieldmap.analysis3.fluorescence`` from ``shieldmap/`` (outputs in ``data/phase3/F/``,
figures ``reports/phase3/figures/F_*``).

Geometry and coordinates
------------------------
A laterally infinite slab of areal mass t (g/cm²) is irradiated at normal incidence by a uniform broad
beam; the person is behind the back face. Depth is mass depth s = ρz (g/cm²), so every quantity is
density-independent (the DIAG basis). The primary is narrow-beam (no Compton/Rayleigh build-up), exactly
as in Table C; the only secondary radiation added is K fluorescence.

Formulas
--------
Quadrature pieces k of a stage 1 spectrum (``spectrum_bins.parquet``; pieces never straddle an absorption
edge) have energy E_k and fluence Φ_k; R(E) is the response (air kerma R = E·(μ_en/ρ)_air, fluence R = 1,
energy fluence R = E). a_k = (μ/ρ)(E_k) of the compound (mixture rule, XCOM port, with or without
coherent as the variant says).

K-shell photoabsorption of constituent i (mass fraction w_i, K edge E_K,i, K jump ratio J_i)::

    τ_K,i(E) = w_i · τ_i(E) · (1 − 1/J_i)        for E ≥ E_K,i, else 0

τ_i is the element photoelectric μ/ρ (XCOM ``pe``). J_i is the XCOM photoabsorption discontinuity at the K
edge (``pe`` above / below, the same data as the map); xraylib ``JumpFactor`` is a cross-check and a
sensitivity. A K vacancy emits line l with probability ω_K,i · p_l (xraylib ``FluorYield``, ``RadRate``).
Lines are merged into four groups per element (Kα1 = KL3; Kα2 = KL2 + KL1; Kβ1,3 = KM*; Kβ2 = KN*, KO*,
KP*), at the rate-weighted mean energy; b_l = (μ/ρ)(E_l) of the compound at the group energy.

Escape toward the back face of a photon born isotropically at depth s (forward hemisphere only; c = cos θ)::

    P_esc,1(s) = ½ ∫₀¹ exp(−b (t − s)/c) dc/c = ½ E₁(b (t − s))       ("fluence", 1/c weight; PRIMARY)
    P_esc(s) = ½ ∫₀¹ exp(−b (t − s)/c) dc = ½ E₂(b (t − s))           ("planar current"; sensitivity only)

L lines (revision): Y_kgl(t) = w_Z σ_g(E_k) t G(a_k t, b_g t), σ_g = Σ_{lines in g} xraylib
CS_FluorLine_Kissel_Cascade(Z, line, E_k) (cm²/g of element Z); groups g merge lines of one subshell whose
energies lie within L_GROUP_TOL of the group's lowest line (rate ratios within a subshell are energy-
independent, so the merge only averages the line energy). Air gap: G is evaluated with the extra factor
exp(−β_air/c), β_air = (μ/ρ)_air(E_l) ρ_air d (air composition and density: NIST SRD 126 Table 2, the
archived stage 1 page; μ/ρ by the XCOM mixture rule, with/without coherent as the variant).

Escaping line-l photons per incident photon of energy E_k::

    Y_kl(t) = ω p_l τ_K(E_k) ∫₀ᵗ e^{−a_k s} P_esc(s) ds = ω p_l τ_K(E_k) · t · G(a_k t, b_l t)
    G(α, β) = ½ ∫₀¹ h(α, β/c) dc,      h(α, γ) = (e^{−α} − e^{−γ}) / (γ − α) = e^{−min(α,γ)} φ(|γ − α|)
    φ(x) = (1 − e^{−x}) / x      (φ(0) = 1; evaluated with expm1, so it is stable for all α, γ ≥ 0)

(The s integral is done exactly: ∫₀ᵗ e^{−as} e^{−(b/c)(t−s)} ds = t·h(at, bt/c).) The direction integral is
Gauss–Legendre in u with c = u² (32 nodes; relative error ≤ 1e-7 for β ≥ 0.01 against adaptive quadrature;
the E₁ variant uses the weight 1/c = 1/u²). Limits: G → ½ as t → 0 (thin slab: Y ≈ ½ ω p τ_K t);
G → 0 as b → ∞; Y → 0 as t → ∞.

Effective transmission (normalised to the unattenuated response, as Table C)::

    T_eff(t) = [ Σ_k Φ_k R(E_k) e^{−a_k t} + Σ_k Φ_k Σ_{i,l} Y_kil(t) R(E_il) ] / Σ_k Φ_k R(E_k)

The required areal mass solves ln T_eff(t) = ln T* with Brent's method (``transmission.brentq_vec``) on the
exact bracket [q_narrow, hi] (T_eff ≥ T_prim, so q_eff ≥ q_narrow). With fluorescence off the solver is
``transmission.solve_thickness`` on μ/ρ and reproduces Table C areal mass.

Targets
-------
fixed-T (primary)   T* = Table C ``T_target`` (narrow-beam air-kerma transmission of 0.25/0.35/0.50 mm Pb):
                    the protection level is a kerma level, so Pb itself needs more than its nominal areal
                    mass once its own K fluorescence is counted.
like-for-like       T*_eff = T_eff of Pb at its nominal areal mass (pb_thickness_cm × 11.34 g/cm³).

Which emitters
--------------
Constituents with a K edge in ``WINDOW_KEV`` = 20–150 keV; a 10–150 keV sensitivity adds
Ga–Mo (K lines 9–17.5 keV).

Omissions (each makes the escape of a given material a lower bound; NOT a bound on the sign of the error
of any ratio between materials, because the omitted terms differ between materials —)
------------------------------------------------------------------------------------------------------
* Compton/Rayleigh build-up of the primary: omitted for every material (narrow-beam primary). Raises the
  transmitted kerma for all materials, most for materials diluted with low-Z elements.
* Fluorescence of fluorescence (secondary fluorescence when an escaping line is absorbed above another
  constituent's edge, e.g. lanthanide Kα above the K edges of I, Sb, Sn or Pd): omitted; it acts
  specifically on lanthanide + partner compounds.
* M lines and L lines below 3.2029 keV: omitted (uncertain M-shell yields; 2–3 keV photons are absorbed by
  any cover and a few cm of air). They add about 1–2 % to Bi, Pb and W (the hardening revision).
* Secondary fluorescence (re-emission after absorption of a fluorescence photon in the slab): omitted.
* Scattered fluorescence photons are counted as removed (attenuation by total μ/ρ incl. coherent when the
  variant includes it).
* K lines that xraylib does not tabulate (Σ p_l = 0.986–1.000 for Z ≤ 100) are omitted (≤ 1.4 % of the
  K term).
* (1 − 1/J) treats the K share of photoabsorption as energy-independent. The true share (xraylib
  CS_Photo_Partial) is slightly lower at the edge and 1–2 % higher at 1.3–2 E_K; the net effect on the
  required areal mass is about +0.05 to +0.2 % (#7; e.g. LaBi 1.04866 vs 1.04818,
  PrSbPd 1.11671 vs 1.11515, Sn 1.18731 vs 1.18470 under E₂, K only). Not corrected.
* Backward-escaping fluorescence (toward the source) is ignored: it does not reach the person.
* Laterally infinite, uniformly irradiated slab: for a small field and a distant point detector the
  fluorescence contribution is smaller (narrow-beam limit → 0); the correction is the broad-beam value.
  The geometry is mixed: broad-beam for the fluorescence, narrow-beam (no build-up) for the primary.

Public API
----------
k_shell_data(Z), kshell_table(), escape_kernel(α, β), slab_yield(...), FluorescenceModel(W, energies,
fluence, ...), solve_areal_mass_effective(W, T, spectrum=... | energies=..., fluence=...),
mono_effective(W, e_keV, T), run().
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sps

from shieldmap import config, paths
from shieldmap.analysis3 import common
from shieldmap.physics import mixture, response as resp_mod, transmission

OUT = common.P3 / "F"
WINDOW_KEV = (20.0, 150.0)
WINDOW_SENS_KEV = (10.0, 150.0)
SEED = 20261012
TOP_K = 200
N_RANDOM = 2000
N_NODES = 32
KERNEL_CHUNK = 4_000_000                 # elements of the (pairs × nodes) work array
TEXTBOOK = ("W", "Bi", "Sn", "Sb2O3", "BaSO4", "Gd2O3", "WO3", "Bi2O3", "PbO", "Pb")
TEXTBOOK_NO_PB = tuple(f for f in TEXTBOOK if f not in ("Pb", "PbO"))
DISPLAY = ("RQR6_th12", 0.35)
F3_THRESHOLD = 0.10
PB_Z = 82
ESCAPE_PRIMARY = "E1"
L_CUTOFF_KEV = 3.2029           # Ar K edge, where the stage 1 response spline starts (see docstring)
L_GROUP_TOL = 0.02              # relative energy span of a merged L line group (one subshell)
AIR_GAP_CM = 0.0                # primary: kerma point at the back face
MODEL_VERSION = "F2: E1 primary, K+L lines, air gap, node rule"     # part of the checkpoint key
SENS_OPTS = {                   # The fluorescence module sensitivities (union of B4 top-200 + textbook + pure elements)
    "E2": {"escape": "E2"},                                   # planar current, not an air-kerma quantity
    "K_only": {"l_shell": False},                             # E1, no L lines
    "E2_K_only": {"escape": "E2", "l_shell": False},          # the original fluorescence-module primary model
    "J_xraylib": {"jump": "xraylib"},
    "window10": {"window": (10.0, 150.0)},
    "cutoff4": {"l_cutoff_keV": 4.0},
    "air10cm": {"air_cm": 10.0},
    "air100cm": {"air_cm": 100.0},
}
LINE_GROUPS = {
    "KA1": ("KL3",),
    "KA2": ("KL1", "KL2"),
    "KB13": ("KM1", "KM2", "KM3", "KM4", "KM5"),
    "KB2": tuple([f"KN{i}" for i in range(1, 8)] + [f"KO{i}" for i in range(1, 8)] + [f"KP{i}" for i in range(1, 6)]),
}


def xtol_rel() -> float:
    return float(config.get("solver.xtol_rel"))


# =============================================================================================
# K-shell data
# =============================================================================================
@lru_cache(maxsize=1)
def _xcom_elements() -> dict:
    mixture.element_mu_rho_at(np.array([100.0]))          # loads and caches the XCOM port
    return mixture._XCOM_CACHE["els"]


def jump_ratio_xcom(Z: int) -> float:
    """K jump ratio from the XCOM photoabsorption table: pe(E_K⁺) / pe(E_K⁻) (file points at the edge)."""
    el = _xcom_elements()[Z]
    if not el.edge_labels or el.edge_labels[0] != "K":
        return float("nan")
    i = el.edge_index[0]
    if not abs(el.e[i] - el.e[i - 1]) <= 1.0:          # XCOM stores the below-edge point ≤ 0.1 eV lower
        raise ValueError(f"Z={Z}: K-edge points do not coincide ({el.e[i - 1]}, {el.e[i]} eV)")
    return float(el.pe[i] / el.pe[i - 1])


@dataclass(frozen=True)
class KShell:
    Z: int
    symbol: str
    edge_keV: float                  # XCOM K edge
    edge_keV_xraylib: float
    J_xcom: float
    J_xraylib: float
    omega_K: float
    groups: tuple = ()               # names
    group_energy_keV: tuple = ()
    group_prob: tuple = ()           # Σ p_l within the group (radiative-rate fractions, not × ω)
    group_span_keV: tuple = ()       # (min, max) line energy inside the group
    lines: tuple = field(default=())  # (name, energy keV, rate) of every tabulated line

    def jump(self, source: str = "xcom") -> float:
        return self.J_xcom if source == "xcom" else self.J_xraylib


@lru_cache(maxsize=128)
def k_shell_data(Z: int) -> KShell | None:
    """K-shell data for element Z (None when XCOM has no K edge, i.e. H and He)."""
    import xraylib as xrl
    el = _xcom_elements()[Z]
    if not el.edge_labels or el.edge_labels[0] != "K":
        return None
    sym = common.elements_table().set_index("Z").loc[Z, "symbol"]
    lines = []
    for grp, names in LINE_GROUPS.items():
        for n in names:
            macro = getattr(xrl, f"{n}_LINE", None)
            if macro is None:
                continue
            try:
                r = xrl.RadRate(Z, macro); e = xrl.LineEnergy(Z, macro)
            except ValueError:
                continue
            if r > 0 and e > 0:
                lines.append((grp, n, float(e), float(r)))
    gn, ge, gp, gs = [], [], [], []
    for grp in LINE_GROUPS:
        sel = [x for x in lines if x[0] == grp]
        if not sel:
            continue
        p = sum(x[3] for x in sel)
        gn.append(grp); gp.append(p)
        ge.append(sum(x[2] * x[3] for x in sel) / p)
        gs.append((min(x[2] for x in sel), max(x[2] for x in sel)))
    try:
        omega = float(xrl.FluorYield(Z, xrl.K_SHELL))
        jx = float(xrl.JumpFactor(Z, xrl.K_SHELL))
        ekx = float(xrl.EdgeEnergy(Z, xrl.K_SHELL))
    except ValueError:
        omega, jx, ekx = 0.0, float("nan"), float("nan")
    return KShell(Z=Z, symbol=sym, edge_keV=float(el.edge_energies[0]) / 1000.0, edge_keV_xraylib=ekx,
                  J_xcom=jump_ratio_xcom(Z), J_xraylib=jx, omega_K=omega, groups=tuple(gn),
                  group_energy_keV=tuple(ge), group_prob=tuple(gp), group_span_keV=tuple(gs),
                  lines=tuple((n, e, r) for _, n, e, r in lines))


def kshell_table() -> pd.DataFrame:
    rows = []
    for Z in range(1, 101):
        k = k_shell_data(Z)
        if k is None:
            continue
        g = dict(zip(k.groups, zip(k.group_energy_keV, k.group_prob)))
        rows.append({"Z": Z, "symbol": k.symbol, "E_K_keV_xcom": k.edge_keV, "E_K_keV_xraylib": k.edge_keV_xraylib,
                     "J_xcom": k.J_xcom, "J_xraylib": k.J_xraylib,
                     "J_rel_diff": k.J_xraylib / k.J_xcom - 1.0 if k.J_xcom > 0 else np.nan,
                     "omega_K": k.omega_K, "sum_p_lines": float(sum(k.group_prob)),
                     "in_window_20_150": WINDOW_KEV[0] <= k.edge_keV <= WINDOW_KEV[1],
                     **{f"E_{n}_keV": g.get(n, (np.nan, 0))[0] for n in LINE_GROUPS},
                     **{f"p_{n}": g.get(n, (np.nan, 0.0))[1] for n in LINE_GROUPS},
                     "lines": ";".join(f"{n}:{e:.4f}:{r:.5f}" for n, e, r in k.lines)})
    return pd.DataFrame(rows)


# =============================================================================================
# L-shell data (revision)
# =============================================================================================
@dataclass(frozen=True)
class LGroup:
    subshell: str                 # 'L1' | 'L2' | 'L3'
    name: str                     # e.g. 'L3:L3M5+L3M4'
    energy_keV: float             # radiative-rate-weighted mean line energy
    span_keV: tuple               # (min, max) line energy in the group
    lines: tuple                  # xraylib macro names
    rate: float                   # Σ radiative rate within the subshell (fraction of that subshell's lines)


@lru_cache(maxsize=1)
def _l_line_macros() -> tuple:
    import re
    import xraylib as xrl
    return tuple(sorted(n for n in dir(xrl) if re.fullmatch(r"L[123][M-Q]\d_LINE", n)))


@lru_cache(maxsize=256)
def l_shell_groups(Z: int, cutoff_keV: float = L_CUTOFF_KEV, tol: float = L_GROUP_TOL) -> tuple:
    """L line groups of element Z with energy ≥ cutoff (xraylib LineEnergy, RadRate)."""
    import xraylib as xrl
    by_sub: dict = {}
    for n in _l_line_macros():
        try:
            e = float(xrl.LineEnergy(Z, getattr(xrl, n))); r = float(xrl.RadRate(Z, getattr(xrl, n)))
        except ValueError:
            continue
        if e >= cutoff_keV and r > 0:
            by_sub.setdefault(n[:2], []).append((e, n[:-5], r))
    out = []
    for sub in sorted(by_sub):
        lines = sorted(by_sub[sub])
        cur: list = []
        for ln in lines + [None]:
            if ln is not None and (not cur or ln[0] <= cur[0][0] * (1 + tol)):
                cur.append(ln); continue
            if cur:
                r = sum(x[2] for x in cur)
                out.append(LGroup(subshell=sub, name=f"{sub}:" + "+".join(x[1] for x in cur),
                                  energy_keV=min(max(sum(x[0] * x[2] for x in cur) / r, cur[0][0]), cur[-1][0]),
                                  span_keV=(cur[0][0], cur[-1][0]),
                                  lines=tuple(x[1] for x in cur), rate=r))
            cur = [ln] if ln is not None else []
    return tuple(out)


def l_production(Z: int, energies_keV, cutoff_keV: float = L_CUTOFF_KEV) -> tuple[tuple, np.ndarray]:
    """(groups, σ) with σ[g, k] = Σ_lines CS_FluorLine_Kissel_Cascade(Z, line, E_k) in cm²/g of element Z
    (zero below the subshell edge)."""
    import xraylib as xrl
    groups = l_shell_groups(Z, cutoff_keV)
    e = np.atleast_1d(np.asarray(energies_keV, dtype=float))
    sig = np.zeros((len(groups), e.size))
    for g, grp in enumerate(groups):
        for n in grp.lines:
            macro = getattr(xrl, f"{n}_LINE")
            for k, ek in enumerate(e):
                try:
                    sig[g, k] += xrl.CS_FluorLine_Kissel_Cascade(Z, macro, float(ek))
                except ValueError:                       # below the subshell edge
                    pass
    return groups, sig


@lru_cache(maxsize=1)
def air_definition() -> dict:
    """Dry air near sea level, NIST SRD 126 Table 2 (archived stage 1 page): composition and density."""
    from shieldmap.ingest import references
    d = references.parse_tab2(references.SRD126_DIR / "tab2.html")["Air, Dry (near sea level)"]
    w = np.zeros(100)
    for z, f in d["composition"]:
        w[z - 1] = f
    return {"W": w / w.sum(), "density_g_cm3": float(d["density_g_cm3"]),
            "source": "NIST SRD 126 Table 2 (data/raw/srd126/tab2.html), Air, Dry (near sea level)"}


def air_mu_rho(e_keV, coherent: bool = True) -> np.ndarray:
    key = "total_wcoh" if coherent else "total_wocoh"
    return air_definition()["W"] @ _xs_at(np.atleast_1d(np.asarray(e_keV, dtype=float)))[key]


def lshell_table(cutoff_keV: float = L_CUTOFF_KEV) -> pd.DataFrame:
    rows = []
    for Z in range(1, 101):
        for g in l_shell_groups(Z, cutoff_keV):
            rows.append({"Z": Z, "group": g.name, "subshell": g.subshell, "energy_keV": g.energy_keV,
                         "span_lo_keV": g.span_keV[0], "span_hi_keV": g.span_keV[1], "rate_sum": g.rate})
    return pd.DataFrame(rows)


# =============================================================================================
# escape kernel
# =============================================================================================
@lru_cache(maxsize=8)
def _nodes(n: int, escape: str) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (x + 1.0)
    c = u * u
    wc = w * u                                   # ½ (from x→u) × 2u (from c = u²)
    if escape == "E1":
        wc = wc / c
    elif escape != "E2":
        raise ValueError(escape)
    return c, wc


def escape_kernel(alpha, beta, escape: str = ESCAPE_PRIMARY, n_nodes: int = N_NODES,
                  air: float = 0.0) -> np.ndarray:
    """G(α, β) = ½ ∫₀¹ h(α, β/c) w(c) e^{−air/c} dc with w = 1/c (E1, fluence, primary) or 1 (E2, planar
    current); air = (μ/ρ)_air ρ_air d of the gap behind the slab (scalar); see module doc."""
    al, be = np.broadcast_arrays(np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float))
    shape = al.shape
    al = al.ravel(); be = be.ravel()
    c, wc = _nodes(n_nodes, escape)
    if air > 0:
        wc = wc * np.exp(-air / c)
    out = np.empty(al.size)
    step = max(1, KERNEL_CHUNK // c.size)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for s in range(0, al.size, step):
            a_ = al[s:s + step, None]
            g = be[s:s + step, None] / c[None, :]
            d = np.abs(g - a_)
            h = -np.expm1(-d)
            h /= d
            h[d == 0] = 1.0
            h *= np.exp(-np.minimum(a_, g))
            out[s:s + step] = h @ wc
    return 0.5 * out.reshape(shape)


# Node count used by the model, by β = b t of the row (escape_kernel_rows): the 32-node rule is kept below
# β = 0.3; above it fewer nodes reach the same accuracy. Against 128 nodes for α ≤ 20 (pieces with α > 20
# carry < e^-20 of the primary): max |ΔG/G| ≤ 1e-7 (24 nodes, β ≥ 0.3), ≤ 3e-8 (16, β ≥ 3), ≤ 1e-9 (12,
# β ≥ 30), for E1 and E2 (tested). On 250 B4 rows at RQR9 the solved areal masses agree with the fixed
# 32-node kernel to 4e-11 relative, 3.3× faster.
NODE_RULE = ((30.0, 12), (3.0, 16), (0.3, 24))


def nodes_for_beta(beta: float) -> int:
    for thr, n in NODE_RULE:
        if beta >= thr:
            return n
    return N_NODES


def escape_kernel_rows(alpha: np.ndarray, beta: np.ndarray, escape: str = ESCAPE_PRIMARY,
                       air: float = 0.0) -> np.ndarray:
    """G for alpha (m, K) and one β per row (m,), with the node count of NODE_RULE per row."""
    beta = np.asarray(beta, dtype=float)
    out = np.empty(alpha.shape)
    cls = np.array([nodes_for_beta(b) for b in beta])
    for n in np.unique(cls):
        sel = cls == n
        out[sel] = escape_kernel(alpha[sel], beta[sel][:, None], escape, n_nodes=int(n), air=air)
    return out


def slab_yield(tau_k: float, a: float, b: float, t, omega_p: float = 1.0, escape: str = ESCAPE_PRIMARY,
               air: float = 0.0) -> np.ndarray:
    """Escaping fluorescence photons per incident photon: ω p τ_K t G(a t, b t) (one energy, one line)."""
    t = np.asarray(t, dtype=float)
    return omega_p * tau_k * t * escape_kernel(a * t, b * t, escape, air=air)


# =============================================================================================
# the model
# =============================================================================================
def _xs_at(e_keV: np.ndarray) -> dict:
    return mixture.element_mu_rho_at(np.asarray(e_keV, dtype=float), keys=("pe", "total_wcoh", "total_wocoh"))


def _dense(W) -> np.ndarray:
    if sps.issparse(W):
        W = W.toarray()
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.shape[1] != 100:
        raise ValueError("mass-fraction rows must have 100 columns (Z = 1..100)")
    return W


@dataclass
class _Emitter:
    Z: int
    rows: np.ndarray        # model rows containing Z
    cols: np.ndarray        # spectrum pieces with E ≥ the shell's edge
    tauw: np.ndarray        # K: (m, K_e) w_Z τ_Z (1 − 1/J); L: (m, 1) w_Z   [cm²/g or 1]
    b: np.ndarray           # (m, L) compound μ/ρ at the group energies
    coef: np.ndarray        # (L,) K: ω p_l R(E_l); L: R(E_l)
    energy: np.ndarray      # (L,) group energies (keV)
    shell: str = "K"
    sig: np.ndarray | None = None    # L only: (L, K_e) Kissel cascade production per g of element
    air: np.ndarray | None = None    # (L,) air-gap optical depth of each line


@lru_cache(maxsize=4096)
def _l_production_cached(Z: int, e_bytes: bytes, cutoff_keV: float):
    return l_production(Z, np.frombuffer(e_bytes, dtype=float), cutoff_keV)


class FluorescenceModel:
    """Effective transmission T_eff(t) of n compounds (mass-fraction rows W) for one source.

    energies, fluence  quadrature pieces (keV, fluence); a line source is a single piece.
    response           'air_kerma' | 'fluence' | 'energy_fluence'
    coherent           μ/ρ with (True) or without coherent scattering, for the primary and the lines
    window             K-edge window (keV) of the constituents allowed to emit K lines
    jump               'xcom' (primary) | 'xraylib'
    escape             'E1' (fluence, primary) | 'E2' (planar current, not an air-kerma quantity)
    l_shell            True (primary) → add L lines ≥ l_cutoff_keV of every constituent
    l_cutoff_keV       lowest L line energy kept (primary 3.2029 keV, the Ar K edge)
    air_cm             air gap between back face and kerma point (primary 0)
    fluorescence       False → plain narrow beam (reproduces Table C)
    """

    def __init__(self, W, energies, fluence, response: str = "air_kerma", coherent: bool = True,
                 window=WINDOW_KEV, jump: str = "xcom", escape: str = ESCAPE_PRIMARY, fluorescence: bool = True,
                 l_shell: bool = True, l_cutoff_keV: float = L_CUTOFF_KEV, air_cm: float = AIR_GAP_CM):
        self.W = _dense(W)
        e = np.atleast_1d(np.asarray(energies, dtype=float))
        phi = np.atleast_1d(np.asarray(fluence, dtype=float))
        R = resp_mod.response(response, e)
        w = phi * R
        keep = w > 0
        self.E, self.phi, self.R, self.w = e[keep], phi[keep], R[keep], w[keep]
        self.S = float(self.w.sum())
        self.response, self.coherent, self.window = response, coherent, tuple(window)
        self.jump, self.escape, self.fluorescence = jump, escape, fluorescence
        self.l_shell, self.l_cutoff_keV, self.air_cm = l_shell, float(l_cutoff_keV), float(air_cm)
        if l_shell and self.l_cutoff_keV < L_CUTOFF_KEV - 1e-9 and response == "air_kerma":
            raise ValueError("the Phase 1 air-kerma response spline starts at the Ar K edge (3.2029 keV); "
                             "lower cutoffs are not supported")
        key = "total_wcoh" if coherent else "total_wocoh"
        xs = _xs_at(self.E)
        self.a = self.W @ xs[key]                     # (n, K) μ/ρ
        self.n = self.W.shape[0]
        self.emitters: list[_Emitter] = []
        self.has_fluor = np.zeros(self.n, dtype=bool)
        if not fluorescence:
            return
        rho_d = air_definition()["density_g_cm3"] * self.air_cm

        def air_of(el):
            return air_mu_rho(el, coherent) * rho_d if rho_d > 0 else np.zeros(len(el))

        present = np.flatnonzero((self.W > 0).any(axis=0)) + 1
        for Z in present:
            rows = np.flatnonzero(self.W[:, Z - 1] > 0)
            k = k_shell_data(int(Z))
            if k is not None and k.omega_K > 0 and self.window[0] <= k.edge_keV <= self.window[1]:
                cols = np.flatnonzero(self.E >= k.edge_keV)
                if cols.size:
                    J = k.jump(jump)
                    tau = xs["pe"][Z - 1, cols] * (1.0 - 1.0 / J)
                    el = np.array(k.group_energy_keV)
                    xl = _xs_at(el)[key]                      # (100, L)
                    self.emitters.append(_Emitter(
                        Z=int(Z), rows=rows, cols=cols, tauw=self.W[rows, Z - 1][:, None] * tau[None, :],
                        b=self.W[rows] @ xl,
                        coef=k.omega_K * np.array(k.group_prob) * resp_mod.response(response, el),
                        energy=el, shell="K", air=air_of(el)))
                    self.has_fluor[rows] = True
            if not l_shell:
                continue
            groups, sig = _l_production_cached(int(Z), np.ascontiguousarray(self.E).tobytes(), self.l_cutoff_keV)
            if not groups:
                continue
            cols = np.flatnonzero((sig > 0).any(axis=0))
            if cols.size == 0:
                continue
            el = np.array([g.energy_keV for g in groups])
            xl = _xs_at(el)[key]
            self.emitters.append(_Emitter(
                Z=int(Z), rows=rows, cols=cols, tauw=self.W[rows, Z - 1][:, None], b=self.W[rows] @ xl,
                coef=resp_mod.response(response, el), energy=el, shell="L", sig=sig[:, cols], air=air_of(el)))
            self.has_fluor[rows] = True

    # ---- transmission ----------------------------------------------------------------------
    def primary(self, t, idx=None) -> np.ndarray:
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        t = np.broadcast_to(np.asarray(t, dtype=float), idx.shape)
        return np.exp(-self.a[idx] * t[:, None]) @ self.w / self.S

    def fluorescence_parts(self, t, idx=None, shells=("K", "L")) -> dict[int, np.ndarray]:
        """Escaping fluorescence response per emitter Z (K + L lines of the shells asked for; normalised
        like T), arrays over idx."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        t = np.broadcast_to(np.asarray(t, dtype=float), idx.shape)
        pos = np.full(self.n, -1); pos[idx] = np.arange(idx.size)
        out = {}
        for em in self.emitters:
            if em.shell not in shells:
                continue
            sel = pos[em.rows] >= 0
            if not sel.any():
                continue
            rr = em.rows[sel]; p = pos[rr]; tt = t[p]
            alpha = self.a[rr][:, em.cols] * tt[:, None]
            A = em.tauw[sel] * self.phi[em.cols][None, :]
            acc = np.zeros(idx.size)
            for l in range(em.coef.size):
                Al = A if em.sig is None else A * em.sig[l][None, :]
                air = float(em.air[l]) if em.air is not None else 0.0
                G = escape_kernel_rows(alpha, em.b[sel, l] * tt, self.escape, air=air)
                acc[p] += em.coef[l] * tt * np.einsum("ij,ij->i", Al, G)
            out[em.Z] = out.get(em.Z, 0.0) + acc / self.S
        return out

    def transmission(self, t, idx=None, parts: bool = False):
        prim = self.primary(t, idx)
        fp = self.fluorescence_parts(t, idx) if self.emitters else {}
        fl = np.sum(list(fp.values()), axis=0) if fp else np.zeros_like(prim)
        if parts:
            return prim + fl, prim, fl, fp
        return prim + fl

    # ---- solvers ---------------------------------------------------------------------------
    def solve_narrow(self, T, idx=None) -> np.ndarray:
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        return transmission.solve_thickness(self.a[idx], self.w, np.broadcast_to(T, idx.shape), xtol_rel())

    def solve_effective(self, T, idx=None, lo=None) -> np.ndarray:
        """Areal mass with T_eff = T (fluorescence-free rows return the narrow-beam value exactly)."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        T = np.broadcast_to(np.asarray(T, dtype=float), idx.shape).copy()
        lo = self.solve_narrow(T, idx) if lo is None else np.asarray(lo, dtype=float).copy()
        out = lo.copy()
        cand = np.flatnonzero(self.has_fluor[idx])
        if cand.size == 0:
            return out
        lt = np.log(T)

        def f(x, j):                                  # j: positions into idx
            return np.log(self.transmission(x, idx[j])) - lt[j]

        flo = f(lo[cand], cand)
        ia = cand[flo > 0]                            # rows whose fluorescence moves the solution
        if ia.size == 0:
            return out
        hi = lo[ia] * 1.25
        for _ in range(60):
            fh = f(hi, ia)
            grow = fh > 0
            if not grow.any():
                break
            hi[grow] *= 1.5
        else:
            raise RuntimeError("could not bracket the effective solution")
        x, _ = transmission.brentq_vec(lambda x, j: f(x, ia[j]), lo[ia], hi, xtol=1e-14 * hi, rtol=xtol_rel())
        out[ia] = x
        return out

    # ---- item 1: K-shell share of the absorbed kerma ---------------------------------------
    def kerma_fraction_K(self, t=None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Per emitter Z: (rows, fraction). t None → fraction of first interactions (thick slab)
        Σ w τ_K/μ / Σ w; t given (n,) → fraction of the removed response at t:
        Σ w (τ_K/μ)(1 − e^{−μt}) / Σ w (1 − e^{−μt})."""
        out = {}
        for em in self.emitters:
            if em.shell != "K":
                continue
            a = self.a[em.rows][:, em.cols]
            r = em.tauw / a
            if t is None:
                out[em.Z] = (em.rows, (r * self.w[em.cols]).sum(1) / self.S)
            else:
                tt = np.asarray(t, dtype=float)[em.rows]
                rem_all = (-np.expm1(-self.a[em.rows] * tt[:, None])) @ self.w
                rem = -np.expm1(-a * tt[:, None]) * self.w[em.cols]
                out[em.Z] = (em.rows, (r * rem).sum(1) / rem_all)
        return out


# =============================================================================================
# convenience API
# =============================================================================================
def _source(spectrum=None, energies=None, fluence=None):
    if spectrum is not None:
        return common.spectrum(spectrum)
    if energies is None:
        raise ValueError("give a spectrum name or energies/fluence")
    e = np.atleast_1d(np.asarray(energies, dtype=float))
    f = np.ones_like(e) if fluence is None else np.atleast_1d(np.asarray(fluence, dtype=float))
    return e, f


def solve_areal_mass_effective(W, T_target, spectrum: str | None = None, energies=None, fluence=None,
                               response: str = "air_kerma", **model_kw) -> dict:
    """Narrow-beam and fluorescence-corrected required areal mass (g/cm²) for mass-fraction rows W.

    Returns q_narrow, q_eff, T_prim / T_fl at q_eff and the fluorescence share of T_eff at q_eff."""
    e, f = _source(spectrum, energies, fluence)
    m = FluorescenceModel(W, e, f, response=response, **model_kw)
    qn = m.solve_narrow(T_target)
    qe = m.solve_effective(T_target, lo=qn)
    Te, Tp, Tf, _ = m.transmission(qe, parts=True)
    return {"q_narrow": qn, "q_eff": qe, "T_eff": Te, "T_prim": Tp, "T_fl": Tf, "fl_fraction": Tf / Te}


def mono_effective(W, e_keV: float, T, response: str = "air_kerma", **model_kw) -> dict:
    """Monoenergetic line source at e_keV: q_narrow = ln(1/T)/(μ/ρ), q_eff, and the effective attenuation
    coefficient (μ/ρ)_eff = ln(1/T)/q_eff (cm²/g) that can be compared with an elemental envelope."""
    r = solve_areal_mass_effective(W, T, energies=[e_keV], fluence=[1.0], response=response, **model_kw)
    r["mu_rho_narrow"] = math.log(1.0 / T) / r["q_narrow"]
    r["mu_rho_eff"] = math.log(1.0 / T) / r["q_eff"]
    return r


def pb_row() -> np.ndarray:
    w = np.zeros((1, 100)); w[0, PB_Z - 1] = 1.0
    return w


# =============================================================================================
# workers (one spectrum per task; top-level for Windows spawn)
# =============================================================================================
def _job(task: dict) -> dict:
    kind = task["kind"]
    sp = task["spectrum"]
    resp = task.get("response", "air_kerma")
    opts = task.get("opts", {})
    e, phi = common.spectrum(sp)
    out = {"task": {k: v for k, v in task.items() if k not in ("W",)}, "t0": time.time()}
    # Pb reference (same options): like-for-like targets
    pbm = FluorescenceModel(pb_row(), e, phi, response=resp, **opts)
    tstar = np.array([v["T_target"] for v in task["variants"]])
    t_pb = np.array([v["pb_areal"] for v in task["variants"]])
    T_lfl = np.array([pbm.transmission(np.array([t]))[0] for t in t_pb])
    out["T_lfl"] = T_lfl
    out["pb_q_eff_fixed"] = np.array([pbm.solve_effective(T)[0] for T in tstar])
    out["pb_q_narrow"] = np.array([pbm.solve_narrow(T)[0] for T in tstar])
    W = task["W"]
    if kind == "narrow_lfl":
        m = FluorescenceModel(W, e, phi, response=resp, fluorescence=False)
        out["q_narrow_lfl"] = np.stack([m.solve_narrow(T) for T in T_lfl], axis=1)
        return out
    m = FluorescenceModel(W, e, phi, response=resp, **opts)
    qn = np.stack([m.solve_narrow(T) for T in tstar], axis=1)                  # (n, 3)
    qe = np.stack([m.solve_effective(T, lo=qn[:, j]) for j, T in enumerate(tstar)], axis=1)
    qn_l = np.stack([m.solve_narrow(T) for T in T_lfl], axis=1)
    ql = np.stack([m.solve_effective(T, lo=qn_l[:, j]) for j, T in enumerate(T_lfl)], axis=1)
    out.update(q_narrow=qn, q_eff_fixed=qe, q_eff_lfl=ql, q_narrow_lfl=qn_l)
    if kind == "main":
        flfrac = np.zeros_like(qe); shares = []
        kf = []
        thick = m.kerma_fraction_K(None)
        for j in range(len(tstar)):
            Te, Tp, Tf, fp = m.transmission(qe[:, j], parts=True)
            flfrac[:, j] = Tf / Te
            for Z, arr in fp.items():
                sel = m.W[:, Z - 1] > 0
                shares.append(pd.DataFrame({"pos": np.flatnonzero(sel), "j": j, "Z": Z,
                                            "share_of_T_eff": arr[sel] / Te[sel]}))
            atq = m.kerma_fraction_K(qn[:, j])
            for Z, (rows, fr) in atq.items():
                kf.append(pd.DataFrame({"pos": rows, "j": j, "Z": Z, "frac_K_first_interaction_thick": thick[Z][1],
                                        "frac_K_removed_at_q_narrow": fr}))
        out["fl_fraction"] = flfrac
        out["shares"] = pd.concat(shares, ignore_index=True) if shares else pd.DataFrame()
        out["kerma_K"] = pd.concat(kf, ignore_index=True) if kf else pd.DataFrame()
        out["has_fluor"] = m.has_fluor
    out["seconds"] = time.time() - out.pop("t0")
    return out


CACHE = OUT / "_cache"          # per-task checkpoints (resume after an interruption); removed by run()


def _task_key(task: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    W = sps.csr_matrix(task["W"])
    for arr in (W.data, W.indices, W.indptr):
        h.update(np.ascontiguousarray(arr).tobytes())
    meta = {k: v for k, v in task.items() if k != "W"}
    h.update(json.dumps(meta, sort_keys=True, default=str).encode())
    h.update(MODEL_VERSION.encode())
    return h.hexdigest()[:24]


def _run_tasks(tasks: list[dict], n_workers: int | None) -> list[dict]:
    """Run tasks in a process pool; each result is checkpointed in CACHE and reused on a rerun."""
    import pickle
    from concurrent.futures import as_completed
    paths.ensure(CACHE)
    keys = [_task_key({k: v for k, v in t.items() if k != "_group"}) for t in tasks]
    out: list = [None] * len(tasks)
    todo = []
    for i, k in enumerate(keys):
        f = CACHE / f"{k}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                out[i] = pickle.load(fh)
        else:
            todo.append(i)

    done = [0]

    def _save(i, r):
        tmp = CACHE / f"{keys[i]}.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(r, fh)
        os.replace(tmp, CACHE / f"{keys[i]}.pkl")
        out[i] = r
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(todo):
            print(f"    {done[0]}/{len(todo)} tasks done ({tasks[i]['tag']})", flush=True)

    n_workers = n_workers or min(len(todo), max(1, (os.cpu_count() or 3) - 2))
    if todo and n_workers <= 1:
        for i in todo:
            _save(i, _job(tasks[i]))
    elif todo:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_job, tasks[i]): i for i in todo}
            for fu in as_completed(futs):
                _save(futs[fu], fu.result())
    return _merge(tasks, out) if "_group" in tasks[0] else out


# =============================================================================================
# the fluorescence-module analysis
# =============================================================================================
def _diag_variants(response: str = "air_kerma") -> pd.DataFrame:
    v = common.variants()
    v = v[(v.scenario == "DIAG") & (v.response == response) & v.coherent].sort_values("variant_id").copy()
    v["pb_areal"] = v["pb_thickness_cm"] * v["pb_density_g_cm3"]
    v["pb_mm"] = v["target_value"]
    return v.reset_index(drop=True)


TASK_ROWS = 250                  # rows per task: short tasks, frequent checkpoints


def _tasks_for(kind, rows_W, vdf, response="air_kerma", opts=None, tag=None):
    """One task per (spectrum, variant, row chunk); _merge reassembles per-spectrum results."""
    rows_W = sps.csr_matrix(rows_W)
    n = rows_W.shape[0]
    step = n if kind == "narrow_lfl" else TASK_ROWS
    tasks = []
    for sp, g in vdf.groupby("spectrum", sort=False):
        g = g.sort_values("pb_mm")
        recs = g[["variant_id", "T_target", "pb_areal", "pb_mm"]].to_dict("records")
        for jv, rec in enumerate(recs):
            for s0 in range(0, n, step):
                tasks.append({"kind": kind, "spectrum": sp, "response": response, "opts": opts or {},
                              "tag": tag or kind, "W": rows_W[s0:s0 + step], "variants": [rec],
                              "_group": (sp, len(recs), jv, s0)})
    return tasks


def _merge(tasks: list[dict], results: list[dict]) -> list[dict]:
    """Reassemble chunk × single-variant results into one result per spectrum (rows × variants)."""
    groups: dict = {}
    for t, r in zip(tasks, results):
        sp, nv, jv, s0 = t["_group"]
        groups.setdefault((t["tag"], t["response"], sp), {}).setdefault(jv, []).append((s0, t, r))
    merged = []
    for (_, _, sp), byv in groups.items():
        nv = len(byv)
        out = {"task": {**{k: v for k, v in byv[0][0][1].items() if k not in ("W", "variants", "_group")},
                        "variants": [byv[j][0][1]["variants"][0] for j in range(nv)]}}
        for j in range(nv):
            byv[j].sort(key=lambda x: x[0])
        first = byv[0][0][2]
        for key in ("T_lfl", "pb_q_eff_fixed", "pb_q_narrow"):
            out[key] = np.array([byv[j][0][2][key][0] for j in range(nv)])
        for key in ("q_narrow", "q_eff_fixed", "q_eff_lfl", "q_narrow_lfl", "fl_fraction"):
            if key in first:
                out[key] = np.stack([np.concatenate([r[key][:, 0] for _, _, r in byv[j]]) for j in range(nv)], axis=1)
        if "has_fluor" in first:
            out["has_fluor"] = np.concatenate([r["has_fluor"] for _, _, r in byv[0]])
        for key in ("shares", "kerma_K"):
            if key in first:
                parts = []
                for j in range(nv):
                    for s0, _, r in byv[j]:
                        d = r[key]
                        if len(d):
                            d = d.copy(); d["pos"] += s0; d["j"] = j
                            parts.append(d)
                out[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        out["seconds"] = float(sum(r.get("seconds", 0.0) for j in range(nv) for _, _, r in byv[j]))
        merged.append(out)
    return merged


def _collect(results, key, n, vids_order) -> np.ndarray:
    """Assemble (n × V) from per-spectrum (n × 3) arrays, columns in vids_order."""
    col = {v: j for j, v in enumerate(vids_order)}
    out = np.full((n, len(vids_order)), np.nan)
    for r in results:
        for j, v in enumerate(r["task"]["variants"]):
            out[:, col[v["variant_id"]]] = r[key][:, j]
    return out


def _collect_vec(results, key, vids_order) -> np.ndarray:
    col = {v: j for j, v in enumerate(vids_order)}
    out = np.full(len(vids_order), np.nan)
    for r in results:
        for j, v in enumerate(r["task"]["variants"]):
            out[col[v["variant_id"]]] = r[key][j]
    return out


def _textbook_rows() -> dict[str, int]:
    u = common.universe(); base = common.base_mask("R2")
    out = {}
    for f in TEXTBOOK:
        r = np.flatnonzero((u["formula_reduced"] == f).to_numpy() & base)
        if r.size != 1:
            raise RuntimeError(f"{f}: {r.size} R2 representatives")
        out[f] = int(r[0])
    return out


def _lanthanide_mask() -> np.ndarray:
    return np.array([bool(set(e) & common.LANTHANIDES) for e in common.universe()["elements"]])


def _textbook_L_fraction(vdf, tb, QF, tb_pos, WU) -> dict:
    """Share of T_eff carried by L lines (all emitters) at the corrected requirement, textbook materials."""
    forms = list(tb)
    W = WU[[tb[f] for f in forms]].toarray()
    out = {f: [] for f in forms}
    for j, r in enumerate(vdf.itertuples()):
        e, phi = common.spectrum(r.spectrum)
        m = FluorescenceModel(W, e, phi)
        q = np.array([QF[tb_pos[f], j] for f in forms])
        Te = m.transmission(q)
        fl = m.fluorescence_parts(q, shells=("L",))
        lsum = np.sum(list(fl.values()), axis=0) if fl else np.zeros(len(forms))
        for i, f in enumerate(forms):
            out[f].append(float(lsum[i] / Te[i]))
    return {f: _summ(v) for f, v in out.items()}


def _summ(a) -> dict:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def run(n_workers: int | None = None, verbose: bool = True) -> dict:
    t_start = time.time()
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    paths.ensure(OUT)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    import xraylib
    u = common.universe()
    P = common.profiles("areal")
    base = common.base_mask("R2")
    b4 = base & common.bundle_mask("B4")
    lan = _lanthanide_mask()
    vdf = _diag_variants("air_kerma")
    vids = vdf["variant_id"].to_numpy()
    V = vids.size
    Pn = np.asarray(P[:, vids], dtype=np.float64)                       # (N × 36) Table C areal

    # ---- K-shell data ----
    ks = kshell_table()
    ks.to_csv(OUT / "kshell_data.csv", index=False)
    ls_ = lshell_table()
    ls_.to_csv(OUT / "lshell_groups.csv", index=False)
    log(f"[F] kshell table: {len(ks)} elements; L groups: {len(ls_)} (≥ {L_CUTOFF_KEV} keV)")

    # ---- row sets ----
    top = {int(v): common.top_k(Pn[:, j], b4, TOP_K) for j, v in enumerate(vids)}
    union = np.unique(np.concatenate(list(top.values())))
    tb = _textbook_rows()
    rng = np.random.default_rng(SEED)
    rnd = np.sort(rng.choice(np.flatnonzero(base), N_RANDOM, replace=False))
    S = np.unique(np.concatenate([union, np.array(list(tb.values())), rnd]))
    WU = common.mass_fractions()
    W_el = np.eye(100)
    log(f"[F] rows: top-{TOP_K} union {union.size}, textbook {len(tb)}, random {rnd.size}, total {S.size}")

    # ---- round 1: main (S + 100 pure elements), sensitivities on union+textbook+elements, B4 lfl bounds
    W_main = sps.vstack([WU[S], sps.csr_matrix(W_el)]).tocsr()
    U = np.unique(np.concatenate([union, np.array(list(tb.values()))]))
    W_u = sps.vstack([WU[U], sps.csr_matrix(W_el)]).tocsr()
    sens_opts = SENS_OPTS
    tasks = _tasks_for("main", W_main, vdf)
    for name, o in sens_opts.items():
        tasks += _tasks_for("sens", W_u, vdf, opts=o, tag=name)
    b4_rows = np.flatnonzero(b4)
    tasks += _tasks_for("narrow_lfl", WU[b4_rows], vdf)
    t0 = time.time()
    res = _run_tasks(tasks, n_workers)
    log(f"[F] round 1: {len(tasks)} tasks in {time.time() - t0:.0f} s")
    by = {}
    for r in res:
        by.setdefault(r["task"]["tag"], []).append(r)

    nS = S.size
    main = by["main"]
    q_n = _collect(main, "q_narrow", nS + 100, vids)
    q_f = _collect(main, "q_eff_fixed", nS + 100, vids)
    q_l = _collect(main, "q_eff_lfl", nS + 100, vids)
    q_nl = _collect(main, "q_narrow_lfl", nS + 100, vids)
    flf = _collect(main, "fl_fraction", nS + 100, vids)
    T_lfl = _collect_vec(main, "T_lfl", vids)
    pb_qf = _collect_vec(main, "pb_q_eff_fixed", vids)
    pb_qn = _collect_vec(main, "pb_q_narrow", vids)

    # validation (d): narrow solver vs Table C areal profiles
    rel_d = np.abs(q_n[:nS] / Pn[S] - 1.0)
    val_d = float(np.nanmax(rel_d))
    log(f"[F] max |q_narrow / Table C − 1| over {nS}×{V}: {val_d:.2e}")

    # ---- certificates and expansion (fixed-T and like-for-like) ----
    inS = np.zeros(len(u), dtype=bool); inS[S] = True
    narrow_lfl_b4 = np.full((len(u), V), np.nan)
    narrow_lfl_b4[b4_rows] = _collect(by["narrow_lfl"], "q_narrow_lfl", b4_rows.size, vids)
    extra_rows = np.zeros(0, dtype=int)
    q_ext = {}
    cert = {}
    for it in range(4):
        rows_all = np.concatenate([S, extra_rows])
        qf_all = np.concatenate([q_f[:nS], q_ext.get("q_eff_fixed", np.zeros((0, V)))])
        ql_all = np.concatenate([q_l[:nS], q_ext.get("q_eff_lfl", np.zeros((0, V)))])
        computed = np.zeros(len(u), dtype=bool); computed[rows_all] = True
        need = np.zeros(len(u), dtype=bool)
        for j in range(V):
            mb4 = b4[rows_all]; ml = (b4 & lan)[rows_all]
            fr_f = np.nanmin(qf_all[mb4, j]); fr_l = np.nanmin(qf_all[ml, j])
            fr_fl = np.nanmin(ql_all[mb4, j]); fr_ll = np.nanmin(ql_all[ml, j])
            out_ = b4 & ~computed
            need |= out_ & (Pn[:, j] < max(fr_f, fr_l))
            need |= out_ & (narrow_lfl_b4[:, j] < max(fr_fl, fr_ll))
        cert[it] = int(need.sum())
        log(f"[F] certificate pass {it}: {int(need.sum())} B4 rows outside the computed set could still lead")
        if not need.any():
            break
        new = np.flatnonzero(need)
        rr = _run_tasks(_tasks_for("main", WU[new], vdf), n_workers)
        for key in ("q_narrow", "q_eff_fixed", "q_eff_lfl", "fl_fraction", "q_narrow_lfl"):
            q_ext[key] = np.concatenate([q_ext.get(key, np.zeros((0, V))), _collect(rr, key, new.size, vids)])
        extra_rows = np.concatenate([extra_rows, new])
        by.setdefault("main_extra", []).extend(rr)
    else:
        raise RuntimeError("certificate expansion did not terminate")

    rows_all = np.concatenate([S, extra_rows])
    QN = np.concatenate([q_n[:nS], q_ext.get("q_narrow", np.zeros((0, V)))])
    QF = np.concatenate([q_f[:nS], q_ext.get("q_eff_fixed", np.zeros((0, V)))])
    QL = np.concatenate([q_l[:nS], q_ext.get("q_eff_lfl", np.zeros((0, V)))])
    FLF = np.concatenate([flf[:nS], q_ext.get("fl_fraction", np.zeros((0, V)))])
    pos_of = {int(r): i for i, r in enumerate(rows_all)}
    el_q_n, el_q_f, el_q_l, el_flf = q_n[nS:], q_f[nS:], q_l[nS:], flf[nS:]

    # ---- corrected_areal.parquet ----
    var_lab = vdf.set_index("variant_id")
    inU = np.isin(rows_all, union); inTB = np.isin(rows_all, list(tb.values())); inR = np.isin(rows_all, rnd)
    rec = []
    for j, v in enumerate(vids):
        rec.append(pd.DataFrame({
            "row": rows_all, "material_id": u["material_id"].to_numpy()[rows_all],
            "formula": u["formula_reduced"].to_numpy()[rows_all], "variant_id": int(v),
            "spectrum": var_lab.loc[v, "spectrum"], "pb_mm": var_lab.loc[v, "pb_mm"],
            "in_b4": b4[rows_all], "lanthanide": lan[rows_all],
            "set_top200_union": inU, "set_textbook": inTB, "set_random2000": inR,
            "set_certificate_extra": ~(inU | inTB | inR),
            "rank_b4_narrow": np.where(b4[rows_all], common.core.ranks(Pn[:, j], b4)[rows_all], np.nan),
            "q_tablec": Pn[rows_all, j], "q_narrow": QN[:, j], "q_eff_fixedT": QF[:, j],
            "q_eff_likeforlike": QL[:, j], "ratio_fixedT": QF[:, j] / QN[:, j],
            "ratio_likeforlike": QL[:, j] / QN[:, j], "fl_kerma_fraction_at_q_eff": FLF[:, j]}))
    cor = pd.concat(rec, ignore_index=True)
    cor.to_parquet(OUT / "corrected_areal.parquet", index=False)

    # ---- decision metric ----
    tb_pos = {f: pos_of[r] for f, r in tb.items()}
    b4p = b4[rows_all]; lanp = (b4 & lan)[rows_all]
    syms = [k_shell_data(z).symbol if k_shell_data(z) else common.elements_table().set_index("Z").loc[z, "symbol"]
            for z in range(1, 101)]

    def metrics(Q):
        best_lan = np.array([np.nanmin(Q[lanp, j]) for j in range(V)])
        arg_lan = np.array([rows_all[lanp][np.nanargmin(Q[lanp, j])] for j in range(V)])
        fr = np.array([np.nanmin(Q[b4p, j]) for j in range(V)])
        arg_fr = np.array([rows_all[b4p][np.nanargmin(Q[b4p, j])] for j in range(V)])
        tbq = {f: Q[p] for f, p in tb_pos.items()}
        best_tb = np.min(np.stack([tbq[f] for f in TEXTBOOK_NO_PB]), axis=0)
        arg_tb = np.array(TEXTBOOK_NO_PB)[np.argmin(np.stack([tbq[f] for f in TEXTBOOK_NO_PB]), axis=0)]
        return {"best_lan": best_lan, "arg_lan": arg_lan, "frontier": fr, "arg_frontier": arg_fr,
                "tb": tbq, "best_tb": best_tb, "arg_tb": arg_tb,
                "adv_tb": best_tb / best_lan - 1.0, "adv_pb": tbq["Pb"] / best_lan - 1.0,
                "adv_pbo": tbq["PbO"] / best_lan - 1.0,
                "frontier_is_lan": lan[arg_fr]}

    M = {"narrow": metrics(QN), "fixedT": metrics(QF), "likeforlike": metrics(QL)}
    # Pb itself as a row (pure element) for the vs-Pb metric: identical μ/ρ to the R2 Pb row
    n_stand = {k: int((m["adv_tb"] >= F3_THRESHOLD).sum()) for k, m in M.items()}
    f3 = "F3 stands with caveat" if n_stand["fixedT"] > V / 2 else "F3 reframed"

    # ---- sensitivities (union + textbook; no certificate expansion) ----
    nU = U.size
    posU = {int(r): i for i, r in enumerate(U)}
    lanU = (b4 & lan)[U]; b4U = b4[U]
    sens = {}
    for name in SENS_OPTS:
        rr = by[name]
        out_s = {}
        for key, tgt in (("q_eff_fixed", "fixedT"), ("q_eff_lfl", "likeforlike")):
            Q = _collect(rr, key, nU + 100, vids)[:nU]
            best_lan = np.nanmin(np.where(lanU[:, None], Q, np.nan), axis=0)
            best_tb = np.min(np.stack([Q[posU[tb[f]]] for f in TEXTBOOK_NO_PB]), axis=0)
            adv = best_tb / best_lan - 1.0
            # certificate: no outside B4 lanthanide row can beat best_lan (narrow is a lower bound)
            outside = b4 & lan & ~np.isin(np.arange(len(u)), U)
            lb = Pn if tgt == "fixedT" else narrow_lfl_b4
            # like-for-like targets differ per option; the narrow bound at the primary T*_eff is only
            # valid when this option's T*_eff ≤ the primary one, so the lfl certificate is reported for E2 only
            cert_ok = [bool(np.all(lb[outside, j] >= best_lan[j])) for j in range(V)] if tgt == "fixedT" else None
            ratio_pb = _collect_vec(rr, "pb_q_eff_fixed", vids) / _collect_vec(rr, "pb_q_narrow", vids)
            out_s[tgt] = {"adv_tb": adv, "n_ge_10pct": int((adv >= F3_THRESHOLD).sum()),
                          "certificate_ok_variants": (int(sum(cert_ok)) if cert_ok is not None else None),
                          "ratio_textbook": {f: Q[posU[tb[f]]] / Pn[tb[f]] for f in TEXTBOOK},
                          "best_lan": best_lan}
        out_s["pb_self_ratio"] = ratio_pb
        sens[name] = out_s
        log(f"[F] sensitivity {name}: adv ≥ 10 % in {out_s['fixedT']['n_ge_10pct']}/{V} (fixed-T)")

    # ---- response sensitivity (fluence, energy fluence) on textbook + leaders ----
    lead = set(tb.values())
    for j in range(V):
        for key in ("fixedT",):
            Q = QF
            ordl = np.argsort(np.where(lanp, Q[:, j], np.inf))[:10]
            ordf = np.argsort(np.where(b4p, Q[:, j], np.inf))[:10]
            lead |= set(rows_all[ordl].tolist()) | set(rows_all[ordf].tolist())
    lead = np.array(sorted(lead))
    resp_out = {}
    for rname in ("fluence", "energy_fluence"):
        vr = _diag_variants(rname)
        rr = _run_tasks(_tasks_for("sens", WU[lead], vr, response=rname, tag=rname), n_workers)
        vids_r = vr["variant_id"].to_numpy()
        Qn = _collect(rr, "q_narrow", lead.size, vids_r); Qf = _collect(rr, "q_eff_fixed", lead.size, vids_r)
        lanL = (b4 & lan)[lead]
        posL = {int(r): i for i, r in enumerate(lead)}
        o = {}
        for lab, Q in (("narrow", Qn), ("fixedT", Qf)):
            bl = np.nanmin(np.where(lanL[:, None], Q, np.nan), axis=0)
            bt = np.min(np.stack([Q[posL[tb[f]]] for f in TEXTBOOK_NO_PB]), axis=0)
            o[lab] = {"adv_tb": bt / bl - 1.0, "n_ge_10pct": int(((bt / bl - 1.0) >= F3_THRESHOLD).sum())}
        o["pb_self_ratio"] = _collect_vec(rr, "pb_q_eff_fixed", vids_r) / _collect_vec(rr, "pb_q_narrow", vids_r)
        o["n_rows"] = int(lead.size)
        resp_out[rname] = o
        log(f"[F] response {rname}: adv ≥ 10 % narrow {o['narrow']['n_ge_10pct']}/{V}, "
            f"corrected {o['fixedT']['n_ge_10pct']}/{V}")

    # ---- kerma_fraction_K.parquet and per-emitter shares ----
    kfr, shr = [], []
    for r in main:
        for j, v in enumerate(r["task"]["variants"]):
            for df_key, acc in (("kerma_K", kfr), ("shares", shr)):
                d = r[df_key]
                if d.empty:
                    continue
                d = d[d.j == j]
                acc.append(pd.DataFrame({"pos": d["pos"].to_numpy(), "variant_id": int(v["variant_id"]),
                                         "spectrum": r["task"]["spectrum"], "pb_mm": v["pb_mm"],
                                         "Z": d["Z"].to_numpy(),
                                         **{c: d[c].to_numpy() for c in d.columns if c not in ("pos", "j", "Z")}}))
    kfr = pd.concat(kfr, ignore_index=True); shr = pd.concat(shr, ignore_index=True)
    for d in (kfr, shr):
        p = d.pop("pos").to_numpy()
        is_el = p >= nS
        d.insert(0, "row", np.where(is_el, -1, S[np.minimum(p, nS - 1)]))
        d.insert(1, "material_id", np.where(is_el, "element:" + pd.Series(p - nS + 1).astype(str).to_numpy(),
                                            u["material_id"].to_numpy()[S[np.minimum(p, nS - 1)]]))
        d.insert(2, "formula", np.where(is_el, np.array(syms, dtype=object)[np.clip(p - nS, 0, 99)],
                                        u["formula_reduced"].to_numpy()[S[np.minimum(p, nS - 1)]]))
        d.insert(3, "symbol", np.array(syms, dtype=object)[d["Z"].to_numpy() - 1])
    kfr.to_parquet(OUT / "kerma_fraction_K.parquet", index=False)

    # ---- offenders ----
    el_ratio = el_q_f / el_q_n
    off = []
    shr_c = shr[shr.row >= 0]
    shr_u = shr_c[shr_c.row.isin(union)]
    for z in sorted(set(shr["Z"])):
        k = k_shell_data(z)
        s_all = shr_c.loc[shr_c.Z == z, "share_of_T_eff"]
        s_u = shr_u.loc[shr_u.Z == z, "share_of_T_eff"]
        off.append({"Z": z, "symbol": k.symbol, "E_K_keV": k.edge_keV, "omega_K": k.omega_K,
                    "E_KA1_keV": dict(zip(k.groups, k.group_energy_keV)).get("KA1", np.nan),
                    "pure_ratio_min": np.nanmin(el_ratio[z - 1]), "pure_ratio_median": np.nanmedian(el_ratio[z - 1]),
                    "pure_ratio_max": np.nanmax(el_ratio[z - 1]),
                    "pure_fl_fraction_median": np.nanmedian(el_flf[z - 1]),
                    "n_compounds_evaluated": int(shr_c.loc[shr_c.Z == z, "row"].nunique()),
                    "share_median_all": float(s_all.median()) if len(s_all) else np.nan,
                    "share_max_all": float(s_all.max()) if len(s_all) else np.nan,
                    "n_compounds_top200": int(shr_u.loc[shr_u.Z == z, "row"].nunique()),
                    "share_median_top200": float(s_u.median()) if len(s_u) else np.nan})
    off = pd.DataFrame(off).sort_values("pure_ratio_median", ascending=False)
    off.to_csv(OUT / "offenders.csv", index=False)

    # ---- monoenergetic: elemental envelope 47–67 keV ----
    eg = common.energy_keV()
    em = eg[(eg >= 47.0) & (eg <= 67.0)]
    mono = []
    for T in (0.1, 0.01, 0.001):
        for e0 in em:
            r = solve_areal_mass_effective(W_el, T, energies=[e0], fluence=[1.0])
            mono.append(pd.DataFrame({"Z": np.arange(1, 101), "symbol": syms, "energy_keV": e0, "T": T,
                                      "q_narrow": r["q_narrow"], "q_eff": r["q_eff"],
                                      "mu_rho_narrow": math.log(1 / T) / r["q_narrow"],
                                      "mu_rho_eff": math.log(1 / T) / r["q_eff"], "fl_fraction": r["fl_fraction"]}))
    mono = pd.concat(mono, ignore_index=True)
    mono.to_parquet(OUT / "mono_elements_47_67.parquet", index=False)
    env = mono.groupby(["T", "energy_keV"]).agg(env_narrow=("mu_rho_narrow", "max"), env_eff=("mu_rho_eff", "max"))
    env_arg = mono.loc[mono.groupby(["T", "energy_keV"])["mu_rho_eff"].idxmax(), ["T", "energy_keV", "symbol"]]

    # ---- random-sample distribution ----
    rp = np.array([pos_of[r] for r in rnd])
    rr_ratio = QF[rp] / QN[rp]
    dvj = int(np.flatnonzero((vdf.spectrum == DISPLAY[0]).to_numpy() & np.isclose(vdf.pb_mm, DISPLAY[1]))[0])

    # ---- monotonicity diagnostic on Pb and textbook (T_eff decreasing on a grid) ----
    mono_ok = True
    for sp in vdf.spectrum.unique():
        e, phi = common.spectrum(sp)
        Wt = np.vstack([pb_row(), WU[list(tb.values())].toarray()])
        mm = FluorescenceModel(Wt, e, phi)
        grid = np.geomspace(0.01, 3.0, 60)
        Tg = np.stack([mm.transmission(np.full(Wt.shape[0], g)) for g in grid], axis=1)
        mono_ok &= bool(np.all(np.diff(Tg, axis=1) < 0))

    # ---- line-group edge straddles (any constituent edge inside a merged group) ----
    ed = common.edges()
    straddle = []
    for z in range(1, 101):
        k = k_shell_data(z)
        spans = []
        if k is not None and WINDOW_KEV[0] <= k.edge_keV <= WINDOW_KEV[1]:
            spans += list(zip(k.groups, k.group_span_keV))
        spans += [(g.name, g.span_keV) for g in l_shell_groups(z)]
        for gname, (lo_, hi_) in spans:
            hit = ed[(ed.energy_keV > lo_) & (ed.energy_keV < hi_)]
            for h in hit.itertuples():
                straddle.append({"emitter_Z": z, "group": gname, "edge_Z": int(h.Z), "shell": h.shell,
                                 "edge_keV": h.energy_keV})
    straddle = pd.DataFrame(straddle, columns=["emitter_Z", "group", "edge_Z", "shell", "edge_keV"])
    if not straddle.empty:
        Wd = WU[rows_all].toarray() > 0
        straddle["n_compounds_affected"] = [int((Wd[:, s.emitter_Z - 1] & Wd[:, s.edge_Z - 1]).sum())
                                            for s in straddle.itertuples()]
    straddle.to_csv(OUT / "line_group_edge_straddles.csv", index=False)

    # ---- summary ----
    def per_variant(a):
        return {int(v): float(x) for v, x in zip(vids, a)}

    name_of = lambda r: f"{u.loc[r, 'formula_reduced']} ({u.loc[r, 'material_id']})"
    tb_ratio_frontier = {}
    for lab in ("narrow", "fixedT", "likeforlike"):
        m = M[lab]
        tb_ratio_frontier[lab] = {f: _summ(m["tb"][f] / m["frontier"]) for f in TEXTBOOK}
    tb_rank = {}
    for lab in ("narrow", "fixedT"):
        stack = np.stack([M[lab]["tb"][f] for f in TEXTBOOK])
        rk = stack.argsort(0).argsort(0) + 1
        tb_rank[lab] = {f: _summ(rk[i]) for i, f in enumerate(TEXTBOOK)}
    summary = {
        "agent": "F (Step 10 revision F2)",
        "step": "handover Step 2: K- and L-fluorescence escape (first-order, broad-beam slab, not Monte Carlo)",
        "model": {"escape": ESCAPE_PRIMARY, "l_shell": True, "l_cutoff_keV": L_CUTOFF_KEV,
                  "l_group_tol": L_GROUP_TOL, "air_gap_cm": AIR_GAP_CM, "jump": "xcom",
                  "k_production": "XCOM pe x (1 - 1/J_XCOM) x omega_K x p_l (xraylib omega_K, RadRate)",
                  "l_production": "xraylib CS_FluorLine_Kissel_Cascade (L photoionisation, Coster-Kronig, K->L transfer)",
                  "air": air_definition()["source"], "air_density_g_cm3": air_definition()["density_g_cm3"]},
        "sensitivity_options": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in o.items()}
                                for k, o in SENS_OPTS.items()},
        "xraylib_version": xraylib.__version__, "seed": SEED, "top_k": TOP_K, "n_random": N_RANDOM,
        "window_keV": WINDOW_KEV, "n_nodes": N_NODES,
        "rows": {"top200_union": int(union.size), "textbook": tb, "random": int(rnd.size), "total_S": int(nS),
                 "certificate_extra": int(extra_rows.size), "certificate_passes": cert},
        "validation": {"d_max_rel_vs_tablec": val_d, "T_eff_monotone_textbook_grid": mono_ok,
                       "line_group_edge_straddles": int(len(straddle)),
                       "line_group_edge_straddles_affecting_rows": int(straddle["n_compounds_affected"].sum()) if not straddle.empty else 0},
        "pb": {"T_star": per_variant(vdf.T_target.to_numpy()), "T_star_likeforlike": per_variant(T_lfl),
               "T_lfl_over_T": _summ(T_lfl / vdf.T_target.to_numpy()),
               "q_eff_over_nominal_fixedT": _summ(pb_qf / vdf.pb_areal.to_numpy()),
               "q_eff_over_nominal_by_quality": {q: _summ((pb_qf / vdf.pb_areal.to_numpy())[vdf.spectrum.str.startswith(q).to_numpy()])
                                                for q in ("RQR5", "RQR6", "RQR8", "RQR9")},
               "pb_q_narrow_over_nominal": _summ(pb_qn / vdf.pb_areal.to_numpy()),
               "fl_fraction_at_q_eff": _summ(el_flf[PB_Z - 1])},
        "decision": {"rule": "adv = best corrected textbook (excl. Pb, PbO) / best corrected B4 lanthanide − 1; "
                             "F3 stands if adv ≥ 0.10 in > 18 of 36 variants (fixed-T primary)",
                     "outcome": f3, "n_variants_ge_10pct": n_stand,
                     "adv_tb": {k: _summ(m["adv_tb"]) for k, m in M.items()},
                     "adv_pb": {k: _summ(m["adv_pb"]) for k, m in M.items()},
                     "adv_pbo": {k: _summ(m["adv_pbo"]) for k, m in M.items()},
                     "n_adv_pb_ge_10pct": {k: int((m["adv_pb"] >= F3_THRESHOLD).sum()) for k, m in M.items()},
                     "n_adv_pbo_ge_10pct": {k: int((m["adv_pbo"] >= F3_THRESHOLD).sum()) for k, m in M.items()},
                     "best_textbook_material": {k: pd.Series(m["arg_tb"]).value_counts().to_dict() for k, m in M.items()},
                     "frontier_is_lanthanide": {k: int(m["frontier_is_lan"].sum()) for k, m in M.items()},
                     "frontier_shift_fixedT": _summ(M["fixedT"]["frontier"] / M["narrow"]["frontier"]),
                     "best_lan_shift_fixedT": _summ(M["fixedT"]["best_lan"] / M["narrow"]["best_lan"]),
                     "frontier_shift_likeforlike": _summ(M["likeforlike"]["frontier"] / M["narrow"]["frontier"]),
                     "advantage_shrink_ratio_fixedT": _summ((1 + M["fixedT"]["adv_tb"]) / (1 + M["narrow"]["adv_tb"])),
                     "per_variant": {int(v): {"spectrum": var_lab.loc[v, "spectrum"], "pb_mm": float(var_lab.loc[v, "pb_mm"]),
                                              **{f"adv_tb_{k}": float(m["adv_tb"][j]) for k, m in M.items()},
                                              **{f"adv_pb_{k}": float(m["adv_pb"][j]) for k, m in M.items()},
                                              **{f"best_tb_{k}": str(m["arg_tb"][j]) for k, m in M.items()},
                                              **{f"best_lan_{k}": name_of(int(m["arg_lan"][j])) for k, m in M.items()},
                                              **{f"frontier_{k}": name_of(int(m["arg_frontier"][j])) for k, m in M.items()},
                                              **{f"q_frontier_{k}": float(m["frontier"][j]) for k, m in M.items()}}
                                     for j, v in enumerate(vids)}},
        "textbook_ratio_to_frontier": tb_ratio_frontier,
        "textbook_rank_among_textbook": tb_rank,
        "textbook_self_ratio_fixedT": {f: _summ(QF[p] / QN[p]) for f, p in tb_pos.items()},
        "textbook_self_ratio_likeforlike": {f: _summ(QL[p] / QN[p]) for f, p in tb_pos.items()},
        "textbook_fl_fraction_at_q_eff": {f: _summ(FLF[p]) for f, p in tb_pos.items()},
        "top200_union_ratio_fixedT": _summ((QF / QN)[np.isin(rows_all, union)]),
        "top200_union_fl_fraction": _summ(FLF[np.isin(rows_all, union)]),
        "random": {"ratio_fixedT_all_variants": _summ(rr_ratio),
                   "ratio_display_variant": _summ(rr_ratio[:, dvj]),
                   "pct_exactly_zero_correction": float(100 * np.mean(rr_ratio == 1.0)),
                   "p90_display": float(np.percentile(rr_ratio[:, dvj], 90)),
                   "p99_display": float(np.percentile(rr_ratio[:, dvj], 99)),
                   "lanthanide_display": _summ(rr_ratio[lan[rnd], dvj]),
                   "non_lanthanide_display": _summ(rr_ratio[~lan[rnd], dvj])},
        "sensitivity": {k: {t: {"n_ge_10pct": s[t]["n_ge_10pct"], "adv_tb": _summ(s[t]["adv_tb"]),
                                "certificate_ok_variants": s[t]["certificate_ok_variants"],
                                "textbook_self_ratio": {f: _summ(x) for f, x in s[t]["ratio_textbook"].items()}}
                            for t in ("fixedT", "likeforlike")} | {"pb_self_ratio": _summ(s["pb_self_ratio"])}
                        for k, s in sens.items()},
        "response_sensitivity": {k: {"narrow": {"n_ge_10pct": o["narrow"]["n_ge_10pct"], "adv_tb": _summ(o["narrow"]["adv_tb"])},
                                     "fixedT": {"n_ge_10pct": o["fixedT"]["n_ge_10pct"], "adv_tb": _summ(o["fixedT"]["adv_tb"])},
                                     "pb_self_ratio": _summ(o["pb_self_ratio"]), "n_rows": o["n_rows"]}
                                 for k, o in resp_out.items()},
        "mono_envelope_47_67": {"max_rel_drop_env": float((1 - env.env_eff / env.env_narrow).max()),
                                "argmax_eff": env_arg.groupby("T")["symbol"].agg(lambda s: sorted(set(s))).to_dict()},
        "offenders_top10": off.head(10)[["symbol", "E_K_keV", "pure_ratio_median", "pure_ratio_max",
                                         "share_median_top200"]].to_dict("records"),
        "textbook_L_fraction_at_q_eff": _textbook_L_fraction(vdf, tb, QF, tb_pos, WU),
    }
    common.write_json(OUT / "summary.json", summary)
    common.write_json(OUT / "_timing.json", {"seconds": time.time() - t_start})
    np.savez(OUT / "decision_arrays.npz", variant_id=vids,
             **{f"{k}_{x}": np.asarray(m[x], dtype=float) for k, m in M.items()
                for x in ("adv_tb", "adv_pb", "adv_pbo", "best_lan", "frontier", "best_tb")},
             **{f"sens_{k}_adv_tb": s["fixedT"]["adv_tb"] for k, s in sens.items()},
             pb_self_ratio=pb_qf / vdf.pb_areal.to_numpy(), T_lfl=T_lfl, el_ratio=el_ratio)

    # ---- figures ----
    figures()
    import shutil
    shutil.rmtree(CACHE, ignore_errors=True)
    log(f"[F] done in {time.time() - t_start:.0f} s; decision: {f3} ({n_stand['fixedT']}/{V} variants ≥ 10 %)")
    return summary


# =============================================================================================
# figures
# =============================================================================================
C_BEFORE = "#8c8c8c"
C_AFTER = "#2a78d6"
C_TB = "#eb6834"
C_PB = "#6250d6"
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def figures() -> list[str]:
    """F_plateau_before_after, F_element_ratio, F_random_distribution from the files run() wrote."""
    plt = common.style()
    cor = pd.read_parquet(OUT / "corrected_areal.parquet")
    arr = np.load(OUT / "decision_arrays.npz")
    vdf = _diag_variants("air_kerma")
    vids = vdf["variant_id"].to_numpy()
    V = vids.size
    dvj = int(np.flatnonzero((vdf.spectrum == DISPLAY[0]).to_numpy() & np.isclose(vdf.pb_mm, DISPLAY[1]))[0])
    v = int(vids[dvj])
    out = []

    # ---- F_plateau_before_after (double column) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(common.DOUBLE_COL_IN, 3.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.35]})
    d = cor[cor.variant_id == v]
    top = d[d.rank_b4_narrow <= TOP_K]
    qn = np.sort(top.q_narrow.to_numpy())
    qf = np.sort(top.q_eff_fixedT.to_numpy())
    r = np.arange(1, qn.size + 1)
    ax1.plot(r, qn, color=C_BEFORE, lw=2, label="B4 top-200, narrow beam")
    ax1.plot(r, qf, color=C_AFTER, lw=2, label="same 200, corrected (E₁, K+L, fixed T*)")
    best = float(d[d.in_b4].q_eff_fixedT.min())
    ax1.plot([4], [best + 0.007], marker="v", color=C_AFTER, ms=5, ls="none",
             label="corrected B4 best (certified)")
    dd = d.set_index("formula")
    show = ["Sn", "Bi", "Sb2O3", "W", "Gd2O3", "Bi2O3", "Pb", "BaSO4"]
    xs = TOP_K * 1.06
    lab = []
    for f in show:
        y0, y1 = float(dd.loc[f, "q_narrow"]), float(dd.loc[f, "q_eff_fixedT"])
        col = C_PB if f == "Pb" else C_TB
        ax1.plot([xs, xs + 14], [y0, y1], color=col, lw=0.7)
        ax1.plot([xs], [y0], marker="o", mfc="white", mec=col, ms=4.5, ls="none")
        ax1.plot([xs + 14], [y1], marker="o", color=col, ms=4.5, ls="none")
        lab.append([y1, f])
    lab.sort()
    gap = 0.016
    for k in range(1, len(lab)):
        lab[k][0] = max(lab[k][0], lab[k - 1][0] + gap)
    for y, f in lab:
        ax1.text(xs + 22, y, f.translate(_SUB), va="center", fontsize=6.5, color="#333333")
    ax1.set_xlim(0, TOP_K * 1.40)
    ax1.set_xticks([1, 50, 100, 150, 200])
    ax1.set_xlabel("rank within the B4 top-200 (each curve sorted)")
    ax1.set_ylabel("required areal mass (g/cm²)")
    ax1.set_title("(a) RQR6, 12°, T* of 0.35 mm Pb", loc="left")
    ax1.text(xs + 7, ax1.get_ylim()[0] + 0.005, "textbook:\nopen = narrow\nfilled = corr.", fontsize=5.3,
             ha="center", va="bottom", color="#555555")
    ax1.legend(loc="upper left", frameon=False, fontsize=6.3)
    ax1.grid(axis="y", color="#e6e6e6", lw=0.5)

    x = np.arange(V)
    ax2.axhline(100 * F3_THRESHOLD, color="#333333", lw=0.8, ls=":")
    ax2.axhline(0, color="#999999", lw=0.5)
    ax2.plot(x, 100 * arr["narrow_adv_pb"], "s", mfc="white", mec=C_PB, ms=3.8, label="Pb, narrow")
    ax2.plot(x, 100 * arr["fixedT_adv_pb"], "s", color=C_PB, ms=3.8, label="Pb, corrected")
    ax2.plot(x, 100 * arr["narrow_adv_tb"], "o", mfc="white", mec=C_TB, ms=4.2,
             label="best textbook excl. Pb/PbO, narrow")
    ax2.plot(x, 100 * arr["fixedT_adv_tb"], "o", color=C_TB, ms=4.2, label="best textbook excl. Pb/PbO, corrected")
    if "sens_E2_adv_tb" in arr.files:
        ax2.plot(x, 100 * arr["sens_E2_adv_tb"], "x", color="#333333", ms=3.5, mew=0.8,
                 label="same, planar-current (E₂) sensitivity")
    if "sens_E2_K_only_adv_tb" in arr.files:
        ax2.plot(x, 100 * arr["sens_E2_K_only_adv_tb"], "+", color="#999999", ms=3.5, mew=0.8,
                 label="same, original model (E₂, K only)")
    for k in range(1, 12):
        ax2.axvline(3 * k - 0.5, color="#eeeeee", lw=0.5, zorder=0)
    ax2.set_xticks(3 * np.arange(12) + 1)
    ax2.set_xticklabels([s_.replace("_th", " ") + "°" for s_ in vdf.spectrum.unique()], rotation=60, ha="right",
                        fontsize=6)
    ax2.set_xlim(-0.8, V - 0.2)
    ax2.set_ylim(-3, 80)
    ax2.set_ylabel("excess areal mass over best B4\nlanthanide compound (%)")
    ax2.set_title("(b) 36 DIAG variants (0.25 / 0.35 / 0.50 mm Pb per group)", loc="left")
    ax2.legend(loc="upper left", frameon=False, fontsize=5.8, ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax2.text(V - 0.6, 100 * F3_THRESHOLD - 0.8, "10 % decision threshold", ha="right", va="top", fontsize=6)
    fig.tight_layout(w_pad=1.5)
    out += common.savefig(fig, "F_plateau_before_after")
    plt.close(fig)

    # ---- F_element_ratio: pure elements ----
    el_ratio = arr["el_ratio"]
    fig, ax = plt.subplots(figsize=(common.SINGLE_COL_IN, 2.4))
    Z = np.arange(1, 101)
    j9 = None
    for qual, col, mk in (("RQR5", "#1baf7a", "o"), ("RQR9", C_AFTER, "s")):
        j = np.flatnonzero((vdf.spectrum == f"{qual}_th12").to_numpy() & np.isclose(vdf.pb_mm, 0.35))[0]
        j9 = j
        ax.plot(Z, 100 * (el_ratio[:, j] - 1), mk, ms=3, color=col, label=f"{qual} 12°, T* of 0.35 mm Pb")
    for z, sym in ((50, "Sn"), (64, "Gd"), (74, "W"), (82, "Pb"), (83, "Bi")):
        ax.annotate(sym, (z, 100 * (el_ratio[z - 1, j9] - 1)), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=6)
    ax.set_xlabel("Z (pure element)")
    ax.set_ylabel("areal-mass increase, fixed T* (%)")
    ax.set_xlim(40, 101)
    ax.grid(color="#eeeeee", lw=0.5)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    out += common.savefig(fig, "F_element_ratio")
    plt.close(fig)

    # ---- F_random_distribution ----
    rd = cor[(cor.variant_id == v) & cor.set_random2000]
    x = 100 * (rd.ratio_fixedT.to_numpy() - 1)
    ln = rd.lanthanide.to_numpy()
    fig, ax = plt.subplots(figsize=(common.SINGLE_COL_IN, 2.4))
    bins = np.linspace(0, max(1.0, np.nanmax(x)) * 1.02, 40)
    ax.hist(x[~ln], bins=bins, color=C_BEFORE, label=f"no La–Lu (n={int((~ln).sum())})")
    ax.hist(x[ln], bins=bins, color=C_AFTER, alpha=0.85, label=f"contains La–Lu (n={int(ln.sum())})")
    ax.set_yscale("log")
    ax.set_xlabel("areal-mass increase, fixed T* (%)")
    ax.set_ylabel("compounds")
    ax.set_title("random 2,000 (R2 base), RQR6 12°, T* of 0.35 mm Pb", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    out += common.savefig(fig, "F_random_distribution")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys as _sys
    figures() if "--figures" in _sys.argv else run()
