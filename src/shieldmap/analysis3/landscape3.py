"""Landscape completion for RQ1.

Regenerate with ``python -m shieldmap.analysis3.landscape3`` from ``shieldmap/`` (or ``landscape3.run()``).
``run()`` is idempotent: every output in ``data/phase3/L/`` and every ``fig1_*``/``fig8_*``/``L_*`` figure
is rewritten from Tables B/C, the stage 1 element partials, the stage 2 universe, the fluorescence module
(``analysis3.fluorescence``) and the COD module (``data/phase3/D/matches.parquet``). Stage results are cached in
``data/phase3/L/`` (``force=True`` recomputes them).

Item 1  Variance decomposition + physics
    decomposition()       stage 2's A1 decomposition, recomputed with stage 2's own functions
                          (``analysis.landscape.decompose`` / ``_decompose_top_mu`` / ``find_crossings``)
                          for R1/R2/R3 × with/without coherent × the four populations, and compared
                          with ``data/phase2/A1/variance_decomposition.parquet`` and ``crossings.csv``.
    z_exponents()         effective Z-exponents of the element partials per energy: least squares of
                          ln σ_k(Z) = a + n_k ln Z (per atom, σ = (μ/ρ)_k A/N_A) and of
                          ln (μ/ρ)_k = a' + n'_k ln Z (per mass) over a stated Z range. For the
                          photoelectric partial three element sets: all elements in the range, the
                          K-active ones (K edge < E) and the K-inactive ones (K edge > E).
    partial_shares()      compound partials by the mixture rule (Table C mass fractions × element
                          partials), share f_k = (μ/ρ)_k/(μ/ρ) per compound, and per energy the
                          quantiles of f_k over the R2 universe. Also: var(log μ/ρ) without the
                          photoelectric partial (counterfactual), the exact split
                          var(f̄_pe Δlog τ + (1 − f̄_pe) Δlog σ_rest), and the medoid compound.
Item 2  Envelope exceedance window
    envelope_window()     every non-radioactive eligible compound whose μ exceeds the max μ of the
                          non-radioactive elemental solids at some grid energy; mechanism per compound.
    envelope_survival()   fluorescence-corrected (the fluorescence module, monoenergetic line, air kerma) required
                          thickness of each such compound vs the best non-radioactive elemental solid,
                          also corrected, at T = 0.1 and 0.01. the hardening revision (F2): the fluorescence module's revised
                          primary model (E₁ fluence kernel, K + L lines); E₂ / K-only sensitivities.
    envelope_density_axis()  the same counts with the compound density moved by its COD density bias
                          (class median and class p95 from analysis3.cod, and the measured COD δ where
                          the entry is matched); elemental DFT-density entries are counted separately
                          (they cannot exceed the experimental elemental envelope physically).
Item 3  Polymorph spread, DFT vs measured
    polymorph_cod()       formulas with ≥ 2 MP polymorphs matched to disjoint COD structure sets.

Definitions (as stage 2)
    x = log10 ρ, y(E) = log10 (μ/ρ)(E); var(log μ) = var(x) + var(y) + 2 cov(x, y) (population moments).
    The crossing (composition share = density share) is var(y) = var(x).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import core
from shieldmap.analysis import landscape as p2
from shieldmap.analysis3 import common

OUT = common.P3 / "L"
P2_A1 = core.P2 / "A1"
N_A = 0.602214076                       # Avogadro / 1e24 (barn conversion): σ[b] = (μ/ρ)[cm²/g]·A/N_A
PARTIALS = ("pe", "incoh", "coh", "pair")
Z_RANGE = (6, 83)                       # C … Bi: primary fit range for the Z-exponents
Z_RANGES_SENS = ((3, 94), (20, 83), (6, 56))
KEY_E = (20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 235.0, 500.0, 661.657, 1000.0, 1250.0, 5000.0)
WINDOW_REPORT_KEV = (47.0, 67.0)        # Stage 2's reported window (gate.md)
T_LEVELS = (0.1, 0.01)
ENVELOPE_SENS = {"E2": {"escape": "E2"}, "K_only": {"l_shell": False},             # T = 0.1 sensitivities
                 "E2_K_only": {"escape": "E2", "l_shell": False}, "xraylib_jump": {"jump": "xraylib"}}
SEED = 20261012

# dataviz reference palette (light, categorical slots 1-4 in fixed order); ink tokens
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


# =============================================================================================
# helpers
# =============================================================================================
def energies() -> np.ndarray:
    return common.energy_keV()


def _nearest_idx(e_keV: float) -> int:
    return int(np.argmin(np.abs(np.log(energies()) - math.log(e_keV))))


def _save(df: pd.DataFrame, name: str, csv: bool = True) -> None:
    paths.ensure(OUT)
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    if csv:
        df.to_csv(OUT / f"{name}.csv", index=False)


def _dist(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(v.size), "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q95": q[4],
            "min": float(v.min()), "max": float(v.max())}


def _ax_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def log_crossing(e: np.ndarray, d: np.ndarray, direction: str = "down") -> list[float]:
    """Zero crossings of d on the grid e, linear interpolation in log10 E.

    direction 'down' (+ → −), 'up' (− → +) or 'any'."""
    out = []
    le = np.log10(e)
    for i in range(len(d) - 1):
        a, b = d[i], d[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a == b:
            continue
        if direction in ("down", "any") and a > 0 >= b or direction in ("up", "any") and a < 0 <= b:
            t = a / (a - b)
            out.append(float(10 ** (le[i] + t * (le[i + 1] - le[i]))))
    return out


# =============================================================================================
# item 1a: decomposition (stage 2 functions) and comparison with stage 2
# =============================================================================================
def decomposition(rows_elig: np.ndarray | None = None, block: int = p2.BLOCK) -> pd.DataFrame:
    """Stage 2's per-energy decomposition (all rules × coherent × four subsets), recomputed."""
    u = common.universe()
    e = energies()
    elig = np.flatnonzero(u.eligible.to_numpy()) if rows_elig is None else rows_elig
    x = np.log10(u.density.to_numpy(dtype=np.float64)[elig])
    rule_masks = {r: u[f"in_{r}"].to_numpy(dtype=bool)[elig] for r in core.RULES}
    parts = []
    for coh in p2.COHS:
        for sl, L, _blk in p2.iter_blocks(coh, elig, block):
            js = np.arange(sl.start, sl.stop)
            for rule, m in rule_masks.items():
                xr, Yr = x[m], L[m]
                thr = np.quantile(xr, 0.9)
                top = xr >= thr
                res = {"all": p2.decompose(xr, Yr), "top10_rho": p2.decompose(xr[top], Yr[top]),
                       "top10_mu": p2._decompose_top_mu(xr, Yr, 0.10), "top1_mu": p2._decompose_top_mu(xr, Yr, 0.01)}
                for sub, d in res.items():
                    df = pd.DataFrame(d)
                    df.insert(0, "energy_keV", e[js]); df.insert(1, "rule", rule)
                    df.insert(2, "coherent", coh == "wcoh"); df.insert(3, "subset", sub)
                    parts.append(df)
    return pd.concat(parts, ignore_index=True)


def crossings(dec: pd.DataFrame) -> pd.DataFrame:
    edges = common.edges().merge(common.elements_table()[["Z", "symbol"]], on="Z")
    present = set(core.element_list(common.universe()))
    edges = edges[edges.shell.isin(["K", "L1", "L2", "L3"]) & edges.symbol.isin(present)]
    cr = []
    for (rule, coh, sub), d in dec.groupby(["rule", "coherent", "subset"]):
        d = d.sort_values("energy_keV")
        for c in p2.find_crossings(d.energy_keV.to_numpy(), (d.share_composition - d.share_density).to_numpy(), edges):
            c.update({"rule": rule, "coherent": coh, "subset": sub})
            cr.append(c)
    return pd.DataFrame(cr)[["subset", "rule", "coherent", "E_cross_keV", "direction", "E_lo_keV", "E_hi_keV",
                             "d_lo", "d_hi", "edge_between_nodes"]]


def compare_phase2(dec: pd.DataFrame, cross: pd.DataFrame) -> dict:
    """Max abs differences against stage 2's stored decomposition and crossings."""
    old = pd.read_parquet(P2_A1 / "variance_decomposition.parquet")
    key = ["energy_keV", "rule", "coherent", "subset"]
    m = dec.merge(old, on=key, suffixes=("", "_p2"))
    out = {"n_rows_p3": int(len(dec)), "n_rows_p2": int(len(old)), "n_rows_matched": int(len(m))}
    for c in ("var_logmu", "var_logrho", "var_logmurho", "cov", "share_density", "share_composition", "share_cov"):
        out[f"max_abs_diff_{c}"] = float(np.nanmax(np.abs(m[c] - m[f"{c}_p2"])))
    oc = pd.read_csv(P2_A1 / "crossings.csv")
    k2 = ["subset", "rule", "coherent", "direction", "E_lo_r"]
    a = cross.assign(E_lo_r=cross.E_lo_keV.round(6)); b = oc.assign(E_lo_r=oc.E_lo_keV.round(6))
    mc = a.merge(b, on=k2, suffixes=("", "_p2"), how="outer", indicator=True)
    out["crossings_n_p3"] = int(len(cross)); out["crossings_n_p2"] = int(len(oc))
    out["crossings_unmatched"] = int((mc["_merge"] != "both").sum())
    both = mc[mc["_merge"] == "both"]
    out["crossings_max_abs_diff_keV"] = float(np.abs(both.E_cross_keV - both.E_cross_keV_p2).max())
    return out


# =============================================================================================
# item 1b: Z-exponents of the element partials
# =============================================================================================
def fit_power_law(Z: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares ln y = a + n ln Z over the finite, positive points. Returns (n, a, R²)."""
    Z = np.asarray(Z, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > 0) & (Z > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(Z[ok]), np.log(y[ok])
    n, a = np.polyfit(lx, ly, 1)
    res = ly - (a + n * lx)
    ss = ((ly - ly.mean()) ** 2).sum()
    return float(n), float(a), float(1 - (res ** 2).sum() / ss) if ss > 0 else float("nan")


def element_partials() -> dict:
    """Element partial μ/ρ (100 × 545), pair = nuclear + electron field, plus Z, A, K edges."""
    xs = common.element_xsec()
    ed = common.edges()
    k = ed[ed.shell == "K"].set_index("Z").energy_keV
    Z = xs["Z"].astype(int)
    return {"Z": Z, "A": xs["mass_used"].astype(float), "pe": xs["pe"], "incoh": xs["incoh"], "coh": xs["coh"],
            "pair": xs["pair_nuc"] + xs["pair_el"], "total_wcoh": xs["total_wcoh"],
            # Z without a tabulated K edge (Z ≤ 10, K edge < 1.07 keV, below the 10 keV grid) → 0: K-active
            "K_edge_keV": np.array([k.get(z, 0.0) for z in Z], dtype=float)}


def z_exponents(z_range=Z_RANGE, ep: dict | None = None) -> pd.DataFrame:
    """Per energy: per-atom (n) and per-mass (n') exponents of each partial over z_range."""
    ep = element_partials() if ep is None else ep
    e = energies()
    Z, A, K = ep["Z"], ep["A"], ep["K_edge_keV"]
    inr = (Z >= z_range[0]) & (Z <= z_range[1])
    rows = []
    for j, ek in enumerate(e):
        rec = {"energy_keV": ek, "Z_lo": z_range[0], "Z_hi": z_range[1]}
        sets = {"": inr}
        kact = inr & (K < ek)
        kin = inr & ~(K < ek)
        for part in PARTIALS:
            v = ep[part][:, j]
            psets = {"": inr, "_Kactive": kact, "_Kinactive": kin} if part == "pe" else sets
            for tag, s in psets.items():
                n_at, _, r2_at = fit_power_law(Z[s], v[s] * A[s] / N_A)
                n_m, _, r2_m = fit_power_law(Z[s], v[s])
                rec[f"n_atom_{part}{tag}"] = n_at; rec[f"r2_atom_{part}{tag}"] = r2_at
                rec[f"n_mass_{part}{tag}"] = n_m; rec[f"r2_mass_{part}{tag}"] = r2_m
                rec[f"n_el_{part}{tag}"] = int((s & (v > 0)).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================================
# item 1c: partial shares over the universe, counterfactual variance, medoid compound
# =============================================================================================
def share_quantiles(P: dict[str, np.ndarray], qs=(0.25, 0.5, 0.75)) -> dict[str, np.ndarray]:
    """P: partial → (n × k) compound partial μ/ρ. Returns partial → (len(qs) × k) quantiles of the share
    f_k = P_k / Σ P over the n compounds (columns independently)."""
    tot = sum(P.values())
    return {k: np.quantile(v / tot, qs, axis=0) for k, v in P.items()}


def partial_shares(rule: str = "R2", block: int = 48, check_rows: int = 2000) -> dict:
    """Per energy over the compounds of `rule`: quantiles of the four partial shares, variance of log10 of
    the full μ/ρ (mixture rule), of μ/ρ without the photoelectric partial, of each partial, the
    split y = log10(μ/ρ − τ) − log10(1 − f_pe) and the medoid compound (closest log μ/ρ curve to the pointwise median, RMS over
    the grid). Also checks the mixture-rule total against Table C (with coherent)."""
    u = common.universe()
    rows = np.flatnonzero(u[f"in_{rule}"].to_numpy(bool))
    W = common.mass_fractions()[rows]
    ep = element_partials()
    e = energies()
    x = np.log10(u.density.to_numpy(float)[rows])
    tab = common.mu_rho(True)
    rng = np.random.default_rng(SEED)
    chk = np.sort(rng.choice(rows.size, size=min(check_rows, rows.size), replace=False))
    recs = []
    med_logs = np.empty(e.size)
    dev2 = np.zeros(rows.size)
    max_rel_tabc = 0.0
    for j0 in range(0, e.size, block):
        js = np.arange(j0, min(e.size, j0 + block))
        P = {k: np.asarray(W @ ep[k][:, js]) for k in PARTIALS}
        tot = P["pe"] + P["incoh"] + P["coh"] + P["pair"]
        tc = np.asarray(tab[rows[chk]][:, js], dtype=np.float64)
        max_rel_tabc = max(max_rel_tabc, float(np.max(np.abs(tot[chk] / tc - 1))))
        y = np.log10(tot)
        nope = np.log10(tot - P["pe"])
        q = share_quantiles(P, (0.05, 0.25, 0.5, 0.75, 0.95))
        f_pe = P["pe"] / tot
        med = np.median(y, axis=0)
        med_logs[js] = med
        dev2 += ((y - med) ** 2).sum(axis=1)
        for t, j in enumerate(js):
            rec = {"energy_keV": e[j], "rule": rule, "n": rows.size}
            for k in PARTIALS:
                for qi, qq in enumerate((5, 25, 50, 75, 95)):
                    rec[f"share_{k}_q{qq:02d}"] = q[k][qi, t]
                rec[f"share_{k}_mean"] = float((P[k][:, t] / tot[:, t]).mean())
                pk = P[k][:, t]
                rec[f"var_log_{k}"] = float(np.var(np.log10(pk))) if (pk > 0).all() else np.nan
            yt = y[:, t]
            rec["var_logmurho"] = float(np.var(yt))
            rec["var_logmurho_no_pe"] = float(np.var(nope[:, t]))
            rec["var_logrho"] = float(np.var(x))
            rec["cov_y_x"] = float(np.mean((yt - yt.mean()) * (x - x.mean())))
            # exact split y = log10(rest) + g, g = −log10(1 − f_pe) = log10(1 + τ/rest):
            # var(y) = var(log rest) + var(g) + 2 cov(log rest, g); g depends on the pe share only
            g = -np.log10(1.0 - f_pe[:, t])
            nr = nope[:, t]
            rec["var_pe_term"] = float(np.var(g))
            rec["cov2_rest_pe_term"] = float(2 * np.mean((g - g.mean()) * (nr - nr.mean())))
            rec["var_log_pe_over_rest"] = float(np.var(np.log10(P["pe"][:, t]) - nr))
            rec["sd_share_pe"] = float(np.std(f_pe[:, t]))
            recs.append(rec)
    df = pd.DataFrame(recs)
    medoid = int(rows[np.argmin(dev2)])
    rms = float(np.sqrt(dev2.min() / e.size))
    return {"table": df, "medoid_row": medoid, "medoid_rms_log10": rms, "median_log10_mu_rho": med_logs,
            "max_rel_mixture_vs_tableC": max_rel_tabc, "n_check_rows": int(chk.size)}


def compound_shares(row: int) -> pd.DataFrame:
    """Partial shares of one compound (Table C row) on the grid."""
    W = common.mass_fractions()[[row]]
    ep = element_partials()
    P = {k: np.asarray(W @ ep[k]).ravel() for k in PARTIALS}
    tot = sum(P.values())
    return pd.DataFrame({"energy_keV": energies(), **{f"share_{k}": P[k] / tot for k in PARTIALS},
                         "mu_rho": tot})


# =============================================================================================
# item 2: envelope exceedance window
# =============================================================================================
def nonrad_elements() -> pd.DataFrame:
    """Non-radioactive elements with an experimental solid density (the envelope's members)."""
    el = common.elements_table()
    return el[~el.is_radioactive & np.isfinite(el.density_solid_exp.astype(float))].reset_index(drop=True)


def envelope_scan(block: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-grid scan of eligible non-radioactive entries: μ > max μ of the non-radioactive elemental solids.

    Returns (per energy counts R2/R3, long table of every (entry, energy) exceedance)."""
    u = common.universe()
    e = energies()
    nre = p2.nonradioactive_envelope()
    env = nre.nonrad_max_mu_wcoh.to_numpy()
    nr = np.flatnonzero(u.eligible.to_numpy() & ~u.contains_radioactive.to_numpy())
    rho = u.density.to_numpy(float)[nr]
    in_r2 = u.in_R2.to_numpy(bool)[nr]
    tab = common.mu_rho(True)
    counts, hits = [], []
    for j0 in range(0, e.size, block):
        js = np.arange(j0, min(e.size, j0 + block))
        mr = np.asarray(tab[:, j0:js[-1] + 1][nr], dtype=np.float64)
        ratio = mr * rho[:, None] / env[js]
        for t, j in enumerate(js):
            over = ratio[:, t] > 1
            counts.append({"energy_keV": e[j], "n_R3": int(over.sum()), "n_R2": int((over & in_r2).sum()),
                           "env_element": nre.nonrad_argmax_mu_wcoh.iloc[j], "env_mu": env[j]})
            for i in np.flatnonzero(over):
                hits.append((int(nr[i]), float(e[j]), int(j), float(ratio[i, t]), float(mr[i, t])))
    h = pd.DataFrame(hits, columns=["row", "energy_keV", "e_idx", "ratio_mu", "mu_rho"])
    return pd.DataFrame(counts), h


def mechanism(row: int, e_keV: float, e_idx: int, ep: dict, el: pd.DataFrame, nre: pd.DataFrame) -> dict:
    """Mechanism attribution of one exceedance (compound row at grid energy e_idx).

    edge element  among constituents whose K edge lies 0–15 keV below E ('K edge just below'), the one with
                  the largest photoelectric contribution w_Z τ_Z(E); if there is none, the constituent with
                  the largest w_Z τ_Z(E) (mechanism 'other'). Reported with its K edge and gap E − E_K.
    partner       the other constituent with the highest elemental solid density.
    ratios        compound vs envelope element: μ/ρ ratio × ρ ratio = μ ratio; compound vs edge-element
                  solid (ρ and μ)."""
    u = common.universe()
    W = common.mass_fractions()[row].toarray().ravel()
    zs = np.flatnonzero(W > 0) + 1
    pe = W[zs - 1] * ep["pe"][zs - 1, e_idx]
    tot = float(W @ ep["total_wcoh"][:, e_idx])
    gap = e_keV - ep["K_edge_keV"][zs - 1]
    near = (gap >= 0) & (gap <= 15.0)
    # the K-edge constituent: largest w·τ among those with a K edge 0–15 keV below E, else largest w·τ
    ze = int(zs[near][np.argmax(pe[near])]) if near.any() else int(zs[np.argmax(pe)])
    sym = el.set_index("Z").symbol
    rho_el = el.set_index("Z").density_solid_exp.astype(float)
    ek = float(ep["K_edge_keV"][ze - 1])
    others = [z for z in zs if z != ze]
    zp = int(max(others, key=lambda z: (np.nan_to_num(rho_el.get(z, np.nan), nan=-1), z))) if others else None
    env_sym = nre.nonrad_argmax_mu_wcoh.iloc[e_idx]
    zenv = int(el.loc[el.symbol == env_sym, "Z"].iloc[0])
    rho_env = float(rho_el[zenv]); mr_env = float(ep["total_wcoh"][zenv - 1, e_idx])
    rho_c = float(u.density.iloc[row])
    mr_edge = float(ep["total_wcoh"][ze - 1, e_idx]); rho_edge = float(rho_el.get(ze, np.nan))
    return {"edge_element": sym[ze], "edge_Z": ze, "edge_K_keV": ek, "gap_E_minus_K_keV": e_keV - ek,
            "K_edge_just_below": bool(0 <= e_keV - ek <= 15.0), "edge_mass_fraction": float(W[ze - 1]),
            "edge_pe_share_of_compound_mu_rho": float(pe[list(zs).index(ze)] / tot),
            "partner": sym[zp] if zp else "", "partner_Z": zp or 0,
            "partner_mass_fraction": float(W[zp - 1]) if zp else np.nan,
            "partner_solid_density": float(rho_el.get(zp, np.nan)) if zp else np.nan,
            "env_element": env_sym, "env_density": rho_env, "env_mu_rho": mr_env,
            "ratio_mu_rho_vs_env": tot / mr_env, "ratio_rho_vs_env": rho_c / rho_env,
            "edge_solid_density": rho_edge, "ratio_rho_vs_edge_solid": rho_c / rho_edge,
            "edge_solid_mu_over_env_mu": mr_edge * rho_edge / (mr_env * rho_env)}


def mechanism_label(m: dict) -> str:
    if not m["partner"]:
        return "elemental entry (DFT ρ > experimental ρ)"
    if not m["K_edge_just_below"]:
        return "other"
    if m["partner"] and m["partner_solid_density"] > m["edge_solid_density"]:
        return "K edge + dense partner"
    return "K edge + compact packing (light partner)"


def envelope_window(hits: pd.DataFrame) -> pd.DataFrame:
    """One row per exceeding entry (at its energy of maximum μ ratio) with mechanism columns."""
    if hits.empty:
        return pd.DataFrame()
    u = common.universe()
    ep = element_partials()
    el = common.elements_table()
    nre = p2.nonradioactive_envelope()
    g = hits.groupby("row")
    best = hits.loc[g.ratio_mu.idxmax()].set_index("row")
    agg = g.agg(first_energy_keV=("energy_keV", "min"), last_energy_keV=("energy_keV", "max"),
                n_grid_energies=("energy_keV", "size"))
    b4 = core.bundle_mask("B4", u)
    recs = []
    for r, b in best.iterrows():
        m = mechanism(int(r), float(b.energy_keV), int(b.e_idx), ep, el, nre)
        ur = u.iloc[int(r)]
        recs.append({"row": int(r), "material_id": ur.material_id, "formula_pretty": ur.formula_pretty,
                     "density": float(ur.density), "energy_above_hull": float(ur.energy_above_hull),
                     "theoretical": bool(ur.theoretical), "in_R2": bool(ur.in_R2),
                     "in_B4": bool(b4[int(r)]),
                     "max_ratio_mu_nonrad_env": float(b.ratio_mu), "energy_at_max_keV": float(b.energy_keV),
                     **agg.loc[r].to_dict(), **m, "mechanism": mechanism_label(m)})
    return pd.DataFrame(recs).sort_values(["max_ratio_mu_nonrad_env", "material_id"], ascending=[False, True]
                                          ).reset_index(drop=True)


def survival_compare(t_c: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """Thickness advantage of compounds over the reference (best elemental solid): t_ref/t_c − 1 (> 0: the
    compound needs less thickness, i.e. it survives)."""
    return np.asarray(t_ref, float) / np.asarray(t_c, float) - 1.0


def envelope_survival(hits: pd.DataFrame, T_levels=T_LEVELS, **model_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fluorescence-corrected required thickness (cm) of every exceeding (entry, energy) vs the best
    non-radioactive elemental solid at that energy (all non-radioactive solids corrected, minimum taken).

    Monoenergetic line source, air-kerma response, the fluorescence module primary model (E₁ fluence escape, K + L lines,
    XCOM jump ratios, 20–150 keV K-edge window, with coherent) unless model_kw says otherwise. Compounds
    use MP (DFT) densities, elements experimental solid densities; envelope_density_axis() varies the
    compound density."""
    from shieldmap.analysis3 import fluorescence as fl
    u = common.universe()
    W = common.mass_fractions()
    nrel = nonrad_elements()
    Wel = np.zeros((len(nrel), 100)); Wel[np.arange(len(nrel)), nrel.Z.to_numpy() - 1] = 1.0
    rho_el = nrel.density_solid_exp.to_numpy(float)
    recs, elrecs = [], []
    for ek, h in hits.groupby("energy_keV"):
        rows = h.row.to_numpy()
        Wc = W[rows].toarray()
        rho_c = u.density.to_numpy(float)[rows]
        for T in T_levels:
            rc = fl.mono_effective(Wc, float(ek), T, **model_kw)
            re_ = fl.mono_effective(Wel, float(ek), T, **model_kw)
            th_el_n = re_["q_narrow"] / rho_el
            th_el_e = re_["q_eff"] / rho_el
            i_n, i_e = int(np.argmin(th_el_n)), int(np.argmin(th_el_e))
            elrecs.append({"energy_keV": ek, "T": T, "best_narrow": nrel.symbol.iloc[i_n],
                           "t_best_narrow_cm": th_el_n[i_n], "best_eff": nrel.symbol.iloc[i_e],
                           "t_best_eff_cm": th_el_e[i_e], "best_eff_fl_fraction": float(re_["fl_fraction"][i_e])})
            tn = rc["q_narrow"] / rho_c; te = rc["q_eff"] / rho_c
            for k, r in enumerate(rows):
                recs.append({"row": int(r), "energy_keV": float(ek), "T": T, "t_narrow_cm": tn[k],
                             "t_eff_cm": te[k], "fl_fraction": float(rc["fl_fraction"][k]),
                             "eff_increase": te[k] / tn[k] - 1,
                             "best_el_narrow": nrel.symbol.iloc[i_n], "t_best_el_narrow_cm": th_el_n[i_n],
                             "best_el_eff": nrel.symbol.iloc[i_e], "t_best_el_eff_cm": th_el_e[i_e],
                             "advantage_narrow": float(survival_compare(tn[k], th_el_n[i_n])),
                             "advantage_eff": float(survival_compare(te[k], th_el_e[i_e]))})
    s = pd.DataFrame(recs)
    s["survives"] = s.advantage_eff > 0
    s = s.merge(u[["row", "material_id", "formula_pretty", "in_R2"]], on="row", how="left")
    s["model_version"] = fl.MODEL_VERSION
    el = pd.DataFrame(elrecs)
    el["model_version"] = fl.MODEL_VERSION
    return s, el


DENSITY_CASES = ("nominal", "class_median", "class_p95", "measured_or_nominal", "measured_or_p95")


def envelope_deltas(rows: np.ndarray) -> pd.DataFrame:
    """Density-bias δ = ρ_MP/ρ_true − 1 for each row under DENSITY_CASES (thickness scales by 1 + δ).

    class_*: the row's exclusive primary class, COD compound distribution (analysis3.cod); measured: the
    entry's own COD δ when it is matched and not excluded by the COD-module screen."""
    from shieldmap.analysis3 import cod
    rows = np.asarray(rows, dtype=int)
    cls = common.universe_primary_class()[rows]
    dist = cod.class_delta_distribution()
    med = {c: float(np.median(dist[c])) for c in set(cls)}
    p95 = {c: float(cod.band(c)[1]) for c in set(cls)}
    m = pd.read_parquet(common.P3 / "D" / "matches.parquet", columns=["row", "delta", "excluded"])
    m = m[~m.excluded.astype(bool)].groupby("row").delta.median()
    meas = m.reindex(rows).to_numpy()
    out = pd.DataFrame({"row": rows, "primary_class": cls,
                        "nominal": 0.0, "class_median": [med[c] for c in cls], "class_p95": [p95[c] for c in cls],
                        "measured_delta": meas})
    out["measured_or_nominal"] = np.where(np.isfinite(meas), meas, 0.0)
    out["measured_or_p95"] = np.where(np.isfinite(meas), meas, out.class_p95)
    return out


def envelope_density_axis(win: pd.DataFrame, surv: pd.DataFrame | None,
                          sens: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per entry × density case: narrow-beam exceedance (max μ ratio / (1 + δ) > 1) and corrected survival
    ((1 + advantage) / (1 + δ) − 1 > 0) at each T; long table and a count table (R2 and R3, multi-element
    entries and elemental entries separately)."""
    dl = envelope_deltas(win.row.to_numpy())
    base = win[["row", "material_id", "formula_pretty", "in_R2", "theoretical", "mechanism",
                "max_ratio_mu_nonrad_env"]].merge(dl, on="row")
    base["elemental_entry"] = base.mechanism.str.startswith("elemental entry")
    models = {"primary": surv} | {f"sens_{k}": v for k, v in (sens or {}).items()}
    recs = []
    for case in DENSITY_CASES:
        d = base[case].to_numpy()
        rec = base[["row", "in_R2", "elemental_entry"]].copy()
        rec["case"] = case
        rec["delta"] = d
        rec["exceeds_narrow"] = base.max_ratio_mu_nonrad_env.to_numpy() / (1 + d) > 1
        for name, sv in models.items():
            if sv is None:
                continue
            for T in sorted(sv["T"].unique()):
                st = sv[sv["T"] == T].groupby("row").advantage_eff.max().reindex(base.row).to_numpy()
                rec[f"survives_{name}_T{T:g}"] = (1 + st) / (1 + d) - 1 > 0
        recs.append(rec)
    long = pd.concat(recs, ignore_index=True)
    cols = [c for c in long.columns if c.startswith(("exceeds_", "survives_"))]
    cnt = []
    lp = long.assign(pop=np.where(long.elemental_entry, "elemental", "multi_element"))
    for (case, pop), g in lp.groupby(["case", "pop"]):
        for scope, gg in (("R3", g), ("R2", g[g.in_R2])):
            cnt.append({"case": case, "population": pop, "scope": scope, "n_entries": int(len(gg)),
                        **{c: int(gg[c].sum()) for c in cols}})
    return long, pd.DataFrame(cnt)


# =============================================================================================
# item 3: polymorph spread, DFT vs measured (COD)
# =============================================================================================
def spread(v) -> float:
    v = np.asarray(v, float)
    return float(v.max() / v.min() - 1.0)


def pair_concordance(a, b, min_rel_b: float = 0.0) -> tuple[int, int, int]:
    """(concordant, discordant, tied) pairs of the orderings of a and b (ties in either count as tied).

    min_rel_b > 0 keeps only pairs whose b values differ by more than that relative amount
    (|b_i/b_j − 1| > min_rel_b, larger over smaller): resolvable measured differences."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = d = t = 0
    for i in range(a.size):
        for j in range(i + 1, a.size):
            if min_rel_b > 0 and max(b[i], b[j]) / min(b[i], b[j]) - 1 <= min_rel_b:
                continue
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
            else:
                t += 1
    return c, d, t


def disjoint_polymorphs(g: pd.DataFrame) -> pd.DataFrame:
    """Keep MP entries of one formula whose COD equivalence sets are pairwise disjoint and whose selected
    COD ids differ (greedy in order of e_hull, then material_id): each kept entry is a distinct structure
    in both databases."""
    g = g.sort_values(["e_hull", "material_id"])
    used: set[str] = set()
    keep = []
    for i, r in g.iterrows():
        ids = set(str(r.cod_ids_equivalent).split(",")) | {str(r.cod_id)}
        if ids & used:
            continue
        used |= ids
        keep.append(i)
    return g.loc[keep]


def polymorph_table(m: pd.DataFrame, min_n: int = 2) -> pd.DataFrame:
    recs = []
    for f, g in m.groupby("formula_reduced", sort=True):
        if len(g) < min_n:
            continue
        k = disjoint_polymorphs(g)
        if len(k) < min_n:
            continue
        c, d, t = pair_concordance(k.rho_mp, k.rho_cod)
        c1, d1, _ = pair_concordance(k.rho_mp, k.rho_cod, 0.01)
        c3, d3, _ = pair_concordance(k.rho_mp, k.rho_cod, 0.03)
        imp, imc = int(np.argmax(k.rho_mp.to_numpy())), int(np.argmax(k.rho_cod.to_numpy()))
        recs.append({"formula_reduced": f, "n_poly": len(k), "material_ids": ",".join(k.material_id),
                     "cod_ids": ",".join(map(str, k.cod_id)),
                     "spread_dft": spread(k.rho_mp), "spread_cod": spread(k.rho_cod),
                     "rho_mp_min": float(k.rho_mp.min()), "rho_mp_max": float(k.rho_mp.max()),
                     "rho_cod_min": float(k.rho_cod.min()), "rho_cod_max": float(k.rho_cod.max()),
                     "pairs_concordant": c, "pairs_discordant": d, "pairs_tied": t,
                     "pairs_conc_cod_gt1pct": c1, "pairs_disc_cod_gt1pct": d1,
                     "pairs_conc_cod_gt3pct": c3, "pairs_disc_cod_gt3pct": d3,
                     "densest_same": bool(imp == imc),
                     "max_ehull": float(k.e_hull.max()), "all_ehull_le_005": bool((k.e_hull <= 0.05).all()),
                     "any_hp_marker": bool(k.get("cod_title_hp_marker", pd.Series(False, index=k.index)).any()),
                     "any_ht_marker": bool(k.get("cod_title_ht_marker", pd.Series(False, index=k.index)).any()),
                     "max_abs_delta": float(np.abs(k.delta).max()),
                     "primary_class": k.primary_class.iloc[0] if "primary_class" in k else ""})
    return pd.DataFrame(recs)


def polymorph_summary(pt: pd.DataFrame) -> dict:
    if pt.empty:
        return {"n_formulas": 0}
    out = {"n_formulas": int(len(pt)), "n_entries": int(pt.n_poly.sum()),
           "spread_dft": _dist(pt.spread_dft), "spread_cod": _dist(pt.spread_cod),
           "diff_dft_minus_cod": _dist(pt.spread_dft - pt.spread_cod),
           "ratio_dft_over_cod": _dist((pt.spread_dft / pt.spread_cod).replace([np.inf], np.nan))}
    c, d, t = int(pt.pairs_concordant.sum()), int(pt.pairs_discordant.sum()), int(pt.pairs_tied.sum())
    out.update({"pairs_concordant": c, "pairs_discordant": d, "pairs_tied": t,
                "frac_pairs_concordant": c / max(1, c + d),
                "frac_formulas_order_fully_preserved": float(((pt.pairs_discordant == 0)).mean()),
                "frac_formulas_densest_same": float(pt.densest_same.mean()),
                "frac_concordant_formula_mean": float((pt.pairs_concordant / (pt.pairs_concordant + pt.pairs_discordant)
                                                       ).mean()),
                "frac_dft_spread_gt_cod": float((pt.spread_dft > pt.spread_cod).mean())})
    big = pt.n_poly.idxmax()
    rest = pt.drop(index=big)
    cr, dr = int(rest.pairs_concordant.sum()), int(rest.pairs_discordant.sum())
    out["largest_formula"] = {"formula": pt.loc[big, "formula_reduced"], "n_poly": int(pt.loc[big, "n_poly"]),
                              "frac_pairs_concordant_without_it": cr / max(1, cr + dr)}
    try:
        from scipy.stats import spearmanr
        r = spearmanr(pt.spread_dft, pt.spread_cod)
        out["spearman_spread_dft_vs_cod"] = float(r.statistic)
    except Exception:  # pragma: no cover
        pass
    # pair level: only pairs whose measured densities differ by > 1 % / > 3 %
    for tag in ("1pct", "3pct"):
        if f"pairs_conc_cod_gt{tag}" in pt:
            cc, dd = int(pt[f"pairs_conc_cod_gt{tag}"].sum()), int(pt[f"pairs_disc_cod_gt{tag}"].sum())
            out[f"pairs_measured_diff_gt_{tag}"] = {"concordant": cc, "discordant": dd, "frac_concordant": cc / max(1, cc + dd)}
    # concordance by measured spread (resolvable differences)
    for thr in (0.01, 0.03):
        s = pt[pt.spread_cod > thr]
        cc, dd = int(s.pairs_concordant.sum()), int(s.pairs_discordant.sum())
        out[f"frac_pairs_concordant_cod_spread_gt_{thr:g}"] = cc / max(1, cc + dd)
        out[f"n_formulas_cod_spread_gt_{thr:g}"] = int(len(s))
    return out


def polymorph_cod(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pt = polymorph_table(m)
    s = {"primary": polymorph_summary(pt)}
    if not pt.empty:
        s["ehull_le_005"] = polymorph_summary(pt[pt.all_ehull_le_005])
        s["no_hp_ht_marker"] = polymorph_summary(pt[~pt.any_hp_marker & ~pt.any_ht_marker])
        s["n_poly_2"] = polymorph_summary(pt[pt.n_poly == 2])
        s["n_poly_ge3"] = polymorph_summary(pt[pt.n_poly >= 3])
        s["all_members_abs_delta_le_0.2"] = polymorph_summary(pt[pt.max_abs_delta <= 0.2])
        s["ehull_le_005_and_abs_delta_le_0.2"] = polymorph_summary(pt[pt.all_ehull_le_005 & (pt.max_abs_delta <= 0.2)])
        s["n_formulas_any_abs_delta_gt_0.2"] = int((pt.max_abs_delta > 0.2).sum())
    # sensitivity: distinct selected COD ids only (equivalence sets may overlap)
    alt = []
    for f, g in m.groupby("formula_reduced"):
        g2 = g.drop_duplicates("cod_id")
        if len(g2) >= 2:
            alt.append({"spread_dft": spread(g2.rho_mp), "spread_cod": spread(g2.rho_cod)})
    alt = pd.DataFrame(alt)
    if len(alt):
        s["sens_distinct_selected_ids_only"] = {"n_formulas": int(len(alt)), "spread_dft": _dist(alt.spread_dft),
                                                "spread_cod": _dist(alt.spread_cod)}
    return pt, s


# =============================================================================================
# figures
# =============================================================================================
def fig_variance_physics(dec, cross, sh, zx, diag_e):
    """Dossier Figure 1 (double column): (a)-(d) stage 2's four populations; (e) var(log μ/ρ) vs var(log ρ)
    with the no-photoelectric counterfactual; (f) partial shares of μ/ρ over the R2 map; (g)/(h) Z-exponents
    per atom and per mass."""
    plt = common.style()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(4, 2, figsize=(common.DOUBLE_COL_IN, 9.4), sharex=True)
    d_all = dec.assign(ratio=dec.var_logmurho / dec.var_logrho)
    xc = cross[(cross.subset == "all") & (cross.rule == "R2") & cross.coherent & (cross.direction == "comp->dens")]
    e_cross = float(xc.E_cross_keV.iloc[-1])
    CR = "#6250d6"                                   # variance-ratio colour (c, d), distinct from the shares
    titles = {"all": "(a) R2 map: shares of var(log μ)", "top10_rho": "(b) densest 10 %: shares",
              "top10_mu": "(c) best 10 % by μ: var(log μ/ρ)/var(log ρ)",
              "top1_mu": "(d) best 1 % by μ: var(log μ/ρ)/var(log ρ)"}
    for ax, sub in zip(axes.flat[:4], p2.SUBSETS):
        _ax_style(ax)
        d = d_all[d_all.subset == sub]
        prim = d[(d.rule == "R2") & d.coherent].sort_values("energy_keV")
        e = prim.energy_keV.to_numpy()
        ax.axvspan(diag_e.min(), diag_e.max(), color="#d9d8d3", alpha=0.5, lw=0)
        cr = cross[(cross.subset == sub) & (cross.rule == "R2") & cross.coherent]
        if sub in ("all", "top10_rho"):
            for col, c in (("share_density", C1), ("share_composition", C2), ("share_cov", C3)):
                piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values=col)
                ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=c, alpha=0.2, lw=0)
                ax.plot(e, prim[col], color=c, lw=1.2)
            if sub == "all":
                t = sh.sort_values("energy_keV")
                ax.fill_between(t.energy_keV, t.share_pe_q25, t.share_pe_q75, color=INK, alpha=0.10, lw=0)
                ax.plot(t.energy_keV, t.share_pe_q50, color=INK, lw=1.0, ls="--")
            ycr = lambda ec, p=prim: np.interp(np.log10(ec), np.log10(p.energy_keV), p.share_density)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_ylim(-0.1 if sub == "all" else -0.2, 1.05)
            ax.set_ylabel("share")
        else:
            piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values="ratio")
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=CR, alpha=0.2, lw=0)
            ax.plot(e, prim.ratio, color=CR, lw=1.2)
            ax.axhline(1, color=INK, lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(min(0.03, float(piv.min().min()) * 0.8), 5)
            ycr = lambda ec: 1.0
            ax.set_ylabel("variance ratio")
        for _, r in cr.iterrows():
            ax.plot(r.E_cross_keV, ycr(r.E_cross_keV), "o", ms=4, mfc="white", mec=INK, mew=0.9, zorder=5)
        main = cr[cr.direction == "comp->dens"]
        lab = cr if sub in ("top10_mu", "top1_mu") else main.tail(1)       # every crossing is labelled in (c), (d)
        for k, (_, r) in enumerate(lab.iterrows()):
            ec = r.E_cross_keV
            ax.annotate(f"{ec:.0f}", (ec, ycr(ec)), xytext=(3, 5 if k % 2 == 0 else -9), textcoords="offset points",
                        fontsize=6, color=INK)
        ax.set_xscale("log")
        ax.set_title(titles[sub], loc="left")
    # (e) variances
    ax = axes[2, 0]; _ax_style(ax)
    t = sh.sort_values("energy_keV")
    ax.plot(t.energy_keV, t.var_logmurho, color=C2, lw=1.2, label="var(log μ/ρ)")
    ax.plot(t.energy_keV, t.var_logmurho_no_pe, color=C2, lw=1.0, ls=":", label="… without photoelectric")
    ax.plot(t.energy_keV, t.var_pe_term, color=INK, lw=0.9, ls="--", label="var(−log(1 − f_pe)): pe term")
    ax.axhline(float(t.var_logrho.iloc[0]), color=C1, lw=1.2, label="var(log ρ)")
    ax.axvline(e_cross, color=MUTED, lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_ylim(5e-5, 4.0)
    ax.set_ylabel("variance (log₁₀ units²)")
    ax.set_title("(e) composition spread falls to the density spread", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    # (f) partial shares
    ax = axes[2, 1]; _ax_style(ax)
    for k, c, lab in (("pe", C2, "photoelectric"), ("incoh", C1, "Compton"), ("coh", C3, "coherent"),
                      ("pair", C4, "pair")):
        ax.fill_between(t.energy_keV, t[f"share_{k}_q25"], t[f"share_{k}_q75"], color=c, alpha=0.2, lw=0)
        ax.plot(t.energy_keV, t[f"share_{k}_q50"], color=c, lw=1.2, label=lab)
    ax.axvline(e_cross, color=MUTED, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of μ/ρ (R2 median, IQR)")
    ax.set_title("(f) interaction shares over the R2 map", loc="left")
    ax.legend(frameon=False, loc="center right", fontsize=6)
    # (g)/(h) exponents
    for ax, kind, ttl in ((axes[3, 0], "atom", "(g) per-atom Z-exponent n: σ ∝ Zⁿ"),
                          (axes[3, 1], "mass", "(h) per-mass Z-exponent n′: μ/ρ ∝ Zⁿ′")):
        _ax_style(ax)
        z = zx.sort_values("energy_keV")
        ax.plot(z.energy_keV, z[f"n_{kind}_pe_Kactive"], color=C2, lw=1.2, label="photoelectric (K-active)")
        ax.plot(z.energy_keV, z[f"n_{kind}_pe"], color=C2, lw=0.8, ls=":", label="photoelectric (all Z)")
        ax.plot(z.energy_keV, z[f"n_{kind}_incoh"], color=C1, lw=1.2, label="Compton")
        ax.plot(z.energy_keV, z[f"n_{kind}_coh"], color=C3, lw=1.0, label="coherent")
        pz = z[z.energy_keV >= 2000]
        ax.plot(pz.energy_keV, pz[f"n_{kind}_pair"], color=C4, lw=1.0, label="pair")
        ax.axvline(e_cross, color=MUTED, lw=0.6, ls=":")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_xscale("log")
        ax.set_ylabel("exponent")
        ax.set_title(ttl + f" (Z {Z_RANGE[0]}–{Z_RANGE[1]})", loc="left")
        ax.set_xlabel("photon energy (keV)")
    axes[3, 0].set_ylim(-0.3, 6.4); axes[3, 1].set_ylim(-0.8, 4.9)
    axes[3, 0].legend(frameon=False, fontsize=5.8, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.6)
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=C1, lw=1.2, label="density share var(log ρ)/var(log μ)"),
               Line2D([], [], color=C2, lw=1.2, label="composition share var(log μ/ρ)/var(log μ)"),
               Line2D([], [], color=C3, lw=1.2, label="covariance share 2cov/var(log μ) (can be < 0)"),
               Line2D([], [], color=CR, lw=1.2, label="(c, d) variance ratio var(log μ/ρ)/var(log ρ)"),
               Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK, label="crossing (keV)"),
               Line2D([], [], color=INK, lw=1.0, ls="--", label="(a) photoelectric share of μ/ρ, R2 median (band: IQR)"),
               Patch(color="#d9d8d3", alpha=0.5, lw=0, label="energy range of the diagnostic tube spectra")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=6.5)
    fig.text(0.0, 0.0, "Shares sum to 1; where the covariance share is negative a single share can exceed 1. "
             "Bands: rules R1/R2/R3 × coherent on/off. Population: eligible Materials Project entries (see the "
             "reweighting test, data/phase3/RW).", fontsize=6, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.015, 1, 0.955), h_pad=0.6, w_pad=0.8)
    out = common.savefig(fig, "fig1_variance_physics")
    plt.close(fig)
    return out


def fig_exponent_fits(ep):
    plt = common.style()
    fig, axes = plt.subplots(1, 2, figsize=(common.DOUBLE_COL_IN, 2.8))
    Z, A = ep["Z"], ep["A"]
    inr = (Z >= Z_RANGE[0]) & (Z <= Z_RANGE[1])
    for ax, part, ttl in ((axes[0], "pe", "photoelectric"), (axes[1], "incoh", "Compton (incoherent)")):
        _ax_style(ax)
        for ek, c in zip((60.0, 235.0, 1250.0), (C1, C2, C3)):
            j = _nearest_idx(ek)
            sig = ep[part][:, j] * A / N_A
            ax.plot(Z, sig, ".", ms=3, color=c)
            s = inr & (ep["K_edge_keV"] < energies()[j]) if part == "pe" else inr
            n, a, r2 = fit_power_law(Z[s], sig[s])
            zz = np.array([Z[s].min(), Z[s].max()], float)
            ax.plot(zz, np.exp(a) * zz ** n, color=c, lw=1.0, label=f"{energies()[j]:.0f} keV: n = {n:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Z"); ax.set_ylabel("σ per atom (b)")
        ax.set_title(f"{ttl}: per-atom σ vs Z" + (" (fit: K-active)" if part == "pe" else ""), loc="left")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = common.savefig(fig, "L_exponent_fits")
    plt.close(fig)
    return out


def fig_envelope_window(counts, win, surv, dax=None):
    plt = common.style()
    fig, axes = plt.subplots(1, 2, figsize=(common.DOUBLE_COL_IN, 2.9))
    ax = axes[0]; _ax_style(ax)
    c = counts[(counts.energy_keV >= 40) & (counts.energy_keV <= 75)]
    ax.step(c.energy_keV, c.n_R3, where="mid", color=C1, lw=1.2, label="R3 (all eligible)")
    ax.step(c.energy_keV, c.n_R2, where="mid", color=C2, lw=1.2, label="R2")
    if surv is not None and len(surv):
        s = surv[surv["T"] == T_LEVELS[0]].groupby("energy_keV").survives.sum()
        s2 = surv[(surv["T"] == T_LEVELS[0]) & surv.in_R2].groupby("energy_keV").survives.sum()
        ax.step(s.index, s.values, where="mid", color=C1, lw=0.8, ls="--", label="R3, fluorescence-corrected")
        ax.step(s2.index, s2.values, where="mid", color=C2, lw=0.8, ls="--", label="R2, fluorescence-corrected")
        if dax is not None:
            d95 = dax[dax.case == "class_p95"].set_index("row").delta
            keep = set(dax[(dax.case == "class_p95") & dax.in_R2 & ~dax.elemental_entry].row)
            sp = surv[(surv["T"] == T_LEVELS[0]) & surv.row.isin(keep)]
            sp = sp[(1 + sp.advantage_eff) / (1 + sp.row.map(d95)) - 1 > 0].groupby("energy_keV").size()
            ax.step(sp.index, sp.values, where="mid", color=C2, lw=0.8, ls=":",
                    label="R2, corrected, ρ at class p95 of δ")
    ax.set_xlabel("photon energy (keV)"); ax.set_ylabel("non-radioactive compounds\nabove the envelope")
    ax.set_title(f"(a) envelope exceedance window (corrected: T = {T_LEVELS[0]:g})", loc="left")
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]; _ax_style(ax)
    for lab, col in (("K edge + dense partner", C1), ("K edge + compact packing (light partner)", C2),
                     ("elemental entry (DFT ρ > experimental ρ)", C3), ("other", MUTED)):
        w = win[win.mechanism == lab]
        if len(w):
            ax.scatter(w.ratio_rho_vs_env, w.ratio_mu_rho_vs_env, s=8, color=col, lw=0, alpha=0.8,
                       label=f"{lab} ({len(w)})")
    xx = np.linspace(0.3, 2.25, 80)
    ax.plot(xx, 1 / xx, color=INK, lw=0.8, ls="--")
    ax.text(0.95, 1 / 0.95 * 1.02, "μ = envelope", fontsize=6, color=INK, ha="right", va="bottom")
    ax.set_xlabel("ρ / ρ(envelope element)"); ax.set_ylabel("(μ/ρ) / (μ/ρ)(envelope element)")
    ax.set_title("(b) mechanism at the energy of max ratio", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    out = common.savefig(fig, "L_envelope_window")
    plt.close(fig)
    return out


def fig_polymorph_spread(pt: pd.DataFrame, s: dict):
    """Dossier Figure 8 (single column): DFT vs measured polymorph density spread."""
    plt = common.style()
    fig, ax = plt.subplots(figsize=(common.SINGLE_COL_IN, 3.3))
    _ax_style(ax)
    floor = 1e-4
    x = np.clip(pt.spread_cod.to_numpy(), floor, None); y = np.clip(pt.spread_dft.to_numpy(), floor, None)
    ok = pt.pairs_discordant.to_numpy() == 0
    bad = pt.max_abs_delta.to_numpy() > 0.2
    ax.scatter(x[ok & ~bad], y[ok & ~bad], s=7, color=C1, lw=0, alpha=0.75,
               label=f"ordering preserved ({(ok & ~bad).sum()})")
    ax.scatter(x[~ok & ~bad], y[~ok & ~bad], s=9, color=C2, lw=0, alpha=0.9,
               label=f"ordering changed ({(~ok & ~bad).sum()})")
    ax.scatter(x[bad], y[bad], s=9, facecolor="none", edgecolor=INK2, lw=0.6,
               label=f"a member with |δ| > 0.2 ({bad.sum()})")
    lim = (floor * 0.8, max(2.0, float(max(x.max(), y.max())) * 1.3))
    ax.plot(lim, lim, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("measured spread (COD), ρ_max/ρ_min − 1  (< 10⁻⁴ drawn at 10⁻⁴)")
    ax.set_ylabel("DFT spread (MP), ρ_max/ρ_min − 1")
    p = s["primary"]
    ax.set_title(f"{p['n_formulas']} formulas; median DFT {100 * p['spread_dft']['median']:.1f} %, "
                 f"COD {100 * p['spread_cod']['median']:.1f} %", loc="left", fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    out = common.savefig(fig, "fig8_polymorph_spread")
    plt.close(fig)
    return out


# =============================================================================================
# run
# =============================================================================================
def _cached(name: str, fn, force: bool):
    p = OUT / f"{name}.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    df = fn()
    _save(df, name, csv=False)
    return df


def run_item1(force: bool = False, verbose: bool = True) -> dict:
    paths.ensure(OUT)
    dec = _cached("variance_decomposition", decomposition, force)
    cross = crossings(dec)
    cross.to_csv(OUT / "crossings.csv", index=False)
    cmp = compare_phase2(dec, cross)
    if verbose:
        print("phase 2 comparison:", cmp)
    ep = element_partials()
    zx = z_exponents(Z_RANGE, ep)
    _save(zx, "z_exponents")
    zs = pd.concat([z_exponents(zr, ep) for zr in Z_RANGES_SENS], ignore_index=True)
    _save(zs, "z_exponents_sensitivity", csv=False)
    shp = OUT / "partial_shares_R2.parquet"
    if shp.exists() and not force:
        sh = pd.read_parquet(shp)
        meta = json.loads((OUT / "partial_shares_meta.json").read_text(encoding="utf-8"))
    else:
        res = partial_shares("R2")
        sh = res["table"]
        _save(sh, "partial_shares_R2")
        res3 = partial_shares("R3")
        _save(res3["table"], "partial_shares_R3", csv=False)
        u = common.universe()
        meta = {"medoid_row": res["medoid_row"], "medoid_material_id": u.material_id.iloc[res["medoid_row"]],
                "medoid_formula": u.formula_pretty.iloc[res["medoid_row"]],
                "medoid_rms_log10": res["medoid_rms_log10"],
                "max_rel_mixture_vs_tableC_R2": res["max_rel_mixture_vs_tableC"],
                "max_rel_mixture_vs_tableC_R3": res3["max_rel_mixture_vs_tableC"],
                "n_check_rows": res["n_check_rows"]}
        common.write_json(OUT / "partial_shares_meta.json", meta)
        sh3 = res3["table"]
        meta["R3_pe_half_energy_keV"] = log_crossing(sh3.energy_keV.to_numpy(), sh3.share_pe_q50.to_numpy() - 0.5)
        common.write_json(OUT / "partial_shares_meta.json", meta)
    med = compound_shares(int(meta["medoid_row"]))
    _save(med, "medoid_compound_shares")
    diag_e = p2.diag_mean_energies()
    figs = fig_variance_physics(dec, cross, sh, zx, diag_e) + fig_exponent_fits(ep)
    return {"dec": dec, "cross": cross, "cmp": cmp, "zx": zx, "zs": zs, "sh": sh, "meta": meta, "med": med,
            "figs": figs, "diag_e": diag_e}


def summarize_item1(r: dict) -> dict:
    dec, cross, sh, zx, med = r["dec"], r["cross"], r["sh"], r["zx"], r["med"]
    e = sh.energy_keV.to_numpy()
    main = cross[(cross.subset == "all") & (cross.direction == "comp->dens")]
    lastc = main.groupby(["rule", "coherent"]).E_cross_keV.max()
    e_x = float(lastc[("R2", True)])
    at = lambda col, ek, df=sh: float(np.interp(np.log10(ek), np.log10(df.energy_keV), df[col]))
    out = {"phase2_comparison": r["cmp"],
           "crossing_R2_wcoh_keV": e_x,
           "crossing_range_keV": [float(lastc.min()), float(lastc.max())],
           "crossing_by_series": {f"{k[0]}|{'wcoh' if k[1] else 'wocoh'}": float(v) for k, v in lastc.items()},
           "var_logrho_R2": float(sh.var_logrho.iloc[0]),
           "pe_share_median_at_crossing": at("share_pe_q50", e_x),
           "pe_share_IQR_at_crossing": [at("share_pe_q25", e_x), at("share_pe_q75", e_x)],
           "pe_share_q05_q95_at_crossing": [at("share_pe_q05", e_x), at("share_pe_q95", e_x)],
           "incoh_share_median_at_crossing": at("share_incoh_q50", e_x),
           "coh_share_median_at_crossing": at("share_coh_q50", e_x),
           "energy_median_pe_share_half_keV": log_crossing(e, sh.share_pe_q50.to_numpy() - 0.5),
           "energy_pe_share_q75_half_keV": log_crossing(e, sh.share_pe_q75.to_numpy() - 0.5),
           "energy_pe_share_q25_half_keV": log_crossing(e, sh.share_pe_q25.to_numpy() - 0.5),
           "energy_median_pair_share_exceeds_pe_keV": log_crossing(e, (sh.share_pair_q50 - sh.share_pe_q50).to_numpy(), "up"),
           "energy_var_logmurho_equals_var_logrho_keV": log_crossing(e, (sh.var_logmurho - sh.var_logrho).to_numpy()),
           "energy_var_no_pe_equals_var_logrho_keV": log_crossing(e, (sh.var_logmurho_no_pe - sh.var_logrho).to_numpy(), "any"),
           "max_var_no_pe_over_var_logrho": float((sh.var_logmurho_no_pe / sh.var_logrho).max()),
           "energy_pe_term_equals_var_logrho_keV": log_crossing(e, (sh.var_pe_term - sh.var_logrho).to_numpy()),
           "split_identity_max_abs": float(np.abs(sh.var_logmurho_no_pe + sh.var_pe_term + sh.cov2_rest_pe_term
                                                  - sh.var_logmurho).max()),
           "var_log_pe_over_rest_range_30_1000keV": [float(sh.var_log_pe_over_rest[(e >= 30) & (e <= 1000)].min()),
                                                     float(sh.var_log_pe_over_rest[(e >= 30) & (e <= 1000)].max())],
           "frac_var_from_pe_term_at_crossing": float(np.interp(np.log10(e_x), np.log10(e),
                                                                (sh.var_pe_term / sh.var_logmurho).to_numpy())),
           "medoid": {k: r["meta"][k] for k in ("medoid_material_id", "medoid_formula", "medoid_rms_log10", "medoid_row")},
           "medoid_pe_share_half_keV": log_crossing(med.energy_keV.to_numpy(), med.share_pe.to_numpy() - 0.5),
           "medoid_pe_share_at_crossing": at("share_pe", e_x, med),
           "R3_pe_half_energy_keV": r["meta"].get("R3_pe_half_energy_keV"),
           "mixture_vs_tableC_max_rel": {"R2": r["meta"]["max_rel_mixture_vs_tableC_R2"],
                                         "R3": r["meta"]["max_rel_mixture_vs_tableC_R3"]},
           "figures": r["figs"]}
    # variance shares and pe shares at key energies
    keys = {}
    for ek in KEY_E:
        keys[f"{ek:g}"] = {"var_logmurho": at("var_logmurho", ek), "var_logmurho_no_pe": at("var_logmurho_no_pe", ek),
                           "var_pe_term": at("var_pe_term", ek), "cov2_rest_pe_term": at("cov2_rest_pe_term", ek),
                           "var_log_pe_over_rest": at("var_log_pe_over_rest", ek), "sd_share_pe": at("sd_share_pe", ek),
                           "pe_share_q25_q50_q75": [at("share_pe_q25", ek), at("share_pe_q50", ek), at("share_pe_q75", ek)],
                           "incoh_share_q50": at("share_incoh_q50", ek), "coh_share_q50": at("share_coh_q50", ek),
                           "pair_share_q50": at("share_pair_q50", ek)}
        z = zx
        for col in ("n_atom_pe_Kactive", "n_atom_pe_Kinactive", "n_atom_pe", "n_atom_incoh", "n_atom_coh",
                    "n_mass_pe_Kactive", "n_mass_pe", "n_mass_incoh", "n_mass_coh", "r2_atom_pe_Kactive",
                    "r2_atom_incoh", "n_el_pe_Kactive"):
            keys[f"{ek:g}"][col] = at(col, ek, z)
    out["at_energies"] = keys
    # exponent ranges between 100 keV and 1 MeV (all K edges below E for Z ≤ 83 above 90.5 keV)
    zr = zx[(zx.energy_keV >= 100) & (zx.energy_keV <= 1000)]
    out["exponent_ranges_100_1000keV"] = {c: [float(zr[c].min()), float(zr[c].max())] for c in
                                          ("n_atom_pe_Kactive", "n_mass_pe_Kactive", "n_atom_incoh", "n_mass_incoh",
                                           "n_atom_coh", "n_mass_coh")}
    zs = r["zs"]
    sens = {}
    for (lo, hi), g in zs.groupby(["Z_lo", "Z_hi"]):
        sens[f"Z{lo}-{hi}"] = {f"{ek:g}": {c: at(c, ek, g) for c in ("n_atom_pe_Kactive", "n_atom_incoh", "n_mass_pe_Kactive", "n_mass_incoh")}
                               for ek in (60.0, 235.0, 661.657, 1250.0)}
    out["exponent_sensitivity_Z_range"] = sens
    return out


def run_item2(force: bool = False, verbose: bool = True) -> dict:
    cp, hp = OUT / "envelope_counts.parquet", OUT / "envelope_hits.parquet"
    if cp.exists() and hp.exists() and not force:
        counts, hits = pd.read_parquet(cp), pd.read_parquet(hp)
    else:
        counts, hits = envelope_scan()
        _save(counts, "envelope_counts", csv=False); _save(hits, "envelope_hits", csv=False)
    win = envelope_window(hits)
    # cross-check against stage 2's list
    old = pd.read_csv(P2_A1 / "nonrad_above_nonrad_envelope.csv")
    mm = win.merge(old, on="material_id", how="outer", suffixes=("", "_p2"), indicator=True)
    both = mm[mm["_merge"] == "both"]
    p2chk = {"n_p3": int(len(win)), "n_p2": int(len(old)), "n_unmatched": int((mm["_merge"] != "both").sum()),
             "max_abs_diff_ratio": float(np.abs(both.max_ratio_mu_nonrad_env - both.max_ratio_mu_nonrad_env_p2).max()),
             "energy_at_max_identical": bool((both.energy_at_max_keV == both.energy_at_max_keV_p2).all())}
    if verbose:
        print("envelope list vs phase 2:", p2chk)
    surv = el = None
    sens: dict = {}
    fl_status = "not run"
    try:
        from shieldmap.analysis3 import fluorescence as fl

        def _current(pth):
            if force or not pth.exists():
                return None
            d = pd.read_parquet(pth)
            return d if "model_version" in d.columns and set(d.model_version.unique()) == {fl.MODEL_VERSION} else None

        sp = OUT / "envelope_survival.parquet"
        surv = _current(sp)
        el = _current(OUT / "envelope_best_element.parquet") if surv is not None else None
        if surv is None or el is None:
            surv, el = envelope_survival(hits)
            _save(surv, "envelope_survival"); _save(el, "envelope_best_element")
        fl_status = "ok"
        for old in OUT.glob("envelope_survival_sens_*.parquet"):            # sensitivities of the old model
            if old.stem.replace("envelope_survival_sens_", "") not in ENVELOPE_SENS:
                old.unlink()
        for tag, kw in ENVELOPE_SENS.items():
            pth = OUT / f"envelope_survival_sens_{tag}.parquet"
            ss = _current(pth)
            if ss is None:
                ss, _ = envelope_survival(hits, (T_LEVELS[0],), **kw)
                _save(ss, f"envelope_survival_sens_{tag}", csv=False)
            sens[tag] = ss
    except Exception as ex:  # fluorescence module unavailable: list and mechanism still written
        fl_status = f"failed: {type(ex).__name__}: {ex}"
        if verbose:
            print("fluorescence survival:", fl_status)
    if surv is not None:
        s1 = surv[surv["T"] == T_LEVELS[0]]
        best = s1.loc[s1.groupby("row").advantage_eff.idxmax(), ["row", "advantage_eff", "advantage_narrow", "energy_keV"]]
        win = win.merge(best.rename(columns={"advantage_eff": "max_advantage_eff_T0.1",
                                             "advantage_narrow": "max_advantage_narrow_T0.1",
                                             "energy_keV": "energy_at_max_advantage_eff_T0.1"}), on="row", how="left")
        win["survives_T0.1"] = win["max_advantage_eff_T0.1"] > 0
        s2 = surv[surv["T"] == T_LEVELS[1]]
        b2 = s2.groupby("row").advantage_eff.max().rename("max_advantage_eff_T0.01")
        win = win.merge(b2, left_on="row", right_index=True, how="left")
        win["survives_T0.01"] = win["max_advantage_eff_T0.01"] > 0
    _save(win, "envelope_window")
    dax = dcnt = None
    try:
        dax, dcnt = envelope_density_axis(win, surv, sens)
        _save(dax, "envelope_density_axis", csv=False); _save(dcnt, "envelope_density_counts")
    except Exception as ex:                                     # COD-module output unavailable
        if verbose:
            print("density axis:", type(ex).__name__, ex)
    figs = fig_envelope_window(counts, win, surv, dax)
    return {"counts": counts, "hits": hits, "win": win, "surv": surv, "el": el, "p2chk": p2chk, "sens": sens,
            "fl_status": fl_status, "figs": figs, "density_counts": dcnt}


def summarize_item2(r: dict) -> dict:
    counts, win, surv = r["counts"], r["win"], r["surv"]
    pos3 = counts[counts.n_R3 > 0]; pos2 = counts[counts.n_R2 > 0]
    w2 = win[win.in_R2]
    out = {"phase2_check": r["p2chk"], "fluorescence_status": r["fl_status"],
           "window_R3_keV": [float(pos3.energy_keV.min()), float(pos3.energy_keV.max())] if len(pos3) else None,
           "window_R2_keV": [float(pos2.energy_keV.min()), float(pos2.energy_keV.max())] if len(pos2) else None,
           "n_entries_R3": int(len(win)), "n_compounds_R2": int(len(w2)), "n_R2_B4": int(w2.in_B4.sum()),
           "n_multi_element_R3": int((win.partner != "").sum()), "n_multi_element_R2": int((w2.partner != "").sum()),
           "n_multi_element_R2_icsd": int(((w2.partner != "") & ~w2.theoretical).sum()),
           "n_multi_element_R2_icsd_ehull0": int(((w2.partner != "") & ~w2.theoretical & (w2.energy_above_hull <= 0)).sum()),
           "max_count_R2": int(counts.n_R2.max()), "energy_max_count_R2": float(counts.loc[counts.n_R2.idxmax(), "energy_keV"]),
           "max_count_R3": int(counts.n_R3.max()), "energy_max_count_R3": float(counts.loc[counts.n_R3.idxmax(), "energy_keV"]),
           "envelope_elements_in_window": counts[(counts.energy_keV >= 45) & (counts.energy_keV <= 70)]
               .groupby("env_element").energy_keV.agg(["min", "max"]).reset_index().to_dict("records"),
           "mechanism_counts_R3": win.mechanism.value_counts().to_dict(),
           "mechanism_counts_R2": w2.mechanism.value_counts().to_dict(),
           "edge_element_counts_R2": w2.edge_element.value_counts().to_dict(),
           "partner_counts_R2_top": w2.partner.value_counts().head(12).to_dict(),
           "gap_E_minus_K_keV_R3": _dist(win.gap_E_minus_K_keV),
           "ratio_mu_rho_vs_env_R3": _dist(win.ratio_mu_rho_vs_env), "ratio_rho_vs_env_R3": _dist(win.ratio_rho_vs_env),
           "ratio_rho_vs_edge_solid_R3": _dist(win.ratio_rho_vs_edge_solid),
           "max_ratio_R3": _dist(win.max_ratio_mu_nonrad_env), "max_ratio_R2": _dist(w2.max_ratio_mu_nonrad_env),
           "top_R2": w2.head(10)[["material_id", "formula_pretty", "density", "energy_above_hull", "theoretical",
                                  "max_ratio_mu_nonrad_env", "energy_at_max_keV", "edge_element", "partner",
                                  "env_element", "mechanism"]].to_dict("records")}
    if surv is not None:
        for T in T_LEVELS:
            s = surv[surv["T"] == T]
            byrow = s.groupby("row").agg(adv=("advantage_eff", "max"), advn=("advantage_narrow", "max"),
                                         in_R2=("in_R2", "first"))
            surv_rows = byrow[byrow.adv > 0]
            multi = set(win.loc[win.partner != "", "row"])
            mi = byrow.index.isin(multi)
            icsd = set(win.loc[(win.partner != "") & ~win.theoretical, "row"])
            ii = byrow.index.isin(icsd)
            out[f"survival_T{T:g}_multi_element"] = {
                "n_entries": int(mi.sum()), "n_surviving": int((byrow.adv[mi] > 0).sum()),
                "n_R2": int(byrow.in_R2[mi].sum()), "n_R2_surviving": int(((byrow.adv > 0) & byrow.in_R2)[mi].sum()),
                "n_R2_icsd": int((byrow.in_R2 & ii).sum()), "n_R2_icsd_surviving": int(((byrow.adv > 0) & byrow.in_R2 & ii).sum()),
                "advantage_eff_surviving": _dist(byrow.adv[mi & (byrow.adv > 0)])}
            out[f"survival_T{T:g}"] = {
                "n_entry_energy_pairs": int(len(s)), "n_pairs_surviving": int(s.survives.sum()),
                "n_entries": int(len(byrow)), "n_entries_surviving": int(len(surv_rows)),
                "n_R2": int(byrow.in_R2.sum()), "n_R2_surviving": int(surv_rows.in_R2.sum()),
                "advantage_narrow_max_per_entry": _dist(byrow.advn), "advantage_eff_max_per_entry": _dist(byrow.adv),
                "advantage_eff_surviving": _dist(surv_rows.adv),
                "advantage_loss_pp": _dist(100 * (byrow.advn - byrow.adv)),
                "energies_with_survivor_keV": sorted(s[s.survives].energy_keV.unique().tolist()),
                "fl_fraction_at_q_eff": _dist(s.fl_fraction), "eff_increase": _dist(s.eff_increase)}
            el = r["el"][r["el"]["T"] == T]
            out[f"best_element_T{T:g}"] = el[["energy_keV", "best_narrow", "best_eff", "best_eff_fl_fraction"]].to_dict("records")
        top = surv[(surv["T"] == T_LEVELS[0]) & surv.in_R2].sort_values("advantage_eff", ascending=False)
        top = top.drop_duplicates("row").head(10)
        out["top_R2_survivors_T0.1"] = top[["material_id", "formula_pretty", "energy_keV", "advantage_narrow",
                                            "advantage_eff", "fl_fraction", "best_el_eff"]].to_dict("records")
    if r.get("density_counts") is not None:
        dc = r["density_counts"]
        out["density_axis_counts"] = dc.to_dict("records")
        mr2 = dc[(dc.population == "multi_element") & (dc.scope == "R2")].set_index("case")
        rng = {}
        for c in mr2.columns:
            if c.startswith(("exceeds_", "survives_")):
                v = mr2[c]
                rng[c] = {"min": int(v.min()), "max": int(v.max()), "by_case": {k: int(x) for k, x in v.items()}}
        out["density_axis_ranges_R2_multi_element"] = rng
        out["density_cases"] = list(DENSITY_CASES)
    out["n_elemental_entries"] = {"R3": int((win.partner == "").sum()), "R2": int((w2.partner == "").sum())}
    multi = set(win.loc[win.partner != "", "row"])
    for tag, ss in r.get("sens", {}).items():
        b = ss.groupby("row").agg(adv=("advantage_eff", "max"), in_R2=("in_R2", "first"))
        m = b.index.isin(multi)
        out[f"survival_T{T_LEVELS[0]:g}_sens_{tag}"] = {
            "n_entries_surviving": int((b.adv > 0).sum()), "n_multi_element_surviving": int((b.adv[m] > 0).sum()),
            "n_multi_element_R2_surviving": int(((b.adv > 0) & b.in_R2)[m].sum()),
            "advantage_eff_surviving": _dist(b.adv[b.adv > 0])}
    out["figures"] = r["figs"]
    return out


def run_item3(force: bool = False, verbose: bool = True) -> dict:
    d = common.P3 / "D"
    if not (d / "summary.json").exists() or not (d / "matches.parquet").exists():
        return {"status": "COD Step 3 output not available"}
    dsum = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    m = pd.read_parquet(d / "matches.parquet")
    pt, s = polymorph_cod(m)
    _save(pt, "polymorph_cod")
    figs = fig_polymorph_spread(pt, s) if len(pt) else []
    s["n_matches"] = int(len(m)); s["n_formulas_in_matches"] = int(m.formula_reduced.nunique())
    s["n_formulas_ge2_mp_entries"] = int((m.groupby("formula_reduced").size() >= 2).sum())
    s["figures"] = figs
    s["status"] = "ok"
    s["D_summary_keys"] = sorted(dsum)[:30]
    # Stage 2 comparison: DFT spread of all eligible polymorphs for the same formulas
    pg = pd.read_parquet(P2_A1 / "polymorph_groups.parquet")
    j = pt.merge(pg[["formula_reduced", "rho_spread", "n_poly"]], on="formula_reduced", how="left", suffixes=("", "_p2"))
    s["phase2_all_polymorph_spread_same_formulas"] = _dist(j.rho_spread)
    s["n_formulas_also_in_phase2_ge3"] = int(j.rho_spread.notna().sum())
    return {"pt": pt, "summary": s}


def run(force: bool = False, verbose: bool = True, items=(1, 2, 3)) -> dict:
    paths.ensure(OUT)
    sp = OUT / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    summary["definitions"] = {
        "Z_range_primary": Z_RANGE, "Z_ranges_sensitivity": Z_RANGES_SENS, "T_levels_survival": T_LEVELS,
        "median_compound": "pointwise median (and IQR) of each partial share over the R2 universe; the medoid "
                           "compound (closest log μ/ρ curve to the pointwise median, RMS over the grid) is "
                           "reported as an illustration",
        "envelope": "max μ over non-radioactive elemental solids (Phase 2 nonradioactive_envelope, with coherent)"}
    if 1 in items:
        r1 = run_item1(force, verbose)
        summary["item1"] = summarize_item1(r1)
        common.write_json(sp, summary)
    if 2 in items:
        r2 = run_item2(force, verbose)
        summary["item2"] = summarize_item2(r2)
        common.write_json(sp, summary)
    if 3 in items:
        r3 = run_item3(force, verbose)
        summary["item3"] = r3["summary"] if "summary" in r3 else r3
        common.write_json(sp, summary)
    return summary


if __name__ == "__main__":
    import sys
    its = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (1, 2, 3)
    run(force="--force" in sys.argv, items=its)
