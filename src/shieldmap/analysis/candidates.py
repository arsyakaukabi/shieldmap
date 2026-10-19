"""A3: candidates and method battery (RQ3, supporting).

Everything here is built on ``analysis.core``; frontier, ranks, constraints, bundles, Pareto layers,
TOPSIS, variants and the universe are imported from there, never re-implemented.

Primary settings: representative rule R2; primary variants (air kerma, with coherent); primary basis
(DIAG areal mass, CS137/CO60 thickness); bundles B4 (main), B5 and B6 (variants).

Method battery (ranks within the bundle's R2 set, one ranking per primary variant):
  a requirement            rank by requirement q (lower is better)
  b  Pareto + requirement   2-D non-dominated layers on (q ↓, log10 min crustal abundance ↑); layers
                            1–3 from core.pareto_layers, every deeper compound shares layer 4; sort by
                            (layer, q, row)
  c  TOPSIS (weight-swept)  core.topsis on {q (lower is better), log10 min abundance shifted to ≥ 0 within the
                            set (benefit), band gap (benefit)} for each of the 66 weight vectors of
                            core.simplex_grid(3, 0.1); a compound's per-weight rank is its rank by
                            closeness; the method orders compounds by the MEDIAN of their 66 ranks
                            (ties by q, then row). Weighted scores appear only here.
  d  Pb-shape distance      DIAG only: cosine distance between μ/ρ(E) and Pb's μ/ρ(E) over the grid
                            nodes 20–150 keV (= cosine distance of log T(E) at any fixed areal mass);
                            ties by q, then row
  e  rule robustness        the formula's WORST formula-level requirement rank across R1, R2, R3
                            (R3: the formula's best entry); a formula absent from a rule's bundle set
                            counts as rank ∞; sort by (worst rank, R2 rank)
Each method's top-20 is the first 20 compounds of its ordering.

Consensus (formula level): in the top-20 of ≥ 4 of the 5 methods (DIAG) or ≥ 3 of the 4 methods
(radionuclides, where d is undefined), in ≥ 80 % of the scenario's primary variants.

``run()`` regenerates data/phase2/A3/, reports/phase2/figures/A3_*.png and reports/phase2/A3_report.md.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import core

OUT = core.P2 / "A3"
FIG = core.FIGURES
REPORT = core.REPORTS / "A3_report.md"

SCENARIOS = ("DIAG", "CS137", "CO60")
A3_BUNDLES = ("B4", "B5", "B6")
TOPN = 20
OVERLAP_N = 50
CONSENSUS_FRAC = 0.8
PB_ROW_MATERIAL = "mp-20483"                       # MP Pb, R2 representative (Fm-3m, ICSD-matched)
PB_BAND_KEV = (20.0, 150.0)
METHODS = ("a", "b", "c", "d", "e")
METHOD_NAMES = {"a": "requirement", "b": "Pareto layer + requirement", "c": "TOPSIS (weight-swept, median rank)",
                "d": "Pb-shape cosine distance (20–150 keV)", "e": "worst rank over R1/R2/R3"}

# displayed variant per scenario (figures and single-variant tables)
DISPLAY = {"DIAG": dict(quality="RQR6", anode_angle=12.0, target_value=0.35),
           "CS137": dict(T_target=0.01), "CO60": dict(T_target=0.01)}

# "textbook shielding material?" rule
TEXTBOOK = {"W", "Pb", "Bi", "Ba", "Gd", "Sn", "Sb", "Ta", "Fe", "Cu"}
LIGHT = {"H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca"}
CONCRETE = {"H", "C", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Fe"}

SANITY = ["W", "WO3", "WC", "Bi", "Bi2O3", "BaSO4", "Gd2O3", "Sn", "Sb2O3", "PbO", "Fe", "Cu",
          "CaCO3", "SiO2"]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def textbook(elements) -> str:
    els = set(elements)
    if (els <= TEXTBOOK | LIGHT and els & TEXTBOOK) or els <= CONCRETE:
        return "yes"
    return "no"


def lex_rank(idx: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """1-based positions of ``idx`` (row indices) sorted by keys (primary first), ties by row."""
    order = np.lexsort((idx,) + tuple(k for k in reversed(keys)))
    r = np.empty(idx.size, dtype=np.int64)
    r[order] = np.arange(1, idx.size + 1)
    return r


@lru_cache(maxsize=1)
def attrs() -> pd.DataFrame:
    u = core.universe().copy()
    u["fcode"] = pd.factorize(u["formula_reduced"])[0]
    with np.errstate(divide="ignore"):
        u["log10_abund"] = np.log10(u["min_crustal_abundance_ppm"].astype(float))
    return u


def scen_matrix(scenario: str) -> tuple[np.ndarray, pd.DataFrame]:
    """(N × V) float64 requirement matrix for the scenario's primary variants and the variant table."""
    vs = core.scenario_variants(scenario).reset_index(drop=True)
    q = np.asarray(core.profiles(core.default_basis(scenario))[:, vs.variant_id.to_numpy()], dtype=np.float64)
    return q, vs


def display_col(scenario: str, vs: pd.DataFrame) -> int:
    m = np.ones(len(vs), dtype=bool)
    for k, val in DISPLAY[scenario].items():
        m &= (vs[k] == val).to_numpy() if isinstance(val, str) else np.isclose(vs[k].astype(float), val)
    return int(np.flatnonzero(m)[0])


def variant_label(row: pd.Series) -> str:
    if row.scenario == "DIAG":
        return f"{row.quality} {int(row.anode_angle)}° {row.target_value:.2f} mm Pb"
    return f"T={row.T_target:g}"


@lru_cache(maxsize=1)
def pb_cosine_distance() -> tuple[np.ndarray, dict]:
    """Cosine distance between each row's μ/ρ(E) and Pb's over grid nodes 20–150 keV (Table C)."""
    u = attrs()
    E = np.load(paths.PROCESSED / "energy_keV.npy")
    sel = np.flatnonzero((E >= PB_BAND_KEV[0]) & (E <= PB_BAND_KEV[1]))
    mu = np.load(paths.PROCESSED / "mu_rho_wcoh.npy", mmap_mode="r")
    block = np.asarray(mu[:, sel.min():sel.max() + 1], dtype=np.float64)
    pb_row = int(np.flatnonzero(u.material_id.to_numpy() == PB_ROW_MATERIAL)[0])
    ref = block[pb_row]
    xs = np.load(paths.INTERIM / "elements" / "xsec_grid.npz")
    zi = int(np.flatnonzero(xs["Z"] == 82)[0])
    tab = xs["total_wcoh"][zi][sel]
    dev = float(np.max(np.abs(ref / tab - 1)))
    cos = block @ ref / (np.linalg.norm(block, axis=1) * np.linalg.norm(ref))
    info = {"pb_row": pb_row, "pb_material_id": PB_ROW_MATERIAL, "n_nodes": int(sel.size),
            "E_min": float(E[sel].min()), "E_max": float(E[sel].max()),
            "max_rel_dev_vs_xsec_grid": dev}
    return 1.0 - cos, info


def formula_ranks(q: np.ndarray, rule: str, bundle: np.ndarray) -> np.ndarray:
    """Formula-level requirement ranks (n_formulas × V) under a rule within a bundle (∞ if absent).

    R1/R2 hold one entry per formula; under R3 each formula is represented by its best entry
    in each variant. Ranks come from core.ranks on the per-formula requirement vector.
    """
    u = attrs()
    nf = int(u.fcode.max()) + 1
    m = core.base_mask(rule) & bundle
    rows = np.flatnonzero(m)
    codes = u.fcode.to_numpy()[rows]
    V = q.shape[1]
    best = np.full((nf, V), np.inf)
    np.minimum.at(best, codes, q[rows])
    present = np.isfinite(best[:, 0])
    out = np.full((nf, V), np.inf)
    for v in range(V):
        r = core.ranks(best[:, v], present)
        out[present, v] = r[present]
    return out


# ---------------------------------------------------------------------------------------------
# battery
# ---------------------------------------------------------------------------------------------
def battery(scenario: str, bundle: str, q: np.ndarray, vs: pd.DataFrame, weights: np.ndarray) -> dict:
    u = attrs()
    bmask = core.bundle_mask(bundle)
    S = np.flatnonzero(core.base_mask("R2") & bmask)
    codes = u.fcode.to_numpy()[S]
    la = u.log10_abund.to_numpy()[S]
    gap = u.band_gap.to_numpy(dtype=float)[S]
    la_shift = la - np.nanmin(la[np.isfinite(la)])
    la_shift = np.where(np.isfinite(la_shift), la_shift, 0.0)
    rule_rk = {r: formula_ranks(q, r, bmask)[codes] for r in core.RULES}
    worst = np.maximum.reduce([rule_rk[r] for r in core.RULES])
    dist = pb_cosine_distance()[0][S] if scenario == "DIAG" else None
    methods = [m for m in METHODS if not (m == "d" and scenario != "DIAG")]
    need = 4 if scenario == "DIAG" else 3

    per_variant, topsis_rows, pareto_rows = [], [], []
    n_in_top = np.zeros((S.size, len(vs)), dtype=np.int8)
    in_top = {m: np.zeros((S.size, len(vs)), dtype=bool) for m in methods}
    for v in range(len(vs)):
        qv = q[S, v]
        rk = {}
        rk["a"] = lex_rank(S, qv)
        lay_ab = core.pareto_layers(qv, la)
        lay_gap = core.pareto_layers(qv, gap)
        rk["b"] = lex_rank(S, np.where(lay_ab == 0, 4, lay_ab), qv)
        M = np.column_stack([qv, la_shift, gap])
        benefit = np.array([False, True, True])
        wr = np.empty((weights.shape[0], S.size), dtype=np.int32)
        for i, w in enumerate(weights):
            wr[i] = lex_rank(S, -core.topsis(M, w, benefit))
        med = np.median(wr, axis=0)
        rk["c"] = lex_rank(S, med, qv)
        if dist is not None:
            rk["d"] = lex_rank(S, dist, qv)
        rk["e"] = lex_rank(S, worst[:, v], rule_rk["R2"][:, v])
        top = np.zeros(S.size, dtype=np.int8)
        for m in methods:
            in_top[m][:, v] = rk[m] <= TOPN
            top += in_top[m][:, v]
        n_in_top[:, v] = top
        keep = np.zeros(S.size, dtype=bool)
        for m in methods:
            keep |= rk[m] <= OVERLAP_N
        df = pd.DataFrame({"scenario": scenario, "bundle": bundle, "variant_id": int(vs.variant_id[v]),
                           "row": S[keep], "q": qv[keep], "pareto_layer_abund": lay_ab[keep],
                           "pareto_layer_gap": lay_gap[keep], "topsis_median_rank": med[keep],
                           "worst_rule_rank": worst[keep, v], "n_methods_top20": top[keep]})
        for r in core.RULES:
            df[f"rank_{r}"] = rule_rk[r][keep, v]
        for m in methods:
            df[f"rank_{m}"] = rk[m][keep]
        per_variant.append(df)
        tk = keep
        t = pd.DataFrame(wr[:, tk].T, columns=[f"w{i:02d}" for i in range(weights.shape[0])])
        t.insert(0, "row", S[tk]); t.insert(0, "variant_id", int(vs.variant_id[v]))
        t.insert(0, "bundle", bundle); t.insert(0, "scenario", scenario)
        topsis_rows.append(t)
        for crit, lay in (("abundance", lay_ab), ("band_gap", lay_gap)):
            sel = lay > 0
            pareto_rows.append(pd.DataFrame({"scenario": scenario, "bundle": bundle,
                                             "variant_id": int(vs.variant_id[v]), "criterion": crit,
                                             "row": S[sel], "layer": lay[sel], "q": qv[sel]}))
    passing = (n_in_top >= need)
    frac = passing.mean(axis=1)
    anytop = n_in_top.max(axis=1) > 0
    summ = pd.DataFrame({"scenario": scenario, "bundle": bundle, "row": S[anytop],
                         "frac_variants_consensus": frac[anytop],
                         "max_methods_top20": n_in_top[anytop].max(axis=1),
                         "median_methods_top20": np.median(n_in_top[anytop], axis=1),
                         "methods_available": len(methods), "methods_needed": need,
                         "frac_variants_ge3_methods": (n_in_top[anytop] >= 3).mean(axis=1)})
    for m in methods:
        summ[f"frac_variants_top20_{m}"] = in_top[m][anytop].mean(axis=1)
    summ["status"] = np.where(summ.frac_variants_consensus >= CONSENSUS_FRAC, "consensus", "method-dependent")
    return {"ranks": pd.concat(per_variant), "topsis": pd.concat(topsis_rows),
            "pareto": pd.concat(pareto_rows), "summary": summ, "set_size": int(S.size)}


# ---------------------------------------------------------------------------------------------
# rule overlap and sanity ranks
# ---------------------------------------------------------------------------------------------
def rule_overlap(scenario: str, q: np.ndarray, vs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bundle in ("B0", "B4"):
        bm = core.bundle_mask(bundle)
        fr = {r: formula_ranks(q, r, bm) for r in core.RULES}
        for v in range(len(vs)):
            sets = {r: set(np.flatnonzero(fr[r][:, v] <= OVERLAP_N)) for r in core.RULES}
            s1, s2, s3 = sets["R1"], sets["R2"], sets["R3"]
            rows.append({"scenario": scenario, "bundle": bundle, "variant_id": int(vs.variant_id[v]),
                         "n_R1": len(s1), "n_R2": len(s2), "n_R3": len(s3),
                         "change_R1_R2": 1 - len(s1 & s2) / OVERLAP_N,
                         "change_R2_R3": 1 - len(s2 & s3) / OVERLAP_N,
                         "change_R1_R3": 1 - len(s1 & s3) / OVERLAP_N,
                         "all3": len(s1 & s2 & s3), "R1R2_only": len((s1 & s2) - s3),
                         "R1R3_only": len((s1 & s3) - s2), "R2R3_only": len((s2 & s3) - s1),
                         "R1_only": len(s1 - s2 - s3), "R2_only": len(s2 - s1 - s3),
                         "R3_only": len(s3 - s1 - s2)})
    return pd.DataFrame(rows)


def rule_overlap_members(scenario: str, q: np.ndarray, vs: pd.DataFrame, v: int) -> pd.DataFrame:
    """Formulas that are in some rule's top-50 but not all, at one variant (for the report)."""
    u = attrs()
    first = u.drop_duplicates("fcode").set_index("fcode")
    rows = []
    for bundle in ("B0", "B4"):
        bm = core.bundle_mask(bundle)
        fr = {r: formula_ranks(q, r, bm)[:, v] for r in core.RULES}
        sets = {r: set(np.flatnonzero(fr[r] <= OVERLAP_N)) for r in core.RULES}
        allf = sets["R1"] | sets["R2"] | sets["R3"]
        for f in sorted(allf - (sets["R1"] & sets["R2"] & sets["R3"])):
            rows.append({"scenario": scenario, "bundle": bundle, "variant_id": int(vs.variant_id[v]),
                         "formula": first.loc[f, "formula_reduced"],
                         **{f"rank_{r}": fr[r][f] for r in core.RULES}})
    return pd.DataFrame(rows)


def sanity_formulas() -> list[str]:
    u = attrs()
    r2 = u[u.in_R2]
    tw = r2[r2.formula_reduced.str.fullmatch(r"[A-Z][a-z]?WO4") & ~r2.theoretical]
    return SANITY + sorted(tw.formula_reduced.unique())


def sanity_ranks(scenario: str, q: np.ndarray, vs: pd.DataFrame) -> pd.DataFrame:
    u = attrs()
    vd = display_col(scenario, vs)
    out = []
    for bundle in ("B0", "B4"):
        m = core.base_mask("R2") & core.bundle_mask(bundle)
        front = core.frontier(q, m, "best")
        rk = np.column_stack([core.ranks(q[:, v], m) for v in range(q.shape[1])])
        for f in sanity_formulas():
            hit = np.flatnonzero(u.in_R2.to_numpy() & (u.formula_reduced.to_numpy() == f))
            if hit.size == 0:
                out.append({"scenario": scenario, "bundle": bundle, "formula": f, "in_map": False})
                continue
            i = int(hit[0])
            rec = {"scenario": scenario, "bundle": bundle, "formula": f, "in_map": True,
                   "material_id": u.material_id[i], "in_bundle": bool(m[i])}
            if not m[i]:
                fails = [c for c, p in core.BUNDLES[bundle] if not core.constraint_mask(c, p)[i]]
                rec["fails"] = "+".join(fails)
            else:
                r = rk[i]
                ratio = q[i] / front
                rec.update({"rank_display": int(r[vd]), "rank_min": int(r.min()), "rank_max": int(r.max()),
                            "set_size": int(m.sum()), "q_display": float(q[i, vd]),
                            "ratio_display": float(ratio[vd]), "ratio_min": float(ratio.min()),
                            "ratio_max": float(ratio.max())})
            out.append(rec)
    return pd.DataFrame(out)


LANTHANIDES = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
SCARCE_PPM = 0.1


def frontier_diagnostics(qmats: dict, vss: dict) -> pd.DataFrame:
    """Per scenario × bundle × variant: leader, flatness of the top-20, and composition of the top-20."""
    u = attrs()
    els = u["elements"].to_numpy()
    rows = []
    for sc in SCENARIOS:
        q, vs = qmats[sc], vss[sc]
        for b in ("B4", "B5", "B6"):
            m = core.base_mask("R2") & core.bundle_mask(b)
            f4 = core.frontier(q, core.base_mask("R2") & core.bundle_mask("B4"), "best")
            for v in range(len(vs)):
                t = core.top_k(q[:, v], m, TOPN)
                rows.append({"scenario": sc, "bundle": b, "variant_id": int(vs.variant_id[v]),
                             "variant": variant_label(vs.iloc[v]), "leader": u.formula_pretty[t[0]],
                             "q_best": q[t[0], v], "q_20th": q[t[-1], v],
                             "spread_top20": q[t[-1], v] / q[t[0], v] - 1,
                             "frontier_vs_B4": q[t[0], v] / f4[v] - 1,
                             "share_lanthanide": float(np.mean([bool(set(els[i]) & LANTHANIDES) for i in t])),
                             "share_scarce": float(np.mean(u.min_crustal_abundance_ppm.to_numpy()[t] < SCARCE_PPM)),
                             "share_textbook": float(np.mean([textbook(els[i]) == "yes" for i in t])),
                             "top5": ", ".join(u.formula_pretty[i] for i in t[:5])})
    return pd.DataFrame(rows)


def topsis_log_check(qmats: dict, vss: dict, weights: np.ndarray) -> pd.DataFrame:
    """Sensitivity of method c to the requirement transform: raw q (primary) vs log10(q / q_min)."""
    u = attrs()
    rows = []
    for sc in SCENARIOS:
        q, vs = qmats[sc], vss[sc]
        vd = display_col(sc, vs)
        for b in A3_BUNDLES:
            S = np.flatnonzero(core.base_mask("R2") & core.bundle_mask(b))
            qv = q[S, vd]
            la = u.log10_abund.to_numpy()[S]; la = la - la.min()
            gap = u.band_gap.to_numpy(dtype=float)[S]
            tops = {}
            for name, qq in (("raw", qv), ("log", np.log10(qv / qv.min()))):
                M = np.column_stack([qq, la, gap])
                wr = np.array([lex_rank(S, -core.topsis(M, w, np.array([False, True, True]))) for w in weights])
                r = lex_rank(S, np.median(wr, axis=0), qv)
                tops[name] = set(S[r <= TOPN])
            rows.append({"scenario": sc, "bundle": b, "variant_id": int(vs.variant_id[vd]),
                         "top20_overlap_raw_vs_log": len(tops["raw"] & tops["log"]),
                         "top20_raw": ", ".join(sorted(u.formula_pretty[i] for i in tops["raw"]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# consensus table
# ---------------------------------------------------------------------------------------------
def consensus_table(summary: pd.DataFrame, ranks: pd.DataFrame, qmats: dict, vss: dict) -> pd.DataFrame:
    u = attrs()
    rows = []
    for _, s in summary.iterrows():
        i = int(s.row)
        q, vs = qmats[s.scenario], vss[s.scenario]
        vd = display_col(s.scenario, vs)
        r3 = u[(u.fcode == u.fcode[i]) & core.base_mask("R3") & core.bundle_mask(s.bundle)]
        sub = ranks[(ranks.scenario == s.scenario) & (ranks.bundle == s.bundle) & (ranks.row == i)]
        rec = {"scenario": s.scenario, "bundle": s.bundle, "formula": u.formula_pretty[i],
               "formula_reduced": u.formula_reduced[i], "material_id_R2": u.material_id[i],
               "material_ids_R3_in_bundle": ";".join(r3.material_id),
               "status": s.status, "frac_variants_consensus": s.frac_variants_consensus,
               "median_methods_top20": s.median_methods_top20,
               "frac_variants_ge3_methods": s.frac_variants_ge3_methods,
               **{c: s[c] for c in s.index if c.startswith("frac_variants_top20_")},
               "methods": f"{int(s.methods_needed)}/{int(s.methods_available)}",
               "q_display": float(q[i, vd]), "q_min": float(q[i].min()), "q_max": float(q[i].max()),
               "density": float(u.density[i]), "e_hull": float(u.energy_above_hull[i]),
               "theoretical": bool(u.theoretical[i]), "band_gap": float(u.band_gap[i]),
               "min_abund_ppm": float(u.min_crustal_abundance_ppm[i]),
               "elements": "-".join(u.elements[i]), "textbook": textbook(u.elements[i])}
        for m in METHODS:
            c = f"rank_{m}"
            if c in sub and sub[c].notna().any():
                rec[f"median_rank_{m}"] = float(np.median(sub[c]))
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values(["scenario", "bundle", "status", "frac_variants_consensus", "q_display"],
                          ascending=[True, True, True, False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
LAYER_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}   # dataviz reference slots 1–3 (all-pairs safe)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _log_x(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_pareto(qmats, vss, cons, pareto, criterion="abundance", name="A3_pareto_B4.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    u = attrs()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURF)
    S = np.flatnonzero(core.base_mask("R2") & core.bundle_mask("B4"))
    yall = u.log10_abund.to_numpy() if criterion == "abundance" else u.band_gap.to_numpy(dtype=float)
    for ax, sc in zip(axes, SCENARIOS):
        _style(ax)
        q, vs = qmats[sc], vss[sc]
        vd = display_col(sc, vs)
        vid = int(vs.variant_id[vd])
        ax.scatter(q[S, vd], yall[S], s=3, c="#b9b8b2", lw=0, alpha=0.5, rasterized=True, zorder=1)
        p = pareto[(pareto.scenario == sc) & (pareto.bundle == "B4") & (pareto.variant_id == vid)
                   & (pareto.criterion == criterion)]
        counts = {}
        for L in (3, 2, 1):
            pl = p[p.layer == L].sort_values("q")
            counts[L] = len(pl)
            ax.step(pl.q, yall[pl.row], where="post", color=LAYER_COLORS[L], lw=1.0, alpha=0.6, zorder=2)
            ax.scatter(pl.q, yall[pl.row], s=22, c=LAYER_COLORS[L], edgecolors=SURF, linewidths=0.8, zorder=3)
        c = cons[(cons.scenario == sc) & (cons.bundle == "B4") & (cons.status == "consensus")]
        note = f"layers 1/2/3: n = {counts[1]}/{counts[2]}/{counts[3]}"
        if len(c):
            ci = [int(np.flatnonzero(u.material_id.to_numpy() == m)[0]) for m in c.material_id_R2]
            ax.scatter(q[ci, vd], yall[ci], s=70, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
            order = np.argsort(q[ci, vd])
            for k, j in enumerate(order[:10]):
                i = ci[j]
                ax.annotate(u.formula_pretty[i], (q[i, vd], yall[i]), xytext=(45, 80 - 15 * k),
                            textcoords="offset points", fontsize=7.5, color=INK, va="center",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
            note += f"; consensus candidates circled (n={len(ci)})"
        else:
            lead = int(p[p.layer == 1].sort_values("q").row.iloc[0])
            ax.annotate(f"{u.formula_pretty[lead]} (lowest requirement; not consensus)",
                        (q[lead, vd], yall[lead]), xytext=(30, -30), textcoords="offset points",
                        fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
            note += "; no consensus candidate"
        ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
        _log_x(ax)
        basis = "areal mass required (g/cm²)" if core.default_basis(sc) == "areal" else "thickness required (cm)"
        ax.set_xlabel(f"{basis}, log scale; lower is better", fontsize=9, color=INK)
        ylab = ("log10 min crustal abundance (ppm), scarcity proxy" if criterion == "abundance"
                else "PBE band gap (eV)")
        ax.set_ylabel(ylab, fontsize=9, color=INK)
        ax.set_title(f"{sc}: {variant_label(vs.iloc[vd])}", fontsize=10, color=INK, loc="left")
    handles = [Line2D([], [], ls="", marker="o", ms=3, color="#b9b8b2", label=f"B4 set, R2 rule (n={S.size:,})")]
    handles += [Line2D([], [], ls="-", marker="o", ms=5, color=LAYER_COLORS[L], label=f"Pareto layer {L}")
                for L in (1, 2, 3)]
    handles += [Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec=INK, label="consensus candidate")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Requirement vs {'min crustal abundance' if criterion == 'abundance' else 'band gap'} "
                 f"under B4 (R2 rule, air kerma, with coherent): non-dominated layers 1–3",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / name, dpi=300, facecolor=SURF)
    plt.close(fig)


def fig_rule_overlap(ov: pd.DataFrame, vss):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURF)
    groups = [(sc, b) for sc in SCENARIOS for b in ("B0", "B4")]
    labels = [f"{sc}\n{b}" for sc, b in groups]
    ax = axes[0]; _style(ax)
    pairs = [("change_R1_R2", "R1→R2", LAYER_COLORS[1]), ("change_R2_R3", "R2→R3", LAYER_COLORS[2]),
             ("change_R1_R3", "R1→R3", LAYER_COLORS[3])]
    w = 0.26
    xs = np.arange(len(groups))
    for k, (col, lab, colr) in enumerate(pairs):
        med, lo, hi = [], [], []
        for sc, b in groups:
            s = ov[(ov.scenario == sc) & (ov.bundle == b)][col] * 100
            med.append(s.median()); lo.append(s.median() - s.min()); hi.append(s.max() - s.median())
        ax.bar(xs + (k - 1) * w, med, width=w - 0.03, color=colr, label=lab, zorder=2)
        ax.errorbar(xs + (k - 1) * w, med, yerr=[lo, hi], fmt="none", ecolor=INK2, lw=0.8, capsize=2, zorder=3)
    ax.axhline(20, color=INK2, ls="--", lw=0.8)
    ax.text(len(groups) - 0.5, 20.5, "~20 % flag", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of formula-level top-50 that changes (%)", fontsize=9, color=INK)
    ax.set_title("Top-50 turnover between representative rules\n(bar = median over primary variants; whisker = range)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]; _style(ax)
    regions = [("all3", "in all three"), ("R1R2_only", "R1 & R2 only"), ("R2R3_only", "R2 & R3 only"),
               ("R1R3_only", "R1 & R3 only"), ("R1_only", "R1 only"), ("R2_only", "R2 only"),
               ("R3_only", "R3 only")]
    cols = ["#2a78d6", "#86b6ef", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#6250d6"]
    bottom = np.zeros(len(groups))
    for (key, lab), colr in zip(regions, cols):
        vals = []
        for sc, b in groups:
            vs = vss[sc]
            vid = int(vs.variant_id[display_col(sc, vs)])
            vals.append(ov[(ov.scenario == sc) & (ov.bundle == b) & (ov.variant_id == vid)][key].iloc[0])
        vals = np.array(vals, dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=colr, label=lab, width=0.6, edgecolor=SURF, linewidth=1.0, zorder=2)
        bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("number of formulas in the union of the three top-50s", fontsize=9, color=INK)
    ax.set_title("Overlap regions at the displayed variant\n(DIAG RQR6 12° 0.35 mm Pb; CS137, CO60 T=0.01)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "A3_rule_overlap.png", dpi=300, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------
def compute() -> dict:
    t0 = time.time()
    paths.ensure(OUT); paths.ensure(FIG)
    weights = core.simplex_grid(3, 0.1)
    assert weights.shape == (66, 3)
    qmats, vss = {}, {}
    res = {"ranks": [], "topsis": [], "pareto": [], "summary": [], "overlap": [], "overlap_members": [],
           "sanity": []}
    set_sizes = {}
    for sc in SCENARIOS:
        q, vs = scen_matrix(sc)
        qmats[sc], vss[sc] = q, vs
        for b in A3_BUNDLES:
            r = battery(sc, b, q, vs, weights)
            set_sizes[f"{sc}/{b}"] = r["set_size"]
            for k in ("ranks", "topsis", "pareto", "summary"):
                res[k].append(r[k])
        res["overlap"].append(rule_overlap(sc, q, vs))
        res["overlap_members"].append(rule_overlap_members(sc, q, vs, display_col(sc, vs)))
        res["sanity"].append(sanity_ranks(sc, q, vs))
    out = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    u = attrs()
    for k in ("ranks", "pareto"):
        out[k]["formula"] = u.formula_pretty.to_numpy()[out[k].row]
        out[k]["material_id"] = u.material_id.to_numpy()[out[k].row]
    out["consensus"] = consensus_table(out["summary"], out["ranks"], qmats, vss)
    out["frontier_diag"] = frontier_diagnostics(qmats, vss)
    out["topsis_check"] = topsis_log_check(qmats, vss, weights)
    out["weights"] = pd.DataFrame(weights, columns=["w_requirement", "w_log10_abundance", "w_band_gap"])
    out["weights"].insert(0, "weight_id", [f"w{i:02d}" for i in range(len(weights))])
    out["qmats"], out["vss"], out["set_sizes"] = qmats, vss, set_sizes
    out["pb_info"] = pb_cosine_distance()[1]
    out["runtime_compute_s"] = time.time() - t0
    return out


def save(out: dict) -> None:
    out["ranks"].to_parquet(OUT / "method_ranks.parquet", index=False)
    out["topsis"].to_parquet(OUT / "topsis_per_weight_ranks.parquet", index=False)
    out["weights"].to_csv(OUT / "topsis_weights.csv", index=False)
    out["pareto"].to_parquet(OUT / "pareto_layers.parquet", index=False)
    out["consensus"].to_csv(OUT / "candidates_top20_union.csv", index=False)
    out["consensus"][out["consensus"].status == "consensus"].to_csv(OUT / "consensus.csv", index=False)
    out["overlap"].to_csv(OUT / "rule_overlap.csv", index=False)
    out["overlap_members"].to_csv(OUT / "rule_overlap_changed_formulas_display_variant.csv", index=False)
    out["sanity"].to_csv(OUT / "sanity_ranks.csv", index=False)
    out["frontier_diag"].to_csv(OUT / "frontier_diagnostics.csv", index=False)
    out["topsis_check"].to_csv(OUT / "topsis_requirement_transform_check.csv", index=False)
    meta = {"set_sizes_R2": out["set_sizes"], "pb_reference": out["pb_info"],
            "display_variants": {sc: {"variant_id": int(out["vss"][sc].variant_id[display_col(sc, out["vss"][sc])]),
                                      "label": variant_label(out["vss"][sc].iloc[display_col(sc, out["vss"][sc])])}
                                 for sc in SCENARIOS},
            "n_primary_variants": {sc: len(out["vss"][sc]) for sc in SCENARIOS},
            "topn": TOPN, "overlap_n": OVERLAP_N, "consensus_frac": CONSENSUS_FRAC,
            "runtime_compute_s": out["runtime_compute_s"]}
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run() -> dict:
    out = compute()
    save(out)
    fig_pareto(out["qmats"], out["vss"], out["consensus"], out["pareto"])
    fig_pareto(out["qmats"], out["vss"], out["consensus"], out["pareto"], criterion="band_gap",
               name="A3_pareto_gap_B4.png")
    fig_rule_overlap(out["overlap"], out["vss"])
    write_report(out)
    return out


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def _md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _rng(s: pd.Series, fmt: str = "{:.1f}", scale: float = 1.0) -> str:
    s = s.dropna() * scale
    if s.empty:
        return "–"
    lo, hi = s.min(), s.max()
    return fmt.format(lo) if np.isclose(lo, hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def write_report(out: dict) -> None:
    u = attrs()
    cons, ov, san, fd = out["consensus"], out["overlap"], out["sanity"], out["frontier_diag"]
    ranks, topsis_w, pareto = out["ranks"], out["topsis"], out["pareto"]
    vss = out["vss"]
    disp = {sc: int(vss[sc].variant_id[display_col(sc, vss[sc])]) for sc in SCENARIOS}
    dlab = {sc: variant_label(vss[sc].iloc[display_col(sc, vss[sc])]) for sc in SCENARIOS}
    L: list[str] = []
    w = L.append

    w("# A3: Candidates and method battery (RQ3, supporting)\n")
    w("Agent C, Phase 2 wave 1. Regenerate with `python -m shieldmap.analysis.candidates` from `shieldmap/` "
      f"(compute ≈ {out['runtime_compute_s']:.0f} s, plus figures). Every number derives from Tables A/B/C through "
      "`analysis.core`; no new data enters.\n")

    # ---------------- headline
    w("## Headline\n")
    hl = []
    for sc in SCENARIOS:
        for b in A3_BUNDLES:
            c = cons[(cons.scenario == sc) & (cons.bundle == b) & (cons.status == "consensus")]
            txt = ", ".join(f"{r.formula} ({r.textbook})" for r in c.itertuples()) or "none"
            n_md = int(((cons.scenario == sc) & (cons.bundle == b) & (cons.status != "consensus")).sum())
            hl.append({"scenario": sc, "bundle": b, "consensus candidates (textbook?)": txt,
                       "method-dependent": n_md})
    w(_md(pd.DataFrame(hl)) + "\n")
    diag = cons[cons.scenario == "DIAG"]
    best3 = diag.sort_values("frac_variants_ge3_methods", ascending=False).iloc[0]
    w(f"- **DIAG has no consensus candidate under any bundle.** No compound reaches the top-20 of ≥ 4 of the 5 methods "
      f"in even one of the 36 primary variants. A relaxed ≥ 3-method criterion would still fail the 80 % bar: the "
      f"closest compound is {best3.formula} under {best3.bundle}, at {best3.frac_variants_ge3_methods:.0%} of variants. "
      "The requirement-only top-20 itself "
      f"rotates with tube quality and target: under B4 the most stable compound is in it in only "
      f"{diag[diag.bundle == 'B4'].frac_variants_top20_a.max():.0%} of the variants. The DIAG frontier is a flat "
      "plateau (see below), so no individual DIAG leader is robust. The *family* that leads is robust: rare-earth "
      "intermetallics, antimonides and iodides, explained by K-edge physics.\n")
    w("- **CS137 and CO60 consensus = the requirement leaders, confirmed by Pareto layering and rule robustness** "
      "(methods a, b and e).\n"
      "  - TOPSIS (c) never agrees with them, and the Pb-shape distance (d) is undefined for the radionuclides.\n"
      "  - Under B4 the consensus is the densest stable metals (Ir, Os, Pt and their W alloys). They are non-textbook "
      "by the rule below, but they are the physically expected outcome of density control at 0.66–1.3 MeV.\n"
      "  - Under B6 the consensus is W and Ta phases, which are textbook.\n"
      "  - Under B5 the consensus is dense wide-gap oxides and fluorides (HfO₂, Lu₂WO₆, ErTaO₄, Dy₂Hf₂O₇, TlF), which "
      "are non-textbook; see Surprises.\n")
    om = ov[["change_R1_R2", "change_R2_R3", "change_R1_R3"]].max()
    w(f"- **The representative rule barely moves the top-50.** The largest turnover anywhere is "
      f"{100 * om.max():.0f} %, and R1→R2 is at most {100 * om.change_R1_R2:.0f} %. Both are well below the ~20 % "
      "flag.\n")

    # ---------------- method
    w("## Method\n")
    w("**Settings.**\n"
      "- Rule **R2**.\n"
      "- Primary variants: air-kerma response, with coherent scattering. DIAG has 36 "
      "(RQR 5/6/8/9 × anode 10/12/16° × 0.25/0.35/0.50 mm Pb); CS137 and CO60 have 4 transmission targets each.\n"
      "- Basis: **areal mass** for DIAG, **thickness** for CS137 and CO60.\n"
      "- Bundles: B4 (main), B5 = B4 + GAP(3.1 eV), B6 = B4 + ABUND(1 ppm).\n"
      "- R2 candidate-set sizes, the same for every scenario: "
      + ", ".join(f"{k.split('/')[1]} {v:,}" for k, v in out["set_sizes"].items() if k.startswith("DIAG")) + ".\n")
    w("**Displayed variant** for the figures and single-variant tables: "
      + "; ".join(f"{sc} {dlab[sc]}" for sc in SCENARIOS)
      + ". CS137 rankings are identical for every T (a single line), and CO60 rankings change only marginally "
        "with T.\n")
    w("**Unweighted Pareto sets.** `core.pareto_layers` is applied to two criterion pairs:\n"
      "- (requirement ↓, log10 min crustal abundance ↑);\n"
      "- (requirement ↓, PBE band gap ↑).\n\n"
      "Layers 1–3 are computed for every scenario × bundle × primary variant "
      "(`data/phase2/A3/pareto_layers.parquet`).\n")
    w("**Method battery.** Each method ranks the bundle's R2 set at formula level, once per variant:\n")
    w("| | Method | Definition |\n|---|---|---|\n"
      "| a | Requirement | Rank by requirement (`core.ranks`). |\n"
      "| b | Pareto layer + requirement | Sort by layer on (requirement, log10 abundance) from `core.pareto_layers` "
      "(layers 1–3; all deeper compounds share layer 4), then by requirement, then by row. |\n"
      "| c | TOPSIS, **weight-swept** | `core.topsis` on three criteria: requirement (lower is better), log10 abundance minus its "
      "minimum over the set (benefit) and band gap (benefit). It is run for each of the 66 weight vectors of "
      "`core.simplex_grid(3, 0.1)`. The method's ordering is the **median of the 66 per-weight ranks**, ties broken by "
      "requirement. This is the only weighted score in A3, and it is never a headline. |\n"
      "| d | Pb-shape distance (DIAG only) | Cosine distance between the compound's μ/ρ(E) and Pb's μ/ρ(E) over the 261 "
      "grid nodes from 20 to 150 keV, ties broken by requirement. At a fixed areal mass M, log T(E) = −M·μ/ρ(E), so "
      "the cosine distance on log T equals the cosine distance on μ/ρ for every M. |\n"
      "| e | Rule robustness | The formula's worst formula-level requirement rank across R1, R2 and R3; under R3 the "
      "formula's best entry is used. A formula missing from a rule's bundle set counts as rank ∞. Ties broken by the "
      "R2 rank. |\n")
    pb = out["pb_info"]
    w(f"**Pb reference for method d.** The Table C μ/ρ row of {pb['pb_material_id']} (Materials Project Pb, Fm-3m, the R2 "
      f"representative), over {pb['E_min']:.1f}–{pb['E_max']:.1f} keV ({pb['n_nodes']} nodes). It agrees with the "
      f"Table A source array (`xsec_grid.npz`, `total_wcoh`, Z = 82) to {pb['max_rel_dev_vs_xsec_grid']:.1e}, "
      "i.e. to float32 precision.\n")
    w("**Top-20 and consensus.**\n"
      "- Each method's **top-20** is the first 20 compounds of its ordering.\n"
      "- A **consensus** candidate (formula level) is in the top-20 of enough methods in ≥ 80 % of the scenario's "
      "primary variants: ≥ 4 of 5 methods for DIAG, or ≥ 3 of 4 for CS137 and CO60, where method d is undefined.\n"
      "- Every other compound that appears in any method's top-20 in any variant is **method-dependent**.\n")

    # ---------------- consensus tables
    w("## Consensus candidates (scenario × bundle)\n")
    w("**`textbook?` rule.** A compound is marked **yes** only if one of these holds:\n"
      "- every constituent is either a textbook shielding element (W, Pb, Bi, Ba, Gd, Sn, Sb, Ta, Fe, Cu) or a light "
      "matrix element (Z ≤ 20, not a noble gas), and at least one constituent is a textbook element;\n"
      "- the compound is concrete-like: only H, C, O, Na, Mg, Al, Si, S, K, Ca and Fe.\n\n"
      "Everything else is marked **no**.\n")
    w("`q` is the requirement at the displayed variant (cm; g/cm² for DIAG), followed by its range over the primary "
      "variants; for CS137 and CO60 that range is just the T sweep (0.5 to 0.001). `share` is the fraction of primary "
      "variants that meet the consensus criterion.\n")
    cc = cons[cons.status == "consensus"].copy()
    cc["q (range)"] = [f"{a:.3f} ({b:.3f}–{c:.3f})" for a, b, c in zip(cc.q_display, cc.q_min, cc.q_max)]
    cc["share"] = cc.frac_variants_consensus.map("{:.0%}".format)
    cc = cc.rename(columns={"material_id_R2": "material_id (R2)", "material_ids_R3_in_bundle": "R3 entries in bundle",
                            "density": "ρ (g/cm³)", "e_hull": "e_hull (eV/atom)", "band_gap": "gap (eV)",
                            "min_abund_ppm": "min abund. (ppm)", "textbook": "textbook?"})
    cols = ["scenario", "bundle", "formula", "material_id (R2)", "R3 entries in bundle", "q (range)", "ρ (g/cm³)",
            "e_hull (eV/atom)", "theoretical", "gap (eV)", "min abund. (ppm)", "share", "textbook?"]
    w(_md(cc[cols], floatfmt=".3g") + "\n")
    w("For the radionuclide scenarios the three agreeing methods are a, b and e. Method c (TOPSIS) gives every "
      "consensus compound a median rank in the thousands; its own top-20 is fluorides and wide-gap oxides of abundant "
      "elements (below). The consensus therefore says three things: *these compounds are the requirement leaders, "
      "they sit on Pareto layers 1–2 for scarcity, and the representative rule does not move them*. It does not say "
      "they are good on every criterion.\n")

    w("### Each method's top-10 under B4 at the displayed variant\n")
    rows = []
    for sc in SCENARIOS:
        r = ranks[(ranks.scenario == sc) & (ranks.bundle == "B4") & (ranks.variant_id == disp[sc])]
        for m in METHODS:
            col = f"rank_{m}"
            if col not in r or r[col].isna().all():
                continue
            t = r.sort_values(col).head(10)
            rows.append({"scenario": sc, "method": f"{m}: {METHOD_NAMES[m]}", "top-10": ", ".join(t.formula)})
    w(_md(pd.DataFrame(rows)) + "\n")
    w("The full list of the method-dependent candidates is in `data/phase2/A3/candidates_top20_union.csv`. For each "
      "compound it gives the share of variants in each method's top-20 (`frac_variants_top20_<m>`) and the median "
      "rank under each method.\n")

    # TOPSIS per-weight
    w("### TOPSIS per-weight behaviour (method c, weight-swept)\n")
    w("The table lists the three requirement leaders and the three TOPSIS leaders under B4 at the displayed "
      "variant.\n")
    wts = out["weights"]
    wcols = [c for c in topsis_w.columns if c.startswith("w") and c[1:].isdigit()]
    w_req = f"w{int(wts.index[wts.w_requirement == 1.0][0]):02d}"
    trows = []
    for sc in SCENARIOS:
        tw = topsis_w[(topsis_w.scenario == sc) & (topsis_w.bundle == "B4") & (topsis_w.variant_id == disp[sc])]
        rr = ranks[(ranks.scenario == sc) & (ranks.bundle == "B4") & (ranks.variant_id == disp[sc])]
        pick = list(rr.sort_values("rank_a").row.head(3)) + list(rr.sort_values("rank_c").row.head(3))
        for i in dict.fromkeys(pick):
            x = tw[tw.row == i]
            if x.empty:
                continue
            v = x[wcols].to_numpy().ravel()
            trows.append({"scenario": sc, "formula": u.formula_pretty[i], "median rank": float(np.median(v)),
                          "best": int(v.min()), "worst": int(v.max()),
                          "rank at w = (1, 0, 0)": int(x[w_req].iloc[0]),
                          "share of 66 weights in top-20": f"{np.mean(v <= TOPN):.0%}"})
    w(_md(pd.DataFrame(trows), floatfmt=".0f") + "\n")
    tc = out["topsis_check"]
    w("**Fluorides dominate.** In every scenario the median-rank TOPSIS top-20 is dominated by Ba and rare-earth "
      "fluorides (band gap 7–10 eV, F abundant). The reason is that most of the simplex puts most of the weight on "
      "abundance or band gap rather than on the requirement.\n")
    w("**Requirement transform check.** The method uses the raw requirement; the check repeats it with "
      "log10(q / q_min). The two top-20s share " + _rng(tc.top20_overlap_raw_vs_log, "{:.0f}")
      + " of 20 compounds across scenario × bundle (`topsis_requirement_transform_check.csv`).\n")
    w("Per-weight ranks for every compound in any method's top-50 are in `topsis_per_weight_ranks.parquet`; columns "
      "w00…w65 map to the weight vectors in `topsis_weights.csv`.\n")

    # ---------------- Pareto
    w("## Unweighted Pareto sets (B4)\n")
    w("![Pareto B4](figures/A3_pareto_B4.png)\n")
    w("**Figure #4.** Requirement vs log10 min crustal abundance under B4 at the displayed variants. Pareto layers 1–3 "
      "are coloured and consensus candidates are circled. The companion plot of requirement vs band gap is "
      "`figures/A3_pareto_gap_B4.png`.\n")
    prow = []
    for sc in SCENARIOS:
        for crit in ("abundance", "band_gap"):
            p = pareto[(pareto.scenario == sc) & (pareto.bundle == "B4") & (pareto.variant_id == disp[sc])
                       & (pareto.criterion == crit) & (pareto.layer == 1)].sort_values("q")
            prow.append({"scenario": sc, "criterion": crit, "n (layer 1)": len(p),
                         "layer 1, by increasing requirement": ", ".join(p.formula)})
    w(_md(pd.DataFrame(prow)) + "\n")
    st = pareto[(pareto.bundle == "B4") & (pareto.layer == 1) & (pareto.criterion == "abundance")]
    srow = []
    for sc in SCENARIOS:
        s2 = st[st.scenario == sc].groupby("formula").size() / len(vss[sc])
        srow.append({"scenario": sc, "layer-1 members in 100 % of variants": ", ".join(s2[s2 >= 0.999].index) or "none",
                     "distinct layer-1 members over all variants": len(s2)})
    w("Stability of layer 1 (abundance criterion) across the primary variants:\n")
    w(_md(pd.DataFrame(srow)) + "\n")

    # ---------------- DIAG frontier diagnostics
    w("## DIAG frontier: plateau, leaders and families\n")
    f4 = fd[(fd.scenario == "DIAG") & (fd.bundle == "B4")]
    w(f"**Leaders.** Under B4 the 36 DIAG variants have {f4.leader.nunique()} different leaders: "
      f"{', '.join(f'{k} ×{v}' for k, v in f4.leader.value_counts().items())}.\n")
    w(f"**Plateau.** The 20th-best compound needs only {_rng(f4.spread_top20, '{:.1f}', 100)} % more areal mass than the "
      f"best, so the ordering inside the top-20 is set by sub-percent differences.\n")
    w("**Composition of the top-20**, as the range of shares over the 36 variants:\n"
      f"- containing a lanthanide: {_rng(f4.share_lanthanide, '{:.0f}', 100)} %;\n"
      f"- containing an element with crustal abundance < {SCARCE_PPM} ppm: {_rng(f4.share_scarce, '{:.0f}', 100)} %;\n"
      f"- textbook by the rule: {_rng(f4.share_textbook, '{:.0f}', 100)} %.\n")
    byq = []
    for qual in ("RQR5", "RQR6", "RQR8", "RQR9"):
        x = f4[f4.variant.str.startswith(qual)]
        byq.append({"quality": qual, "leaders (9 variants)": ", ".join(sorted(x.leader.unique())),
                    "top-20 spread (%)": _rng(x.spread_top20, "{:.1f}", 100),
                    "top-5 at 12°, 0.35 mm": x[x.variant.str.contains("12° 0.35")].top5.iloc[0]})
    w(_md(pd.DataFrame(byq)) + "\n")
    f6 = fd[(fd.scenario == "DIAG") & (fd.bundle == "B6")]
    f5 = fd[(fd.scenario == "DIAG") & (fd.bundle == "B5")]
    b6q = "; ".join(f"{qq} {_rng(f6[f6.variant.str.startswith(qq)].frontier_vs_B4, '{:.1f}', 100)} %"
                    for qq in ("RQR5", "RQR6", "RQR8", "RQR9"))
    w(f"**Excluding scarce constituents moves the DIAG frontier little.** B6 (ABUND ≥ 1 ppm) removes Pd, Rh, Ir, Pt, Au, Sb, Bi, I, "
      f"Ag and other scarce elements. Its frontier is {_rng(f6.frontier_vs_B4, '{:.1f}', 100)} % above B4's "
      f"(by quality: {b6q}) and is led by {', '.join(f6.leader.value_counts().index[:4])}. For comparison, the B5 frontier (gap ≥ 3.1 eV) is "
      f"{_rng(f5.frontier_vs_B4, '{:.1f}', 100)} % above B4's.\n")
    w("The full per-variant table is `data/phase2/A3/frontier_diagnostics.csv`.\n")

    # ---------------- sanity
    w("## Sanity ranks of known materials (R2; plausibility only, not validation)\n")
    w("Each cell gives the rank within the bundle's R2 set at the displayed variant, the range over the primary "
      "variants in parentheses when it varies, and `×ratio` = requirement / bundle frontier (best). Set sizes: "
      "B0 102,806; B4 29,846. When the material is not in the bundle, the cell names the failing constraint.\n")
    srows = []
    for f in san.formula.unique():
        rec = {"formula": f}
        for sc in SCENARIOS:
            for b in ("B0", "B4"):
                r = san[(san.formula == f) & (san.scenario == sc) & (san.bundle == b)].iloc[0]
                if not r.in_map:
                    rec[f"{sc} {b}"] = "absent"
                elif not r.in_bundle:
                    rec[f"{sc} {b}"] = f"fails {r.fails}"
                else:
                    rr = (f"{int(r.rank_display)}" if r.rank_min == r.rank_max
                          else f"{int(r.rank_display)} ({int(r.rank_min)}–{int(r.rank_max)})")
                    rec[f"{sc} {b}"] = f"{rr}; ×{r.ratio_display:.2f}"
        srows.append(rec)
    w(_md(pd.DataFrame(srows)) + "\n")
    w("**Notes on the table.**\n"
      "- The R2 representative of Sn is α-Sn (mp-117, Fd-3m, 5.56 g/cm³; β-Sn is 7.29 g/cm³). Its thickness ranks "
      "(CS137, CO60) are therefore pessimistic; its DIAG areal-mass rank is unaffected.\n"
      "- The R2 entry of WO₃ (mp-2928451) has 6.76 g/cm³, against about 7.2 g/cm³ measured.\n"
      "- The AWO₄ rows are every ICSD-matched R2 formula of that stoichiometry. AlWO₄ and NaWO₄ are not classic "
      "scheelite or wolframite tungstates.\n"
      "- PbO and PbWO₄ fail B4 through the PB constraint.\n")

    # ---------------- rule overlap
    w("## Representative-rule overlap (top-50, formula level)\n")
    w("![rule overlap](figures/A3_rule_overlap.png)\n")
    w("**Figure #7.** Left: share of the top-50 that changes between rules, as the median and range over the primary "
      "variants. Right: overlap regions of the three top-50 sets at the displayed variant.\n")
    orow = []
    for (sc, b), g in ov.groupby(["scenario", "bundle"]):
        orow.append({"scenario": sc, "bundle": b, "variants": len(g),
                     "R1→R2 (%)": _rng(g.change_R1_R2, "{:.0f}", 100), "R2→R3 (%)": _rng(g.change_R2_R3, "{:.0f}", 100),
                     "R1→R3 (%)": _rng(g.change_R1_R3, "{:.0f}", 100)})
    w(_md(pd.DataFrame(orow)) + "\n")
    w(f"**Not flagged.** No R1 vs R2 turnover exceeds the ~20 % flag; the maximum is "
      f"{100 * ov.change_R1_R2.max():.0f} %.\n"
      "- **DIAG** uses areal mass, which does not depend on density. Polymorphs of one formula therefore tie, and the "
      "rules can differ only through set membership: the B4 EXP and STAB filters act on the entry each rule "
      "chooses.\n"
      "- **CS137 and CO60** use thickness. Under B0, R3 admits denser metastable polymorphs, mostly of Pu and Np "
      "intermetallics: at CS137, Pu₃Au and Pu₃Ta jump from rank 138 and 63 under R1/R2 to 9 and 8 under R3. Under B4, "
      "RAD, STAB and EXP remove them. The remaining B4 differences come from formulas whose R1 entry is theoretical "
      "(Re, VIr₃, NbIr₃, FeIr₃, PrSbPd), which EXP removes under R1 only.\n")
    mem = out["overlap_members"]
    if len(mem):
        w("Formulas that are not in all three top-50s at the displayed variant (∞ = absent from that rule's set):\n")
        w(_md(mem.drop(columns=["variant_id"]), floatfmt=".0f") + "\n")

    # ---------------- surprises
    w("## Surprises (with evidence)\n")
    pr = u[(u.formula_reduced == "PrSbPd") & u.eligible]
    w("**1. Rare-earth intermetallics, not textbook materials, lead the DIAG B4 frontier, and no individual leader is "
      "robust.**\n\n"
      "The coordinator's smoke-run leaders (PrSbPd, CeSnRh, CeSbRh, CeSbPd, PrSnRh) are exactly the RQR5 0.25 mm Pb "
      "leaders.\n\n"
      "*They are not data artefacts:*\n"
      "- all five are ICSD-matched (theoretical = False), with e_hull 0–0.004 eV/atom;\n"
      "- DIAG uses areal mass (μ/ρ only), so density plays no role and the CsI-type polymorph problem cannot arise; "
      "composition is the only input.\n\n"
      "*But they are not robust candidates either:*\n"
      "- they lead only at RQR5 and RQR6. At RQR8, Tb and Eu iodides and antimonides lead; at RQR9, Er/Ho–Ir/Pt "
      "iodides and Eu–Au compounds lead;\n"
      "- the top-20 spans ≤ 2 % in areal mass, so their order is at noise level;\n"
      "- Pd (0.015 ppm) and Rh (0.001 ppm) are at most modestly load-bearing. Removing every constituent below "
      f"1 ppm (B6) raises the frontier by {b6q}, and leaves rare-earth–Sn binaries (Pr₅Sn₄, Nd₁₁Sn₁₀, EuSn, Eu₅Sn₃H, Ho₅Sn₃).\n\n"
      "*The robust statement is family-level.* Per unit mass, the best absorbers for 70–120 kV beams combine two "
      "ingredients:\n"
      "- a lanthanide whose K-edge sits just below the bulk of the spectrum (Ce 40.4, Pr 42.0, Nd 43.6, Eu 48.5, "
      "Tb 52.0, Er 57.5 keV). The preferred lanthanide moves up the series as the tube voltage rises;\n"
      "- a partner that absorbs just below that edge: the Sn, Sb or I K-edges (29–33 keV) or 5d L-edges.\n\n"
      "This is K-edge physics a specialist would predict, and lead-free aprons already use the multi-element "
      "principle. The specific intermetallics are nevertheless not textbook materials.\n")
    w(f"**2. PrSbPd's rank depends on the representative rule.**\n\n"
      f"- Its R1 representative ({', '.join(pr[pr.in_R1].material_id)}, e_hull 0) is a theoretical polymorph, which the "
      f"B4 EXP filter removes. The formula is therefore absent from R1 ∩ B4.\n"
      f"- Its ICSD-matched entry ({', '.join(pr[pr.in_R2].material_id)}, e_hull 0.004 eV/atom) leads under R2.\n"
      "- Method e therefore ranks this DIAG leader near the bottom.\n\n"
      "The same mechanism (an R1 representative that is theoretical, so EXP removes it under R1) produces the R1 vs R2 "
      "differences under B4 in all three scenarios.\n")
    w("**3. The radionuclide scenarios have non-textbook consensus candidates. This is a gate trigger, so it needs "
      "careful reading.**\n"
      "- **B4:** Ir, Os, Pt, Ir₃W and Pt₂W (ρ 20.9–22.4 g/cm³; all stable and ICSD-matched; crustal abundance "
      "0.001–0.005 ppm). These are simply the densest stable solids. A physicist would predict them, so they signal "
      "*density control at MeV energies*, not a new material.\n"
      "- **B5:** HfO₂ (mp-352, monoclinic, 10.24 g/cm³, gap 4.0 eV), Lu₂WO₆, ErTaO₄, Dy₂Hf₂O₇ and, for CS137, TlF. These "
      "are dense wide-gap oxides and fluorides: non-textbook as shielding materials, but known dense optical and "
      "dielectric hosts. PBE underestimates band gaps, so the GAP(3.1) filter is conservative for them.\n"
      "- **B6:** W, BW₂, Ta and a Ta subnitride/hydride, which are textbook by the rule.\n\n"
      "*Caveat:* for the radionuclides the consensus test discriminates weakly. Once the leaders are stable "
      "single-polymorph phases, methods a, b and e are near-duplicates of the requirement order.\n")
    w("**4. Weight-swept TOPSIS and the Pb-shape distance pick entirely different compounds.**\n"
      "- TOPSIS (median over the simplex) is led by Ba and rare-earth fluorides (Ba₃Al₂F₁₂, NdF₃, BaSiF₆) in every "
      "scenario.\n"
      "- The Pb-shape distance (d) is led by Tl–Bi compounds (TlBiS₂ and others), Pb's periodic-table neighbours, "
      "whose areal mass is not competitive.\n\n"
      "Neither method ever joins a DIAG consensus. That is why DIAG has none, and why even a relaxed ≥ 3-method "
      f"criterion is met by no compound in ≥ 80 % of variants (closest: {best3.formula} under {best3.bundle}, "
      f"{best3.frac_variants_ge3_methods:.0%}).\n")
    sb4 = san[(san.bundle == "B4") & san.in_bundle.fillna(False).astype(bool)].set_index(["scenario", "formula"])

    def rr(sc, f):
        r = sb4.loc[(sc, f)]
        a, b = f"{r.ratio_min:.2f}", f"{r.ratio_max:.2f}"
        return f"×{a}" if a == b else f"×{a}–{b}"
    w("**5. Per unit mass, textbook materials sit well behind the B4 frontier at DIAG.** Required areal mass relative "
      "to the frontier, over the 36 variants:\n"
      f"- W {rr('DIAG', 'W')};\n- Bi {rr('DIAG', 'Bi')};\n- Sn {rr('DIAG', 'Sn')};\n"
      f"- Gd₂O₃ {rr('DIAG', 'Gd2O3')};\n- BaSO₄ {rr('DIAG', 'BaSO4')};\n- CaWO₄ {rr('DIAG', 'CaWO4')}.\n\n"
      f"At CS137 and CO60 (thickness basis), W ranks {int(sb4.loc[('CS137', 'W')].rank_display)} and "
      f"{int(sb4.loc[('CO60', 'W')].rank_display)} of 29,846 under B4 ({rr('CS137', 'W')} and {rr('CO60', 'W')} of the "
      "Ir frontier). This is plausible and consistent with density control.\n")

    # ---------------- decisions
    w("## Decisions\n")
    w("- **A3-D-001 Formula level.** R2 holds one entry per formula. R3 ranks are formula-level, using the formula's "
      "best entry in each variant. *Alternative:* rank R3 entries individually, which lets one formula's duplicates "
      "push other formulas down.\n")
    w("- **A3-D-002 Method b ordering.** Sort by Pareto layer (1, 2, 3; all deeper compounds share layer 4), then by "
      "requirement, then by row. Layer 1 of the abundance front spans the whole requirement range, ending at abundant "
      "phases of the O₂, Si or Fe type. By design, b's top-20 is therefore roughly layers 1–2, and it includes "
      "abundant compounds with high requirements.\n")
    w("- **A3-D-003 TOPSIS inputs.**\n"
      "  - Requirement: raw, as a lower-is-better criterion.\n"
      "  - log10 min abundance: shifted by its minimum within the set, so that every column is ≥ 0 before "
      "`core.topsis` applies vector normalisation; a benefit.\n"
      "  - Band gap: a benefit.\n"
      "  - Each weight vector ranks by closeness (ties by row); the method's rank is the median of the 66 ranks, "
      "ties by requirement.\n"
      "  - Cross-checked against a log-requirement variant (overlap reported above). *Reverse:* the column "
      "construction in `battery()`.\n")
    w("- **A3-D-004 Method d reference.** The Table C μ/ρ row of mp-20483 (Pb), cross-checked against the Table A "
      "source array. All 261 grid nodes in [20, 150] keV are used, unweighted.\n")
    w("- **A3-D-005 Method e, missing formulas.** A formula absent from a rule's bundle set counts as rank ∞, because "
      "it is not robust to the rule. *Alternative:* ignore the missing rule, which would hide PrSbPd's absence "
      "under R1.\n")
    w("- **A3-D-006 Top-20 and consensus.** Each method's top-20 is the first 20 of its ordering, so there are no ties "
      "at the cut. For the radionuclides method d is undefined, so the threshold is ≥ 3 of 4. The 80 % share is taken "
      "over the scenario's primary variants (36, 4 and 4); CS137's 4 variants give identical rankings.\n")
    w("- **A3-D-007 Textbook rule.** Coded as stated in the consensus section (`textbook()`) and deliberately strict: "
      "Lu₂WO₆, ErTaO₄ and ZnWO₄ count as *no* because Lu, Er and Zn are outside the list. The rule is a reporting "
      "label, not a filter.\n")
    w("- **A3-D-008 Displayed variants.** DIAG RQR6 12° 0.35 mm Pb (mid-sweep); CS137 and CO60 T = 0.01.\n")
    w("- **A3-D-009 Sanity tungstates.** Every ICSD-matched R2 formula that matches AWO₄.\n")

    # ---------------- files
    w("## Files\n")
    w("**Code.** `src/shieldmap/analysis/candidates.py`; `run()` regenerates everything. Tests: "
      "`tests/test_candidates.py`.\n")
    w("**Data** (`data/phase2/A3/`):\n"
      "- `consensus.csv`, `candidates_top20_union.csv`;\n"
      "- `method_ranks.parquet` (per variant, every compound in any method's top-50);\n"
      "- `topsis_per_weight_ranks.parquet`, `topsis_weights.csv`, `topsis_requirement_transform_check.csv`;\n"
      "- `pareto_layers.parquet`;\n"
      "- `rule_overlap.csv`, `rule_overlap_changed_formulas_display_variant.csv`;\n"
      "- `sanity_ranks.csv`, `frontier_diagnostics.csv`, `summary.json`.\n")
    w("**Figures:** `figures/A3_pareto_B4.png` (#4), `figures/A3_rule_overlap.png` (#7), "
      "`figures/A3_pareto_gap_B4.png`.\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    run()
