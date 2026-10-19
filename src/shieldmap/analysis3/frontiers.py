"""Robust frontiers, artefact control, tiers and textbook ambient phases.

1. Every stage 2 price (single constraints and bundles B1–B6, all 264 profile variants, rules R1/R2/R3,
   both bases) is evaluated with six frontier definitions side by side: best, 10th, median50,
   best_icsd, best_clean and median50_clean (common.py). A price is flagged when the best-compound
   frontier on either side of the ratio (F0 or F0 ∩ K) is an artefact entry (theoretical, e_hull >
   0.1 eV/atom). The headline kind is best_clean in every cell; the original per-cell rule
   (median50 where flagged, else best) is kept as `legacy_rule_kind` in headline_kinds.csv.
   The hardening revision adds: top-N medians (N = 1…100) on the full and the artefact-free map, the composition of
   each top-50 set, marginal prices given the other B4 constraints, and absolute increments.
2. Tiers on the thickness basis. With δ = ρ_MP/ρ_true − 1 drawn from the compound's class
   distribution, the true requirement of compound m lies in [q_m(1+lo_m), q_m(1+hi_m)], where
   (lo, hi) is the 90 % interval of δ for m's bootstrap class. Tier 1 is every compound that could
   be the best: q_m(1+lo_m) ≤ U* = min_j q_j(1+hi_j). Tier 2 repeats on the remainder, and so on.
   Ordered lists inside a tier are never reported.
3. Textbook ambient phases: each named textbook material is pinned to its ambient phase by
   ICSD match plus space group (lowest e_hull among those), and the choice is recorded. On the areal
   basis this is immaterial (μ/ρ is identical across polymorphs); on the thickness basis it is not.

Outputs: data/phase3/S4/{frontier_prices.parquet, headline_kinds.csv, artefact_frontiers.csv, topn_prices.csv,
top50_composition.csv, marginal_prices.csv, absolute_increments.csv,
textbook_phases.csv, tiers.parquet, tier_summary.csv, band_used.json, summary.json}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import prices as p2prices
from shieldmap.analysis3 import common

OUT = common.P3 / "S4"
SCENARIOS = ("DIAG", "CS137", "CO60")
BUNDLE_NAMES = ("B1", "B2", "B3", "B4", "B5", "B6")

# Ambient phase of each named textbook material: (reduced formula, space-group number, phase name).
# Selection: eligible MP entries of that formula, ICSD-matched (theoretical == False), with this space
# group; lowest e_hull among them.
TEXTBOOK_PHASES = [
    ("W", 229, "bcc W (Im-3m)"),
    ("Bi", 166, "rhombohedral Bi (R-3m)"),
    ("Sn", 141, "beta-Sn, white tin (I4_1/amd); R2 picks alpha-Sn"),
    ("Sb2O3", 227, "senarmontite (Fd-3m); R2 picks valentinite"),
    ("BaSO4", 62, "barite (Pnma)"),
    ("Gd2O3", 206, "cubic C-type bixbyite (Ia-3)"),
    ("WO3", 14, "monoclinic gamma-WO3 (P2_1/n); R2 picks a triclinic entry"),
    ("Bi2O3", 14, "alpha-Bi2O3 (P2_1/c)"),
    ("PbO", 129, "litharge, alpha-PbO (P4/nmm)"),
    ("Pb", 225, "fcc Pb (Fm-3m)"),
    ("TiO2", 136, "rutile (P4_2/mnm); anatase (I4_1/amd, 141) is R2's choice and is listed too"),
    ("TiO2", 141, "anatase (I4_1/amd)"),
]


# ---------------------------------------------------------------------------------------------
# textbook phases
# ---------------------------------------------------------------------------------------------
def textbook_phases() -> pd.DataFrame:
    u = common.universe()
    recs = []
    for f, sg, name in TEXTBOOK_PHASES:
        s = u[(u.formula_reduced == f) & u.eligible]
        r2 = s[s.in_R2]
        cand = s[(~s.theoretical) & (s.spacegroup_number == sg)].sort_values("energy_above_hull")
        if cand.empty:
            raise RuntimeError(f"no ICSD-matched {f} entry in space group {sg}")
        c = cand.iloc[0]
        recs.append({"formula": f, "phase": name, "spacegroup_number": sg, "row": int(c.row),
                     "material_id": c.material_id, "density": float(c.density),
                     "energy_above_hull": float(c.energy_above_hull), "n_icsd_ids": int(c.n_icsd_ids),
                     "n_candidates_same_sg": int(len(cand)),
                     "R2_material_id": r2.material_id.iloc[0], "R2_spacegroup": int(r2.spacegroup_number.iloc[0]),
                     "R2_density": float(r2.density.iloc[0]),
                     "density_ratio_ambient_vs_R2": float(c.density / r2.density.iloc[0]),
                     "same_as_R2": bool(c.material_id == r2.material_id.iloc[0])})
    return pd.DataFrame(recs)


def textbook_rows() -> dict[str, int]:
    """formula → universe row of the ambient phase (rutile for TiO2)."""
    tp = textbook_phases()
    tp = tp[~((tp.formula == "TiO2") & (tp.spacegroup_number == 141))]
    return dict(zip(tp.formula, tp.row))


# ---------------------------------------------------------------------------------------------
# prices with four frontier kinds
# ---------------------------------------------------------------------------------------------
def constraint_sets() -> list[tuple[str, str, np.ndarray]]:
    """(family, label, mask satisfying the constraint) for stage 2's single constraints and bundles."""
    out = []
    for code, p in p2prices.SINGLES:
        out.append(("single", p2prices.label(code, p), common.constraint_mask(code, p)))
    for b in BUNDLE_NAMES:
        out.append(("bundle", b, common.bundle_mask(b)))
    return out


def _frontier_all(q: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Six frontier kinds plus the best, best-ICSD and best-clean row for every column of q."""
    out = {}
    for tag, m in (("", mask), ("_icsd", mask & common.icsd_mask()), ("_clean", mask & common.clean_mask())):
        idx = np.flatnonzero(m)
        sub = q[idx]
        sub = np.where(np.isfinite(sub), sub, np.inf)
        k = min(50, idx.size)
        part = np.argpartition(sub, k - 1, axis=0)[:k] if idx.size > k else np.broadcast_to(
            np.arange(idx.size)[:, None], (idx.size, q.shape[1]))
        vals = np.take_along_axis(sub, part, axis=0)
        order = np.argsort(vals, axis=0, kind="stable")
        top = np.take_along_axis(vals, order, axis=0)                       # k × V, ascending
        rows = idx[np.take_along_axis(part, order, axis=0)]
        nan = np.full(q.shape[1], np.nan)
        if tag == "":
            out["best"] = top[0]
            out["10th"] = top[9] if k >= 10 else nan
            out["median50"] = 0.5 * (top[24] + top[25]) if k >= 50 else nan
            out["best_row"] = rows[0]
        elif tag == "_icsd":
            out["best_icsd"] = top[0]
            out["best_icsd_row"] = rows[0]
        else:
            out["best_clean"] = top[0]
            out["median50_clean"] = 0.5 * (top[24] + top[25]) if k >= 50 else nan
            out["best_clean_row"] = rows[0]
    for kk in common.KINDS6:
        out[kk] = np.where(np.isfinite(out[kk]), out[kk], np.nan)
    return out


def frontier_prices() -> pd.DataFrame:
    v = common.variants()
    sets = constraint_sets()
    recs = []
    for basis in ("thickness", "areal"):
        q = np.asarray(common.profiles(basis), dtype=np.float64)        # N × 264
        for rule in common.RULES:
            base = common.base_mask(rule)
            f0 = _frontier_all(q, base)
            for fam, lab, km in sets:
                fk = _frontier_all(q, base & km)
                for kind in common.KINDS6:
                    pr = fk[kind] / f0[kind] - 1.0
                    recs.append(pd.DataFrame({
                        "basis": basis, "rule": rule, "family": fam, "constraint": lab, "kind": kind,
                        "variant_id": v.variant_id.to_numpy(), "price": pr,
                        "q_base": f0[kind], "q_constrained": fk[kind],
                        "best_row_base": f0["best_row"], "best_row_constrained": fk["best_row"],
                        "clean_row_base": f0["best_clean_row"], "clean_row_constrained": fk["best_clean_row"]}))
    df = pd.concat(recs, ignore_index=True)
    df = df.merge(v[["variant_id", "scenario", "spectrum", "target_value", "response", "coherent", "primary"]],
                  on="variant_id", how="left")
    fl_b = common.artefact_flag(df.best_row_base.to_numpy())
    fl_c = common.artefact_flag(df.best_row_constrained.to_numpy())
    df["artefact_base"] = fl_b
    df["artefact_constrained"] = fl_c
    df["artefact_any"] = fl_b | fl_c
    return df


def headline_kinds(fp: pd.DataFrame) -> pd.DataFrame:
    """Per basis × rule × constraint × variant: the headline kind (best_clean everywhere) and
    the kind the superseded per-cell rule would have chosen (median50 where 'best' is flagged)."""
    b = fp[fp.kind == "best"][["basis", "rule", "constraint", "variant_id", "artefact_any"]]
    b = b.assign(headline_kind=common.HEADLINE_KIND,
                 legacy_rule_kind=np.where(b.artefact_any, "median50", "best"))
    return b.drop(columns="artefact_any")


def artefact_frontiers(fp: pd.DataFrame) -> pd.DataFrame:
    """Every distinct artefact entry that sets a best-compound frontier, with where it does so."""
    u = common.universe()
    b = fp[(fp.kind == "best")]
    recs = []
    for col, side in (("best_row_base", "F0"), ("best_row_constrained", "F0∩K")):
        fl = common.artefact_flag(b[col].to_numpy())
        sub = b[fl]
        for r, g in sub.groupby(col):
            recs.append({"row": int(r), "material_id": u.material_id.iat[r], "formula": u.formula_pretty.iat[r],
                         "density": float(u.density.iat[r]), "e_hull": float(u.energy_above_hull.iat[r]),
                         "side": side, "n_cells": int(len(g)),
                         "scenarios": ",".join(sorted(g.scenario.unique())),
                         "rules": ",".join(sorted(g.rule.unique())), "bases": ",".join(sorted(g.basis.unique())),
                         "constraints": ",".join(sorted(g.constraint.unique()))[:400]})
    return pd.DataFrame(recs).sort_values(["side", "n_cells"], ascending=[True, False])


# ---------------------------------------------------------------------------------------------
# The hardening revision additions: top-N sensitivity, top-50 composition, marginal prices, absolute increments
# ---------------------------------------------------------------------------------------------
TOPN = (1, 5, 10, 20, 50, 100)
KEY_CONSTRAINTS = ("RAD", "PB", "STAB(0.05)", "EXP", "GAP(3.1)", "ABUND(1)", "B4", "B5", "B6")
MARGINAL = {  # label: (bundle without X, bundle with X) — the price of X given the other practical constraints
    "RAD | B4 without RAD": ("B4-RAD", "B4"), "PB | B4 without PB": ("B4-PB", "B4"),
    "STAB(0.05) | B4 without STAB": ("B4-STAB", "B4"), "EXP | B4 without EXP": ("B4-EXP", "B4"),
    "GAP(3.1) | B4": ("B4", "B5"), "ABUND(1) | B4": ("B4", "B6"),
}


def _named_mask(name: str) -> np.ndarray:
    """Masks for KEY_CONSTRAINTS, the bundles and 'B4-X' (B4 with constraint X removed)."""
    if name.startswith("B4-"):
        drop = {"RAD": "RAD", "PB": "PB", "STAB": "STAB", "EXP": "EXP"}[name[3:]]
        m = np.ones(len(common.universe()), bool)
        for code, param in common.BUNDLES["B4"]:
            if code != drop:
                m &= common.constraint_mask(code, param)
        return m
    if name in common.BUNDLES:
        return common.bundle_mask(name)
    code, _, par = name.partition("(")
    return common.constraint_mask(code, float(par[:-1]) if par else None)


def key_columns() -> list[dict]:
    """Requirement columns for the hardening-revision tables: R2 primary profile variants (Cs-137, Co-60 both bases;
    DIAG areal) and monoenergetic key energies (both bases, with coherent; requirement for T = 0.01)."""
    v = common.variants()
    cols = []
    for basis in ("thickness", "areal"):
        q = common.profiles(basis)
        for sc in ("CS137", "CO60", "DIAG"):
            if sc == "DIAG" and basis == "thickness":
                continue
            for r in v[(v.scenario == sc) & v.primary].itertuples():
                cols.append({"source": sc, "basis": basis, "variant_id": int(r.variant_id), "energy_keV": np.nan,
                             "T": float(r.T_target), "q": np.asarray(q[:, r.variant_id], dtype=np.float64)})
    ln100 = np.log(100.0)
    for basis in ("thickness", "areal"):
        idx = [common.energy_index(e) for e in p2prices.KEY_E]
        qm = common.mono_requirement(basis, True, e_idx=idx)
        for j, e in enumerate(p2prices.KEY_E):
            cols.append({"source": "mono", "basis": basis, "variant_id": -1, "energy_keV": float(e), "T": 0.01,
                         "q": ln100 * np.asarray(qm[:, j], dtype=np.float64)})
    return cols


def _top_sorted(q: np.ndarray, mask: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask & np.isfinite(q))
    k = min(n, idx.size)
    part = idx[np.argpartition(q[idx], k - 1)[:k]] if idx.size > k else idx
    o = np.lexsort((part, q[part]))
    return q[part[o]], part[o]


def _median_topn(v: np.ndarray, n: int) -> float:
    if v.size < n:
        return np.nan
    return float(v[0]) if n == 1 else float(np.median(v[:n]))


def extras_tables() -> dict[str, pd.DataFrame]:
    """Top-N sensitivity, top-50 composition, marginal prices and absolute increments (R2)."""
    u = common.universe()
    base = common.base_mask(common.PRIMARY_RULE)
    clean = common.clean_mask()
    rad_mask = common.constraint_mask("RAD")
    theo = u["theoretical"].to_numpy(bool)
    eh = u["energy_above_hull"].to_numpy(float)
    rho = common.density()
    art = ~clean
    masks = {n: _named_mask(n) for n in set(KEY_CONSTRAINTS) | {"B4-RAD", "B4-PB", "B4-STAB", "B4-EXP"}}
    topn, comp, marg, absinc = [], [], [], []
    for c in key_columns():
        q = c["q"]
        key = {k: c[k] for k in ("source", "basis", "variant_id", "energy_keV", "T")}
        unit = "cm" if c["basis"] == "thickness" else "g/cm2"
        tops = {}
        for name in ("F0",) + KEY_CONSTRAINTS + ("B4-RAD", "B4-PB", "B4-STAB", "B4-EXP"):
            m = base if name == "F0" else base & masks[name]
            for mapname, mm in (("full", m), ("clean", m & clean)):
                tops[(name, mapname)] = _top_sorted(q, mm, max(TOPN))
        for name in KEY_CONSTRAINTS:
            for mapname in ("full", "clean"):
                v0, _ = tops[("F0", mapname)]; vk, _ = tops[(name, mapname)]
                for n in TOPN:
                    topn.append({**key, "constraint": name, "map": mapname, "N": n,
                                 "price": _median_topn(vk, n) / _median_topn(v0, n) - 1.0})
        for name in ("F0",) + KEY_CONSTRAINTS:
            for mapname in ("full", "clean"):
                _, r = tops[(name, mapname)]
                r = r[:50]
                comp.append({**key, "set": name, "map": mapname, "n": int(r.size),
                             "share_radioactive": float((~rad_mask[r]).mean()), "share_theoretical": float(theo[r].mean()),
                             "share_ehull_gt_0.05": float((eh[r] > 0.05).mean()), "share_artefact": float(art[r].mean()),
                             "rho_median": float(np.median(rho[r])), "rho_max": float(rho[r].max())})
        for lab, (a, b) in MARGINAL.items():
            for kind, mapname, n in (("best_clean", "clean", 1), ("median50_clean", "clean", 50), ("best", "full", 1),
                                     ("median50", "full", 50)):
                va, _ = tops[(a, mapname)]; vb, _ = tops[(b, mapname)]
                marg.append({**key, "marginal": lab, "kind": kind,
                             "price": _median_topn(vb, n) / _median_topn(va, n) - 1.0})
        for name in KEY_CONSTRAINTS:
            for kind, mapname, n in (("best_clean", "clean", 1), ("median50_clean", "clean", 50)):
                v0, r0 = tops[("F0", mapname)]; vk, rk = tops[(name, mapname)]
                q0, qk = _median_topn(v0, n), _median_topn(vk, n)
                rec = {**key, "constraint": name, "kind": kind, "unit": unit, "q_F0": q0, "q_K": qk,
                       "delta": qk - q0, "price": qk / q0 - 1.0}
                if n == 1:
                    rec.update({"F0_compound": u.formula_pretty.iat[r0[0]], "K_compound": u.formula_pretty.iat[rk[0]]})
                absinc.append(rec)
    return {"topn": pd.DataFrame(topn), "composition": pd.DataFrame(comp), "marginal": pd.DataFrame(marg),
            "absolute": pd.DataFrame(absinc)}


# ---------------------------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------------------------
def class_band() -> tuple[dict[str, tuple[float, float]], str]:
    """90 % interval of δ per bootstrap class: compound-based (COD, the COD module) when available, else elemental."""
    try:
        from shieldmap.analysis3 import cod
        bands = {c: tuple(cod.band(c)) for c in common.PRIMARY_CLASS_ORDER}
        return bands, "compound (COD, step 3)"
    except Exception as exc:                                         # COD not (yet) available
        from shieldmap.analysis import sensitivity
        x = sensitivity.bias_distribution("empirical")
        lo, hi = float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))
        return {c: (lo, hi) for c in common.PRIMARY_CLASS_ORDER}, f"elemental (Phase 1; COD unavailable: {exc!r})"


def tiers(q: np.ndarray, mask: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_tier: int = 3) -> np.ndarray:
    """Tier 1..max_tier (0 = deeper or outside mask). q, lo, hi are per-row arrays (n,).

    Tier t: among the remaining members, U* = min q(1+hi); members with q(1+lo) ≤ U* form the tier.
    """
    q = np.asarray(q, dtype=float)
    out = np.zeros(q.size, dtype=int)
    alive = np.asarray(mask, bool) & np.isfinite(q)
    lower = q * (1 + lo); upper = q * (1 + hi)
    for t in range(1, max_tier + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        u_star = upper[idx].min()
        mem = idx[lower[idx] <= u_star]
        out[mem] = t
        alive[mem] = False
    return out


def tier_table(max_tier: int = 3) -> tuple[pd.DataFrame, dict]:
    u = common.universe()
    bands, source = class_band()
    cls = common.universe_primary_class()
    lo = np.array([bands[c][0] for c in cls]); hi = np.array([bands[c][1] for c in cls])
    q = common.profiles("thickness")
    recs = []
    for sc in ("CS137", "CO60"):
        for vid in common.scenario_variants(sc)["variant_id"]:
            col = np.asarray(q[:, vid], dtype=float)
            for rule in common.RULES:
                base = common.base_mask(rule)
                for b in ("B0", "B4", "B5", "B6"):
                    m = base & common.bundle_mask(b)
                    t = tiers(col, m, lo, hi, max_tier)
                    for r in np.flatnonzero(t):
                        recs.append({"scenario": sc, "variant_id": int(vid), "rule": rule, "bundle": b,
                                     "tier": int(t[r]), "row": int(r), "q_cm": col[r],
                                     "q_lo": col[r] * (1 + lo[r]), "q_hi": col[r] * (1 + hi[r]),
                                     "bootstrap_class": cls[r]})
    df = pd.DataFrame(recs)
    df = df.merge(u[["row", "material_id", "formula_pretty", "density", "energy_above_hull", "theoretical",
                     "band_gap", "min_crustal_abundance_ppm"]], on="row", how="left")
    return df, {"band_source": source, "bands": bands}


# ---------------------------------------------------------------------------------------------
def run() -> dict:
    paths.ensure(OUT)
    tp = textbook_phases()
    tp.to_csv(OUT / "textbook_phases.csv", index=False)
    fp = frontier_prices()
    fp.to_parquet(OUT / "frontier_prices.parquet", index=False)
    hk = headline_kinds(fp)
    hk.to_csv(OUT / "headline_kinds.csv", index=False)
    af = artefact_frontiers(fp)
    af.to_csv(OUT / "artefact_frontiers.csv", index=False)
    tt, band = tier_table()
    tt.to_parquet(OUT / "tiers.parquet", index=False)
    common.write_json(OUT / "band_used.json", band)
    ts = (tt.groupby(["scenario", "variant_id", "rule", "bundle", "tier"])
            .agg(n=("row", "size"), n_formulas=("formula_pretty", "nunique"),
                 q_min=("q_cm", "min"), q_max=("q_cm", "max")).reset_index())
    ts.to_csv(OUT / "tier_summary.csv", index=False)
    ex = extras_tables()
    for k, f in EXTRA_FILES.items():
        ex[k].to_csv(OUT / f, index=False)
    for p in OUT.glob("_*"):                                         # hand-made logs are not pipeline outputs
        p.unlink()
    summ = _summary(fp, hk, ts, tp, band)
    summ.update(_extras_summary(ex))
    common.write_json(OUT / "summary.json", summ)
    return summ


EXTRA_FILES = {"topn": "topn_prices.csv", "composition": "top50_composition.csv", "marginal": "marginal_prices.csv",
               "absolute": "absolute_increments.csv"}


def _rng(x) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(x.min()), float(x.max())] if x.size else [float("nan"), float("nan")]


def _extras_summary(ex: dict) -> dict:
    """Headline numbers of the hardening-revision tables (R2 primary variants; percentages)."""
    out = {}
    tn = ex["topn"]
    rad = tn[tn.source.isin(["CS137", "CO60"]) & (tn.basis == "thickness")]
    out["topn_thickness_pct"] = {f"{sc}|{c}|{mp}": {int(n): [round(100 * v, 2) for v in _rng(g[g.N == n].price)]
                                                    for n in TOPN}
                                 for (sc, c, mp), g in rad[rad.constraint.isin(["RAD", "B4", "GAP(3.1)", "B5", "B6"])]
                                 .groupby(["source", "constraint", "map"])}
    cp = ex["composition"]
    f0 = cp[(cp.set == "F0") & (cp.source != "mono")]
    out["top50_F0_composition"] = {f"{sc}|{b}|{mp}": {k: _rng(g[k]) for k in ("share_radioactive", "share_theoretical",
                                                                           "share_ehull_gt_0.05", "share_artefact",
                                                                           "rho_median")}
                                   for (sc, b, mp), g in f0.groupby(["source", "basis", "map"])}
    mg = ex["marginal"]
    out["marginal_pct"] = {f"{sc}|{b}|{m}": {k: [round(100 * v, 2) for v in _rng(g[g.kind == k].price)]
                                             for k in ("best_clean", "median50_clean")}
                           for (sc, b, m), g in mg[mg.source != "mono"].groupby(["source", "basis", "marginal"])}
    ab = ex["absolute"]
    a = ab[(ab.kind == "best_clean") & (ab.source != "mono")]
    out["absolute_best_clean"] = {f"{sc}|{b}|T={t:g}|{c}": {"q_F0": float(g.q_F0.iloc[0]), "q_K": float(g.q_K.iloc[0]),
                                                           "delta": float(g.delta.iloc[0]), "unit": g.unit.iloc[0],
                                                           "F0": g.F0_compound.iloc[0], "K": g.K_compound.iloc[0]}
                                  for (sc, b, t, c), g in a[a.source != "DIAG"].groupby(["source", "basis", "T", "constraint"])}
    d = a[a.source == "DIAG"]
    out["absolute_best_clean_DIAG_areal_g_cm2"] = {c: {"delta": _rng(g.delta), "price_pct": [100 * v for v in _rng(g.price)]}
                                                   for c, g in d.groupby("constraint")}
    return out


def _summary(fp, hk, ts, tp, band) -> dict:
    prim = fp[fp.primary & (fp.rule == common.PRIMARY_RULE)]
    out = {"band_source": band["band_source"], "headline_kind": common.HEADLINE_KIND,
           "n_price_cells": int(len(fp)),
           "share_best_cells_flagged": float(fp[fp.kind == "best"].artefact_any.mean()),
           "flagged_by_scenario_basis_R2_primary": (
               prim[prim.kind == "best"].groupby(["scenario", "basis"]).artefact_any.mean().round(4)
               .reset_index().to_dict("records")),
           "textbook_not_R2": tp.loc[~tp.same_as_R2, ["formula", "material_id", "R2_material_id",
                                                     "density_ratio_ambient_vs_R2"]].to_dict("records")}
    hl = prim[prim.kind.isin(common.KINDS6)]
    out["headline_R2_primary_pct"] = {
        f"{sc}|{b}|{c}": {k: round(100 * float(g[g.kind == k].price.median()), 2) for k in common.KINDS6}
        for (sc, b, c), g in hl[hl.constraint.isin(["PB", "RAD", "STAB(0.05)", "EXP", "GAP(3.1)", "ABUND(1)", "B4", "B5",
                                                     "B6"])].groupby(["scenario", "basis", "constraint"])}
    t1 = ts[(ts.tier == 1) & (ts.rule == common.PRIMARY_RULE)]
    out["tier1_size_R2"] = t1.groupby(["scenario", "bundle"]).n.agg(["min", "median", "max"]).reset_index().to_dict("records")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str))
