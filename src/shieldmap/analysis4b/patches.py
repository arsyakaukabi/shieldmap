"""Re-presentation patches on the reference set.

Every function here re-presents numbers stage 4a already computed, or recomputes them with the stage 4a code
on the B4-internal reference; nothing new is estimated.

  tl_axis(): lead-free price in the 88.5-90.5 keV window with Tl also excluded (the Tl axis of H63),
                 narrow beam, same certified routine as analysis4.edge4.
  h72_noh(): extra areal mass Pb needs relative to the best B4 compound, H-containing compounds excluded.
  fig2_data()    Fig. 2 price curves (B4-internal, best_clean, R2, measured-density thickness); areal mass from
                 the H-excluded variant at and above the energy where NH₃·H₂O enters (800 keV).
  fig3_data(): GAP(3.1)|B4 on both bases across energy (B4-internal, measured-density point),
                 split into a density factor and a composition factor; Monte Carlo points.
  fig4_data(): diagnostic advantage per variant (narrow, first-order corrected) plus the Monte Carlo
                 estimators at the two simulated variants; families at the headline variant.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import edgewin as ew
from shieldmap.analysis4 import common, refsets

P4A = common.P4
S4A = P4A / "S"
P4B = paths.DATA / "phase4b"
R4B = paths.ROOT / "reports" / "phase4b"
LABELS = ("PB|B4", "STAB|B4", "EXP|B4", "GAP(3.1)|B4", "ABUND(1)|B4", "RAD|B4")
AMBIENT_H = ("H5NO",)                      # NH₃·H₂O (MP formula H5NO): liquid under ambient conditions
HEADLINE_DIAG_VARIANT = 13                 # RQR6, 12°, 0.35 mm Pb
MC_DIAG = {"c3_rqr5": (3, "advantage Bi/LaBi − 1 (R3.1 knife edge)", "LaBi"),
           "c4_rqr8": (22, "advantage Bi/Pr3OsI3 − 1", "Pr3OsI3")}
MC_GAP = {"c1_cs137": ("CS137|v38", "GAP(3.1) given B4 thickness price (Cs-137)"),
          "c2_co60": ("CO60|v42", "GAP(3.1) given B4 thickness price (Co-60)")}
MC_EST = ("narrow", "fl", "broad")


def noh_from_kev() -> float:
    """Switch point: the lowest single-line energy at which NH₃·H₂O is a reference compound (either side) of a
    plotted B4-internal areal price (the figure review found it enters at 800 keV, below 1 MeV)."""
    b = _bands()
    b = b[(b.scenario == "MONO") & (b.basis == "areal") & b.label.isin(LABELS)]
    h = b[b.formula_den.isin(AMBIENT_H) | b.formula_num.isin(AMBIENT_H)]
    return float(h.energy_keV.min())


def _bands(practical: str = "primary") -> pd.DataFrame:
    if practical == "primary":
        b = pd.read_parquet(S4A / "bands.parquet")
        b = b[b.reference == "B4-internal"]
    else:
        b = pd.read_parquet(S4A / "practical_bands.parquet")
        b = b[b.practical == practical]
    return b[b.artefact_def == common.PRIMARY_ARTEFACT]


def tl_axis() -> pd.DataFrame:
    """PB|B4 in the Pb K-edge window with Tl excluded from the practical set, narrow beam (edge4 routine)."""
    u = c3.universe()
    e = c3.energy_keV()
    sets = refsets.practical_sets(headroom=False)
    base = c3.base_mask("R2") & sets["B4-PB"] & ~ew._contains(81)
    km = ~ew._contains(82)
    art = common.artefact_mask()
    caches: dict = {}
    recs = []
    for j in np.flatnonzero((e >= 88.5 - 1e-9) & (e <= 90.5 + 1e-9)):
        for T in ew.T_LEVELS:
            for kind, v in ew.price_cell(base, km, int(j), T, "areal", None, caches, art).items():
                recs.append({"energy_keV": float(e[j]), "T": T, "model": "narrow", "kind": kind, **v,
                             "excluded": "Tl", "base_formula": u.formula_pretty.iat[v["row_base"]],
                             "constrained_formula": u.formula_pretty.iat[v["row_constrained"]],
                             "base_mpid": u.material_id.iat[v["row_base"]],
                             "constrained_mpid": u.material_id.iat[v["row_constrained"]]})
    out = pd.DataFrame(recs)
    out.to_csv(P4B / "pb_window_b4_noTl.csv", index=False)
    return out


def h72_noh() -> dict:
    """analysis4.headline.misc_rows (H72), with and without H-containing compounds; per-energy table saved."""
    u = c3.universe()
    pb_row = int(np.flatnonzero((u.formula_pretty.to_numpy() == "Pb") & c3.base_mask("R2"))[0])
    cols = refsets.columns()
    mono = cols[(cols.kind == "mono") & cols.coherent & (cols.energy_keV >= 100)]
    q = c3.mono_requirement("areal", True, e_idx=mono.e_idx.to_numpy())
    b4 = refsets.practical_sets(headroom=False)["B4"] & c3.base_mask("R2") & common.clean_mask()
    noh = b4 & ~c3.core.contains("H")
    out = {"energy_keV": mono.energy_keV.to_numpy()}
    for name, m in (("all", b4), ("noH", noh)):
        qq = np.where(m[:, None], q, np.inf)
        best = qq.min(axis=0)
        arg = qq.argmin(axis=0)
        out[f"best_{name}"] = u.formula_pretty.to_numpy()[arg]
        out[f"ratio_{name}"] = q[pb_row] / best - 1.0
        out[f"delta_g_cm2_{name}"] = (q[pb_row] - best) * math.log(100)
    t = pd.DataFrame(out)
    t.to_csv(P4B / "h72_pb_vs_b4_mass.csv", index=False)
    r = t.ratio_noH.to_numpy()
    return {"median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()),
            "e_min": float(t.energy_keV[r.argmin()]), "e_max": float(t.energy_keV[r.argmax()]),
            "delta_median_g_cm2": float(np.median(t.delta_g_cm2_noH)),
            "median_all": float(np.median(t.ratio_all)), "max_all": float(t.ratio_all.max()),
            "frontier_noH_at_max_all": str(t.best_noH[t.ratio_all.argmax()]),
            "frontier_all_at_max_all": str(t.best_all[t.ratio_all.argmax()]), "pb_mpid": u.material_id.iat[pb_row]}


# --------------------------------------------------------------------------------------------- Fig. 2
def fig2_data() -> pd.DataFrame:
    """Price curves on the MONO grid: thickness (measured-density point) and areal mass; areal from the
    H-excluded variant at and above noh_from_kev() . Band = R1-R3 × kinds × coherent (stage 4a credible band)."""
    keep = ["label", "basis", "density", "energy_keV", "price", "band_lo", "band_hi", "formula_den", "formula_num"]
    b = _bands()
    b = b[(b.scenario == "MONO") & b.label.isin(LABELS)]
    th = b[(b.basis == "thickness") & (b.density == "measured")][keep].assign(variant="primary")
    NOH_FROM_KEV = noh_from_kev()
    ar = b[(b.basis == "areal") & (b.energy_keV < NOH_FROM_KEV)][keep].assign(variant="primary")
    n = _bands("noH")
    n = n[(n.scenario == "MONO") & n.label.isin(LABELS) & (n.basis == "areal") & (n.energy_keV >= NOH_FROM_KEV)]
    out = pd.concat([th, ar, n[keep].assign(variant="noH")], ignore_index=True)
    out["label"] = out.label.astype(str)
    out = out.sort_values(["label", "basis", "energy_keV"]).reset_index(drop=True)
    out.to_csv(P4B / "fig2_prices.csv", index=False)
    return out


# --------------------------------------------------------------------------------------------- Fig. 3
def _mc(case_id: str, metric: str) -> pd.DataFrame:
    m = pd.read_csv(P4A / "mc" / "metrics.csv")
    return m[(m.case_id == case_id) & (m.metric == metric)].set_index("estimator")


def fig3_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GAP(3.1)|B4 across energy, B4-internal, measured-density point. For the thickness pair
    (reference c0 -> GAP frontier cK): 1 + P_t = [ρ_eff(c0)/ρ_eff(cK)] × [q(cK)/q(c0)], with ρ_eff = ρ_MP/(1 + δ_point)
    (the density factor) and q the areal-mass requirement of the same pair (the composition factor)."""
    u = c3.universe()
    idx = pd.Series(np.arange(len(u)), index=u.material_id.to_numpy())
    idx = idx[~idx.index.duplicated()]
    rho_eff = u.density.to_numpy() / refsets.measured_scale()
    b = _bands()
    g = b[(b.label == "GAP(3.1)|B4") & b.scenario.isin(["MONO", "CS137", "CO60"])]
    th = g[(g.basis == "thickness") & (g.density == "measured")].set_index("key")
    NOH_FROM_KEV = noh_from_kev()
    gn = _bands("noH")
    gn = gn[(gn.label == "GAP(3.1)|B4") & (gn.basis == "areal")]
    hi = lambda x: (x.scenario == "CO60") | ((x.scenario == "MONO") & (x.energy_keV >= NOH_FROM_KEV))
    ar = pd.concat([g[(g.basis == "areal") & ~hi(g) & g.scenario.isin(["MONO", "CS137"])].assign(mass_variant="primary"),
                    gn[hi(gn)].assign(mass_variant="noH")]).set_index("key")
    cols = refsets.columns().set_index("key")
    rows = []
    for key, t in th.iterrows():
        a = ar.loc[key]
        r0, rk = int(idx[t.mpid_den]), int(idx[t.mpid_num])
        c = cols.loc[key]
        c = c[c.coherent].iloc[0] if isinstance(c, pd.DataFrame) else c
        if t.scenario == "MONO":
            qa = c3.mono_requirement("areal", True, e_idx=np.array([int(c.e_idx)]))[[r0, rk], 0]
        else:
            qa = refsets.requirement(cols.reset_index()[cols.reset_index().col == c.col], "areal", "any")[[r0, rk], 0]
        dens = rho_eff[r0] / rho_eff[rk]
        comp = qa[1] / qa[0]
        rows.append({"key": key, "scenario": t.scenario, "energy_keV": t.energy_keV,
                     "P_thickness": t.price, "P_thickness_lo": t.band_lo, "P_thickness_hi": t.band_hi,
                     "P_mass": a.price, "P_mass_lo": a.band_lo, "P_mass_hi": a.band_hi, "mass_variant": a.mass_variant,
                     "thk_ref": t.formula_den, "thk_gap": t.formula_num, "mass_ref": a.formula_den, "mass_gap": a.formula_num,
                     "rho_eff_ref": rho_eff[r0], "rho_eff_gap": rho_eff[rk],
                     "density_factor": dens, "composition_factor": comp,
                     "identity_residual": dens * comp - (1 + t.price)})
    d = pd.DataFrame(rows).sort_values(["scenario", "energy_keV"]).reset_index(drop=True)
    if d.identity_residual.abs().max() > 1e-6:
        raise RuntimeError(f"Fig. 3 decomposition does not close: {d.identity_residual.abs().max():.2e}")
    d.to_csv(P4B / "fig3_gap_b4internal.csv", index=False)
    mc = []
    for cid, (key, metric) in MC_GAP.items():
        m = _mc(cid, metric)
        for est in MC_EST + ("T_tablec", "T_endf_narrow"):
            mc.append({"case_id": cid, "key": key, "num": m.loc[est, "num"], "den": m.loc[est, "den"], "estimator": est,
                       "price": float(m.loc[est, "value"]), "se": float(m.loc[est, "se"]) if pd.notna(m.loc[est, "se"]) else np.nan})
    mc = pd.DataFrame(mc)
    mc.to_csv(P4B / "fig3_mc_points.csv", index=False)
    return d, mc


# --------------------------------------------------------------------------------------------- Fig. 4
def fig4_data() -> dict:
    """(b) advantage of the best lanthanide over the best textbook material per DIAG variant: narrow beam and
    first-order corrected (stage 4a xs table, XCOM, map T*), plus Monte Carlo estimators at variants 3 and 22.
    (a) families and textbook materials relative to the B4 areal-mass frontier at the headline variant."""
    from shieldmap.analysis4 import figures4
    x = pd.read_csv(P4A / "xs" / "diag_advantage_by_variant.csv")
    x = x[(x.lib == "xcom") & (x.tstar == "XCOM Pb (map)")]
    keep = ["variant_id", "quality", "anode_angle", "pb_mm", "adv_narrow", "best_ln_narrow", "best_textbook_narrow",
            "adv_corrected_estimate", "best_ln_corrected", "best_textbook_corrected"]
    v = x[keep].sort_values("variant_id").reset_index(drop=True)
    v.to_csv(P4B / "fig4b_advantage_by_variant.csv", index=False)
    mc = []
    for cid, (vid, metric, ln) in MC_DIAG.items():
        m = _mc(cid, metric)
        for est in MC_EST + ("T_tablec", "T_firstorder"):
            mc.append({"case_id": cid, "variant_id": vid, "lanthanide": ln, "textbook": "Bi", "estimator": est,
                       "advantage": float(m.loc[est, "value"]),
                       "se": float(m.loc[est, "se"]) if pd.notna(m.loc[est, "se"]) else np.nan})
    mc = pd.DataFrame(mc)
    mc.to_csv(P4B / "fig4b_mc_points.csv", index=False)
    d = figures4.fig5_inputs()
    f = d["fams"][d["fams"].variant_id == HEADLINE_DIAG_VARIANT]
    f.to_csv(P4B / "fig4a_families_v13.csv", index=False)
    t = d["tbv"][d["tbv"].variant_id == HEADLINE_DIAG_VARIANT]
    t.to_csv(P4B / "fig4a_textbook_v13.csv", index=False)
    pv = d["pv"][d["pv"].variant_id == HEADLINE_DIAG_VARIANT]
    pv.to_csv(P4B / "fig4a_plateau_v13.csv", index=False)
    return {"by_variant": v, "mc": mc, "families": f, "textbook": t, "plateau": pv}


def run() -> dict:
    paths.ensure(P4B)
    paths.ensure(R4B)
    res = {}
    res["noh_from_keV"] = noh_from_kev()
    res["fig2"] = len(fig2_data())
    d3, mc3 = fig3_data()
    res["fig3"] = {"n": len(d3), "mc": len(mc3)}
    f4 = fig4_data()
    res["fig4"] = {k: len(v) for k, v in f4.items()}
    res["h72_noh"] = h72_noh()
    tl = tl_axis()
    w = tl[(tl.kind == "best_clean")]
    i = w.price.idxmax()
    res["tl_axis"] = {"max_price": float(w.price.max()), "at_keV": float(w.loc[i, "energy_keV"]), "T": float(w.loc[i, "T"]),
                      "pair": f"{w.loc[i, 'base_formula']} → {w.loc[i, 'constrained_formula']}",
                      "band_lo": float(tl.price.min()), "band_hi": float(tl.price.max()),
                      "certified": bool(tl.certified.all())}
    common.write_json(P4B / "patches.json", res)
    return res
