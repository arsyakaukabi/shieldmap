"""Headline table v3 (data/phase4b/headline_table_v3.csv, reports/phase4b/headline_table_v3.md).

The stage 4a table (every ID kept, values unchanged) plus:
  quoted_in       main / supplementary / not quoted
  manuscript_id   where the number appears (T1-r3, Fig2b, S6.2, ...); planned initially, replaced by the usage
                  that paper/check_numbers.py extracts from the built manuscript when that file exists
  statement_v3    the revised statement
  sensitivity_values_v3, decision, v3_origin, v3_note
and new rows H133+ that re-present existing data on the reference (H-excluded Co-60 mass prices and
H72; Tl axis; Monte Carlo estimators shown in Figs. 3 and 4; the Fig. 3 decomposition at Cs-137/Co-60).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from shieldmap.analysis4 import common
from shieldmap.analysis4b import patches

P4A = common.P4
P4B = patches.P4B
R4B = patches.R4B
USAGE = patches.paths.ROOT / "paper" / "build" / "number_usage.json"
MC_NOTE = ("[first-order model: the Monte Carlo spot check finds escaping fluorescence under-counted by 5–30 %; "
           "the Monte Carlo values lead in the main text (D2)]")

RAD = ["H01", "H02", "H13", "H14", "H25", "H26", "H41", "H47", "H48", "H49", "H50"]
FIRST_ORDER = ["H64", "H82", "H83", "H85", "H86", "H130", "H131", "H132"]
CO60_MASS_PRIMARY = {"H16": "H133", "H18": "H134", "H20": "H135", "H22": "H136", "H24": "H137", "H14": "H138"}

# planned placement; paper/check_numbers.py records the actual placement after the build
PLAN_MAIN = {
    "H03": "T1", "H04": "T1", "H05": "T1", "H06": "T1", "H07": "T1", "H08": "T1", "H09": "T1; Fig3a", "H10": "T1; Fig3a",
    "H11": "T1", "H12": "T1", "H15": "T1", "H17": "T1", "H19": "T1", "H21": "T1; Fig3a", "H23": "T1",
    "H133": "T1", "H134": "T1", "H135": "T1", "H136": "T1; Fig3a", "H137": "T1",
    "H27": "T1", "H28": "T1", "H29": "T1", "H30": "T1", "H31": "T1", "H32": "T1", "H33": "T1", "H34": "T1",
    "H35": "T1", "H36": "T1", "H42": "Sec3.2", "H43": "Sec3.2", "H44": "Sec3.2", "H45": "Sec3.2", "H46": "Sec3.2",
    "H56": "Sec3.2", "H63": "Sec3.2", "H140": "Sec3.2", "H121": "Sec3.2", "H139": "Sec3.2",
    "H51": "Sec3.3", "H73": "Sec3.3", "H109": "Fig3a", "H117": "Fig3a", "H146": "Fig3a", "H147": "Fig3a",
    "H148": "Fig3a", "H149": "Fig3a", "H150": "Fig3b", "H151": "Fig3b", "H152": "Fig3b", "H153": "Fig3b",
    "H84": "Sec3.4", "H112": "Fig4b", "H118": "Fig4b", "H141": "Fig4b", "H142": "Fig4b", "H143": "Fig4b",
    "H144": "Fig4b", "H145": "Abstract; Sec3.4", "H116": "Sec3.4", "H120": "Sec3.4",
    "H79": "Fig1b", "H80": "Sec3.1", "H87": "Sec2", "H75": "Sec2", "H77": "Sec2",
    "H66": "T2", "H67": "T2", "H68": "T2", "H69": "T2", "H70": "T2", "H71": "T2",
    "H89": "T2", "H90": "T2", "H91": "T2", "H92": "T2", "H93": "T2", "H94": "T2", "H124": "Sec3.3",
    **{f"H{i}": "T2" for i in range(154, 168)},                                                     # the rows_tables
    # revision
    **{f"H{i}": "Fig4a; Fig4b; Sec3.4" for i in range(180, 189)},                                   # E2 blend line
    **{f"H{i}": "Abstract; Sec3.4; Fig4c; TS4" for i in range(189, 200)},                           # E3 like-for-like MC
    **{f"H{i}": "Sec3.4; T1 (ambient tier)" for i in range(200, 220)},                              # E5 ambient tier
    **{f"H{i}": "Sec3.3; T2 footnote" for i in range(220, 242)},                                    # E7 GAP sweep, cubic
    **{f"H{i}": "Sec2.6" for i in range(242, 246)},                                                 # E12 under-count
    **{f"H{i}": "Fig4c caption" for i in range(252, 258)},                                          # O12 estimators
}
NOT_QUOTED = {"H65": "F0 element-exclusion row; the F0 version is retired (D4)",
              "H74": "MP-density twin of H73; the measured-density point is primary"}


def _usage() -> dict | None:
    if USAGE.exists():
        return json.loads(USAGE.read_text(encoding="utf-8"))
    return None


def _row(rid, statement, value, lo, hi, unit, band_type, basis, refset, refc, source_file, source_key, *,
         delta=np.nan, delta_unit="", delta_band="", delta_note="", kind="best_clean", density="n/a",
         fluor="no (narrow beam)", sens="", mc_status="", dossier="", decision="", note="") -> dict:
    pct = unit == "%"
    return {"id": rid, "block": "headline", "dossier_result": dossier, "statement": statement, "quantity": "",
            "value": value / 100 if pct else value, "value_pct": value if pct else np.nan, "unit": unit,
            "band_lo_pct": lo, "band_hi_pct": hi, "band_type": band_type, "density_band_pct": "",
            "delta_abs": delta, "delta_unit": delta_unit, "delta_band": delta_band, "delta_note": delta_note,
            "basis": basis, "frontier_kind": kind, "reference_set": refset, "reference_compounds": refc,
            "reference_mpids": "", "density_source": density, "fluorescence_corrected": fluor,
            "sensitivity_axes": "", "sensitivity_values": sens, "band_lo_set_by": "", "band_hi_set_by": "",
            "source_file": source_file, "source_key": source_key, "artefact_reference": False, "xs_label": "",
            "reference_practical": "", "xs_sensitivity": "", "mc_status": mc_status,
            "v3_origin": "phase4b-step1", "decision": decision, "v3_note": note}


def new_rows(pat: dict) -> list[dict]:
    rows = []
    # Co-60 areal prices on the H-excluded variant
    pb = pd.read_parquet(P4A / "S" / "practical_bands.parquet")
    pb = pb[(pb.practical == "noH") & (pb.artefact_def == common.PRIMARY_ARTEFACT) & (pb.key == "CO60|v42")
            & (pb.basis == "areal")].set_index("label")
    text = {"PB|B4": "lead-free (PB) given the rest of B4", "STAB|B4": "stability STAB(0.05) given the rest of B4",
            "EXP|B4": "ICSD match (EXP) given the rest of B4", "GAP(3.1)|B4": "band gap ≥ 3.1 eV (GAP) given B4",
            "ABUND(1)|B4": "scarcity proxy (crustal abundance) ≥ 1 ppm given B4",
            "RAD|B4": "non-radioactive (RAD) given the rest of B4 (actinide headroom; Supplementary only)"}
    for (old, rid), lab in zip(CO60_MASS_PRIMARY.items(), ["PB|B4", "STAB|B4", "EXP|B4", "GAP(3.1)|B4", "ABUND(1)|B4", "RAD|B4"]):
        r = pb.loc[lab]
        rows.append(_row(rid, f"Price of {text[lab]}, Co-60 (T = 0.01), areal basis, H-containing compounds excluded (D3; replaces {old} in the main text)",
                         100 * r.price, 100 * r.band_lo, 100 * r.band_hi, "%",
                         "credible (R1–R3 × 3 kinds × coherent; Phase 4a flag), H-excluded variant", "areal",
                         f"B4-internal ({lab}), H-containing compounds excluded", f"{r.formula_den} → {r.formula_num}",
                         "data/phase4a/S/practical_bands.parquet", f"practical=noH; key=CO60|v42; label={lab}; basis=areal",
                         delta=float(r.delta_abs), delta_unit="g/cm2",
                         delta_band=f"{r.delta_band_lo:.4g}–{r.delta_band_hi:.4g} g/cm2", delta_note="at T = 0.01",
                         density="n/a (areal)", dossier="R2.6" if lab in ("STAB|B4", "EXP|B4") else "R2.1" if lab == "PB|B4" else "R2.2" if lab.startswith("GAP") else "R2.3",
                         decision="D3", note=f"{old} (primary set) has NH₃·H₂O as the Co-60 mass frontier"))
    h = pat["h72_noh"]
    rows.append(_row("H139", "Extra areal mass Pb needs relative to the best B4 (lead-free practical) compound, H-containing compounds excluded, single lines 100 keV–10 MeV (median over energies; band = range over energies)",
                     100 * h["median"], 100 * h["min"], 100 * h["max"], "%", "range over energies", "areal",
                     "B4 (best_clean), H-containing compounds excluded", f"Pb ({h['pb_mpid']}) vs B4 frontier without H",
                     "data/phase4b/h72_pb_vs_b4_mass.csv", "ratio_noH (analysis4b.patches.h72_noh)",
                     delta=h["delta_median_g_cm2"], delta_unit="g/cm2", delta_note="median extra areal mass at T = 0.01",
                     dossier="R2.1", decision="D3", note=f"re-states H72 (unfiltered: median {100 * h['median_all']:.1f} %, max {100 * h['max_all']:.1f} % where the frontier is NH₃·H₂O)"))
    t = pat["tl_axis"]
    rows.append(_row("H140", "Lead-free (PB) given B4 with Tl also excluded (Tl axis of H63), areal mass, 88.5–90.5 keV window, narrow beam (maximum over window nodes and T 0.1/0.01; band = window energies × kinds × T)",
                     100 * t["max_price"], 100 * t["band_lo"], 100 * t["band_hi"], "%",
                     "window energies × kinds (best, best_clean, median50) × T 0.1/0.01 (narrow)", "areal",
                     "B4-internal (PB|B4), Tl excluded", t["pair"], "data/phase4b/pb_window_b4_noTl.csv",
                     "model=narrow; kind=best_clean; 88.5–90.5 keV; Tl excluded", dossier="R2.1", decision="D5",
                     note=f"maximum at {t['at_keV']:g} keV, T = {t['T']:g}; all cells certified: {t['certified']}"))
    m = pd.read_csv(P4A / "mc" / "metrics.csv")

    def mc(rid, cid, metric, est, stmt, basis, refc, dossier, decision="D2"):
        x = m[(m.case_id == cid) & (m.metric == metric) & (m.estimator == est)].iloc[0]
        se = 100 * float(x.se)
        rows.append(_row(rid, stmt, 100 * float(x.value), 100 * float(x.value) - 1.96 * se, 100 * float(x.value) + 1.96 * se,
                         "%", "Monte Carlo statistical 95 % interval (± 1.96 SE)", basis, refc, refc,
                         "data/phase4a/mc/metrics.csv", f"case_id={cid}; metric={metric}; estimator={est}",
                         fluor="Monte Carlo (fluorescence transported)" if est != "narrow" else "Monte Carlo narrow beam (scored uncollided)",
                         mc_status="Monte Carlo result", dossier=dossier, decision=decision))
    adv3, adv4 = "advantage Bi/LaBi − 1 (R3.1 knife edge)", "advantage Bi/Pr3OsI3 − 1"
    mc("H141", "c3_rqr5", adv3, "narrow", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c3_rqr5 (RQR5, 12°, 0.25 mm Pb): advantage Bi/LaBi − 1, narrow-beam estimator", "areal", "Bi vs LaBi", "R3.1")
    mc("H142", "c3_rqr5", adv3, "broad", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c3_rqr5 (RQR5, 12°, 0.25 mm Pb): advantage Bi/LaBi − 1, broad-beam estimator", "areal", "Bi vs LaBi", "R3.1")
    mc("H143", "c4_rqr8", adv4, "narrow", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c4_rqr8 (RQR8, 12°, 0.35 mm Pb): advantage Bi/Pr3OsI3 − 1, narrow-beam estimator", "areal", "Bi vs Pr3OsI3", "R3.1")
    mc("H144", "c4_rqr8", adv4, "broad", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c4_rqr8 (RQR8, 12°, 0.35 mm Pb): advantage Bi/Pr3OsI3 − 1, broad-beam estimator", "areal", "Bi vs Pr3OsI3", "R3.1")
    sel = m[((m.case_id == "c3_rqr5") & (m.metric == adv3) | (m.case_id == "c4_rqr8") & (m.metric == adv4)) & m.estimator.isin(["fl", "broad"])]
    rows.append(_row("H145", "Monte Carlo advantage of the best lanthanide compound over Bi (the best textbook material) on areal mass, fluorescence-inclusive and broad-beam estimators, cases c3_rqr5 (LaBi) and c4_rqr8 (Pr3OsI3): range (value = lower end)",
                     100 * sel.value.min(), 100 * sel.value.min(), 100 * sel.value.max(), "%",
                     "range over the two cases × estimators fl/broad", "areal", "B4 lanthanide vs Bi", "LaBi, Pr3OsI3 vs Bi",
                     "data/phase4a/mc/metrics.csv", "cases c3_rqr5/c4_rqr8; estimators fl, broad",
                     fluor="Monte Carlo (fluorescence transported)", mc_status="Monte Carlo result", dossier="R3.1", decision="D2",
                     note="summarises H112, H118, H142, H144"))
    g1, g2 = "GAP(3.1) given B4 thickness price (Cs-137)", "GAP(3.1) given B4 thickness price (Co-60)"
    mc("H146", "c1_cs137", g1, "broad", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c1_cs137: GAP(3.1) given B4 thickness price (Cs-137), broad-beam estimator (build-up included)", "thickness", "BiOF vs Ir", "R2.2", "")
    mc("H147", "c2_co60", g2, "broad", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c2_co60: GAP(3.1) given B4 thickness price (Co-60), broad-beam estimator (build-up included)", "thickness", "HfO2 vs Ir", "R2.2", "")
    mc("H148", "c1_cs137", g1, "narrow", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c1_cs137: GAP(3.1) given B4 thickness price (Cs-137), narrow-beam estimator", "thickness", "BiOF vs Ir", "R2.2", "")
    mc("H149", "c2_co60", g2, "narrow", "Monte Carlo (OpenMC, ENDF/B-VIII.0), case c2_co60: GAP(3.1) given B4 thickness price (Co-60), narrow-beam estimator", "thickness", "HfO2 vs Ir", "R2.2", "")
    f3 = pd.read_csv(P4B / "fig3_gap_b4internal.csv").set_index("key")
    for rid, key, fac, name in (("H150", "CS137|v38", "density_factor", "density factor ρ(ref)/ρ(GAP frontier)"),
                                ("H151", "CS137|v38", "composition_factor", "composition factor q(GAP frontier)/q(ref)"),
                                ("H152", "CO60|v42", "density_factor", "density factor ρ(ref)/ρ(GAP frontier)"),
                                ("H153", "CO60|v42", "composition_factor", "composition factor q(GAP frontier)/q(ref)")):
        r = f3.loc[key]
        src = "Cs-137" if key.startswith("CS") else "Co-60"
        rows.append(_row(rid, f"GAP(3.1) given B4 thickness price at {src} (T = 0.01), measured-density point: {name}; 1 + P_thickness = density factor × composition factor",
                         float(r[fac]), np.nan, np.nan, "ratio", "exact decomposition of the headline pair (no band)", "thickness",
                         "B4-internal (GAP(3.1)|B4)", f"{r.thk_ref} → {r.thk_gap}", "data/phase4b/fig3_gap_b4internal.csv",
                         f"key={key}; column={fac}", density="measured-density point", dossier="R2.2",
                         note=f"ρ_eff {r.rho_eff_ref:.3f} / {r.rho_eff_gap:.3f} g/cm³"))
    return rows


def build() -> pd.DataFrame:
    pat = json.loads((P4B / "patches.json").read_text(encoding="utf-8"))
    t = pd.read_csv(P4A / "headline_table.csv")
    t["v3_origin"] = "phase4a"
    t["decision"] = ""
    t["v3_note"] = ""
    from shieldmap.analysis4b import rows_tables                                                    # Table 2 rows H154+
    from shieldmap.analysis4b import rows_text                                                      # text rows H168+
    from shieldmap.analysis4b import rows_revision                                                  # revision 1: rows H180+
    t = pd.concat([t, pd.DataFrame(new_rows(pat)), pd.DataFrame(rows_tables.extra_rows()),
                   pd.DataFrame(rows_text.extra_rows()), pd.DataFrame(rows_revision.extra_rows())], ignore_index=True)
    t["statement_v3"] = t.statement
    t["sensitivity_values_v3"] = t.sensitivity_values.fillna("")
    ix = t.set_index("id").index
    loc = lambda i: int(np.flatnonzero(ix == i)[0])
    for i in RAD:
        j = loc(i)
        s = t.at[j, "statement"]
        if not s.startswith("Actinide headroom"):
            s = "Actinide headroom inside B4 (not an achievable alternative): " + s
        t.at[j, "statement_v3"] = s + " — Supplementary only (D1)"
        t.at[j, "decision"] = "D1"
    for i in FIRST_ORDER:
        j = loc(i)
        s = t.at[j, "statement"]
        t.at[j, "statement_v3"] = s if "first-order model" in s else f"{s} {MC_NOTE}"
        t.at[j, "decision"] = "D2"
    for old, new in CO60_MASS_PRIMARY.items():
        j = loc(old)
        t.at[j, "statement_v3"] = t.at[j, "statement"] + f" — primary set, mass frontier NH₃·H₂O; the main text quotes the H-excluded {new} (D3)"
        t.at[j, "decision"] = "D3"
    for old in ("H72", "H52", "H54", "H107", "H106"):
        j = loc(old)
        t.at[j, "decision"] = "D3"
    j = loc("H72")
    t.at[j, "statement_v3"] = t.at[j, "statement"] + " — Supplementary only (unfiltered frontier); the main text quotes the H-excluded H139 (D3)"
    j = loc("H65")
    t.at[j, "decision"] = "D4"
    j = loc("H63")
    tl = pat["tl_axis"]
    t.at[j, "sensitivity_values_v3"] = (t.at[j, "sensitivity_values"] + f"; Tl excluded (element-exclusion sensitivity, D5): "
                                       f"max {100 * tl['max_price']:.1f} % ({tl['pair']}) at {tl['at_keV']:g} keV, T = {tl['T']:g} (H140)")
    t.at[j, "statement_v3"] = t.at[j, "statement"] + "; the B4 frontier without Pb is Tl (Tl replaces Pb in the window, D5)"
    t.at[j, "decision"] = "D5"
    # placement
    usage = _usage()
    q, mid = [], []
    for r in t.itertuples():
        if usage is not None:
            u = usage.get(r.id, {})
            places = sorted(set(u.get("main", [])) | set(u.get("supplementary", [])))
            q.append("main" if u.get("main") else "supplementary" if u.get("supplementary") else "not quoted")
            mid.append("; ".join(places))
            continue
        if r.id in NOT_QUOTED:
            q.append("not quoted"); mid.append(NOT_QUOTED[r.id])
        elif r.id in PLAN_MAIN and r.id not in RAD:
            q.append("main"); mid.append(PLAN_MAIN[r.id] + " (planned)")
        else:
            q.append("supplementary"); mid.append("TS2 (full table)" + ("; S6.1 headroom" if r.id in RAD or r.block == "appendix" else ""))
    t["quoted_in"] = q
    t["manuscript_id"] = mid
    t["placement_source"] = "paper/build/number_usage.json" if usage is not None else "Step 1 plan"
    first = ["id", "block", "quoted_in", "manuscript_id", "decision", "v3_origin", "statement_v3"]
    t = t[first + [c for c in t.columns if c not in first]]
    return t


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}g}"


def write(t: pd.DataFrame) -> None:
    t.to_csv(P4B / "headline_table_v3.csv", index=False)
    L = ["# Headline table v3 (Phase 4b)", "",
         "Generated by `src/shieldmap/analysis4b/table_v3.py`; do not edit by hand. The Phase 4a table "
         "(`data/phase4a/headline_table.csv`) is kept row for row with its values unchanged. New rows H133+ re-present "
         "existing data (handover decisions D1–D5). Full columns: `data/phase4b/headline_table_v3.csv`.", "",
         f"Placement source: {t.placement_source.iat[0]}.", "",
         "| ID | quoted in | manuscript | decision | statement (v3) | value | band | Δ | MC |", "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        v = f"{_f(r.value_pct)} %" if r.unit == "%" else f"{_f(r.value, 4)} {r.unit}"
        b = f"{_f(r.band_lo_pct)}–{_f(r.band_hi_pct)}" if pd.notna(r.band_lo_pct) else ""
        d = f"{_f(r.delta_abs)} {r.delta_unit}" if pd.notna(r.delta_abs) else ""
        s = str(r.statement_v3).replace("|", "\\|")
        L.append(f"| {r.id} | {r.quoted_in} | {r.manuscript_id} | {r.decision} | {s} | {v} | {b} | {d} | {r.mc_status if pd.notna(r.mc_status) else ''} |")
    (R4B / "headline_table_v3.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run() -> dict:
    t = build()
    write(t)
    return {"rows": len(t), "new_rows": int((t.v3_origin == "phase4b-step1").sum()),
            "quoted": t.quoted_in.value_counts().to_dict()}
