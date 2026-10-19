"""Headline rows H168+ that the manuscript text needs and the table does not yet hold.

Re-presentation only; every value is read from existing stage 4a / outputs:

  H168-H172  textbook materials and single-lanthanide families against the B4 frontier at the diagnostic headline
             variant (13: RQR6, 12°, 0.35 mm Pb), narrow beam and first-order fluorescence model
             (data/phase4b/fig4a_textbook_v13.csv, fig4a_families_v13.csv; the Fig. 4a data of)
  H173       number of the 36 DIAG variants in which β-Sn is the closest textbook material under narrow beam
             (data/phase4b/fig4b_advantage_by_variant.csv)
  H174       Monte Carlo narrow-beam estimator against the analytic ENDF/B-VIII.0 transmission: chi-square over the
             78 slab points of the six cases (sum of z_narrow_vs_endf², data/phase4a/mc/points.csv)
  H175-H179  writing-review rows: the other end of the headline curves behind H45, H46, H51, H53 (bands.parquet)
             and P(best) of BiOF under rule R3 at Cs-137 (data/phase3/T/tier_summary.csv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis4 import common
from shieldmap.analysis4b import patches

NOTE = "text (Phase 4b step 4)"
P4B = patches.P4B
MC_CAVEAT = ("[first-order model: the Monte Carlo spot check finds escaping fluorescence under-counted by 5–30 %; "
             "the Monte Carlo values lead in the main text (D2)]")
V13 = "DIAG RQR6, 12°, 0.35 mm Pb target (variant 13)"


def _textbook_rows() -> list[dict]:
    from shieldmap.analysis4b.table_v3 import _row

    tb = pd.read_csv(P4B / "fig4a_textbook_v13.csv")
    fam = pd.read_csv(P4B / "fig4a_families_v13.csv")
    src_tb = "data/phase4b/fig4a_textbook_v13.csv"
    pct = lambda r: 100 * (float(r) - 1)
    before = tb[tb.state == "before"].set_index("formula")
    after = tb[tb.state == "after"].set_index("formula")
    if len(before) != 10 or len(after) != 10:
        raise RuntimeError("expected ten textbook materials per state")
    closest = before.ratio_to_B4_frontier.idxmin()
    if closest != "Sn":
        raise RuntimeError(f"closest textbook material under narrow beam is {closest}, not β-Sn")
    if after.ratio_to_B4_frontier.idxmin() != "Bi":
        raise RuntimeError("closest textbook material under the first-order model is not Bi")
    rows = [
        _row("H168", f"Areal mass above the B4 frontier of the ten textbook materials, {V13}, narrow beam: range over the "
                     f"materials (value = closest, β-Sn at its ambient phase; upper end {before.ratio_to_B4_frontier.idxmax()})",
             pct(before.ratio_to_B4_frontier.min()), pct(before.ratio_to_B4_frontier.min()),
             pct(before.ratio_to_B4_frontier.max()), "%", "range over the ten textbook materials", "areal",
             "B4 frontier (best_clean, R2)", "textbook materials vs B4 frontier", src_tb,
             "variant_id=13; state=before; column=ratio_to_B4_frontier", kind="best_clean", density="n/a (areal)",
             dossier="R3.2", note=NOTE),
        _row("H169", f"Areal mass above the B4 frontier of Bi, {V13}, narrow beam",
             pct(before.at["Bi", "ratio_to_B4_frontier"]), np.nan, np.nan, "%", "none (single value)", "areal",
             "B4 frontier (best_clean, R2)", "Bi vs B4 frontier", src_tb,
             "variant_id=13; state=before; formula=Bi; column=ratio_to_B4_frontier", density="n/a (areal)",
             dossier="R3.2", note=NOTE),
        _row("H170", f"Areal mass above the B4 frontier of Bi, {V13}, fluorescence-corrected (first order, E₁, K + L; "
                     f"closest textbook material in this state) {MC_CAVEAT}",
             pct(after.at["Bi", "ratio_to_B4_frontier"]), np.nan, np.nan, "%", "none (single value)", "areal",
             "B4 frontier (best_clean, R2), corrected", "Bi vs corrected B4 frontier", src_tb,
             "variant_id=13; state=after; formula=Bi; column=ratio_to_B4_frontier", density="n/a (areal)",
             fluor="yes (first order; no secondary fluorescence, no M lines)", dossier="R3.2", decision="D2", note=NOTE),
        _row("H171", f"Areal mass above the B4 frontier of β-Sn, {V13}, fluorescence-corrected (first order, E₁, K + L) "
                     f"{MC_CAVEAT}",
             pct(after.at["Sn", "ratio_to_B4_frontier"]), np.nan, np.nan, "%", "none (single value)", "areal",
             "B4 frontier (best_clean, R2), corrected", "β-Sn vs corrected B4 frontier", src_tb,
             "variant_id=13; state=after; formula=Sn; column=ratio_to_B4_frontier", density="n/a (areal)",
             fluor="yes (first order; no secondary fluorescence, no M lines)", dossier="R3.2", decision="D2", note=NOTE),
    ]
    ln = fam[(fam.level == "ln") & (fam.state == "before") & (fam.variant_id == 13) & ~fam.family.str.contains(r"\+")]
    if len(ln) != 14:
        raise RuntimeError(f"expected 14 single-lanthanide families, found {len(ln)}")
    worst = ln.loc[ln.best_ratio.idxmax()]
    rows.append(_row("H172", f"Areal mass above the B4 frontier of the best member of each of the 14 single-lanthanide "
                             f"families, {V13}, narrow beam: largest over the families ({worst.family}; band = range over "
                             f"the families)",
                     pct(ln.best_ratio.max()), pct(ln.best_ratio.min()), pct(ln.best_ratio.max()), "%",
                     "range over the 14 single-lanthanide families", "areal", "B4 frontier (best_clean, R2)",
                     "lanthanide families vs B4 frontier", "data/phase4b/fig4a_families_v13.csv",
                     "level=ln; state=before; variant_id=13; single-lanthanide families; column=best_ratio",
                     density="n/a (areal)", dossier="R3.3", note=NOTE))
    return rows


def _variant_count_row() -> dict:
    from shieldmap.analysis4b.table_v3 import _row

    d = pd.read_csv(P4B / "fig4b_advantage_by_variant.csv")
    if len(d) != 36:
        raise RuntimeError("expected 36 DIAG variants")
    n = d.best_textbook_narrow.value_counts()
    c = d.best_textbook_corrected.value_counts()
    sens = ("narrow beam: " + ", ".join(f"{k} {v}" for k, v in n.items())
            + "; first-order model: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    return _row("H173", "Number of the 36 DIAG variants in which Sn (β-Sn, ambient phase) is the closest textbook material "
                        "to the B4 lanthanide frontier, narrow beam",
                float(n.get("Sn", 0)), np.nan, np.nan, "count", "none (count)", "areal",
                "B4 lanthanide vs textbook materials", "best textbook material per variant",
                "data/phase4b/fig4b_advantage_by_variant.csv", "column=best_textbook_narrow (count of Sn)",
                delta=36.0, delta_unit="variants", sens=sens, density="n/a (areal)", dossier="R3.2", note=NOTE)


def _mc_chi2_row() -> dict:
    from shieldmap.analysis4b.table_v3 import _row

    p = pd.read_csv(common.P4 / "mc" / "points.csv")
    chi2 = float((p.z_narrow_vs_endf ** 2).sum())
    n2 = int((p.z_narrow_vs_endf.abs() > 2).sum())
    return _row("H174", f"Monte Carlo narrow-beam estimator vs the analytic narrow-beam transmission with ENDF/B-VIII.0 "
                        f"cross sections: chi-square over the {len(p)} slab points of the six handover cases",
                chi2, np.nan, np.nan, "chi2", "none (statistic)", "both", "n/a (not a price)",
                "Monte Carlo set-up validation", "data/phase4a/mc/points.csv", "sum of z_narrow_vs_endf**2 over all rows",
                delta=float(len(p)), delta_unit="slab points", sens=f"{n2} points beyond 2 sigma",
                mc_status="Monte Carlo result", fluor="Monte Carlo narrow beam (scored uncollided)", dossier="R-BU",
                note=NOTE)


def _curve_range_rows() -> list[dict]:
    """H175-H178: the other end of the headline curve (R2, best_clean, coherent) over the diagnostic variants or the
    single lines, where the stage 4a row gives the median or maximum and the band envelope.
    H179: P(best) of BiOF under rule R3 at Cs-137."""
    from shieldmap.analysis4b.table_v3 import _row

    b = pd.read_parquet(common.P4 / "S" / "bands.parquet")
    b = b[(b.reference == "B4-internal") & (b.artefact_def == common.PRIMARY_ARTEFACT)]
    src = "data/phase4a/S/bands.parquet"
    rows = []
    d = b[(b.scenario == "DIAG") & (b.basis == "areal")]
    for rid, lab, name, twin in (("H175", "GAP(3.1)|B4", "band gap ≥ 3.1 eV (GAP)", "H45"),
                                 ("H176", "ABUND(1)|B4", "scarcity proxy (crustal abundance) ≥ 1 ppm", "H46")):
        x = d[d.label == lab]
        if len(x) != 36:
            raise RuntimeError(f"{lab}: {len(x)} DIAG variants")
        rows.append(_row(rid, f"Price of {name} given B4 over the 36 DIAG variants, areal basis: largest headline value over "
                              f"the variants (at {x.loc[x.price.idxmax(), 'key']}; {twin} gives the median and the band envelope)",
                         100 * float(x.price.max()), 100 * float(x.price.min()), 100 * float(x.price.max()), "%",
                         "range of the headline value over the 36 variants", "areal", f"B4-internal ({lab})", "",
                         src, f"scenario=DIAG; label={lab}; basis=areal; reference=B4-internal; max of price",
                         density="n/a (areal)", dossier="R2.5", note=NOTE))
    m = b[(b.scenario == "MONO") & (b.basis == "thickness") & (b.density == "measured")
          & (b.energy_keV >= 100) & (b.energy_keV <= 10000)]
    for rid, lab, name, twin in (("H177", "GAP(3.1)|B4", "GAP(3.1) given B4", "H51"),
                                 ("H178", "ABUND(1)|B4", "ABUND(1) given B4", "H53")):
        x = m[m.label == lab]
        rows.append(_row(rid, f"{name}, thickness (measured-density point), single lines 100 keV–10 MeV: smallest headline value "
                              f"over the energies (at {x.loc[x.price.idxmin(), 'energy_keV']:.0f} keV; {twin} is the largest, "
                              f"its band the envelope)",
                         100 * float(x.price.min()), 100 * float(x.price.min()), 100 * float(x.price.max()), "%",
                         "range of the headline value over the single-line energies", "thickness", f"B4-internal ({lab})", "",
                         src, f"scenario=MONO; label={lab}; basis=thickness; density=measured; 100-10000 keV; min of price",
                         density="measured-density point", dossier="R2.4", note=NOTE))
    t = pd.read_csv(paths.DATA / "phase3" / "T" / "tier_summary.csv")
    s = t[(t.scenario == "CS137") & (t.variant_id == 38) & (t.set == "B5")].set_index("rule")
    if s.at["R3", "top_winner_copula"] != "BiOF" or s.at["R1", "top_winner_copula"] != "HfO2":
        raise RuntimeError("unexpected B5 Cs-137 leaders by rule")
    rows.append(_row("H179", "Thinnest B5 compound at CS137 (T = 0.01) under rule R3: P(best) of BiOF (R2: H90; under R1 the "
                             "leader is HfO2 with P(best) 0.596, the low end of H90's band)",
                     float(s.at["R3", "top_winner_p_copula"]), np.nan, np.nan, "P(best) of BiOF", "none (single rule)",
                     "thickness", "B5 (Phase 3 set; contains no artefact entry)", "",
                     "data/phase3/T/tier_summary.csv", "scenario=CS137; variant_id=38; rule=R3; set=B5",
                     kind="n/a", density="density model (copula)", dossier="R3.4", note=NOTE))
    return rows


def _panel_rows() -> list[dict]:
    """H300+: values the revision quotes that exist in stage 1-4a outputs without a row
    (IDs H180-H299 are reserved for the revision analyses). Re-presentation only."""
    from shieldmap.analysis4b.table_v3 import _row

    rows = []
    tb = pd.read_parquet(paths.DATA / "processed" / "table_b_compounds.parquet").set_index("material_id")
    src_b = "data/processed/table_b_compounds.parquet"
    # H300 (BiOF gap) dropped: duplicate of H241 (rows_revision)
    rows.append(_row("H301", "Energy above the Materials Project hull of beta-Sn (mp-84, ambient phase), which excludes it from B4",
                     float(tb.at["mp-84", "energy_above_hull"]), np.nan, np.nan, "eV/atom", "none (database value)", "n/a",
                     "n/a (not a price)", "Sn (mp-84)", src_b, "material_id=mp-84; column=energy_above_hull",
                     kind="n/a", dossier="R4.3", note=NOTE))
    c = pd.read_csv(paths.DATA / "phase3" / "D" / "bias_by_class.csv")
    c = c[(c.population == "R3") & (c.class_kind == "report (overlapping)")].set_index("class")
    lo, hi = c["median"].idxmin(), c["median"].idxmax()
    rows.append(_row("H302", f"Median DFT density bias by compound class (Fig. S1 report classes, R3): range over classes, from {lo} to {hi}",
                     100 * float(c["median"].min()), 100 * float(c["median"].min()), 100 * float(c["median"].max()), "%",
                     "range over the overlapping report classes", "thickness only", "n/a (not a price)", "",
                     "data/phase3/D/bias_by_class.csv", "population=R3; class_kind=report (overlapping); column=median",
                     kind="n/a", dossier="R4.1", note=NOTE))
    p = pd.read_csv(common.P4 / "mc" / "points.csv")
    f1 = p[np.isclose(p.factor, 1.0)]
    src_p = "data/phase4a/mc/points.csv"
    for rid, cases, label in (("H303", ("c6_cs137_pb", "c1_cs137"), "Cs-137 (Pb, Ir, HfO2, BiOF)"),
                              ("H304", ("c2_co60",), "Co-60 (Ir, HfO2)")):
        x = f1[f1.case_id.isin(cases)]
        v = (x.coh_primary / x.narrow)
        rows.append(_row(rid, f"Monte Carlo: coherently scattered source-energy photons as a fraction of the uncollided air kerma at the "
                              f"narrow-beam T = 0.01 slab, {label}: range over materials (counted in broad, not in narrow)",
                         float(v.min()), float(v.min()), float(v.max()), "fraction", "range over the simulated materials",
                         "thickness", "n/a (not a price)", "", src_p, f"factor=1.0; case_id in {list(cases)}; coh_primary/narrow",
                         kind="n/a", mc_status="Monte Carlo result", dossier="R-BU", note=NOTE))
    x = f1[f1.case_id.isin(("c3_rqr5", "c4_rqr8")) & f1.material.isin(("Bi", "Pb"))]
    v = x.lines_L_hi / x.lines_hi
    rows.append(_row("H305", "Monte Carlo: share of L lines in the escaping fluorescence air kerma (lines >= 3.2029 keV) of Bi and Pb at "
                             "the two diagnostic beams (RQR5 0.25 mm, RQR8 0.35 mm), headline slab: range",
                     100 * float(v.min()), 100 * float(v.min()), 100 * float(v.max()), "%", "range over Bi, Pb x two beams", "areal",
                     "n/a (not a price)", "", src_p, "factor=1.0; case_id in [c3_rqr5, c4_rqr8]; material in [Bi, Pb]; lines_L_hi/lines_hi",
                     kind="n/a", mc_status="Monte Carlo result", dossier="R3.1", note=NOTE))
    # H306-H309 (cone1, fl_all) dropped: duplicates of H253, H256, H252, H255 (rows_revision)
    d = pd.read_csv(P4B / "fig4b_advantage_by_variant.csv")
    n = d.best_textbook_narrow.value_counts()
    for rid, mat, name in (("H310", "Bi", "Bi"), ("H311", "Gd2O3", "Gd2O3")):
        rows.append(_row(rid, f"Number of the 36 DIAG variants in which {name} is the closest textbook material to the B4 lanthanide frontier, narrow beam",
                         float(n.get(mat, 0)), np.nan, np.nan, "count", "none (count)", "areal",
                         "B4 lanthanide vs textbook materials", "best textbook material per variant",
                         "data/phase4b/fig4b_advantage_by_variant.csv", f"column=best_textbook_narrow (count of {mat})",
                         kind="n/a", dossier="R3.2", note=NOTE))
    a = pd.read_parquet(paths.DATA / "processed" / "table_a_elements.parquet").set_index("symbol")
    for rid, el in (("H312", "Bi"), ("H313", "Ir"), ("H314", "Hf")):
        rows.append(_row(rid, f"Crustal abundance of {el} (scarcity proxy input; CRC Handbook 95th ed. via mendeleev 1.3.0, P3)",
                         float(a.at[el, "crustal_abundance_ppm"]), np.nan, np.nan, "ppm", "none (tabulated constant)", "n/a",
                         "n/a (not a price)", el, "data/processed/table_a_elements.parquet",
                         f"symbol={el}; column=crustal_abundance_ppm", kind="n/a", dossier="R2.3", note=NOTE))
    rows.append(_row("H315", "Monte Carlo: independent sub-runs per slab point (range over the 78 points)",
                     float(p.n_subruns.min()), float(p.n_subruns.min()), float(p.n_subruns.max()), "count", "range over points",
                     "n/a", "n/a (not a price)", "", src_p, "column=n_subruns", kind="n/a", mc_status="Monte Carlo result",
                     dossier="R-BU", note=NOTE))
    rows.append(_row("H316", "Monte Carlo narrow-beam estimator vs analytic ENDF transmission: reduced chi-square (H174 / number of points)",
                     float((p.z_narrow_vs_endf ** 2).sum() / len(p)), np.nan, np.nan, "ratio", "none (statistic)", "both",
                     "n/a (not a price)", "", src_p, "sum(z_narrow_vs_endf**2)/len(points)", kind="n/a",
                     mc_status="Monte Carlo result", dossier="R-BU", note=NOTE))
    rows.append(_row("H317", "Experimental solid density of Os used at the measured-density point (the Ir-excluded thickness frontier, H37)",
                     float(a.at["Os", "density_solid_exp"]), np.nan, np.nan, "g/cm3", "none (tabulated value)", "thickness",
                     "n/a (not a price)", "Os", "data/processed/table_a_elements.parquet", "symbol=Os; column=density_solid_exp",
                     kind="n/a", density="measured: Phase 1 experimental element density", dossier="R2.7", note=NOTE))
    return rows


def _three_beam_row() -> dict:
    """H318: like-for-like Monte Carlo advantage of the simulated lanthanide compounds over Bi at the three simulated
    beams (c3 RQR5, c7 RQR6, c4 RQR8), fluorescence-inclusive and broad-beam: the envelope quoted as the third answer.
    It combines H189-H192 (e3_like_for_like.csv) with H260, H261, H264, H265 (mc_rev/metrics.csv).."""
    from shieldmap.analysis4b.table_v3 import _row

    e3 = pd.read_csv(P4B / "revision" / "e3_like_for_like.csv")
    e3 = e3[e3.estimator.isin(("fl", "broad")) & e3.pair.str.startswith("Bi/")]
    m = pd.read_csv(P4B / "mc_rev" / "metrics.csv")
    m = m[(m.case_id == "c7_rqr6") & (m.target == "like-for-like") & (m.num == "Bi") & m.den.isin(("Ce5AgSn3", "PrSbPd"))
          & m.estimator.isin(("fl", "broad")) & (m.basis == "areal")]
    if len(e3) != 4 or len(m) != 4:
        raise ValueError(f"H318: expected 4 + 4 like-for-like values, got {len(e3)} + {len(m)}")
    v = 100 * pd.concat([e3.adv_lfl, m.value]).astype(float)
    return _row("H318", "Monte Carlo (OpenMC, ENDF/B-VIII.0): advantage of the simulated reduced (frontier) lanthanide compounds over Bi on areal mass "
                        "with the like-for-like Pb target, fluorescence-inclusive and broad-beam estimators, at the three simulated "
                        "beams (RQR5 behind 0.25 mm Pb, LaBi; RQR6 behind 0.35 mm Pb, Ce5AgSn3 and PrSbPd; RQR8 behind 0.35 mm Pb, "
                        "Pr3OsI3), all at 12°: range (value = lower end; envelope of H189-H192, H260, H261, H264, H265). The fully oxidised "
                        "Ba2SmSbO6, also simulated at RQR6, goes the other way (H268, H269) and is not included",
                float(v.min()), float(v.min()), float(v.max()), "%", "range over beams, compounds and estimators", "areal",
                "simulated lanthanide compounds vs Bi", "LaBi; Ce5AgSn3; PrSbPd; Pr3OsI3",
                "data/phase4b/revision/e3_like_for_like.csv; data/phase4b/mc_rev/metrics.csv",
                "e3: estimator in (fl, broad), column=adv_lfl; mc_rev: case_id=c7_rqr6, target=like-for-like, num=Bi, "
                "den in (Ce5AgSn3, PrSbPd), estimator in (fl, broad), column=value",
                kind="n/a", fluor="yes (Monte Carlo transport, fl and broad)", mc_status="Monte Carlo result",
                dossier="E3/O1-O3", decision="P4B-D-350", note=NOTE)


def _round2_rows() -> list[dict]:
    """Revision changes: re-presentation of existing revision outputs.

    H319-H322  price of AMB-ANION and AMB-ICSD3 together (fully oxidised, at least 3 ICSD ids) given B4, areal mass, at
               the three simulated X-ray beams: range and per beam (data/phase4b/revision/e5_amb_prices.csv)
    H323       Pr3OsI3 at RQR8 (c4): drop of the like-for-like advantage of the lanthanide compound over Bi from the
               narrow-beam to the fluorescence-inclusive estimator, percentage points (e3_like_for_like.csv; H194 - H191)
    """
    from shieldmap.analysis4b.table_v3 import _row

    src = "data/phase4b/revision/e5_amb_prices.csv"
    e5 = pd.read_csv(P4B / "revision" / "e5_amb_prices.csv")
    x = e5[(e5.tier == "AMB-ANION&ICSD3") & (e5.basis == "areal") & (e5.practical == "primary")
           & e5.key.isin(("DIAG|v3", "DIAG|v13", "DIAG|v22"))].set_index("key")
    if len(x) != 3:
        raise ValueError("H319: expected three DIAG rows of AMB-ANION&ICSD3")
    lab = {"DIAG|v13": "DIAG RQR6, 12°, 0.35 mm Pb target (variant 13)", "DIAG|v3": "DIAG RQR5, 12°, 0.25 mm Pb target (variant 3)",
           "DIAG|v22": "DIAG RQR8, 12°, 0.35 mm Pb target (variant 22)"}
    rows = [_row("H319", "Price of the fully oxidised filter with at least 3 ICSD ids (AMB-ANION ∧ AMB-ICSD3) given B4, areal mass, "
                         "narrow beam, at the three simulated X-ray beams (RQR5 0.25 mm, RQR6 0.35 mm, RQR8 0.35 mm Pb, all 12°): "
                         "range (value = lower end)",
                 100 * float(x.price.min()), 100 * float(x.price.min()), 100 * float(x.price.max()), "%",
                 "range over the three beams (primary values)", "areal", "B4-internal", "; ".join(x.reference),
                 src, "tier=AMB-ANION&ICSD3; key in (DIAG|v3, DIAG|v13, DIAG|v22); basis=areal; practical=primary; column=price",
                 kind="best_clean", dossier="R1-N1/R3-N1", decision="P4B-D-362", note=NOTE)]
    for rid, key in (("H320", "DIAG|v13"), ("H321", "DIAG|v3"), ("H322", "DIAG|v22")):
        r = x.loc[key]
        rows.append(_row(rid, f"Price of the fully oxidised filter with at least 3 ICSD ids (AMB-ANION ∧ AMB-ICSD3) given B4, {lab[key]}, "
                              f"areal mass, narrow beam (leader {r.leader})",
                         100 * float(r.price), 100 * float(r.band_lo), 100 * float(r.band_hi), "%", "R1-R3 x frontier kinds x coherent",
                         "areal", "B4-internal", str(r.reference), src,
                         f"tier=AMB-ANION&ICSD3; key={key}; basis=areal; practical=primary; columns=price, band_lo, band_hi",
                         kind="best_clean", dossier="R1-N1/R3-N1", decision="P4B-D-362", note=NOTE))
    e3 = pd.read_csv(P4B / "revision" / "e3_like_for_like.csv").set_index(["case_id", "estimator"])
    d = 100 * (float(e3.loc[("c4_rqr8", "narrow"), "adv_lfl"]) - float(e3.loc[("c4_rqr8", "fl"), "adv_lfl"]))
    rows.append(_row("H323", "Monte Carlo case c4_rqr8 (RQR8, 12°, 0.35 mm Pb): drop of the like-for-like advantage Bi/Pr3OsI3 − 1 from the "
                             "narrow-beam to the fluorescence-inclusive estimator (H194 − H191)",
                     d, np.nan, np.nan, "percentage points", "none (difference of two Monte Carlo values)", "areal",
                     "Pr3OsI3 vs Bi", "Pr3OsI3", "data/phase4b/revision/e3_like_for_like.csv",
                     "case_id=c4_rqr8; adv_lfl(narrow) - adv_lfl(fl)", kind="n/a", mc_status="Monte Carlo result",
                     dossier="R2/R3-N2", decision="P4B-D-362", note=NOTE))
    return rows


def extra_rows() -> list[dict]:
    rows = _textbook_rows() + [_variant_count_row(), _mc_chi2_row()] + _curve_range_rows() + _panel_rows() + [_three_beam_row()] + _round2_rows()
    for r in rows:
        r["v3_origin"] = "phase4b-step4-text"
        r.setdefault("decision", "")
    return sorted(rows, key=lambda r: int(r["id"][1:]))


IDS = [f"H{i}" for i in range(168, 180)]
