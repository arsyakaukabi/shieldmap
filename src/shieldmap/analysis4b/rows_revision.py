"""Headline rows H180+ for the revision analyses.

Every value is read from a CSV that analysis4b.revision.run() writes to data/phase4b/revision/:

  H180-H188  E2  best garment-element blend (Sn, Sb, Bi, W, Ba, Gd) vs the B4 compound frontier and vs Bi, narrow
                 beam, areal mass: variants 13 (headline), 3 and 22 (MC beams), range over the 36 variants
  H189-H199  E3  like-for-like Monte Carlo advantage (Pb target = Pb's own transmission under the same estimator)
                 and Pb/Bi − 1 like-for-like
  H200-H219  E5  ambient-chemistry tiers AMB-ANION and AMB-ICSD3 priced inside B4; best fully oxidised lanthanide
                 compound with ≥ 3 ICSD ids vs the B4 frontier and vs Bi
  H220-H241  E7  GAP threshold sweep (2.5, 3.5, 4.0 eV; 3.2 eV switch), cubic-only variant, BiOF's band gap
  H242-H245  E12 first-order fluorescence under-count per case family
  H246-H251  O5  alternative build-up ratio broad / (uncollided + coherent-scattered primary)
  H252-H257  O12 fl_all, cone1, cone3 advantages for the two diagnostic Monte Carlo pairs
  H260-H279  O1-O3 Monte Carlo case c7_rqr6 at the reference beam (analysis4b.mc_rev; data/phase4b/mc_rev/):
                 Bi/X − 1 for X in {Ce5AgSn3, PrSbPd, Ba2SmSbO6, Sn/Ba/Gd blend} × {like-for-like, fixed} × {fl, broad},
                 blend and Ba2SmSbO6 vs the frontier, range row, narrow-vs-ENDF χ²
E11 (compounds that set each 0 % price; non-ambient frontier compounds) adds names, not numbers: see
data/phase4b/revision/e11_zero_setters.csv and e11_flagged_appearances.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap import paths

REV = paths.DATA / "phase4b" / "revision"
SRC = "data/phase4b/revision/"
WHERE = {"DIAG v13": "DIAG RQR6, 12°, 0.35 mm Pb target (variant 13)",
         "DIAG v3": "DIAG RQR5, 12°, 0.25 mm Pb target (variant 3)",
         "DIAG v22": "DIAG RQR8, 12°, 0.35 mm Pb target (variant 22)",
         "Cs-137 thickness": "Cs-137 (T = 0.01), thickness at the measured-density point",
         "Cs-137 areal": "Cs-137 (T = 0.01), areal basis",
         "Co-60 thickness": "Co-60 (T = 0.01), thickness at the measured-density point",
         "Co-60 areal (H excl.)": "Co-60 (T = 0.01), areal basis, H-containing compounds excluded (D3)"}
CREDIBLE = "credible (R1–R3 × best_clean/best_icsd/median50_clean × coherent; Phase 4a flag)"


def _note(n) -> str:
    return f"revision round 1 (editor item {n})"


def _density(basis: str) -> str:
    return "measured-density point" if basis == "thickness" else "n/a (areal)"


# --------------------------------------------------------------------------------------------- E2
def _e2(_row) -> list[dict]:
    b = pd.read_csv(REV / "e2_blend_by_variant.csv").set_index("variant_id")
    src = SRC + "e2_blend_by_variant.csv"
    rows = []
    rid = 180
    for vid, tag in ((13, "DIAG v13"), (3, "DIAG v3"), (22, "DIAG v22")):
        r = b.loc[vid]
        rows.append(_row(f"H{rid}", f"Best binary/ternary blend of the garment elements Sn, Sb, Bi, W, Ba, Gd (mass fractions, "
                                    f"μ/ρ by the mixture rule) relative to the B4 compound frontier, {WHERE[tag]}, narrow beam, "
                                    f"areal mass (negative: the blend needs less than the frontier compound)",
                         100 * r.blend_vs_B4, 100 * r.blend_vs_B4_lo, 100 * r.blend_vs_B4_hi, "%",
                         "B4 frontier R1–R3 × best_clean/best_icsd/median50_clean × coherent (blend at the same coherence)",
                         "areal", "B4 frontier (best_clean, R2) vs best elemental blend",
                         f"{r.best_blend} vs {r.B4_frontier}", src, f"variant_id={vid}; column=blend_vs_B4",
                         delta=float(r.q_blend - r.q_B4), delta_unit="g/cm2", delta_note="q(blend) − q(B4 frontier) at T*",
                         density="n/a (areal)", sens=f"best binary {r.best_binary}: {100 * r.binary_vs_B4:+.2f} %; "
                                                     f"q(blend) {r.q_blend:.4f}, q(B4) {r.q_B4:.4f} g/cm²",
                         dossier="R3.2", decision="E2", note=_note(2)))
        rows.append(_row(f"H{rid + 1}", f"Best binary/ternary blend of Sn, Sb, Bi, W, Ba, Gd relative to Bi, {WHERE[tag]}, "
                                        f"narrow beam, areal mass (negative: the blend needs less than Bi)",
                         100 * r.blend_vs_Bi, 100 * r.blend_vs_Bi_lo, 100 * r.blend_vs_Bi_hi, "%",
                         "coherent on/off (blend and Bi at the same coherence)", "areal", "Bi (mp-23152) vs best elemental blend",
                         f"{r.best_blend} vs Bi", src, f"variant_id={vid}; column=blend_vs_Bi",
                         delta=float(r.q_blend - r.q_Bi), delta_unit="g/cm2", delta_note="q(blend) − q(Bi) at T*",
                         density="n/a (areal)", sens=f"B4 frontier vs Bi {100 * r.B4_vs_Bi:+.1f} %",
                         dossier="R3.2", decision="E2", note=_note(2)))
        rid += 2
    for col, name, ref in (("blend_vs_B4", "the B4 compound frontier", "B4 frontier (best_clean, R2)"),
                           ("blend_vs_Bi", "Bi", "Bi (mp-23152)")):
        x = b[col]
        rows.append(_row(f"H{rid}", f"Best blend of Sn, Sb, Bi, W, Ba, Gd relative to {name} over the 36 DIAG variants, narrow beam, "
                                    f"areal mass: median over the variants (band = range over the variants; minimum at "
                                    f"variant {int(x.idxmin())}, maximum at variant {int(x.idxmax())})",
                         100 * float(x.median()), 100 * float(x.min()), 100 * float(x.max()), "%",
                         "range of the headline value over the 36 DIAG variants", "areal", f"{ref} vs best elemental blend",
                         "best blend per variant", src, f"all variants; column={col}", density="n/a (areal)",
                         sens=f"variants where the blend beats the B4 frontier: {int((b.blend_vs_B4 < 0).sum())} of 36",
                         dossier="R3.2", decision="E2", note=_note(2)))
        rid += 1
    r = b.loc[13]
    rows.append(_row(f"H{rid}", f"Best binary blend (two of Sn, Sb, Bi, W, Ba, Gd) relative to the B4 compound frontier, "
                                f"{WHERE['DIAG v13']}, narrow beam, areal mass",
                     100 * r.binary_vs_B4, np.nan, np.nan, "%", "none (single value)", "areal",
                     "B4 frontier (best_clean, R2) vs best binary elemental blend", f"{r.best_binary} vs {r.B4_frontier}",
                     src, "variant_id=13; column=binary_vs_B4", density="n/a (areal)", dossier="R3.2", decision="E2",
                     note=_note(2)))
    return rows


# --------------------------------------------------------------------------------------------- E3
def _e3(_row) -> list[dict]:
    d = pd.read_csv(REV / "e3_like_for_like.csv")
    src = SRC + "e3_like_for_like.csv"
    beam = {"c3_rqr5": "RQR5, 12°, behind 0.25 mm Pb", "c4_rqr8": "RQR8, 12°, behind 0.35 mm Pb"}
    est_txt = {"narrow": "narrow-beam", "fl": "fluorescence-inclusive", "broad": "broad-beam"}
    rows = []
    rid = 189

    def g(cid, est):
        return d[(d.case_id == cid) & (d.estimator == est)].iloc[0]
    for cid, est in (("c3_rqr5", "fl"), ("c3_rqr5", "broad"), ("c4_rqr8", "fl"), ("c4_rqr8", "broad"),
                     ("c3_rqr5", "narrow"), ("c4_rqr8", "narrow")):
        r = g(cid, est)
        v, s = 100 * r.adv_lfl, 100 * r.adv_lfl_se
        rows.append(_row(f"H{rid}", f"Monte Carlo (OpenMC, ENDF/B-VIII.0), case {cid} ({beam[cid]}): advantage {r.pair} − 1 "
                                    f"with the like-for-like Pb target (Pb's own transmission under the {est_txt[est]} "
                                    f"estimator at the nominal Pb areal mass {r.q_Pb_nominal:.4f} g/cm²)",
                         v, v - 1.96 * s, v + 1.96 * s, "%", "Monte Carlo statistical 95 % interval (± 1.96 SE, joint bootstrap)",
                         "areal", "Bi vs best lanthanide compound (MC pair)", r.pair.replace("/", " vs "), src,
                         f"case_id={cid}; estimator={est}; column=adv_lfl",
                         fluor="Monte Carlo (fluorescence transported)" if est != "narrow" else "Monte Carlo narrow beam (scored uncollided)",
                         mc_status="Monte Carlo result",
                         sens=f"fixed narrow-beam target T = {r.T_target_fixed:.4f}: {100 * r.adv_fixedT:.2f} %; "
                              f"like-for-like target T(Pb) = {r.T_Pb_same_estimator:.4f}",
                         dossier="R3.1", decision="E3", note=_note(3)))
        rid += 1
    sel = d[d.estimator.isin(["fl", "broad"])]
    rows.append(_row(f"H{rid}", "Monte Carlo advantage of the best lanthanide compound over Bi on areal mass with the like-for-like "
                                "Pb target, fluorescence-inclusive and broad-beam estimators, the two simulated beams (RQR5 behind "
                                "0.25 mm Pb, LaBi; RQR8 behind 0.35 mm Pb, Pr3OsI3): range (value = lower end)",
                     100 * sel.adv_lfl.min(), 100 * sel.adv_lfl.min(), 100 * sel.adv_lfl.max(), "%",
                     "range over the two cases × estimators fl/broad", "areal", "B4 lanthanide vs Bi", "LaBi, Pr3OsI3 vs Bi",
                     src, "estimator in (fl, broad); column=adv_lfl", fluor="Monte Carlo (fluorescence transported)",
                     mc_status="Monte Carlo result", dossier="R3.1", decision="E3",
                     note=_note(3) + "; summarises the four like-for-like fl/broad rows; the fixed-T range is H145"))
    rid += 1
    for cid, est in (("c3_rqr5", "fl"), ("c3_rqr5", "broad"), ("c4_rqr8", "fl"), ("c4_rqr8", "broad")):
        r = g(cid, est)
        v, s = 100 * r.Pb_vs_num_lfl, 100 * r.Pb_vs_num_lfl_se
        rows.append(_row(f"H{rid}", f"Monte Carlo, case {cid} ({beam[cid]}): Pb/Bi − 1 on areal mass, like-for-like "
                                    f"({est_txt[est]} estimator; Pb at its nominal areal mass, Bi at Pb's transmission)",
                         v, v - 1.96 * s, v + 1.96 * s, "%", "Monte Carlo statistical 95 % interval (± 1.96 SE, joint bootstrap)",
                         "areal", "Pb vs Bi (MC)", "Pb vs Bi", src, f"case_id={cid}; estimator={est}; column=Pb_vs_num_lfl",
                         fluor="Monte Carlo (fluorescence transported)", mc_status="Monte Carlo result",
                         sens=f"Pb vs {r.pair.split('/')[1]} like-for-like {100 * r.Pb_vs_den_lfl:.1f} ± {100 * r.Pb_vs_den_lfl_se:.1f} %",
                         dossier="R3.1", decision="E3", note=_note(3)))
        rid += 1
    return rows


# --------------------------------------------------------------------------------------------- E5
def _e5(_row) -> list[dict]:
    a = pd.read_csv(REV / "e5_amb_prices.csv")
    ln = pd.read_csv(REV / "e5_amb_lanthanide.csv")
    rows = []
    rid = 200
    txt = {"AMB-ANION": "fully oxidised tier AMB-ANION (anions only O and F, every other element in a positive "
                        "oxidation state, i.e. oxides, fluorides and oxyanion salts)",
           "AMB-ICSD3": "ICSD-count tier AMB-ICSD3 (at least 3 ICSD ids)"}
    for tier in ("AMB-ANION", "AMB-ICSD3"):
        for r in a[a.tier == tier].itertuples():
            rows.append(_row(f"H{rid}", f"Price of {txt[tier]} given B4, {WHERE[r.where]}",
                             100 * r.price, 100 * r.band_lo, 100 * r.band_hi, "%", CREDIBLE, r.basis,
                             f"B4-internal ({tier}|B4)" + (", H-containing compounds excluded" if r.practical == "noH" else ""),
                             f"{r.reference} → {r.leader}", SRC + "e5_amb_prices.csv",
                             f"tier={tier}; key={r.key}; basis={r.basis}; practical={r.practical}",
                             delta=float(r.delta_abs), delta_unit=r.delta_unit, delta_note="at the target T",
                             density=_density(r.basis), dossier="R3.3", decision="E5", note=_note(5)))
            rid += 1
    x = ln[ln.tier == "AMB-ANION&ICSD3"].set_index("variant_id")
    alt = ln[ln.tier == "AMB-ANION"].set_index("variant_id")
    for vid, tag in ((13, "DIAG v13"), (3, "DIAG v3"), (22, "DIAG v22")):
        r = x.loc[vid]
        s = alt.loc[vid]
        rows.append(_row(f"H{rid}", f"Best fully oxidised lanthanide compound with at least 3 ICSD ids (B4 ∧ lanthanide ∧ "
                                    f"AMB-ANION ∧ AMB-ICSD3) relative to the B4 frontier, {WHERE[tag]}, narrow beam, areal mass",
                         100 * r.vs_B4, 100 * r.vs_B4_lo, 100 * r.vs_B4_hi, "%", CREDIBLE, "areal",
                         "B4-internal (LN&AMB-ANION&ICSD3|B4)", f"{r.B4_frontier} → {r.leader}", SRC + "e5_amb_lanthanide.csv",
                         f"tier=AMB-ANION&ICSD3; variant_id={vid}; column=vs_B4", density="n/a (areal)",
                         sens=f"AMB-ANION only (any ICSD count): {s.leader} {100 * s.vs_B4:+.1f} % vs B4, {100 * s.vs_Bi:+.1f} % vs Bi",
                         dossier="R3.3", decision="E5", note=_note(5)))
        rows.append(_row(f"H{rid + 1}", f"Best fully oxidised lanthanide compound with at least 3 ICSD ids relative to Bi, "
                                        f"{WHERE[tag]}, narrow beam, areal mass (negative: needs less than Bi)",
                         100 * r.vs_Bi, 100 * r.vs_Bi_lo, 100 * r.vs_Bi_hi, "%",
                         "R1–R3 × kinds × coherent of the lanthanide side; Bi (R2) at the same coherence", "areal",
                         "Bi vs B4 ∧ lanthanide ∧ AMB-ANION ∧ AMB-ICSD3", f"{r.leader} vs Bi", SRC + "e5_amb_lanthanide.csv",
                         f"tier=AMB-ANION&ICSD3; variant_id={vid}; column=vs_Bi", density="n/a (areal)",
                         dossier="R3.3", decision="E5", note=_note(5)))
        rid += 2
    return rows


# --------------------------------------------------------------------------------------------- E7
def _e7(_row) -> list[dict]:
    g = pd.read_csv(REV / "e7_gap_sweep.csv")
    src = SRC + "e7_gap_sweep.csv"
    rows = []
    rid = 220

    def one(r, rid, what):
        return _row(f"H{rid}", f"{what}, {WHERE[r.where]}", 100 * r.price, 100 * r.band_lo, 100 * r.band_hi, "%", CREDIBLE,
                    r.basis, f"B4-internal ({r.set})" + (", H-containing compounds excluded" if r.practical == "noH" else ""),
                    f"{r.reference} → {r.leader}", src, f"set={r.set}; key={r.key}; basis={r.basis}; practical={r.practical}",
                    delta=float(r.delta_abs), delta_unit=r.delta_unit, delta_note="at the target T",
                    density=_density(r.basis),
                    sens=f"leader {r.leader} ({r.leader_mpid}; gap {r.leader_gap_eV:.2f} eV; {r.leader_crystal_system})",
                    dossier="R2.2", decision="E7", note=_note(7))
    for t in (2.5, 3.5, 4.0):
        for r in g[(g.threshold_eV == t) & ~g.cubic_only].itertuples():
            rows.append(one(r, rid, f"Threshold sweep: price of GGA(+U) band gap ≥ {t:g} eV (GAP) given B4"))
            rid += 1
    r = next(g[(g.threshold_eV == 3.2) & ~g.cubic_only & (g["where"] == "Cs-137 thickness")].itertuples())
    rows.append(one(r, rid, "Threshold sweep: price of GGA(+U) band gap ≥ 3.2 eV (GAP) given B4 (the leader switches from BiOF)"))
    rid += 1
    for r in g[g.cubic_only].itertuples():
        rows.append(one(r, rid, "Cubic-only variant: price of GGA(+U) band gap ≥ 3.1 eV and cubic crystal system given B4"))
        rid += 1
    bg = pd.read_csv(REV / "e7_biof_gap.csv").iloc[0]
    rows.append(_row(f"H{rid}", f"GGA(+U) band gap of BiOF ({bg.material_id}, the GAP(3.1) leader at Cs-137), Materials Project",
                     float(bg.band_gap_eV), np.nan, np.nan, "eV", "none (database value)", "n/a", "n/a (not a price)",
                     f"BiOF ({bg.material_id})", SRC + "e7_biof_gap.csv", f"material_id={bg.material_id}; column=band_gap_eV",
                     kind="n/a", density="n/a", sens=f"margin over 3.1 eV: {bg['margin_over_3.1_eV']:.3f} eV; "
                                                       f"{bg.crystal_system}, space group {int(bg.spacegroup_number)}",
                     dossier="R2.2", decision="E7", note=_note(7)))
    return rows


# --------------------------------------------------------------------------------------------- E12
def _e12(_row) -> list[dict]:
    f = pd.read_csv(REV / "e12_fluor_undercount_families.csv").set_index("family")
    src = SRC + "e12_fluor_undercount_families.csv"
    rows = []
    rid = 242
    for fam in ("single emitter below its K edge (Bi, Pb at RQR5/RQR8)",
                "single emitter above the Pb K edge (Pb, Tl at 89.5 keV)",
                "partner compounds (LaBi, PrSbPd, Pr3OsI3, HfAu2)",
                "50 keV diagnostic cases (La, Bi, LaBi)"):
        r = f.loc[fam]
        rows.append(_row(f"H{rid}", f"First-order fluorescence model vs Monte Carlo, {fam}: under-count of the escaping line air "
                                    f"kerma (E ≥ 3.2029 keV), 1 − (first-order line term)/(MC line term), over every slab of the "
                                    f"family (value = lower end; band = range)",
                         100 * r.undercount_lines_min, 100 * r.undercount_lines_min, 100 * r.undercount_lines_max, "%",
                         "range over the slabs of the family", "areal", "n/a (model check)", r.materials, src,
                         f"family={fam}; columns=undercount_lines_min/max",
                         kind="n/a", density="n/a", fluor="Monte Carlo (fluorescence transported) vs first-order model",
                         mc_status="Monte Carlo result",
                         sens=f"total fl transmission under-count 1 − T_firstorder/T_fl: {100 * r.undercount_fl_min:.1f}–"
                              f"{100 * r.undercount_fl_max:.1f} %; {int(r.n_points)} slabs",
                         dossier="R-BU", decision="E12", note=_note(12)))
        rid += 1
    return rows


# --------------------------------------------------------------------------------------------- O5, O12
def _o5(_row) -> list[dict]:
    d = pd.read_csv(REV / "o5_buildup_alt.csv")
    rows = []
    for k, r in enumerate(d.itertuples()):
        v, s = r.B_alt, r.B_alt_se
        rows.append(_row(f"H{246 + k}", f"Monte Carlo build-up factor, alternative convention B' = broad / (uncollided + "
                                        f"coherent-scattered primary), {r.material}, case {r.case_id}, at the headline thickness "
                                        f"(T = 0.01); conventional B = broad/uncollided is {r.id_conventional}",
                         v, v - 1.96 * s, v + 1.96 * s, "ratio", "Monte Carlo statistical 95 % interval (± 1.96 SE, delta method)",
                         "thickness", "n/a (not a price)", r.material, SRC + "o5_buildup_alt.csv",
                         f"case_id={r.case_id}; material={r.material}; factor=1.0; column=B_alt", kind="n/a",
                         density="measured-density point", fluor="Monte Carlo (fluorescence transported)",
                         mc_status="Monte Carlo result",
                         sens=f"conventional B {r.B_conventional:.4f}; coherent-scattered primary / uncollided "
                              f"{r.coh_primary_over_narrow:.3f}", dossier="R-BU", decision="O5", note=_note("optional 5")))
    return rows


def _o12(_row) -> list[dict]:
    d = pd.read_csv(REV / "o12_mc_estimators.csv")
    txt = {"fl_all": "fluorescence-inclusive estimator with lines below 3.2029 keV (fl_all)",
           "cone1": "1° acceptance-cone estimator (cone1)", "cone3": "3° acceptance-cone estimator (cone3)"}
    rows = []
    rid = 252
    for cid in ("c3_rqr5", "c4_rqr8"):
        for est in ("fl_all", "cone1", "cone3"):
            r = d[(d.case_id == cid) & (d.estimator == est)].iloc[0]
            v, s = 100 * r.value, 100 * r.se
            rows.append(_row(f"H{rid}", f"Monte Carlo (OpenMC, ENDF/B-VIII.0), case {cid}: advantage {r.pair} − 1, {txt[est]}, "
                                        f"fixed narrow-beam Pb target", v, v - 1.96 * s, v + 1.96 * s, "%",
                             "Monte Carlo statistical 95 % interval (± 1.96 SE)", "areal", r.pair.replace("/", " vs "),
                             r.pair.replace("/", " vs "), SRC + "o12_mc_estimators.csv",
                             f"case_id={cid}; estimator={est}; column=value (= data/phase4a/mc/metrics.csv)",
                             fluor="Monte Carlo (fluorescence transported)", mc_status="Monte Carlo result",
                             dossier="R3.1", decision="O12", note=_note("optional 12")))
            rid += 1
    return rows


# --------------------------------------------------------------------------------------------- O1-O3 (MC case c7)
MC7 = paths.DATA / "phase4b" / "mc_rev"
MC7_SRC = "data/phase4b/mc_rev/metrics.csv"
MC7_NOTE = "revision round 1, MC case c7 (editor optional 1-3)"
MC7_BEAM = "RQR6, 12°, behind 0.35 mm Pb (variant 13, reference beam)"
MC7_NAME = {"Ce5AgSn3": "Ce5AgSn3 (narrow-beam B4 frontier)", "PrSbPd": "PrSbPd (first-order-corrected leader)",
            "Ba2SmSbO6": "Ba2SmSbO6 (best fully oxidised lanthanide compound with ≥ 3 ICSD ids)",
            "SnBaGd_blend": "the Sn 0.325 / Ba 0.330 / Gd 0.345 blend (homogeneous mixture)"}
MC7_EST = {"fl": "fluorescence-inclusive", "broad": "broad-beam"}
MC7_TGT = {"like-for-like": "like-for-like Pb target (Pb's own transmission under the same estimator at 0.3969 g/cm²)",
           "fixed": "fixed narrow-beam Pb target T* = 0.02877"}


def _mc7(_row) -> list[dict]:
    m = pd.read_csv(MC7 / "metrics.csv")
    x2 = pd.read_json(MC7 / "chi2.json", typ="series")

    def g(num, den, est, tgt):
        return m[(m.num == num) & (m.den == den) & (m.estimator == est) & (m.target == tgt)].iloc[0]

    def interval(tgt):
        return ("Monte Carlo statistical 95 % interval (± 1.96 SE, joint bootstrap)" if tgt == "like-for-like"
                else "Monte Carlo statistical 95 % interval (± 1.96 SE, compare.py propagation)")

    rows, rid = [], 260
    for x in ("Ce5AgSn3", "PrSbPd", "Ba2SmSbO6", "SnBaGd_blend"):
        for tgt in ("like-for-like", "fixed"):
            for est in ("fl", "broad"):
                r = g("Bi", x, est, tgt)
                v, s = 100 * r.value, 100 * r.se
                other = "fixed" if tgt == "like-for-like" else "like-for-like"
                pb, nar = g("Pb", x, est, tgt), g("Bi", x, "narrow", tgt)
                rows.append(_row(f"H{rid}", f"Monte Carlo (OpenMC, ENDF/B-VIII.0), case c7_rqr6 ({MC7_BEAM}): advantage Bi/X − 1 "
                                            f"on areal mass, X = {MC7_NAME[x]}, {MC7_EST[est]} estimator, {MC7_TGT[tgt]}"
                                            + (" (negative: X needs more areal mass than Bi)" if v < 0 else ""),
                                 v, v - 1.96 * s, v + 1.96 * s, "%", interval(tgt), "areal",
                                 "Bi vs X (MC case c7)", f"Bi vs {x}", MC7_SRC,
                                 f"case_id=c7_rqr6; target={tgt}; metric=Bi/{x} − 1; estimator={est}; column=value",
                                 fluor="Monte Carlo (fluorescence transported)", mc_status="Monte Carlo result",
                                 sens=f"{other} target: {100 * g('Bi', x, est, other).value:.2f} %; MC narrow ({tgt}): "
                                      f"{100 * nar.value:.2f} %; Pb/X − 1 ({tgt}, {est}): {100 * pb.value:.2f} ± "
                                      f"{100 * pb.se:.2f} %",
                                 dossier="R3.1", decision="O1-O3", note=MC7_NOTE))
                rid += 1
    for x, what in (("SnBaGd_blend", "the Sn/Ba/Gd blend relative to the B4 frontier Ce5AgSn3"),
                    ("Ba2SmSbO6", "Ba2SmSbO6 relative to the B4 frontier Ce5AgSn3")):
        r = g(x, "Ce5AgSn3", "fl", "like-for-like")
        v, s = 100 * r.value, 100 * r.se
        rows.append(_row(f"H{rid}", f"Monte Carlo, case c7_rqr6 ({MC7_BEAM}): areal requirement of {what}, X/Ce5AgSn3 − 1, "
                                    f"fluorescence-inclusive estimator, {MC7_TGT['like-for-like']}",
                         v, v - 1.96 * s, v + 1.96 * s, "%", interval("like-for-like"), "areal",
                         "B4 frontier (Ce5AgSn3) vs X (MC case c7)", f"{x} vs Ce5AgSn3", MC7_SRC,
                         f"case_id=c7_rqr6; target=like-for-like; metric={x}/Ce5AgSn3 − 1; estimator=fl; column=value",
                         fluor="Monte Carlo (fluorescence transported)", mc_status="Monte Carlo result",
                         sens=f"broad (like-for-like): {100 * g(x, 'Ce5AgSn3', 'broad', 'like-for-like').value:.2f} %; "
                              f"MC narrow: {100 * g(x, 'Ce5AgSn3', 'narrow', 'like-for-like').value:.2f} %; "
                              f"Table C: {100 * g(x, 'Ce5AgSn3', 'T_tablec', 'fixed').value:.2f} %",
                         dossier="R3.2" if x == "SnBaGd_blend" else "R3.3", decision="O1-O3", note=MC7_NOTE))
        rid += 1
    sel = m[(m.target == "like-for-like") & (m.num == "Bi") & m.den.isin(["Ce5AgSn3", "PrSbPd", "SnBaGd_blend"])
            & m.estimator.isin(["fl", "broad"])]
    pbbi = g("Pb", "Bi", "fl", "like-for-like")
    rows.append(_row(f"H{rid}", f"Monte Carlo, case c7_rqr6 ({MC7_BEAM}): advantage of the frontier materials over Bi on areal "
                                f"mass with the like-for-like Pb target, fluorescence-inclusive and broad-beam estimators, "
                                f"X in {{Ce5AgSn3, PrSbPd, Sn/Ba/Gd blend}}: range (value = lower end)",
                     100 * sel.value.min(), 100 * sel.value.min(), 100 * sel.value.max(), "%",
                     "range over the three materials × estimators fl/broad", "areal", "Bi vs frontier materials (MC case c7)",
                     "Ce5AgSn3, PrSbPd, Sn/Ba/Gd blend vs Bi", MC7_SRC,
                     "case_id=c7_rqr6; target=like-for-like; num=Bi; den in (Ce5AgSn3, PrSbPd, SnBaGd_blend); "
                     "estimator in (fl, broad); column=value", fluor="Monte Carlo (fluorescence transported)",
                     mc_status="Monte Carlo result",
                     sens=f"Pb/Bi − 1 like-for-like fl {100 * pbbi.value:.2f} ± {100 * pbbi.se:.2f} %, broad "
                          f"{100 * g('Pb', 'Bi', 'broad', 'like-for-like').value:.2f} %; fixed-target range "
                          f"{100 * m[(m.target == 'fixed') & (m.num == 'Bi') & m.den.isin(['Ce5AgSn3', 'PrSbPd', 'SnBaGd_blend']) & m.estimator.isin(['fl', 'broad'])].value.min():.1f}–"
                          f"{100 * m[(m.target == 'fixed') & (m.num == 'Bi') & m.den.isin(['Ce5AgSn3', 'PrSbPd', 'SnBaGd_blend']) & m.estimator.isin(['fl', 'broad'])].value.max():.1f} %",
                     dossier="R3.1", decision="O1-O3", note=MC7_NOTE + "; summarises the like-for-like rows H260, H261, H264, H265, H272, H273"))
    rid += 1
    rows.append(_row(f"H{rid}", f"Monte Carlo case c7_rqr6 ({MC7_BEAM}), consistency check: χ² of the narrow-beam (uncollided) "
                                f"MC transmission against the ENDF/B-VIII.0 analytic narrow-beam transmission over the "
                                f"{int(x2.dof)} simulated slabs (6 materials)",
                     float(x2.chi2_narrow_vs_endf), np.nan, np.nan, "chi2", f"none (statistic; {int(x2.dof)} degrees of freedom)",
                     "areal", "n/a (model check)", "Pb, Bi, Ce5AgSn3, PrSbPd, Ba2SmSbO6, Sn/Ba/Gd blend",
                     "data/phase4b/mc_rev/chi2.json", "column=chi2_narrow_vs_endf", kind="n/a", density="n/a",
                     fluor="Monte Carlo narrow beam (scored uncollided)", mc_status="Monte Carlo result",
                     sens=f"dof {int(x2.dof)}; p = {x2.p_value:.2f}; max |z| {x2.max_abs_z:.2f}; MC narrow / Table C "
                          f"{x2.ratio_narrow_mc_tablec_min:.4f}–{x2.ratio_narrow_mc_tablec_max:.4f}",
                     dossier="R-BU", decision="O1-O3", note=MC7_NOTE))
    return rows


def extra_rows() -> list[dict]:
    from shieldmap.analysis4b.table_v3 import _row
    rows = _e2(_row) + _e3(_row) + _e5(_row) + _e7(_row) + _e12(_row) + _o5(_row) + _o12(_row)
    ids = [r["id"] for r in rows]
    if ids != [f"H{i}" for i in range(180, 180 + len(rows))]:
        raise RuntimeError(f"revision rows are not contiguous from H180: {ids}")
    mc7 = _mc7(_row)                                  # O1-O3: MC case c7 at the reference beam, block H260-H279
    if [r["id"] for r in mc7] != [f"H{i}" for i in range(260, 280)]:
        raise RuntimeError(f"MC case c7 rows are not H260-H279: {[r['id'] for r in mc7]}")
    rows += mc7
    for r in rows:
        r["v3_origin"] = "phase4b-revision1"
    return rows


IDS = [f"H{i}" for i in range(180, 258)] + [f"H{i}" for i in range(260, 280)]
# planned placement: main-text guesses; paper/check_numbers.py records the actual placement
PLAN = {**{f"H{i}": "Fig4a; Fig4b; Sec3.4" for i in range(180, 189)},
        **{f"H{i}": "Abstract; Sec3.4; Fig4c; TS4" for i in range(189, 200)},
        **{f"H{i}": "Sec3.4; T1 (ambient tier rows)" for i in range(200, 220)},
        **{f"H{i}": "Sec3.3; T2 footnote" for i in range(220, 242)},
        **{f"H{i}": "Sec2.6; Sec4 (limitations)" for i in range(242, 246)},
        **{f"H{i}": "Sec2.7; TS4" for i in range(246, 252)},
        **{f"H{i}": "Fig4c caption" for i in range(252, 258)},
        **{f"H{i}": "Sec3.4; TS4 (MC case c7)" for i in range(260, 280)}}
