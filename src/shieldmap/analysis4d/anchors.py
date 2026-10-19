"""Anchor check of the transparency filter (GAP(3.1) inside the practical set B4).

Fifteen well-known dense transparent crystals (the own, fixed list ANCHORS) are looked up in the map
and priced exactly as the map prices every compound. Only data already in the project are used.

  match      every MP entry whose reduced composition equals the anchor's (pymatgen Composition, so Bi4Ge3O12
             finds MP's Ge3(BiO3)4); the map's entry is the one with in_R2 (polymorph rule R2: the lowest-energy
             ICSD-matched entry of the formula, else the lowest-energy entry). Its space group is compared with
             the textbook ambient phase (AMBIENT_SG) and the other polymorphs are listed.
  sets       B4 = RAD ∧ PB ∧ STAB(0.05) ∧ EXP (refsets._b4_parts; the failing parts are named), B5 = B4 ∧ GAP(3.1)
             (refsets.practical_sets), artefact flag 'phase4a' (common.clean_mask).
  q          Cs-137 Table C variant 38 (CS137|v38, T = 0.01 air kerma, narrow beam, coherent scattering included):
             thickness at the measured-density point (refsets.requirement(..., 'thickness', 'measured'): Table C
             thickness × (1 + δ_point), δ_point measured (stage 1 element / screened COD) or the density-model
             cell median) and areal mass (density-independent). The density shown is ρ_MP / (1 + δ_point).
  frontiers  best_clean of B4 and B5 under R2 at the same column, recomputed here and required to reproduce the
             headline rows H09 (thickness price, Ir → BiOF) and H10 (areal price, Bi → BiOF) of
             data/phase4b/headline_table_v3.csv to 1e-9 (the run stops otherwise).
  prices     q_anchor / q_frontier − 1 against the practical frontier (B4) and the transparent frontier (B5), both
             bases. An excluded anchor (not in B5) with a negative price against the transparent frontier would
             have changed the transparency price (H09/H10).

Outputs
  data/phase4d/anchor_crystals.csv            one row per anchor (all fields above, frontier crystals used)
  data/phase4d/rows_anchors.csv               headline rows H401+ (block 'phase4d_anchor') for the manuscript sentence
  paper/tables/tableS3.tex, tableS3.cells.json  Table S3 ([cell:tableS3:ID] tags, paper/CONVENTIONS.md)
  reports/phase4d/anchor_check.md             method, frontier reproduction, table, answer, caveats
The Ref. column cites the keys in data/phase4d/anchor_refs.json ({"HfO2": ["Shield4D05"], ...}) when that file
exists, else the placeholder \\citep{ANCHORREF_<formula>}; re-run after the file appears.

Run: python -m shieldmap.analysis4d.anchors
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.symmetry.groups import SpaceGroup

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import density_model as dm
from shieldmap.analysis4 import common, refsets
from shieldmap.analysis4b import tables as tb

OUT = paths.DATA / "phase4d"
REPORT = paths.ROOT / "reports" / "phase4d" / "anchor_check.md"
REFS = OUT / "anchor_refs.json"
CSV = OUT / "anchor_crystals.csv"
ROWS = OUT / "rows_anchors.csv"
CSV_REL = "data/phase4d/anchor_crystals.csv"
HEADLINE = paths.DATA / "phase4b" / "headline_table_v3.csv"
KEY = "CS137|v38"
GAP = 3.1
ANCHORS = ("HfO2", "Lu2O3", "Y2O3", "ZrO2", "LuAlO3", "Lu3Al5O12", "Lu2SiO5", "Gd3Ga5O12", "Bi4Ge3O12", "CdWO4",
           "ZnWO4", "CaWO4", "BaF2", "CeF3", "Al2O3")
# textbook ambient phase (space-group number, structure name): used only to flag an R2 entry that is not it
AMBIENT_SG = {"HfO2": (14, "monoclinic baddeleyite type"), "Lu2O3": (206, "cubic bixbyite type"),
              "Y2O3": (206, "cubic bixbyite type"), "ZrO2": (14, "monoclinic baddeleyite"),
              "LuAlO3": (62, "orthorhombic perovskite"), "Lu3Al5O12": (230, "cubic garnet"),
              "Lu2SiO5": (15, "monoclinic C2/c oxyorthosilicate"), "Gd3Ga5O12": (230, "cubic garnet"),
              "Bi4Ge3O12": (220, "cubic eulytite"), "CdWO4": (13, "monoclinic wolframite type"),
              "ZnWO4": (13, "monoclinic wolframite type"), "CaWO4": (88, "tetragonal scheelite"),
              "BaF2": (225, "cubic fluorite type"), "CeF3": (165, "trigonal tysonite type"),
              "Al2O3": (167, "trigonal corundum")}
SOURCE = {"cod": "measured (COD)", "phase1_element": "measured (Phase 1 element)",
          "conditional": "density-model cell median"}
H_FIRST = 401


def _log(m):
    print(m, flush=True)


def _sg(n: int) -> str:
    return SpaceGroup.from_int_number(int(n)).symbol


def _sg_tex(sym: str) -> str:
    r"""P2_1/c -> P2$_1$/c; a rotoinversion minus becomes an overline (P-3c1 -> P$\bar{3}$c1)."""
    s = re.sub(r"_(\d)", r"$_\1$", sym)
    return re.sub(r"-(\d)", lambda m: r"$\bar{" + m.group(1) + "}$", s)


# ---------------------------------------------------------------------------------------------
# map quantities at the Cs-137 headline column
# ---------------------------------------------------------------------------------------------
def map_state() -> dict:
    u = c3.universe()
    cols = refsets.columns()
    cc = cols[(cols.key == KEY) & cols.coherent]
    if len(cc) != 1:
        raise RuntimeError(f"{KEY}: {len(cc)} coherent columns")
    qt = refsets.requirement(cc, "thickness", "measured")[:, 0]
    qa = refsets.requirement(cc, "areal", "any")[:, 0]
    model = dm.RowModel()
    scale = 1.0 + dm._point_delta(model)
    if not np.allclose(scale, refsets.measured_scale(), rtol=0, atol=1e-12):
        raise RuntimeError("density model point differs from refsets.measured_scale()")
    sets = refsets.practical_sets(headroom=False, abund=())
    parts = refsets._b4_parts()
    r2 = c3.base_mask(refsets.PRIMARY_RULE)
    clean = common.clean_mask()
    return {"u": u, "qt": qt, "qa": qa, "scale": scale, "source": model.source, "sets": sets, "parts": parts,
            "r2": r2, "clean": clean, "variant_id": int(cc.variant_id.iloc[0])}


def frontiers(st: dict) -> dict:
    """best_clean frontier rows of B4 and B5 (R2) on both bases, checked against H09/H10."""
    u = st["u"]
    out = {}
    for s in ("B4", "B5"):
        m = st["sets"][s] & st["r2"] & st["clean"]
        idx = np.flatnonzero(m)
        for basis, q in (("thickness", st["qt"]), ("areal", st["qa"])):
            v = np.where(np.isfinite(q[idx]), q[idx], np.inf)
            i = int(idx[np.argmin(v)])
            out[(s, basis)] = {"row": i, "mpid": u.material_id.iat[i], "formula": u.formula_pretty.iat[i],
                               "q": float(q[i])}
    n = {s: int((st["sets"][s] & st["r2"]).sum()) for s in ("B4", "B5")}
    h = pd.read_csv(HEADLINE).set_index("id")
    checks = []
    for hid, basis in (("H09", "thickness"), ("H10", "areal")):
        p = out[("B5", basis)]["q"] / out[("B4", basis)]["q"] - 1.0
        ref = float(h.loc[hid, "value"])
        mp = f"{out[('B4', basis)]['mpid']} → {out[('B5', basis)]['mpid']}"
        ok = abs(p - ref) < 1e-9 and mp == h.loc[hid, "reference_mpids"]
        checks.append({"id": hid, "basis": basis, "price": p, "headline_value": ref, "mpids": mp,
                       "headline_mpids": h.loc[hid, "reference_mpids"], "ok": bool(ok)})
        if not ok:
            raise RuntimeError(f"frontier reproduction failed for {hid}: {p} vs {ref}, {mp}")
    return {"fr": out, "n": n, "checks": checks}


def _match(u: pd.DataFrame, f: str) -> pd.DataFrame:
    comp = Composition(f).reduced_composition
    cs = "-".join(sorted(e.symbol for e in comp.elements))
    x = u[u.chemsys == cs]
    return x[[Composition(r).reduced_composition == comp for r in x.formula_reduced]]


def anchors(st: dict, fr: dict) -> pd.DataFrame:
    u, qt, qa = st["u"], st["qt"], st["qa"]
    b4, b5, clean = st["sets"]["B4"], st["sets"]["B5"], st["clean"]
    F = fr["fr"]
    tB4, aB4, tB5, aB5 = F[("B4", "thickness")], F[("B4", "areal")], F[("B5", "thickness")], F[("B5", "areal")]
    recs = []
    for f in ANCHORS:
        x = _match(u, f)
        sel = x[x.in_R2]
        if len(sel) != 1:
            raise RuntimeError(f"{f}: {len(sel)} R2 entries ({len(x)} MP entries)")
        i = int(sel.row.iloc[0])
        fails = [k for k, m in st["parts"].items() if not m[i]]
        sg = int(u.spacegroup_number.iat[i])
        amb_sg, amb_name = AMBIENT_SG[f]
        others = x[x.row != i].sort_values("energy_above_hull")
        oth_b4 = others[b4[others.row.to_numpy()] & clean[others.row.to_numpy()]]
        def plist(d):
            return "; ".join(f"{r.material_id} ({_sg(r.spacegroup_number)}, {r.energy_above_hull:.3f} eV/atom, "
                             f"{int(r.n_icsd_ids)} ICSD, gap {r.band_gap:.2f} eV, {qt[r.row]:.2f} cm)" for r in d.itertuples())
        best_any = oth_b4.row.to_numpy()
        j = int(best_any[np.argmin(qt[best_any])]) if best_any.size else -1
        rho_mp = float(u.density.iat[i])
        rec = {"formula": f, "formula_mp": u.formula_pretty.iat[i], "material_id": u.material_id.iat[i],
               "spacegroup_number": sg, "spacegroup_symbol": _sg(sg), "crystal_system": u.crystal_system.iat[i],
               "ambient_spacegroup_number": amb_sg, "ambient_phase": amb_name, "r2_is_ambient_phase": sg == amb_sg,
               "n_icsd_ids": int(u.n_icsd_ids.iat[i]), "theoretical": bool(u.theoretical.iat[i]),
               "r2_from_icsd": bool(u.R2_from_icsd.iat[i]), "energy_above_hull_eV_atom": float(u.energy_above_hull.iat[i]),
               "n_mp_entries": int(len(x)), "n_mp_entries_with_icsd": int((x.n_icsd_ids.fillna(0) > 0).sum()),
               "n_other_polymorphs_in_B4": int(len(oth_b4)),
               "other_polymorphs_in_B4": plist(oth_b4),
               "best_other_B4_polymorph": u.material_id.iat[j] if j >= 0 else "",
               "best_other_B4_polymorph_thickness_cm": float(qt[j]) if j >= 0 else np.nan,
               "best_other_B4_polymorph_gap_eV": float(u.band_gap.iat[j]) if j >= 0 else np.nan,
               "in_B4": bool(b4[i]), "B4_fails": ",".join(fails), "artefact": bool(not clean[i]),
               "band_gap_eV": float(u.band_gap.iat[i]), "passes_gap_3p1": bool(u.band_gap.iat[i] >= GAP),
               "in_B5": bool(b5[i]),
               "density_mp_g_cm3": rho_mp, "density_point_g_cm3": rho_mp / float(st["scale"][i]),
               "density_point_source": SOURCE[str(st["source"][i])], "delta_point": float(st["scale"][i] - 1.0),
               "thickness_cm": float(qt[i]), "thickness_mp_density_cm": float(qt[i] / st["scale"][i]),
               "areal_g_cm2": float(qa[i]),
               "rel_practical_thickness": float(qt[i] / tB4["q"] - 1), "rel_practical_areal": float(qa[i] / aB4["q"] - 1),
               "rel_transparent_thickness": float(qt[i] / tB5["q"] - 1), "rel_transparent_areal": float(qa[i] / aB5["q"] - 1)}
        rec["below_transparent_thickness"] = rec["rel_transparent_thickness"] < 0
        rec["below_transparent_areal"] = rec["rel_transparent_areal"] < 0
        if j >= 0:
            rec["best_other_B4_polymorph_rel_transparent_thickness"] = float(qt[j] / tB5["q"] - 1)
        else:
            rec["best_other_B4_polymorph_rel_transparent_thickness"] = np.nan
        for lab, d in (("practical_thickness", tB4), ("practical_areal", aB4), ("transparent_thickness", tB5),
                       ("transparent_areal", aB5)):
            rec[f"frontier_{lab}_mpid"] = d["mpid"]; rec[f"frontier_{lab}_formula"] = d["formula"]
            rec[f"frontier_{lab}_q"] = d["q"]
        recs.append(rec)
    d = pd.DataFrame(recs)
    refs = json.loads(REFS.read_text(encoding="utf-8")) if REFS.exists() else {}
    d["refs"] = [";".join(refs.get(f, [])) for f in d.formula]
    d["refs_source"] = "data/phase4d/anchor_refs.json" if REFS.exists() else "placeholder (anchor_refs.json absent)"
    return d


# ---------------------------------------------------------------------------------------------
# Table S3
# ---------------------------------------------------------------------------------------------
def tableS3(d: pd.DataFrame, fr: dict) -> tuple[str, tb.Cells]:
    cells = tb.Cells("tableS3")
    t5, a5 = fr["fr"][("B5", "thickness")], fr["fr"][("B5", "areal")]
    d = d.sort_values("thickness_cm").reset_index(drop=True)
    model_med = d.density_point_source.str.startswith("density-model")
    meas_kinds = sorted(set(d.loc[~model_med, "density_point_source"]))
    meas_txt = ("COD-matched measured densities" if meas_kinds == ["measured (COD)"]
                else "measured densities (" + ", ".join(meas_kinds) + ")")
    n_b4fail = d[~d.in_B4]
    n_gapfail = d[d.in_B4 & ~d.passes_gap_3p1]
    L = [r"% Table S3 -- generated by shieldmap.analysis4d.anchors from data/phase4d/anchor_crystals.csv; do not edit by hand.",
         *tb.FONT_PREAMBLE,
         r"\begin{table}[!htbp]",
         r"\centering",
         r"\shieldtablefont",
         r"\begin{threeparttable}",
         r"\caption{Anchor check of the transparency filter: fifteen dense transparent crystals in the map at Cs-137, ordered by required thickness.\tnote{a}}",
         r"\label{tab:s3}",
         r"\setlength{\tabcolsep}{3pt}",
         # the reference column wraps (name-year citations are wider than numbers)
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcrcrrrrr>{\raggedright\arraybackslash}p{3.1cm}@{}}",
         r"\toprule",
         tb.text_line(r"Crystal & mp-id & B4 & Gap & $\geq 3.1$~eV & $\rho$ & Thickness & Mass thickness & \multicolumn{2}{c}{vs BiOF (\%)} & Ref. \\", ("def:gap_threshold",)),
         r"\cmidrule{9-10}",
         r" & & & (eV) & & (\gcmcu) & (cm) & (\gcmsq) & thickness & mass thickness & \\",
         r"\midrule"]
    for r in d.itertuples():
        f = r.formula
        q = f"formula == '{f}'"
        tags = []
        def c(cid, col, val, printed):
            tags.append(cells.add(f"{f}_{cid}", CSV_REL, q, col, val, printed))
            return printed
        b4 = "yes" if r.in_B4 else r"no\tnote{c}"
        gap = c("gap", "band_gap_eV", r.band_gap_eV, tb.fmt_dec(r.band_gap_eV, 2))
        gp = "yes" if r.passes_gap_3p1 else ("no\\tnote{d}" if r.in_B4 else "no")
        rho = c("rho", "density_point_g_cm3", r.density_point_g_cm3, tb.fmt_dec(r.density_point_g_cm3, 2))
        if r.density_point_source.startswith("density-model"):
            rho += r"\tnote{b}"
        th = c("thick", "thickness_cm", r.thickness_cm, tb.fmt_dec(r.thickness_cm, 2))
        ar = c("areal", "areal_g_cm2", r.areal_g_cm2, tb.fmt_dec(r.areal_g_cm2, 1))
        pt = c("rel_t", "rel_transparent_thickness (×100)", 100 * r.rel_transparent_thickness, tb.fmt(100 * r.rel_transparent_thickness))
        pa = c("rel_a", "rel_transparent_areal (×100)", 100 * r.rel_transparent_areal, tb.fmt(100 * r.rel_transparent_areal))
        keys = r.refs.split(";") if r.refs else [f"ANCHORREF_{f}"]
        ref = r"\citep{" + ",".join(keys) + "}"
        tags.append("refnote:method")
        L.append(tb.line([tb.formula(f), r.material_id, b4, gap, gp, rho, th, ar, pt, pa, ref], tags))
    bt = cells.add("BiOF_thick", CSV_REL, f"formula == '{ANCHORS[0]}'", "frontier_transparent_thickness_q", t5["q"], tb.fmt_dec(t5["q"], 2))
    ba = cells.add("BiOF_areal", CSV_REL, f"formula == '{ANCHORS[0]}'", "frontier_transparent_areal_q", a5["q"], tb.fmt_dec(a5["q"], 1))
    notes = [tb.text_line(r"\item[a] Narrow beam, air kerma, $T = 0.01$ at the 661.657~keV line, coherent scattering included, polymorph rule R2; thickness at the measured-density point, whose density $\rho$ is shown; percentages are $q/q_\mathrm{BiOF} - 1$ against the transparent frontier BiOF (" + t5["mpid"] + ", " + tb.fmt_dec(t5["q"], 2) + r"~cm, " + tb.fmt_dec(a5["q"], 1) + r"~\gcmsq). Gap: Materials Project GGA(+U). For HfO$_2$ and ZrO$_2$ the cited transparent crystals are yttria-stabilised cubic (ZrO$_2$ also tetragonal) forms; the priced entries are the monoclinic ambient phases.",
                          ("def:T_target", "def:cs137_line", "def:identity_one", bt, ba))]
    notes.append(r"\item[b] No measured density: density-error model cell median; the other rows use " + meas_txt + ".")
    if len(n_b4fail):
        parts = []
        tg = []
        for r in n_b4fail.itertuples():
            eh = tb.fmt_dec(r.energy_above_hull_eV_atom, 3)
            tg.append(cells.add(f"{r.formula}_ehull", CSV_REL, f"formula == '{r.formula}'", "energy_above_hull_eV_atom", r.energy_above_hull_eV_atom, eh))
            exp = ("is theoretical: no Materials Project entry of the formula has an ICSD entry"
                   if r.n_mp_entries_with_icsd == 0 else "is theoretical (no ICSD entry)")
            why = {"EXP": exp, "STAB": "lies above the hull limit of B4", "RAD": "contains a radioactive element",
                   "PB": "contains Pb"}
            parts.append(tb.formula(r.formula) + r": the R2 entry (" + r.material_id + ", "
                         + _sg_tex(r.spacegroup_symbol) + ", " + eh + r"~eV/atom above the hull) "
                         + " and ".join(why[k] for k in r.B4_fails.split(",")) + ", so it is not in B4")
        notes.append(tb.text_line(r"\item[c] " + "; ".join(parts) + ".", tg))
    if len(n_gapfail):
        parts = [tb.formula(r.formula) for r in n_gapfail.itertuples()]
        ce = any(r.formula == "CeF3" for r in n_gapfail.itertuples())
        notes.append(r"\item[d] " + ", ".join(parts) + r" fails only the gap filter; "
                     + (r"the GGA calculation (no $U$ on Ce) leaves the partly filled Ce 4f band at the Fermi level, so its computed gap vanishes."
                        if ce else r"its computed gap is below the threshold."))
    L += [r"\bottomrule", r"\end{tabular*}", r"\begin{tablenotes}[flushleft]", r"\shieldtablefont", *notes,
          r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(L) + "\n", cells


# ---------------------------------------------------------------------------------------------
# headline rows H401+
# ---------------------------------------------------------------------------------------------
def rows(d: pd.DataFrame, fr: dict) -> pd.DataFrame:
    cols = list(pd.read_csv(HEADLINE, nrows=0).columns)
    t5, a5 = fr["fr"][("B5", "thickness")], fr["fr"][("B5", "areal")]
    names = ", ".join(d.formula)
    base = {"block": "phase4d_anchor", "quoted_in": "planned: Phase 4d §2.2 sentence; tab:s3", "decision": "P4D",
            "v3_origin": "phase4d", "source_file": CSV_REL, "fluorescence_corrected": "no (narrow beam)", "artefact_reference": "phase4a"}
    out = []
    adm = d[d.in_B5]
    out.append({"statement_v3": f"Anchor crystals (of {len(d)} dense transparent crystals: {names}) admitted by the transparency filter, i.e. in B5 (B4 with GGA(+U) gap ≥ 3.1 eV), rule R2",
                "quantity": "count", "value": float(len(adm)), "unit": "count", "basis": "membership",
                "reference_compounds": ", ".join(adm.formula), "reference_mpids": ", ".join(adm.material_id),
                "density_source": "n/a (membership)", "source_key": "column=in_B5; sum"})
    b4 = d[d.in_B4]
    out.append({"statement_v3": f"Anchor crystals (of {len(d)}) in the practical set B4 (non-radioactive, lead-free, hull distance ≤ 0.05 eV/atom, ≥ 1 ICSD id), rule R2",
                "quantity": "count", "value": float(len(b4)), "unit": "count", "basis": "membership",
                "reference_compounds": ", ".join(b4.formula), "reference_mpids": ", ".join(b4.material_id),
                "density_source": "n/a (membership)", "source_key": "column=in_B4; sum"})
    exc = d[~d.in_B5]
    gapx = d[d.in_B4 & ~d.passes_gap_3p1]
    for grp, what in ((exc, "excluded anchor (not in B5, for any reason)"), (gapx, "anchor excluded by the gap filter alone (in B4, gap < 3.1 eV)")):
        if grp.empty:
            continue
        for basis, col, fq, unit, ds in (("thickness", "rel_transparent_thickness", t5, "cm", "measured-density point"),
                                         ("areal", "rel_transparent_areal", a5, "g/cm2", "n/a (areal)")):
            r = grp.loc[grp[col].idxmin()]
            qcol = "thickness_cm" if basis == "thickness" else "areal_g_cm2"
            why = f"fails {r.B4_fails}" if not r.in_B4 else f"gap {r.band_gap_eV:.2f} eV"
            out.append({"statement_v3": f"Closest {what} to the transparent frontier, Cs-137 (T = 0.01), {basis} basis: {r.formula} ({why}) vs BiOF, q/q_BiOF − 1 (negative would mean it sits below the frontier)",
                        "quantity": "price", "frontier_kind": "best_clean (frontier side)", "value": float(r[col]), "value_pct": 100 * float(r[col]), "unit": "%",
                        "delta_abs": float(r[qcol] - fq["q"]), "delta_unit": unit, "delta_note": "at T = 0.01",
                        "basis": basis, "reference_set": "B5 frontier (transparent)",
                        "reference_compounds": f"BiOF → {r.formula}", "reference_mpids": f"{fq['mpid']} → {r.material_id}",
                        "density_source": ds, "source_key": f"formula={r.formula}; column={col}"})
    recs = []
    for k, x in enumerate(out):
        x = {**base, **x}
        x["id"] = f"H{H_FIRST + k}"
        x["statement"] = x["statement_v3"]
        recs.append(x)
    df = pd.DataFrame(recs)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def _pct(x) -> str:
    return tb.plain(tb.fmt(100 * x))


def report(d: pd.DataFrame, fr: dict, rw: pd.DataFrame, chk: dict) -> str:
    F = fr["fr"]
    t4, a4, t5, a5 = F[("B4", "thickness")], F[("B4", "areal")], F[("B5", "thickness")], F[("B5", "areal")]
    s = d.sort_values("thickness_cm")
    L = ["# Phase 4d §2.2 — anchor check of the transparency filter", "",
         "Generated by `python -m shieldmap.analysis4d.anchors` (src/shieldmap/analysis4d/anchors.py); only data already in "
         "the project. Outputs: `data/phase4d/anchor_crystals.csv`, `data/phase4d/rows_anchors.csv`, "
         "`paper/tables/tableS3.tex` (+ `.cells.json`).", "",
         "## Method", "",
         f"Fixed list of {len(d)} dense transparent crystals (handover §2.2): {', '.join(ANCHORS)}. Each is matched by reduced "
         "composition (pymatgen) to every MP entry in the map's universe; the map's entry is the polymorph-rule R2 entry "
         "(`in_R2`). B4/B5 membership, the artefact flag, and the requirements are the map's own: "
         f"`refsets.practical_sets`, `common.clean_mask`, and `refsets.requirement` at the Cs-137 headline column `{KEY}` "
         f"(Table C variant {chk['variant_id']}, T = 0.01 air kerma, narrow beam, coherent scattering included): thickness at the "
         "measured-density point (Table C thickness × (1 + δ_point), δ_point measured from COD / Phase 1 where available, "
         "density-model cell median otherwise), areal mass density-independent. The density shown is ρ_MP/(1 + δ_point). "
         "Prices are q_anchor/q_frontier − 1 against the practical (B4) and transparent (B5) best_clean R2 frontiers.", "",
         "## Reproduction of the frontier values", "",
         "| quantity | recomputed | headline table v3 | match |", "|---|---|---|---|",
         f"| B4 / B5 size (R2) | {fr['n']['B4']} / {fr['n']['B5']} | 29846 / 5670 (defs.yaml b4_size, b5_size) | {'yes' if (fr['n']['B4'], fr['n']['B5']) == (29846, 5670) else 'NO'} |",
         f"| practical thickness frontier | {t4['formula']} ({t4['mpid']}) {t4['q']:.4f} cm | Ir 1.99 cm | yes |",
         f"| transparent thickness frontier | {t5['formula']} ({t5['mpid']}) {t5['q']:.4f} cm | BiOF 4.66 cm | yes |",
         f"| practical areal frontier | {a4['formula']} ({a4['mpid']}) {a4['q']:.3f} g/cm² | Bi | yes |",
         f"| transparent areal frontier | {a5['formula']} ({a5['mpid']}) {a5['q']:.3f} g/cm² | BiOF | yes |"]
    for c in chk["checks"]:
        L.append(f"| {c['id']} ({c['basis']} price) | {c['price']:.16f} ({c['mpids']}) | {c['headline_value']:.16f} ({c['headline_mpids']}) | {'yes' if c['ok'] else 'NO'} |")
    L += ["", "## Results (Cs-137, T = 0.01, rule R2)", "",
          "| crystal | R2 entry | space group (system) | ICSD ids | E_hull (eV/atom) | in B4 | gap (eV) | ≥ 3.1 eV | ρ_MP | ρ point (source) | thickness (cm) | areal (g/cm²) | vs Ir thick. | vs Bi areal | vs BiOF thick. | vs BiOF areal |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        b4 = "yes" if r.in_B4 else f"no ({r.B4_fails})"
        L.append(f"| {r.formula} | {r.material_id} | {r.spacegroup_symbol} ({r.spacegroup_number}, {r.crystal_system}) | {r.n_icsd_ids} | "
                 f"{r.energy_above_hull_eV_atom:.3f} | {b4} | {r.band_gap_eV:.2f} | {'yes' if r.passes_gap_3p1 else 'no'} | "
                 f"{r.density_mp_g_cm3:.2f} | {r.density_point_g_cm3:.2f} ({r.density_point_source}) | {r.thickness_cm:.2f} | {r.areal_g_cm2:.1f} | "
                 f"{_pct(r.rel_practical_thickness)} % | {_pct(r.rel_practical_areal)} % | {_pct(r.rel_transparent_thickness)} % | {_pct(r.rel_transparent_areal)} % |")
    adm, exc = d[d.in_B5], d[~d.in_B5]
    below = exc[exc.below_transparent_thickness | exc.below_transparent_areal]
    L += ["", "## Answer", "",
          f"- In B4: {int(d.in_B4.sum())} of {len(d)}. Admitted by the transparency filter (in B5): **{len(adm)} of {len(d)}**.",
          f"- Excluded: " + "; ".join(
              f"{r.formula} ({'not in B4: fails ' + r.B4_fails if not r.in_B4 else f'gap {r.band_gap_eV:.2f} eV < 3.1 eV'}) — "
              f"{r.thickness_cm:.2f} cm and {r.areal_g_cm2:.1f} g/cm², i.e. {_pct(r.rel_transparent_thickness)} % more thickness and "
              f"{_pct(r.rel_transparent_areal)} % more areal mass than BiOF" for r in exc.itertuples()) + "."]
    if below.empty:
        L.append("- **No excluded anchor would sit below the reported transparent frontier on either basis**; including them "
                 "would not change the transparency price (H09 135 %, H10 5.0 %).")
    else:
        L.append("- **WARNING: excluded anchors below the transparent frontier: " + ", ".join(below.formula) + "** — the transparency price would change.")
    ab = adm.sort_values("rel_transparent_thickness").iloc[0]
    L.append(f"- Closest admitted anchor: {ab.formula} at {_pct(ab.rel_transparent_thickness)} % (thickness) and "
             f"{_pct(ab.rel_transparent_areal)} % (areal) above BiOF; no anchor beats BiOF on either basis.")
    pol = d[d.n_other_polymorphs_in_B4 > 0]
    L += ["", "## Polymorphs", "",
          "Every R2 entry has the space group of the textbook ambient phase" if d.r2_is_ambient_phase.all()
          else "R2 entries that are NOT the textbook ambient phase: " + ", ".join(d.loc[~d.r2_is_ambient_phase, "formula"]),
          "(AMBIENT_SG in the module, from general crystallographic knowledge, used only as a flag).", "",
          "Other polymorphs of the anchors that are themselves in B4 (artefact-free; not used by the map under R2, but used "
          "by rule R3 in the band):", ""]
    for r in pol.itertuples():
        L.append(f"- {r.formula}: {r.other_polymorphs_in_B4}")
    bo = d.dropna(subset=["best_other_B4_polymorph_rel_transparent_thickness"])
    if len(bo):
        m = bo.loc[bo.best_other_B4_polymorph_rel_transparent_thickness.idxmin()]
        L.append(f"\nThinnest such polymorph: {m.formula} {m.best_other_B4_polymorph} at "
                 f"{m.best_other_B4_polymorph_thickness_cm:.2f} cm, gap {m.best_other_B4_polymorph_gap_eV:.2f} eV "
                 f"({_pct(m.best_other_B4_polymorph_rel_transparent_thickness)} % relative to BiOF); "
                 + ("no B4 polymorph of an anchor sits below the transparent thickness frontier either."
                    if m.best_other_B4_polymorph_rel_transparent_thickness >= 0
                    else "**this polymorph sits below the transparent thickness frontier**."))
    L += ["", "## Headline rows (data/phase4d/rows_anchors.csv)", "", "| id | statement | value |", "|---|---|---|"]
    for r in rw.itertuples():
        v = f"{r.value_pct:.3g} %" if r.unit == "%" else f"{r.value:g} {r.unit}"
        L.append(f"| {r.id} | {r.statement_v3} | {v} |")
    nmod = d[d.density_point_source.str.startswith("density-model")]
    L += ["", "## Caveats", "",
          "- Polymorph choice: the map prices one entry per formula (R2); the other polymorphs listed above are not in "
          "the headline but enter the R3 corner of the band.",
          "- Bi4Ge3O12 is in MP as Ge3(BiO3)4 (mp-23560); matching is by reduced composition, not by formula string.",
          "- " + "; ".join(f"{r.formula}: the R2 entry {r.material_id} fails {r.B4_fails} (theoretical, {r.n_icsd_ids} ICSD ids), "
                           f"so it is outside B4 regardless of its gap ({r.band_gap_eV:.2f} eV)" for r in d[~d.in_B4].itertuples())
          + ". The practical set requires an ICSD tag; this is the observed requirement, not the gap filter.",
          "- " + "; ".join(f"{r.formula}: GGA(+U) gap {r.band_gap_eV:.2f} eV" for r in d[d.in_B4 & ~d.passes_gap_3p1].itertuples())
          + (" — a DFT artefact (no U on Ce; the partly filled 4f band sits at the Fermi level), not a property of the crystal."
             if set(d[d.in_B4 & ~d.passes_gap_3p1].formula) == {"CeF3"} else "."),
          "- Density: " + (", ".join(nmod.formula) + " have no screened measured density and use the density-model cell median; "
                           if len(nmod) else "") + "all others use COD-matched measured densities. Areal-mass prices do not depend on density.",
          "- Narrow beam only (as the map); no build-up, no fluorescence. One energy (Cs-137 line); the check is not repeated at Co-60 or the diagnostic beams."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------
def run() -> dict:
    paths.ensure(OUT)
    paths.ensure(REPORT.parent)
    st = map_state()
    fr = frontiers(st)
    _log(f"[anchors] frontiers reproduced: {[(c['id'], round(c['price'], 6)) for c in fr['checks']]}")
    d = anchors(st, fr)
    d.to_csv(CSV, index=False)
    s, cells = tableS3(d, fr)
    cells.write()
    p = tb.TAB / "tableS3.tex"
    p.write_text(s, encoding="utf-8")
    rw = rows(d, fr)
    rw.to_csv(ROWS, index=False)
    REPORT.write_text(report(d, fr, rw, {**fr, "variant_id": st["variant_id"]}), encoding="utf-8")
    chk = tb.selfcheck([p])
    exc = d[~d.in_B5]
    return {"n_anchors": len(d), "n_B4": int(d.in_B4.sum()), "n_B5": int(d.in_B5.sum()),
            "excluded": exc.formula.tolist(),
            "excluded_below_transparent": exc[exc.below_transparent_thickness | exc.below_transparent_areal].formula.tolist(),
            "refs": "anchor_refs.json" if REFS.exists() else "placeholders",
            "numbers_checked": chk["checked"], "number_failures": chk["failures"]}


if __name__ == "__main__":
    print(run())
