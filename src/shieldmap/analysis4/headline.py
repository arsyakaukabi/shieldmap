"""The locked headline table (reports/phase4a/headline_table.md, data/phase4a/headline_table.csv).

One row per number that stage 4b may quote. Price rows are generated from the outputs (refsets,
practical, density4) under the reference primaries: B4-internal marginals, best_clean, phase4a artefact flag,
R2, with coherent, thickness at the measured-density point (areal mass alongside). Rows that quote a stage 3
result that does not depend on the reference set cite the stage 3 file and key they come from. Headroom
(F0-based) rows form a separate appendix block. Optional inputs from the cross-section sensitivity and
The Monte Carlo are joined when their files exist.

Every row carries: statement, value, band (and its type), absolute Δ, basis, frontier kind, reference set,
reference compounds, density source, fluorescence correction, sensitivity axes and values, source file/key,
and an artefact check (neither reference compound is an artefact entry under the phase4a artefact flag).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common, practical, refsets

S = common.P4 / "S"
P3 = c3.P3
HEAD_KEYS = {"CS137|v38": "Cs-137 (T = 0.01)", "CO60|v42": "Co-60 (T = 0.01)"}
LABELS = ("RAD|B4", "PB|B4", "STAB|B4", "EXP|B4", "GAP(3.1)|B4", "ABUND(1)|B4")
LABEL_TEXT = {"RAD|B4": "non-radioactive (RAD) given the rest of B4", "PB|B4": "lead-free (PB) given the rest of B4",
              "STAB|B4": "stability STAB(0.05) given the rest of B4", "EXP|B4": "ICSD match (EXP) given the rest of B4",
              "GAP(3.1)|B4": "band gap ≥ 3.1 eV (GAP) given B4",
              "ABUND(1)|B4": "scarcity proxy (crustal abundance) ≥ 1 ppm given B4"}
HEADROOM = ("RAD|F0", "PB|F0", "GAP(3.1)|F0", "B4|F0", "B5|F0", "B6|F0")
PB_REL_MATERIALS = ("Ir", "W", "Ta", "Bi", "BiOF", "HfO2")
EDGE_ELEMENTS = ("W", "Ta", "Hf")
POLY_SUBSETS = ("primary", "ehull_le_005", "no_hp_ht_marker", "n_poly_2", "n_poly_ge3", "all_members_abs_delta_le_0.2")


def _minmax(v):
    return float(min(v)), float(max(v))
AXES_PRICE = ("rules R1/R2/R3", "frontier kind (best_clean, best_icsd, median50_clean)", "coherent on/off",
              "artefact flag (Phase 3 vs Phase 4a)", "natural radioactivity (strict/moderate/named)",
              "H-containing excluded", "element exclusion Tl/As/Cd/Hg/Be/Se")


# ---------------------------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------------------------
def _bands() -> pd.DataFrame:
    return pd.read_parquet(S / "bands.parquet")


def _pbands() -> pd.DataFrame:
    return pd.read_parquet(S / "practical_bands.parquet")


def _dbands() -> pd.DataFrame:
    return pd.read_csv(S / "density_bands.csv")


def _jget(rel: str, path: str):
    d = json.loads((paths.ROOT / rel).read_text(encoding="utf-8"))
    for k in path.strip("/").split("/"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def _artefact(mpids) -> bool:
    u = c3.universe()
    m = dict(zip(u.material_id, common.artefact_mask()))
    return any(bool(m.get(x, False)) for x in mpids if x)


def _xs_sens() -> pd.DataFrame | None:
    f = common.P4 / "xs" / "headline_sensitivity.csv"
    return pd.read_csv(f) if f.exists() else None


def _mc_status() -> pd.DataFrame | None:
    f = common.P4 / "mc" / "headline_status.csv"
    return pd.read_csv(f) if f.exists() else None


def _fmt(x, pct=True, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------------------------
# price rows
# ---------------------------------------------------------------------------------------------
def _price_row(b: pd.DataFrame, pb: pd.DataFrame, db: pd.DataFrame, key: str, label: str, basis: str,
               rid: str, dossier: str, where: str, statement: str | None = None) -> dict:
    den = "measured" if basis == "thickness" else "any"
    sel = b[(b.key == key) & (b.label == label) & (b.basis == basis) & (b.density == den) & (b.artefact_def == "phase4a")]
    if len(sel) != 1:
        raise RuntimeError(f"{key} {label} {basis}: {len(sel)} rows")
    r = sel.iloc[0]
    mp = b[(b.key == key) & (b.label == label) & (b.basis == basis) & (b.density == "mp") & (b.artefact_def == "phase4a")]
    old = b[(b.key == key) & (b.label == label) & (b.basis == basis) & (b.density == den) & (b.artefact_def == "phase3")]
    sens = []
    if basis == "thickness" and len(mp):
        sens.append(f"MP-map value {_fmt(mp.price.iloc[0])} %")
    if len(old):
        sens.append(f"Phase 3 artefact flag: band {_fmt(old.band_lo.iloc[0])}–{_fmt(old.band_hi.iloc[0])} %")
    pv = pb[(pb.key == key) & (pb.label == label) & (pb.basis == basis) & (pb.density == den)]
    for v in practical.VARIANTS[1:]:
        x = pv[pv.practical == v]
        if len(x):
            sens.append(f"{v} {_fmt(x.price.iloc[0])} %")
    dband = ""
    if basis == "thickness":
        d = db[(db.key == key) & (db.label == label)]
        if len(d):
            dband = f"{_fmt(d.density_p05.iloc[0])}–{_fmt(d.density_p95.iloc[0])}"
            sens.append(f"density model (copula, 5–95 %) {dband} %")
    unit = r.delta_unit
    return {"id": rid, "block": "headline", "dossier_result": dossier,
            "statement": statement or f"Price of {LABEL_TEXT.get(label, label)}, {where}, {basis} basis",
            "quantity": "price", "value": r.price, "value_pct": 100 * r.price, "unit": "%",
            "band_lo_pct": 100 * r.band_lo, "band_hi_pct": 100 * r.band_hi, "band_type": "credible (R1–R3 × 3 kinds × coherent; Phase 4a flag)",
            "density_band_pct": dband,
            "delta_abs": r.delta_abs, "delta_unit": unit,
            "delta_band": f"{r.delta_band_lo:.4g}–{r.delta_band_hi:.4g} {unit}",
            "delta_note": f"at T = {r.T_for_delta:g}", "basis": basis, "frontier_kind": "best_clean",
            "reference_set": f"B4-internal ({label})", "reference_compounds": f"{r.formula_den} → {r.formula_num}",
            "reference_mpids": f"{r.mpid_den} → {r.mpid_num}",
            "density_source": "measured-density point" if basis == "thickness" else "n/a (areal)",
            "fluorescence_corrected": "no (narrow beam)", "sensitivity_axes": "; ".join(AXES_PRICE + (("density model",) if basis == "thickness" else ())),
            "sensitivity_values": "; ".join(sens), "band_lo_set_by": r.band_lo_set_by, "band_hi_set_by": r.band_hi_set_by,
            "source_file": "data/phase4a/S/bands.parquet; data/phase4a/S/practical_bands.parquet" + ("; data/phase4a/S/density_bands.csv" if basis == "thickness" else ""),
            "source_key": f"key={key}; label={label}; basis={basis}; density={den}; artefact_def=phase4a",
            "artefact_reference": _artefact([r.mpid_den, r.mpid_num])}


def price_rows() -> list[dict]:
    b, pb, db = _bands(), _pbands(), _dbands()
    rows, n = [], 0
    dk = practical.diag_head_key()
    cells = dict(HEAD_KEYS); cells[dk] = "DIAG RQR6, 0.35 mm Pb target (air kerma)"
    for key, where in cells.items():
        for label in LABELS:
            for basis in ("thickness", "areal"):
                n += 1
                rows.append(_price_row(b, pb, db, key, label, basis, f"H{n:02d}",
                                       {"RAD|B4": "R2.3" if key != dk else "R2.5", "PB|B4": "R2.1",
                                        "STAB|B4": "R2.6", "EXP|B4": "R2.6", "GAP(3.1)|B4": "R2.2",
                                        "ABUND(1)|B4": "R2.3" if key != dk else "R2.5"}[label], where))
    # element-exclusion filter at the radionuclides (the only non-zero ones)
    for key, where in HEAD_KEYS.items():
        for el, basis in (("Ir", "thickness"), ("Bi", "areal")):
            n += 1
            rows.append(_price_row(b, pb, db, key, f"EXCL({el})|B4", basis, f"H{n:02d}", "R2.7", where,
                                   f"Element-exclusion filter: excluding {el} from B4, {where}, {basis} basis"))
    return rows


def diag_range_rows(start: int) -> list[dict]:
    """Median and range over the 36 DIAG variants of each B4-internal price (areal, headline kind)."""
    b = _bands()
    d = b[(b.scenario == "DIAG") & (b.basis == "areal") & (b.artefact_def == "phase4a") & b.label.isin(LABELS)]
    rows, n = [], start
    for label in LABELS:
        x = d[d.label == label]
        n += 1
        rows.append({"id": f"H{n:02d}", "block": "headline", "dossier_result": "R2.5",
                     "statement": f"Price of {LABEL_TEXT[label]} over the 36 DIAG variants (RQR5/6/8/9 × 3 angles × 3 Pb targets), areal basis",
                     "quantity": "price (median over variants)", "value": float(x.price.median()),
                     "value_pct": 100 * float(x.price.median()), "unit": "%",
                     "band_lo_pct": 100 * float(x.band_lo.min()), "band_hi_pct": 100 * float(x.band_hi.max()),
                     "band_type": "credible band, envelope over the 36 variants",
                     "density_band_pct": "", "delta_abs": float(x.delta_abs.median()), "delta_unit": "g/cm2",
                     "delta_band": f"{x.delta_abs.min():.4g}–{x.delta_abs.max():.4g} g/cm2",
                     "delta_note": "at each variant's Pb-relative target", "basis": "areal", "frontier_kind": "best_clean",
                     "reference_set": f"B4-internal ({label})",
                     "reference_compounds": ", ".join(sorted(set(x.formula_den + " → " + x.formula_num)))[:400],
                     "reference_mpids": "", "density_source": "n/a (areal)", "fluorescence_corrected": "no (narrow beam)",
                     "sensitivity_axes": "; ".join(AXES_PRICE[:4]), "sensitivity_values": f"headline range over variants {_fmt(x.price.min())}–{_fmt(x.price.max())} %",
                     "band_lo_set_by": "", "band_hi_set_by": "", "source_file": "data/phase4a/S/bands.parquet",
                     "source_key": f"scenario=DIAG; label={label}; basis=areal; artefact_def=phase4a",
                     "artefact_reference": _artefact(list(x.mpid_den) + list(x.mpid_num))})
    return rows


# ---------------------------------------------------------------------------------------------
# energy-structure rows (single lines, R2, best_clean, phase4a artefact flag)
# ---------------------------------------------------------------------------------------------
def _mono(label: str, basis: str) -> pd.DataFrame:
    b = _bands()
    den = "measured" if basis == "thickness" else "any"
    x = b[(b.scenario == "MONO") & (b.label == label) & (b.basis == basis) & (b.density == den)
          & (b.artefact_def == "phase4a")].sort_values("energy_keV")
    return x


def _zero_ranges(x: pd.DataFrame, tol: float = 1e-12) -> str:
    z = (x.price.abs() <= tol).to_numpy()
    e = x.energy_keV.to_numpy()
    out, i = [], 0
    while i < len(z):
        if z[i]:
            j = i
            while j + 1 < len(z) and z[j + 1]:
                j += 1
            out.append(f"{e[i]:g}–{e[j]:g}" if j > i else f"{e[i]:g}")
            i = j + 1
        else:
            i += 1
    return ", ".join(out)


def energy_rows(start: int) -> list[dict]:
    rows, n = [], start

    def row(stmt, label, basis, sub: pd.DataFrame, dossier, value_kind="max"):
        nonlocal n
        n += 1
        if value_kind == "max":
            i = sub.price.idxmax(); r = sub.loc[i]
            val, where = r.price, f"maximum at {r.energy_keV:g} keV"
        else:
            r = sub.iloc[0]; val, where = r.price, f"at {r.energy_keV:g} keV"
        return {"id": f"H{n:02d}", "block": "headline", "dossier_result": dossier, "statement": stmt,
                "quantity": f"price ({where})", "value": float(val), "value_pct": 100 * float(val), "unit": "%",
                "band_lo_pct": 100 * float(sub.band_lo.min()), "band_hi_pct": 100 * float(sub.band_hi.max()),
                "band_type": "credible band, envelope over the energy range",
                "density_band_pct": "", "delta_abs": float(r.delta_abs), "delta_unit": r.delta_unit,
                "delta_band": f"{sub.delta_abs.min():.4g}–{sub.delta_abs.max():.4g} {r.delta_unit} over the range",
                "delta_note": "single line, T = 0.01", "basis": basis, "frontier_kind": "best_clean",
                "reference_set": f"B4-internal ({label})", "reference_compounds": f"{r.formula_den} → {r.formula_num}",
                "reference_mpids": f"{r.mpid_den} → {r.mpid_num}",
                "density_source": "measured-density point" if basis == "thickness" else "n/a (areal)",
                "fluorescence_corrected": "no (narrow beam)", "sensitivity_axes": "; ".join(AXES_PRICE[:4]),
                "sensitivity_values": f"range over the energies {_fmt(sub.price.min())}–{_fmt(sub.price.max())} %; zero at {_zero_ranges(sub) or 'none'} keV",
                "band_lo_set_by": "", "band_hi_set_by": "", "source_file": "data/phase4a/S/bands.parquet",
                "source_key": f"scenario=MONO; label={label}; basis={basis}; {sub.energy_keV.min():g}–{sub.energy_keV.max():g} keV",
                "artefact_reference": _artefact([r.mpid_den, r.mpid_num])}

    rt = _mono("RAD|B4", "thickness")
    out = [row("RAD given B4, thickness, single lines 20–63 keV (actinide K edges)", "RAD|B4", "thickness",
               rt[rt.energy_keV <= 63.0], "R2.4"),
           row("RAD given B4, thickness, single lines 63.5–300 keV", "RAD|B4", "thickness",
               rt[(rt.energy_keV >= 63.5) & (rt.energy_keV <= 300)], "R2.4"),
           row("RAD given B4, thickness, single lines above 300 keV", "RAD|B4", "thickness",
               rt[rt.energy_keV > 300], "R2.4")]
    ra = _mono("RAD|B4", "areal")
    out.append(row("RAD given B4, areal mass, single lines 20 keV–10 MeV", "RAD|B4", "areal", ra, "R2.4"))
    for lab, txt in (("GAP(3.1)|B4", "GAP(3.1) given B4"), ("ABUND(1)|B4", "ABUND(1) given B4")):
        g = _mono(lab, "thickness")
        out.append(row(f"{txt}, thickness, single lines ≥ 100 keV", lab, "thickness", g[g.energy_keV >= 100], "R2.4"))
        ga = _mono(lab, "areal")
        out.append(row(f"{txt}, areal mass, single lines ≥ 300 keV", lab, "areal", ga[ga.energy_keV >= 300], "R2.4"))
    pbm = _mono("PB|B4", "areal")
    out.append(row("Lead-free (PB) given B4, areal mass, narrow beam, 88.5–90.5 keV window", "PB|B4", "areal",
                   pbm[(pbm.energy_keV >= 88.5) & (pbm.energy_keV <= 90.5)], "R2.1"))
    pv = _pbands()
    tl = pv[(pv.practical == "excl_Tl") & (pv.scenario == "MONO") & (pv.label == "PB|B4") & (pv.basis == "areal")
            & pv.energy_keV.between(88.5, 90.5)]
    if len(tl):
        j = tl.price.idxmax()
        out[-1]["sensitivity_values"] += (f"; Tl excluded (element-exclusion sensitivity): max {_fmt(tl.price.max())} % "
                                          f"({tl.loc[j, 'formula_den']} → {tl.loc[j, 'formula_num']})")
    _add_endf_library(out[-1])
    pbt = _mono("PB|B4", "thickness")
    out.append(row("Lead-free (PB) given B4, thickness, single lines 20 keV–10 MeV", "PB|B4", "thickness", pbt, "R2.1"))
    for el in EDGE_ELEMENTS:
        for basis in ("areal", "thickness"):
            x = _mono(f"EXCL({el})|B4", basis)
            out.append(row(f"Element-exclusion filter {el} given B4, {basis}, narrow beam, single lines (maximum; set at the "
                           f"grid node just above the {el} K edge, where single-node values carry several points of "
                           f"cross-section-library interpolation uncertainty, see the xs column)",
                           f"EXCL({el})|B4", basis, x, "R2.7"))
            out[-1]["xs_label"] = f"EXCL({el})|B4 {basis}"
    return out


def _add_endf_library(r: dict) -> None:
    """Widen the narrow-beam Pb-window band by the library axis: the ENDF/B-VIII.0 analytic narrow-beam price of
    the Monte Carlo case 5 (Tl vs Pb at 89.5 keV)."""
    f = common.P4 / "mc" / "metrics.csv"
    if not f.exists():
        return
    m = pd.read_csv(f)
    x = m[(m.case_id == "c5_89keV") & m.metric.str.startswith("Tl/Pb") & (m.estimator == "T_endf_narrow")]
    if x.empty:
        return
    v = 100 * float(x.value.iloc[0])
    r["band_lo_pct"] = min(r["band_lo_pct"], v); r["band_hi_pct"] = max(r["band_hi_pct"], v)
    r["band_type"] += " ∪ library (ENDF/B-VIII.0 analytic, MC case 5)"
    r["sensitivity_values"] += f"; ENDF/B-VIII.0 analytic narrow-beam {v:.2f} % at 89.5 keV (MC case 5)"
    r["source_file"] += "; data/phase4a/mc/metrics.csv"


def corrected_rows(start: int) -> list[dict]:
    """Fluorescence-corrected values from stage 3 (E₁, K + L), carried over to the B4-internal reference only
    where the corrected frontier compounds are members of the B4-internal sets (checked row by row)."""
    rows, n = [], start
    u = c3.universe()
    pw = pd.read_csv(S / "pb_window_b4.csv")                       # analysis4.edge4 (B4-internal, phase4a artefact flag)
    win = pw.energy_keV.between(88.5, 90.5)
    for model, label in (("narrow", "narrow beam"), ("E1", "fluorescence-corrected (first order, E₁, K + L)")):
        w = pw[win & (pw.model == model) & (pw.kind == "best_clean")]
        i = w.price.idxmax(); r = w.loc[i]
        allw = pw[win & (pw.model == model)]
        n += 1
        rows.append({"id": f"H{n:02d}", "block": "headline", "dossier_result": "R2.1",
                     "statement": f"Lead-free (PB) given B4, areal mass, 88.5–90.5 keV window, {label}",
                     "quantity": f"price (maximum at {r.energy_keV:g} keV, T = {r['T']:g})", "value": float(r.price),
                     "value_pct": 100 * float(r.price), "unit": "%",
                     "band_lo_pct": 100 * float(allw.price.min()), "band_hi_pct": 100 * float(allw.price.max()),
                     "band_type": f"window energies × kinds (best, best_clean, median50) × T 0.1/0.01 ({model})",
                     "density_band_pct": "", "delta_abs": float("nan"), "delta_unit": "g/cm2",
                     "delta_band": "not computed (corrected requirement is T-dependent)", "delta_note": "", "basis": "areal",
                     "frontier_kind": "best_clean", "reference_set": "B4-internal (PB|B4)",
                     "reference_compounds": f"{r.base_formula} → {r.constrained_formula}",
                     "reference_mpids": f"{r.base_mpid} → {r.constrained_mpid}",
                     "density_source": "n/a (areal)",
                     "fluorescence_corrected": "no (narrow beam)" if model == "narrow" else "yes (first order; no secondary fluorescence, no M lines)",
                     "sensitivity_axes": "escape kernel E₁/E₂; T 0.1/0.01; kinds",
                     "sensitivity_values": f"E₂ max {_fmt(pw[win & (pw.model == 'E2')].price.max())} %; all cells certified: {bool(pw[win].certified.all())}",
                     "band_lo_set_by": "", "band_hi_set_by": "", "source_file": "data/phase4a/S/pb_window_b4.csv",
                     "source_key": f"model={model}; kind=best_clean; 88.5–90.5 keV",
                     "artefact_reference": _artefact([r.base_mpid, r.constrained_mpid]),
                     "xs_label": "PB|B4 areal window" if model == "narrow" else ""})
        if model == "narrow":
            _add_endf_library(rows[-1])
    rows += _corrected_lb_rows(n, ("U",))
    return rows


def _corrected_lb_rows(n: int, elements) -> list[dict]:
    """Stage 3 corrected element-exclusion cells (E₁, K + L; thickness at MP densities). The stage 3 file labels
    the practical baseline 'F0&B4' (= R2 ∧ B4): q*(B4 ∧ ¬X)/q*(B4) − 1, the B4-internal definition (B4 contains
    no artefact entry under either flag, so best = best_clean)."""
    u = c3.universe()
    rows = []
    lb = pd.read_csv(P3 / "E" / "load_bearing_corrected.csv")
    for el in elements:
        base = "F0" if el == "U" else "F0&B4"
        x = lb[(lb.element == el) & (lb.baseline == base) & (lb.model == "E1") & (lb.kind.isin(["best", "best_clean"]))]
        if x.empty:
            continue
        n += 1
        i = x.price.idxmax(); r = x.loc[i]
        rows.append({"id": f"H{n:02d}", "block": "headline" if base != "F0" else "appendix", "dossier_result": "R2.7",
                     "statement": f"Element-exclusion filter {el} against {'B4 (Phase 3 F0∩B4 = B4-internal definition)' if base != 'F0' else 'F0'}, thickness at MP densities, fluorescence-corrected (first order, E₁, K + L)",
                     "quantity": f"price (maximum over 3 sampled energies; {r.energy_keV:g} keV, T = {r['T']:g})",
                     "value": float(r.price), "value_pct": 100 * float(r.price), "unit": "%",
                     "band_lo_pct": 100 * float(x.price.min()), "band_hi_pct": 100 * float(x.price.max()),
                     "band_type": "3 sampled energies × T 0.1/0.01 × kinds best/best_clean under E₁",
                     "density_band_pct": "", "delta_abs": float("nan"), "delta_unit": "cm",
                     "delta_band": "not computed for corrected cells", "delta_note": "", "basis": "thickness",
                     "frontier_kind": "best / best_clean (Phase 3 flag)", "reference_set": ("B4-internal (element-exclusion filter)" if base != "F0" else "F0 (element-exclusion filter) — headroom appendix"),
                     "reference_compounds": f"{r.best_base_formula} → {r.best_constrained_formula}",
                     "reference_mpids": f"{u.material_id.iloc[int(r.row_base)]} → {u.material_id.iloc[int(r.row_constrained)]}",
                     "density_source": "MP (Phase 3 edge-window run)", "fluorescence_corrected": "yes (first order; no secondary fluorescence, no M lines)",
                     "sensitivity_axes": "escape kernel E₁/E₂; T 0.1/0.01; kinds",
                     "sensitivity_values": f"narrow beam {_fmt(x.p5_narrow_this_kind.max())} %; E₂ {_fmt(lb[(lb.element == el) & (lb.baseline == base) & (lb.model == 'E2')].price.max())} %",
                     "band_lo_set_by": "", "band_hi_set_by": "", "source_file": "data/phase3/E/load_bearing_corrected.csv",
                     "source_key": f"element={el}; baseline={base}; model=E1", "artefact_reference": _artefact(
                         [u.material_id.iloc[int(r.row_base)], u.material_id.iloc[int(r.row_constrained)]])})
    return rows


def corrected_b4_rows(start: int) -> list[dict]:
    """The corrected W/Ta/Hf windows against B4 (appended after the MC rows so earlier row ids stay stable)."""
    return _corrected_lb_rows(start, EDGE_ELEMENTS)


# ---------------------------------------------------------------------------------------------
# non-price rows
# ---------------------------------------------------------------------------------------------
def _np(rid, dossier, statement, value, unit, lo, hi, band_type, delta, delta_unit, delta_note, basis, axes, sens,
        src, key, fl="no (narrow beam)", ref="n/a (not a price)", refc="", dens="n/a", kind="n/a", block="headline"):
    return {"id": rid, "block": block, "dossier_result": dossier, "statement": statement, "quantity": "value",
            "value": value, "value_pct": float("nan"), "unit": unit, "band_lo_pct": lo, "band_hi_pct": hi,
            "band_type": band_type, "density_band_pct": "", "delta_abs": delta, "delta_unit": delta_unit,
            "delta_band": "", "delta_note": delta_note, "basis": basis, "frontier_kind": kind, "reference_set": ref,
            "reference_compounds": refc, "reference_mpids": "", "density_source": dens, "fluorescence_corrected": fl,
            "sensitivity_axes": axes, "sensitivity_values": sens, "band_lo_set_by": "", "band_hi_set_by": "",
            "source_file": src, "source_key": key, "artefact_reference": False}


def pb_relative_rows(start: int) -> list[dict]:
    """Pb-relative thickness (narrow-beam) at the measured-density point for named materials (R2 rows)."""
    cols = refsets.columns()
    u = c3.universe()
    r2 = c3.base_mask("R2")
    rows, n = [], start
    q = {}
    for key in HEAD_KEYS:
        cc = cols[(cols.key == key) & cols.coherent]
        q[key] = (refsets.requirement(cc, "thickness", "measured")[:, 0], float(cc.pb_thickness_cm.iloc[0]))
    for f in PB_REL_MATERIALS:
        idx = np.flatnonzero((u.formula_pretty.to_numpy() == f) & r2)
        if idx.size != 1:
            raise RuntimeError(f"{f}: {idx.size} R2 rows")
        i = int(idx[0])
        n += 1
        vals = {k: q[k][0][i] / q[k][1] for k in HEAD_KEYS}
        thick = {k: q[k][0][i] for k in HEAD_KEYS}
        mpv = refsets.requirement(cols[(cols.key == "CS137|v38") & cols.coherent], "thickness", "mp")[i, 0] / q["CS137|v38"][1]
        rows.append(_np(f"H{n:02d}", "R2.1", f"Pb-relative thickness (narrow-beam) of {f} at the measured-density point, Cs-137 / Co-60, T = 0.01",
                        vals["CS137|v38"], "ratio (Cs-137); Co-60 in sensitivity column", *_minmax([vals["CS137|v38"], mpv]),
                        "measured-density point vs MP density (Cs-137)", thick["CS137|v38"], "cm",
                        f"absolute thickness for T = 0.01 at Cs-137 (Co-60 {thick['CO60|v42']:.3f} cm); Pb {q['CS137|v38'][1]:.3f} / {q['CO60|v42'][1]:.3f} cm",
                        "thickness", "density point vs MP", f"Co-60 {vals['CO60|v42']:.3f}; MP-density Cs-137 {refsets.requirement(cols[(cols.key == 'CS137|v38') & cols.coherent], 'thickness', 'mp')[i, 0] / q['CS137|v38'][1]:.3f}",
                        "data/processed (Table C profiles) × data/phase3 density model point; data/phase4a/S/columns.csv",
                        f"material_id={u.material_id.iloc[i]}; variants 38/42", dens="measured-density point",
                        refc=f"{f} ({u.material_id.iloc[i]}) vs Pb at 11.34 g/cm³"))
    return rows


def phase3_rows(start: int) -> list[dict]:
    n = start
    rows = []

    def nid():
        nonlocal n
        n += 1
        return f"H{n:02d}"
    L = "data/phase3/L/summary.json"
    rows.append(_np(nid(), "R1.1", "Energy at which composition and density have equal direct shares of var(log μ) over the map (R2, with coherent)",
                    _jget(L, "item1/crossing_R2_wcoh_keV"), "keV", *_jget(L, "item1/crossing_range_keV"),
                    "R1/R2/R3 × coherent on/off", float("nan"),
                    "", "n/a (not a price; the value is itself absolute)", "n/a (population moments)",
                    "rules; coherent; eight population weightings", f"population weightings {_jget('data/phase3/RW/summary.json', 'range_wcoh_keV')[0]:.1f}–{_jget('data/phase3/RW/summary.json', 'range_wcoh_keV')[1]:.1f} keV",
                    f"{L}; data/phase3/RW/summary.json", "item1/crossing_R2_wcoh_keV; item1/crossing_range_keV; range_wcoh_keV"))
    rows.append(_np(nid(), "R1.2", "Share of var(log μ/ρ) carried by the photoelectric term at the crossing",
                    _jget(L, "item1/frac_var_from_pe_term_at_crossing"), "fraction", float("nan"), float("nan"),
                    "none (single population statistic)", float("nan"), "",
                    "n/a (not a price)", "n/a", "Z-fit range; R2 vs R3",
                    f"median share IQR {_jget(L, 'item1/pe_share_IQR_at_crossing')[0]:.2f}–{_jget(L, 'item1/pe_share_IQR_at_crossing')[1]:.2f}; share passes ½ at {_jget(L, 'item1/energy_median_pe_share_half_keV')[0]:.0f} keV",
                    L, "item1/frac_var_from_pe_term_at_crossing; item1/pe_share_median_at_crossing"))
    p = "item3/primary"
    rows.append(_np(nid(), "R1.3", "Polymorph density spread, DFT vs measured (1,212 formulas with ≥ 2 MP polymorphs matched to distinct COD structures), median ρ_max/ρ_min − 1",
                    _jget(L, f"{p}/spread_dft/median"), "fraction (DFT)",
                    *_minmax([_jget(L, f"item3/{s}/spread_dft/median") for s in POLY_SUBSETS]),
                    "median DFT spread over the tested subsets", float("nan"), "",
                    "n/a (not a price)", "thickness only (structure acts through density)",
                    "e_hull ≤ 0.05; no HP/HT marker; 2 vs ≥ 3 polymorphs; |δ| ≤ 0.2",
                    f"measured (COD) median spread {100 * _jget(L, f'{p}/spread_cod/median'):.2f} %; pairwise ordering preserved|δ| ≤ 0.2",
                    f"pairwise ordering preserved {_jget(L, f'{p}/frac_pairs_concordant'):.3f}; Spearman of spreads {_jget(L, f'{p}/spearman_spread_dft_vs_cod'):.2f}",
                    L, f"{p}/spread_dft/median; {p}/spread_cod/median; {p}/frac_pairs_concordant"))
    rows.append(_np(nid(), "R1.4", "Non-radioactive R2 multi-element compounds exceeding the non-radioactive elemental envelope (47–67 keV) that survive the first-order fluorescence correction (E₁, K + L, T = 0.1, nominal density)",
                    _jget(L, "item2/survival_T0.1_multi_element/n_R2_surviving"), "count", _jget(L, "item2/density_axis_ranges_R2_multi_element/survives_primary_T0.1/min"),
                    _jget(L, "item2/density_axis_ranges_R2_multi_element/survives_primary_T0.1/max"), "five density cases",
                    _jget(L, "item2/survival_T0.1_multi_element/n_R2"), "count", "of this many narrow-beam exceeders",
                    "thickness", "escape kernel; K only; density; T", "E₂ K + L 100; E₁ K only 45 (Phase 3 R1.4)",
                    L, "item2/survival_T0.1_multi_element; item2/density_axis_ranges_R2_multi_element/survives_primary_T0.1",
                    fl="yes (first order; no secondary fluorescence, no M lines)", dens="nominal (MP) with a five-case density axis"))
    C = "data/phase3/C/summary.json"
    rows.append(_np(nid(), "R3.1", "DIAG lanthanide plateau: advantage of the best B4 lanthanide compound over the best non-Pb textbook material, areal mass, fluorescence-corrected (first order, E₁, K + L), median over 36 variants. Weakened by Monte Carlo: like-for-like at RQR5 0.25 mm the advantage is below 10 % (H112), so the first-order count of variants ≥ 10 % is not robust and the median is biased high",
                    _jget(C, "lanthanide_advantage_over_best_textbook/after/range/median"), "fraction",
                    _jget(C, "lanthanide_advantage_over_best_textbook/after/range/min"), _jget(C, "lanthanide_advantage_over_best_textbook/after/range/max"),
                    "range over the 36 variants", _jget(C, "lanthanide_advantage_over_best_textbook/after/n_ge_10pct"), "variants",
                    "variants with advantage ≥ 10 % (of 36) under the first-order model only; not robust under Monte Carlo", "areal", "fixed-T vs like-for-like; air gap; E₁/E₂; K only; responses",
                    f"narrow beam median {_jget(C, 'lanthanide_advantage_over_best_textbook/before/range/median'):.3f}; knife-edge at RQR5 0.25 mm (10.4 %)",
                    C, "lanthanide_advantage_over_best_textbook/after", fl="yes (first order; no secondary fluorescence, no M lines)",
                    ref="B4 (lanthanide) vs textbook materials", kind="best"))
    rows[-1]["xs_label"] = "DIAG advantage corrected (estimate)"
    rows.append(_np(nid(), "R3.1", "DIAG lanthanide plateau: advantage of the best B4 lanthanide compound over the best non-Pb textbook material, areal mass, narrow beam, median over 36 variants",
                    _jget(C, "lanthanide_advantage_over_best_textbook/before/range/median"), "fraction",
                    _jget(C, "lanthanide_advantage_over_best_textbook/before/range/min"), _jget(C, "lanthanide_advantage_over_best_textbook/before/range/max"),
                    "range over the 36 variants", _jget(C, "lanthanide_advantage_over_best_textbook/before/n_ge_10pct"), "variants",
                    "variants with advantage ≥ 10 % (of 36)", "areal", "36 variants", "", C,
                    "lanthanide_advantage_over_best_textbook/before", ref="B4 (lanthanide) vs textbook materials", kind="best"))
    rows[-1]["xs_label"] = "DIAG advantage narrow"
    rows.append(_np(nid(), "R3.2", "Bi / B4 frontier areal-mass ratio over the 36 DIAG variants, fluorescence-corrected (best textbook material in 33 of 36)",
                    _jget(C, "textbook/1/ratio_frontier_min_after"), "ratio (min)", _jget(C, "textbook/1/ratio_frontier_min_after"),
                    _jget(C, "textbook/1/ratio_frontier_max_after"), "range over 36 variants", float("nan"), "",
                    "ratio of areal masses (absolute masses per variant in C/textbook_by_variant.csv)", "areal",
                    "correction on/off", "", C, "textbook/1 (Bi): ratio_frontier_min_after / max_after",
                    fl="yes (first order)", ref="B4 frontier", kind="best"))
    rows.append(_np(nid(), "R3.3", "Lanthanide-level rank agreement across the 36 DIAG variants after correction (pairwise Spearman, median)",
                    _jget(C, "rank_correlation/ln/after/pairwise_spearman/median"), "Spearman ρ",
                    _jget(C, "rank_correlation/ln/after/pairwise_spearman/min"), _jget(C, "rank_correlation/ln/after/pairwise_spearman/max"),
                    "min–max over the 630 variant pairs", float("nan"), "", "n/a (rank statistic)", "areal", "three family granularities; correction on/off",
                    "", C, "rank_correlation/ln/after/pairwise_spearman/median", fl="yes (first order)"))
    D = "data/phase3/D/summary.json"
    rows.append(_np(nid(), "R4.1", "Median DFT density bias on compounds, δ = ρ_MP/ρ_COD − 1 (21,489 structure-matched compounds)",
                    _jget(D, "overall/median"), "fraction", *_jget(D, "overall/subset_median_range"), "range of the median over tested subsets",
                    float("nan"), "", "n/a (relative bias)", "thickness only", "selection rule; markers; temperature; screens",
                    "90 % band −8.41 to +3.14 % (Phase 3 R4.1)", D, "overall/median; overall/subset_median_range", dens="COD"))
    tp = pd.read_csv(P3 / "S4" / "textbook_phases.csv")
    sn = tp[tp.formula == "Sn"].iloc[0]
    rows.append(_np(nid(), "R4.3", "Ambient-phase density ratio, β-Sn vs the R2 representative (α-Sn)",
                    float(sn.density_ratio_ambient_vs_R2), "ratio", float("nan"), float("nan"), "none", float(sn.density - sn.R2_density),
                    "g/cm3", "density difference", "thickness", "", "Sb₂O₃ 0.940, γ-WO₃ 0.991, rutile 1.099 (Phase 3 R4.3)",
                    "data/phase3/S4/textbook_phases.csv", "formula=Sn", dens="MP"))
    tsum = pd.read_csv(P3 / "T" / "tier_summary.csv")
    for sc, vid in (("CS137", 38), ("CO60", 42)):
        for st in ("B4", "B5", "B6"):
            t = tsum[(tsum.scenario == sc) & (tsum.variant_id == vid) & (tsum.rule == "R2") & (tsum["set"] == st)]
            if t.empty:
                continue
            t = t.iloc[0]
            others = tsum[(tsum.scenario == sc) & (tsum.variant_id == vid) & (tsum["set"] == st)]
            rows.append(_np(nid(), "R3.4", f"Thinnest {st} compound at {sc} (T = 0.01, R2): share of density-model draws in which it is thinnest, P(best)",
                            float(t.top_winner_p_copula), f"P(best) of {t.top_winner_copula}",
                            float(others.top_winner_p_copula.min()), float(others.top_winner_p_copula.max()), "R1/R2/R3 (copula)",
                            float("nan"), "", "n/a (probability)", "thickness",
                            "eight density-model modes; rules; T; ABUND 0.5/1/2 ppm",
                            f"no ceiling {t.top_winner_p_copula_noceiling:.2f} ({t.top_winner_copula_noceiling}); unscreened residual {t.top_winner_p_copula_rawresid:.2f} ({t.top_winner_copula_rawresid})",
                            "data/phase3/T/tier_summary.csv", f"scenario={sc}; variant_id={vid}; rule=R2; set={st}",
                            dens="density model (copula)", ref=f"{st} (Phase 3 set; contains no artefact entry)"))
    return rows


def misc_rows(start: int) -> list[dict]:
    """Rows recomputed under stage 4a from refsets outputs: Pb near the practical mass frontier; the
    thickness/mass ratio of the GAP price; artefact share of best-compound frontiers."""
    rows, n = [], start
    b = _bands()
    # Pb vs best non-Pb B4 compound, areal, single lines 100 keV–10 MeV (= 1 / (1 + PB|B4)) when Pb sets B4-PB
    u = c3.universe()
    pb_row = int(np.flatnonzero((u.formula_pretty.to_numpy() == "Pb") & c3.base_mask("R2"))[0])
    cols = refsets.columns()
    mono = cols[(cols.kind == "mono") & cols.coherent & (cols.energy_keV >= 100)]
    q = c3.mono_requirement("areal", True, e_idx=mono.e_idx.to_numpy())
    b4 = refsets.practical_sets(headroom=False)["B4"] & c3.base_mask("R2") & common.clean_mask()
    best = np.min(np.where(b4[:, None], q, np.inf), axis=0)
    ratio = q[pb_row] / best - 1.0
    noh = b4 & ~c3.core.contains("H")
    best_noh = np.min(np.where(noh[:, None], q, np.inf), axis=0)
    ratio_noh = q[pb_row] / best_noh - 1.0
    n += 1
    rows.append(_np(f"H{n:02d}", "R2.1", "Extra areal mass Pb needs relative to the best B4 (lead-free practical) compound, single lines 100 keV–10 MeV (the upper end is set where the B4 mass frontier is NH₃·H₂O; see the H-excluded range)",
                    float(np.median(ratio)), "fraction (median over energies)", float(ratio.min()), float(ratio.max()),
                    "range over energies", float(np.median((q[pb_row] - best) * math.log(100))), "g/cm2",
                    "median extra areal mass at T = 0.01", "areal", "rules/kinds not varied (Pb fixed); H-containing compounds excluded",
                    f"H-containing compounds excluded: median {_fmt(np.median(ratio_noh))} %, range {_fmt(ratio_noh.min())}–{_fmt(ratio_noh.max())} %",
                    "data/processed (μ/ρ); data/phase4a/S/columns.csv", f"Pb {u.material_id.iloc[pb_row]} vs B4 best_clean R2",
                    ref="B4 (best_clean)", refc=f"Pb ({u.material_id.iloc[pb_row]}) vs B4 frontier", kind="best_clean"))
    # GAP thickness / mass ratio at Cs-137 (MP densities and measured)
    for den in ("mp", "measured"):
        t = b[(b.key == "CS137|v38") & (b.label == "GAP(3.1)|B4") & (b.basis == "thickness") & (b.density == den) & (b.artefact_def == "phase4a")].iloc[0]
        a = b[(b.key == "CS137|v38") & (b.label == "GAP(3.1)|B4") & (b.basis == "areal") & (b.artefact_def == "phase4a")].iloc[0]
        n += 1
        rows.append(_np(f"H{n:02d}", "R2.2", f"Derived from the GAP(3.1) given B4 price rows (not an independent result). Transparency: thickness-to-mass price ratio (1 + P_thickness)/(1 + P_mass) of GAP(3.1) given B4 at Cs-137 ({'MP densities' if den == 'mp' else 'measured-density point'})",
                        (1 + t.price) / (1 + a.price), "ratio", float("nan"), float("nan"), "see the two price rows' bands",
                        float(t.delta_abs), "cm", f"thickness increment at T = 0.01 (mass increment {a.delta_abs:.3f} g/cm²)",
                        "both", "as the GAP rows", f"P_thickness {_fmt(t.price)} %, P_mass {_fmt(a.price)} %",
                        "data/phase4a/S/bands.parquet", f"key=CS137|v38; label=GAP(3.1)|B4; density={den}",
                        ref="B4-internal (GAP(3.1)|B4)", refc=f"{t.formula_den} → {t.formula_num}", kind="best_clean",
                        dens="MP" if den == "mp" else "measured-density point"))
    # artefact share of the unconstrained best-compound frontier (R2.8) under the phase4a artefact flag
    ft = pd.read_parquet(S / "frontiers.parquet", columns=["col", "basis", "density", "rule", "set", "artefact_def", "kind", "row"])
    s = ft[(ft["set"] == "F0") & (ft.kind == "best") & (ft.artefact_def == "phase4a") & (ft.row >= 0)]
    cc = cols.set_index("col")
    s = s.assign(scenario=cc.loc[s.col].scenario.to_numpy(), coherent=cc.loc[s.col].coherent.to_numpy())
    s = s[s.coherent]
    flags = {d: common.artefact_mask(d) for d in common.ARTEFACT_DEFS}
    art = flags["phase4a"]
    for sc, basis, den in (("CS137", "thickness", "mp"), ("DIAG", "areal", "any"), ("MONO", "thickness", "mp"), ("MONO", "areal", "any")):
        xa = s[(s.scenario.isin([sc] if sc != "CS137" else ["CS137", "CO60"])) & (s.basis == basis) & (s.density == den)]
        x = xa[xa.rule == "R2"]
        alt = [float(flags[d][xa[xa.rule == r].row.to_numpy()].mean()) for d in flags for r in refsets.RULES]
        n += 1
        rows.append(_np(f"H{n:02d}", "R2.8", f"Share of unconstrained best-compound (F0, kind best) frontier cells set by an artefact entry, Phase 4a flag: {('radionuclide profiles' if sc == 'CS137' else 'DIAG profiles' if sc == 'DIAG' else 'single lines')}, {basis}",
                        float(art[x.row.to_numpy()].mean()), "fraction", min(alt), max(alt), "rules R1/R2/R3 × artefact flag (Phase 3, Phase 4a)",
                        float(len(x)), "cells", "number of cells", basis, "Phase 3 vs Phase 4a flag; rules", "",
                        "data/phase4a/S/frontiers.parquet", f"set=F0; kind=best; rule=R2; coherent; {sc}; {basis}; density={den}",
                        kind="best", dens="MP" if den == "mp" else "n/a"))
    return rows


def step3_rows(start: int) -> list[dict]:
    """Practical-set definitions (the practical-set module): abundance curve, natural-radioactivity flag, H exclusion; and the
    MP-map vs measured-density gap of the thickness headlines (R4.2)."""
    rows, n = [], start

    def nid():
        nonlocal n
        n += 1
        return f"H{n:02d}"
    pb = _pbands()
    pr = pb[(pb.practical == "primary") & pb.label.str.startswith("ABUND(")]
    for key, where in HEAD_KEYS.items():
        for basis in ("thickness", "areal"):
            den = "measured" if basis == "thickness" else "any"
            x = pr[(pr.key == key) & (pr.basis == basis) & (pr.density == den)].copy()
            x["a"] = x.label.str.extract(r"ABUND\(([\d.]+)\)")[0].astype(float)
            x = x.sort_values("a")
            one = x[x.a == 1.0].iloc[0]
            rows.append(_np(nid(), "R2.3", f"Scarcity proxy (crustal abundance) curve: ABUND(a) given B4 for a = 0.1–10 ppm, {where}, {basis} (value at 1 ppm)",
                            float(one.price), "fraction", float(x.price.min()), float(x.price.max()), "range over the thresholds 0.1–10 ppm",
                            float(one.delta_abs), one.delta_unit, "at 1 ppm, T = 0.01", basis, "threshold a ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10} ppm",
                            "; ".join(f"{a:g} ppm {_fmt(p)} % ({f})" for a, p, f in zip(x.a, x.price, x.formula_num)),
                            "data/phase4a/S/practical_bands.parquet", f"practical=primary; key={key}; label=ABUND(a)|B4; basis={basis}",
                            ref="B4-internal (ABUND(a)|B4)", refc=f"{one.formula_den} → {one.formula_num}", kind="best_clean",
                            dens="measured-density point" if basis == "thickness" else "n/a (areal)"))
    act = pd.read_csv(S / "natrad_elements.csv")
    for el in ("Rb", "Sm", "Lu", "Re", "La", "In", "K"):
        r = act[act.element == el].iloc[0]
        flag = "moderate and named" if (r.moderate and r.named) else "moderate" if r.moderate else "named" if r.named else "none"
        rows.append(_np(nid(), "R3.4", f"Natural specific activity of {el} relative to natural potassium (flag: {flag})",
                        float(r.ratio_to_K), "ratio to K", float("nan"), float("nan"), "none (nuclear data; lower-limit half-lives give upper bounds)",
                        float(r.A_Bq_per_g), "Bq per g of element", "absolute specific activity", "n/a",
                        "half-life source ENSDF; abundance IUPAC (mendeleev)", f"isotopes: {r.radioactive_isotopes}",
                        "data/phase4a/S/natrad_elements.csv", f"element={el}", ref="natural K (31.72 Bq/g)", dens="n/a"))
    nh = pd.read_csv(S / "noH_frontier_changes.csv")
    m = nh[nh.key.str.startswith("MONO") & (nh.basis == "areal")]
    i = m.price_noH_vs_B4.idxmax()
    rows.append(_np(nid(), "R2.4", "H-containing compounds excluded: change of the B4 areal-mass frontier on single lines (maximum)",
                    float(m.loc[i, "price_noH_vs_B4"]), "fraction", float(m.price_noH_vs_B4.min()), float(m.price_noH_vs_B4.max()),
                    "over the energies where the frontier changes", float(m.loc[i, "energy_keV"]), "keV",
                    "energy of the maximum", "areal", "H exclusion on/off",
                    f"frontier changes at {m.energy_keV.min():g}–{m.energy_keV.max():g} keV ({', '.join(sorted(set(m.formula_B4 + ' → ' + m.formula_noH)))})",
                    "data/phase4a/S/noH_frontier_changes.csv", "key=MONO; basis=areal", ref="B4 vs B4 without H", kind="best_clean"))
    for key, where in HEAD_KEYS.items():
        x = nh[(nh.key == key) & (nh.basis == "areal")]
        xs_ = nh[nh.key.str.startswith(key.split("|")[0] + "|") & (nh.basis == "areal")]
        if len(x):
            rows.append(_np(nid(), "R2.4", f"H-containing compounds excluded: change of the B4 areal-mass frontier at {where}",
                            float(x.price_noH_vs_B4.iloc[0]), "fraction", float(xs_.price_noH_vs_B4.min()),
                            float(xs_.price_noH_vs_B4.max()), "over the four transmission targets T = 0.5–0.001",
                            float(x.value_B4.iloc[0] * x.price_noH_vs_B4.iloc[0]), "g/cm2", "extra areal mass", "areal",
                            "H exclusion on/off", f"{x.formula_B4.iloc[0]} → {x.formula_noH.iloc[0]}",
                            "data/phase4a/S/noH_frontier_changes.csv", f"key={key}; basis=areal", ref="B4 vs B4 without H", kind="best_clean"))
    b = _bands()
    th = b[b.key.isin(list(HEAD_KEYS)) & (b.basis == "thickness") & (b.artefact_def == "phase4a") & (b.reference == "B4-internal")]
    w = th.pivot_table(index=["key", "label"], columns="density", values="price", observed=True)
    gap = (w["mp"] - w["measured"]) * 100
    j = gap.abs().idxmax()
    rows.append(_np(nid(), "R4.2", "Largest gap between the MP-map value and the measured-density point among the B4-internal thickness prices at Cs-137/Co-60 (percentage points)",
                    float(gap.loc[j]), "percentage points", float(gap.min()), float(gap.max()), "over all B4-internal labels at both radionuclides",
                    float("nan"), "", "n/a (difference of prices)", "thickness", "density source MP vs measured",
                    f"largest at {j[0]} {j[1]}", "data/phase4a/S/bands.parquet", "density=mp vs measured; reference=B4-internal",
                    dens="MP vs measured"))
    return rows


MC_DOSSIER = {"c1_cs137": "R2.2", "c2_co60": "R2.2", "c3_rqr5": "R3.1", "c4_rqr8": "R3.1", "c5_89keV": "R2.1",
              "c6_cs137_pb": "R2.1"}
MC_EST = ("narrow", "cone1", "cone3", "fl", "fl_all", "broad")


def mc_rows(start: int) -> list[dict]:
    """The Monte Carlo spot check Monte Carlo results (OpenMC, ENDF/B-VIII.0) that stage 4b may quote: each metric's value with the
    fluorescence-inclusive narrow-geometry estimator ('fl', like-for-like with the first-order model), its spread
    over the MC estimators (narrow … broad beam), and the build-up factor B = broad/narrow at the headline thickness."""
    f = common.P4 / "mc" / "metrics.csv"
    if not f.exists():
        return []
    m = pd.read_csv(f)
    pts = pd.read_csv(common.P4 / "mc" / "points.csv")
    rows, n = [], start
    for (cid, metric), g in m.groupby(["case_id", "metric"], sort=False):
        e = g.set_index("estimator")
        if "fl" not in e.index:
            continue
        mc = e.loc[[x for x in MC_EST if x in e.index]]
        dossier = MC_DOSSIER.get(cid, "R3.1")
        if metric.startswith("HfO2 vs BiOF"):
            dossier = "R3.4"
        elif metric.startswith("Pb/Bi"):
            dossier = "R3.2"
        n += 1
        sens = "; ".join(f"{k} {100 * e.loc[k, 'value']:.2f} %" + (f" ± {100 * e.loc[k, 'se']:.2f}" if np.isfinite(e.loc[k, "se"]) else "")
                         for k in e.index)
        rows.append(_np(f"H{n:02d}", dossier, f"Monte Carlo (OpenMC, ENDF/B-VIII.0), case {cid}: {metric}",
                        float(e.loc["fl", "value"]), "fraction", float(mc.value.min()), float(mc.value.max()),
                        "MC estimators: narrow, 1° and 3° cones, with fluorescence (≥ 3.2 keV and all lines), broad beam",
                        float(e.loc["fl", "se"]), "fraction (1 SE)", "statistical standard error of the value",
                        str(g.basis.iloc[0]), "MC estimator; library (XCOM Table C vs ENDF analytic); first-order model",
                        sens, "data/phase4a/mc/metrics.csv", f"case_id={cid}; metric={metric}; estimator=fl",
                        fl="yes (Monte Carlo, full fluorescence cascade)", ref=f"{g.num.iloc[0]} vs {g.den.iloc[0]}",
                        kind="named materials"))
        rows[-1]["mc_status_row"] = "Monte Carlo result"
    b = pts[np.isclose(pts.factor, 1.0) & pts.case_id.isin(["c6_cs137_pb", "c1_cs137", "c2_co60"])]
    for r in b.itertuples():
        n += 1
        allf = pts[(pts.case_id == r.case_id) & (pts.material == r.material)]
        rows.append(_np(f"H{n:02d}", MC_DOSSIER[r.case_id],
                        f"Monte Carlo build-up factor B = broad-beam / narrow-beam air-kerma transmission, {r.material}, case {r.case_id}, at the headline thickness (T = 0.01)",
                        float(r.buildup), "ratio", float(allf.buildup.min()), float(allf.buildup.max()),
                        "over the simulated thicknesses (0.6–1.6 × headline)", float(r.t_cm), "cm",
                        "slab thickness", "thickness", "slab thickness ± 20–60 %", f"B ± {r.buildup_se:.3g} (1 SE)",
                        "data/phase4a/mc/points.csv", f"case_id={r.case_id}; material={r.material}; factor=1.0",
                        fl="yes (Monte Carlo)", ref=f"{r.material}", kind="named material", dens="measured-density point"))
        rows[-1]["mc_status_row"] = "Monte Carlo result"
    return rows


def headroom_rows(start: int) -> list[dict]:
    b, pb, db = _bands(), _pbands(), _dbands()
    rows, n = [], start
    dk = practical.diag_head_key()
    cells = dict(HEAD_KEYS); cells[dk] = "DIAG RQR6, 0.35 mm Pb target"
    for key, where in cells.items():
        for label in HEADROOM:
            for basis in ("thickness", "areal"):
                n += 1
                r = _price_row(b, pb, db, key, label, basis, f"A{n - start:02d}", "R2.3" if key != dk else "R2.5", where,
                               f"Headroom: price of {label.split('|')[0]} against the unconstrained artefact-free frontier F0, {where}, {basis} basis")
                r["block"] = "appendix"
                r["reference_set"] = f"F0 headroom ({label})"
                rows.append(r)
    return rows


def build() -> pd.DataFrame:
    rows = price_rows()
    rows += diag_range_rows(len(rows))
    rows += energy_rows(len(rows))
    rows += corrected_rows(len(rows))
    rows += pb_relative_rows(len(rows))
    rows += misc_rows(len(rows))
    rows += phase3_rows(len(rows))
    rows += step3_rows(len(rows))
    rows += mc_rows(len(rows))
    rows += corrected_b4_rows(len(rows))
    rows += headroom_rows(0)
    t = pd.DataFrame(rows)
    # reference practicality: are both reference compounds members of B4? (RAD rows are not, by construction)
    u = c3.universe()
    in_b4 = dict(zip(u.material_id, c3.bundle_mask("B4")))
    def _practical(s):
        ids = [x.strip() for x in str(s).split("→")] if isinstance(s, str) and "→" in s else []
        return "" if not ids else ("yes" if all(in_b4.get(x, False) for x in ids) else "no")
    t["reference_practical"] = t["reference_mpids"].map(_practical)
    rad = t["reference_set"].astype(str).str.contains(r"\(RAD\|B4\)")
    t.loc[rad, "statement"] = "Actinide headroom inside B4 (the reference admits radioactive elements; not an achievable alternative): " + t.loc[rad, "statement"]
    # every first-order fluorescence-corrected number carries the Monte Carlo finding
    fo = t["fluorescence_corrected"].astype(str).str.startswith("yes (first order") & ~t["statement"].str.contains("Monte Carlo")
    t.loc[fo, "statement"] = t.loc[fo, "statement"] + (" [first-order model: the Monte Carlo spot check finds escaping "
                                                       "fluorescence under-counted by 5–30 %, BLOCKERS O4-5]")
    # a row without a computed band gets an explicit point band, labelled as such
    nob = t.band_lo_pct.isna() | t.band_hi_pct.isna()
    pv = np.where(t.unit.astype(str) == "%", t.value_pct, t.value)
    t.loc[nob, "band_lo_pct"] = pv[nob]
    t.loc[nob, "band_hi_pct"] = pv[nob]
    t.loc[nob, "band_type"] = "point value only: " + t.loc[nob, "band_type"].astype(str) + " (no band computed)"
    t["sensitivity_axes"] = t["sensitivity_axes"].replace("", "none tested")
    frac = t.unit.astype(str).str.startswith("fraction")          # one scale for every relative quantity
    t.loc[frac, "value_pct"] = 100 * t.loc[frac, "value"]
    t.loc[frac, "band_lo_pct"] = 100 * t.loc[frac, "band_lo_pct"]
    t.loc[frac, "band_hi_pct"] = 100 * t.loc[frac, "band_hi_pct"]
    t.loc[frac, "unit"] = "%" + t.loc[frac, "unit"].str[len("fraction"):]
    if "xs_label" not in t:
        t["xs_label"] = ""
    t["xs_label"] = t["xs_label"].fillna("")
    xs = _xs_sens()
    if xs is not None:
        m = dict(zip(xs.match_label, xs.xs_sensitivity))
        t["xs_sensitivity"] = [m.get(k, "not in Step 7 scope") if k else "not in Step 7 scope" for k in t.xs_label]
    else:
        t["xs_sensitivity"] = "not yet computed (Step 7)"
    mc = _mc_status()
    if mc is not None:
        t = t.merge(mc[["id", "mc_status"]], on="id", how="left")
        t["mc_status"] = t["mc_status"].fillna("not tested")
    else:
        t["mc_status"] = "not tested"
    if "mc_status_row" in t:
        t["mc_status"] = t["mc_status_row"].fillna(t["mc_status"])
        t = t.drop(columns="mc_status_row")
    return t


def write_md(t: pd.DataFrame) -> str:
    def cell(x, nd=2):
        if isinstance(x, float):
            return "—" if not np.isfinite(x) else f"{x:.{nd}f}"
        return str(x)
    lines = ["# Phase 4a headline table", "",
             "Generated by `python -m shieldmap.analyze4a` from `data/phase4a/headline_table.csv`; do not edit by hand.",
             "**Phase 4b may quote only numbers that appear in this table.**", "",
             "Conventions (DECISIONS P4A-D-010 to P4A-D-016, P4A-D-020 onward):",
             "- **Primary reference: B4-internal marginals.**",
             "  - For X ∈ {GAP, ABUND, EXCL}: q*(B4 ∧ X)/q*(B4) − 1.",
             "  - For X ∈ {RAD, PB, STAB, EXP}: q*(B4)/q*(B4 without X) − 1.",
             "- **Frontier kind:** best_clean under the Phase 4a artefact flag. **Rule:** R2. **Coherent scattering:** included.",
             "- **Thickness basis:** at the Phase 3 measured-density point (the MP-map value is in the sensitivity column). Areal mass is density-independent.",
             "- **Credible band:** R1/R2/R3 × {best_clean, best_icsd, median50_clean} × coherent on/off. **Density band:** the copula model's 5–95 %.",
             "- **Δ** is the absolute increment at the stated transmission.",
             "- **Beam:** everything is narrow-beam unless the fluorescence column says otherwise.",
             "- Terminology: Pb-relative thickness (narrow-beam); element-exclusion filter; scarcity proxy (crustal abundance).", ""]
    for blk, title in (("headline", "## Headline rows"), ("appendix", "## Appendix: headroom (F0-based) and other non-headline rows")):
        x = t[t.block == blk]
        lines += [title, "", "| id | result | statement | value | band | Δ | basis | reference (compounds) | density | fluor. | xs sens. | MC | source |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in x.itertuples():
            if str(r.unit).startswith("%"):
                val = f"**{r.value_pct:.2f} %**{str(r.unit)[1:]}"
                band = f"{cell(r.band_lo_pct)}–{cell(r.band_hi_pct)} %" + (f"; density {r.density_band_pct} %" if r.density_band_pct else "")
            else:
                val = f"**{cell(r.value, 3)}** {r.unit}"
                band = "—" if not (isinstance(r.band_lo_pct, float) and np.isfinite(r.band_lo_pct)) else f"{cell(r.band_lo_pct, 3)}–{cell(r.band_hi_pct, 3)}"
            d = f"{cell(r.delta_abs, 3)} {r.delta_unit}" if isinstance(r.delta_abs, float) and np.isfinite(r.delta_abs) else (r.delta_band or "—")
            lines.append(f"| {r.id} | {r.dossier_result} | {r.statement} | {val} | {band} ({r.band_type}) | {d} {('(' + r.delta_note + ')') if r.delta_note else ''} | {r.basis} | {r.reference_set}: {r.reference_compounds} | {r.density_source} | {r.fluorescence_corrected} | {r.xs_sensitivity} | {r.mc_status} | `{r.source_file}` [{r.source_key}] |")
        lines.append("")
    lines += ["## Sensitivity values per row", "", "| id | axes tested | values |", "|---|---|---|"]
    for r in t.itertuples():
        lines.append(f"| {r.id} | {r.sensitivity_axes} | {r.sensitivity_values} |")
    lines.append("")
    bad = t[t.artefact_reference.astype(bool) & (t.block == "headline")]
    lines.append(f"**Artefact check:** {len(bad)} headline rows rest on an artefact reference under the Phase 4a flag"
                 + (f": {', '.join(bad.id)}." if len(bad) else "."))
    return "\n".join(lines) + "\n"


def run() -> dict:
    t = build()
    t.to_csv(common.P4 / "headline_table.csv", index=False)
    (common.REPORTS4 / "headline_table.md").write_text(write_md(t), encoding="utf-8")
    return {"n_rows": int(len(t)), "n_headline": int((t.block == "headline").sum()),
            "n_appendix": int((t.block == "appendix").sum()),
            "n_artefact_headline": int((t.artefact_reference.astype(bool) & (t.block == "headline")).sum())}


if __name__ == "__main__":
    print(run())
