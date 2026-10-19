"""LaTeX tables (paper/tables/*.tex) and their CSV copies (data/phase4b/tables/).

  table1   price of each requirement inside the practical set (B4-internal, best_clean, R2) at Cs-137, Co-60
           and the reference diagnostic beam, both bases, value and absolute increment     (main, table*)
  table2   leaders of B4/B5/B6 at Cs-137 and Co-60: P(best), Pb-relative thickness          (main, table)
  tableS1  data sources and definitions                                                     (table)
  tableS2  Monte Carlo cases (a) and the compounds behind the diagnostic frontier (b)        (table*)

Bands left Table 1 for the dataset; the stage 4b headline longtable and the
natural-radioactivity table moved to the dataset and docs/ (natural_radioactivity.md); the stage 4b Table S4 (Monte
Carlo cases) became panel (a) of Table S2.

Number tags follow paper/CONVENTIONS.md: one table row per source line, every number tagged on that line.
Headline numbers use [Hnn] tags; definitional numbers [def:NAME] / [count:NAME] from paper/defs.yaml (entries
written here are merged into the file, entries of other writers are kept); cells generated straight from a data
file use [cell:FILE:ID] with the sidecar paper/tables/FILE.cells.json. run() re-checks every generated line with
paper/check_numbers.py (when present) and returns the failures.

Text reproduced from the headline table goes through one fixed substitution table (TEX_CHARS, UNIT_TEX,
DELTA_UNIT_TEX below); nothing is shortened.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import yaml

from shieldmap import paths

ROOT = paths.ROOT
PAPER = ROOT / "paper"
TAB = PAPER / "tables"
DEFS_FILE = PAPER / "defs.yaml"
P4A = paths.DATA / "phase4a"
P4B = paths.DATA / "phase4b"
P4D = paths.DATA / "phase4d"
CSV_OUT = P4B / "tables"
HEADLINE = P4B / "headline_table_v3.csv"

# ---------------------------------------------------------------------------------------------
# fixed substitution tables (documented in reports/phase4b/step3_tables.md)
# ---------------------------------------------------------------------------------------------
TEX_CHARS = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
    "_": r"\_", "^": r"\^{}", "~": r"\textasciitilde{}", "<": r"$<$", ">": r"$>$", "|": r"$|$",
    "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$", "×": r"$\times$", "−": r"$-$", "–": "--",
    "—": "---", "μ": r"$\mu$", "ρ": r"$\rho$", "δ": r"$\delta$", "α": r"$\alpha$", "β": r"$\beta$",
    "σ": r"$\sigma$", "Δ": r"$\Delta$", "°": r"$^{\circ}$", "·": r"$\cdot$", "∩": r"$\cap$", "∧": r"$\wedge$",
    "≈": r"$\approx$", "±": r"$\pm$", "₀": r"$_0$", "₁": r"$_1$", "₂": r"$_2$", "₃": r"$_3$", "₄": r"$_4$",
    "₆": r"$_6$", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}", "…": r"\ldots{}", " ": r"\,",
    "’": "'", "‘": "`",
}
UNIT_TEX = {          # headline-table unit -> printed unit (S2 value column)
    "%": r"\%", "ratio": "ratio", "ratio to K": "ratio to K", "ratio (min)": "ratio (min)",
    "ratio (Cs-137); Co-60 in sensitivity column": "ratio (Cs-137)",
    "% (median over energies)": r"\% (median over energies)", "% (DFT)": r"\% (DFT)", "chi2": "chi-square",
    "percentage points": "pp", "Spearman ρ": r"Spearman $\rho$", "keV": "keV", "count": "count", "ppm": "ppm",
    "g/cm3": r"g\,cm\textsuperscript{$-$3}", "fraction": "fraction", "eV": "eV", "eV/atom": "eV/atom",
}
DELTA_UNIT_TEX = {
    "cm": "cm", "g/cm2": r"g\,cm\textsuperscript{$-$2}", "g/cm3": r"g\,cm\textsuperscript{$-$3}",
    "fraction (1 SE)": "SE (fraction)", "Bq per g of element": r"Bq\,g\textsuperscript{$-$1}", "cells": "cells",
    "variants": "variants", "count": "count", "keV": "keV", "slab points": "slab points",
}
G_CM2 = r"g\,cm\textsuperscript{$-$2}"
G_CM3 = r"g\,cm\textsuperscript{$-$3}"
# one absolute table font size (8 pt) in both builds: \footnotesize of the 10 pt two-column 5p build,
# \scriptsize of the 12 pt single-column preprint build
FONT_PREAMBLE = (r"\makeatletter",
                 r"\providecommand{\shieldtablefont}{\if@twocolumn\footnotesize\else\scriptsize\fi}",
                 r"\makeatother")


def tex(s) -> str:
    return "".join(TEX_CHARS.get(c, c) for c in str(s))


def formula(f: str) -> str:
    """HfO2 -> HfO$_2$ (digits after a letter or a closing parenthesis become subscripts)."""
    return re.sub(r"(?<=[A-Za-z)])(\d+)", r"$_{\1}$", f)


def fmt(x, sig: int = 2) -> str:
    """Two (or `sig`) significant figures, half-up; no decimals at or above 10**sig; zero printed as 0."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x):
        return "--"
    x = float(x)
    if x == 0:
        return "0"
    ax = abs(x)
    d = 0 if ax >= 10 ** sig else sig - 1 - int(math.floor(math.log10(ax)))
    s = str(Decimal(repr(ax)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return ("$-$" + s) if x < 0 and Decimal(s) != 0 else s


def fmt_trim(x, sig: int = 2) -> str:
    """fmt(), with trailing zeros dropped when the shorter number is exact (0.010 -> 0.01, 1.60 -> 1.6)."""
    s = fmt(x, sig)
    if "." in s and "$" not in s:
        t = s.rstrip("0").rstrip(".")
        if abs(float(t) - float(x)) < 1e-12:
            return t
    return s


def fmt_dec(x, d: int) -> str:
    s = str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return ("$-$" + s.lstrip("-")) if s.startswith("-") else s


def plain(s: str) -> str:
    return s.replace("$-$", "-")


def line(cells: list[str], tags=(), end: str = r" \\") -> str:
    tg = " ".join(f"[{t}]" for t in dict.fromkeys(tags))
    return " & ".join(cells) + end + (f" % {tg}" if tg else "")


def text_line(s: str, tags=()) -> str:
    tg = " ".join(f"[{t}]" for t in dict.fromkeys(tags))
    return s + (f" % {tg}" if tg else "")


class Cells:
    """Sidecar paper/tables/<name>.cells.json: ID -> {file, row, column, value, printed}."""

    def __init__(self, name: str):
        self.name, self.d = name, {}

    def add(self, cid: str, file: str, row: str, column: str, value: float, printed: str) -> str:
        self.d[cid] = {"file": file, "row": row, "column": column, "value": float(value), "printed": plain(printed)}
        return f"cell:{self.name}:{cid}"

    def write(self) -> None:
        (TAB / f"{self.name}.cells.json").write_text(json.dumps(self.d, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# definitional numbers (paper/defs.yaml)
# ---------------------------------------------------------------------------------------------
def defs() -> dict:
    from shieldmap.analysis3 import density_model as dm
    from shieldmap.analysis4b import patches
    summ = json.loads((P4A / "S" / "summary.json").read_text(encoding="utf-8"))
    nat = pd.read_csv(P4A / "S" / "natrad_elements.csv")
    t = pd.read_csv(HEADLINE).set_index("id")
    n81 = int(re.search(r"\(([\d,]+) formulas", t.at["H81", "statement_v3"]).group(1).replace(",", ""))
    n87 = int(re.search(r"\(([\d,]+) structure-matched", t.at["H87", "statement_v3"]).group(1).replace(",", ""))
    d = {
        "T_target": (0.01, "transmission (air kerma)", "Table C headline target for Cs-137/Co-60 (variants 38/42) and single lines; analysis4.headline.HEAD_KEYS"),
        "stab_ehull": (0.05, "eV/atom", "STAB threshold; analysis4.refsets._b4_parts (constraint_mask('STAB', 0.05))"),
        "gap_threshold": (3.1, "eV", "GAP threshold; analysis4.refsets.practical_sets (constraint_mask('GAP', 3.1))"),
        "abund_threshold": (1, "ppm", "ABUND threshold of B6; analysis4.refsets.practical_sets (abund=(1.0,))"),
        "abund_tier_lo": (0.5, "ppm", "B6@ABUND0.5 tier run; analysis3.density_model.TIER_SETS"),
        "abund_tier_hi": (2, "ppm", "B6@ABUND2 tier run; analysis3.density_model.TIER_SETS"),
        "abund_curve": ([0.1, 10], "ppm", "ABUND(a) curve range; headline rows H95-H98"),
        "diag_anode_angle": (12, "degree", "diagnostic headline variant 13 (RQR6, 12°, 0.35 mm Pb); analysis4b.patches.HEADLINE_DIAG_VARIANT"),
        "diag_pb_target": (0.35, "mm Pb", "diagnostic headline variant 13 (RQR6, 12°, 0.35 mm Pb); analysis4b.patches.HEADLINE_DIAG_VARIANT"),
        "pb_density": (11.34, "g/cm3", "Pb reference density, Phase 1 D-011 (pymatgen density_of_solid, P1)"),
        "artefact_ehull_theoretical": (0.1, "eV/atom", "artefact flag condition 1 (Phase 3 P3-D-005; Phase 4a step 2)"),
        "artefact_rho_factor": (1.05, "factor", "artefact flag condition 2 (Phase 4a step 2): rho_MP > 1.05 x rho_max"),
        "rho_max": (22.65, "g/cm3", "densest Phase 1 experimental element density (Ir); analysis3.density_model.RowModel.rho_max"),
        "artefact_ehull_icsd": (0.3, "eV/atom", "artefact flag condition 3 (Phase 4a step 2)"),
        "baseline_min_density": (1.0, "g/cm3", "baseline eligibility, handover_phase2 §2.1"),
        "rad_z_min": (84, "Z", "no-stable-isotope rule (Tc, Pm, Z >= 84), handover_phase1 Table A is_radioactive"),
        "excl_topn": (10, "rank", "element-exclusion set: every element in a B4 top 10 at any column; analysis4.refsets.EXCL_TOP"),
        "noh_from": (patches.noh_from_kev() / 1000, "MeV", "D3 switch point (P4B-D-015): Co-60 areal prices and areal curves at and above it use the H-excluded variant; analysis4b.patches.noh_from_kev()"),
        "cs137_line": (661.657, "keV", "ENSDF Cs-137 (Ba-137m) gamma line; data/phase4a/mc/inputs/cases.json"),
        "co60_lines": ([1173.228, 1332.492], "keV", "ENSDF Co-60 gamma lines; data/phase4a/mc/inputs/cases.json"),
        "pb_k_edge": (88.0, "keV", "Pb K edge (XCOM edges)"),
        "xcom_version": (3.1, "version", "NIST XCOM 3.1 standalone package (Phase 1 step 0)"),
        "spekpy_version": (2.5, "version (2.5.4)", "SpekPy 2.5.4 (Phase 1 data-principle audit)"),
        "mendeleev_version": (1.3, "version (1.3.0)", "mendeleev 1.3.0 (crustal abundance, isotopic abundances)"),
        "openmc_version": (0.16, "version (0.16.0)", "OpenMC 0.16.0 (Phase 4a step 5)"),
        "mp_release": (2026.04, "release (2026.04.13)", "Materials Project database version 2026.04.13 (Phase 1 step 0)"),
        "cod_revision": (297631, "revision", "COD bulk release rev. 297631 of 2025-02-07 (Phase 3 P3-D-004)"),
        "crc_edition": (95, "edition", "CRC Handbook of Chemistry and Physics, 95th ed. (crustal abundance, P3)"),
        "mc_photon_cutoff": (1, "keV", "OpenMC photon cutoff (Phase 4a mc_spot_check §1)"),
        "mc_line_threshold": (3.2029, "keV", "fl estimator line threshold (Phase 4a mc_spot_check §1)"),
        "mc_lateral_half_width": (5, "cm", "reflective planes at x, y = ±5 cm (Phase 4a mc_spot_check §1)"),
        "density_draws": (int(dm.N_DRAWS), "draws", "Phase 3 density-model draws per tier run; analysis3.density_model.N_DRAWS"),
        "excl_elements": (len(summ["excl_elements"]), "elements", "data/phase4a/S/summary.json (excl_elements)"),
        "natrad_elements": (int(len(nat)), "elements", "data/phase4a/S/natrad_elements.csv (rows)"),
        "polymorph_formulas": (n81, "formulas", "statement of H81 (Phase 3 R4.2 polymorph spread)"),
        "cod_matched_compounds": (n87, "compounds", "statement of H87 (Phase 3 COD density bias)"),
        "spekpy_cell_kev": (_spekpy_cells()[0], "keV", "SpekPy cell width used by the quadrature; physics.spectra (D-013)"),
        "spekpy_cell0_kev": (_spekpy_cells()[1], "keV", "lower edge of the first SpekPy cell, fluence below it dropped; physics.spectra.CELL0_KEV"),
        "frontier_median_n": (50, "members", "median50_clean frontier kind: median of the 50 best non-artefact members; analysis4.refsets"),
        "band_corners": (18, "combinations", "band = min-max envelope over 3 polymorph rules x 3 frontier kinds x coherent on/off; analysis4.refsets.bands"),
    }
    assert patches.HEADLINE_DIAG_VARIANT == 13   # NOH_FROM_KEV became patches.noh_from_kev() in (fix)
    return {k: {"value": v[0], "unit": v[1], "source": v[2]} for k, v in d.items()}


def _spekpy_cells() -> tuple[float, float]:
    from shieldmap.physics import spectra
    dk = getattr(spectra, "CELL_KEV", None) or getattr(spectra, "DK_KEV", None) or 0.25
    return float(dk), float(spectra.CELL0_KEV)


def write_defs(mine: dict) -> None:
    cur = yaml.safe_load(DEFS_FILE.read_text(encoding="utf-8")) if DEFS_FILE.exists() else {}
    cur = cur or {}
    cur.update(mine)
    head = ("# Definitional numbers and counts for [def:NAME] / [count:NAME] tags (paper/CONVENTIONS.md).\n"
            "# Flat mapping NAME -> {value, unit, source}; value may be a list. Entries written by\n"
            "# shieldmap.analysis4b.tables are refreshed on every run; other entries are kept.\n")
    DEFS_FILE.write_text(head + yaml.safe_dump(cur, sort_keys=True, allow_unicode=True, width=200), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------------------------
REQS = (("PB", "Lead-free", "lead-free (PB) given the rest of B4", ()),              # printed labels in plain words
        ("STAB", "Thermodynamically stable", "stability STAB(0.05) given the rest of B4", ()),
        ("EXP", "Experimentally observed", "ICSD match (EXP) given the rest of B4", ()),
        ("GAP", "Transparent (band gap ≥ 3.1 eV)", "band gap ≥ 3.1 eV (GAP) given B4", ("def:gap_threshold",)),
        ("ABUND", "Scarcity proxy (≥ 1 ppm)", "scarcity proxy (crustal abundance) ≥ 1 ppm given B4", ("def:abund_threshold",)))
T1_IDS = {
    "CS137": (r"Cs-137 ($T = 0.01$)", "Cs-137 (T = 0.01)", ("H03", "H05", "H07", "H09", "H11"), ("H04", "H06", "H08", "H10", "H12")),
    "CO60": (r"Co-60 ($T = 0.01$)", "Co-60 (T = 0.01)", ("H15", "H17", "H19", "H21", "H23"), ("H133", "H134", "H135", "H136", "H137")),
    "DIAG": (r"Reference diagnostic beam: RQR6, 12$^{\circ}$ anode angle, 0.35~mm Pb target (air kerma)", "DIAG RQR6, 0.35 mm Pb target",
             ("H27", "H29", "H31", "H33", "H35"), ("H28", "H30", "H32", "H34", "H36")),
}
GROUP_TAGS = {"CS137": ("def:T_target",), "CO60": ("def:T_target",), "DIAG": ("def:diag_anode_angle", "def:diag_pb_target")}


def _check_t1(t: pd.DataFrame) -> list[dict]:
    """Verify every Table 1 row against its statement, basis, reference and Δ unit; return the checks."""
    out = []
    for sc, (_, where, th, ar) in T1_IDS.items():
        for (code, _, stmt, _), a, b in zip(REQS, th, ar):
            for rid, basis, unit in ((a, "thickness", "cm"), (b, "areal", "g/cm2")):
                r = t.loc[rid]
                s = r.statement_v3
                ok = {"statement": stmt in s and where in s and f"{basis} basis" in s, "basis": r.basis == basis,
                      "delta_unit": r.delta_unit == unit, "unit": r.unit == "%",
                      "reference": str(r.reference_set).startswith("B4-internal"),
                      "kind": r.frontier_kind == "best_clean",
                      "density": (r.density_source == "measured-density point") if basis == "thickness" else True,
                      "noH": ("H-containing compounds excluded" in s) == (sc == "CO60" and basis == "areal")}
                bad = [k for k, v in ok.items() if not v]
                if bad:
                    raise RuntimeError(f"Table 1 row {rid} fails checks {bad}: {s}")
                out.append({"scenario": sc, "requirement": code, "basis": basis, "id": rid, "delta_unit": r.delta_unit,
                            "reference_compounds": r.reference_compounds})
    return out


def table1(t: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Layout: value and absolute increment per basis, bands moved to the dataset, one-line
    title carrying the scope sentence, three one-sentence notes."""
    checks = _check_t1(t)
    L = [r"% Table 1 -- generated by shieldmap.analysis4b.tables; do not edit by hand.",
         *FONT_PREAMBLE,
         r"\begin{table*}[!htbp]",
         r"\centering",
         r"\shieldtablefont",
         r"\begin{threeparttable}",
         r"\caption{Narrow-beam price of each requirement inside the practical set, for ideal single-phase catalogued crystals.}",
         r"\label{tab:t1}",
         r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}lrrrr@{}}",
         r"\toprule",
         r" & \multicolumn{2}{c}{Thickness\tnote{c}} & \multicolumn{2}{c}{Mass thickness} \\",
         r"\cmidrule{2-3} \cmidrule{4-5}",
         r"Requirement\tnote{a} & Price (\%)\tnote{b} & $\Delta$ (mm) & Price (\%)\tnote{b} & $\Delta$ (" + G_CM2 + r") \\",
         r"\midrule"]
    recs = []
    titles = {"CS137": r"Cs-137 ($T = 0.01$)", "CO60": r"Co-60 ($T = 0.01$)\tnote{c}",
              "DIAG": r"Reference beam (RQR6, 12$^{\circ}$ anode angle, 0.35~mm Pb)"}
    for k, (sc, (_, _, th, ar)) in enumerate(T1_IDS.items()):
        if k:
            L.append(r"\addlinespace")
        L.append(text_line(r"\multicolumn{5}{@{}l}{\textit{" + titles[sc] + r"}} \\", GROUP_TAGS[sc]))
        for (code, label, _, ltags), a, b in zip(REQS, th, ar):
            ra, rb = t.loc[a], t.loc[b]
            dmm = 10 * float(ra.delta_abs)
            cells = [tex(label), fmt(ra.value_pct), fmt(dmm), fmt(rb.value_pct), fmt(rb.delta_abs)]
            tags = list(ltags) + [a, f"{a}:delta_mm", b, f"{b}:delta"]
            L.append(line(cells, tags))
            recs.append({"scenario": sc, "requirement": code, "thickness_id": a, "areal_id": b,
                         "thickness_price_pct": ra.value_pct, "thickness_band_lo_pct": ra.band_lo_pct,
                         "thickness_band_hi_pct": ra.band_hi_pct, "thickness_delta_mm": dmm,
                         "areal_price_pct": rb.value_pct, "areal_band_lo_pct": rb.band_lo_pct,
                         "areal_band_hi_pct": rb.band_hi_pct, "areal_delta_g_cm2": rb.delta_abs,
                         "thickness_reference": ra.reference_compounds, "areal_reference": rb.reference_compounds,
                         "printed": " | ".join(plain(c) for c in cells[1:5])})
    L += [r"\bottomrule",
          r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]",
          r"\shieldtablefont",
          text_line(r"\item[a] Stable: energy above the hull at most 0.05~eV/atom on the Materials Project mixed GGA/GGA+U/r$^2$SCAN hull; observed: at least one ICSD entry; neither means stability in air.", ("def:stab_ehull",)),
          text_line(r"\item[b] Each 0 is a reference value: the same crystal sets the frontier with and without the requirement, namely Ir (thickness) and Bi (mass thickness) at Cs-137 and Co-60, and the lanthanide intermetallics SmPt$_3$ (thickness) and Ce$_5$AgSn$_3$ (mass thickness) at the reference beam; none is a liquid.", ("def:zero",)),
          r"\item[c] Thickness uses measured densities where they exist, Co-60 mass thickness excludes hydrogen-containing compounds (the liquid NH$_3\cdot$H$_2$O would otherwise set the mass-thickness frontier), and the bands over polymorph rules, frontier definitions and coherent scattering are in the dataset.",
          r"\end{tablenotes}",
          r"\end{threeparttable}",
          r"\end{table*}"]
    df = pd.DataFrame(recs)
    df.attrs["checks"] = checks
    return "\n".join(L) + "\n", df


# ---------------------------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------------------------
T2_ROWS = (  # set, source, P(best) row, Pb-relative row, density row
    ("B4", "Cs-137", "H89", "H66", "H158"), ("B4", "Co-60", "H92", "H154", "H158"),
    ("B5", "Cs-137", "H90", "H70", "H160"), ("B5", "Co-60", "H93", "H157", "H161"),
    ("B6", "Cs-137", "H91", "H67", "H159"), ("B6", "Co-60", "H94", "H155", "H159"),
)
T2_PBREL_FORMULA = {"H66": "Ir", "H154": "Ir", "H70": "BiOF", "H157": "HfO2", "H67": "W", "H155": "W"}


def table2(t: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Layout: leaders only (source, set, leader, P(best), Pb-relative thickness), three notes."""
    L = [r"% Table 2 -- generated by shieldmap.analysis4b.tables; do not edit by hand.",
         *FONT_PREAMBLE,
         r"\begin{table}[!htbp]",
         r"\centering",
         r"\shieldtablefont",
         r"\begin{threeparttable}",
         r"\caption{Thickness frontier crystals of the practical, transparent and abundant sets under density uncertainty.}",
         r"\label{tab:t2}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lllrr@{}}",
         r"\toprule",
         r"Source & Set & \makecell[l]{Frontier\\crystal} & $P(\mathrm{best})$\tnote{b} & \makecell[r]{Pb-relative thickness\\(narrow-beam)\tnote{c}} \\",
         r"\midrule"]
    ranges = []
    recs = []
    order = sorted(range(len(T2_ROWS)), key=lambda i: (T2_ROWS[i][1] != "Cs-137", i))
    for n, k in enumerate(order):
        st, src, pid, xid, did = T2_ROWS[k]
        rp, rx, rd = t.loc[pid], t.loc[xid], t.loc[did]
        leader = str(rp.unit).replace("P(best) of ", "")
        if leader != T2_PBREL_FORMULA[xid] or f"Density of {leader} (" not in rd.statement_v3 or f" of {leader} at" not in rx.statement_v3:
            raise RuntimeError(f"Table 2 row {st}/{src}: leader {leader} does not match {xid}/{did}")
        cs = src == "Cs-137"
        if (cs and not (xid in ("H66", "H67", "H70", "H71") and "CS137" in rp.statement_v3)) or \
                (not cs and not ("Co-60 (T = 0.01)" in rx.statement_v3 and "CO60" in rp.statement_v3)):
            raise RuntimeError(f"Table 2 row {st}/{src}: rows {pid}/{xid} are not for {src}")
        if not ("Phase 1 experimental element density" in rd.statement_v3 or "COD" in rd.statement_v3):
            raise RuntimeError(f"{did}: density is not measured; the Table 2 note says measured densities")
        pb = fmt_dec(rp.value, 2)
        if fmt_dec(rp.band_lo_pct, 2) != fmt_dec(rp.band_hi_pct, 2):
            ranges.append((st, src, pid, fmt_dec(rp.band_lo_pct, 2), fmt_dec(rp.band_hi_pct, 2)))
        name = {"B4": "Practical", "B5": "Transparent", "B6": "Abundant"}[st]
        leader_tex = formula(leader) + (r"\tnote{a}" if st == "B5" else "")
        cells = [src if n % 3 == 0 else "", name, leader_tex, pb, fmt_dec(rx.value, 2)]
        L.append(line(cells, [pid, xid]))
        recs.append({"set": st, "source": src, "leader": leader, "density_g_cm3": rd.value,
                     "p_best_R2": rp.value, "p_best_R1_R3_lo": rp.band_lo_pct, "p_best_R1_R3_hi": rp.band_hi_pct,
                     "pb_relative_thickness": rx.value, "p_best_id": pid, "pb_relative_id": xid, "density_id": did,
                     "printed": " | ".join(plain(c) for c in cells[3:])})
        if n == 2:
            L.append(r"\addlinespace")
    if {st for st, *_ in ranges} != {"B5"}:
        raise RuntimeError(f"Table 2 note says P(best) varies over rules only for the transparent set; it varies for {ranges}")
    L += [r"\bottomrule",
          r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]",
          r"\shieldtablefont",
          text_line(r"\item[a] BiOF (tetragonal) and HfO$_2$ (monoclinic) are the frontier crystals of the transparent set; both are birefringent, transparent only as single crystals or fine-grained ceramics \citep{Shield4B18}, and neither is a demonstrated window material.", ("refnote:method",)),
          text_line(r"\item[b] Share of the " + r"5\,000" + r" density-error model draws in which the compound is the frontier crystal; under the lowest-energy polymorph rule HfO$_2$, in its high-pressure Pbca form, is the frontier crystal of the transparent set at Cs-137 (" + fmt_dec(t.at["H90", "band_lo_pct"], 2) + ").", ("def:density_draws", "H90:lo")),
          text_line(r"\item[c] Thickness for $T = 0.01$ divided by that of Pb at 11.34~" + G_CM3 + r", with measured densities; not the lead attenuation equivalent of IEC 61331-1.", ("def:T_target", "def:pb_density")),
          r"\end{tablenotes}",
          r"\end{threeparttable}",
          r"\end{table}"]
    return "\n".join(L) + "\n", pd.DataFrame(recs)


# ---------------------------------------------------------------------------------------------
# Table S1: constraint definitions and data sources
# ---------------------------------------------------------------------------------------------
def tableS1() -> tuple[str, pd.DataFrame]:
    """Table S1: data sources and definitions merged into one table."""
    summ = json.loads((P4A / "S" / "summary.json").read_text(encoding="utf-8"))
    from shieldmap.analysis4 import refsets
    from shieldmap.analysis4b import patches
    hazard = ", ".join(refsets.HAZARD_ELEMENTS)
    load_bearing = ", ".join(e for e in refsets.ALWAYS_EXCL if e not in refsets.HAZARD_ELEMENTS)
    nat = pd.read_csv(P4A / "S" / "natrad_elements.csv")
    moderate = ", ".join(e for e in nat[nat.moderate & ~nat.strict].element)
    named = ", ".join(e for e in nat[nat.named & ~nat.strict].element)
    rows = [  # (group, item, definition or source, class, tags)
        ("Data", "Photon cross sections, absorption edges", r"NIST XCOM 3.1 package, ported line by line", "P1", ("def:xcom_version",)),
        ("Data", "Air-kerma response", r"NIST SRD 126 ($\mu_\mathrm{en}/\rho$ of air)", "P1", ()),
        ("Data", "Radionuclide lines, half-lives", r"ENSDF", "P1", ()),
        ("Data", "Atomic masses, element densities", r"element table distributed with pymatgen (compiled reference values)", "P1", ()),
        ("Data", "Compounds: composition, density, energy above the hull, ICSD entries, band gap", r"Materials Project release 2026.04.13, one archived query", "P2", ("def:mp_release",)),
        ("Data", "Crustal abundance; isotopic abundances", r"CRC Handbook, 95th ed., and IUPAC, via mendeleev 1.3.0", "P3", ("def:crc_edition", "def:mendeleev_version")),
        ("Data", "Elements without a stable isotope; noble gases", r"definitional lists", "P3", ()),
        ("Data", "X-ray beams", r"SpekPy 2.5.4, matched to IEC 61267 RQR qualities", "model", ("def:spekpy_version",)),
        ("Data", "Measured compound densities", r"COD revision 297631; density-error model and, through the measured-density point, thickness prices; the map itself uses Materials Project densities", "comparison", ("def:cod_revision",)),
        ("Data", "Fluorescence line data; library sensitivity", r"xraylib 4.3.0", "model", ("def:xraylib_version",)),
        ("Data", "Monte Carlo spot check", r"OpenMC 0.16.0 with ENDF/B-VIII.0; never in the map", "check", ("def:openmc_version",)),
        ("Def", "Practical set (B4)", r"Non-radioactive (no element without a stable isotope: Tc, Pm, $Z \geq 84$), lead-free, stable (energy above the hull $\leq 0.05$~eV/atom) and observed (at least one ICSD entry).", "", ("def:rad_z_min", "def:stab_ehull")),
        ("Def", "Transparent (B5), abundant (B6)", r"B4 with a GGA(+U) band gap $\geq 3.1$~eV; B4 with every element at a crustal abundance $\geq 1$~ppm.", "", ("def:gap_threshold", "def:abund_threshold")),
        ("Def", "Polymorph rules", r"R2 (primary): per formula the lowest-energy ICSD-matched entry, else the lowest-energy entry (R1); R3: all polymorphs.", "", ()),
        ("Def", "Frontier definitions", r"best\_clean (headline): the best non-artefact member; best\_icsd: the best non-artefact member with an ICSD entry; median50\_clean: the median of the 50 best non-artefact members.", "", ("def:frontier_median_n",)),
        ("Def", "Band", r"Minimum--maximum envelope over the 18 combinations of three polymorph rules, three frontier definitions and coherent scattering on or off, at a fixed density source (thickness: measured-density point) and fixed artefact flag; it does not include the density-error model draws.", "", ("def:band_corners",)),
        ("Def", "Artefact flag", r"Theoretical with energy above the hull $> 0.1$~eV/atom, or $\rho_\mathrm{MP} > 1.05 \times 22.65$~" + G_CM3 + r" (the densest measured element), or with an ICSD entry and an energy above the hull $> 0.3$~eV/atom; removed from every frontier; B4 contains none.",
         "", ("def:artefact_ehull_theoretical", "def:artefact_rho_factor", "def:rho_max", "def:artefact_ehull_icsd")),
        ("Def", "Natural-radioactivity flag", r"Sensitivity only: moderate adds the elements whose natural specific activity exceeds that of natural K (" + moderate + r"); named adds " + named + ".", "", ()),
        ("Def", "Fully oxidised filter", r"AMB-ANION: only O and F as anions, with charge balance $\sum_i n_i\,\mathrm{ox}_i = 2n_\mathrm{O} + n_\mathrm{F}$, every other element in one of its positive common oxidation states (pymatgen), extended by its ICSD-reported states for d-block metals and lanthanides, mixed valence allowed (necessary, not sufficient, for stability in air); AMB-ICSD3: at least 3 ICSD entries.", "", ("def:amb_icsd_min",)),
        ("Def", "Hydrogen-containing compounds excluded", r"No H; used for Co-60 mass-thickness prices and mass-thickness curves from " + f"{patches.noh_from_kev() / 1000:g}" + r"~MeV, where the liquid NH$_3\cdot$H$_2$O would otherwise set the mass-thickness frontier of some of the sets compared (up to 2.9~MeV).", "", ("def:noh_from", "def:noh_compared_range_MeV")),
        ("Def", "Element-exclusion filter", r"No element el, inside B4, for " + str(len(summ["excl_elements"])) + r" elements: every element in a B4 top 10 at any energy, the hazard list (" + hazard + r") and the load-bearing list (" + load_bearing + r").", "", ("count:excl_elements", "def:excl_topn")),
        ("Def", "Price, headroom", r"Price: B4 without X against B4 (X defines B4), or B4 against B4 with X; headroom: against the unconstrained artefact-free frontier F0.", "", ()),
    ]
    L = [r"% Table S1 -- generated by shieldmap.analysis4b.tables; do not edit by hand.",
         *FONT_PREAMBLE,
         r"\begin{table}[!htbp]",
         r"\centering",
         r"\shieldtablefont",
         r"\caption{Data sources and definitions. Classes: P1, a physical table maintained by a standards body; P2, the database of record; P3, a definitional or geochemical constant.}",
         r"\label{tab:s1}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.09\textwidth}@{}}",
         r"\toprule",
         r"Item & Source or definition & Class \\",
         r"\midrule"]
    recs, grp = [], None
    for g, item, d, cls, tags in rows:
        if g != grp:
            if grp is not None:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{3}{@{}l}{\textit{" + {"Data": "Data sources", "Def": "Definitions"}[g] + r"}} \\")
            grp = g
        L.append(line([item, d, cls], tags))
        recs.append({"group": g, "item": item, "definition_or_source": d, "class": cls, "tags": " ".join(tags)})
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(L) + "\n", pd.DataFrame(recs)


# ---------------------------------------------------------------------------------------------
# Table S2: Monte Carlo cases (a) and the compounds behind the diagnostic frontier (b)
# ---------------------------------------------------------------------------------------------
S2_COMPOUNDS = (  # (compound, role, ICSD entries, tags): compounds named in the manuscript; counts in paper/defs.yaml
    (r"Ce$_5$AgSn$_3$", r"narrow-beam frontier crystal at the reference beam; case c7", "1", ("def:nicsd_single",)),
    (r"PrSbPd", r"narrow-beam frontier crystal at RQR5; frontier crystal under the fluorescence model at the reference beam; cases c3, c7", "1", ("def:nicsd_single",)),
    (r"LaBi", r"lanthanide compound simulated at RQR5; case c3", "4", ("def:nicsd_LaBi",)),
    (r"Pr$_3$OsI$_3$\tnote{h}", r"lanthanide compound simulated at RQR8; case c4", "1", ("def:nicsd_single",)),
    (r"Er$_6$IrI$_{10}$, Tb$_6$IrI$_{10}$\tnote{h}", r"narrow-beam frontier crystals of other X-ray variants", "1, 1", ("def:nicsd_single",)),
    (r"CeSnRh, Nd$_5$Sn$_4$, Nd$_{11}$Sn$_{10}$", r"narrow-beam frontier crystals of other X-ray variants", "3, 3, 2", ("def:nicsd_CeSnRh", "def:nicsd_Nd5Sn4", "def:nicsd_Nd11Sn10")),
    (r"Ba$_2$SmSbO$_6$", r"fully oxidised lanthanide frontier crystal with at least 3 ICSD entries at the reference beam; case c7", "3", ("def:nicsd_Ba2SmSbO6", "def:amb_icsd_min")),
    (r"Ba$_4$Tb(RuO$_4$)$_3$, LaHoO$_3$", r"the same at RQR5 and RQR8 (not transported)", "3, 4", ("def:nicsd_Ba4TbRuO43", "def:nicsd_LaHoO3")),
    (r"BaNdInO$_4$", r"fully oxidised lanthanide frontier crystal without the ICSD condition at the reference beam (not transported)", "1", ("def:nicsd_BaNdInO4",)),
    (r"CsAgO, CsSmO$_2$", r"hygroscopic frontier crystals of the fully oxidised filter (reference beam and RQR5; RQR8)", "2, 1", ("def:nicsd_CsAgO", "def:nicsd_CsSmO2")),
    (r"Ir, W, Bi, BiOF, HfO$_2$", r"radionuclide frontier crystals and the textbook material Bi", "7, 14, 8, 2, 11", ("def:nicsd_Ir", "def:nicsd_W", "def:nicsd_Bi", "def:nicsd_BiOF", "def:nicsd_HfO2")),
)


def tableS2() -> tuple[str, pd.DataFrame, Cells]:
    mc = P4A / "mc"
    cases = json.loads((mc / "inputs" / "cases.json").read_text(encoding="utf-8"))
    dcases = json.loads((mc / "inputs" / "diag_cases.json").read_text(encoding="utf-8"))
    summ = json.loads((mc / "summary.json").read_text(encoding="utf-8"))
    dsum = json.loads((mc / "diag_summary.json").read_text(encoding="utf-8"))
    pts = pd.concat([pd.read_csv(mc / "points.csv").assign(_f="data/phase4a/mc/points.csv"),
                     pd.read_csv(mc / "diag_points.csv").assign(_f="data/phase4a/mc/diag_points.csv")])
    # revision case c7 at the reference beam: data/phase4b/mc_rev
    rv = P4B / "mc_rev"
    rev = json.loads((rv / "inputs" / "cases.json").read_text(encoding="utf-8"))
    rsum = json.loads((rv / "summary.json").read_text(encoding="utf-8"))
    pts = pd.concat([pts, pd.read_csv(rv / "points.csv").assign(_f="data/phase4b/mc_rev/points.csv")])
    frev = "data/phase4b/mc_rev/inputs/cases.json"
    # Stage 4d case c8 (Cs-137, W vs Ir; analysis4d.mc_c8): data/phase4d/mc_c8. Only the W points are new; the Ir
    # and Pb points are those of stage 4a cases c1 and c6, reused (not re-simulated), so only W enters the row.
    c8d = P4D / "mc_c8"
    c8 = json.loads((c8d / "inputs" / "cases.json").read_text(encoding="utf-8"))
    c8sum = json.loads((c8d / "summary.json").read_text(encoding="utf-8"))
    c8all = pd.read_csv(c8d / "points.csv")
    f8, f8p = "data/phase4d/mc_c8/inputs/cases.json", "data/phase4d/mc_c8/points.csv"
    c8ids = {x["id"] for x in c8["cases"]}
    c8new_mats = {r["material"] for x in c8["cases"] for r in x["runs"]}
    c8reused = {r["material"]: r["reused_from"] for x in c8["cases"] for r in x["reused_runs"]}
    c8new = c8all[c8all.material.isin(c8new_mats)]
    if (c8new.origin.str.contains("reused").any() or not c8all[~c8all.material.isin(c8new_mats)].origin.str.contains("reused").all()
            or set(c8all.material) != c8new_mats | set(c8reused)):
        raise RuntimeError("Table S2: data/phase4d/mc_c8/points.csv origins do not match inputs/cases.json (runs / reused_runs)")
    # the reused points must be the stage 4a points of the cases they come from (same factor, sub-runs, histories)
    for m, src in c8reused.items():
        a = c8all[c8all.material == m][["factor", "n_subruns", "n_particles"]].sort_values("factor").reset_index(drop=True)
        b = pts[(pts.case_id == src.split("/")[1]) & (pts.material == m)][["factor", "n_subruns", "n_particles"]] \
            .sort_values("factor").reset_index(drop=True)
        if len(a) != len(b) or not np.allclose(a.values.astype(float), b.values.astype(float), rtol=1e-12, atol=1e-9):
            raise RuntimeError(f"Table S2: c8 reused {m} points differ from {src} in data/phase4a/mc/points.csv")
    pts = pd.concat([pts, c8new.assign(_f=f8p)])
    cells = Cells("tableS2")
    fc, fd = "data/phase4a/mc/inputs/cases.json", "data/phase4a/mc/inputs/diag_cases.json"
    L = [r"% Table S2 -- generated by shieldmap.analysis4b.tables from data/phase4a/mc/, data/phase4b/mc_rev/, data/phase4d/mc_c8/ and paper/defs.yaml; do not edit by hand.",
         *FONT_PREAMBLE,
         r"\begin{table*}[!htbp]",
         r"\centering",
         r"\shieldtablefont",
         r"\begin{threeparttable}",
         r"\caption{Monte Carlo cases (a) and the compounds named in the text, with their ICSD entries (b).\tnote{a}}",
         r"\label{tab:s2}",
         r"\par\smallskip{\raggedright\textit{(a) Monte Carlo cases}\par}\smallskip",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.08\textwidth}rrr@{}}",
         r"\toprule",
         r"Case & Source & Basis, target & Materials ($\rho$, " + G_CM3 + r")\tnote{b} & Factor\tnote{c} & Points & \makecell[r]{Sub-\\runs} & \makecell[r]{Histories\\(millions)} \\",
         r"\midrule"]
    recs = []
    allc = sorted(cases["cases"], key=lambda c: c["case"]) + sorted(rev["cases"], key=lambda c: c["case"]) + \
        sorted(c8["cases"], key=lambda c: c["case"]) + sorted(dcases["cases"], key=lambda c: c["case"])
    for c in allc:
        diag = c["case"] >= 100
        isrev = c["id"] in {x["id"] for x in rev["cases"]}
        is8 = c["id"] in c8ids
        f = frev if isrev else f8 if is8 else fd if diag else fc
        mats = (rev if isrev else c8 if is8 else dcases if diag else cases)["materials"]
        cid = c["id"]
        key = f"cases[id={cid}]"
        tags = []
        s = c["source"]
        if s["kind"] == "line":
            es = [fmt_dec(e, 3) for e in s["energies_keV"]]
            for j, (e, p) in enumerate(zip(s["energies_keV"], es)):
                tags.append(cells.add(f"{cid}_E{j}", f, f"{key}.source.energies_keV[{j}]", "energy_keV", e, p))
            src = f"{s['name']} line{'s' if len(es) > 1 else ''} ({', '.join(es)}~keV)"
        elif s["kind"] == "mono":
            e = s["energies_keV"][0]
            p = fmt_dec(e, 1)
            tags.append(cells.add(f"{cid}_E0", f, f"{key}.source.energies_keV[0]", "energy_keV", e, p))
            src = f"{p}~keV, single line"
        else:
            q, th = re.match(r"(RQR\d+)_th(\d+)", s["name"]).groups()
            tags.append(cells.add(f"{cid}_angle", f, f"{key}.source.name", "anode angle (deg) from the name", float(th), th))
            src = f"{q} spectrum, {th}" + r"$^{\circ}$ anode angle"
        T = c["T_target"]
        Ts = fmt_trim(T, 2)
        tags.append(cells.add(f"{cid}_T", f, f"{key}.T_target", "T_target", T, Ts))
        tgt = f"{c['basis'].replace('areal', 'mass thickness')}, $T = {Ts}$"
        if c.get("pb_mm"):
            pm = fmt_dec(c["pb_mm"], 2)
            tags.append(cells.add(f"{cid}_pb", f, f"{key}.pb_mm", "pb_mm", c["pb_mm"], pm))
            tgt += f" (narrow-beam T of {pm}~mm Pb)"
        ms = []
        for m in c["materials"]:
            if is8 and m not in c8new_mats:
                continue
            md = mats[m]
            rs = fmt_dec(md["rho_used"], 2)
            tags.append(cells.add(f"{cid}_{m}_rho", f, f"materials.{m}.rho_used", "rho_used", md["rho_used"], rs))
            nm = "Sn/Ba/Gd blend" if m == "SnBaGd_blend" else formula(m)
            ms.append(f"{nm} ({rs}" + (r"\tnote{d}" if md["rho_source"] == "conditional" else
                                       r"\tnote{g}" if m == "SnBaGd_blend" else "") + ")")
        if is8:   # reused materials: "with Ir and Pb from c1 and c6" (case labels from reused_from)
            ru = [m for m in c["materials"] if m not in c8new_mats]
            ms[-1] += (r", with " + " and ".join(formula(m) for m in ru) + " from " +
                       " and ".join(c8reused[m].split("/")[1].split("_")[0] for m in ru))
        p = pts[pts.case_id == cid]
        pf = p._f.iloc[0]
        fmin, fmax = fmt_trim(p.factor.min(), 3), fmt_trim(p.factor.max(), 3)
        npts, nsub, hist = len(p), int(p.n_subruns.sum()), float(p.n_particles.sum()) / 1e6
        hs = fmt_dec(hist, 0) if hist >= 100 else fmt(hist, 3)
        q = f"case_id == '{cid}'" + (f" and material in {sorted(c8new_mats)}" if is8 else "")
        tags += [cells.add(f"{cid}_fmin", pf, q, "factor (min)", p.factor.min(), fmin),
                 cells.add(f"{cid}_fmax", pf, q, "factor (max)", p.factor.max(), fmax),
                 cells.add(f"{cid}_points", pf, q, "rows (count)", npts, str(npts)),
                 cells.add(f"{cid}_subruns", pf, q, "n_subruns (sum)", nsub, str(nsub)),
                 cells.add(f"{cid}_hist", pf, q, "n_particles (sum) / 1e6", hist, hs)]
        name = tex(cid).replace(r"\_", r"\_\allowbreak{}") + (r"\tnote{e}" if diag else r"\tnote{f}" if isrev else
                                                              r"\tnote{i}" if is8 else "")
        L.append(line([name, src, tgt, ", ".join(ms), f"{fmin}--{fmax}", str(npts), str(nsub), hs], tags))
        recs.append({"case": c["case"], "case_id": cid, "title": c["title"], "source": plain(src), "target": plain(tgt),
                     "materials": "; ".join(f"{m} {mats[m]['rho_used']:.4g} ({mats[m]['rho_source']}"
                                            + (f"; reused from {c8reused[m]}" if is8 and m in c8reused else "") + ")"
                                            for m in c["materials"]),
                     "factor_min": p.factor.min(), "factor_max": p.factor.max(), "points": npts, "subruns": nsub,
                     "histories": float(p.n_particles.sum())})
    tot = summ["total_particles"] / 1e6
    ts = fmt_dec(tot, 0)
    tags = [cells.add("total_points", "data/phase4a/mc/summary.json", "n_points", "n_points", summ["n_points"], str(summ["n_points"])),
            cells.add("total_subruns", "data/phase4a/mc/summary.json", "n_subruns", "n_subruns", summ["n_subruns"], str(summ["n_subruns"])),
            cells.add("total_hist", "data/phase4a/mc/summary.json", "total_particles", "total_particles / 1e6", tot, ts)]
    hc = [r for r in recs if r["case"] < 100 and r["case_id"] not in {x["id"] for x in rev["cases"]} | c8ids]
    rc = [r for r in recs if r["case_id"] in {x["id"] for x in rev["cases"]}]
    if sum(r["points"] for r in rc) != rsum["n_points"] or sum(r["subruns"] for r in rc) != rsum["n_subruns"] or \
            abs(sum(r["histories"] for r in rc) - rsum["total_particles"]) > 0.5:
        raise RuntimeError("Table S2: the revision case does not reproduce data/phase4b/mc_rev/summary.json")
    # c8: the row (new W points only) must reproduce the new-simulation totals of data/phase4d/mc_c8/summary.json,
    # and the new plus reused points its n_points and n_subruns_reused
    c8r = [r for r in recs if r["case_id"] in c8ids]
    c8np = sum(r["points"] for r in c8r)
    c8old = c8all[~c8all.material.isin(c8new_mats)]
    if c8np != len(c8all) - len(c8old) or len(c8all) != c8sum["n_points"] or \
            sum(r["subruns"] for r in c8r) != c8sum["n_subruns_new"] or int(c8old.n_subruns.sum()) != c8sum["n_subruns_reused"] or \
            abs(sum(r["histories"] for r in c8r) - c8sum["particles_new"]) > 0.5:
        raise RuntimeError("Table S2: case c8 does not reproduce data/phase4d/mc_c8/summary.json")
    ap = summ["n_points"] + rsum["n_points"] + c8np
    asub = summ["n_subruns"] + rsum["n_subruns"] + c8sum["n_subruns_new"]
    at = (summ["total_particles"] + rsum["total_particles"] + c8sum["particles_new"]) / 1e6
    ats = fmt_dec(at, 0)
    fa = "data/phase4a/mc/summary.json + data/phase4b/mc_rev/summary.json + data/phase4d/mc_c8/summary.json"
    atags = [cells.add("all_points", fa, "n_points (sum; c8: new W points of points.csv)", "n_points", ap, str(ap)),
             cells.add("all_subruns", fa, "n_subruns (sum; c8: n_subruns_new)", "n_subruns", asub, str(asub)),
             cells.add("all_hist", fa, "total_particles (sum; c8: particles_new)", "total_particles / 1e6", at, ats)]
    if sum(r["points"] for r in hc) != summ["n_points"] or sum(r["subruns"] for r in hc) != summ["n_subruns"] or \
            abs(sum(r["histories"] for r in hc) - summ["total_particles"]) > 0.5:
        raise RuntimeError("Table S2: per-case sums do not reproduce data/phase4a/mc/summary.json")
    if sum(r["points"] for r in recs if r["case"] >= 100) != dsum["n_points"]:
        raise RuntimeError("Table S2: diagnostic points do not reproduce diag_summary.json")
    minsub = int(pts.n_subruns.min())
    L += [r"\midrule",
          line([r"\multicolumn{5}{@{}l}{Cases c1--c6, total}", str(summ["n_points"]), str(summ["n_subruns"]), ts], tags),
          line([r"\multicolumn{5}{@{}l}{Cases c1--c8, total}", str(ap), str(asub), ats], atags),
          r"\bottomrule",
          r"\end{tabularx}",
          r"\par\medskip{\raggedright\textit{(b) Compounds behind the X-ray-beam frontier}\par}\smallskip",
          r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.27\textwidth}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{0.12\textwidth}@{}}",
          r"\toprule",
          r"Compound & Role & ICSD entries \\",
          r"\midrule"]
    for comp, role, n, ntags in S2_COMPOUNDS:
        L.append(line([comp, role, n], ntags))
        recs.append({"case": None, "case_id": "", "title": f"compound {plain(comp)}", "source": "", "target": plain(role),
                     "materials": "", "factor_min": np.nan, "factor_max": np.nan, "points": np.nan, "subruns": np.nan,
                     "histories": np.nan})
    L += [r"\bottomrule",
          r"\end{tabularx}",
          r"\begin{tablenotes}[flushleft]",
          r"\shieldtablefont",
          text_line(r"\item[a] OpenMC 0.16.0 with ENDF/B-VIII.0; laterally infinite slab, free-air kerma at the back face, narrow-beam, fluorescence-inclusive and broad-beam estimators; each point has at least " + str(minsub) + r" sub-runs with distinct seeds.",
                    ("def:openmc_version", cells.add("min_subruns", "data/phase4a/mc/points.csv; diag_points.csv; data/phase4b/mc_rev/points.csv; data/phase4d/mc_c8/points.csv (new points)", "all", "n_subruns (min)", minsub, str(minsub)))),
          r"\item[b] Densities at the measured-density point; mass-thickness results do not depend on them.",
          r"\item[c] Slab thickness as a multiple of the map's required thickness or mass thickness at the target (minimum--maximum).",
          r"\item[d] No measured density: density-error model cell median.",
          r"\item[e] Check of the first-order fluorescence model.",
          r"\item[f] Simulated at the reference beam (dataset folder data/phase4b/mc\_rev); same model, physics, estimators and statistics.",
          r"\item[g] Homogeneous mixture; density from the inverse-density mixture of the element densities.",
          text_line(r"\item[h] The R$_3$I$_3$Z chain and R$_6$I$_{10}$Z cluster iodides are made in sealed Ta containers and handled under inert gas \citep{Shield4B08,Shield4B09,Shield4B10}, and the lanthanide-rich intermetallics are moisture-sensitive; none of these compounds is proposed as a shielding filler.", ("refnote:data",)),
          r"\item[i] Dataset folder data/phase4d/mc\_c8; the Ir and Pb points of cases c1 and c6 are reused, same model, physics, estimators and statistics.",
          r"\end{tablenotes}",
          r"\end{threeparttable}",
          r"\end{table*}"]
    return "\n".join(L) + "\n", pd.DataFrame(recs), cells


# ---------------------------------------------------------------------------------------------
# self-check with paper/check_numbers.py
# ---------------------------------------------------------------------------------------------
def selfcheck(files) -> dict:
    p = PAPER / "check_numbers.py"
    if not p.exists():
        return {"checked": 0, "failures": [], "note": "paper/check_numbers.py absent"}
    spec = importlib.util.spec_from_file_location("check_numbers_selfcheck", p)
    cn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cn)
    t, dfs = cn.load_targets()
    cells = cn._cells()
    fails, checked, banned = [], 0, []
    for f in files:
        for i, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code, com = cn.split_comment(raw)
            for b in cn.BANNED:
                if re.search(b, code, flags=re.I):
                    # "first-order" (the fluorescence model) is a technical term reproduced from statement_v3, not a
                    # priority claim; it is reported separately
                    term = b == r"\bfirst\b" and not re.search(r"\bfirst\b(?![- ]order)", code, flags=re.I)
                    banned.append({"file": f.name, "line": i, "word": b, "first_order_term": term})
            nums = cn.numbers_in(code)
            if not nums:
                continue
            tags = cn.TAG.findall(com)
            cand = []
            for tg in tags:
                v, e = cn.resolve(tg, t, dfs, cells)
                cand += v
                if e:
                    fails.append({"file": f.name, "line": i, "number": "", "problem": e})
            for n in nums:
                checked += 1
                if not tags or not any(cn.matches(n, x) for x in cand):
                    fails.append({"file": f.name, "line": i, "number": n, "problem": "untagged" if not tags else "no tag matches",
                                  "text": code.strip()[:120]})
    return {"checked": checked, "failures": fails, "banned": banned}


# ---------------------------------------------------------------------------------------------
def run() -> dict:
    TAB.mkdir(parents=True, exist_ok=True)
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(HEADLINE).set_index("id")
    missing = [i for i in ("H154", "H167") if i not in t.index]
    if missing:
        raise RuntimeError(f"headline table v3 lacks {missing}: run analysis4b.table_v3 first")
    write_defs(defs())
    out = {}
    s1, d1 = table1(t)
    s2, d2 = table2(t)
    sS1, dS1 = tableS1()
    sS2, dS2, cS2 = tableS2()
    cS2.write()
    for old in ("tableS4.tex", "tableS4.cells.json"):   # Stage 4b table, retired (tableS3 is the stage 4d anchor table, analysis4d.anchors)
        (TAB / old).unlink(missing_ok=True)
    files = []
    for name, s, d in (("table1", s1, d1), ("table2", s2, d2), ("tableS1", sS1, dS1), ("tableS2", sS2, dS2)):
        p = TAB / f"{name}.tex"
        p.write_text(s, encoding="utf-8")
        d.to_csv(CSV_OUT / f"{name}.csv", index=False)
        files.append(p)
        out[name] = len(d)
    (CSV_OUT / "table1_row_checks.json").write_text(json.dumps(d1.attrs["checks"], indent=1, ensure_ascii=False), encoding="utf-8")
    chk = selfcheck(files)
    (CSV_OUT / "selfcheck.json").write_text(json.dumps(chk, indent=1, ensure_ascii=False), encoding="utf-8")
    out["numbers_checked"] = chk["checked"]
    out["failures"] = len(chk["failures"])
    out["banned"] = sum(not b["first_order_term"] for b in chk.get("banned", []))
    out["banned_first_order_term"] = sum(b["first_order_term"] for b in chk.get("banned", []))
    return out


if __name__ == "__main__":
    print(run())
