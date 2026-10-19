"""Results dossier v2 (reports/phase4a/results_dossier_v2.md), generated.

The stage 3 dossier re-based on the stage 4a primaries. For each result the hand-written part is only
the statement, the status and what changed since stage 3; every number comes from the headline table
(data/phase4a/headline_table.csv) and is rendered from it, so the dossier cannot disagree with the table. Each
result also carries its reference-map classification (data/phase4a/reference_map.csv, the reference reconciliation) and its Monte Carlo
status (data/phase4a/mc/result_status.csv, the Monte Carlo spot check) when those files exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap.analysis4 import common

RESULTS = [
    # (id, title, statement, status, change since stage 3)
    ("R1.1", "Composition-to-density crossover",
     "Over the map, composition (log μ/ρ) and density (log ρ) have equal direct shares of the spread of log μ near "
     "235 keV; below it composition dominates, above it density. The energy depends on the sampled population.",
     "robust for the map as delivered; conditional as a general statement (population dependence)",
     "none (reference-independent)"),
    ("R1.2", "Physics of the crossover",
     "The composition spread is carried by the photoelectric term and decays with it; the crossing is where that "
     "decaying spread meets the population's fixed density spread. It is not where the photoelectric effect stops "
     "dominating.", "robust as a description of the population", "none"),
    ("R1.3", "Structure acts only through density; DFT polymorph spread vs measured",
     "μ/ρ is identical across polymorphs; for observed polymorphs the DFT density spread is close to the measured "
     "spread.", "robust for the μ/ρ identity; conditional for the spread comparison", "none"),
    ("R1.4", "Non-radioactive compounds exceeding the elemental envelope (47–67 keV)",
     "In the narrow beam some non-radioactive compounds attenuate more per unit length than every non-radioactive "
     "element; under the first-order fluorescence correction about a quarter survive, all lanthanide compounds with a "
     "dense 5d partner.", "conditional (escape kernel, density, omitted second-order terms)", "none"),
    ("R2.0", "Where each constraint is expensive, where it is free, and why",
     "Measured inside the practical bundle B4, the price of a constraint depends on energy and basis: transparency "
     "(GAP) is a large thickness price and a small mass price; the scarcity proxy is a staircase of single elements; "
     "lead-free, stability and ICSD match are free at the radionuclides; excluding radioactive elements forgoes a "
     "few to twenty per cent of thickness at the radionuclides, by comparison with α-Np metal.",
     "robust as a map of regimes", "re-based on B4-internal marginals (P4A-D-011)"),
    ("R2.1", "Lead-free (PB)",
     "Requiring lead-free adds nothing for every polyenergetic profile and for single lines on the thickness basis; "
     "on the mass basis, inside B4, it adds a few per cent only in a narrow window above the Pb K edge in the narrow "
     "beam, which the first-order fluorescence correction reduces. Pb needs 1–13 % more mass than the best practical "
     "lead-free compound at 0.1–10 MeV. The size of the narrow-beam window depends on the cross-section library "
     "(ENDF/B-VIII.0 gives a smaller value than XCOM, H121); the Monte Carlo confirms that Pb metal leaves the "
     "frontier once fluorescence is counted.",
     "robust for profiles and the thickness basis; the mass window is a narrow-beam, library-dependent statement",
     "PB priced against B4 without PB; corrected window recomputed for the B4-internal reference (analysis4.edge4)"),
    ("R2.2", "Transparency (GAP 3.1 eV): a thickness price, not a mass price",
     "Inside B4, requiring a band gap ≥ 3.1 eV more than doubles the thickness at the radionuclides but adds little "
     "mass, because wide-gap compounds are about half as dense as the practical frontier. The Monte Carlo reproduces "
     "the narrow-beam price and shows that build-up moves it by only a few points (H109, H117).",
     "robust as a structure; the PBE gap is a proxy for transparency", "B4-internal is now the primary reference"),
    ("R2.3", "Non-radioactive (RAD) given B4, and the scarcity proxy, at the radionuclides",
     "Excluding radioactive elements from B4 forgoes the thickness advantage of α-Np metal over Ir: about a fifth at "
     "Cs-137, a few per cent at Co-60, zero above 2.28 MeV. The reference α-Np is a real, measured metal but not a "
     "shielding material: the number states what the constraint forgoes, not an achievable alternative. The "
     "practical answer is inside B4: the scarcity proxy (crustal abundance) moves the frontier from Ir to W "
     "(about +20 to +23 % thickness), as a staircase over the threshold.",
     "conditional (single two-element comparison; density model; ceiling)",
     "RAD priced inside B4 (B4 vs B4 without RAD); Pu entries no longer set any band (Phase 4a artefact flag); "
     "abundance reported as a curve"),
    ("R2.4", "Price structure along energy",
     "B4-internal prices vary along energy with distinct regimes on each basis; the B4 mass frontier at 1–2.6 MeV is "
     "hydrogen-rich (NH₃·H₂O), and excluding H-containing compounds raises the mass price there by up to about 12 %.",
     "robust as a structure; single-line values near edges are narrow-beam",
     "B4-internal; H-exclusion sensitivity added (P4A-D-021)"),
    ("R2.5", "Diagnostic (tube) prices",
     "For diagnostic tube spectra the B4-internal prices on the mass basis are small: excluding radioactive elements "
     "forgoes about 2 % (median), the scarcity proxy about 1.6 %, and transparency about 9 %.",
     "robust for the mass basis; narrow beam (the fluorescence-corrected frontier is R3.1)",
     "re-based on B4-internal marginals; the Phase 3 F0 levels with radioactive near-artefact references are now "
     "headroom (appendix)"),
    ("R2.6", "Stability (STAB) and ICSD match (EXP)",
     "Inside B4, stability and ICSD match are free at the radionuclides and nearly free in the diagnostic spectra.",
     "robust under best_clean; conditional on the estimator for median-type kinds",
     "B4-internal (drop-one) definition replaces F0-based Shapley shares as the headline"),
    ("R2.7", "Load-bearing elements (element-exclusion filter)",
     "Within B4, excluding a single element adds less than a few per cent except just above a heavy element's K "
     "edge in the narrow beam; under the first-order fluorescence correction the W, Ta and Hf windows against B4 "
     "persist at roughly 6–18 % on the thickness basis.",
     "conditional (first-order model; three sampled energies per element)", "B4-internal values tabulated"),
    ("R2.8", "Artefact entries set most best-compound frontiers",
     "Without artefact control the unconstrained best-compound frontier is set by artefact entries in most cells; "
     "this is why the headline kind is best_clean and the reference set is B4-internal. Unphysical and metastable "
     "entries in DFT databases are a known problem; the map quantifies their effect on shielding frontiers.",
     "robust", "recomputed under the Phase 4a artefact flag; \"New in Phase 3\" withdrawn (reference map: quantifies known)"),
    ("R3.1", "DIAG lanthanide plateau under fluorescence escape",
     "In the narrow beam with a first-order fluorescence-escape correction, the mass-limited diagnostic frontier is "
     "a lanthanide plateau that beats every non-Pb textbook material; at RQR5 0.25 mm the ≥ 10 % count is a "
     "knife-edge. Lanthanide K-edge filling of lead's 40–88 keV window is known; broad-beam garment studies report "
     "that the mass saving of lead-free materials shrinks above about 100 kV, so the ≥ 10 % count above 100 kV is a "
     "narrow-beam, first-order statement. The Monte Carlo spot check confirms the direction at RQR5 0.25 mm and "
     "RQR8 0.35 mm under every estimator, but like-for-like with the first-order model the RQR5 0.25 mm advantage "
     "falls below 10 % (H112): the first-order model under-counts escaping fluorescence (Rayleigh-scattered lines, "
     "secondary fluorescence from Pr K lines in Sb/Pd partners). The \"36 of 36 variants ≥ 10 %\" count is therefore "
     "not robust and the median is biased high.",
     "weakened by Monte Carlo (direction holds; count and median do not)",
     "labelled narrow-beam, first-order (reference map: quantifies known); Monte Carlo cases 3 and 4"),
    ("R3.2", "Textbook materials vs the B4 frontier (mass basis)",
     "Textbook shielding materials need more areal mass than the B4 frontier; once fluorescence is counted Bi is the "
     "closest textbook material.", "robust within the first-order model", "none"),
    ("R3.3", "The plateau is a family of lanthanides, not of partners",
     "Which lanthanide leads is stable across tube settings and under the correction; which partner leads is not.",
     "robust for lanthanide-level families", "none"),
    ("R3.4", "Cs-137 / Co-60 tiers (thickness basis) and natural radioactivity",
     "The thinnest compound is identified for B4 (Ir) and B6 (W); Cs-137 B5 (BiOF) under R2/R3; Co-60 B5 stays "
     "open. Under the moderate natural-radioactivity flag (Rb, Sm, Lu, Re above natural K) the Lu oxides leave the "
     "Co-60 B5 top 10, which leaves HfO₂ and the Ta oxides.",
     "conditional (density model; rule for B5)", "natural-radioactivity flag derived from ENSDF + IUPAC (P4A-D-020)"),
    ("R3.5", "Dense wide-gap B5 candidates",
     "The transparent radionuclide candidates are dense wide-gap oxides and fluorides; they are ICSD-matched, on or "
     "near the hull, and not textbook shielding materials. Dense Lu/Hf oxides are a known class of transparent "
     "shields; the map ranks them under the density model.", "conditional (PBE gap as proxy)",
     "not a new candidate class (reference map); Lu-containing members are flagged under the moderate "
     "natural-radioactivity definition"),
    ("R4.1", "DFT density bias on compounds (COD)",
     "Materials Project densities are lower than measured densities by about 2.5 % on compounds, with clear class "
     "structure. The low bias of PBE densities is known; what is added is the COD-matched measurement over 21,489 "
     "compounds and its class and e_hull breakdown.", "robust",
     "\"New in Phase 3\" withdrawn (reference map: confirms known)"),
    ("R4.2", "Density error and the thickness prices",
     "Areal-mass results are density-independent. On the thickness basis the B4-internal prices move by several "
     "points between MP and measured densities; the density bands around the measured-density point are narrow.",
     "robust for areal invariance and the measured-density points; band widths conditional on the model",
     "density bands recomputed for the B4-internal references"),
    ("R4.3", "Ambient phase of textbook materials",
     "Materials Project's representative choice gives the wrong ambient polymorph for some textbook materials; this "
     "matters only on the thickness basis.", "robust", "none"),
]


def _rows_md(t: pd.DataFrame) -> list[str]:
    out = ["| id | statement | value | band | Δ | basis | reference | fluor. |", "|---|---|---|---|---|---|---|---|"]
    t = t.copy()
    for c in ("reference_compounds", "basis", "delta_unit", "density_band_pct", "delta_band"):
        t[c] = t[c].fillna("")
    for r in t.itertuples():
        if str(r.unit).startswith("%"):
            val = f"{r.value_pct:.2f} %{str(r.unit)[1:]}"
            band = f"{r.band_lo_pct:.2f}–{r.band_hi_pct:.2f} %" if np.isfinite(r.band_lo_pct) else "—"
            if isinstance(r.density_band_pct, str) and r.density_band_pct:
                band += f" (density {r.density_band_pct} %)"
        else:
            val = f"{r.value:.4g} {r.unit}"
            band = f"{r.band_lo_pct:.4g}–{r.band_hi_pct:.4g}" if np.isfinite(r.band_lo_pct) else "—"
        d = f"{r.delta_abs:.3g} {r.delta_unit}" if isinstance(r.delta_abs, float) and np.isfinite(r.delta_abs) else (r.delta_band if isinstance(r.delta_band, str) and r.delta_band else "—")
        out.append(f"| {r.id} | {r.statement} | **{val}** | {band} | {d} | {r.basis} | {r.reference_set}: {r.reference_compounds} | {r.fluorescence_corrected} |")
    return out


def _refmap() -> pd.DataFrame | None:
    f = common.P4 / "reference_map.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)


def _ref_lines(rm: pd.DataFrame | None, rid: str) -> list[str]:
    if rm is None:
        return ["- **Reference map:** not yet available (Step 6)."]
    idc = next((c for c in ("result_id", "dossier_result", "item_id", "id") if c in rm.columns), None)
    if idc is None:
        return ["- **Reference map:** file present but without a result-id column."]
    x = rm[rm[idc].astype(str) == rid]
    if x.empty:
        return [f"- **Reference map:** no row for {rid}."]
    out = []
    for r in x.itertuples():
        d = r._asdict()
        cls = d.get("classification", "")
        pos = d.get("positioning_sentence", d.get("positioning", ""))
        prior = d.get("closest_prior_statement", d.get("closest_prior", d.get("prior", "")))
        dois = d.get("prior_dois", "")
        if isinstance(dois, str) and dois:
            prior = f"{prior} [{dois}]"
        prior = "" if not isinstance(prior, str) else prior
        depth = d.get("reading_depth", "")
        out.append(f"- **Reference map:** **{cls}**" + (f" ({depth})" if depth else "") + (f". Closest prior: {prior}" if prior else "")
                   + (f". Positioning: {pos}" if pos else ""))
    return out


def _mc_line(rid: str) -> str:
    f = common.P4 / "mc" / "result_status.csv"
    if not f.exists():
        return "- **Monte Carlo:** not tested."
    m = pd.read_csv(f)
    x = m[m.dossier_result.astype(str) == rid]
    if x.empty:
        return "- **Monte Carlo:** not tested."
    return "- **Monte Carlo:** " + "; ".join(f"**{r.mc_status}**" + (f" ({r.note})" if isinstance(r.note, str) and r.note else "") for r in x.itertuples())


def build() -> str:
    t = pd.read_csv(common.P4 / "headline_table.csv")
    rm = _refmap()
    L = ["# Phase 4a results dossier v2", "",
         "Generated by `python -m shieldmap.analyze4a` (`analysis4/dossier4.py`). Every number below is rendered from "
         "`data/phase4a/headline_table.csv` and carries its row id; the full row (band type, sensitivity axes and "
         "values, source file and key) is in `headline_table.md`.", "",
         "**Primaries (handover §3, DECISIONS P4A-D-010 to P4A-D-024).**",
         "- Prices are B4-internal marginals: the price of a constraint X is measured inside the practical bundle B4 "
         "(B4 = non-radioactive + lead-free + e_hull ≤ 0.05 eV/atom + ICSD-matched), either by adding X to B4 "
         "(GAP, ABUND, element-exclusion filter) or by dropping X from B4 (RAD, PB, STAB, EXP).",
         "- Frontier kind best_clean under the Phase 4a artefact flag; rule R2; coherent scattering included; "
         "thickness at the measured-density point; areal mass alongside.",
         "- Narrow beam unless the fluorescence column says otherwise. Prices against the unconstrained frontier F0 "
         "are \"headroom\" and appear only in the headline table's appendix.", "",
         "Status words: **robust**, **conditional**, **fell**. MC status: **supported**, **weakened**, "
         "**contradicted** or **not tested** (Step 5). Reference-map classification: **confirms known**, "
         "**quantifies known** or **new** (Step 6).", ""]
    for rid, title, stmt, status, change in RESULTS:
        x = t[(t.dossier_result == rid) & (t.block == "headline")]
        L += [f"## {rid} {title}", "", f"- **Statement.** {stmt}", f"- **Status:** {status}.",
              f"- **Change since Phase 3:** {change}."]
        L += _ref_lines(rm, rid)
        L.append(_mc_line(rid))
        L.append("")
        if len(x):
            L += _rows_md(x)
            L.append("")
        else:
            L += ["(No headline row; see the Phase 3 dossier for this result.)", ""]
    app = t[t.block == "appendix"]
    L += ["## Appendix: headroom (F0-based) rows", "",
          "These measure the distance to the unconstrained artefact-free frontier, which at the radionuclides is an "
          "actinide. They are not prices of practical choices.", ""]
    L += _rows_md(app)
    L.append("")
    return "\n".join(L) + "\n"


def run() -> dict:
    s = build()
    (common.REPORTS4 / "results_dossier_v2.md").write_text(s, encoding="utf-8")
    return {"chars": len(s)}


if __name__ == "__main__":
    print(run())
