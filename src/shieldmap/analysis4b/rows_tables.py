"""Headline rows H154+ that Table 2 needs and the stage 4a table does not hold.

Re-presentation only; every value is read from, or recomputed with the code of, existing stage 3 / stage 4a outputs:

  H154-H157  Pb-relative thickness (narrow-beam) at Co-60 of the Table 2 leaders (Ir, W, BiOF, HfO2): the Co-60
             counterparts of H66, H67, H70, H71 (same routine as analysis4.headline.pb_relative_rows; the stage 4a
             rows carry the Co-60 value only in their sensitivity column)
  H158-H161  density of the same four compounds at the stage 3 measured-density point (analysis4.refsets.
             measured_scale(); all four are measured: stage 1 experimental element density or screened COD match)
  H162-H165  P(best) of the B6 leader at the other abundance thresholds of the stage 3 tier run (2 ppm: Ta;
             0.5 ppm: W), Cs-137 and Co-60, R2 (data/phase3/T/tier_summary.csv)
  H166-H167  crustal abundance of W and Ta (stage 1 Table A, P3 data), which places W between the 1 and 2 ppm
             thresholds
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap import paths

NOTE = "Table 2 (Phase 4b step 3)"
LEADERS = (("H154", "H158", "Ir", "H66"), ("H155", "H159", "W", "H67"), ("H156", "H160", "BiOF", "H70"),
           ("H157", "H161", "HfO2", "H71"))
DENSITY_SOURCE = {"phase1_element": "measured: Phase 1 experimental element density (pymatgen, P1)",
                  "cod": "measured: screened COD structure match (Phase 3)",
                  "conditional": "not measured: cell median of the Phase 3 density model"}


def _pb_relative_and_density() -> list[dict]:
    from shieldmap.analysis3 import common as c3
    from shieldmap.analysis3 import density_model as dm
    from shieldmap.analysis4 import refsets
    from shieldmap.analysis4b.table_v3 import _row

    cols = refsets.columns()
    u = c3.universe()
    r2 = c3.base_mask("R2")
    model = dm.RowModel()
    scale = refsets.measured_scale()
    cc = cols[(cols.key == "CO60|v42") & cols.coherent]
    q_meas = refsets.requirement(cc, "thickness", "measured")[:, 0]
    q_mp = refsets.requirement(cc, "thickness", "mp")[:, 0]
    pb = float(cc.pb_thickness_cm.iloc[0])
    rows = []
    for rid_pb, rid_rho, f, twin in LEADERS:
        idx = np.flatnonzero((u.formula_pretty.to_numpy() == f) & r2)
        if idx.size != 1:
            raise RuntimeError(f"{f}: {idx.size} R2 rows")
        i = int(idx[0])
        mpid = u.material_id.iloc[i]
        v, vmp = q_meas[i] / pb, q_mp[i] / pb
        rows.append(_row(rid_pb, f"Pb-relative thickness (narrow-beam) of {f} at the measured-density point, Co-60 (T = 0.01); Co-60 counterpart of {twin}",
                         float(v), float(min(v, vmp)), float(max(v, vmp)), "ratio",
                         "measured-density point vs MP density (Co-60)", "thickness", "n/a (not a price)",
                         f"{f} ({mpid}) vs Pb at 11.34 g/cm³",
                         "data/processed (Table C profiles) × data/phase3 density model point; data/phase4a/S/columns.csv",
                         f"material_id={mpid}; variant 42 (CO60|v42, coherent); analysis4.headline.pb_relative_rows routine",
                         delta=float(q_meas[i]), delta_unit="cm",
                         delta_note=f"absolute thickness for T = 0.01 at Co-60; Pb {pb:.3f} cm",
                         kind="n/a", density="measured-density point",
                         sens=f"MP-density Co-60 {vmp:.3f}; Cs-137 value in {twin}", dossier="R2.1", note=NOTE))
        rho_mp = float(u.density.iloc[i])
        rho = rho_mp / float(scale[i])
        src = str(model.source[i])
        rows.append(_row(rid_rho, f"Density of {f} ({mpid}) at the measured-density point ({DENSITY_SOURCE[src]})",
                         rho, np.nan, np.nan, "g/cm3", "none (measured value; MP density in the sensitivity column)",
                         "thickness", "n/a (not a price)", f"{f} ({mpid})",
                         "data/interim (Table B MP density) × analysis4.refsets.measured_scale() (Phase 3 density model point)",
                         f"material_id={mpid}; row={i}; source={src}",
                         kind="n/a", density=DENSITY_SOURCE[src], sens=f"MP density {rho_mp:.3f} g/cm³",
                         dossier="R3.4", note=NOTE))
    return rows


def _abundance_rows() -> list[dict]:
    from shieldmap.analysis3 import common as c3
    from shieldmap.analysis4b.table_v3 import _row

    tsum = pd.read_csv(paths.DATA / "phase3" / "T" / "tier_summary.csv")
    rows = []
    for rid, sc, vid, st in (("H162", "CS137", 38, "B6@ABUND2"), ("H163", "CO60", 42, "B6@ABUND2"),
                             ("H164", "CS137", 38, "B6@ABUND0.5"), ("H165", "CO60", 42, "B6@ABUND0.5")):
        sel = tsum[(tsum.scenario == sc) & (tsum.variant_id == vid) & (tsum["set"] == st)]
        t = sel[sel.rule == "R2"].iloc[0]
        a = st.split("ABUND")[1]
        where = {"CS137": "Cs-137", "CO60": "Co-60"}[sc]
        rows.append(_row(rid, f"Thinnest compound of B6 with the scarcity proxy (crustal abundance) at ≥ {a} ppm instead of 1 ppm, {where} (T = 0.01, R2): share of density-model draws in which it is thinnest, P(best)",
                         float(t.top_winner_p_copula), float(sel.top_winner_p_copula.min()), float(sel.top_winner_p_copula.max()),
                         f"P(best) of {t.top_winner_copula}", "R1/R2/R3 (copula)", "thickness",
                         f"{st} (Phase 3 set; contains no artefact entry)", "",
                         "data/phase3/T/tier_summary.csv", f"scenario={sc}; variant_id={vid}; rule=R2; set={st}",
                         kind="n/a", density="density model (copula)",
                         sens=f"no ceiling {t.top_winner_p_copula_noceiling:.2f} ({t.top_winner_copula_noceiling}); unscreened residual {t.top_winner_p_copula_rawresid:.2f} ({t.top_winner_copula_rawresid})",
                         dossier="R3.4", note=NOTE))
    e = c3.elements_table().set_index("symbol")
    for rid, el in (("H166", "W"), ("H167", "Ta")):
        rows.append(_row(rid, f"Crustal abundance of {el} (scarcity proxy input; CRC Handbook 95th ed. via mendeleev 1.3.0, P3)",
                         float(e.at[el, "crustal_abundance_ppm"]), np.nan, np.nan, "ppm", "none (tabulated constant)",
                         "n/a", "n/a (not a price)", el, "data/interim/elements/elements.parquet",
                         f"symbol={el}; column=crustal_abundance_ppm", kind="n/a", density="n/a",
                         dossier="R3.4", note=NOTE))
    return rows


def extra_rows() -> list[dict]:
    rows = _pb_relative_and_density() + _abundance_rows()
    for r in rows:
        r["v3_origin"] = "phase4b-step3-tables"
        r["decision"] = ""
    return sorted(rows, key=lambda r: int(r["id"][1:]))


IDS = [f"H{i}" for i in range(154, 168)]
