"""Logged numbers re-check of H55, H63 and H130-H132.

The check recomputes each row from the raw requirement matrices rather than reading the stage 4a outputs:
  H55/H63    areal requirement on the single-line grid, base R2 ∧ (B4 without its PB constraint), stage 4a
             artefact flag, best_clean: price = min over base ∧ no Pb / min over base − 1, maximum over the
             88.5–90.5 keV nodes.
             For a single line the narrow-beam areal ratio does not depend on T, so H55 and H63 must agree.
  H130-H132  (a) narrow-beam thickness price at MP densities, computed directly; (b) the first-order (E₁)
             price recomputed with the certified stage 3 routine (edgewin.price_cell) on the stage 3 baseline
             (R2 ∧ stage 3 B4, phase3 artefact flag) and on the stage 4a baseline (R2 ∧ stage 4a B4, phase4a artefact flag).
Writes data/phase4b/recheck.csv; the step report logs it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import edgewin as ew
from shieldmap.analysis4 import common, refsets
from shieldmap.analysis4b import patches

TOL_PP = 0.05          # percentage points


def _mono(basis: str, j: int) -> np.ndarray:
    return c3.mono_requirement(basis, True, e_idx=np.array([j]))[:, 0]


def run() -> pd.DataFrame:
    u = c3.universe()
    e = c3.energy_keV()
    v3 = pd.read_csv(patches.P4B / "headline_table_v3.csv").set_index("id")
    # PB is one of B4's own constraints: price = q*(B4) / q*(B4 without PB) − 1, base = R2 ∧ (B4 without PB)
    b4 = refsets.practical_sets(headroom=False)["B4-PB"] & c3.base_mask("R2") & common.clean_mask()
    nopb = ~ew._contains(82)
    recs = []
    best = (-1.0, None)
    for j in np.flatnonzero((e >= 88.5 - 1e-9) & (e <= 90.5 + 1e-9)):
        q = _mono("areal", int(j))
        a, b = np.where(b4, q, np.inf), np.where(b4 & nopb, q, np.inf)
        p = b.min() / a.min() - 1.0
        if p > best[0]:
            best = (p, f"{u.formula_pretty.iat[int(a.argmin())]} → {u.formula_pretty.iat[int(b.argmin())]} at {e[j]:g} keV")
    for rid in ("H55", "H63"):
        recs.append({"id": rid, "check": "direct areal ratio, max over 88.5–90.5 keV nodes", "table_value_pct": v3.at[rid, "value_pct"],
                     "recomputed_pct": 100 * best[0], "detail": best[1]})
    sym = c3.elements_table().set_index("symbol").Z
    lb = pd.read_csv(c3.P3 / "E" / "load_bearing_corrected.csv")
    bases = {"phase3": (c3.base_mask("R2") & c3.bundle_mask("B4"), c3.artefact_flag(np.arange(len(u)))),
             "phase4a": (c3.base_mask("R2") & refsets.practical_sets(headroom=False)["B4"], common.artefact_mask())}
    caches: dict = {}
    for rid, el in (("H130", "W"), ("H131", "Ta"), ("H132", "Hf")):
        x = lb[(lb.element == el) & (lb.baseline == "F0&B4")]
        km = ~ew._contains(int(sym[el]))
        # (a) narrow beam, direct
        nar = []
        for E in sorted(x.energy_keV.unique()):
            j = c3.energy_index(float(E))
            q = _mono("thickness", j)
            base = bases["phase3"][0] & ~bases["phase3"][1]
            nar.append(np.where(base & km, q, np.inf).min() / np.where(base, q, np.inf).min() - 1.0)
        file_nar = x[(x.model == "narrow") & x.kind.isin(["best", "best_clean"])].price.max()
        recs.append({"id": rid, "check": f"narrow-beam counterpart ({el}), direct thickness ratio at MP densities, max over the 3 sampled energies",
                     "table_value_pct": 100 * file_nar, "recomputed_pct": 100 * max(nar), "detail": "compared with the file's narrow rows (sensitivity column of the row)"})
        # (b) E1 with the certified routine
        for name, (base, art) in bases.items():
            vals = []
            for E in sorted(x.energy_keV.unique()):
                j = c3.energy_index(float(E))
                for T in ew.T_LEVELS:
                    res = ew.price_cell(base, km, j, T, "thickness", "E1", caches, art)
                    for kind in ("best", "best_clean"):
                        vals.append((res[kind]["price"], E, T, u.formula_pretty.iat[res[kind]["row_base"]],
                                     u.formula_pretty.iat[res[kind]["row_constrained"]], res[kind]["certified"]))
            m = max(vals, key=lambda t: t[0])
            recs.append({"id": rid, "check": f"E₁ first-order price recomputed (edgewin.price_cell), {name} baseline, max over energies × T × kinds",
                         "table_value_pct": v3.at[rid, "value_pct"], "recomputed_pct": 100 * m[0],
                         "detail": f"{m[3]} → {m[4]} at {m[1]:g} keV, T = {m[2]:g}; certified: {all(v[5] for v in vals)}; band {100 * min(v[0] for v in vals):.2f}–{100 * m[0]:.2f} %"})
    out = pd.DataFrame(recs)
    out["diff_pp"] = out.recomputed_pct - out.table_value_pct
    out["pass"] = out.diff_pp.abs() <= TOL_PP
    out.to_csv(patches.P4B / "recheck.csv", index=False)
    return out
