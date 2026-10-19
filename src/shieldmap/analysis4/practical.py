"""Practical-set variants and their memberships and prices.

Variants of "practical" (each changes B4 and therefore B5 and B6):
  primary        RAD strict (as in stages 1-3)
  rad_moderate   RAD moderate: strict + elements whose natural specific activity exceeds potassium's (natrad)
  rad_named      RAD named: strict + {Lu, Re, La}
  noH            H-containing compounds excluded (ambient-state sensitivity; the map has no state field)
  excl_<el>      element-exclusion sensitivity for el ∈ {Tl, As, Cd, Hg, Be, Se} (not toxicology)
Abundance curve: B6(a) = B4 ∧ ABUND(a) for a ∈ ABUND_GRID ppm (scarcity proxy (crustal abundance)).
All prices use the refsets engine (refsets) with the phase4a artefact flag.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common, natrad, refsets

OUT = common.P4 / "S"
ABUND_GRID = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
VARIANTS = ("primary", "rad_moderate", "rad_named", "noH") + tuple(f"excl_{e}" for e in refsets.HAZARD_ELEMENTS)
HEAD_COLS = {"CS137|v38": "thickness", "CO60|v42": "thickness"}   # + the DIAG headline column (diag_head_key)
TOP = 10


def diag_head_key() -> str:
    """DIAG headline column: RQR6, 0.35 mm Pb target, the primary anode angle of stage 3 (the median one)."""
    c = refsets.columns()
    d = c[(c.scenario == "DIAG") & c.coherent & (c.quality == "RQR6")
          & np.isclose(c.pb_thickness_cm, 0.035)].sort_values("anode_angle")
    if d.empty:
        raise RuntimeError("no RQR6 0.35 mm column")
    return d.key.iloc[len(d) // 2]


def variant_masks(v: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(rad mask or None, extra exclusion mask or None) for a variant."""
    if v == "primary":
        return None, None
    if v.startswith("rad_"):
        return natrad.rad_mask(v[4:]), None
    if v == "noH":
        return None, ~c3.core.contains("H")
    if v.startswith("excl_"):
        return None, ~c3.core.contains(v[5:])
    raise ValueError(v)


def all_sets() -> tuple[dict, dict]:
    """Sets for the engine and the B4-internal specs per variant."""
    sets, specs = {}, {}
    for v in VARIANTS:
        rad, extra = variant_masks(v)
        p = "" if v == "primary" else f"{v}:"
        ab = ABUND_GRID if v == "primary" else (1.0,)
        sets.update(refsets.practical_sets(rad=rad, extra=extra, prefix=p, headroom=False, abund=ab))
        specs[v] = refsets.marginal_specs(prefix=p, abund=ab)
    return sets, specs


def memberships(sets: dict) -> pd.DataFrame:
    r2 = c3.base_mask("R2")
    clean = common.clean_mask()
    recs = []
    for v in VARIANTS:
        p = "" if v == "primary" else f"{v}:"
        for b in ("B4", "B5", "B6(1)"):
            m = sets[p + b] & r2
            m0 = sets[b] & r2
            recs.append({"variant": v, "bundle": b, "n_R2": int(m.sum()), "n_R2_clean": int((m & clean).sum()),
                         "delta_vs_primary": int(m.sum() - m0.sum()), "removed": int((m0 & ~m).sum()),
                         "added": int((m & ~m0).sum())})
    return pd.DataFrame(recs)


def top_moves(sets: dict) -> pd.DataFrame:
    """Top-10 (best_clean order, R2, with coherent, phase4a) of B4/B5/B6 at the headline columns per variant,
    and the compounds that move in or out relative to the primary definition."""
    cols = refsets.columns()
    heads = dict(HEAD_COLS); heads[diag_head_key()] = "areal"
    r2 = c3.base_mask("R2") & common.clean_mask()
    u = c3.universe()
    form = u.formula_pretty.to_numpy(); mid = u.material_id.to_numpy()
    recs = []
    for key, basis in heads.items():
        cc = cols[(cols.key == key) & cols.coherent]
        for den in ([d for d in refsets.densities(basis) if d != "mp"] or ["any"]):
            q = refsets.requirement(cc, basis, den)[:, 0]
            for b in ("B4", "B5", "B6(1)"):
                def top(m):
                    idx = np.flatnonzero(m & r2)
                    return list(idx[np.argsort(q[idx], kind="stable")[:TOP]])
                t0 = top(sets[b])
                for v in VARIANTS[1:]:
                    tv = top(sets[f"{v}:{b}"])
                    for r in sorted(set(t0) ^ set(tv), key=lambda r: q[r]):
                        recs.append({"key": key, "basis": basis, "density": den, "bundle": b, "variant": v,
                                     "move": "out" if r in t0 else "in", "rank_primary": (t0.index(r) + 1) if r in t0
                                     else None, "rank_variant": (tv.index(r) + 1) if r in tv else None,
                                     "formula": form[r], "material_id": mid[r], "q": float(q[r])})
    return pd.DataFrame(recs)


def noh_frontier_changes(ft: pd.DataFrame) -> pd.DataFrame:
    """Mono energies where the B4 best_clean frontier compound (R2, coherent, phase4a) changes under noH."""
    cols = refsets.columns().set_index("col")
    s = ft[(ft.rule == "R2") & (ft.kind == "best_clean") & ft["set"].isin(["B4", "noH:B4"])]
    s = s.assign(coherent=cols.loc[s.col].coherent.to_numpy(), key=cols.loc[s.col].key.to_numpy(),
                 energy_keV=cols.loc[s.col].energy_keV.to_numpy())
    s = s[s.coherent]
    w = s.pivot_table(index=["key", "energy_keV", "basis", "density"], columns="set", values=["row", "value"],
                      observed=True, aggfunc="first", dropna=False)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    form = c3.universe().formula_pretty.to_numpy()
    w = w[(w["row_B4"] >= 0) & (w["row_noH:B4"] >= 0)]           # NaN (no frontier) compares False
    w = w[w["row_B4"] != w["row_noH:B4"]].copy()
    w["formula_B4"] = form[w["row_B4"].astype(int)]
    w["formula_noH"] = form[w["row_noH:B4"].astype(int)]
    w["price_noH_vs_B4"] = w["value_noH:B4"] / w["value_B4"] - 1.0
    return w


def run() -> dict:
    paths.ensure(OUT)
    t0 = time.time()
    act, iso, a_k = natrad.activities()
    act.to_csv(OUT / "natrad_elements.csv", index=False)
    iso.to_csv(OUT / "natrad_isotopes.csv", index=False)
    sets, specs = all_sets()
    mem = memberships(sets); mem.to_csv(OUT / "practical_membership.csv", index=False)
    mv = top_moves(sets); mv.to_csv(OUT / "practical_top10_moves.csv", index=False)
    ft = refsets.frontier_table(sets, defs=("phase4a",), log=lambda m: print(m, flush=True))
    ft.to_parquet(OUT / "practical_frontiers.parquet", index=False)
    pr = []
    for v, sp in specs.items():
        p = refsets.prices(ft, sp, "B4-internal")
        p["practical"] = v
        pr.append(p)
    pr = pd.concat(pr, ignore_index=True)
    for k in ("basis", "density", "rule", "artefact_def", "kind", "label", "reference", "key", "scenario", "practical"):
        pr[k] = pr[k].astype("category")
    pr.to_parquet(OUT / "practical_prices.parquet", index=False)
    bd = []
    for v in VARIANTS:
        b = refsets.bands(pr[pr.practical == v])
        b["practical"] = v
        bd.append(b)
    pd.concat(bd, ignore_index=True).to_parquet(OUT / "practical_bands.parquet", index=False)
    nh = noh_frontier_changes(ft); nh.to_csv(OUT / "noH_frontier_changes.csv", index=False)
    summ = {"A_K_Bq_per_g": a_k,
            "moderate_added": act.loc[act.moderate & ~act.strict, "element"].tolist(),
            "named_added": act.loc[act.named & ~act.strict, "element"].tolist(),
            "diag_head_key": diag_head_key(), "seconds": round(time.time() - t0, 1)}
    common.write_json(OUT / "practical_summary.json", summ)
    return summ


if __name__ == "__main__":
    print(run())
