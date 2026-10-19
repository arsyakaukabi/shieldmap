"""Frontiers and prices under the phase4a artefact flag; the engine reused by.

Columns (``columns()``)
  profile   every primary Table C variant (DIAG, CS137, CO60) and its without-coherent twin; the value is
            the requirement at the variant's target (thickness cm or areal mass g/cm²)
  mono      the 511 grid nodes 20 keV–10 MeV, with and without coherent; the value is 1/μ (cm) or 1/(μ/ρ)
            (g/cm²); absolute increments are quoted for T = 0.01 (ln 100 × Δq), as in stage 3
Densities
  mp        Materials Project densities (the map as built)
  measured  the stage 3 measured-density point: every row's thickness × (1 + δ_point), δ_point measured
            (stage 1 element or screened COD) where available, else the cell median (density_model.RowModel,
            ceiling 'unmeasured'); frontiers are recomputed after scaling. Thickness only.
  any       areal mass (density-independent)
Kinds (both sides of a price use the same kind)
  best             best member, artefacts included (diagnostic only)
  best_clean       best member after removing artefact entries under the given definition   HEADLINE
  best_icsd        best ICSD-tagged member; under 'phase4a' artefact entries are removed too (an ICSD tag
                   with e_hull > 0.3 eV/atom or ρ above the cap is an artefact); under 'phase3' as in stage 3
  median50_clean   median of the 50 best artefact-free members
Prices
  B4-internal: for X ∈ {GAP(3.1), ABUND(1), EXCL(el)} P = q*(B4 ∧ X)/q*(B4) − 1;
  for X ∈ {RAD, PB, STAB, EXP} P = q*(B4)/q*(B4 without X) − 1.
  Headroom (appendix): P = q*(F0 ∧ K)/q*(F0) − 1, as in stage 3.
Credible band: envelope over rules R1/R2/R3 × kinds {best_clean, best_icsd, median50_clean} × coherent
(with/without), at fixed density source and artefact definition.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import continuum, frontiers as f3
from shieldmap.analysis4 import common

OUT = common.P4 / "S"
CHUNK = 64
TOPK = 4000
RULES = ("R1", "R2", "R3")
PRIMARY_RULE = "R2"
KINDS = ("best", "best_clean", "best_icsd", "median50_clean")
CREDIBLE_KINDS = ("best_clean", "best_icsd", "median50_clean")
HEADLINE_KIND = "best_clean"
LN100 = math.log(100.0)
B4_DROP = ("RAD", "PB", "STAB", "EXP")
HEADROOM = ("RAD", "PB", "STAB(0.05)", "EXP", "GAP(3.1)", "ABUND(1)", "B1", "B2", "B3", "B4", "B5", "B6")
HAZARD_ELEMENTS = ("Tl", "As", "Cd", "Hg", "Be", "Se")      # element-exclusion sensitivity
ALWAYS_EXCL = ("W", "Ta", "Hf", "Bi", "Ir", "Os", "Pt", "Au", "Re", "Gd", "Sn", "Sb", "Ba") + HAZARD_ELEMENTS
EXCL_TOP = 10                                                 # elements of the B4 top 10 at any column


# ---------------------------------------------------------------------------------------------
# columns, requirements, densities
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def columns() -> pd.DataFrame:
    v = c3.variants()
    recs = []
    for r in v[v.primary & v.scenario.isin(["DIAG", "CS137", "CO60"])].itertuples():
        twin = v[(v.spectrum == r.spectrum) & (v.target_type == r.target_type) & (v.target_value == r.target_value)
                 & (v.response == r.response) & ~v.coherent]
        if len(twin) != 1:
            raise RuntimeError(f"no unique without-coherent twin for variant {r.variant_id}")
        for coh, vid in ((True, int(r.variant_id)), (False, int(twin.variant_id.iloc[0]))):
            recs.append({"key": f"{r.scenario}|v{int(r.variant_id)}", "kind": "profile", "scenario": r.scenario,
                         "variant_id": vid, "primary_variant_id": int(r.variant_id), "coherent": coh,
                         "spectrum": r.spectrum, "quality": r.quality, "anode_angle": r.anode_angle,
                         "T_target": r.T_target, "pb_thickness_cm": r.pb_thickness_cm, "energy_keV": np.nan,
                         "e_idx": -1})
    gi = continuum.grid_indices()
    e = c3.energy_keV()
    for coh in (True, False):
        for i in gi:
            recs.append({"key": f"MONO|{e[i]:g}", "kind": "mono", "scenario": "MONO", "variant_id": -1,
                         "primary_variant_id": -1, "coherent": coh, "spectrum": "", "quality": None,
                         "anode_angle": np.nan, "T_target": 0.01, "pb_thickness_cm": np.nan,
                         "energy_keV": float(e[i]), "e_idx": int(i)})
    out = pd.DataFrame(recs)
    out["col"] = np.arange(len(out))
    return out


@lru_cache(maxsize=1)
def measured_scale() -> np.ndarray:
    """1 + δ_point per universe row (stage 3 measured-density point)."""
    from shieldmap.analysis3 import density_model as dm
    return 1.0 + dm._point_delta(dm.RowModel())


def requirement(cols: pd.DataFrame, basis: str, density: str) -> np.ndarray:
    """N × len(cols) requirement (float64) for a block of columns sharing ``coherent``."""
    coh = bool(cols.coherent.iloc[0])
    if not (cols.coherent == coh).all():
        raise ValueError("a block must share 'coherent'")
    parts = []
    prof = cols[cols.kind == "profile"]
    mono = cols[cols.kind == "mono"]
    if len(prof):
        parts.append(np.asarray(c3.profiles(basis)[:, prof.variant_id.to_numpy()], dtype=np.float64))
    if len(mono):
        parts.append(c3.mono_requirement(basis, coh, e_idx=mono.e_idx.to_numpy()))
    q = np.concatenate(parts, axis=1)
    if density == "measured":
        if basis != "thickness":
            raise ValueError("measured density applies to thickness only")
        q = q * measured_scale()[:, None]
    return q


def densities(basis: str) -> tuple[str, ...]:
    return ("mp", "measured") if basis == "thickness" else ("any",)


def _blocks(cols: pd.DataFrame):
    """Column blocks (profile columns first, then mono) of ≤ CHUNK columns sharing 'coherent'."""
    for coh in (True, False):
        c = cols[cols.coherent == coh]
        c = pd.concat([c[c.kind == "profile"], c[c.kind == "mono"]])
        for s in range(0, len(c), CHUNK):
            yield c.iloc[s:s + CHUNK]


# ---------------------------------------------------------------------------------------------
# sets
# ---------------------------------------------------------------------------------------------
def _b4_parts(rad: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The four B4 constraints; ``rad`` replaces the strict RAD mask (the practical-set module variants)."""
    return {"RAD": c3.constraint_mask("RAD") if rad is None else np.asarray(rad, bool),
            "PB": c3.constraint_mask("PB"), "STAB": c3.constraint_mask("STAB", 0.05),
            "EXP": c3.constraint_mask("EXP")}


def practical_sets(rad: np.ndarray | None = None, extra: np.ndarray | None = None, elements=(),
                   prefix: str = "", headroom: bool = True, abund=(1.0,)) -> dict[str, np.ndarray]:
    """Named masks for one definition of 'practical'.

    B4 = RAD ∧ PB ∧ STAB(0.05) ∧ EXP (∧ extra); B4-X drops X; B5 = B4 ∧ GAP(3.1); B6(a) = B4 ∧ ABUND(a);
    B4&EXCL(el) for the given elements; with ``headroom`` also F0 ∧ {each constraint and bundle}.
    """
    parts = _b4_parts(rad)
    ex = np.ones(len(c3.universe()), bool) if extra is None else np.asarray(extra, bool)
    b4 = parts["RAD"] & parts["PB"] & parts["STAB"] & parts["EXP"] & ex
    p = prefix
    s = {f"{p}B4": b4}
    for x in B4_DROP:
        m = ex.copy()
        for k, v in parts.items():
            if k != x:
                m &= v
        s[f"{p}B4-{x}"] = m
    s[f"{p}B5"] = b4 & c3.constraint_mask("GAP", 3.1)
    for a in abund:
        s[f"{p}B6({a:g})"] = b4 & c3.constraint_mask("ABUND", a)
    for el in elements:
        s[f"{p}B4&EXCL({el})"] = b4 & c3.constraint_mask("EXCL", el)
    if headroom:
        s["F0"] = np.ones(len(c3.universe()), bool)
        for name in HEADROOM:
            s[f"F0&{name}"] = f3._named_mask(name)
    return s


def marginal_specs(elements=(), prefix: str = "", abund=(1.0,)) -> dict[str, tuple[str, str]]:
    """B4-internal marginal prices: label → (numerator set, denominator set)."""
    p = prefix
    out = {"GAP(3.1)|B4": (f"{p}B5", f"{p}B4")}
    for a in abund:
        out[f"ABUND({a:g})|B4"] = (f"{p}B6({a:g})", f"{p}B4")
    for el in elements:
        out[f"EXCL({el})|B4"] = (f"{p}B4&EXCL({el})", f"{p}B4")
    for x in B4_DROP:
        out[f"{x}|B4"] = (f"{p}B4", f"{p}B4-{x}")
    return out


def headroom_specs() -> dict[str, tuple[str, str]]:
    return {f"{k}|F0": (f"F0&{k}", "F0") for k in HEADROOM}


# ---------------------------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------------------------
def frontier_table(sets: dict[str, np.ndarray], cols: pd.DataFrame | None = None, bases=("thickness", "areal"),
                   rules=RULES, coherents=(True, False), defs=common.ARTEFACT_DEFS, kinds=KINDS,
                   dens: dict | None = None, log=None) -> pd.DataFrame:
    """Long table: col, basis, density, rule, set, artefact_def, kind, value, row (−1 for median50)."""
    cols = columns() if cols is None else cols
    cols = cols[cols.coherent.isin(coherents)]
    icsd = c3.icsd_mask()
    clean = {d: common.clean_mask(d) for d in defs}
    names = list(sets)
    out = {k: [] for k in ("col", "basis", "density", "rule", "set", "artefact_def", "kind", "value", "row")}

    def emit(cc, basis, den, rule, sname, d, kind, val, row):
        n = len(cc)
        out["col"].append(cc); out["basis"].append(np.repeat(basis, n)); out["density"].append(np.repeat(den, n))
        out["rule"].append(np.repeat(rule, n)); out["set"].append(np.repeat(sname, n))
        out["artefact_def"].append(np.repeat(d, n)); out["kind"].append(np.repeat(kind, n))
        out["value"].append(np.asarray(val, float)); out["row"].append(np.asarray(row, np.int64))

    t0 = time.time()
    for basis in bases:
        for den in (dens or {}).get(basis, densities(basis)):
            for blk in _blocks(cols):
                q = requirement(blk, basis, den)
                cc = blk.col.to_numpy()
                for rule in rules:
                    tk = continuum.TopK(q, c3.base_mask(rule), TOPK)
                    for sname in names:
                        m = sets[sname]
                        plain = tk.frontiers(m, kinds=("best",)) if "best" in kinds else None
                        for d in defs:
                            if plain is not None:
                                emit(cc, basis, den, rule, sname, d, "best", plain["best"], plain["best_row"])
                            mc = m & clean[d]
                            o = tk.frontiers(mc, kinds=("best", "median50"))
                            if "best_clean" in kinds:
                                emit(cc, basis, den, rule, sname, d, "best_clean", o["best"], o["best_row"])
                            if "median50_clean" in kinds:
                                emit(cc, basis, den, rule, sname, d, "median50_clean", o["median50"],
                                     np.full(len(cc), -1))
                            if "best_icsd" in kinds:
                                mi = m & icsd & (clean[d] if d == "phase4a" else True)
                                oi = tk.frontiers(mi, kinds=("best",))
                                emit(cc, basis, den, rule, sname, d, "best_icsd", oi["best"], oi["best_row"])
                if log:
                    log(f"  {basis}/{den} block {int(cc[0])}–{int(cc[-1])}  {time.time() - t0:.0f} s")
    df = pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})
    for k in ("basis", "density", "rule", "set", "artefact_def", "kind"):
        df[k] = df[k].astype("category")
    return df


def excl_elements(cols: pd.DataFrame | None = None) -> list[str]:
    """Elements present in the B4 top EXCL_TOP (best_clean, phase4a, R2, with coherent) at any column, basis
    and density, plus ALWAYS_EXCL."""
    cols = columns() if cols is None else cols
    cols = cols[cols.coherent]
    u = c3.universe()
    b4 = practical_sets(headroom=False)["B4"] & common.clean_mask() & c3.base_mask(PRIMARY_RULE)
    idx = np.flatnonzero(b4)
    rows = set()
    for basis in ("thickness", "areal"):
        for den in densities(basis):
            for blk in _blocks(cols):
                q = requirement(blk, basis, den)[idx]
                q = np.where(np.isfinite(q), q, np.inf)
                top = np.argpartition(q, EXCL_TOP - 1, axis=0)[:EXCL_TOP]
                rows.update(idx[top.ravel()].tolist())
    els = {e for r in rows for e in u["elements"].iloc[r]}
    return sorted(els | set(ALWAYS_EXCL))


# ---------------------------------------------------------------------------------------------
# prices and bands
# ---------------------------------------------------------------------------------------------
def prices(ft: pd.DataFrame, specs: dict[str, tuple[str, str]], reference: str) -> pd.DataFrame:
    """Price, absolute increment and frontier rows for each spec (label → (num set, den set))."""
    cols = columns().set_index("col")
    keys = ["col", "basis", "density", "rule", "artefact_def", "kind"]
    need = {s for pair in specs.values() for s in pair}
    sub = ft[ft["set"].isin(need)]
    val = sub.pivot_table(index=keys, columns="set", values="value", observed=True, aggfunc="first")
    row = sub.pivot_table(index=keys, columns="set", values="row", observed=True, aggfunc="first")
    recs = []
    for label, (num, den) in specs.items():
        if num not in val.columns or den not in val.columns:
            continue
        d = pd.DataFrame({"q_den": val[den], "q_num": val[num], "row_den": row[den], "row_num": row[num]})
        d["price"] = d.q_num / d.q_den - 1.0
        d["label"] = label
        d["reference"] = reference
        recs.append(d.reset_index())
    p = pd.concat(recs, ignore_index=True)
    c = cols.loc[p.col]
    p["key"] = c.key.to_numpy(); p["scenario"] = c.scenario.to_numpy(); p["coherent"] = c.coherent.to_numpy()
    p["energy_keV"] = c.energy_keV.to_numpy(); p["T_target"] = c.T_target.to_numpy()
    mono = (c.kind == "mono").to_numpy()
    p["delta_abs"] = np.where(mono, LN100 * (p.q_num - p.q_den), p.q_num - p.q_den)
    p["delta_unit"] = np.where(p.basis == "thickness", "cm", "g/cm2")
    p["T_for_delta"] = np.where(mono, 0.01, p.T_target)
    u = c3.universe()
    form = u["formula_pretty"].to_numpy(); mid = u["material_id"].to_numpy()
    for side in ("den", "num"):
        r = p[f"row_{side}"].fillna(-1).astype(np.int64).to_numpy()
        ok = r >= 0
        p[f"formula_{side}"] = np.where(ok, form[np.where(ok, r, 0)], "")
        p[f"mpid_{side}"] = np.where(ok, mid[np.where(ok, r, 0)], "")
    return p


def bands(p: pd.DataFrame) -> pd.DataFrame:
    """Headline (R2, with coherent, best_clean) and credible band per key × basis × density × def × label."""
    keys = ["key", "basis", "density", "artefact_def", "label", "reference"]
    p = p.reset_index(drop=True)
    head = p[(p.rule == PRIMARY_RULE) & p.coherent & (p.kind == HEADLINE_KIND)].set_index(keys)
    cr = p[p.kind.isin(CREDIBLE_KINDS)].groupby(keys, observed=True)
    out = head[["price", "delta_abs", "delta_unit", "T_for_delta", "q_den", "q_num", "formula_den", "formula_num",
                "mpid_den", "mpid_num", "scenario", "energy_keV", "T_target"]].copy()
    out["band_lo"] = cr.price.min(); out["band_hi"] = cr.price.max()
    out["delta_band_lo"] = cr.delta_abs.min(); out["delta_band_hi"] = cr.delta_abs.max()
    # which corner sets each end of the band (rule|coherent|kind|numerator formula)
    tag = (p.rule.astype(str) + "|" + np.where(p.coherent, "coh", "nocoh") + "|" + p.kind.astype(str) + "|"
           + p.formula_num.astype(str) + "/" + p.formula_den.astype(str))
    pp = p.assign(tag=tag)[p.kind.isin(CREDIBLE_KINDS)].dropna(subset=["price"])
    lo = pp.loc[pp.groupby(keys, observed=True).price.idxmin()].set_index(keys).tag
    hi = pp.loc[pp.groupby(keys, observed=True).price.idxmax()].set_index(keys).tag
    out["band_lo_set_by"] = lo; out["band_hi_set_by"] = hi
    return out.reset_index()


# ---------------------------------------------------------------------------------------------
# refsets reports
# ---------------------------------------------------------------------------------------------
def flag_counts() -> pd.DataFrame:
    comp = common.artefact_components()
    old = common.artefact_mask("phase3"); new = common.artefact_mask("phase4a")
    sets = {"eligible": c3.universe()["eligible"].to_numpy(bool)}
    for r in RULES:
        sets[f"F0 {r}"] = c3.base_mask(r)
    sets["B4 R2"] = practical_sets(headroom=False)["B4"] & c3.base_mask("R2")
    recs = []
    for name, m in sets.items():
        recs.append({"set": name, "n": int(m.sum()), "phase3_flag": int((m & old).sum()),
                     "phase4a_flag": int((m & new).sum()), "new_only": int((m & new & ~old).sum()),
                     "dense": int((m & comp.dense.to_numpy()).sum()),
                     "icsd_far": int((m & comp.icsd_far.to_numpy()).sum()),
                     "dense_and_icsd_far": int((m & comp.dense.to_numpy() & comp.icsd_far.to_numpy()).sum())})
    return pd.DataFrame(recs)


def removed_frontier(ft: pd.DataFrame) -> pd.DataFrame:
    """Rows that set a best_clean / best_icsd frontier under 'phase3' and are artefacts under 'phase4a'."""
    new = common.artefact_mask("phase4a")
    comp = common.artefact_components()
    s = ft[(ft.artefact_def == "phase3") & ft.kind.isin(["best_clean", "best_icsd"]) & (ft.row >= 0)]
    s = s[new[s.row.to_numpy()]]
    if s.empty:
        return pd.DataFrame()
    cols = columns().set_index("col")
    s = s.assign(scenario=cols.loc[s.col].scenario.to_numpy())
    g = s.groupby("row", observed=True)
    u = c3.universe()
    out = pd.DataFrame({"n_cells": g.size(), "sets": g["set"].agg(lambda x: ",".join(sorted(set(map(str, x))))),
                        "kinds": g.kind.agg(lambda x: ",".join(sorted(set(map(str, x))))),
                        "scenarios": g.scenario.agg(lambda x: ",".join(sorted(set(x)))),
                        "rules": g.rule.agg(lambda x: ",".join(sorted(set(map(str, x)))))})
    r = out.index.to_numpy()
    out.insert(0, "material_id", u.material_id.to_numpy()[r])
    out.insert(1, "formula", u.formula_pretty.to_numpy()[r])
    out["density"] = u.density.to_numpy()[r]
    out["e_hull"] = u.energy_above_hull.to_numpy()[r]
    out["theoretical"] = u.theoretical.to_numpy()[r]
    out["n_icsd_ids"] = u.n_icsd_ids.to_numpy()[r]
    for k in ("dense", "icsd_far", "conjunctive"):
        out[k] = comp[k].to_numpy()[r]
    return out.sort_values("n_cells", ascending=False).reset_index(names="row")


def run() -> dict:
    paths.ensure(OUT)
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    els = excl_elements()
    log(f"[S] element-exclusion set: {len(els)} elements ({time.time() - t0:.0f} s)")
    sets = practical_sets(elements=els)
    ft = frontier_table(sets, log=log)
    ft.to_parquet(OUT / "frontiers.parquet", index=False)
    pm = prices(ft, marginal_specs(els), "B4-internal")
    ph = prices(ft, headroom_specs(), "headroom")
    pr = pd.concat([pm, ph], ignore_index=True)
    for k in ("basis", "density", "rule", "artefact_def", "kind", "label", "reference", "key", "scenario"):
        pr[k] = pr[k].astype("category")
    pr.to_parquet(OUT / "prices.parquet", index=False)
    bd = bands(pr)
    bd.to_parquet(OUT / "bands.parquet", index=False)
    fc = flag_counts(); fc.to_csv(OUT / "artefact_flag_counts.csv", index=False)
    rf = removed_frontier(ft); rf.to_csv(OUT / "artefact_removed_frontier.csv", index=False)
    columns().to_csv(OUT / "columns.csv", index=False)
    summ = {"excl_elements": els, "n_columns": int(len(columns())), "n_frontier_rows": int(len(ft)),
            "n_price_rows": int(len(pr)), "seconds": round(time.time() - t0, 1)}
    common.write_json(OUT / "summary.json", summ)
    return summ


if __name__ == "__main__":
    print(run())
