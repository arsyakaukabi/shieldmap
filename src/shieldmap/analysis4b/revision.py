"""The small revision analyses.

Only data already in the project are used (Table C machinery, stage 3/4a outputs, Monte Carlo records as
summarised in data/phase4a/mc/). Outputs go to data/phase4b/revision/; rows H180+ are built from them by
analysis4b.rows_revision.

  blend_line()       E2  best binary / ternary elemental blend of {Sn, Sb, Bi, W, Ba, Gd} on the areal-mass
                         basis, narrow beam, per DIAG variant (36): same spectrum quadrature, air-kerma weights and
                         T* as Table C; μ/ρ of a blend = Σ w_Z (μ/ρ)_Z (XCOM port, exactly as Table C builds every
                         compound). Grid ≤ 5 % in mass fraction, refined near the optimum.
  like_for_like()    E3  Monte Carlo diagnostic advantage with the Pb target taken as Pb's own transmission under
                         the same estimator at the nominal Pb areal mass (log T interpolated in areal mass as
                         analysis4.mc.compare does; SE by a joint bootstrap over the sub-run SEs of all three slabs).
  ambient_tier()     E5  ambient-chemistry tiers priced inside B4 with the stage 4a engine (refsets):
                         AMB-ANION (fully oxidised: anions only O/F, every other element in a positive oxidation state)
                         and AMB-ICSD3 (n_icsd_ids ≥ 3); best lanthanide member of each.
  gap_sweep()        E7  GAP(t)|B4 for t = 2.5, 3.1, 3.2, 3.5, 4.0 eV and GAP(3.1) ∧ cubic, same engine.
  zero_setters()     E11 compounds that set every 0 % price of Table 1; flags for frontier compounds that are not
                         air-stable solids (NH₃·H₂O, Hg, Ca metal, Tl) wherever they appear in the headline table.
  fluor_undercount() E12 first-order fluorescence under-count per case family (points.csv, diag_points.csv).
  buildup_alt()      O5  build-up broad / (uncollided + coherent-scattered primary) at the T = 0.01 slab.
  mc_estimators()    O12 fl_all, cone1, cone3 for the two diagnostic Monte Carlo pairs (metrics.csv).

The engine runs once (engine()) for E2's B4 frontier band, E5 and E7: sets × columns {DIAG v0–v35, CS137|v38,
CO60|v42} × both bases (thickness at the measured-density point) × R1–R3 × {best_clean, best_icsd,
median50_clean} × coherent on/off, phase4a artefact flag; every set also exists with H-containing compounds
excluded (prefix 'noH:': Co-60 areal mass).
"""
from __future__ import annotations

import itertools
import math
import re
import time
from functools import lru_cache
from math import gcd

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common, refsets
from shieldmap.analysis4b import patches

OUT = paths.DATA / "phase4b" / "revision"
P4A = common.P4
MC = P4A / "mc"
GARMENT = ("Sn", "Sb", "Bi", "W", "Ba", "Gd")
ZNUM = {"Sn": 50, "Sb": 51, "Ba": 56, "Gd": 64, "W": 74, "Pb": 82, "Bi": 83}
STEP = 0.05                       # coarse mass-fraction grid
REFINE_BIN, REFINE_TER = 0.001, 0.0025
HEAD_VARIANTS = (13, 3, 22)       # RQR6 12° 0.35 mm (headline); RQR5 12° 0.25 mm and RQR8 12° 0.35 mm (MC)
GAP_THRESHOLDS = (2.5, 3.1, 3.2, 3.5, 4.0)
RAD_KEYS = ("CS137|v38", "CO60|v42")
MC_PAIRS = {"c3_rqr5": ("Bi", "LaBi", 3), "c4_rqr8": ("Bi", "Pr3OsI3", 22)}
MC_ESTIMATORS = ("narrow", "cone1", "cone3", "fl", "fl_all", "broad")
N_BOOT = 4000
FLAGGED = {"H5NO": "NH₃·H₂O (MP H5NO): a liquid at room temperature",
           "Hg": "Hg: a liquid metal at room temperature",
           "Ca": "Ca metal: reacts with moist air and water (not an air-stable solid)",
           "Tl": "Tl metal: oxidises in air; highly toxic"}


def _log(m):
    print(m, flush=True)


# ============================================================================================ E2
def _solve_areal(mu_rho_rows: np.ndarray, w: np.ndarray, t_star: float) -> np.ndarray:
    """Areal mass (g/cm²) with narrow-beam T = t_star for each row of μ/ρ (n × K): Table C's solver at ρ = 1."""
    from shieldmap.physics import transmission
    return transmission.solve_thickness(np.atleast_2d(mu_rho_rows), w, t_star)


def _simplex(n_el: int, step: float) -> np.ndarray:
    m = int(round(1 / step))
    pts = [c for c in itertools.product(range(m + 1), repeat=n_el - 1) if sum(c) <= m]
    a = np.array(pts, float) * step
    return np.column_stack([a, 1.0 - a.sum(axis=1)])


def _local(center: np.ndarray, half: float, step: float) -> np.ndarray:
    """Grid of compositions within ±half of the first n−1 fractions of center (clipped to the simplex)."""
    k = int(round(half / step))
    offs = np.arange(-k, k + 1) * step
    grids = np.meshgrid(*[center[i] + offs for i in range(center.size - 1)], indexing="ij")
    a = np.column_stack([g.ravel() for g in grids])
    a = a[(a >= -1e-12).all(axis=1) & (a.sum(axis=1) <= 1 + 1e-12)]
    a = np.clip(a, 0, 1)
    return np.column_stack([a, np.clip(1.0 - a.sum(axis=1), 0, 1)])


def _best_combo(X: np.ndarray, w: np.ndarray, t_star: float, n_el: int, top_refine: int | None = None):
    """Best blend of every n_el-subset of the garment elements: coarse 5 % grid, then local refinement."""
    combos = list(itertools.combinations(range(len(GARMENT)), n_el))
    S = _simplex(n_el, STEP)
    F = np.concatenate([S] * len(combos))
    M = np.concatenate([S @ X[list(c)] for c in combos])
    q = _solve_areal(M, w, t_star).reshape(len(combos), len(S))
    coarse = [(float(q[i].min()), i, S[q[i].argmin()]) for i in range(len(combos))]
    coarse.sort(key=lambda r: r[0])
    refine = coarse if top_refine is None else coarse[:top_refine]
    out = []
    for qc, i, f0 in coarse:
        c = combos[i]
        if any(r[1] == i for r in refine):
            L = _local(f0, STEP, REFINE_BIN if n_el == 2 else REFINE_TER)
            ql = _solve_areal(L @ X[list(c)], w, t_star)
            j = int(ql.argmin())
            qb, fb = float(ql[j]), L[j]
        else:
            qb, fb = qc, f0
        out.append({"elements": "+".join(GARMENT[k] for k in c), "fractions": fb, "q": qb, "q_coarse": qc,
                    "refined": any(r[1] == i for r in refine)})
    return sorted(out, key=lambda r: r["q"])


def _label(elements: str, fr) -> str:
    els = elements.split("+")
    parts = [f"{e}{f:.3f}" for e, f in zip(els, fr) if f > 5e-4]
    return " + ".join(parts)


@lru_cache(maxsize=1)
def _bi_row() -> int:
    u = c3.universe()
    idx = np.flatnonzero((u.formula_pretty.to_numpy() == "Bi") & c3.base_mask("R2"))
    if idx.size != 1:
        raise RuntimeError("Bi: expected one R2 row")
    return int(idx[0])


def blend_line(eng: dict) -> pd.DataFrame:
    from shieldmap.physics import mixture
    v = c3.variants()
    diag = v[(v.scenario == "DIAG") & (v.response == "air_kerma")].sort_values("variant_id")
    prim = diag[diag.coherent]
    if len(prim) != 36:
        raise RuntimeError("expected 36 DIAG variants")
    pr = eng["prices"]
    b4f = pr[(pr.label == "AMB-ANION|B4") & (pr.basis == "areal") & (pr.practical == "primary")]   # q_den = B4 frontier
    cols = refsets.columns()
    prof = c3.profiles("areal")
    ibi = _bi_row()
    recs, detail = [], []
    for r in prim.itertuples():
        twin = diag[(diag.spectrum == r.spectrum) & (diag.target_value == r.target_value) & ~diag.coherent].iloc[0]
        e, w = c3.spectrum_weights(r.spectrum, "air_kerma")
        xs = mixture.element_mu_rho_at(e)
        res = {}
        for coh, vv in ((True, r), (False, twin)):
            X = xs["total_wcoh" if coh else "total_wocoh"][[ZNUM[s] - 1 for s in GARMENT]]
            t_star = float(vv.T_target)
            b2 = _best_combo(X, w, t_star, 2)
            b3 = _best_combo(X, w, t_star, 3, top_refine=3)
            pure = _solve_areal(X, w, t_star)
            res[coh] = {"b2": b2, "b3": b3, "pure": pure, "T": t_star, "vid": int(vv.variant_id)}
            if coh and int(r.variant_id) in HEAD_VARIANTS:
                for kind, lst in (("binary", b2), ("ternary", b3)):
                    for x in lst:
                        detail.append({"variant_id": int(r.variant_id), "kind": kind, "blend": _label(x["elements"], x["fractions"]),
                                       "elements": x["elements"], "q_g_cm2": x["q"], "q_coarse_g_cm2": x["q_coarse"],
                                       "refined": x["refined"]})
                for s, q in zip(GARMENT, pure):
                    detail.append({"variant_id": int(r.variant_id), "kind": "pure", "blend": s, "elements": s,
                                   "q_g_cm2": float(q), "q_coarse_g_cm2": float(q), "refined": False})
        h = res[True]
        best2, best3 = h["b2"][0], h["b3"][0]
        q_best = min(best2["q"], best3["q"])
        key = f"DIAG|v{int(r.variant_id)}"
        f = b4f[b4f.key == key]
        head = f[(f.rule == "R2") & f.coherent & (f.kind == "best_clean")].iloc[0]
        q_bi = {True: float(prof[ibi, h["vid"]]), False: float(prof[ibi, res[False]["vid"]])}
        qb = {coh: min(res[coh]["b2"][0]["q"], res[coh]["b3"][0]["q"]) for coh in (True, False)}
        cr = f[f.kind.isin(refsets.CREDIBLE_KINDS)]
        ratios = np.array([qb[bool(c)] / qd - 1 for c, qd in zip(cr.coherent, cr.q_den)])
        recs.append({"variant_id": int(r.variant_id), "key": key, "quality": r.quality, "anode_angle": r.anode_angle,
                     "pb_mm": 10 * r.pb_thickness_cm, "T_target": r.T_target,
                     "best_binary": _label(best2["elements"], best2["fractions"]), "q_binary": best2["q"],
                     "best_ternary": _label(best3["elements"], best3["fractions"]), "q_ternary": best3["q"],
                     "best_blend": _label(best3["elements"], best3["fractions"]) if best3["q"] <= best2["q"]
                     else _label(best2["elements"], best2["fractions"]),
                     "q_blend": q_best, "q_blend_nocoh": qb[False],
                     "q_B4": float(head.q_den), "B4_frontier": head.formula_den, "B4_frontier_mpid": head.mpid_den,
                     "q_Bi": q_bi[True], "q_Bi_nocoh": q_bi[False],
                     **{f"q_pure_{s}": float(q) for s, q in zip(GARMENT, h["pure"])},
                     "blend_vs_B4": q_best / float(head.q_den) - 1, "blend_vs_B4_lo": float(ratios.min()),
                     "blend_vs_B4_hi": float(ratios.max()),
                     "binary_vs_B4": best2["q"] / float(head.q_den) - 1,
                     "blend_vs_Bi": q_best / q_bi[True] - 1,
                     "blend_vs_Bi_lo": min(qb[c] / q_bi[c] - 1 for c in (True, False)),
                     "blend_vs_Bi_hi": max(qb[c] / q_bi[c] - 1 for c in (True, False)),
                     "B4_vs_Bi": float(head.q_den) / q_bi[True] - 1})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "e2_blend_by_variant.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "e2_blend_detail_v3_v13_v22.csv", index=False)
    return out


# ============================================================================================ E3
def _interp_T(q: np.ndarray, T: np.ndarray, q0: float) -> float:
    """T at areal mass q0: quadratic in q through the three nearest points of log T (compare._solve's model)."""
    o = np.sort(np.argsort(np.abs(q - q0))[:3])
    c = np.polyfit(q[o], np.log(T[o]), 2)
    return float(np.exp(np.polyval(c, q0)))


def _lfl_point(pts: dict, num: str, den: str, est: str, q_pb: float, draw=None):
    from shieldmap.analysis4.mc import compare
    g = {}
    for m in ("Pb", num, den):
        q, T, se = pts[m]["q"], pts[m][est], pts[m][est + "_se"]
        g[m] = (q, T if draw is None else np.maximum(T + draw[m] * se, 1e-300))
    t_pb = _interp_T(*g["Pb"], q_pb)
    qa, ina = compare._solve(*g[num], t_pb)
    qb, inb = compare._solve(*g[den], t_pb)
    return t_pb, qa, qb, ina and inb


def like_for_like() -> pd.DataFrame:
    from shieldmap.analysis4.mc import compare
    cases, refs, _ = compare._load()
    p = pd.read_csv(MC / "points.csv")
    met = pd.read_csv(MC / "metrics.csv")
    v = c3.variants().set_index("variant_id")
    rng = np.random.default_rng(20261013)
    recs = []
    for cid, (num, den, vid) in MC_PAIRS.items():
        q_pb = float(v.at[vid, "pb_thickness_cm"] * v.at[vid, "pb_density_g_cm3"])
        t_fix = float(refs[cid]["T_target"])
        pts = {}
        for m in ("Pb", num, den):
            g = p[(p.case_id == cid) & (p.material == m)].sort_values("q_g_cm2")
            pts[m] = {"q": g.q_g_cm2.to_numpy(float), **{c: g[c].to_numpy(float) for c in g.columns if c in
                                                         [e for e in MC_ESTIMATORS] + [e + "_se" for e in MC_ESTIMATORS]}}
        for est in MC_ESTIMATORS:
            t_pb, qa, qb, inside = _lfl_point(pts, num, den, est, q_pb)
            boots = []
            for _ in range(N_BOOT):
                d = {m: rng.standard_normal(pts[m]["q"].size) for m in pts}
                boots.append(_lfl_point(pts, num, den, est, q_pb, d))
            b = np.array([x[:3] for x in boots])
            adv_b = b[:, 1] / b[:, 2] - 1
            pbbi_b = q_pb / b[:, 1] - 1
            pbln_b = q_pb / b[:, 2] - 1
            # fixed-T check (reproduces metrics.csv)
            qa_f, _ = compare._solve(pts[num]["q"], pts[num][est], t_fix)
            qb_f, _ = compare._solve(pts[den]["q"], pts[den][est], t_fix)
            mrow = met[(met.case_id == cid) & (met.num == num) & (met.den == den) & (met.estimator == est)]
            recs.append({"case_id": cid, "variant_id": vid, "pair": f"{num}/{den}", "estimator": est,
                         "q_Pb_nominal": q_pb, "T_target_fixed": t_fix, "T_Pb_same_estimator": t_pb,
                         "T_Pb_se": float(b[:, 0].std(ddof=1)),
                         "q_num_lfl": qa, "q_num_lfl_se": float(b[:, 1].std(ddof=1)),
                         "q_den_lfl": qb, "q_den_lfl_se": float(b[:, 2].std(ddof=1)),
                         "adv_lfl": qa / qb - 1, "adv_lfl_se": float(adv_b.std(ddof=1)),
                         "Pb_vs_num_lfl": q_pb / qa - 1, "Pb_vs_num_lfl_se": float(pbbi_b.std(ddof=1)),
                         "Pb_vs_den_lfl": q_pb / qb - 1, "Pb_vs_den_lfl_se": float(pbln_b.std(ddof=1)),
                         "adv_fixedT": qa_f / qb_f - 1,
                         "adv_fixedT_metrics_csv": float(mrow.value.iloc[0]) if len(mrow) else np.nan,
                         "adv_fixedT_metrics_se": float(mrow.se.iloc[0]) if len(mrow) else np.nan,
                         "inside_bracket": inside, "n_boot": N_BOOT})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "e3_like_for_like.csv", index=False)
    return out


# ============================================================================================ E5 / E7 sets
@lru_cache(maxsize=None)
def _states(symbol: str, oxyhalide: bool = False) -> tuple[int, ...]:
    """Admissible oxidation states of a non-O/F element in a fully oxidised compound: the positive common oxidation
    states of pymatgen (cations and oxyanion centres), plus the positive ICSD states for d-block metals and
    lanthanides (e.g. Pr⁴⁺, Tb⁴⁺, Cu⁺); p-block low states seen only in ICSD (Bi⁺, Sb⁻…) are not admitted.
    With ``oxyhalide`` Cl/Br/I may also be −1."""
    from pymatgen.core import Element
    e = Element(symbol)
    extra = tuple(e.icsd_oxidation_states) if (e.is_transition_metal or e.is_lanthanoid) else ()
    s = {int(x) for x in tuple(e.common_oxidation_states) + extra if x > 0}
    if oxyhalide and symbol in ("Cl", "Br", "I"):
        s.add(-1)
    return tuple(sorted(s))


def _fully_oxidised(formula: str, oxyhalide: bool = False) -> bool:
    """True when the anions are only O (−2) and F (−1) and every other element can take an admissible state with
    Σ n_i ox_i = 2 n_O + n_F (mixed valence allowed; feasibility on the lattice Σ n_i min_i + G·k within
    [Σ n_i min_i, Σ n_i max_i], G = gcd of the state spacings)."""
    from pymatgen.core import Composition
    amt = Composition(formula).get_el_amt_dict()
    if not ({"O", "F"} & set(amt)):
        return False
    target = 2 * amt.get("O", 0) + amt.get("F", 0)
    lo = hi = 0.0
    g = 0
    for el, n in amt.items():
        if el in ("O", "F"):
            continue
        st = _states(el, oxyhalide)
        if not st:
            return False
        lo += n * st[0]; hi += n * st[-1]
        for a, b in zip(st, st[1:]):
            g = gcd(g, int(b - a))
    if not (lo - 1e-9 <= target <= hi + 1e-9):
        return False
    if g == 0:
        return abs(target - lo) < 1e-9
    k = (target - lo) / g
    return abs(k - round(k)) < 1e-9


@lru_cache(maxsize=1)
def amb_masks() -> dict[str, np.ndarray]:
    u = c3.universe()
    has_of = u["elements"].map(lambda e: ("O" in e) or ("F" in e)).to_numpy()
    form = u.formula_reduced.to_numpy()
    anion = np.zeros(len(u), bool)
    oxhal = np.zeros(len(u), bool)
    for i in np.flatnonzero(has_of):
        anion[i] = _fully_oxidised(form[i])
        oxhal[i] = anion[i] or _fully_oxidised(form[i], oxyhalide=True)
    gap0 = (u.band_gap.fillna(0).to_numpy() > 0)
    icsd3 = (u.n_icsd_ids.fillna(0).to_numpy() >= 3)
    ln = u["elements"].map(lambda e: bool(set(e) & c3.LANTHANIDES)).to_numpy()
    cubic = (u.crystal_system == "Cubic").to_numpy()
    return {"AMB-ANION": anion, "AMB-ANION-OXHAL": oxhal, "AMB-ANION-GAP0": anion & gap0, "AMB-ICSD3": icsd3,
            "AMB-ANION&ICSD3": anion & icsd3, "LN": ln, "CUBIC": cubic}


def _sets() -> tuple[dict, dict]:
    a = amb_masks()
    sets, specs = {}, {}
    for pre, extra in (("", None), ("noH:", ~c3.core.contains("H"))):
        b4 = refsets.practical_sets(extra=extra, prefix=pre, headroom=False, abund=())[f"{pre}B4"]
        sets[f"{pre}B4"] = b4
        for name in ("AMB-ANION", "AMB-ANION-OXHAL", "AMB-ANION-GAP0", "AMB-ICSD3", "AMB-ANION&ICSD3"):
            sets[f"{pre}B4&{name}"] = b4 & a[name]
            sets[f"{pre}B4&LN&{name}"] = b4 & a[name] & a["LN"]
        sets[f"{pre}B4&LN"] = b4 & a["LN"]
        for t in GAP_THRESHOLDS:
            sets[f"{pre}B4&GAP({t:g})"] = b4 & c3.constraint_mask("GAP", t)
        sets[f"{pre}B4&GAP(3.1)&CUBIC"] = b4 & c3.constraint_mask("GAP", 3.1) & a["CUBIC"]
        for s in list(sets):
            if s.startswith(f"{pre}B4&"):
                specs.setdefault(pre, {})[s[len(pre) + 3:] + "|B4"] = (s, f"{pre}B4")
    return sets, specs


def engine() -> dict:
    """Frontiers, prices and bands for every revision set (one engine run)."""
    t0 = time.time()
    sets, specs = _sets()
    cols = refsets.columns()
    cols = cols[(cols.kind == "profile") & (cols.scenario.isin(["DIAG"]) | cols.key.isin(RAD_KEYS))]
    ft = refsets.frontier_table(sets, cols=cols, defs=(common.PRIMARY_ARTEFACT,), kinds=refsets.KINDS,
                                dens={"thickness": ("measured",), "areal": ("any",)},
                                log=lambda m: _log("[engine]" + m))
    pr = []
    for pre, sp in specs.items():
        p = refsets.prices(ft, sp, "B4-internal")
        p["practical"] = "noH" if pre else "primary"
        pr.append(p)
    pr = pd.concat(pr, ignore_index=True)
    bd = []
    for v in ("primary", "noH"):
        b = refsets.bands(pr[pr.practical == v])
        b["practical"] = v
        bd.append(b)
    bd = pd.concat(bd, ignore_index=True)
    for k in ("basis", "density", "artefact_def", "label", "reference", "key", "scenario", "practical"):
        bd[k] = bd[k].astype(str)
    for k in ("basis", "density", "rule", "artefact_def", "kind", "label", "reference", "key", "scenario", "practical"):
        pr[k] = pr[k].astype(str)
    pr.to_parquet(OUT / "engine_prices.parquet", index=False)
    bd.to_csv(OUT / "engine_bands.csv", index=False)
    mem = []
    r2c = c3.base_mask("R2") & common.clean_mask()
    for s, m in sets.items():
        mem.append({"set": s, "n_all": int(m.sum()), "n_R2_clean": int((m & r2c).sum())})
    pd.DataFrame(mem).to_csv(OUT / "engine_membership.csv", index=False)
    _log(f"[engine] done in {time.time() - t0:.0f} s")
    return {"prices": pr, "bands": bd, "sets": sets}


def _pick(bd: pd.DataFrame, key: str, basis: str, label: str, practical: str = "primary") -> pd.Series:
    d = "measured" if basis == "thickness" else "any"
    x = bd[(bd.key == key) & (bd.basis == basis) & (bd.density == d) & (bd.label == label) & (bd.practical == practical)]
    if len(x) != 1:
        raise RuntimeError(f"{key} {basis} {label} {practical}: {len(x)} rows")
    return x.iloc[0]


def _scenario_list():
    """(tag, key, basis, practical) of every place a price is quoted (Co-60 areal on noH)."""
    out = [(f"DIAG v{v}", f"DIAG|v{v}", "areal", "primary") for v in HEAD_VARIANTS]
    out += [("Cs-137 thickness", "CS137|v38", "thickness", "primary"), ("Cs-137 areal", "CS137|v38", "areal", "primary"),
            ("Co-60 thickness", "CO60|v42", "thickness", "primary"), ("Co-60 areal (H excl.)", "CO60|v42", "areal", "noH")]
    return out


def _vs_bi(pr: pd.DataFrame, key: str, label: str) -> tuple[float, float, float]:
    """q*(numerator set)/q(Bi) − 1 (areal): headline and band over the credible corners, Bi at the same coherence."""
    cols = refsets.columns().set_index("col")
    prof = c3.profiles("areal")
    ibi = _bi_row()
    x = pr[(pr.key == key) & (pr.basis == "areal") & (pr.label == label) & (pr.practical == "primary")]
    vid = cols.loc[x.col.to_numpy()].variant_id.to_numpy()
    r = x.q_num.to_numpy() / prof[ibi, vid].astype(float) - 1
    head = r[((x.rule == "R2") & (x.coherent) & (x.kind == "best_clean")).to_numpy()]
    cr = r[x.kind.isin(refsets.CREDIBLE_KINDS).to_numpy()]
    return float(head[0]), float(np.nanmin(cr)), float(np.nanmax(cr))


def ambient_tier(eng: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bd, pr = eng["bands"], eng["prices"]
    recs = []
    for name in ("AMB-ANION", "AMB-ICSD3", "AMB-ANION&ICSD3", "AMB-ANION-OXHAL", "AMB-ANION-GAP0"):
        for tag, key, basis, prac in _scenario_list():
            b = _pick(bd, key, basis, f"{name}|B4", prac)
            recs.append({"tier": name, "where": tag, "key": key, "basis": basis, "practical": prac,
                         "price": b.price, "band_lo": b.band_lo, "band_hi": b.band_hi, "delta_abs": b.delta_abs,
                         "delta_unit": b.delta_unit, "reference": b.formula_den, "reference_mpid": b.mpid_den,
                         "leader": b.formula_num, "leader_mpid": b.mpid_num,
                         "band_lo_set_by": b.band_lo_set_by, "band_hi_set_by": b.band_hi_set_by})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "e5_amb_prices.csv", index=False)
    u = c3.universe()
    mid = pd.Series(np.arange(len(u)), index=u.material_id)
    lns = []
    for name in ("AMB-ANION&ICSD3", "AMB-ANION", "AMB-ICSD3", "AMB-ANION-OXHAL", "AMB-ANION-GAP0", None):
        lab = f"LN&{name}|B4" if name else "LN|B4"
        for v in HEAD_VARIANTS:
            key = f"DIAG|v{v}"
            b = _pick(bd, key, "areal", lab)
            h, lo, hi = _vs_bi(pr, key, lab)
            i = int(mid[b.mpid_num]) if isinstance(b.mpid_num, str) and b.mpid_num else -1
            lns.append({"tier": name or "none (any B4 lanthanide)", "variant_id": v, "key": key,
                        "leader": b.formula_num, "leader_mpid": b.mpid_num,
                        "n_icsd": int(u.n_icsd_ids.iat[i]) if i >= 0 else -1,
                        "spacegroup": int(u.spacegroup_number.iat[i]) if i >= 0 else -1,
                        "q_leader": b.q_num, "q_B4": b.q_den, "B4_frontier": b.formula_den,
                        "vs_B4": b.price, "vs_B4_lo": b.band_lo, "vs_B4_hi": b.band_hi,
                        "vs_Bi": h, "vs_Bi_lo": lo, "vs_Bi_hi": hi,
                        "band_lo_set_by": b.band_lo_set_by, "band_hi_set_by": b.band_hi_set_by})
    ln = pd.DataFrame(lns)
    ln.to_csv(OUT / "e5_amb_lanthanide.csv", index=False)
    return out, ln


def gap_sweep(eng: dict) -> pd.DataFrame:
    bd = eng["bands"]
    u = c3.universe()
    mid = pd.Series(np.arange(len(u)), index=u.material_id)
    recs = []
    places = [x for x in _scenario_list() if not x[0].startswith("DIAG v") or x[0] == "DIAG v13"]
    for lab in [f"GAP({t:g})|B4" for t in GAP_THRESHOLDS] + ["GAP(3.1)&CUBIC|B4"]:
        for tag, key, basis, prac in places:
            b = _pick(bd, key, basis, lab, prac)
            i = int(mid[b.mpid_num])
            recs.append({"set": lab, "threshold_eV": float(re.search(r"GAP\(([\d.]+)\)", lab).group(1)),
                         "cubic_only": "CUBIC" in lab, "where": tag, "key": key, "basis": basis, "practical": prac,
                         "price": b.price, "band_lo": b.band_lo, "band_hi": b.band_hi,
                         "delta_abs": b.delta_abs, "delta_unit": b.delta_unit,
                         "reference": b.formula_den, "leader": b.formula_num, "leader_mpid": b.mpid_num,
                         "leader_gap_eV": float(u.band_gap.iat[i]), "leader_crystal_system": u.crystal_system.iat[i],
                         "leader_spacegroup": int(u.spacegroup_number.iat[i]),
                         "band_lo_set_by": b.band_lo_set_by, "band_hi_set_by": b.band_hi_set_by})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "e7_gap_sweep.csv", index=False)
    return out


def biof_gap() -> dict:
    u = c3.universe()
    x = u[(u.formula_pretty == "BiOF") & u.in_R2].iloc[0]
    d = {"material_id": x.material_id, "formula": x.formula_pretty, "band_gap_eV": float(x.band_gap),
         "margin_over_3.1_eV": float(x.band_gap) - 3.1, "crystal_system": x.crystal_system,
         "spacegroup_number": int(x.spacegroup_number), "energy_above_hull": float(x.energy_above_hull),
         "n_icsd_ids": int(x.n_icsd_ids)}
    pd.DataFrame([d]).to_csv(OUT / "e7_biof_gap.csv", index=False)
    return d


# ============================================================================================ E11
def _parse_key(s: str) -> dict:
    return dict(p.strip().split("=", 1) for p in str(s).split(";") if "=" in p)


def zero_setters() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(patches.P4B / "headline_table_v3.csv")
    ids = [f"H{i:02d}" for i in range(3, 37)] + [f"H{i}" for i in range(133, 138)]
    z = t[t.id.isin(ids) & (t.value_pct.abs() < 1e-12)]
    b = pd.read_parquet(P4A / "S" / "bands.parquet")
    b = b[(b.reference == "B4-internal") & (b.artefact_def == common.PRIMARY_ARTEFACT)]
    pb = pd.read_parquet(P4A / "S" / "practical_bands.parquet")
    pb = pb[pb.artefact_def == common.PRIMARY_ARTEFACT]
    recs = []
    for r in z.itertuples():
        k = _parse_key(r.source_key)
        src = pb[pb.practical == k["practical"]] if "practical" in k else b
        dens = k.get("density", "any" if k["basis"] == "areal" else "measured")
        x = src[(src.key == k["key"]) & (src.label == k["label"]) & (src.basis == k["basis"]) & (src.density == dens)]
        if len(x) != 1:
            raise RuntimeError(f"{r.id}: {len(x)} band rows")
        x = x.iloc[0]
        corners = set()
        for tag in (x.band_lo_set_by, x.band_hi_set_by):
            corners.update(str(tag).split("|")[-1].split("/"))
        fl = sorted({f for f in (x.formula_den, x.formula_num) if f in FLAGGED})
        flc = sorted({f for f in corners if f in FLAGGED})
        recs.append({"id": r.id, "quoted_in": r.quoted_in, "key": k["key"], "label": k["label"], "basis": k["basis"],
                     "density": dens, "practical": k.get("practical", "primary"), "price_pct": 100 * x.price,
                     "band_lo_pct": 100 * x.band_lo, "band_hi_pct": 100 * x.band_hi,
                     "setter_den": x.formula_den, "mpid_den": x.mpid_den, "setter_num": x.formula_num, "mpid_num": x.mpid_num,
                     "same_compound": x.mpid_den == x.mpid_num, "band_lo_set_by": x.band_lo_set_by,
                     "band_hi_set_by": x.band_hi_set_by, "flag_setter": "; ".join(FLAGGED[f] for f in fl),
                     "flag_band_corner": "; ".join(FLAGGED[f] for f in flc),
                     "table_reference_compounds": r.reference_compounds,
                     "consistent_with_table": r.reference_compounds == f"{x.formula_den} → {x.formula_num}"})
    zs = pd.DataFrame(recs)
    zs.to_csv(OUT / "e11_zero_setters.csv", index=False)
    # every headline row whose reference pair or band corners include a flagged compound
    tok = re.compile(r"[A-Z][A-Za-z0-9()·.]*")
    fl = []
    for r in t.itertuples():
        for col in ("reference_compounds", "band_lo_set_by", "band_hi_set_by", "statement_v3"):
            s = getattr(r, col)
            if not isinstance(s, str):
                continue
            words = set(tok.findall(s.replace("→", " ").replace("/", " ").replace("|", " ")))
            for f in FLAGGED:
                if f in words:
                    fl.append({"id": r.id, "quoted_in": r.quoted_in, "column": col, "compound": f, "flag": FLAGGED[f],
                               "value_pct": r.value_pct, "statement_v3": r.statement_v3})
    fg = pd.DataFrame(fl).drop_duplicates(["id", "column", "compound"])
    fg.to_csv(OUT / "e11_flagged_appearances.csv", index=False)
    flagged_in_bands(b, pb)
    return zs, fg


def flagged_in_bands(b: pd.DataFrame, pb: pd.DataFrame) -> pd.DataFrame:
    """Where a flagged compound is the headline reference (either side) of a B4-internal price in the stage 4a
    band tables (primary set and the H-excluded variant): count of cells and energy range per label × basis."""
    recs = []
    for prac, src in (("primary", b), ("noH", pb[pb.practical == "noH"])):
        for f in FLAGGED:
            for side in ("den", "num"):
                x = src[(src[f"formula_{side}"] == f) & src.label.isin(patches.LABELS)]
                for (sc, lab, bas, den), g in x.groupby(["scenario", "label", "basis", "density"], observed=True):
                    recs.append({"compound": f, "flag": FLAGGED[f], "practical": prac, "side": side, "scenario": sc,
                                 "label": lab, "basis": bas, "density": den, "n_cells": len(g),
                                 "energy_keV_min": float(g.energy_keV.min()), "energy_keV_max": float(g.energy_keV.max()),
                                 "keys": "; ".join(sorted(set(map(str, g.key))))[:300]})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "e11_flagged_in_bands.csv", index=False)
    return out


# ============================================================================================ E12
FAMILIES = {
    "single emitter below its K edge (Bi, Pb at RQR5/RQR8)": [("c3_rqr5", "Bi"), ("c3_rqr5", "Pb"), ("c4_rqr8", "Bi"),
                                                              ("c4_rqr8", "Pb")],
    "single emitter above the Pb K edge (Pb, Tl at 89.5 keV)": [("c5_89keV", "Pb"), ("c5_89keV", "Tl")],
    "partner compounds (LaBi, PrSbPd, Pr3OsI3, HfAu2)": [("c3_rqr5", "LaBi"), ("c3_rqr5", "PrSbPd"),
                                                        ("c4_rqr8", "Pr3OsI3"), ("c5_89keV", "HfAu2")],
    "50 keV diagnostic cases (La, Bi, LaBi)": [("d1_50keV_La", "La"), ("d2_50keV_Bi", "Bi"), ("d3_50keV_LaBi", "LaBi")],
}


def fluor_undercount() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.concat([pd.read_csv(MC / "points.csv").assign(source="points.csv"),
                   pd.read_csv(MC / "diag_points.csv").assign(source="diag_points.csv")], ignore_index=True)
    p = p[p.T_firstorder_fluor > 0].copy()
    p["lines_ratio"] = p.lines_hi / p.T_firstorder_fluor
    p["undercount_lines"] = 1 - p.T_firstorder_fluor / p.lines_hi
    p["undercount_lines_se"] = p.T_firstorder_fluor / p.lines_hi ** 2 * p.lines_hi_se
    p["undercount_fl"] = 1 - p.T_firstorder / p.fl
    p["undercount_fl_se"] = p.T_firstorder / p.fl ** 2 * p.fl_se
    fam = {(c, m): f for f, lst in FAMILIES.items() for c, m in lst}
    p["family"] = [fam.get((c, m), "radionuclide cases (Cs-137, Co-60; excluded)") for c, m in zip(p.case_id, p.material)]
    keep = ["family", "source", "case_id", "material", "factor", "q_g_cm2", "lines_hi", "lines_hi_se", "T_firstorder_fluor",
            "lines_ratio", "undercount_lines", "undercount_lines_se", "fl", "fl_se", "T_firstorder", "undercount_fl",
            "undercount_fl_se", "lines_K_hi", "lines_L_hi"]
    pts = p[keep].sort_values(["family", "case_id", "material", "factor"]).reset_index(drop=True)
    pts.to_csv(OUT / "e12_fluor_undercount_points.csv", index=False)
    g = pts.groupby("family")
    fams = pd.DataFrame({"n_points": g.size(), "materials": g.apply(lambda x: ", ".join(sorted(set(x.case_id + ":" + x.material)))),
                         "undercount_lines_min": g.undercount_lines.min(), "undercount_lines_max": g.undercount_lines.max(),
                         "undercount_lines_median": g.undercount_lines.median(),
                         "undercount_fl_min": g.undercount_fl.min(), "undercount_fl_max": g.undercount_fl.max()}).reset_index()
    fams.to_csv(OUT / "e12_fluor_undercount_families.csv", index=False)
    return pts, fams


# ============================================================================================ O5 / O12
BUILDUP_ROWS = {"H124": ("c6_cs137_pb", "Pb"), "H125": ("c1_cs137", "Ir"), "H126": ("c1_cs137", "HfO2"),
                "H127": ("c1_cs137", "BiOF"), "H128": ("c2_co60", "Ir"), "H129": ("c2_co60", "HfO2")}


def buildup_alt() -> pd.DataFrame:
    s = pd.read_csv(MC / "subrun_sums.csv")
    p = pd.read_csv(MC / "points.csv")
    recs = []
    for rid, (cid, mat) in BUILDUP_ROWS.items():
        pt = p[(p.case_id == cid) & (p.material == mat) & np.isclose(p.factor, 1.0)].iloc[0]
        g = s[(s.case == int(pt.case)) & (s.material == mat) & np.isclose(s.factor, 1.0)]
        n = len(g)
        B = g.broad.to_numpy(); N = g.narrow.to_numpy(); C = g.coh_primary.to_numpy()
        D = N + C

        def ratio_se(a, d):
            ma, md = a.mean(), d.mean()
            cov = np.cov(a, d, ddof=1) / len(a)
            r = ma / md
            return r, r * math.sqrt(cov[0, 0] / ma ** 2 + cov[1, 1] / md ** 2 - 2 * cov[0, 1] / (ma * md))
        b0, s0 = ratio_se(B, N)
        b1, s1 = ratio_se(B, D)
        recs.append({"id_conventional": rid, "case_id": cid, "material": mat, "factor": 1.0, "n_subruns": n,
                     "B_conventional": b0, "B_conventional_se": s0, "B_points_csv": float(pt.buildup),
                     "coh_primary_over_narrow": C.mean() / N.mean(),
                     "B_alt": b1, "B_alt_se": s1})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "o5_buildup_alt.csv", index=False)
    return out


def mc_estimators() -> pd.DataFrame:
    m = pd.read_csv(MC / "metrics.csv")
    recs = []
    for cid, (num, den, vid) in MC_PAIRS.items():
        for est in MC_ESTIMATORS:
            x = m[(m.case_id == cid) & (m.num == num) & (m.den == den) & (m.estimator == est)].iloc[0]
            recs.append({"case_id": cid, "variant_id": vid, "pair": f"{num}/{den}", "metric": x.metric, "estimator": est,
                         "value": float(x.value), "se": float(x.se), "inside_bracket": bool(x.inside_bracket)})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "o12_mc_estimators.csv", index=False)
    return out


# ============================================================================================ run
def run() -> dict:
    paths.ensure(OUT)
    t0 = time.time()
    res = {}
    eng = engine()
    b = blend_line(eng)
    res["e2"] = {int(r.variant_id): {"blend": r.best_blend, "vs_B4": r.blend_vs_B4, "vs_Bi": r.blend_vs_Bi}
                 for r in b[b.variant_id.isin(HEAD_VARIANTS)].itertuples()}
    res["e2_range_vs_B4"] = [float(b.blend_vs_B4.min()), float(b.blend_vs_B4.max())]
    _log(f"[E2] {time.time() - t0:.0f} s")
    lf = like_for_like()
    res["e3"] = {f"{r.case_id}:{r.estimator}": [r.adv_lfl, r.adv_lfl_se] for r in lf.itertuples()}
    _log(f"[E3] {time.time() - t0:.0f} s")
    a, ln = ambient_tier(eng)
    res["e5_n"] = {k: int(v.sum()) for k, v in amb_masks().items()}
    g = gap_sweep(eng)
    res["e7_biof"] = biof_gap()
    zs, fg = zero_setters()
    res["e11"] = {"n_zero": len(zs), "n_flagged_rows": int(fg.id.nunique())}
    pts, fams = fluor_undercount()
    res["e12"] = {r.family: [r.undercount_lines_min, r.undercount_lines_max] for r in fams.itertuples()}
    bu = buildup_alt()
    res["o5"] = {r.id_conventional: r.B_alt for r in bu.itertuples()}
    mc_estimators()
    res["seconds"] = round(time.time() - t0, 1)
    common.write_json(OUT / "revision_summary.json", res)
    return res


if __name__ == "__main__":
    print(run())
