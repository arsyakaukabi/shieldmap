"""Candidate families and tiers (RQ3, deliberately small).

Regenerate with ``python -m shieldmap.analysis3.families`` from ``shieldmap/`` (outputs in
``data/phase3/C/``, figures ``reports/phase3/figures/fig5_diag_plateau`` and ``C_*``). the hardening revision (F2):
the corrected values use the fluorescence module's revised model (E₁ fluence kernel, K + L lines); fig6_tiers is
drawn by the hardening-revision density-model module; the tier tables here are the class-band model (legacy);
the external glass-density range was removed (reproducibility review F-4); the D-input fingerprint is
written by run() (``d_inputs_sha256.json``); runtimes go to ``_timing.json``.
``--corrected`` only computes the fluorescence-corrected areal mass of the B4 rows that the fluorescence module did not
evaluate (checkpointed per spectrum × row chunk in ``data/phase3/C/_cache/``; resumable).

1. DIAG lanthanide plateau as families (areal basis, rule R2, bundle B4, 36 primary DIAG variants)
   family (fine)    lanthanide set × partner set: the La–Lu constituents and the sorted set of all other
                    elements (the compound's chemical system split into its lanthanide and partner parts)
   family (coarse)  lanthanide set × principal partner: the heaviest non-lanthanide element (largest Z),
                    or "none" for lanthanide-only compounds. On the areal basis μ/ρ is the mass-fraction
                    mean of the constituents, so the heaviest partner carries the second absorption edge
                    that complements the lanthanide K edge; light partners (H, B, C, N, O, F, ...) dilute.
   Ln               lanthanide set alone
   For each family and variant: n members, best and median areal mass narrow-beam ("before") and with
   the first-order K + L fluorescence escape correction of the fluorescence module ("after", fixed-T, E₁ fluence), the
   family best relative to the B4 frontier in the same state, and the number of members within 2 % and
   5 % of that frontier. Family rank (by family best) stability over the 36 variants.
   Corrected values come from ``data/phase3/F/corrected_areal.parquet``; B4 rows missing there are
   computed with the fluorescence module's API (``fluorescence.FluorescenceModel``, same options) in this module.
   Textbook comparison: the ambient phases of ``frontiers.textbook_rows()``; on the areal basis μ/ρ is
   identical across polymorphs, so the corrected value of the R2 representative of the same reduced
   formula is used (checked: identical narrow-beam areal mass).
2. Cs-137 / Co-60 tiers (thickness basis): ``frontiers.tiers`` with ``frontiers.class_band`` (COD band
   from the COD module when available, else the stage 1 elemental band), bundles B4/B5/B6, rules R1/R2/R3,
   all four T targets; tier-1 member lists with ρ, e_hull, ICSD flag, band gap, scarcity proxy (crustal
   abundance), bootstrap class. Areal-basis (density-free) view of the same bundles for contrast.
3. B5 dense wide-gap oxides: densities and gaps of the B5 tier members (map data only).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.sparse as sps

from shieldmap import paths
from shieldmap.analysis3 import common

OUT = common.P3 / "C"
CACHE = OUT / "_cache"
F_DIR = common.P3 / "F"
P_DIR = common.P3 / "P"
D_SUMMARY = common.P3 / "D" / "summary.json"

WITHIN = (0.02, 0.05)                    # "within x % of the frontier"
TEXTBOOK10 = ("W", "Bi", "Sn", "Sb2O3", "BaSO4", "Gd2O3", "WO3", "Bi2O3", "PbO", "Pb")
DISPLAY_DIAG = ("RQR6_th12", 0.35)       # same display variant as the fluorescence module
DISPLAY_T = 0.01                         # radionuclide display target (T-independent for Cs-137)
TOP_RANK = 10                            # family "top-10" count for rank stability
CHUNK = 500                              # rows per corrected-solve task


# =============================================================================================
# family keys (pure functions; tested on synthetic inputs)
# =============================================================================================
def _z_of() -> dict[str, int]:
    et = common.elements_table()
    return dict(zip(et["symbol"], et["Z"].astype(int)))


def family_keys(elements, z_of: dict[str, int] | None = None) -> dict | None:
    """Family keys of one compound from its element list; None if it contains no lanthanide.

    ln       lanthanide set, '+'-joined in Z order (e.g. 'Gd', 'Ce+La')
    partners non-lanthanide elements, '-'-joined in Z order ('' for lanthanide-only compounds)
    fine     'ln | partners'
    principal heaviest non-lanthanide element ('none' if there is none)
    coarse   'ln | principal'
    """
    z_of = z_of or _z_of()
    els = sorted(set(elements), key=lambda s: z_of[s])
    ln = [e for e in els if e in common.LANTHANIDES]
    if not ln:
        return None
    part = [e for e in els if e not in common.LANTHANIDES]
    lns = "+".join(ln)
    principal = part[-1] if part else "none"
    return {"ln": lns, "partners": "-".join(part), "fine": f"{lns} | {'-'.join(part) or 'none'}",
            "principal": principal, "coarse": f"{lns} | {principal}"}


def family_frame(rows: np.ndarray, elements: list, z_of: dict[str, int] | None = None) -> pd.DataFrame:
    """One row per lanthanide-containing input row with its family keys (others dropped)."""
    z_of = z_of or _z_of()
    recs = []
    for r, el in zip(rows, elements):
        k = family_keys(el, z_of)
        if k is not None:
            recs.append({"row": int(r), **k})
    return pd.DataFrame(recs, columns=["row", "ln", "partners", "fine", "principal", "coarse"])


def within_counts(q: np.ndarray, frontier: float, fractions=WITHIN) -> dict[float, int]:
    """Number of finite q with q ≤ frontier·(1 + x) for each x."""
    q = np.asarray(q, dtype=float)
    q = q[np.isfinite(q)]
    return {x: int((q <= frontier * (1.0 + x)).sum()) for x in fractions}


def family_stats(fam: pd.Series, q: np.ndarray, frontier: float, fractions=WITHIN) -> pd.DataFrame:
    """Per family: n, best, median, best/frontier, n within each fraction of the frontier.

    fam  family label per member (index-aligned with q); q member requirements; frontier reference value.
    """
    df = pd.DataFrame({"family": np.asarray(fam), "q": np.asarray(q, dtype=float)})
    df = df[np.isfinite(df.q)]
    g = df.groupby("family", sort=False).q
    out = pd.DataFrame({"n": g.size(), "best": g.min(), "median": g.median()})
    out["best_ratio"] = out["best"] / frontier
    out["median_ratio"] = out["median"] / frontier
    for x in fractions:
        lab = f"n_within_{int(round(100 * x))}pct"
        out[lab] = df.assign(w=df.q <= frontier * (1.0 + x)).groupby("family", sort=False).w.sum().astype(int)
    out["rank"] = out["best"].rank(method="min").astype(int)
    return out.reset_index()


def rank_stability(ranks: pd.DataFrame, top: int = TOP_RANK) -> pd.DataFrame:
    """ranks: family × variant table of ranks (NaN allowed). Median/min/max rank, variants in top-k."""
    r = ranks.astype(float)
    return pd.DataFrame({"rank_median": r.median(axis=1), "rank_min": r.min(axis=1), "rank_max": r.max(axis=1),
                         f"n_variants_top{top}": (r <= top).sum(axis=1),
                         "n_variants_rank1": (r == 1).sum(axis=1)})


# =============================================================================================
# corrected areal mass for the B4 rows fluorescence-module did not evaluate
# =============================================================================================
def _diag_vdf() -> pd.DataFrame:
    from shieldmap.analysis3 import fluorescence as F
    return F._diag_variants("air_kerma")


def _b4_r2() -> np.ndarray:
    return common.base_mask("R2") & common.bundle_mask("B4")


def _lanthanide_mask() -> np.ndarray:
    return np.array([bool(set(e) & common.LANTHANIDES) for e in common.universe()["elements"]])


def f_corrected() -> pd.DataFrame:
    """The fluorescence module's corrected areal masses (fixed-T, primary model): row, variant_id, q_tablec, q_eff_fixedT."""
    return pd.read_parquet(F_DIR / "corrected_areal.parquet",
                           columns=["row", "variant_id", "q_tablec", "q_narrow", "q_eff_fixedT"])


def rows_to_correct(n_check: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(rows to compute, check rows already in the fluorescence module's output).

    Rows to compute: every R2 ∧ B4 lanthanide row without a fluorescence-module value (family statistics), plus every
    R2 ∧ B4 row without a value whose narrow-beam areal mass is within 5 % of the corrected B4 frontier
    in some variant (q_eff ≥ q_narrow, so no other B4 row can be within 5 % after correction).
    Check rows: a fixed, evenly spaced sample of the fluorescence module rows, recomputed to verify the API reproduces it.
    """
    b4 = _b4_r2()
    lan = _lanthanide_mask()
    fc = f_corrected()
    have = np.zeros(len(b4), dtype=bool); have[fc.row.unique()] = True
    vids = _diag_vdf()["variant_id"].to_numpy()
    P = np.asarray(common.profiles("areal")[:, vids], dtype=np.float64)
    fb = fc[b4[fc.row.to_numpy()]].groupby("variant_id").q_eff_fixedT.min().reindex(vids).to_numpy()
    near = (P <= fb[None, :] * (1.0 + max(WITHIN))).any(axis=1)
    need = b4 & ~have & (lan | near)
    hv = np.flatnonzero(have & b4)
    check = hv[np.linspace(0, hv.size - 1, n_check).astype(int)]
    return np.flatnonzero(need), check


def _solve_task(args) -> dict:
    """Narrow and fixed-T corrected areal mass for one spectrum × row chunk (the fluorescence module options)."""
    sp, recs, W = args
    from shieldmap.analysis3 import fluorescence as F
    e, phi = common.spectrum(sp)
    m = F.FluorescenceModel(W, e, phi, response="air_kerma")
    qn = np.stack([m.solve_narrow(r["T_target"]) for r in recs], axis=1)
    qe = np.stack([m.solve_effective(r["T_target"], lo=qn[:, j]) for j, r in enumerate(recs)], axis=1)
    return {"q_narrow": qn, "q_eff": qe, "vids": [int(r["variant_id"]) for r in recs]}


def compute_corrected(n_workers: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Corrected areal mass of rows_to_correct() (+ check rows); writes OUT/corrected_extra.parquet.

    Idempotent: if the parquet exists for the same row set it is returned unchanged. Each task result is
    checkpointed in CACHE, so an interrupted run resumes."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    from shieldmap.analysis3 import fluorescence as F
    need, check = rows_to_correct()
    rows = np.concatenate([need, check])
    target = OUT / "corrected_extra.parquet"
    if target.exists():
        old = pd.read_parquet(target)
        if ("model_version" in old.columns and set(old.model_version.unique()) == {F.MODEL_VERSION}
                and set(old.row.unique()) == set(rows.tolist())):
            return old
    import hashlib
    ver = hashlib.sha256(F.MODEL_VERSION.encode()).hexdigest()[:8]
    paths.ensure(CACHE)
    vdf = _diag_vdf()
    W = common.mass_fractions()[rows].toarray()
    tasks, keys = [], []
    for sp, g in vdf.groupby("spectrum", sort=True):
        recs = g.sort_values("pb_mm")[["variant_id", "T_target"]].to_dict("records")
        for s0 in range(0, rows.size, CHUNK):
            tasks.append((sp, recs, W[s0:s0 + CHUNK]))
            keys.append(f"{sp}_{ver}_{s0:06d}_{rows.size}")
    todo = [i for i, k in enumerate(keys) if not (CACHE / f"{k}.npz").exists()]
    log(f"[C] corrected solve: {need.size} rows + {check.size} check rows; {len(todo)}/{len(tasks)} tasks to run")
    t0 = time.time()

    def save(i, r):
        tmp = CACHE / f"{keys[i]}.tmp.npz"
        np.savez(tmp, q_narrow=r["q_narrow"], q_eff=r["q_eff"], vids=np.array(r["vids"]))
        os.replace(tmp, CACHE / f"{keys[i]}.npz")

    if todo:
        if n_workers <= 1:
            for n, i in enumerate(todo):
                save(i, _solve_task(tasks[i]))
                log(f"    {n + 1}/{len(todo)} ({time.time() - t0:.0f} s)")
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_solve_task, tasks[i]): i for i in todo}
                for n, fu in enumerate(as_completed(futs)):
                    save(futs[fu], fu.result())
                    log(f"    {n + 1}/{len(todo)} ({time.time() - t0:.0f} s)")
    recs = []
    for i, (sp, _, _) in enumerate(tasks):
        with np.load(CACHE / f"{keys[i]}.npz") as z:
            qn_, qe_, vv_ = z["q_narrow"], z["q_eff"], z["vids"]
        s0 = int(keys[i].split("_")[-2])
        rr = rows[s0:s0 + qe_.shape[0]]
        for j, v in enumerate(vv_):
            recs.append(pd.DataFrame({"row": rr, "variant_id": int(v), "q_narrow": qn_[:, j],
                                      "q_eff_fixedT": qe_[:, j], "is_check": np.isin(rr, check)}))
    df = pd.concat(recs, ignore_index=True).sort_values(["variant_id", "row"]).reset_index(drop=True)
    df["model_version"] = F.MODEL_VERSION
    paths.ensure(OUT)
    df.to_parquet(target, index=False)
    for f in CACHE.glob("*.npz"):
        f.unlink()
    try:
        CACHE.rmdir()
    except OSError:
        pass
    log(f"[C] corrected solve done in {time.time() - t0:.0f} s")
    return df


# =============================================================================================
# item 1: DIAG families
# =============================================================================================
LEVELS = ("ln", "coarse", "fine")
STATES = ("before", "after")


def _summ(a) -> dict:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def diag_inputs() -> dict:
    """Narrow-beam (Table C) and corrected areal mass on the 36 primary DIAG variants, frontiers, checks."""
    u = common.universe()
    vdf = _diag_vdf().reset_index(drop=True)
    vids = vdf["variant_id"].to_numpy()
    col = {int(v): j for j, v in enumerate(vids)}
    b4 = _b4_r2()
    lan = _lanthanide_mask()
    P = np.asarray(common.profiles("areal")[:, vids], dtype=np.float64)
    QE = np.full(P.shape, np.nan)
    fc = f_corrected()
    QE[fc.row.to_numpy(), fc.variant_id.map(col).to_numpy()] = fc.q_eff_fixedT.to_numpy()
    ex = compute_corrected(verbose=False)
    chk = ex[ex.is_check]
    ref = fc.set_index(["row", "variant_id"]).q_eff_fixedT
    rel_check = np.abs(chk.q_eff_fixedT.to_numpy() / ref.loc[list(zip(chk.row, chk.variant_id))].to_numpy() - 1)
    rel_narrow = np.abs(ex.q_narrow.to_numpy() / P[ex.row.to_numpy(), ex.variant_id.map(col).to_numpy()] - 1)
    new = ex[~ex.is_check]
    QE[new.row.to_numpy(), new.variant_id.map(col).to_numpy()] = new.q_eff_fixedT.to_numpy()
    f_n = np.nanmin(np.where(b4[:, None], P, np.nan), axis=0)
    f_e = np.nanmin(np.where(b4[:, None], QE, np.nan), axis=0)
    # completeness: every B4 row that could lie within the widest window after correction has a value
    could = b4[:, None] & (P <= f_e[None, :] * (1 + max(WITHIN)))
    missing = int((could & ~np.isfinite(QE)).sum())
    if missing:
        raise RuntimeError(f"{missing} B4 row × variant cells near the corrected frontier lack a corrected value")
    if not np.isfinite(QE[b4 & lan]).all():
        raise RuntimeError("some B4 lanthanide rows lack a corrected value")
    fam = family_frame(np.flatnonzero(b4 & lan), u["elements"].to_numpy()[b4 & lan])
    return {"u": u, "vdf": vdf, "vids": vids, "b4": b4, "lan": lan, "P": P, "QE": QE,
            "f": {"before": f_n, "after": f_e}, "fam": fam,
            "checks": {"n_check_rows": int(chk.row.nunique()),
                       "max_rel_diff_check_vs_step2": float(rel_check.max()),
                       "max_rel_diff_narrow_vs_tablec": float(rel_narrow.max()),
                       "n_rows_computed_here": int(new.row.nunique()),
                       "n_b4_lanthanide_rows": int((b4 & lan).sum())}}


def _q(d: dict, state: str) -> np.ndarray:
    return d["P"] if state == "before" else d["QE"]


def family_table(d: dict) -> pd.DataFrame:
    """Long table: level × state × variant × family statistics (family_stats)."""
    fam = d["fam"]
    rows = fam.row.to_numpy()
    out = []
    for state in STATES:
        Q = _q(d, state)
        for j, v in enumerate(d["vids"]):
            fr = d["f"][state][j]
            for level in LEVELS:
                st = family_stats(fam[level], Q[rows, j], fr)
                st.insert(0, "variant_id", int(v))
                st.insert(0, "state", state)
                st.insert(0, "level", level)
                out.append(st)
    return pd.concat(out, ignore_index=True)


def textbook_ambient() -> pd.DataFrame:
    """Ambient phases (frontiers.textbook_phases, rutile/anatase dropped) with the R2 row of the formula."""
    from shieldmap.analysis3 import frontiers
    u = common.universe()
    base = common.base_mask("R2")
    tp = frontiers.textbook_phases()
    tp = tp[tp.formula != "TiO2"].set_index("formula").loc[list(TEXTBOOK10)].reset_index()
    tp["R2_row"] = [int(np.flatnonzero((u["formula_reduced"] == f).to_numpy() & base)[0]) for f in tp.formula]
    return tp


def textbook_by_variant(d: dict, fams: pd.DataFrame) -> pd.DataFrame:
    """Per textbook material × variant × state: q, ratio to the B4 frontier and to the best B4 lanthanide.

    On the areal basis the ambient phase and the R2 representative have identical requirements (checked);
    the corrected value is the fluorescence module's value for the R2 representative."""
    tp = textbook_ambient()
    fc = f_corrected()
    col = {int(v): j for j, v in enumerate(d["vids"])}
    bl = {s: fams[(fams.level == "ln") & (fams.state == s)].groupby("variant_id").best.min() for s in STATES}
    recs = []
    for _, t in tp.iterrows():
        amb, r2 = int(t.row), int(t.R2_row)
        diff = np.abs(d["P"][amb] / d["P"][r2] - 1).max()
        if diff > 1e-6:
            raise RuntimeError(f"{t.formula}: ambient and R2 areal requirements differ ({diff:.2e})")
        qe = fc[fc.row == r2].set_index("variant_id").q_eff_fixedT
        for v in d["vids"]:
            j = col[int(v)]
            for state, q in (("before", float(d["P"][amb, j])), ("after", float(qe.loc[v]))):
                recs.append({"formula": t.formula, "phase": t.phase, "material_id": t.material_id,
                             "R2_material_id": t.R2_material_id, "variant_id": int(v), "state": state, "q": q,
                             "ratio_to_B4_frontier": q / d["f"][state][j],
                             "ratio_to_best_B4_lanthanide": q / bl[state].loc[v]})
    df = pd.DataFrame(recs)
    df["rank_among_10"] = df.groupby(["variant_id", "state"]).q.rank(method="min").astype(int)
    return df


def textbook_table(tbv: pd.DataFrame) -> pd.DataFrame:
    """Range over the 36 variants of each textbook material's ratios, before and after (wide)."""
    out = tbv.groupby(["formula", "state"]).agg(
        ratio_frontier_min=("ratio_to_B4_frontier", "min"), ratio_frontier_median=("ratio_to_B4_frontier", "median"),
        ratio_frontier_max=("ratio_to_B4_frontier", "max"),
        ratio_best_ln_min=("ratio_to_best_B4_lanthanide", "min"),
        ratio_best_ln_max=("ratio_to_best_B4_lanthanide", "max"),
        rank_median=("rank_among_10", "median")).reset_index()
    w = out.pivot(index="formula", columns="state")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    meta = tbv.drop_duplicates("formula").set_index("formula")[["phase", "material_id", "R2_material_id"]]
    return meta.join(w).loc[list(TEXTBOOK10)].reset_index()


def plateau_by_variant(d: dict, fams: pd.DataFrame, tbv: pd.DataFrame) -> pd.DataFrame:
    """Per variant × state: frontier compound, plateau size (all B4 / lanthanide), families on the plateau,
    the best family per level, and the families whose best beats the best textbook material (excl. Pb, PbO)."""
    u, b4, lan = d["u"], d["b4"], d["lan"]
    recs = []
    for state in STATES:
        Q = _q(d, state)
        for j, v in d["vdf"].iterrows():
            vid = int(v.variant_id)
            fr = d["f"][state][j]
            r0 = int(np.nanargmin(np.where(b4, Q[:, j], np.nan)))
            tb = tbv[(tbv.variant_id == vid) & (tbv.state == state) & ~tbv.formula.isin(["Pb", "PbO"])]
            best_tb = tb.loc[tb.q.idxmin()]
            rec = {"variant_id": vid, "spectrum": v.spectrum, "quality": v.quality,
                   "anode_angle": int(v.anode_angle), "pb_mm": float(v.pb_mm), "state": state,
                   "frontier_g_cm2": fr, "frontier_material_id": u.material_id.iat[r0],
                   "frontier_formula": u.formula_pretty.iat[r0], "frontier_is_lanthanide": bool(lan[r0]),
                   "best_textbook_excl_Pb": best_tb.formula, "best_textbook_ratio": best_tb.ratio_to_B4_frontier}
            for x in WITHIN:
                k = int(round(100 * x))
                w = b4 & (Q[:, j] <= fr * (1 + x))
                rec[f"n_b4_within_{k}pct"] = int(w.sum())
                rec[f"n_b4_lanthanide_within_{k}pct"] = int((w & lan).sum())
            f = fams[(fams.state == state) & (fams.variant_id == vid)]
            for level in LEVELS:
                fl = f[f.level == level]
                for x in WITHIN:
                    k = int(round(100 * x))
                    rec[f"n_{level}_families_within_{k}pct"] = int((fl[f"n_within_{k}pct"] > 0).sum())
                rec[f"n_{level}_families_beat_best_textbook"] = int((fl.best < best_tb.q).sum())
                rec[f"best_{level}_family"] = fl.loc[fl.best.idxmin(), "family"]
            rec["best_lanthanide_ratio"] = float(f[f.level == "ln"].best.min() / fr)
            recs.append(rec)
    return pd.DataFrame(recs)


def family_stability(fams: pd.DataFrame) -> pd.DataFrame:
    """Per level × state × family: rank stability over the 36 variants, best-ratio range, member counts."""
    out = []
    for (level, state), g in fams.groupby(["level", "state"], sort=False):
        st = rank_stability(g.pivot(index="family", columns="variant_id", values="rank"))
        agg = g.groupby("family").agg(
            n_members=("n", "max"), best_ratio_min=("best_ratio", "min"), best_ratio_median=("best_ratio", "median"),
            best_ratio_max=("best_ratio", "max"), median_ratio_median=("median_ratio", "median"),
            n_within_2pct_max=("n_within_2pct", "max"), n_within_5pct_max=("n_within_5pct", "max"),
            n_variants_any_within_2pct=("n_within_2pct", lambda s: int((s > 0).sum())))
        st = st.join(agg)
        st.insert(0, "state", state)
        st.insert(0, "level", level)
        out.append(st.reset_index())
    return pd.concat(out, ignore_index=True).sort_values(["level", "state", "rank_median"]).reset_index(drop=True)


def rank_correlation(fams: pd.DataFrame, level: str = "coarse", top: int = 30) -> dict:
    """Spearman ρ of family-best ranks between all variant pairs (families in any variant's top-`top`),
    and between before and after for the same variant (families in the top-`top` of either)."""
    from scipy.stats import spearmanr
    out = {}
    for state in STATES:
        g = fams[(fams.level == level) & (fams.state == state)]
        rk = g.pivot(index="family", columns="variant_id", values="rank")
        keep = (rk <= top).any(axis=1)
        rho = np.atleast_2d(spearmanr(rk[keep].to_numpy(dtype=float)).statistic)
        out[state] = {"n_families": int(keep.sum()),
                      "pairwise_spearman": _summ(rho[np.triu_indices(rho.shape[0], 1)])}
    gb = fams[(fams.level == level) & (fams.state == "before")].set_index(["variant_id", "family"]).best
    ga = fams[(fams.level == level) & (fams.state == "after")].set_index(["variant_id", "family"]).best
    per_v = []
    for v in gb.index.get_level_values(0).unique():
        b, a = gb.loc[v], ga.loc[v].reindex(gb.loc[v].index)
        keep = (b.rank() <= top) | (a.rank() <= top)
        per_v.append(spearmanr(b[keep], a[keep]).statistic)
    out["before_vs_after_same_variant"] = _summ(per_v)
    out["level"], out["top"] = level, top
    return out


# =============================================================================================
# item 2: Cs-137 / Co-60 tiers (thickness basis)
# =============================================================================================
TIER_BUNDLES = ("B4", "B5", "B6")
TIER_SCENARIOS = ("CS137", "CO60")
MEMBER_COLS = ["material_id", "formula_pretty", "density", "energy_above_hull", "theoretical", "n_icsd_ids",
               "band_gap", "min_crustal_abundance_ppm"]


def d_status() -> dict:
    """The COD module status: summary.json present? which band does frontiers.class_band() return?"""
    from shieldmap.analysis3 import frontiers
    bands, source = frontiers.class_band()
    st = {"d_summary_exists": D_SUMMARY.exists(), "band_source": source,
          "bands": {k: [float(v[0]), float(v[1])] for k, v in bands.items()}}
    if D_SUMMARY.exists():
        try:
            s = json.loads(D_SUMMARY.read_text(encoding="utf-8"))
            st["d_summary_keys"] = sorted(s.keys())[:40]
            for k in ("status", "outcome", "cod_status", "failure"):
                if k in s:
                    st[f"d_{k}"] = s[k]
        except Exception as exc:                                   # pragma: no cover
            st["d_summary_error"] = repr(exc)
    return st


def tier_listing(q: np.ndarray, mask: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_tier: int = 3) -> pd.DataFrame:
    """Rows in tiers 1..max_tier with their requirement interval (frontiers.tiers); no order inside a tier.

    Returns row, tier, q, q_lo = q(1+lo), q_hi = q(1+hi), U_star of the row's tier; sorted by tier, row."""
    from shieldmap.analysis3 import frontiers
    q = np.asarray(q, dtype=float)
    t = frontiers.tiers(q, mask, lo, hi, max_tier)
    r = np.flatnonzero(t)
    df = pd.DataFrame({"row": r, "tier": t[r], "q": q[r], "q_lo": q[r] * (1 + lo[r]), "q_hi": q[r] * (1 + hi[r])})
    df["U_star"] = df.groupby("tier").q_hi.transform("min")
    return df.sort_values(["tier", "row"]).reset_index(drop=True)


def tier_members(max_tier: int = 3) -> tuple[pd.DataFrame, dict]:
    """Tier lists for CS137/CO60 (all 4 T targets) × R1/R2/R3 × B4/B5/B6 with member attributes."""
    from shieldmap.analysis3 import frontiers
    u = common.universe()
    bands, source = frontiers.class_band()
    cls = common.universe_primary_class()
    lo = np.array([bands[c][0] for c in cls])
    hi = np.array([bands[c][1] for c in cls])
    Q = common.profiles("thickness")
    recs = []
    for sc in TIER_SCENARIOS:
        for _, v in common.scenario_variants(sc).iterrows():
            col = np.asarray(Q[:, int(v.variant_id)], dtype=float)
            for rule in common.RULES:
                base = common.base_mask(rule)
                for b in TIER_BUNDLES:
                    df = tier_listing(col, base & common.bundle_mask(b), lo, hi, max_tier)
                    df.insert(0, "bundle", b)
                    df.insert(0, "rule", rule)
                    df.insert(0, "T", float(v.target_value))
                    df.insert(0, "variant_id", int(v.variant_id))
                    df.insert(0, "scenario", sc)
                    recs.append(df)
    df = pd.concat(recs, ignore_index=True)
    df["bootstrap_class"] = cls[df.row.to_numpy()]
    df = df.join(u[MEMBER_COLS], on="row")
    df["icsd_matched"] = ~df.pop("theoretical").astype(bool)
    df["artefact_flag"] = common.artefact_flag(df.row.to_numpy())
    df["oxide"] = ["O" in e for e in u["elements"].to_numpy()[df.row.to_numpy()]]
    return df, {"band_source": source, "bands": {k: [float(a), float(b)] for k, (a, b) in bands.items()}}


def alternative_bands() -> dict[str, dict[str, tuple[float, float]]]:
    """Bands for the tier sensitivity: the class band in use (frontiers.class_band), the pooled COD band
    (one interval for every class; when the COD module is available) and the stage 1 elemental band."""
    from shieldmap.analysis3 import frontiers
    from shieldmap.analysis import sensitivity
    out = {"class (in use)": frontiers.class_band()[0]}
    try:
        from shieldmap.analysis3 import cod
        pooled = tuple(cod.band("pooled"))
        out["COD pooled"] = {c: pooled for c in common.PRIMARY_CLASS_ORDER}
    except Exception:                                               # The COD module not available
        pass
    x = sensitivity.bias_distribution("empirical")
    el = (float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95)))
    out["Phase 1 elemental"] = {c: el for c in common.PRIMARY_CLASS_ORDER}
    return out


def band_sensitivity() -> pd.DataFrame:
    """Tier-1 sets (R2, T = DISPLAY_T, B4/B5/B6) under each band of alternative_bands(), vs the band in use."""
    u = common.universe()
    cls = common.universe_primary_class()
    Q = common.profiles("thickness")
    bands = alternative_bands()
    recs = []
    for sc in TIER_SCENARIOS:
        v = common.scenario_variants(sc)
        vid = int(v.loc[np.isclose(v.target_value, DISPLAY_T), "variant_id"].iloc[0])
        col = np.asarray(Q[:, vid], dtype=float)
        for b in TIER_BUNDLES:
            m = common.base_mask(common.PRIMARY_RULE) & common.bundle_mask(b)
            sets = {}
            for name, bd in bands.items():
                lo = np.array([bd[c][0] for c in cls]); hi = np.array([bd[c][1] for c in cls])
                t = tier_listing(col, m, lo, hi, 2)
                sets[name] = (set(t.row[t.tier == 1]), int((t.tier == 2).sum()))
            ref = sets["class (in use)"][0]
            for name, (s1, n2) in sets.items():
                recs.append({"scenario": sc, "variant_id": vid, "bundle": b, "band": name, "n_tier1": len(s1),
                             "n_tier2": n2, "jaccard_tier1_vs_in_use": len(s1 & ref) / len(s1 | ref),
                             "added_vs_in_use": ";".join(sorted(u.formula_pretty.to_numpy()[sorted(s1 - ref)])),
                             "missing_vs_in_use": ";".join(sorted(u.formula_pretty.to_numpy()[sorted(ref - s1)]))})
    return pd.DataFrame(recs)


def tier_sizes(tm: pd.DataFrame) -> pd.DataFrame:
    return (tm.groupby(["scenario", "variant_id", "T", "rule", "bundle", "tier"])
              .agg(n=("row", "size"), n_formulas=("formula_pretty", "nunique"),
                   n_icsd=("icsd_matched", "sum"), n_oxide=("oxide", "sum"),
                   rho_min=("density", "min"), rho_max=("density", "max"),
                   q_min=("q", "min"), q_max=("q", "max"), U_star=("U_star", "first")).reset_index())


def t_dependence(tm: pd.DataFrame) -> pd.DataFrame:
    """Per scenario × rule × bundle × tier: are the member sets identical across the four T targets?"""
    recs = []
    for (sc, rule, b, t), g in tm.groupby(["scenario", "rule", "bundle", "tier"]):
        sets = {T: frozenset(gg.row) for T, gg in g.groupby("T")}
        ref = sets.get(DISPLAY_T, next(iter(sets.values())))
        jac = [len(s & ref) / len(s | ref) for s in sets.values()]
        recs.append({"scenario": sc, "rule": rule, "bundle": b, "tier": t, "n_T": len(sets),
                     "identical_across_T": all(s == ref for s in sets.values()),
                     "min_jaccard_vs_display_T": float(min(jac)),
                     "sizes_by_T": ";".join(f"{T:g}:{len(s)}" for T, s in sorted(sets.items()))})
    return pd.DataFrame(recs)


def rule_sensitivity(tm: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 formula sets under R1/R3 vs R2 (display T): Jaccard on reduced formulas."""
    recs = []
    d = tm[(tm["T"] == DISPLAY_T) & (tm.tier == 1)]
    for (sc, b), g in d.groupby(["scenario", "bundle"]):
        s = {r: set(gg.formula_pretty) for r, gg in g.groupby("rule")}
        ref = s.get(common.PRIMARY_RULE, set())
        for r, x in s.items():
            recs.append({"scenario": sc, "bundle": b, "rule": r, "n_tier1": len(x),
                         "jaccard_formulas_vs_R2": len(x & ref) / len(x | ref) if (x | ref) else np.nan,
                         "only_in_rule": ";".join(sorted(x - ref)), "only_in_R2": ";".join(sorted(ref - x))})
    return pd.DataFrame(recs)


def areal_view(top: int = 5) -> pd.DataFrame:
    """Density-free contrast: areal-basis frontier of the same bundles (R2, display T): the best `top`
    members and the number within 2 %/5 % of the areal frontier (no density band applies)."""
    u = common.universe()
    A = common.profiles("areal")
    recs = []
    for sc in TIER_SCENARIOS:
        v = common.scenario_variants(sc)
        vid = int(v.loc[np.isclose(v.target_value, DISPLAY_T), "variant_id"].iloc[0])
        col = np.asarray(A[:, vid], dtype=float)
        for b in TIER_BUNDLES:
            m = common.base_mask("R2") & common.bundle_mask(b)
            fr = float(np.nanmin(col[m]))
            wc = within_counts(col[m], fr)
            for k, r in enumerate(common.top_k(col, m, top)):
                recs.append({"scenario": sc, "variant_id": vid, "bundle": b, "areal_frontier_g_cm2": fr,
                             "n_within_2pct": wc[0.02], "n_within_5pct": wc[0.05], "listed": k + 1,
                             "row": int(r), "material_id": u.material_id.iat[r], "formula": u.formula_pretty.iat[r],
                             "q_areal_g_cm2": col[r], "ratio_to_areal_frontier": col[r] / fr,
                             "density": float(u.density.iat[r]), "band_gap": float(u.band_gap.iat[r])})
    return pd.DataFrame(recs)


def bootstrap_crossref(tm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap tier stability for CS137/CO60 × B4/B5/B6 (display T), and the per-member
    stay fraction for this step's R2 tier-1 members (empty frames if P's files are absent)."""
    fs, fm = P_DIR / "bootstrap_tier_summary.csv", P_DIR / "bootstrap_tiers.csv"
    if not (fs.exists() and fm.exists()):
        return pd.DataFrame(), pd.DataFrame()
    s = pd.read_csv(fs)
    vids = {sc: int(common.scenario_variants(sc).set_index("target_value").loc[DISPLAY_T, "variant_id"])
            for sc in TIER_SCENARIOS}
    s = s[s.bundle.isin(TIER_BUNDLES) & s.apply(lambda r: vids.get(r.scenario) == r.variant_id, axis=1)]
    mine = tm[(tm.rule == common.PRIMARY_RULE) & (tm.tier == 1) & (tm["T"] == DISPLAY_T)]
    n1 = mine.groupby(["scenario", "bundle"]).size().rename("n_tier1_this_step")
    s = s.join(n1, on=["scenario", "bundle"])
    m = pd.read_csv(fm)
    m = m[m.bundle.isin(TIER_BUNDLES) & m.apply(lambda r: vids.get(r.scenario) == r.variant_id, axis=1)]
    mem = mine[["scenario", "bundle", "row", "formula_pretty"]].merge(
        m[["mode", "scenario", "bundle", "row", "nominal_tier1", "frac_draws_in_tier1", "frac_draws_true_best"]],
        on=["scenario", "bundle", "row"], how="left")
    return s.reset_index(drop=True), mem


# =============================================================================================
# item 3: B5 dense wide-gap oxides
# =============================================================================================
PHASE2_B5 = ("HfO2", "Lu2WO6", "ErTaO4", "Dy2Hf2O7", "TlF")


def b5_table(tm: pd.DataFrame) -> pd.DataFrame:
    """B5 tier 1–2 members (R2, display T, CS137 and CO60) ∪ stage 2's B5 consensus formulas, with
    density, gap, e_hull, ICSD flag, scarcity proxy and the tier per scenario and rule."""
    u = common.universe()
    d = tm[(tm.bundle == "B5") & (tm["T"] == DISPLAY_T)]
    rows = set(d[(d.rule == common.PRIMARY_RULE) & (d.tier <= 2)].row)
    base = common.base_mask("R2")
    for f in PHASE2_B5:
        r = np.flatnonzero((u.formula_reduced == f).to_numpy() & base)
        rows.update(int(x) for x in r)
    rows = sorted(rows)
    out = u.loc[rows, ["row"] + MEMBER_COLS + ["spacegroup_number"]].copy()
    out["icsd_matched"] = ~out.pop("theoretical").astype(bool)
    out["oxide"] = ["O" in e for e in u.elements.to_numpy()[rows]]
    out["phase2_b5_consensus"] = u.formula_reduced.to_numpy()[rows]
    out["phase2_b5_consensus"] = out["phase2_b5_consensus"].isin(PHASE2_B5)
    out["in_R2_B5"] = (base & common.bundle_mask("B5"))[rows]
    for sc in TIER_SCENARIOS:
        for rule in common.RULES:
            t = d[(d.scenario == sc) & (d.rule == rule)].set_index("row").tier
            out[f"tier_{sc}_{rule}"] = t.reindex(rows).fillna(0).astype(int).to_numpy()
        q = np.asarray(common.profiles("thickness")[rows, int(common.scenario_variants(sc).set_index(
            "target_value").loc[DISPLAY_T, "variant_id"])], dtype=float)
        out[f"q_cm_{sc}_T{DISPLAY_T:g}"] = q
    return out.sort_values(["tier_CS137_R2", "density"], ascending=[True, False]).reset_index(drop=True)


# =============================================================================================
# figures
# =============================================================================================
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
C_BEFORE = "#8c8c8c"
C_AFTER = "#2a78d6"
C_TB = "#eb6834"
INK = "#333333"


def _fmt(f: str) -> str:
    return f.translate(_SUB)


def _display_col(d: dict) -> int:
    v = d["vdf"]
    return int(np.flatnonzero((v.spectrum == DISPLAY_DIAG[0]).to_numpy() & np.isclose(v.pb_mm, DISPLAY_DIAG[1]))[0])


def fig5(d: dict, fams: pd.DataFrame, tbv: pd.DataFrame, pv: pd.DataFrame, n_fam: int = 12) -> list[str]:
    """Figure 5: DIAG lanthanide plateau as families, before/after the fluorescence correction.

    (a) the plateau itself (families, zoomed to 0.995–ZOOM_HI of the B4 frontier); (b) textbook materials on
    their own axis; (c) plateau breadth per variant (every variant labelled); (d) textbook ranges over the 36
    variants. The referee's readability points (separate panel for the plateau, legend for every line,
    variant labels, the variants with no family within 2 %) are addressed here."""
    from matplotlib.lines import Line2D
    plt = common.style()
    ZOOM_HI = 1.10
    fig = plt.figure(figsize=(common.DOUBLE_COL_IN, 6.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.85], wspace=0.5, hspace=0.72)
    ax1 = fig.add_subplot(gs[0, 0]); ax1b = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1])
    j = _display_col(d)
    vid = int(d["vids"][j])
    title_v = f"{DISPLAY_DIAG[0].replace('_th', ' ')}°, T* of {DISPLAY_DIAG[1]} mm Pb"

    # (a) the plateau: coarse families (best after correction), members within the zoom window
    fa = fams[(fams.level == "coarse") & (fams.variant_id == vid)]
    aft = fa[fa.state == "after"].nsmallest(n_fam, "best").set_index("family")
    bef = fa[fa.state == "before"].set_index("family").reindex(aft.index)
    fam = d["fam"]
    ylab = []
    for y, name in enumerate(aft.index):
        rows = fam.row[fam.coarse == name].to_numpy()
        ra = d["QE"][rows, j] / d["f"]["after"][j]
        ra = ra[ra <= ZOOM_HI]
        ax1.plot(ra, np.full(ra.size, y), "|", color=C_AFTER, ms=5, alpha=0.45, mew=0.6)
        b = float(bef.loc[name, "best_ratio"])
        if b <= ZOOM_HI:
            ax1.plot(b, y, "o", mfc="white", mec=C_BEFORE, ms=4.2, mew=0.9)
        else:
            ax1.annotate(f"{b:.2f}", (ZOOM_HI, y), xytext=(-2, 0), textcoords="offset points", ha="right",
                         va="center", fontsize=5.2, color=C_BEFORE)
        ax1.plot(aft.loc[name, "best_ratio"], y, "o", color=C_AFTER, ms=4.2)
        ylab.append(f"{name.split(' | ')[0]}–{name.split(' | ')[1]} (n={int(aft.loc[name, 'n'])})")
    for x in WITHIN:
        ax1.axvline(1 + x, color=INK, lw=0.5, ls=":")
    ax1.axvline(1.0, color=INK, lw=0.6)
    ax1.set_yticks(range(len(ylab)))
    ax1.set_yticklabels([_fmt(s) for s in ylab], fontsize=6.3)
    ax1.set_ylim(len(ylab) - 0.4, -0.6)
    ax1.set_xlim(0.995, ZOOM_HI)
    ax1.set_xlabel("areal mass / B4 frontier (same state)")
    ax1.set_title(f"(a) plateau families (Ln–principal partner), {title_v}", loc="left", fontsize=7)
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec=C_BEFORE, label="family best, narrow beam"),
         Line2D([], [], marker="o", ls="none", color=C_AFTER, label="family best, corrected"),
         Line2D([], [], marker="|", ls="none", color=C_AFTER, label=f"members ≤ {ZOOM_HI:.2f}, corrected"),
         Line2D([], [], color=INK, lw=0.5, ls=":", label="2 % and 5 % above the frontier")]
    ax1.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.6, frameon=False)
    ax1.grid(axis="x", color="#eeeeee", lw=0.5)

    # (b) textbook materials, same variant, with the best family for reference
    tb = tbv[tbv.variant_id == vid].pivot(index="formula", columns="state", values="ratio_to_B4_frontier")
    tb = tb.loc[[f for f in TEXTBOOK10]].sort_values("after")
    for y, f in enumerate(tb.index):
        ax1b.plot([tb.loc[f, "before"], tb.loc[f, "after"]], [y, y], color="#dddddd", lw=0.8, zorder=0)
        ax1b.plot(tb.loc[f, "before"], y, "D", mfc="white", mec=C_BEFORE, ms=3.6, mew=0.9)
        ax1b.plot(tb.loc[f, "after"], y, "D", color=C_TB, ms=3.6)
    ax1b.axvline(1.0, color=INK, lw=0.6)
    ax1b.axvline(float(aft.best_ratio.min()), color=C_AFTER, lw=0.6, ls="--")
    ax1b.set_yticks(range(len(tb)))
    ax1b.set_yticklabels([_fmt(f) + (" (β)" if f == "Sn" else "") for f in tb.index], fontsize=6.3)
    ax1b.set_ylim(len(tb) - 0.4, -0.6)
    ax1b.set_xlabel("areal mass / B4 frontier (same state)")
    ax1b.set_title("(b) textbook materials, same variant", loc="left", fontsize=7)
    ax1b.set_xlim(0.97, float(np.nanmax(tb.values)) * 1.04)
    h = [Line2D([], [], marker="D", ls="none", mfc="white", mec=C_BEFORE, label="narrow beam"),
         Line2D([], [], marker="D", ls="none", color=C_TB, label="corrected"),
         Line2D([], [], color=C_AFTER, lw=0.6, ls="--", label="best family, corrected")]
    ax1b.legend(handles=h, loc="upper right", fontsize=5.6, frameon=True, framealpha=0.95, edgecolor="#dddddd")
    ax1b.grid(axis="x", color="#eeeeee", lw=0.5)

    # (c) plateau breadth per variant: coarse families with a member within 2 % of the frontier
    vd = d["vdf"].sort_values(["quality", "pb_mm", "anode_angle"])
    order = vd.index.to_numpy()
    x = np.arange(len(d["vids"]))
    zero_after = []
    for state, kw in (("before", dict(mfc="white", mec=C_BEFORE)), ("after", dict(color=C_AFTER))):
        p = pv[pv.state == state].set_index("variant_id").loc[d["vids"][order]]
        ax2.plot(x, p["n_coarse_families_within_2pct"], "o", ms=3.2, ls="none", mew=0.8, **kw,
                 label="narrow beam" if state == "before" else "corrected")
        if state == "after":
            zero_after = [k for k, n in enumerate(p["n_coarse_families_within_2pct"].to_numpy()) if n == 0]
    for k in zero_after:
        ax2.plot(k, 0, "x", color="#c0392b", ms=4, mew=0.9)
    for k in range(1, 4):
        ax2.axvline(9 * k - 0.5, color="#e6e6e6", lw=0.5, zorder=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(a)}° {p:.2f}" for a, p in zip(vd.anode_angle, vd.pb_mm)], rotation=90, fontsize=4.6)
    ymax = float(pv.n_coarse_families_within_2pct.max())
    for k, q in enumerate(("RQR5", "RQR6", "RQR8", "RQR9")):
        ax2.text(9 * k + 4, 1.01, q, ha="center", va="bottom", fontsize=6.3, transform=ax2.get_xaxis_transform())
    ax2.set_ylim(-0.6, ymax * 1.45 + 1)
    ax2.set_xlim(-0.8, len(x) - 0.2)
    ax2.set_ylabel("families within 2 %")
    ax2.set_xlabel("anode angle, T* (mm Pb)")
    ax2.set_title("(c) families within 2 % of the B4 frontier", loc="left", fontsize=7, pad=11)
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec=C_BEFORE, label="narrow beam"),
         Line2D([], [], marker="o", ls="none", color=C_AFTER, label="corrected"),
         Line2D([], [], marker="x", ls="none", color="#c0392b", label=f"no family within 2 % after correction "
                                                                      f"({len(zero_after)} variants)")]
    ax2.legend(handles=h, fontsize=5.6, frameon=False, loc="upper left", ncol=1)

    # (d) textbook ratio to the B4 frontier: range over 36 variants, before → after
    tt = tbv.groupby(["formula", "state"]).ratio_to_B4_frontier.agg(["min", "median", "max"])
    order_tb = tt.xs("after", level="state").sort_values("median").index
    for i, f in enumerate(order_tb):
        for k, (state, col) in enumerate((("before", C_BEFORE), ("after", C_TB))):
            r = tt.loc[(f, state)]
            yy = i + (k - 0.5) * 0.32
            ax3.plot([r["min"], r["max"]], [yy, yy], color=col, lw=1.6, solid_capstyle="round")
            ax3.plot(r["median"], yy, "|", color=INK, ms=4, mew=0.8)
    ax3.set_yticks(range(len(order_tb)))
    ax3.set_yticklabels([_fmt(f) for f in order_tb], fontsize=6.3)
    ax3.invert_yaxis()
    ax3.axvline(1.0, color=INK, lw=0.6)
    ax3.set_xlabel("textbook areal mass / B4 frontier (36 variants)")
    ax3.set_title("(d) textbook materials: range over 36 variants", loc="left", fontsize=7, pad=11)
    h = [Line2D([], [], color=C_BEFORE, lw=1.6, label="narrow beam"),
         Line2D([], [], color=C_TB, lw=1.6, label="corrected"),
         Line2D([], [], marker="|", ls="none", color=INK, label="median")]
    ax3.legend(handles=h, fontsize=5.6, frameon=True, framealpha=0.95, edgecolor="#dddddd", loc="upper right")
    ax3.grid(axis="x", color="#eeeeee", lw=0.5)
    out = common.savefig(fig, "fig5_diag_plateau")
    plt.close(fig)
    return out


def fig_b5(b5: pd.DataFrame) -> list[str]:
    """C_b5_density_gap: B5 (R2) members, tier 1–2 (class-band model) at Cs-137/Co-60 highlighted."""
    plt = common.style()
    u = common.universe()
    m = common.base_mask("R2") & common.bundle_mask("B5")
    fig, ax = plt.subplots(figsize=(common.SINGLE_COL_IN, 2.8))
    ax.scatter(u.density[m], u.band_gap[m], s=2, color="#cfcfcf", lw=0, label="R2 ∧ B5 (all)", rasterized=True)
    t1 = b5[(b5.tier_CS137_R2 == 1) | (b5.tier_CO60_R2 == 1)]
    t2 = b5[~b5.index.isin(t1.index) & ((b5.tier_CS137_R2 == 2) | (b5.tier_CO60_R2 == 2))]
    ax.scatter(t2.density, t2.band_gap, s=10, color="#9bbfe8", lw=0, label="tier 2 (Cs-137 or Co-60)")
    ax.scatter(t1.density, t1.band_gap, s=14, color=C_AFTER, lw=0, label="tier 1 (Cs-137 or Co-60)")
    offs = {"HfO2": (-6, 16), "ErTaO4": (-38, 14), "Lu2WO6": (-4, -14), "Dy2Hf2O7": (-46, -14), "TlF": (-26, 10)}
    for _, r in b5[b5.phase2_b5_consensus].iterrows():
        ax.annotate(_fmt(r.formula_pretty), (r.density, r.band_gap), fontsize=5.6,
                    xytext=offs.get(r.formula_pretty, (3, 3)), textcoords="offset points", color=INK,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777"))
    ax.set_ylim(2.55, None)
    ax.set_xlabel("MP density (g/cm³)")
    ax.set_ylabel("PBE band gap (eV)")
    ax.legend(fontsize=5.6, frameon=False, loc="lower left")
    out = common.savefig(fig, "C_b5_density_gap")
    plt.close(fig)
    return out


def fig_family_heatmap(fams: pd.DataFrame, d: dict, n_fam: int = 20) -> list[str]:
    """C_family_heatmap: corrected family-best / B4 frontier for the most often top-ranked coarse families
    over the 36 variants (rows ordered by median rank after correction)."""
    plt = common.style()
    g = fams[(fams.level == "coarse") & (fams.state == "after")]
    rk = g.pivot(index="family", columns="variant_id", values="rank")
    sel = rk.median(axis=1).nsmallest(n_fam).index
    order = d["vdf"].sort_values(["quality", "pb_mm", "anode_angle"]).variant_id.to_numpy()
    M = g.pivot(index="family", columns="variant_id", values="best_ratio").loc[sel, order]
    fig, ax = plt.subplots(figsize=(common.DOUBLE_COL_IN, 3.6))
    im = ax.imshow(M.to_numpy(), aspect="auto", cmap="Blues_r", vmin=1.0, vmax=1.10)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([_fmt(s.replace(" | ", "–")) for s in sel], fontsize=6)
    lab = d["vdf"].set_index("variant_id").loc[order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{q} {int(a)}° {p:.2f}" for q, a, p in zip(lab.quality, lab.anode_angle, lab.pb_mm)],
                       rotation=90, fontsize=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("family best / B4 frontier (corrected)", fontsize=6)
    out = common.savefig(fig, "C_family_heatmap")
    plt.close(fig)
    return out


# =============================================================================================
def run(verbose: bool = True) -> dict:
    """Items 1–3 end to end (idempotent: the corrected solve is reused when its row set is unchanged)."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()
    paths.ensure(OUT)
    compute_corrected(verbose=verbose)
    d = diag_inputs()
    log(f"[C] DIAG inputs: {d['checks']}")
    fams = family_table(d)
    fams.to_parquet(OUT / "families_by_variant.parquet", index=False)
    tbv = textbook_by_variant(d, fams)
    tbv.to_csv(OUT / "textbook_by_variant.csv", index=False)
    tbt = textbook_table(tbv)
    tbt.to_csv(OUT / "textbook_comparison.csv", index=False)
    pv = plateau_by_variant(d, fams, tbv)
    pv.to_csv(OUT / "plateau_by_variant.csv", index=False)
    fs = family_stability(fams)
    fs.to_parquet(OUT / "family_stability.parquet", index=False)
    top = fs[(fs.n_variants_top10 > 0)]
    top.to_csv(OUT / "family_top10.csv", index=False)
    fam_members = d["fam"].merge(d["u"][["row", "material_id", "formula_pretty", "energy_above_hull",
                                         "min_crustal_abundance_ppm"]], on="row")
    fam_members.to_csv(OUT / "family_members.csv", index=False)
    rc = {lv: rank_correlation(fams, lv) for lv in ("ln", "coarse")}
    log("[C] families done")

    dst = d_status()
    tm, band = tier_members()
    tm.to_csv(OUT / "tier_members.csv", index=False)
    ts = tier_sizes(tm)
    ts.to_csv(OUT / "tier_sizes.csv", index=False)
    td = t_dependence(tm)
    td.to_csv(OUT / "tier_T_dependence.csv", index=False)
    rs = rule_sensitivity(tm)
    rs.to_csv(OUT / "tier_rule_sensitivity.csv", index=False)
    av = areal_view()
    av.to_csv(OUT / "tier_areal_view.csv", index=False)
    bsens = band_sensitivity()
    bsens.to_csv(OUT / "tier_band_sensitivity.csv", index=False)
    bs, bm = bootstrap_crossref(tm)
    bs.to_csv(OUT / "tier_bootstrap_crossref.csv", index=False)
    bm.to_csv(OUT / "tier_bootstrap_members.csv", index=False)
    b5 = b5_table(tm)
    b5 = b5.assign(_k=b5.tier_CS137_R2.replace(0, 99)).sort_values(["_k", "density"], ascending=[True, False]) \
           .drop(columns="_k").reset_index(drop=True)
    b5.to_csv(OUT / "b5_oxides.csv", index=False)
    log("[C] tiers done")

    figs = fig5(d, fams, tbv, pv) + fig_b5(b5) + fig_family_heatmap(fams, d)
    summ = _summary(d, fams, tbt, tbv, pv, fs, rc, dst, tm, ts, td, rs, av, bs, bm, b5, band)
    summ["figures"] = figs
    summ["d_inputs"] = write_d_fingerprint()
    common.write_json(OUT / "summary.json", summ)
    secs = time.time() - t0
    common.write_json(OUT / "_timing.json", {"seconds": secs})
    log(f"[C] done in {secs:.0f} s")
    return summ


def write_d_fingerprint() -> dict:
    """SHA-256 of the COD-module inputs this step used (written by the pipeline: C/d_inputs_sha256.json)."""
    files = [common.P3 / "D" / "band.json", common.P3 / "D" / "delta_by_primary_class.parquet",
             paths.ROOT / "src" / "shieldmap" / "analysis3" / "cod.py"]
    fp = {str(f.relative_to(paths.ROOT)).replace("\\", "/"): (common.sha256(f) if f.exists() else None) for f in files}
    common.write_json(OUT / "d_inputs_sha256.json", fp)
    return fp


def _rng(s) -> dict:
    return _summ(np.asarray(s, dtype=float))


def _summary(d, fams, tbt, tbv, pv, fs, rc, dst, tm, ts, td, rs, av, bs, bm, b5, band) -> dict:
    out = {"agent": "C", "step": 7, "checks": d["checks"], "within": list(WITHIN),
           "display_diag": list(DISPLAY_DIAG), "display_T": DISPLAY_T}
    n_fam = d["fam"][list(LEVELS)].nunique().to_dict()
    out["n_families"] = {k: int(v) for k, v in n_fam.items()}
    pl = {}
    for state in STATES:
        p = pv[pv.state == state]
        pl[state] = {c: _rng(p[c]) for c in p.columns if c.startswith("n_") or c in ("best_lanthanide_ratio",
                                                                                     "best_textbook_ratio")}
        pl[state]["frontier_is_lanthanide"] = int(p.frontier_is_lanthanide.sum())
        pl[state]["frontier_formulas"] = p.frontier_formula.value_counts().to_dict()
        pl[state]["best_textbook_excl_Pb"] = p.best_textbook_excl_Pb.value_counts().to_dict()
        for lv in LEVELS:
            pl[state][f"best_{lv}_family"] = p[f"best_{lv}_family"].value_counts().to_dict()
    out["plateau"] = pl
    # lanthanide advantage over the best textbook material excluding Pb/PbO (reproduces the fluorescence module's metric)
    adv = {}
    for state in STATES:
        p = pv[pv.state == state].set_index("variant_id")
        t = tbv[(tbv.state == state) & ~tbv.formula.isin(["Pb", "PbO"])].groupby("variant_id") \
            .ratio_to_best_B4_lanthanide.min()
        adv[state] = {"range": _rng(t - 1.0), "n_ge_10pct": int((t - 1.0 >= 0.10).sum())}
    out["lanthanide_advantage_over_best_textbook"] = adv
    out["textbook"] = tbt.round(4).to_dict("records")
    keep = {}
    for lv in ("ln", "coarse"):
        for state in STATES:
            f = fs[(fs.level == lv) & (fs.state == state)].nsmallest(10, "rank_median")
            keep[f"{lv}_{state}"] = f[["family", "rank_median", "rank_min", "rank_max", "n_variants_top10",
                                       "n_variants_rank1", "best_ratio_min", "best_ratio_max", "n_members",
                                       "n_within_2pct_max"]].round(4).to_dict("records")
    out["family_rank_stability_top10"] = keep
    out["rank_correlation"] = rc
    out["d_status"] = dst
    out["band"] = band
    t1 = ts[(ts.tier == 1) & (ts["T"] == DISPLAY_T)]
    out["tier_sizes_display_T"] = ts[ts["T"] == DISPLAY_T][["scenario", "rule", "bundle", "tier", "n", "n_icsd",
                                                             "n_oxide", "rho_min", "rho_max", "q_min", "q_max",
                                                             "U_star"]].round(4).to_dict("records")
    mem = tm[(tm.rule == common.PRIMARY_RULE) & (tm.tier == 1) & (tm["T"] == DISPLAY_T)]
    out["tier1_members_R2"] = {f"{sc}_{b}": g[["formula_pretty", "material_id", "density", "energy_above_hull",
                                                "icsd_matched", "band_gap", "min_crustal_abundance_ppm",
                                                "bootstrap_class"]].round(4).to_dict("records")
                               for (sc, b), g in mem.groupby(["scenario", "bundle"])}
    out["T_dependence"] = {"n_groups": int(len(td)), "n_identical": int(td.identical_across_T.sum()),
                           "not_identical": td[~td.identical_across_T].round(4).to_dict("records")}
    out["rule_sensitivity"] = rs.round(4).to_dict("records")
    out["band_sensitivity"] = pd.read_csv(OUT / "tier_band_sensitivity.csv").round(4).to_dict("records")
    out["areal_view"] = av.round(4).to_dict("records")
    out["bootstrap_crossref"] = bs.round(4).to_dict("records") if len(bs) else "P outputs absent"
    out["b5"] = {"n_rows": int(len(b5)),
                 "tier1_any": b5[(b5.tier_CS137_R2 == 1) | (b5.tier_CO60_R2 == 1)][
                     ["formula_pretty", "material_id", "density", "band_gap", "oxide", "tier_CS137_R2",
                      "tier_CO60_R2"]].round(4).to_dict("records"),
                 "phase2_consensus": b5[b5.phase2_b5_consensus][
                     ["formula_pretty", "material_id", "density", "band_gap", "tier_CS137_R1", "tier_CS137_R2",
                      "tier_CS137_R3", "tier_CO60_R1", "tier_CO60_R2", "tier_CO60_R3"]].round(4).to_dict("records")}
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if "--corrected" in sys.argv:
        compute_corrected()
    else:
        s = run()
        print(json.dumps({k: s[k] for k in ("checks", "n_families", "lanthanide_advantage_over_best_textbook")},
                         indent=1, default=str))
