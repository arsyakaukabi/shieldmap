"""Prices as a continuum.

1. Monoenergetic price curves on every grid node from 20 keV to 10 MeV (511 nodes), for every stage 2
   single constraint (PB, RAD, STAB(τ), EXP, GAP(g), ABUND(a)) and bundle B1–B6. Axes: both bases,
   rules R1/R2/R3, six frontier kinds (best, 10th, median50, best_icsd, best_clean, median50_clean),
   with/without coherent. Headline: R2, with coherent, kind best_clean in every cell; the
   superseded per-cell rule is carried as price_legacy_rule.
2. Tube-quality curves: DIAG profile prices vs RQR 5/6/8/9 × anode angle × Pb target, areal mass,
   from the frontiers-module table data/phase3/S4/frontier_prices.parquet (recomputed when missing).
3. Thickness–mass inversion of GAP: P_thickness(E), P_mass(E), the ratio (1+P_t)/(1+P_m), the
   densities of the frontier compounds and the exact identity
   1 + P_t = [q_a(c_K)/q_a(c_0)] · [ρ(c_0)/ρ(c_K)]   (best kind; c_0, c_K the thickness-frontier compounds).
4. Why PB = 0: the ambient fcc Pb entry (mp-20483) against the best non-Pb compound of F0, F0∩RAD,
   F0∩EXP and F0∩B4 at every energy; the minimum ratio and the compounds that beat Pb there.
5. Shapley attribution of log q*(F0∩PB∩RAD∩STAB(0.05)∩EXP) − log q*(F0) over {PB, RAD, STAB(0.05),
   EXP} vs energy, with the pairwise Shapley interaction index of STAB×EXP (negative = substitutes).
6. Element-exclusion filter (EXCL-X) prices vs energy for all 84 elements, against F0 and F0∩B4.
7. (Superseded in the hardening revision by analysis3.density_model, the functions below are kept for the
   record but run() no longer calls them and removes their old outputs.)

Monoenergetic requirement: thickness q = 1/μ (cm), areal q = 1/(μ/ρ) (g/cm²). For a single line the
Table C requirement is ln(1/T)·q, so prices and ranks are exact and T-independent.

Efficiency: per energy chunk and rule, the baseline set's best TOPK rows are sorted once (TopK); every
frontier of a subset (constraint, bundle, Shapley coalition, element-exclusion filter) is read off by
walking that list and skipping non-members. Columns where the list holds too few members fall back to
the full computation, so results equal core.frontier exactly (checked in tests and against stage 2).

Density convention (item 7): δ = ρ_MP/ρ_true − 1, so the true thickness requirement at fixed μ/ρ is
q_true = q_MP·(1+δ) — the convention of the frontiers module's tiers (q(1+lo), q(1+hi)). Areal mass is exactly
density-invariant and is not bootstrapped.

Outputs: data/phase3/P/ (see run()); figures reports/phase3/figures/{fig2_price_curves, fig3_gap_inversion,
fig4_load_bearing, P_*}. Run: python -m shieldmap.analysis3.continuum
"""
from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import core
from shieldmap.analysis import prices as p2prices
from shieldmap.analysis3 import common, frontiers

OUT = common.P3 / "P"
E_LO, E_HI = 20.0, 10000.0
TOPK = 4000                      # rows kept per energy in the sorted baseline list
CHUNK = 64                       # energies per chunk (memory: N × CHUNK float64 ≈ 79 MB)
KINDS3 = ("best", "10th", "median50")
KINDS_SH = KINDS3 + ("best_clean",)          # Shapley and element-exclusion filter kinds
HEADLINE = common.HEADLINE_KIND              # best_clean
CREDIBLE_KINDS = ("best_clean", "best_icsd", "median50_clean")   # kinds without artefact-dominated frontiers
KEY_E = p2prices.KEY_E
PLAYERS = [p2prices.player_label(p) for p in p2prices.PLAYERS]          # PB, RAD, STAB(0.05), EXP
HEAD_CONSTRAINTS = ["PB", "RAD", "STAB(0.05)", "EXP", "GAP(3.1)", "ABUND(1)", "B1", "B2", "B3", "B4", "B5", "B6"]
PB_SETS = ("F0", "F0&RAD", "F0&EXP", "F0&B4")
N_DRAWS = 400
SEED = 20261012
MIXED_RHO = 0.5                  # within-class correlation of the "mixed" bootstrap variant
BOOT_MODES = ("iid", "mixed", "class_shift")
BOOT_BUNDLES = ("B0", "B4", "B5", "B6")


# ---------------------------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------------------------
def grid_indices(lo: float = E_LO, hi: float = E_HI) -> np.ndarray:
    """Grid-node indices with lo ≤ E ≤ hi (511 nodes for 20 keV–10 MeV)."""
    e = common.energy_keV()
    return np.flatnonzero((e >= lo * (1 - 1e-9)) & (e <= hi * (1 + 1e-9)))


# ---------------------------------------------------------------------------------------------
# sorted top-K walk (exact frontiers of subsets of a baseline set)
# ---------------------------------------------------------------------------------------------
def _sorted_top(sub: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(values, positions) of the n smallest entries of each column of sub, ascending, ties by position."""
    sub = np.where(np.isfinite(sub), sub, np.inf)
    m = sub.shape[0]
    n = min(n, m)
    if n == 0:
        return np.empty((0, sub.shape[1])), np.empty((0, sub.shape[1]), dtype=np.int64)
    if m > n:
        part = np.argpartition(sub, n - 1, axis=0)[:n]
        part.sort(axis=0)                                   # positions ascending → stable sort breaks ties by row
    else:
        part = np.repeat(np.arange(m)[:, None], sub.shape[1], axis=1)
    vals = np.take_along_axis(sub, part, axis=0)
    order = np.argsort(vals, axis=0, kind="stable")
    return np.take_along_axis(vals, order, axis=0), np.take_along_axis(part, order, axis=0)


def frontier_kinds_full(q: np.ndarray, member: np.ndarray, cols=None) -> dict:
    """Brute-force best / 10th / median50 values and best row of `member` (over all rows of q)."""
    idx = np.flatnonzero(member)
    c = np.arange(q.shape[1]) if cols is None else np.asarray(cols)
    out = {k: np.full(c.size, np.nan) for k in KINDS3}
    out["best_row"] = np.full(c.size, -1, dtype=np.int64)
    if idx.size == 0 or c.size == 0:
        return out
    vals, pos = _sorted_top(q[np.ix_(idx, c)], 50)
    k = vals.shape[0]
    if k >= 1:
        out["best"] = vals[0]; out["best_row"] = idx[pos[0]]
    if k >= 10:
        out["10th"] = vals[9]
    if k >= 50:
        out["median50"] = 0.5 * (vals[24] + vals[25])
    for kk in KINDS3:
        out[kk] = np.where(np.isfinite(out[kk]), out[kk], np.nan)
    out["best_row"] = np.where(np.isfinite(out["best"]), out["best_row"], -1)
    return out


class TopK:
    """Best `k` rows of a baseline set per column of q, sorted ascending (ties by row index).

    frontiers(member) returns the exact frontier of baseline ∩ member: it walks the sorted list,
    skipping non-members, and falls back to frontier_kinds_full for columns where the list holds
    fewer members than a kind needs (unless the list already holds the whole baseline set).
    """

    def __init__(self, q: np.ndarray, base: np.ndarray, k: int = TOPK):
        self.q = q
        self.base = np.asarray(base, bool)
        idx = np.flatnonzero(self.base)
        vals, pos = _sorted_top(q[idx], k)
        self.vals = vals
        self.rows = idx[pos]
        self.k = vals.shape[0]
        self.complete = idx.size <= k
        self.n_fallback = 0

    def frontiers(self, member: np.ndarray | None = None, kinds=KINDS3, icsd: np.ndarray | None = None) -> dict:
        """dict with the requested kinds, best_row, and (if icsd given) best_icsd / best_icsd_row."""
        out = self._walk(member, kinds)
        if icsd is not None:
            mi = icsd if member is None else (np.asarray(member, bool) & icsd)
            o2 = self._walk(mi, ("best",))
            out["best_icsd"] = o2["best"]; out["best_icsd_row"] = o2["best_row"]
        return out

    def _walk(self, member, kinds) -> dict:
        C = self.q.shape[1]
        cols = np.arange(C)
        if member is None:
            ok = np.isfinite(self.vals)
        else:
            ok = np.asarray(member, bool)[self.rows] & np.isfinite(self.vals)
        cs = np.cumsum(ok, axis=0, dtype=np.int32)
        cnt = cs[-1] if self.k else np.zeros(C, np.int32)

        def pick(j):
            pos = np.argmax(cs >= j, axis=0)
            have = cnt >= j
            return np.where(have, self.vals[pos, cols], np.nan), np.where(have, self.rows[pos, cols], -1), have

        out = {}
        need = max({"best": 1, "10th": 10, "median50": 50}[k] for k in kinds)
        v1, r1, h1 = pick(1)
        out["best"] = v1; out["best_row"] = r1
        if "10th" in kinds:
            out["10th"] = pick(10)[0]
        if "median50" in kinds:
            a, _, _ = pick(25); b, _, _ = pick(26)
            out["median50"] = np.where(cnt >= 50, 0.5 * (a + b), np.nan)
        lacking = np.flatnonzero(cnt < need)
        if lacking.size and not self.complete:
            mem = self.base if member is None else (self.base & np.asarray(member, bool))
            full = frontier_kinds_full(self.q, mem, lacking)
            for kk in list(kinds) + ["best_row"]:
                out[kk] = out[kk].copy()
                out[kk][lacking] = full[kk]
            self.n_fallback += lacking.size
        return {k: out[k] for k in list(kinds) + ["best_row"]}

    def members(self, member: np.ndarray | None, n: int) -> np.ndarray:
        """Rows of the n best members (n × C, −1 padded), exact (full sort where the list is short)."""
        C = self.q.shape[1]
        ok = np.isfinite(self.vals) if member is None else (np.asarray(member, bool)[self.rows] & np.isfinite(self.vals))
        out = np.full((n, C), -1, dtype=np.int64)
        mem = self.base if member is None else (self.base & np.asarray(member, bool))
        idx = None
        for c in range(C):
            r = self.rows[ok[:, c], c][:n]
            if r.size < n and not self.complete:
                if idx is None:
                    idx = np.flatnonzero(mem)
                v, p = _sorted_top(self.q[idx, c][:, None], n)
                r = idx[p[np.isfinite(v[:, 0]), 0]]
            out[:r.size, c] = r
        return out


# ---------------------------------------------------------------------------------------------
# Shapley on a vector-valued game
# ---------------------------------------------------------------------------------------------
def shapley_vec(players: list[str], v: dict) -> dict[str, np.ndarray]:
    """Exact Shapley values; v maps frozenset(coalition) → value array (v[∅] subtracted implicitly)."""
    return core.shapley(players, lambda s: np.asarray(v[frozenset(s)], float) - np.asarray(v[frozenset()], float))


def shapley_interaction(players: list[str], v: dict, i: str, j: str) -> np.ndarray:
    """Pairwise Shapley interaction index I_ij = Σ_T |T|!(n−|T|−2)!/(n−1)! Δ_ij v(T), T ⊆ N∖{i,j}.

    Δ_ij v(T) = v(T∪{i,j}) − v(T∪{i}) − v(T∪{j}) + v(T). Negative: i and j are substitutes
    (together they add less than the sum of what each adds); positive: complements.
    """
    n = len(players)
    others = [p for p in players if p not in (i, j)]
    tot = 0.0
    for r in range(len(others) + 1):
        w = math.factorial(r) * math.factorial(n - r - 2) / math.factorial(n - 1)
        for t in itertools.combinations(others, r):
            T = frozenset(t)
            tot = tot + w * (np.asarray(v[T | {i, j}]) - np.asarray(v[T | {i}]) - np.asarray(v[T | {j}])
                             + np.asarray(v[T]))
    return tot


# ---------------------------------------------------------------------------------------------
# items 1, 3, 4, 5, 6: one pass over the energy grid
# ---------------------------------------------------------------------------------------------
def _player_masks() -> dict[str, np.ndarray]:
    return {p2prices.player_label(p): common.constraint_mask(*p) for p in p2prices.PLAYERS}


def _subset_masks(pm: dict) -> dict[frozenset, np.ndarray]:
    n = len(next(iter(pm.values())))
    out = {}
    for r in range(len(pm) + 1):
        for s in itertools.combinations(list(pm), r):
            m = np.ones(n, bool)
            for k in s:
                m &= pm[k]
            out[frozenset(s)] = m
    return out


def _with_clean(tk: TopK, member, clean: np.ndarray, icsd: np.ndarray) -> dict:
    """All six frontier kinds of baseline ∩ member (best_clean/median50_clean: artefact entries removed)."""
    out = dict(tk.frontiers(member, KINDS3, icsd))
    mc = clean if member is None else (np.asarray(member, bool) & clean)
    oc = tk.frontiers(mc, ("best", "median50"))
    out["best_clean"], out["median50_clean"], out["best_clean_row"] = oc["best"], oc["median50"], oc["best_row"]
    return out


def compute_grid(topk: int = TOPK, chunk: int = CHUNK, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Single pass over (coherent, basis, energy chunk, rule) computing items 1, 3, 4, 5, 6."""
    e_all = common.energy_keV()
    eidx = grid_indices()
    sets = frontiers.constraint_sets()
    pm = _player_masks()
    subs = _subset_masks(pm)
    icsd = common.icsd_mask()
    clean = common.clean_mask()
    b4 = common.bundle_mask("B4")
    nopb = common.constraint_mask("PB")
    rad_ok = common.constraint_mask("RAD")
    els = core.element_list()
    excl = {x: common.constraint_mask("EXCL", x) for x in els}
    gap31 = common.constraint_mask("GAP", 3.1)
    rho = common.density()
    pb_row = frontiers.textbook_rows()["Pb"]
    pbset_mask = {"F0": None, "F0&RAD": rad_ok, "F0&EXP": icsd, "F0&B4": b4}

    price_recs, shap_recs, excl_recs, pb_recs, gap_recs = [], [], [], [], []
    n_fb = 0
    t0 = time.time()
    for coh in (True, False):
        for basis in core.BASES:
            for c0 in range(0, eidx.size, chunk):
                cols = eidx[c0:c0 + chunk]
                E = e_all[cols]
                q = common.mono_requirement(basis, coh, e_idx=cols)
                q = np.where(np.isfinite(q), q, np.inf)
                for rule in core.RULES:
                    base = common.base_mask(rule)
                    tk = TopK(q, base, topk)
                    f0 = _with_clean(tk, None, clean, icsd)
                    # item 1: constraints and bundles
                    for fam, lab, km in sets:
                        fk = _with_clean(tk, km, clean, icsd)
                        for kind in common.KINDS6:
                            price_recs.append(pd.DataFrame({
                                "energy_keV": E, "e_idx": cols, "basis": basis, "rule": rule, "coherent": coh,
                                "family": fam, "constraint": lab, "kind": kind,
                                "price": fk[kind] / f0[kind] - 1.0, "q_base": f0[kind], "q_constrained": fk[kind],
                                "best_row_base": f0["best_row"], "best_row_constrained": fk["best_row"],
                                "icsd_row_base": f0["best_icsd_row"], "icsd_row_constrained": fk["best_icsd_row"],
                                "clean_row_base": f0["best_clean_row"], "clean_row_constrained": fk["best_clean_row"]}))
                    # item 5: Shapley coalitions (log q*)
                    sub_f = {s: dict(tk.frontiers(m, KINDS3) if s else f0) for s, m in subs.items()}
                    for s_, m in subs.items():
                        sub_f[s_]["best_clean"] = (f0["best_clean"] if not s_ else
                                                   tk.frontiers(m & clean, ("best",))["best"])
                    for kind in KINDS_SH:
                        vals = {s: np.log(f[kind]) for s, f in sub_f.items()}
                        phi = shapley_vec(PLAYERS, vals)
                        tot = vals[frozenset(PLAYERS)] - vals[frozenset()]
                        inter = shapley_interaction(PLAYERS, vals, "STAB(0.05)", "EXP")
                        inter_empty = (vals[frozenset({"STAB(0.05)", "EXP"})] - vals[frozenset({"STAB(0.05)"})]
                                       - vals[frozenset({"EXP"})] + vals[frozenset()])
                        T = frozenset({"PB", "RAD"})
                        inter_pbrad = (vals[T | {"STAB(0.05)", "EXP"}] - vals[T | {"STAB(0.05)"}]
                                       - vals[T | {"EXP"}] + vals[T])
                        d = {"energy_keV": E, "basis": basis, "rule": rule, "coherent": coh, "kind": kind,
                             "total": tot, "I_STAB_EXP": inter, "I_STAB_EXP_given_empty": inter_empty,
                             "I_STAB_EXP_given_PB_RAD": inter_pbrad,
                             "v_STAB": vals[frozenset({"STAB(0.05)"})] - vals[frozenset()],
                             "v_EXP": vals[frozenset({"EXP"})] - vals[frozenset()],
                             "v_STAB_EXP": vals[frozenset({"STAB(0.05)", "EXP"})] - vals[frozenset()]}
                        for p in PLAYERS:
                            d[f"phi_{p}"] = phi[p]
                        shap_recs.append(pd.DataFrame(d))
                    if not coh:
                        n_fb += tk.n_fallback
                        continue
                    # item 6: element-exclusion filter vs F0 and vs F0 ∩ B4
                    tb = TopK(q, base & b4, topk)
                    fb0 = tb.frontiers(None, KINDS3)
                    fb0["best_clean"] = tb.frontiers(clean, ("best",))["best"]
                    for bl, tkk, fz in (("F0", tk, f0), ("F0&B4", tb, fb0)):
                        for x in els:
                            fx = tkk.frontiers(excl[x], KINDS3)
                            fx["best_clean"] = tkk.frontiers(excl[x] & clean, ("best",))["best"]
                            for kind in KINDS_SH:
                                excl_recs.append(pd.DataFrame({
                                    "energy_keV": E, "basis": basis, "rule": rule, "baseline": bl, "element": x,
                                    "kind": kind, "price": (fx[kind] / fz[kind] - 1.0).astype(np.float32),
                                    "best_row_base": fz["best_row"].astype(np.int32),
                                    "best_row_constrained": fx["best_row"].astype(np.int32)}))
                    n_fb += tk.n_fallback + tb.n_fallback
                    if rule != core.PRIMARY_RULE:
                        continue
                    # item 4: Pb anatomy
                    qpb = q[pb_row]
                    for sname, sm in pbset_mask.items():
                        smask = base if sm is None else base & sm
                        fn = tk.frontiers(nopb if sm is None else nopb & sm, ("best", "median50"))
                        fp = tk.frontiers(~nopb if sm is None else (~nopb) & sm, ("best",))
                        cand = np.flatnonzero(smask & nopb)
                        n_beat = (q[cand] < qpb[None, :]).sum(axis=0)
                        allr = np.flatnonzero(base)
                        rank_pb = 1 + (q[allr] < qpb[None, :]).sum(axis=0)
                        pb_recs.append(pd.DataFrame({
                            "energy_keV": E, "basis": basis, "rule": rule, "set": sname, "q_Pb": qpb,
                            "Pb_in_base": bool(base[pb_row]),
                            "q_best_nonPb": fn["best"], "best_nonPb_row": fn["best_row"],
                            "q_median50_nonPb": fn["median50"],
                            "q_best_Pb_containing": fp["best"], "best_Pb_containing_row": fp["best_row"],
                            "ratio_best_nonPb_to_Pb": fn["best"] / qpb,
                            "ratio_median50_nonPb_to_Pb": fn["median50"] / qpb,
                            "ratio_best_Pbcont_to_best_nonPb": fp["best"] / fn["best"],
                            "n_nonPb_beating_Pb": n_beat, "rank_Pb_in_F0": rank_pb}))
                    # item 3: densities of the GAP(3.1) and F0 top-50 members
                    for lab, m in (("F0", None), ("GAP(3.1)", gap31), ("F0_clean", clean),
                                   ("GAP(3.1)_clean", gap31 & clean)):
                        mem = tk.members(m, 50)
                        dens = np.where(mem >= 0, rho[np.maximum(mem, 0)], np.nan)
                        gap_recs.append(pd.DataFrame({
                            "energy_keV": E, "basis": basis, "set": lab,
                            "rho_best": dens[0], "rho_top50_median": np.nanmedian(dens, axis=0),
                            "rho_top50_p10": np.nanquantile(dens, 0.1, axis=0),
                            "rho_top50_p90": np.nanquantile(dens, 0.9, axis=0)}))
                    n_fb += 0
                if verbose:
                    print(f"  [P] coh={coh} {basis} E {E[0]:.0f}–{E[-1]:.0f} keV  {time.time() - t0:.0f} s", flush=True)
                del q
    pc = pd.concat(price_recs, ignore_index=True)
    pc = _add_flags(pc)
    return {"price_curves": pc,
            "shapley": pd.concat(shap_recs, ignore_index=True),
            "excl": _excl_flags(pd.concat(excl_recs, ignore_index=True)),
            "pb": pd.concat(pb_recs, ignore_index=True),
            "gap_density": pd.concat(gap_recs, ignore_index=True),
            "n_fallback_columns": n_fb}


def _flag(rows: np.ndarray) -> np.ndarray:
    r = np.asarray(rows)
    out = np.zeros(r.size, bool)
    ok = r >= 0
    out[ok] = common.artefact_flag(r[ok])
    return out


def _add_flags(pc: pd.DataFrame) -> pd.DataFrame:
    pc["artefact_base"] = _flag(pc.best_row_base.to_numpy())
    pc["artefact_constrained"] = _flag(pc.best_row_constrained.to_numpy())
    pc["artefact_any"] = pc.artefact_base | pc.artefact_constrained
    return pc


def _excl_flags(ex: pd.DataFrame) -> pd.DataFrame:
    fl = _flag(ex.best_row_base.to_numpy()) | _flag(ex.best_row_constrained.to_numpy())
    ex["artefact_any"] = fl
    return ex


# ---------------------------------------------------------------------------------------------
# headline curves and bands (item 1)
# ---------------------------------------------------------------------------------------------
def headline_select(price: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per group `keys` (must include kind-free identifiers): the headline price (kind best_clean,
    ), every other kind side by side, and the superseded per-cell rule (median50 where the
    best-kind price is flagged, else best) as price_legacy_rule."""
    b = price[price.kind == "best"].set_index(keys)
    out = pd.DataFrame(index=b.index)
    out["headline_kind"] = HEADLINE
    for kd in sorted(set(price.kind.unique())):
        out[f"price_{kd}"] = price[price.kind == kd].set_index(keys).price.reindex(b.index)
    out["price_headline"] = out[f"price_{HEADLINE}"]
    out["artefact_any"] = b.artefact_any
    out["legacy_rule_kind"] = np.where(b.artefact_any, "median50", "best")
    out["price_legacy_rule"] = np.where(b.artefact_any, out.price_median50, out.price_best)
    return out.reset_index()


def headline_curves(pc: pd.DataFrame) -> pd.DataFrame:
    """R2/with-coherent headline price per energy × basis × constraint with the bands."""
    keys = ["energy_keV", "basis", "constraint"]
    u = common.universe()
    hk = headline_select(pc, keys + ["rule", "coherent"])
    h = hk[(hk.rule == core.PRIMARY_RULE) & hk.coherent].drop(columns=["rule", "coherent"]).set_index(keys)
    g = pc.groupby(keys).price
    h["band_all_lo"] = g.min(); h["band_all_hi"] = g.max()
    gcr = pc[pc.kind.isin(CREDIBLE_KINDS)].groupby(keys).price
    h["band_credible_lo"] = gcr.min(); h["band_credible_hi"] = gcr.max()
    r2c = pc[(pc.rule == core.PRIMARY_RULE) & pc.coherent].groupby(keys).price
    h["band_kinds_lo"] = r2c.min(); h["band_kinds_hi"] = r2c.max()
    # rules alone and coherent alone: at the R2/with-coherent headline kind of that cell
    sel = pc.merge(h[["headline_kind"]].reset_index(), on=keys)
    sel = sel[sel.kind == sel.headline_kind]
    gr = sel[sel.coherent].groupby(keys).price
    h["band_rules_lo"] = gr.min(); h["band_rules_hi"] = gr.max()
    gc = sel[sel.rule == core.PRIMARY_RULE].groupby(keys).price
    h["band_coherent_lo"] = gc.min(); h["band_coherent_hi"] = gc.max()
    bb = pc[(pc.kind == "best") & (pc.rule == core.PRIMARY_RULE) & pc.coherent].set_index(keys)
    for side in ("base", "constrained"):
        for tag, col in (("best", f"best_row_{side}"), ("clean", f"clean_row_{side}")):
            r = bb[col].reindex(h.index).to_numpy()
            ok = r >= 0
            f = np.full(r.size, None, dtype=object); mid = f.copy()
            f[ok] = u.formula_pretty.to_numpy()[r[ok]]; mid[ok] = u.material_id.to_numpy()[r[ok]]
            h[f"{tag}_{side}_formula"] = f; h[f"{tag}_{side}_material_id"] = mid
    return h.reset_index()


# ---------------------------------------------------------------------------------------------
# item 2: tube-quality curves
# ---------------------------------------------------------------------------------------------
def tube_prices() -> pd.DataFrame:
    """DIAG areal-mass prices for every RQR × anode angle × Pb target × response (rule R2 and bands)."""
    f = common.P3 / "S4" / "frontier_prices.parquet"
    fp = pd.read_parquet(f) if f.exists() else frontiers.frontier_prices()
    fp = fp[(fp.scenario == "DIAG") & (fp.basis == "areal")]
    v = common.variants().set_index("variant_id")[["quality", "anode_angle"]]
    keys = ["variant_id", "constraint", "rule"]
    hk = headline_select(fp, keys)
    meta = fp[fp.kind == "best"][keys + ["response", "coherent", "target_value", "spectrum"]]
    hk = hk.merge(meta, on=keys).join(v, on="variant_id")
    band = fp.groupby(["variant_id", "constraint"]).price.agg(band_rules_kinds_lo="min", band_rules_kinds_hi="max")
    hk = hk.join(band, on=["variant_id", "constraint"])
    return hk.sort_values(["constraint", "rule", "response", "coherent", "quality", "anode_angle", "target_value"])


# ---------------------------------------------------------------------------------------------
# item 3: GAP inversion table
# ---------------------------------------------------------------------------------------------
def gap_inversion(pc: pd.DataFrame, head: pd.DataFrame, dens: pd.DataFrame) -> pd.DataFrame:
    """P_t, P_m, the inversion factor and its decomposition, per energy and g, all in the headline kind.

    With c₀, c_K the thickness-frontier compounds and m₀, m_K the mass-frontier compounds (headline kind
    best_clean, both bases), the definitions q_t = q_a/ρ give the identity
        1 + P_t = [ρ(c₀)/ρ(c_K)] · [q_a(c_K)/q_a(c₀)]                      (density factor × mass factor)
        (1 + P_t)/(1 + P_m) = density factor × [q_a(c_K)/q_a(c₀)] / [q_a(m_K)/q_a(m₀)]
    The second factor ("frontier mismatch") compares the mass requirement of the thickness frontier with
    that of the mass frontier. The identity follows from μ = ρ·(μ/ρ); it is an accounting, not a finding.
    """
    rho = common.density()
    u = common.universe()
    f = u.formula_pretty.to_numpy()
    mr = common.mu_rho(True)
    sel = pc[(pc.rule == core.PRIMARY_RULE) & pc.coherent]
    recs = []
    for g in ("GAP(2.5)", "GAP(3.1)", "GAP(3.5)", "B5"):
        hb = head[head.constraint == g].pivot(index="energy_keV", columns="basis",
                                              values=["price_headline", "price_best", "price_median50",
                                                      "price_median50_clean", "band_all_lo", "band_all_hi"])
        d = pd.DataFrame({"constraint": g, "energy_keV": hb.index.to_numpy()})
        for kd in (HEADLINE, "best"):
            tag = "clean" if kd == HEADLINE else "best"
            st = sel[(sel.constraint == g) & (sel.basis == "thickness") & (sel.kind == kd)].set_index("energy_keV")
            sa = sel[(sel.constraint == g) & (sel.basis == "areal") & (sel.kind == kd)].set_index("energy_keV")
            st = st.reindex(hb.index); sa = sa.reindex(hb.index)
            col = "clean_row" if kd == HEADLINE else "best_row"
            c0 = st[f"{col}_base"].to_numpy(int); cK = st[f"{col}_constrained"].to_numpy(int)
            m0 = sa[f"{col}_base"].to_numpy(int); mK = sa[f"{col}_constrained"].to_numpy(int)
            ei = st.e_idx.to_numpy(int)
            qa = lambda r: 1.0 / np.asarray(mr[r, ei], float)               # noqa: E731
            d[f"P_thickness_{tag}"] = st.price.to_numpy()
            d[f"P_mass_{tag}"] = sa.price.to_numpy()
            d[f"thk_F0_{tag}"] = f[c0]; d[f"thk_K_{tag}"] = f[cK]
            d[f"mass_F0_{tag}"] = f[m0]; d[f"mass_K_{tag}"] = f[mK]
            d[f"rho_thk_F0_{tag}"] = rho[c0]; d[f"rho_thk_K_{tag}"] = rho[cK]
            d[f"density_factor_{tag}"] = rho[c0] / rho[cK]
            d[f"mass_factor_{tag}"] = qa(cK) / qa(c0)
            d[f"mismatch_factor_{tag}"] = (qa(cK) / qa(c0)) / (qa(mK) / qa(m0))
            d[f"inversion_{tag}"] = (1 + d[f"P_thickness_{tag}"]) / (1 + d[f"P_mass_{tag}"])
            d[f"identity_residual_{tag}"] = (1 + d[f"P_thickness_{tag}"]) / (d[f"density_factor_{tag}"]
                                                                             * d[f"mass_factor_{tag}"]) - 1
            d[f"inversion_residual_{tag}"] = d[f"inversion_{tag}"] / (d[f"density_factor_{tag}"]
                                                                      * d[f"mismatch_factor_{tag}"]) - 1
        for kd in ("median50", "median50_clean"):
            d[f"P_thickness_{kd}"] = hb[(f"price_{kd}", "thickness")].to_numpy(float)
            d[f"P_mass_{kd}"] = hb[(f"price_{kd}", "areal")].to_numpy(float)
            d[f"inversion_{kd}"] = (1 + d[f"P_thickness_{kd}"]) / (1 + d[f"P_mass_{kd}"])
        d["P_thickness_headline"] = d.P_thickness_clean
        d["P_mass_headline"] = d.P_mass_clean
        d["ratio_1pPt_over_1pPm_headline"] = d.inversion_clean
        d["ratio_Pt_over_Pm_headline"] = d.P_thickness_clean / d.P_mass_clean.where(d.P_mass_clean > 0)
        for side in ("thickness", "areal"):
            nm = "P_thickness" if side == "thickness" else "P_mass"
            d[f"{nm}_band_lo"] = hb[("band_all_lo", side)].to_numpy(float)
            d[f"{nm}_band_hi"] = hb[("band_all_hi", side)].to_numpy(float)
        recs.append(d)
    out = pd.concat(recs, ignore_index=True)
    dd = dens[dens.basis == "thickness"].pivot(index="energy_keV", columns="set",
                                                 values=["rho_best", "rho_top50_median", "rho_top50_p10", "rho_top50_p90"])
    dd.columns = [f"{a}_{b}_thk" for a, b in dd.columns]
    da = dens[dens.basis == "areal"].pivot(index="energy_keV", columns="set", values=["rho_top50_median"])
    da.columns = [f"{a}_{b}_mass" for a, b in da.columns]
    # join on rounded energies: the density table may have been round-tripped through CSV
    dd.index = np.round(dd.index.to_numpy(float), 6)
    da.index = np.round(da.index.to_numpy(float), 6)
    out["_ek"] = np.round(out.energy_keV.to_numpy(float), 6)
    out = out.join(dd, on="_ek").join(da, on="_ek").drop(columns="_ek")
    if out.filter(like="rho_top50").isna().any().any():
        raise RuntimeError("gap_inversion: density table does not cover every energy")
    return out


# ---------------------------------------------------------------------------------------------
# item 4: Pb anatomy
# ---------------------------------------------------------------------------------------------
def pb_anatomy(pb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-energy table, closest-energy summary, beating compounds at the closest energy)."""
    u = common.universe()
    pb = pb.copy()
    for c in ("best_nonPb_row", "best_Pb_containing_row"):
        r = pb[c].to_numpy(); ok = r >= 0
        f = np.full(r.size, None, dtype=object); f[ok] = u.formula_pretty.to_numpy()[r[ok]]
        pb[c.replace("_row", "_formula")] = f
    pb["margin_best_nonPb"] = 1.0 - pb.ratio_best_nonPb_to_Pb
    # thickness basis: the MP fcc Pb entry has the DFT density 11.08 g/cm³; Table C's Pb-relative
    # quantities use 11.34 g/cm³. Both references are reported.
    rho_pb_mp = float(common.density()[frontiers.textbook_rows()["Pb"]])
    pb_tc = float(common.variants().pb_density_g_cm3.dropna().iloc[0])
    scale = np.where(pb.basis == "thickness", pb_tc / rho_pb_mp, 1.0)
    pb["ratio_best_nonPb_to_Pb_tableC_density"] = pb.ratio_best_nonPb_to_Pb * scale
    pb["rho_Pb_MP"] = rho_pb_mp
    pb["rho_Pb_tableC"] = pb_tc
    closest = []
    beaters = []
    rho = common.density()
    pb_row = frontiers.textbook_rows()["Pb"]
    for (basis, s), g in pb.groupby(["basis", "set"], sort=False):
        # Pb is closest to (or on) the frontier where q*(best non-Pb)/q(Pb) is LARGEST; the minimum of
        # the same ratio is where non-Pb compounds beat Pb by the widest margin.
        i = g.ratio_best_nonPb_to_Pb.idxmax()
        r = g.loc[i]
        j = g.ratio_best_nonPb_to_Pb.idxmin()
        on = g[g.ratio_best_nonPb_to_Pb > 1]
        closest.append({"basis": basis, "set": s, "energy_keV_closest": r.energy_keV,
                        "max_ratio_best_nonPb_to_Pb": r.ratio_best_nonPb_to_Pb,
                        "min_ratio_best_nonPb_to_Pb_tableC_density": g.ratio_best_nonPb_to_Pb_tableC_density.min(),
                        "best_nonPb_at_closest": r.best_nonPb_formula,
                        "n_nonPb_beating_Pb_at_closest": int(r.n_nonPb_beating_Pb),
                        "rank_Pb_in_F0_at_closest": int(r.rank_Pb_in_F0),
                        "min_ratio_best_nonPb_to_Pb": g.loc[j, "ratio_best_nonPb_to_Pb"],
                        "energy_keV_min_ratio": g.loc[j, "energy_keV"],
                        "best_nonPb_at_min_ratio": g.loc[j, "best_nonPb_formula"],
                        "min_n_beating": int(g.n_nonPb_beating_Pb.min()),
                        "max_ratio_median50_nonPb_to_Pb": g.ratio_median50_nonPb_to_Pb.max(),
                        "energy_keV_max_ratio_median50": g.loc[g.ratio_median50_nonPb_to_Pb.idxmax(), "energy_keV"],
                        "n_energies_Pb_beats_all_nonPb": int(len(on)),
                        "energies_keV_Pb_beats_all_nonPb": ";".join(f"{e:g}" for e in on.energy_keV),
                        "n_energies_Pbcontaining_beats_all_nonPb": int((g.ratio_best_Pbcont_to_best_nonPb < 1).sum()),
                        "n_energies": int(len(g))})
        # beating compounds at the closest energy (plus the 20 best non-Pb members if fewer beat Pb)
        e_idx = common.energy_index(r.energy_keV)
        q = common.mono_requirement(basis, True, e_idx=[e_idx])[:, 0]
        base = common.base_mask(core.PRIMARY_RULE)
        sm = {"F0": base, "F0&RAD": base & common.constraint_mask("RAD"), "F0&EXP": base & common.icsd_mask(),
              "F0&B4": base & common.bundle_mask("B4")}[s] & common.constraint_mask("PB")
        cand = np.flatnonzero(sm)
        cand = cand[np.lexsort((cand, q[cand]))]
        n_show = max(int((q[cand] < q[pb_row]).sum()), 20)
        for k, rr in enumerate(cand[:n_show], 1):
            beaters.append({"basis": basis, "set": s, "energy_keV": r.energy_keV, "rank": k, "row": int(rr),
                            "material_id": u.material_id.iat[rr], "formula": u.formula_pretty.iat[rr],
                            "requirement": float(q[rr]), "ratio_to_Pb": float(q[rr] / q[pb_row]),
                            "beats_Pb": bool(q[rr] < q[pb_row]),
                            "density": float(rho[rr]), "e_hull": float(u.energy_above_hull.iat[rr]),
                            "theoretical": bool(u.theoretical.iat[rr]),
                            "artefact": bool(common.artefact_flag([rr])[0]),
                            "elements": "-".join(u.elements.iat[rr])})
    return pb, pd.DataFrame(closest), pd.DataFrame(beaters)


# ---------------------------------------------------------------------------------------------
# item 7: density-bias bootstrap
# ---------------------------------------------------------------------------------------------
def delta_distributions() -> tuple[dict[str, np.ndarray], str]:
    """Class → δ samples (δ = ρ_MP/ρ_true − 1). the COD module compound (COD) distribution when available,
    else the COD module's elemental one, else the stage 1 elemental distribution pooled for every class."""
    summ = common.P3 / "D" / "summary.json"
    errs = []
    if summ.exists():
        try:
            from shieldmap.analysis3 import cod
            for kind in ("compound", "elemental"):
                try:
                    d = cod.class_delta_distribution(kind=kind)
                    d = {str(k): np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for k, v in d.items()}
                    d = {k: v for k, v in d.items() if v.size}
                    if d:
                        try:
                            info = json.dumps(cod.class_delta_info(kind), default=str)
                        except Exception:                                   # info is optional
                            info = "n/a"
                        return (_fill_classes(d),
                                f"Step 3 cod.class_delta_distribution(kind='{kind}'); class info {info}"
                                + (f"; earlier attempts failed: {'; '.join(errs)}" if errs else ""))
                except Exception as exc:                                    # try the next kind
                    errs.append(f"{kind}: {exc!r}")
        except Exception as exc:
            errs.append(repr(exc))
    else:
        errs.append("data/phase3/D/summary.json missing")
    from shieldmap.analysis import sensitivity
    x = sensitivity.bias_distribution("empirical")
    return ({c: x for c in common.PRIMARY_CLASS_ORDER},
            f"Phase 1 elemental (pooled for every class; Step 3 unavailable: {'; '.join(errs)})")


def _fill_classes(d: dict) -> dict:
    pooled = np.concatenate(list(d.values()))
    return {c: d.get(c, pooled) for c in common.PRIMARY_CLASS_ORDER}


def draw_deltas(cls: np.ndarray, dist: dict, rng: np.random.Generator, mode: str = "iid",
                rho: float = MIXED_RHO) -> np.ndarray:
    """One bootstrap draw of δ for every row, by class.

    iid          δ_m drawn independently from the class distribution
    class_shift  one δ per class per draw, shared by every member (full within-class correlation)
    mixed        δ_m = μ_c + √ρ (s_c − μ_c) + √(1−ρ) (e_m − μ_c), s_c and e_m independent class draws:
                 same class mean and variance as iid, within-class correlation ρ
    """
    out = np.empty(cls.size)
    for c in np.unique(cls):
        ix = np.flatnonzero(cls == c)
        x = dist[c]
        if mode == "iid":
            out[ix] = rng.choice(x, ix.size, replace=True)
        elif mode == "class_shift":
            out[ix] = rng.choice(x)
        elif mode == "mixed":
            mu = x.mean()
            out[ix] = mu + math.sqrt(rho) * (rng.choice(x) - mu) + math.sqrt(1 - rho) * (rng.choice(x, ix.size) - mu)
        else:
            raise ValueError(mode)
    return out


def true_requirement(q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Thickness requirement implied by δ = ρ_MP/ρ_true − 1: q_true = q_MP (1 + δ)."""
    q = np.asarray(q, float)
    return q * (1.0 + delta)[:, None] if q.ndim == 2 else q * (1.0 + delta)


def _boot_columns() -> tuple[np.ndarray, pd.DataFrame]:
    """Thickness requirement over R2-base rows: Cs-137/Co-60 primary profile variants + key-energy lines."""
    base = common.base_mask(core.PRIMARY_RULE)
    rows = np.flatnonzero(base)
    qp = common.profiles("thickness")
    meta, mats = [], []
    for sc in ("CS137", "CO60"):
        for vid in common.scenario_variants(sc).variant_id:
            mats.append(np.asarray(qp[rows, int(vid)], float))
            meta.append({"column": f"{sc}/v{int(vid)}", "source": sc, "variant_id": int(vid), "energy_keV": np.nan})
    ki = [common.energy_index(e) for e in KEY_E]
    qm = common.mono_requirement("thickness", True, rows=rows, e_idx=ki)
    for j, e in enumerate(KEY_E):
        mats.append(qm[:, j])
        meta.append({"column": f"mono/{e:g}", "source": "mono", "variant_id": -1, "energy_keV": e})
    return np.column_stack(mats), pd.DataFrame(meta)


def _kinds_from_sorted(vals: np.ndarray) -> dict:
    k = vals.shape[0]
    nan = np.full(vals.shape[1], np.nan)
    return {"best": vals[0] if k else nan, "10th": vals[9] if k >= 10 else nan,
            "median50": 0.5 * (vals[24] + vals[25]) if k >= 50 else nan}


def bootstrap(n_draws: int = N_DRAWS, modes=BOOT_MODES, seed: int = SEED, verbose: bool = True) -> dict:
    """Item 7: price bands and tier stability on the thickness basis under class-resolved δ draws."""
    dist, source = delta_distributions()
    bands, band_source = frontiers.class_band()
    base = common.base_mask(core.PRIMARY_RULE)
    rows = np.flatnonzero(base)
    cls = common.universe_primary_class()[rows]
    lo = np.array([bands[c][0] for c in cls]); hi = np.array([bands[c][1] for c in cls])
    q, meta = _boot_columns()
    cmasks = {lab: m[rows] for _, lab, m in frontiers.constraint_sets() if lab in HEAD_CONSTRAINTS}
    allm = np.ones(rows.size, bool)
    art = common.artefact_flag(rows)
    bmask = {b: common.bundle_mask(b)[rows] for b in BOOT_BUNDLES}
    tier_cols = [j for j, r in meta.iterrows() if r.source in ("CS137", "CO60")]

    def frontier_set(qq, m):
        idx = np.flatnonzero(m)
        v, p = _sorted_top(qq[idx], 50)
        k = _kinds_from_sorted(v)
        k["best_row"] = idx[p[0]]
        return k

    def prices_of(qq):
        f0 = frontier_set(qq, allm)
        res = {}
        for lab, m in cmasks.items():
            fk = frontier_set(qq, m)
            flag = art[f0["best_row"]] | art[fk["best_row"]]
            r = {kd: fk[kd] / f0[kd] - 1 for kd in KINDS3}
            r["headline"] = np.where(flag, r["median50"], r["best"])
            r["flag"] = flag
            res[lab] = r
        return res

    nominal = prices_of(q)
    nominal_t1 = {}
    for b, m in bmask.items():
        for j in tier_cols:
            nominal_t1[(b, j)] = frontiers.tiers(q[:, j], m, lo, hi, 1) == 1
    price_draws = {mode: {lab: {kd: np.empty((n_draws, q.shape[1])) for kd in ("best", "10th", "median50", "headline")}
                          for lab in cmasks} for mode in modes}
    tier_acc = {}
    t0 = time.time()
    for mi, mode in enumerate(modes):
        rng = np.random.default_rng(seed + mi)
        for d in range(n_draws):
            delta = draw_deltas(cls, dist, rng, mode)
            qd = true_requirement(q, delta)
            pr = prices_of(qd)
            for lab in cmasks:
                for kd in ("best", "10th", "median50", "headline"):
                    price_draws[mode][lab][kd][d] = pr[lab][kd]
            for b, m in bmask.items():
                for j in tier_cols:
                    t1 = frontiers.tiers(qd[:, j], m, lo, hi, 1) == 1
                    nom = nominal_t1[(b, j)]
                    true_best = np.flatnonzero(m)[np.argmin(qd[m, j])]
                    a = tier_acc.setdefault((mode, b, j), {"in_t1": np.zeros(rows.size), "same": 0, "jacc": [],
                                                           "cover": 0, "size": [], "true_best": []})
                    a["in_t1"] += t1
                    a["same"] += bool(np.array_equal(t1, nom))
                    a["jacc"].append((t1 & nom).sum() / max((t1 | nom).sum(), 1))
                    a["cover"] += bool(nom[true_best])
                    a["size"].append(int(t1.sum()))
                    a["true_best"].append(int(rows[true_best]))
            if verbose and (d + 1) % 100 == 0:
                print(f"  [P boot] {mode} {d + 1}/{n_draws}  {time.time() - t0:.0f} s", flush=True)
    # tables
    u = common.universe()
    precs = []
    for mode in modes:
        for lab in cmasks:
            for kd in ("best", "10th", "median50", "headline"):
                x = price_draws[mode][lab][kd]
                for j, r in meta.iterrows():
                    col = x[:, j]
                    precs.append({"mode": mode, "column": r.column, "source": r.source, "variant_id": r.variant_id,
                                  "energy_keV": r.energy_keV, "constraint": lab, "kind": kd,
                                  "nominal": float(nominal[lab][kd][j]),
                                  "nominal_flagged": bool(nominal[lab]["flag"][j]),
                                  "p05": float(np.nanquantile(col, 0.05)), "median": float(np.nanmedian(col)),
                                  "p95": float(np.nanquantile(col, 0.95)), "n_draws": int(np.isfinite(col).sum())})
    trecs, tsum = [], []
    for (mode, b, j), a in tier_acc.items():
        nom = nominal_t1[(b, j)]
        r = meta.loc[j]
        for i in np.flatnonzero(nom | (a["in_t1"] > 0)):
            gr = rows[i]
            trecs.append({"mode": mode, "scenario": r.source, "variant_id": r.variant_id, "bundle": b,
                          "row": int(gr), "material_id": u.material_id.iat[gr], "formula": u.formula_pretty.iat[gr],
                          "bootstrap_class": cls[i], "nominal_tier1": bool(nom[i]),
                          "frac_draws_in_tier1": a["in_t1"][i] / n_draws,
                          "frac_draws_true_best": a["true_best"].count(int(gr)) / n_draws})
        tsum.append({"mode": mode, "scenario": r.source, "variant_id": r.variant_id, "bundle": b,
                     "n_tier1_nominal": int(nom.sum()),
                     "frac_draws_tier1_set_unchanged": a["same"] / n_draws,
                     "frac_draws_tier1_set_changed": 1 - a["same"] / n_draws,
                     "mean_jaccard_vs_nominal": float(np.mean(a["jacc"])),
                     "min_jaccard_vs_nominal": float(np.min(a["jacc"])),
                     "tier1_size_median": float(np.median(a["size"])),
                     "tier1_size_p05": float(np.quantile(a["size"], 0.05)),
                     "tier1_size_p95": float(np.quantile(a["size"], 0.95)),
                     "coverage_true_best_in_nominal_tier1": a["cover"] / n_draws,
                     "n_distinct_true_best": len(set(a["true_best"])),
                     "min_member_stay_frac": float((a["in_t1"][nom] / n_draws).min()) if nom.any() else np.nan})
    meta_out = {"delta_source": source, "band_source": band_source, "n_draws": n_draws, "seed": seed,
                "modes": list(modes), "mixed_rho": MIXED_RHO,
                "class_sizes_R2": pd.Series(cls).value_counts().to_dict(),
                "class_delta_quantiles": {c: [float(np.quantile(dist[c], q_)) for q_ in (0.05, 0.5, 0.95)] + [int(dist[c].size)]
                                          for c in common.PRIMARY_CLASS_ORDER},
                "tier_bands": {c: list(map(float, v)) for c, v in bands.items()},
                "convention": "q_true = q_MP (1 + δ), δ = ρ_MP/ρ_true − 1 (Step 4 tiers convention)"}
    return {"prices": pd.DataFrame(precs), "tiers": pd.DataFrame(trecs), "tier_summary": pd.DataFrame(tsum),
            "meta": meta_out, "draws": price_draws, "columns": meta}


# ---------------------------------------------------------------------------------------------
# consistency with stage 2 and the three-scenario comparison
# ---------------------------------------------------------------------------------------------
def phase2_checks(pc: pd.DataFrame, ex: pd.DataFrame, sh: pd.DataFrame, tube: pd.DataFrame) -> dict:
    """Asserted equalities with stage 2 at the 14 key energies and on the DIAG profiles."""
    res = {}
    keys = ["energy_keV", "basis", "rule", "kind", "constraint"]
    p2 = pd.read_parquet(core.P2 / "A2" / "prices_mono.parquet")
    a = p2[(p2.baseline == "F0") & p2.family.isin(["single", "bundle"])]
    m = pc[pc.coherent & pc.kind.isin(KINDS3)].merge(a[keys + ["price"]], on=keys, suffixes=("", "_p2"))
    res["mono_singles_bundles_n"] = int(len(m))
    res["mono_singles_bundles_max_abs_diff"] = float(np.nanmax(np.abs(m.price - m.price_p2)))
    res["mono_singles_bundles_nan_mismatch"] = int((m.price.isna() != m.price_p2.isna()).sum())
    assert res["mono_singles_bundles_n"] == 14 * 20 * 2 * 3 * 3
    assert res["mono_singles_bundles_max_abs_diff"] < 1e-12 and res["mono_singles_bundles_nan_mismatch"] == 0
    e = p2[p2.family == "excl"].rename(columns={"param": "element"})
    k2 = ["energy_keV", "basis", "rule", "kind", "element", "baseline"]
    m = ex.merge(e[k2 + ["price"]], on=k2, suffixes=("", "_p2"))
    res["excl_n"] = int(len(m))
    res["excl_max_abs_diff_float32"] = float(np.nanmax(np.abs(m.price - m.price_p2)))
    assert res["excl_max_abs_diff_float32"] < 1e-6
    s2 = pd.read_parquet(core.P2 / "A2" / "shapley.parquet")
    s2 = s2[s2.source == "mono"]
    worst = 0.0
    for p in PLAYERS:
        mm = sh[sh.coherent].merge(s2[s2.player == p][["energy_keV", "basis", "rule", "kind", "phi"]],
                                   on=["energy_keV", "basis", "rule", "kind"])
        worst = max(worst, float(np.nanmax(np.abs(mm[f"phi_{p}"] - mm.phi))))
    res["shapley_max_abs_diff"] = worst
    assert worst < 1e-12
    pr = pd.read_parquet(core.P2 / "A2" / "prices.parquet", columns=["scenario", "variant_id", "basis", "rule", "kind",
                                                                     "constraint", "baseline", "family", "price"])
    d = pr[(pr.scenario == "DIAG") & (pr.basis == "areal") & (pr.baseline == "F0") & pr.family.isin(["single", "bundle"])]
    t = tube.merge(d[(d.kind == "best")][["variant_id", "rule", "constraint", "price"]],
                   on=["variant_id", "rule", "constraint"])
    res["diag_tube_best_vs_phase2_n"] = int(len(t))
    res["diag_tube_best_vs_phase2_max_abs_diff"] = float(np.nanmax(np.abs(t.price_best - t.price)))
    assert res["diag_tube_best_vs_phase2_max_abs_diff"] < 1e-12
    return res


def scenario_comparison(pc: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 three-scenario prices (R2, primary sweep) vs the continuum at 661.657 and 1250 keV."""
    pr = pd.read_parquet(core.P2 / "A2" / "prices.parquet", columns=["scenario", "variant_id", "primary", "basis", "rule",
                                                                     "kind", "constraint", "baseline", "family", "price"])
    pr = pr[pr.primary & (pr.rule == core.PRIMARY_RULE) & (pr.baseline == "F0") & pr.family.isin(["single", "bundle"])]
    recs = []
    for sc, E in (("CS137", 661.657), ("CO60", 1250.0)):
        for basis in core.BASES:
            for c in HEAD_CONSTRAINTS:
                s = pr[(pr.scenario == sc) & (pr.basis == basis) & (pr.constraint == c)]
                mono = pc[(pc.energy_keV == E) & (pc.basis == basis) & (pc.constraint == c) & pc.coherent
                          & (pc.rule == core.PRIMARY_RULE)].set_index("kind").price
                hh = head[(head.energy_keV == E) & (head.basis == basis) & (head.constraint == c)].iloc[0]
                rec = {"scenario": sc, "energy_keV": E, "basis": basis, "constraint": c}
                for kd in KINDS3:
                    x = s[s.kind == kd].price
                    rec[f"phase2_profile_{kd}_min"] = x.min(); rec[f"phase2_profile_{kd}_max"] = x.max()
                    rec[f"mono_{kd}"] = mono.get(kd, np.nan)
                rec["mono_headline"] = hh.price_headline; rec["mono_headline_kind"] = hh.headline_kind
                rec["mono_band_all_lo"] = hh.band_all_lo; rec["mono_band_all_hi"] = hh.band_all_hi
                rec["abs_diff_best_mono_vs_profile_max"] = (float(np.nanmax(np.abs(s[s.kind == "best"].price - rec["mono_best"])))
                                                            if len(s) else np.nan)
                recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
COL = {"thickness": "#2a78d6", "areal": "#eb6834"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASIS_LABEL = {"thickness": "thickness", "areal": "areal mass"}
TITLES = {"PB": "PB: lead-free", "RAD": "RAD: non-radioactive", "STAB(0.05)": "STAB(0.05 eV/atom)",
          "EXP": "EXP: ICSD-matched", "GAP(3.1)": "GAP(3.1 eV)", "ABUND(1)": "ABUND(1 ppm), scarcity proxy",
          "B1": "B1 = RAD", "B2": "B2 = B1 + PB", "B3": "B3 = B2 + STAB(0.05)", "B4": "B4 = B3 + EXP",
          "B5": "B5 = B4 + GAP(3.1)", "B6": "B6 = B4 + ABUND(1)"}
LINES = {"Cs-137": 661.657, "Co-60": 1250.0}
PB_K_KEV = 88.0045
XT = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
XTL = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k"]


def _logx(ax, lo=E_LO, hi=E_HI):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(XT)
    ax.set_xticklabels(XTL)
    ax.minorticks_off()


def _recess(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e4e3df", lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, color="#8a8984")


FIG2_PANELS = ["PB", "STAB(0.05)", "EXP", "RAD", "ABUND(1)", "B4", "B6", "GAP(3.1)", "B5"]
FIG2_GROUPS = [("PB", "STAB(0.05)", "EXP"), ("RAD", "ABUND(1)", "B4", "B6"), ("GAP(3.1)", "B5")]


def fig_price_curves(head: pd.DataFrame) -> list[str]:
    """fig2: headline (best_clean) price vs energy, both bases, 11 constraints, all-axes bands.

    B1 is RAD itself and B2 (= B1 + PB) equals B1 wherever PB is free (see the PB panel); B3 (= B2 + STAB)
    equals B4 wherever EXP is free (see the EXP panel). None of them has a panel. The band is the credible-kinds
    envelope (best_clean, best_icsd, median50_clean × rules × coherent). Panels of the same group share a
    y-scale so that their heights can be compared.
    """
    plt = common.style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(4, 3, figsize=(common.DOUBLE_COL_IN, 8.8), sharex=True)
    ylim = {}
    for grp in FIG2_GROUPS:
        h = head[head.constraint.isin(grp)]
        top = float(100 * np.nanmax(np.r_[h.band_credible_hi.to_numpy(float), h.price_headline.to_numpy(float)]))
        for c in grp:
            ylim[c] = (-0.03 * top, 1.05 * top)
    for ax, c in zip(axs.flat, FIG2_PANELS):
        _recess(ax)
        for E in LINES.values():
            ax.axvline(E, color="#b9b8b2", lw=0.5, ls=":", zorder=0)
        for basis in core.BASES:
            h = head[(head.constraint == c) & (head.basis == basis)].sort_values("energy_keV")
            ax.fill_between(h.energy_keV, 100 * h.band_credible_lo, 100 * h.band_credible_hi, color=COL[basis],
                            alpha=0.12 if basis == "thickness" else 0.22, lw=0)
            ax.plot(h.energy_keV, 100 * h.price_headline, color=COL[basis], lw=1.1)
        ax.set_title(TITLES[c] if c != "RAD" else "RAD (= B1): non-radioactive", loc="left", fontsize=7.5)
        ax.axhline(0, color="#52514e", lw=0.5)
        ax.set_ylim(*ylim[c])
        _logx(ax)
        for name, E in LINES.items():
            ax.text(E * 0.95, 0.97, name, transform=ax.get_xaxis_transform(), fontsize=6, color="#8a8984",
                    ha="right", va="top", rotation=90)
    for ax in axs.flat[len(FIG2_PANELS):]:
        ax.axis("off")
    ax = axs.flat[len(FIG2_PANELS)]
    hs = [Line2D([], [], color=COL["thickness"], lw=1.2), Line2D([], [], color=COL["areal"], lw=1.2),
          (Patch(color=COL["thickness"], alpha=0.12, lw=0), Patch(color=COL["areal"], alpha=0.22, lw=0))]
    ax.legend(hs, ["thickness basis, 1/μ (cm)", "areal-mass basis, 1/(μ/ρ) (g/cm²)",
                   "band: rules R1/R2/R3 × kinds without\nartefact frontiers (best_clean, best_icsd,\nmedian of top 50 clean) × coherent on/off"],
              loc="center left", frameon=False, fontsize=7,
              title="line: best non-artefact compound,\nrule R2, with coherent; narrow beam;\nMP densities (thickness: see dossier R2.4\nfor the measured-density point).\nB1 = RAD; B2 = B1 + PB and B3 = B2 + STAB\nnot shown (equal to RAD and B4 here)",
              title_fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("price (%)")
    for ax in (axs[2, 0], axs[2, 1], axs[2, 2]):
        ax.set_xlabel("photon energy (keV)")
        ax.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    out = common.savefig(fig, "fig2_price_curves")
    plt.close(fig)
    return out


def fig_price_params(head: pd.DataFrame) -> list[str]:
    """Parameter families STAB(τ), GAP(g), ABUND(a), headline kind, both bases."""
    plt = common.style()
    fams = {"STAB": ["STAB(0)", "STAB(0.025)", "STAB(0.05)", "STAB(0.1)"],
            "GAP": ["GAP(2.5)", "GAP(3.1)", "GAP(3.5)"],
            "ABUND": ["ABUND(0.01)", "ABUND(0.1)", "ABUND(1)", "ABUND(10)"]}
    shades = ["#9ec5f4", "#5b9be6", "#2a78d6", "#0f4c9a"]
    fig, axs = plt.subplots(2, 3, figsize=(common.DOUBLE_COL_IN, 4.2), sharex=True)
    for j, (fam, cs) in enumerate(fams.items()):
        for i, basis in enumerate(core.BASES):
            ax = axs[i, j]
            _recess(ax)
            for k, c in enumerate(cs):
                h = head[(head.constraint == c) & (head.basis == basis)].sort_values("energy_keV")
                ax.plot(h.energy_keV, 100 * h.price_headline, color=shades[k + (4 - len(cs))], lw=0.9, label=c)
            ax.axhline(0, color="#52514e", lw=0.5)
            _logx(ax)
            ax.set_title(f"{fam}, {BASIS_LABEL[basis]}", loc="left", fontsize=7.5)
            ax.legend(frameon=False, fontsize=5.5, loc="upper right")
            if j == 0:
                ax.set_ylabel("headline price (%)")
            if i == 1:
                ax.set_xlabel("photon energy (keV)")
    fig.tight_layout()
    out = common.savefig(fig, "P_price_curves_params")
    plt.close(fig)
    return out


def fig_band_drivers(head: pd.DataFrame) -> list[str]:
    """What drives the band width (B4), and how often the headline kind is median50."""
    plt = common.style()
    fig, axs = plt.subplots(1, 3, figsize=(common.DOUBLE_COL_IN, 2.4))
    srcs = [("all", "all axes", "#52514e"), ("kinds", "6 frontier kinds (R2, coh.)", CAT[0]),
            ("rules", "rules R1/R2/R3", CAT[1]), ("coherent", "coherent on/off", CAT[2])]
    for ax, basis in zip(axs[:2], core.BASES):
        _recess(ax)
        h = head[(head.constraint == "B4") & (head.basis == basis)].sort_values("energy_keV")
        for s, lab, col in srcs:
            ax.plot(h.energy_keV, 100 * (h[f"band_{s}_hi"] - h[f"band_{s}_lo"]), color=col, lw=0.9, label=lab)
        _logx(ax)
        ax.set_title(f"B4 band width, {BASIS_LABEL[basis]}", loc="left", fontsize=7.5)
        ax.set_xlabel("photon energy (keV)")
    axs[0].set_ylabel("band width (percentage points)")
    axs[0].legend(frameon=False, fontsize=5.5)
    ax = axs[2]
    _recess(ax)
    for basis in core.BASES:
        g = head[head.basis == basis].groupby("energy_keV").artefact_any.mean()
        ax.plot(g.index, g.values, color=COL[basis], lw=0.9, label=BASIS_LABEL[basis])
    _logx(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("share of the 20 constraints whose best-\ncompound frontier is an artefact entry", loc="left",
                 fontsize=7.5)
    ax.set_xlabel("photon energy (keV)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = common.savefig(fig, "P_band_drivers")
    plt.close(fig)
    return out


def fig_gap_inversion(gi: pd.DataFrame) -> list[str]:
    """fig3: thickness–mass inversion of the GAP(3.1) price and its density accounting (headline kind)."""
    plt = common.style()
    g = gi[gi.constraint == "GAP(3.1)"].sort_values("energy_keV")
    fig, axs = plt.subplots(2, 2, figsize=(common.DOUBLE_COL_IN, 5.0), sharex=True)
    ax = axs[0, 0]
    _recess(ax)
    for basis, col, lab in (("thickness", "P_thickness", "thickness"), ("areal", "P_mass", "areal mass")):
        ax.fill_between(g.energy_keV, 100 * g[f"{col}_band_lo"], 100 * g[f"{col}_band_hi"], color=COL[basis],
                        alpha=0.16, lw=0)
        ax.plot(g.energy_keV, 100 * g[f"{col}_headline"], color=COL[basis], lw=1.1, label=f"{lab}, GAP(3.1)")
        for other, ls in (("GAP(2.5)", (0, (1, 1))), ("GAP(3.5)", (0, (4, 1.5)))):
            o = gi[gi.constraint == other].sort_values("energy_keV")
            ax.plot(o.energy_keV, 100 * o[f"{col}_headline"], color=COL[basis], lw=0.5, ls=ls)
    ax.set_ylabel("price (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("a  GAP price (thin: g = 2.5 dotted, 3.5 dashed)", loc="left", fontsize=7.5)
    ax = axs[0, 1]
    _recess(ax)
    ax.plot(g.energy_keV, g.inversion_clean, color="#52514e", lw=1.1, label="best non-artefact compound")
    ax.plot(g.energy_keV, g.inversion_median50_clean, color="#52514e", lw=0.6, ls="--",
            label="median of top 50 (artefact-free)")
    ax.axhline(1, color="#8a8984", lw=0.5)
    ax.set_ylabel("(1 + P_thickness) / (1 + P_mass)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("b  inversion factor (same kind on both bases)", loc="left", fontsize=7.5)
    ax = axs[1, 0]
    _recess(ax)
    for sname, col, lab in (("F0_clean", "#52514e", "F0"), ("GAP(3.1)_clean", CAT[2], "F0 ∩ GAP(3.1)")):
        ax.fill_between(g.energy_keV, g[f"rho_top50_p10_{sname}_thk"], g[f"rho_top50_p90_{sname}_thk"], color=col,
                        alpha=0.15, lw=0)
        ax.plot(g.energy_keV, g[f"rho_top50_median_{sname}_thk"], color=col, lw=1.0, label=f"{lab}: top-50 median")
    ax.plot(g.energy_keV, g.rho_thk_F0_clean, color="#52514e", lw=0.6, ls="--", label="F0: best compound")
    ax.plot(g.energy_keV, g.rho_thk_K_clean, color=CAT[2], lw=0.6, ls="--",
            label="GAP(3.1): best compound (ThO₂ at high E: radioactive)")
    ax.plot(g.energy_keV, g.rho_best_F0_thk, color="#b9b8b2", lw=0.6, ls=":",
            label="F0 best incl. artefact entries")
    ax.set_ylabel("density (g/cm³)")
    ax.set_ylim(0, 33)                                       # headroom for the legend
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center")
    ax.set_title("c  thickness-frontier densities, artefact entries removed\n    (shade: top-50 p10–p90)",
                 loc="left", fontsize=7.5)
    ax.set_xlabel("photon energy (keV)")
    ax = axs[1, 1]
    _recess(ax)
    ax.plot(g.energy_keV, g.inversion_clean, color="#52514e", lw=1.1, label="inversion factor")
    ax.plot(g.energy_keV, g.density_factor_clean, color=CAT[2], lw=0.9, label="density factor ρ(c₀)/ρ(c_K)")
    ax.plot(g.energy_keV, g.mismatch_factor_clean, color=COL["areal"], lw=0.9,
            label="frontier-mismatch factor")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3, 4])
    ax.set_yticklabels(["0.5", "0.7", "1", "1.5", "2", "3", "4"])
    ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.axhline(1, color="#8a8984", lw=0.5)
    ax.set_title("d  accounting identity: inversion = density × mismatch", loc="left", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_xlabel("photon energy (keV)")
    for a in axs.flat:
        _logx(a)
    fig.tight_layout()
    out = common.savefig(fig, "fig3_gap_inversion")
    plt.close(fig)
    return out


def excl_headline(ex: pd.DataFrame, rule: str = core.PRIMARY_RULE) -> pd.DataFrame:
    """Headline element-exclusion filter price per element × energy × basis × baseline (the frontiers module kind rule)."""
    e = ex[ex.rule == rule]
    return headline_select(e, ["energy_keV", "basis", "baseline", "element"])


def fig_load_bearing(eh: pd.DataFrame, thresh: float = 0.02) -> tuple[list[str], list[str]]:
    """fig4: element-exclusion filter price heatmap (elements with a headline price ≥ thresh anywhere)."""
    plt = common.style()
    import matplotlib
    from matplotlib.colors import PowerNorm
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("white")                                          # exactly zero (and no data) is white
    et = common.elements_table()
    zmap = dict(zip(et["symbol"], et["Z"])) if "symbol" in et.columns else {}
    mx = eh.groupby("element").price_headline.max()
    els = sorted(mx.index[mx >= thresh].tolist(), key=lambda s: zmap.get(s, 999))
    E = np.sort(eh.energy_keV.unique())
    edges = np.concatenate([[E[0]], np.sqrt(E[1:] * E[:-1]), [E[-1]]])
    vmax = 100 * float(eh.price_headline.max())
    fig, axs = plt.subplots(2, 2, figsize=(common.DOUBLE_COL_IN, 0.12 * len(els) * 2 + 1.4), sharex=True,
                            sharey=True, layout="constrained")
    im = None
    for i, basis in enumerate(core.BASES):
        for j, bl in enumerate(("F0", "F0&B4")):
            ax = axs[i, j]
            z = (eh[(eh.basis == basis) & (eh.baseline == bl)]
                 .pivot(index="element", columns="energy_keV", values="price_headline").reindex(index=els, columns=E))
            zz = 100 * z.to_numpy()
            zz = np.ma.masked_where(~(zz > 0), zz)
            im = ax.pcolormesh(edges, np.arange(len(els) + 1), zz, cmap=cmap,
                               norm=PowerNorm(0.5, vmin=0, vmax=vmax), shading="flat", rasterized=True)
            _logx(ax)
            ax.set_yticks(np.arange(len(els)) + 0.5)
            ax.set_yticklabels(els, fontsize=6)
            ax.set_ylim(len(els), 0)
            ax.set_title(f"{BASIS_LABEL[basis]}, against {'F0' if bl == 'F0' else 'F0 ∩ B4'}", loc="left", fontsize=7.5)
            ax.tick_params(length=2)
            for E0 in LINES.values():
                ax.axvline(E0, color="#8a8984", lw=0.4, ls=":")
            if i == 1:
                ax.set_xlabel("photon energy (keV)")
    cb = fig.colorbar(im, ax=axs, shrink=0.45, pad=0.01, aspect=30)
    cb.set_label("element-exclusion filter price (%), best non-artefact\ncompound, rule R2, narrow beam; "
                 "square-root colour scale, white = 0")
    fig.suptitle(f"{len(els)} elements whose filter price reaches ≥ {100 * thresh:g} % at some energy "
                 f"(of {eh.element.nunique()} tested); narrow beam: under the fluorescence correction the "
                 f"U maximum falls to ≤ 4 % (dossier R2.7)", fontsize=7.5)
    out = common.savefig(fig, "fig4_load_bearing")
    plt.close(fig)
    return out, els


def fig_tube(tube: pd.DataFrame) -> list[str]:
    """Tube-quality curves: DIAG areal-mass prices vs RQR, air kerma (thick), fluence/energy fluence (thin)."""
    plt = common.style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    t = tube[(tube.rule == core.PRIMARY_RULE) & tube.coherent]
    qs = ["RQR5", "RQR6", "RQR8", "RQR9"]
    xl = ["RQR5\n70 kV", "RQR6\n80 kV", "RQR8\n100 kV", "RQR9\n120 kV"]
    x = np.arange(4)
    fig, axs = plt.subplots(4, 3, figsize=(common.DOUBLE_COL_IN, 7.8), sharex=True)
    resp = [("air_kerma", 1.2, "-"), ("fluence", 0.6, (0, (1, 1))), ("energy_fluence", 0.6, (0, (4, 1.5)))]
    for ax, c in zip(axs.flat, HEAD_CONSTRAINTS):
        _recess(ax)
        s = t[t.constraint == c]
        ak = s[s.response == "air_kerma"]
        env = ak.groupby("quality").price_headline.agg(["min", "max"]).reindex(qs)
        ax.fill_between(x, 100 * env["min"], 100 * env["max"], color=COL["areal"], alpha=0.16, lw=0)
        for angle, mk in ((10, "v"), (16, "^")):
            a = ak[(ak.anode_angle == angle) & (ak.target_value == 0.35)].set_index("quality").price_headline.reindex(qs)
            ax.plot(x, 100 * a, ls="none", marker=mk, ms=2.6, color=COL["areal"], mew=0)
        for r, lw, ls in resp:
            a = (s[(s.response == r) & (s.anode_angle == 12) & (s.target_value == 0.35)]
                 .set_index("quality").price_headline.reindex(qs))
            ax.plot(x, 100 * a, color=COL["areal"] if r == "air_kerma" else "#52514e", lw=lw, ls=ls,
                    marker="o" if r == "air_kerma" else None, ms=2.5)
        ax.axhline(0, color="#52514e", lw=0.5)
        ax.set_title(TITLES[c], loc="left", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xl, fontsize=6)
    for ax in axs[:, 0]:
        ax.set_ylabel("areal-mass price (%)")
    hs = [Line2D([], [], color=COL["areal"], lw=1.2, marker="o", ms=2.5),
          Line2D([], [], color="#52514e", lw=0.6, ls=(0, (1, 1))), Line2D([], [], color="#52514e", lw=0.6, ls=(0, (4, 1.5))),
          Line2D([], [], color=COL["areal"], ls="none", marker="v", ms=3),
          Line2D([], [], color=COL["areal"], ls="none", marker="^", ms=3), Patch(color=COL["areal"], alpha=0.25, lw=0)]
    fig.legend(hs, ["air kerma, 12°, 0.35 mm Pb", "fluence", "energy fluence", "anode 10°", "anode 16°",
                    "air kerma: 3 angles × 3 Pb targets"], loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=5.6)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = common.savefig(fig, "P_tube_prices")
    plt.close(fig)
    return out


def fig_pb(pb: pd.DataFrame) -> list[str]:
    """Why PB = 0: best non-Pb / fcc Pb requirement vs energy and the number of non-Pb compounds beating Pb."""
    plt = common.style()
    fig, axs = plt.subplots(1, 3, figsize=(common.DOUBLE_COL_IN, 2.4))
    labs = {"F0": "F0", "F0&RAD": "F0 ∩ RAD", "F0&EXP": "F0 ∩ EXP", "F0&B4": "F0 ∩ B4"}
    for ax, basis in zip(axs[:2], core.BASES):
        _recess(ax)
        for k, (s, lab) in enumerate(labs.items()):
            g = pb[(pb.basis == basis) & (pb.set == s)].sort_values("energy_keV")
            ls = ["-", (0, (4, 1.5)), "-", (0, (1, 1))][k]
            ax.plot(g.energy_keV, g.ratio_best_nonPb_to_Pb, color=CAT[k], lw=1.4 if k == 0 else 0.8, ls=ls, label=lab,
                    zorder=3 - 0.1 * k)
        ax.axhline(1, color="#52514e", lw=0.6)
        ax.axvline(PB_K_KEV, color="#b9b8b2", lw=0.5, ls=":")
        _logx(ax)
        ax.set_title(f"best non-Pb / fcc Pb, {BASIS_LABEL[basis]}", loc="left", fontsize=7)
        ax.set_xlabel("photon energy (keV)")
    axs[0].set_ylabel("requirement ratio (< 1: Pb beaten)")
    axs[0].legend(frameon=False, fontsize=5.5, title="coincident lines overlap", title_fontsize=5)
    ax = axs[2]
    _recess(ax)
    for basis in core.BASES:
        g = pb[(pb.basis == basis) & (pb.set == "F0")].sort_values("energy_keV")
        ax.plot(g.energy_keV, np.maximum(g.n_nonPb_beating_Pb, 0.5), color=COL[basis], lw=0.9, label=BASIS_LABEL[basis])
    ax.set_yscale("log")
    _logx(ax)
    ax.set_title("non-Pb compounds in F0 beating Pb\n(0 plotted at 0.5)", loc="left", fontsize=7)
    ax.set_xlabel("photon energy (keV)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = common.savefig(fig, "P_pb_anatomy")
    plt.close(fig)
    return out


def fig_shapley(sh: pd.DataFrame) -> list[str]:
    """Shapley φ vs energy (best_clean solid, median50 dashed) and the STAB × EXP interaction index."""
    plt = common.style()
    s = sh[(sh.rule == core.PRIMARY_RULE) & sh.coherent]
    fig, axs = plt.subplots(2, 2, figsize=(common.DOUBLE_COL_IN, 4.2), sharex=True)
    for j, basis in enumerate(core.BASES):
        ax = axs[0, j]
        _recess(ax)
        for k, p in enumerate(PLAYERS):
            for kind, ls, lw in ((HEADLINE, "-", 1.0), ("median50", (0, (2, 1.5)), 0.5)):
                g = s[(s.basis == basis) & (s.kind == kind)].sort_values("energy_keV")
                ax.plot(g.energy_keV, g[f"phi_{p}"], color=CAT[k], lw=lw, ls=ls, label=p if kind == HEADLINE else None)
        ax.axhline(0, color="#52514e", lw=0.5)
        _logx(ax)
        ax.set_title(f"Shapley φ on log q*, {BASIS_LABEL[basis]}", loc="left", fontsize=7.5)
        ax = axs[1, j]
        _recess(ax)
        for kind, ls, lw in ((HEADLINE, "-", 1.0), ("median50", (0, (1, 1)), 0.6), ("best", (0, (2, 1.5)), 0.6)):
            g = s[(s.basis == basis) & (s.kind == kind)].sort_values("energy_keV")
            ax.plot(g.energy_keV, g.I_STAB_EXP, color="#52514e", lw=lw, ls=ls, label=kind)
        ax.axhline(0, color="#8a8984", lw=0.5)
        _logx(ax)
        ax.set_title("STAB × EXP interaction index (< 0: substitutes)", loc="left", fontsize=7.5)
        ax.set_xlabel("photon energy (keV)")
    axs[0, 0].set_ylabel("φ (log units)")
    axs[1, 0].set_ylabel("I(STAB, EXP) (log units)")
    axs[0, 0].legend(frameon=False, fontsize=6, title="solid: best non-artefact; dashed: median50",
                     title_fontsize=6)
    axs[1, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = common.savefig(fig, "P_shapley_vs_energy")
    plt.close(fig)
    return out


def fig_bootstrap(bp: pd.DataFrame, ts: pd.DataFrame) -> list[str]:
    """Density-bias bootstrap diagnostics: radionuclide price bands, mono curves, tier-1 stability."""
    plt = common.style()
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator
    fig, axs = plt.subplots(1, 3, figsize=(common.DOUBLE_COL_IN, 2.9))
    mcol = {"iid": CAT[0], "mixed": CAT[2], "class_shift": CAT[1]}
    ax = axs[0]
    _recess(ax)
    k = 0
    ticks, tl = [], []
    for sc, vid in (("CS137", 38), ("CO60", 42)):
        for c in ("RAD", "GAP(3.1)", "B4", "B5", "B6"):
            for mi, mode in enumerate(BOOT_MODES):
                r = bp[(bp["mode"] == mode) & (bp.variant_id == vid) & (bp.constraint == c) & (bp.kind == "headline")].iloc[0]
                xx = k + (mi - 1) * 0.24
                ax.plot([xx, xx], [100 * r.p05, 100 * r.p95], color=mcol[mode], lw=1.4, solid_capstyle="butt")
                ax.plot(xx, 100 * r.nominal, "o", color="#0b0b0b", ms=1.6)
            ticks.append(k)
            tl.append(f"{c.replace('(3.1)', '')} {'Cs' if sc == 'CS137' else 'Co'}")
            k += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl, fontsize=5.5, rotation=90)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, 400)
    ax.set_yticks([0, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["0", "5", "10", "20", "50", "100", "200"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("thickness price (%)")
    ax.set_title("a  5–95 % bands (dot: nominal)", loc="left", fontsize=7)
    ax = axs[1]
    _recess(ax)
    for c, ls in (("B4", "-"), ("RAD", (0, (2, 1.5)))):
        for mode in BOOT_MODES:
            g = bp[(bp["mode"] == mode) & (bp.source == "mono") & (bp.constraint == c)
                   & (bp.kind == "headline")].sort_values("energy_keV")
            if c == "B4":
                ax.plot(g.energy_keV, 100 * g.p05, color=mcol[mode], lw=0.6)
                ax.plot(g.energy_keV, 100 * g.p95, color=mcol[mode], lw=0.6)
        ax.plot(g.energy_keV, 100 * g.nominal, color="#0b0b0b", lw=0.9, ls=ls, marker="o", ms=1.5)
    _logx(ax)
    ax.set_xlabel("photon energy (keV)")
    ax.set_ylabel("price (%)")
    ax.set_title("b  B4 at key energies: nominal (black),\n    5 %/95 % per mode; RAD nominal dashed", loc="left", fontsize=6.3)
    ax = axs[2]
    _recess(ax)
    k = 0
    ticks, tl = [], []
    for sc, vid in (("CS137", 38), ("CO60", 42)):
        for b in ("B4", "B5", "B6"):
            for mi, mode in enumerate(BOOT_MODES):
                r = ts[(ts["mode"] == mode) & (ts.variant_id == vid) & (ts.bundle == b)].iloc[0]
                xx = k + (mi - 1) * 0.26
                ax.bar(xx, r.mean_jaccard_vs_nominal, width=0.22, color=mcol[mode], lw=0)
                ax.plot(xx, r.coverage_true_best_in_nominal_tier1, "_", color="#0b0b0b", ms=5, mew=1)
            ticks.append(k)
            tl.append(f"{b}\n{'Cs' if sc == 'CS137' else 'Co'}")
            k += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl, fontsize=5.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("c  tier 1: Jaccard vs nominal (bars);\n    true best in nominal tier 1 (ticks)", loc="left", fontsize=6.3)
    fig.legend([Patch(color=mcol[m]) for m in BOOT_MODES], ["i.i.d. by class", f"mixed (ρ = {MIXED_RHO})", "class shift"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=6)
    fig.tight_layout()
    out = common.savefig(fig, "P_bootstrap")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------
# summary and run
# ---------------------------------------------------------------------------------------------
def _pct(x) -> float:
    return float(100.0 * x) if x is not None and np.isfinite(x) else float("nan")


def _summary(pc, head, tube, gi, pbt, closest, eh, sh, boot, checks, comp, els_fig4, n_fb) -> dict:
    out = {"energy_grid": {"n_nodes": int(head.energy_keV.nunique()), "min_keV": float(head.energy_keV.min()),
                           "max_keV": float(head.energy_keV.max())},
           "topk": TOPK, "n_fallback_columns": int(n_fb), "phase2_checks": checks,
           "conventions": {"headline": "R2, with coherent, kind best_clean (best compound after removing artefact "
                                       "entries, theoretical and e_hull > 0.1 eV/atom, from both sides; P3-D-011)",
                           "legacy_rule": "superseded per-cell rule P3-D-005: median50 where the best-compound frontier "
                                          "on either side is an artefact entry, else best (price_legacy_rule)",
                           "band_all": "envelope over R1/R2/R3 x {best, 10th, median50, best_icsd, best_clean, "
                                       "median50_clean} x coherent on/off",
                           "band_credible": "envelope over R1/R2/R3 x {best_clean, best_icsd, median50_clean} x "
                                            "coherent on/off (kinds whose frontiers exclude artefact entries)",
                           "band_kinds": "R2, with coherent, 6 kinds",
                           "band_rules": "R1/R2/R3 at the R2 headline kind, with coherent",
                           "band_coherent": "R2 at the R2 headline kind, coherent on/off"}}
    # item 1
    ku = head.groupby("basis").artefact_any.mean().astype(float).to_dict()
    ku_c = head.groupby(["basis", "constraint"]).artefact_any.mean()
    out["share_cells_best_frontier_artefact"] = {"overall_by_basis": ku,
                                                 "by_basis_constraint": {f"{b}|{c}": float(v) for (b, c), v in ku_c.items()}}
    curves = {}
    sel = HEAD_CONSTRAINTS + ["GAP(2.5)", "GAP(3.5)", "STAB(0)", "STAB(0.1)", "ABUND(10)"]
    for (b, c), g in head[head.constraint.isin(sel)].groupby(["basis", "constraint"]):
        g = g.set_index("energy_keV")
        i = g.price_headline.idxmax()
        rec = {"max_headline_pct": _pct(g.price_headline.max()), "energy_keV_at_max": float(i),
               "kind_at_max": g.headline_kind[i], "band_all_at_max_pct": [_pct(g.band_all_lo[i]), _pct(g.band_all_hi[i])],
               "min_headline_pct": _pct(g.price_headline.min()),
               "energy_keV_at_min": float(g.price_headline.idxmin()),
               "median_band_width_pp": {s: float(100 * (g[f"band_{s}_hi"] - g[f"band_{s}_lo"]).median())
                                        for s in ("all", "credible", "kinds", "rules", "coherent")},
               "key_energies": {}}
        for E in KEY_E:
            r = g.loc[E]
            rec["key_energies"][f"{E:g}"] = {"headline_pct": _pct(r.price_headline), "kind": r.headline_kind,
                                             "by_kind_pct": {k: _pct(r[f"price_{k}"]) for k in common.KINDS6},
                                             "legacy_rule_pct": _pct(r.price_legacy_rule),
                                             "band_all_pct": [_pct(r.band_all_lo), _pct(r.band_all_hi)],
                                             "band_credible_pct": [_pct(r.band_credible_lo), _pct(r.band_credible_hi)],
                                             "band_kinds_pct": [_pct(r.band_kinds_lo), _pct(r.band_kinds_hi)],
                                             "band_rules_pct": [_pct(r.band_rules_lo), _pct(r.band_rules_hi)],
                                             "band_coherent_pct": [_pct(r.band_coherent_lo), _pct(r.band_coherent_hi)]}
        curves[f"{b}|{c}"] = rec
    out["price_curves"] = curves
    # item 2
    ak = tube[(tube.rule == core.PRIMARY_RULE) & tube.coherent & (tube.response == "air_kerma")]
    tq = {}
    for (c, q), g in ak[ak.constraint.isin(HEAD_CONSTRAINTS)].groupby(["constraint", "quality"]):
        tq[f"{c}|{q}"] = {"headline_pct_min": _pct(g.price_headline.min()), "headline_pct_max": _pct(g.price_headline.max()),
                          "best_pct_min": _pct(g.price_best.min()), "best_pct_max": _pct(g.price_best.max()),
                          "share_best_frontier_artefact": float(g.artefact_any.mean()),
                          "median50_pct_min": _pct(g.price_median50.min()), "median50_pct_max": _pct(g.price_median50.max()),
                          "legacy_rule_pct_min": _pct(g.price_legacy_rule.min()),
                          "legacy_rule_pct_max": _pct(g.price_legacy_rule.max()),
                          "band_rules_kinds_pct": [_pct(g.band_rules_kinds_lo.min()), _pct(g.band_rules_kinds_hi.max())]}
    out["tube_prices_air_kerma_R2_coh"] = tq
    oth = tube[(tube.rule == core.PRIMARY_RULE) & tube.coherent & (tube.constraint == "B4")]
    out["tube_B4_by_response_pct"] = {r: [_pct(g.price_headline.min()), _pct(g.price_headline.max())]
                                      for r, g in oth.groupby("response")}
    # item 3
    g = gi[gi.constraint == "GAP(3.1)"].set_index("energy_keV")
    gp = {}
    for E in KEY_E:
        r = g.loc[E]
        gp[f"{E:g}"] = {"P_thickness_pct": _pct(r.P_thickness_headline), "P_mass_pct": _pct(r.P_mass_headline),
                        "P_thickness_band_pct": [_pct(r.P_thickness_band_lo), _pct(r.P_thickness_band_hi)],
                        "P_mass_band_pct": [_pct(r.P_mass_band_lo), _pct(r.P_mass_band_hi)],
                        "inversion": float(r.inversion_clean), "inversion_median50_clean": float(r.inversion_median50_clean),
                        "inversion_best": float(r.inversion_best), "inversion_median50": float(r.inversion_median50),
                        "density_factor": float(r.density_factor_clean), "mismatch_factor": float(r.mismatch_factor_clean),
                        "mass_factor": float(r.mass_factor_clean),
                        "thk_F0": r.thk_F0_clean, "thk_K": r.thk_K_clean, "mass_F0": r.mass_F0_clean,
                        "mass_K": r.mass_K_clean,
                        "rho_top50_median_F0_clean_thk": float(r.rho_top50_median_F0_clean_thk),
                        "rho_top50_median_GAP_clean_thk": float(r["rho_top50_median_GAP(3.1)_clean_thk"]),
                        "rho_top50_median_F0_thk": float(r.rho_top50_median_F0_thk)}
    hi = g[g.index >= 200]

    def rr(x):
        return [float(np.nanmin(x)), float(np.nanmax(x))]
    out["gap_inversion"] = {
        "key_energies": gp,
        "inversion_range_all_E": {k: rr(g[f"inversion_{k}"]) for k in ("clean", "median50_clean", "best", "median50")},
        "inversion_range_E_ge_200keV": {k: rr(hi[f"inversion_{k}"]) for k in ("clean", "median50_clean", "best",
                                                                            "median50")},
        "share_E_where_Pt_gt_Pm": float((g.P_thickness_headline > g.P_mass_headline).mean()),
        "identity_max_abs_residual": float(np.nanmax(np.abs(gi[["identity_residual_clean", "identity_residual_best",
                                                                 "inversion_residual_clean",
                                                                 "inversion_residual_best"]].to_numpy()))),
        "density_factor_range_E_ge_200keV": rr(hi.density_factor_clean),
        "mismatch_factor_range_E_ge_200keV": rr(hi.mismatch_factor_clean),
        "rho_top50_median_F0_clean_thk_range": rr(g.rho_top50_median_F0_clean_thk),
        "rho_top50_median_GAP_clean_thk_range": rr(g["rho_top50_median_GAP(3.1)_clean_thk"]),
        "thk_F0_compounds_E_ge_200keV": sorted(set(hi.thk_F0_clean)),
        "thk_K_compounds_E_ge_200keV": sorted(set(hi.thk_K_clean))}
    # item 4
    pbc = head[head.constraint == "PB"]
    pbb = pc[(pc.constraint == "PB") & (pc.kind == "best") & (pc.rule == core.PRIMARY_RULE) & pc.coherent]
    out["pb"] = {"closest": closest.to_dict("records"),
                 "PB_headline_by_basis": {b: {"max_pct": _pct(x.price_headline.max()),
                                              "energies_keV_nonzero": [float(e) for e in x.energy_keV[x.price_headline > 0]],
                                              "kinds_nonzero": sorted(set(x.headline_kind[x.price_headline > 0]))}
                                          for b, x in pbc.groupby("basis")},
                 "PB_band_all_max_pct_by_basis": {b: _pct(x.band_all_hi.max()) for b, x in pbc.groupby("basis")},
                 "PB_best_kind_R2_coh_nonzero_energies": {b: [float(e) for e in x.energy_keV[x.price > 1e-12]]
                                                          for b, x in pbb.groupby("basis")},
                 "PB_all_axes_nonzero": _pb_windows(pc),
                 "practical_mass_margin": _pb_margin(pbt)}
    # item 5
    s = sh[(sh.rule == core.PRIMARY_RULE) & sh.coherent]
    sv = {}
    for (b, k), g in s.groupby(["basis", "kind"]):
        tot = g.total.to_numpy()
        gi_ = g.set_index("energy_keV")
        sv[f"{b}|{k}"] = {
            "share_E_I_STAB_EXP_negative": float((g.I_STAB_EXP < -1e-9).mean()),
            "share_E_I_STAB_EXP_positive": float((g.I_STAB_EXP > 1e-9).mean()),
            "I_STAB_EXP_range": [float(g.I_STAB_EXP.min()), float(g.I_STAB_EXP.max())],
            "I_given_empty_share_negative": float((g.I_STAB_EXP_given_empty < -1e-9).mean()),
            "I_given_PB_RAD_share_negative": float((g.I_STAB_EXP_given_PB_RAD < -1e-9).mean()),
            "phi_share_median_over_E_with_total_gt_0.01": {
                p: float(np.median((g[f"phi_{p}"] / g.total)[tot > 0.01])) for p in PLAYERS},
            "key_energies": {f"{E:g}": {**{p: float(gi_.loc[E, f"phi_{p}"]) for p in PLAYERS},
                                        "total": float(gi_.loc[E, "total"]), "I_STAB_EXP": float(gi_.loc[E, "I_STAB_EXP"])}
                             for E in (20.0, 60.0, 100.0, 200.0, 661.657, 1250.0, 5000.0)}}
    allr = sh[sh.coherent & (sh.kind == "median50")]
    sv["median50_all_rules_both_bases_share_negative"] = float((allr.I_STAB_EXP < -1e-9).mean())
    out["shapley"] = sv
    # item 6
    lb = {}
    for (b, bl), g in eh.groupby(["basis", "baseline"]):
        i = g.price_headline.idxmax()
        top = g.sort_values("price_headline", ascending=False).drop_duplicates("element").head(8)
        mxe = g.groupby("element").price_headline.max()
        lb[f"{b}|{bl}"] = {"max_pct": _pct(g.loc[i, "price_headline"]), "element": g.loc[i, "element"],
                           "energy_keV": float(g.loc[i, "energy_keV"]), "kind": g.loc[i, "headline_kind"],
                           "max_best_kind_pct": _pct(g.price_best.max()),
                           "n_elements_ever_ge_5pct": int((mxe >= 0.05).sum()),
                           "n_elements_ever_ge_1pct": int((mxe >= 0.01).sum()),
                           "top_elements": [{"element": r.element, "max_pct": _pct(r.price_headline),
                                             "energy_keV": float(r.energy_keV)} for r in top.itertuples()],
                           "share_cells_ge_1pct": float((g.price_headline >= 0.01).mean()),
                           "share_energies_any_element_ge_5pct": float(
                               (g.groupby("energy_keV").price_headline.max() >= 0.05).mean()),
                           "per_element_max_pct": {e: _pct(v) for e, v in mxe[mxe >= 0.01].sort_values(ascending=False).items()},
                           "kinds_at_max": {k: _pct(g.loc[i, f"price_{k}"]) for k in KINDS_SH}}
    out["load_bearing"] = lb
    out["load_bearing_fig4_elements"] = els_fig4
    # item 7 moved to density_model (the hardening revision)
    out["phase2_comparison"] = comp[["scenario", "basis", "constraint", "phase2_profile_best_min", "phase2_profile_best_max",
                                     "mono_best", "mono_headline", "mono_headline_kind",
                                     "abs_diff_best_mono_vs_profile_max"]].to_dict("records")
    return out


def _pb_windows(pc: pd.DataFrame) -> dict:
    """Energies where the PB price is non-zero, per basis and kind, over all rules and coherent settings."""
    x = pc[pc.constraint == "PB"]
    out = {}
    for (b, k), g in x.groupby(["basis", "kind"]):
        nz = g[g.price > 1e-12]
        out[f"{b}|{k}"] = {"n_energies": int(nz.energy_keV.nunique()),
                           "energy_range_keV": [float(nz.energy_keV.min()), float(nz.energy_keV.max())] if len(nz) else [],
                           "max_pct": _pct(g.price.max()),
                           "energy_keV_at_max": float(g.loc[g.price.idxmax(), "energy_keV"]) if len(g) else float("nan")}
    return out


def _pb_margin(pbt: pd.DataFrame) -> dict:
    """How close fcc Pb is to the best non-Pb compound of each practical set (R2), 100 keV–10 MeV."""
    out = {}
    for (b, sname), g in pbt.groupby(["basis", "set"]):
        h = g[g.energy_keV >= 100]
        out[f"{b}|{sname}"] = {"ratio_best_nonPb_to_Pb_100keV_10MeV": [float(h.ratio_best_nonPb_to_Pb.min()),
                                                                     float(h.ratio_best_nonPb_to_Pb.max())],
                               "ratio_tableC_density": [float(h.ratio_best_nonPb_to_Pb_tableC_density.min()),
                                                        float(h.ratio_best_nonPb_to_Pb_tableC_density.max())]}
    return out


def _boot_summary(boot: dict) -> dict:
    """Item 7 headline numbers: radionuclide price bands and tier stability."""
    bp = boot["prices"]
    bsum = {}
    sub = bp[bp.source.isin(["CS137", "CO60"]) & bp.constraint.isin(["RAD", "GAP(3.1)", "B4", "B5", "B6"])
             & bp.kind.isin(["headline", "best", "median50"])]
    for (mode, sc, c, kd), g in sub.groupby(["mode", "source", "constraint", "kind"]):
        bsum[f"{mode}|{sc}|{c}|{kd}"] = {"nominal_pct": [_pct(g.nominal.min()), _pct(g.nominal.max())],
                                         "p05_pct": _pct(g.p05.min()), "median_pct": _pct(g["median"].median()),
                                         "p95_pct": _pct(g.p95.max())}
    ts = boot["tier_summary"]
    tsd = {}
    for (mode, sc, b), g in ts.groupby(["mode", "scenario", "bundle"]):
        tsd[f"{mode}|{sc}|{b}"] = {"n_tier1_nominal": [int(g.n_tier1_nominal.min()), int(g.n_tier1_nominal.max())],
                                   "frac_draws_set_changed": [float(g.frac_draws_tier1_set_changed.min()),
                                                              float(g.frac_draws_tier1_set_changed.max())],
                                   "mean_jaccard": [float(g.mean_jaccard_vs_nominal.min()),
                                                    float(g.mean_jaccard_vs_nominal.max())],
                                   "coverage_true_best": [float(g.coverage_true_best_in_nominal_tier1.min()),
                                                          float(g.coverage_true_best_in_nominal_tier1.max())],
                                   "min_member_stay_frac": float(g.min_member_stay_frac.min())}
    return {"meta": boot["meta"], "radionuclide_prices": bsum, "tiers": tsd}


OUT_FILES = {"price_curves": "price_curves.parquet", "shapley": "shapley_vs_energy.parquet",
             "excl": "load_bearing_vs_energy.parquet", "pb": "pb_anatomy.csv", "gap_density": "gap_frontier_density.csv"}


def run(reuse: bool = False, n_draws: int = N_DRAWS, figures: bool = True, reuse_boot: bool = False) -> dict:
    """Regenerate every continuum-module output (idempotent).

    reuse=True reloads the grid pass (items 1, 3–6) from disk; reuse_boot=True reloads the item 7 tables
    (bootstrap_*.csv) instead of redrawing. run() with defaults recomputes everything.
    """
    paths.ensure(OUT)
    t0 = time.time()
    f = {k: OUT / v for k, v in OUT_FILES.items()}
    if reuse and all(p.exists() for p in f.values()):
        pc = pd.read_parquet(f["price_curves"])
        sh = pd.read_parquet(f["shapley"])
        ex = pd.read_parquet(f["excl"])
        pbraw = pd.read_csv(f["pb"])
        dens = pd.read_csv(f["gap_density"])
        old = OUT / "summary.json"
        n_fb = json.loads(old.read_text(encoding="utf-8")).get("n_fallback_columns", -1) if old.exists() else -1
    else:
        g = compute_grid()
        pc, sh, ex, pbraw, dens, n_fb = (g["price_curves"], g["shapley"], g["excl"], g["pb"], g["gap_density"],
                                         g["n_fallback_columns"])
        pc.to_parquet(f["price_curves"], index=False)
        sh.to_parquet(f["shapley"], index=False)
        ex.to_parquet(f["excl"], index=False)
        dens.to_csv(f["gap_density"], index=False)
    print(f"[P] grid pass ready ({time.time() - t0:.0f} s)", flush=True)
    head = headline_curves(pc)
    head.to_csv(OUT / "price_curves_headline.csv", index=False)
    tube = tube_prices()
    tube.to_csv(OUT / "tube_prices.csv", index=False)
    gi = gap_inversion(pc, head, dens)
    gi.to_csv(OUT / "gap_inversion.csv", index=False)
    pbt, closest, beaters = pb_anatomy(pbraw)
    pbt.to_csv(f["pb"], index=False)
    closest.to_csv(OUT / "pb_closest.csv", index=False)
    beaters.to_csv(OUT / "pb_beaters_at_closest.csv", index=False)
    eh = excl_headline(ex)
    eh.to_parquet(OUT / "load_bearing_headline.parquet", index=False)
    checks = phase2_checks(pc, ex, sh, tube)
    comp = scenario_comparison(pc, head)
    comp.to_csv(OUT / "phase2_comparison.csv", index=False)
    print(f"[P] tables written ({time.time() - t0:.0f} s)", flush=True)
    # item 7 (density bootstrap) is superseded by analysis3.density_model: remove stale outputs
    boot = None
    for v in list(BOOT_FILES.values()) + ["bootstrap_meta.json"]:
        (OUT / v).unlink(missing_ok=True)
    for ext in ("png", "svg"):
        (common.FIGURES3 / f"P_bootstrap.{ext}").unlink(missing_ok=True)
    figs, els = {}, []
    if figures:
        figs["fig2_price_curves"] = fig_price_curves(head)
        figs["P_price_curves_params"] = fig_price_params(head)
        figs["P_band_drivers"] = fig_band_drivers(head)
        figs["fig3_gap_inversion"] = fig_gap_inversion(gi)
        figs["fig4_load_bearing"], els = fig_load_bearing(eh)
        figs["P_tube_prices"] = fig_tube(tube)
        figs["P_pb_anatomy"] = fig_pb(pbt)
        figs["P_shapley_vs_energy"] = fig_shapley(sh)
    summ = _summary(pc, head, tube, gi, pbt, closest, eh, sh, boot, checks, comp, els, n_fb)
    summ["figures"] = figs
    common.write_json(OUT / "summary.json", summ)
    for p in OUT.glob("_*"):                                         # scratch files of interactive runs
        p.unlink()
    common.write_json(OUT / "_timing.json", {"runtime_s": round(time.time() - t0, 1)})
    print(f"[P] done ({time.time() - t0:.0f} s)", flush=True)
    return summ


BOOT_FILES = {"prices": "bootstrap_prices.csv", "tiers": "bootstrap_tiers.csv", "tier_summary": "bootstrap_tier_summary.csv"}


def _write_bootstrap(boot: dict) -> dict:
    for k, v in BOOT_FILES.items():
        boot[k].to_csv(OUT / v, index=False)
    common.write_json(OUT / "bootstrap_meta.json", boot["meta"])
    return boot


def load_bootstrap() -> dict | None:
    """Item 7 tables from disk (None if any is missing)."""
    if not all((OUT / v).exists() for v in BOOT_FILES.values()) or not (OUT / "bootstrap_meta.json").exists():
        return None
    out = {k: pd.read_csv(OUT / v) for k, v in BOOT_FILES.items()}
    out["meta"] = json.loads((OUT / "bootstrap_meta.json").read_text(encoding="utf-8"))
    return out


def run_item7(n_draws: int = N_DRAWS) -> dict:
    """Re-run only item 7 (e.g. once the COD module's compound δ distribution exists): tables, P_bootstrap figure,
    and the "bootstrap" section of summary.json. The other items are left untouched."""
    t0 = time.time()
    boot = _write_bootstrap(bootstrap(n_draws=n_draws))
    fig = fig_bootstrap(boot["prices"], boot["tier_summary"])
    sp = OUT / "summary.json"
    summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    summ["bootstrap"] = _boot_summary(boot)
    summ.setdefault("figures", {})["P_bootstrap"] = fig
    summ["item7_runtime_s"] = round(time.time() - t0, 1)
    common.write_json(sp, summ)
    print(f"[P item 7] done ({time.time() - t0:.0f} s): {boot['meta']['delta_source'][:120]}", flush=True)
    return summ['bootstrap']


if __name__ == "__main__":
    import sys
    if "--item7" in sys.argv:
        run_item7()
    else:
        s = run(reuse="--reuse" in sys.argv, reuse_boot="--reuse-boot" in sys.argv)
        print(json.dumps({k: s[k] for k in ("energy_grid", "phase2_checks")}, indent=1, default=str))
