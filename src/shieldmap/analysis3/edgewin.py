"""Narrow-beam K-edge-window claims under the fluorescence correction.

Regenerate with ``python -m shieldmap.analysis3.edgewin`` from ``shieldmap/`` (outputs in ``data/phase3/E/``,
report ``reports/phase3/step10_edge_windows.md``). Physics review findings 3 and 4.

Monoenergetic line source at a grid energy E, air-kerma response, fixed transmission T (0.1 and 0.01), the
Fluorescence-module model (``fluorescence.mono_effective``: E₁ fluence kernel and K + L lines, the primary; E₂ planar current
as a sensitivity). Requirements, as the continuum module (``continuum``) defines them for a line source:

    areal      q = ln(1/T) / (μ/ρ)_eff                  (g/cm²)
    thickness  q = ln(1/T) / ((μ/ρ)_eff · ρ_MP)        (cm)

(narrow beam: (μ/ρ)_eff = μ/ρ of Table C, so q·μ = ln(1/T) and the ratio of requirements is T-independent.)
Frontier kinds per set M (rule R2 base): best = min q over M; best_clean = min q over M without artefact
entries (common.artefact_flag); median50 = mean of the 25th and 26th smallest q. A price is
q_kind(M ∩ K) / q_kind(M) − 1.

Certified candidate sets. Fluorescence only lowers (μ/ρ)_eff, so q_eff ≥ q_narrow. For each set the rows are
taken in narrow-beam order, first N0 = 200, and N is doubled until the k-th smallest corrected q among them
(k = 1 for best/best_clean, 26 for median50) does not exceed the narrow-beam q of the first row left out: then
no row outside can enter the kind's statistic.

(a) Pb mass-basis window: every grid node 85–95 keV; M = F0 (R2 base), K = PB (no Pb). Pb's own corrected
    rank in F0 is reported.
(b) Load-bearing at absorption-edge windows, thickness basis: M = F0 and F0 ∩ B4, K = "contains no X" for every
    (X, baseline) whose continuum-module narrow-beam R2 load-bearing price reaches ≥ 10 % in any kind at E ≤ 150 keV
    (``data/phase3/P/load_bearing_vs_energy.parquet``); evaluated at that pair's LB_TOP energies of largest
    narrow-beam price (any kind), which include its maximum (Np alone has 141 such energies, a density
    effect of α-Np, not an edge window).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common

OUT = common.P3 / "E"
WINDOW_PB_KEV = (85.0, 95.0)
LB_MAX_KEV = 150.0
LB_THRESHOLD = 0.10
LB_TOP = 3
T_LEVELS = (0.1, 0.01)
MODELS = {"E1": {}, "E2": {"escape": "E2"}}          # E1 + L lines = the fluorescence module primary; E2 = planar current
KINDS = ("best", "best_clean", "median50")
N0 = 200


# =============================================================================================
# corrected attenuation, cached per (energy, T, model)
# =============================================================================================
class EffCache:
    """(μ/ρ)_eff of universe rows at one grid energy, one T, one model; computed on demand in batches."""

    def __init__(self, e_idx: int, T: float, model: str):
        self.e_idx, self.T, self.model = int(e_idx), float(T), model
        self.e_keV = float(common.energy_keV()[e_idx])
        self.val: dict[int, float] = {}

    def get(self, rows: np.ndarray) -> np.ndarray:
        from shieldmap.analysis3 import fluorescence as fl
        rows = np.asarray(rows, dtype=int)
        miss = np.array([r for r in rows if r not in self.val], dtype=int)
        if miss.size:
            W = common.mass_fractions()[miss].toarray()
            r = fl.mono_effective(W, self.e_keV, self.T, **MODELS[self.model])
            self.val.update(zip(miss.tolist(), np.asarray(r["mu_rho_eff"], dtype=float).tolist()))
        return np.array([self.val[r] for r in rows])


def _narrow_mu_rho(e_idx: int) -> np.ndarray:
    return np.asarray(common.mu_rho(True)[:, e_idx], dtype=np.float64)


def requirement(mu_rho: np.ndarray, rows: np.ndarray, T: float, basis: str) -> np.ndarray:
    q = math.log(1.0 / T) / np.asarray(mu_rho, dtype=float)
    return q if basis == "areal" else q / common.density()[rows]


def kind_value(mask: np.ndarray, kind: str, e_idx: int, T: float, basis: str, cache: EffCache | None,
               art: np.ndarray) -> dict:
    """Narrow (cache None) or corrected value of one frontier kind over a row mask, with its certificate."""
    m = mask & ~art if kind == "best_clean" else mask
    k = 26 if kind == "median50" else 1
    idx = np.flatnonzero(m)
    qn = requirement(_narrow_mu_rho(e_idx)[idx], idx, T, basis)
    order = idx[np.argsort(qn, kind="stable")]
    qn_sorted = np.sort(qn, kind="stable")
    n = min(N0, order.size)
    while True:
        rows = order[:n]
        q = qn_sorted[:n] if cache is None else requirement(cache.get(rows), rows, T, basis)
        srt = np.argsort(q, kind="stable")
        kth = q[srt[k - 1]]
        nxt = qn_sorted[n] if n < order.size else np.inf
        if kth <= nxt or n == order.size:
            break
        n = min(2 * n, order.size)
    if kind == "median50":
        val = 0.5 * (q[srt[24]] + q[srt[25]])
    else:
        val = q[srt[0]]
    return {"value": float(val), "row": int(rows[srt[0]]), "n_candidates": int(n), "certified": bool(kth <= nxt)}


def price_cell(base: np.ndarray, km: np.ndarray, e_idx: int, T: float, basis: str, model: str | None,
               caches: dict, art: np.ndarray) -> dict:
    cache = None
    if model is not None:
        key = (e_idx, T, model)
        cache = caches.setdefault(key, EffCache(e_idx, T, model))
    out = {}
    for kind in KINDS:
        a = kind_value(base, kind, e_idx, T, basis, cache, art)
        b = kind_value(base & km, kind, e_idx, T, basis, cache, art)
        out[kind] = {"price": b["value"] / a["value"] - 1.0, "row_base": a["row"], "row_constrained": b["row"],
                     "certified": a["certified"] and b["certified"],
                     "n_candidates": max(a["n_candidates"], b["n_candidates"])}
    return out


# =============================================================================================
# (a) Pb mass-basis window
# =============================================================================================
def _contains(Z: int) -> np.ndarray:
    W = common.mass_fractions()
    return np.asarray((W[:, Z - 1] > 0).toarray()).ravel()


def pb_window(caches: dict) -> pd.DataFrame:
    u = common.universe()
    e = common.energy_keV()
    base = common.base_mask("R2")
    art = common.artefact_flag(np.arange(len(u)))
    km = ~_contains(82)
    pb_rows = np.flatnonzero(base & ~km & (u.nelements.to_numpy() == 1))
    recs = []
    for j in np.flatnonzero((e >= WINDOW_PB_KEV[0] - 1e-9) & (e <= WINDOW_PB_KEV[1] + 1e-9)):
        for T in T_LEVELS:
            for model in (None, *MODELS):
                res = price_cell(base, km, int(j), T, "areal", model, caches, art)
                # Pb's rank among F0 by corrected (or narrow) areal requirement, within the certified best set
                idx = np.flatnonzero(base)
                mr = _narrow_mu_rho(int(j))[idx]
                cand = idx[np.argsort(-mr, kind="stable")[:N0]]
                q = (requirement(mr[np.argsort(-mr, kind="stable")[:N0]], cand, T, "areal") if model is None
                     else requirement(caches[(int(j), T, model)].get(cand), cand, T, "areal"))
                srt = cand[np.argsort(q, kind="stable")]
                rank = {int(r): i + 1 for i, r in enumerate(srt)}
                pb_rank = min((rank.get(int(r), 10 ** 6) for r in pb_rows), default=None)
                for kind, v in res.items():
                    recs.append({"energy_keV": float(e[j]), "T": T, "model": model or "narrow", "kind": kind,
                                 **v, "best_base_formula": u.formula_pretty.iat[v["row_base"]],
                                 "best_constrained_formula": u.formula_pretty.iat[v["row_constrained"]],
                                 "pb_rank_in_F0_top200": pb_rank,
                                 "first_three_F0": ";".join(u.formula_pretty.to_numpy()[srt[:3]])})
    return pd.DataFrame(recs)


# =============================================================================================
# (b) load-bearing at absorption-edge windows (thickness)
# =============================================================================================
def lb_targets() -> pd.DataFrame:
    d = pd.read_parquet(common.P3 / "P" / "load_bearing_vs_energy.parquet")
    d = d[(d.rule == common.PRIMARY_RULE) & (d.basis == "thickness") & (d.energy_keV <= LB_MAX_KEV)]
    hit = d[d.price >= LB_THRESHOLD]
    best = hit.groupby(["element", "baseline", "energy_keV"]).price.max().reset_index()
    cells = (best.sort_values("price", ascending=False).groupby(["element", "baseline"]).head(LB_TOP)
             [["element", "baseline", "energy_keV"]])
    nb = d.pivot_table(index=["element", "baseline", "energy_keV"], columns="kind", values="price")
    nb.columns = [f"p5_{c}" for c in nb.columns]
    return cells.merge(nb.reset_index(), on=["element", "baseline", "energy_keV"], how="left") \
                .sort_values(["element", "baseline", "energy_keV"]).reset_index(drop=True)


def load_bearing(caches: dict, targets: pd.DataFrame) -> pd.DataFrame:
    u = common.universe()
    base0 = common.base_mask("R2")
    art = common.artefact_flag(np.arange(len(u)))
    b4 = base0 & common.bundle_mask("B4")
    sym = common.elements_table().set_index("symbol").Z
    recs = []
    for t in targets.itertuples():
        base = base0 if t.baseline == "F0" else b4
        km = ~_contains(int(sym[t.element]))
        j = common.energy_index(float(t.energy_keV))
        for T in T_LEVELS:
            for model in (None, *MODELS):
                res = price_cell(base, km, j, T, "thickness", model, caches, art)
                for kind, v in res.items():
                    recs.append({"element": t.element, "baseline": t.baseline, "energy_keV": float(t.energy_keV),
                                 "T": T, "model": model or "narrow", "kind": kind, **v,
                                 "best_base_formula": u.formula_pretty.iat[v["row_base"]],
                                 "best_constrained_formula": u.formula_pretty.iat[v["row_constrained"]],
                                 "p5_narrow_this_kind": getattr(t, f"p5_{kind}", np.nan)})
    return pd.DataFrame(recs)


# =============================================================================================
def _summ(a) -> dict:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def summarize(pbw: pd.DataFrame, lb: pd.DataFrame) -> dict:
    out = {"models": {"narrow": "Table C μ/ρ", **{k: (v or "Step 2 primary (E1, K + L)") for k, v in MODELS.items()}},
           "T_levels": list(T_LEVELS), "kinds": list(KINDS), "n0": N0,
           "all_certified": bool(pbw.certified.all() and lb.certified.all())}
    pa = {}
    for (model, T, kind), g in pbw.groupby(["model", "T", "kind"]):
        pos = g[g.price > 0]
        pa[f"{model}_T{T:g}_{kind}"] = {
            "price_max": float(g.price.max()), "energy_at_max": float(g.loc[g.price.idxmax(), "energy_keV"]),
            "energies_price_gt0": sorted(pos.energy_keV.tolist()),
            "pb_rank_in_F0_min": int(g.pb_rank_in_F0_top200.min()),
            "best_base_at_max": g.loc[g.price.idxmax(), "best_base_formula"]}
    out["pb_window"] = pa
    # check: narrow values reproduce the continuum for the cells it reports
    lbs = {}
    for (el, bl, model, T, kind), g in lb.groupby(["element", "baseline", "model", "T", "kind"]):
        i = g.price.idxmax()
        lbs[f"{el}|{bl}|{model}|T{T:g}|{kind}"] = {"max": float(g.price.max()), "energy_at_max": float(g.at[i, "energy_keV"]),
                                                   "constrained_best": g.at[i, "best_constrained_formula"],
                                                   "base_best": g.at[i, "best_base_formula"]}
    out["load_bearing_max"] = lbs
    nar = lb[(lb.model == "narrow") & (lb["T"] == T_LEVELS[0])]
    out["check_narrow_vs_step5_max_abs_diff"] = float(np.nanmax(np.abs(nar.price - nar.p5_narrow_this_kind)))
    return out


def run(verbose: bool = True) -> dict:
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()
    paths.ensure(OUT)
    from shieldmap.analysis3 import fluorescence as fl
    caches: dict = {}
    pf = OUT / "pb_window.csv"
    pbw = pd.read_csv(pf) if pf.exists() else None
    if pbw is None or "model_version" not in pbw or set(pbw.model_version) != {fl.MODEL_VERSION}:
        pbw = pb_window(caches)
        pbw["model_version"] = fl.MODEL_VERSION
        pbw.to_csv(pf, index=False)
    log(f"[E] Pb window: {len(pbw)} rows ({time.time() - t0:.0f} s)")
    tg = lb_targets()
    tg.to_csv(OUT / "load_bearing_targets.csv", index=False)
    lb = load_bearing(caches, tg)
    lb["model_version"] = fl.MODEL_VERSION
    lb.to_csv(OUT / "load_bearing_corrected.csv", index=False)
    log(f"[E] load-bearing: {len(tg)} cells, {len(lb)} rows ({time.time() - t0:.0f} s)")
    s = summarize(pbw, lb)
    common.write_json(OUT / "summary.json", s)
    common.write_json(OUT / "_timing.json", {"seconds": time.time() - t0})
    return s


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    r = run()
    print(json.dumps({k: r[k] for k in ("all_certified", "check_narrow_vs_step5_max_abs_diff")}, indent=1))
