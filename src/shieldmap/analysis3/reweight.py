"""The composition/density crossover energy under population reweighting.

The stage 2/3 crossover (235.1 keV, R2, with coherent) is the energy where the variance of log10 μ/ρ
(composition) equals the variance of log10 ρ (density) over the eligible Materials Project population.
The variance of log ρ is a property of that population, so the crossing depends on which compounds the
map contains and how each is weighted. This module recomputes the crossing for
reweighted and restricted populations, with the same population-moment decomposition as stage 2
(analysis.landscape.decompose), using weights:

  all              every eligible R2 representative, weight 1 (reproduces 235.1 keV)
  icsd_only        ICSD-matched entries only (theoretical == False)
  stable_0.05      e_hull ≤ 0.05 eV/atom
  per_chemsys      weight 1/n(chemical system): each chemical system counts once
  per_heaviest     weight 1/n(heaviest element): each heaviest element counts once
  per_nelements    weight 1/n(number of elements): unaries, binaries, ternaries … count equally
  no_oxygen        compounds without oxygen
  oxides_only      compounds containing oxygen

Outputs: data/phase3/RW/{decomposition.parquet, crossings.csv, summary.json}
Run: python -m shieldmap.analysis3.reweight
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common
from shieldmap.analysis3.landscape3 import log_crossing

OUT = common.P3 / "RW"
BLOCK = 64


def populations() -> dict[str, np.ndarray]:
    """name → weight vector over R2 eligible rows (zero weight = excluded)."""
    u = common.universe()
    m = u.eligible.to_numpy(bool) & u.in_R2.to_numpy(bool)
    rows = np.flatnonzero(m)
    s = u.iloc[rows]
    z = dict(zip(common.elements_table().symbol, common.elements_table().Z))
    heaviest = s.elements.map(lambda e: max(e, key=lambda x: z[x])).to_numpy()
    has_o = s.elements.map(lambda e: "O" in set(e)).to_numpy(bool)

    def inv_count(keys):
        k = pd.Series(keys)
        return (1.0 / k.map(k.value_counts())).to_numpy(float)
    w = {"all": np.ones(rows.size),
         "icsd_only": (~s.theoretical.to_numpy(bool)).astype(float),
         "stable_0.05": (s.energy_above_hull.to_numpy(float) <= 0.05).astype(float),
         "per_chemsys": inv_count(s.chemsys.to_numpy()),
         "per_heaviest": inv_count(heaviest),
         "per_nelements": inv_count(s.nelements.to_numpy()),
         "no_oxygen": (~has_o).astype(float),
         "oxides_only": has_o.astype(float)}
    return {"rows": rows, **{k: v / v.sum() for k, v in w.items()}}


def weighted_decomposition(x: np.ndarray, Y: np.ndarray, w: np.ndarray) -> dict:
    """Weighted population moments of log ρ (x), log μ/ρ (Y columns) and their covariance."""
    keep = w > 0
    x, Y, w = x[keep], Y[keep], w[keep] / w[keep].sum()
    xc = x - w @ x
    Yc = Y - w @ Y
    vx = float(w @ xc ** 2)
    vy = w @ Yc ** 2
    cv = (w * xc) @ Yc
    vmu = vx + vy + 2 * cv
    n_eff = 1.0 / float((w ** 2).sum())
    return {"n": int(keep.sum()), "n_eff": n_eff, "var_logrho": np.full(Y.shape[1], vx), "var_logmurho": vy,
            "cov": cv, "share_density": vx / vmu, "share_composition": vy / vmu, "share_cov": 2 * cv / vmu}


def decomposition(coherent: bool = True) -> pd.DataFrame:
    pops = populations()
    rows = pops.pop("rows")
    x = np.log10(common.density()[rows])
    e = common.energy_keV()
    mr = common.mu_rho(coherent)
    parts = []
    for j0 in range(0, e.size, BLOCK):
        j1 = min(e.size, j0 + BLOCK)
        Y = np.log10(np.asarray(mr[rows, j0:j1], dtype=np.float64))
        for name, w in pops.items():
            d = weighted_decomposition(x, Y, w)
            df = pd.DataFrame({k: v for k, v in d.items() if k not in ("n", "n_eff")})
            df.insert(0, "energy_keV", e[j0:j1]); df.insert(1, "population", name)
            df.insert(2, "coherent", coherent); df["n"] = d["n"]; df["n_eff"] = d["n_eff"]
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def crossings(dec: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (pop, coh), d in dec.groupby(["population", "coherent"]):
        d = d.sort_values("energy_keV")
        diff = (d.var_logmurho - d.var_logrho).to_numpy()
        for direction in ("down", "up"):
            for ec in log_crossing(d.energy_keV.to_numpy(), diff, direction):
                recs.append({"population": pop, "coherent": coh, "direction": direction, "E_cross_keV": ec,
                             "n": int(d.n.iloc[0]), "n_eff": float(d.n_eff.iloc[0]),
                             "var_logrho": float(d.var_logrho.iloc[0])})
    return pd.DataFrame(recs)


def run() -> dict:
    paths.ensure(OUT)
    dec = pd.concat([decomposition(True), decomposition(False)], ignore_index=True)
    dec.to_parquet(OUT / "decomposition.parquet", index=False)
    cr = crossings(dec)
    cr.to_csv(OUT / "crossings.csv", index=False)
    main = cr[(cr.direction == "down") & (cr.E_cross_keV > 100)]
    by = {}
    for (pop, coh), g in main.groupby(["population", "coherent"]):
        by[f"{pop}|{'wcoh' if coh else 'wocoh'}"] = {"E_cross_keV": [round(float(v), 2) for v in g.E_cross_keV],
                                                    "n": int(g.n.iloc[0]), "n_eff": round(float(g.n_eff.iloc[0]), 1),
                                                    "var_logrho": float(g.var_logrho.iloc[0])}
    wc = main[main.coherent].groupby("population").E_cross_keV.max()
    summ = {"crossing_main_keV": by,
            "range_wcoh_keV": [float(wc.min()), float(wc.max())],
            "all_crossings_any_direction": int(len(cr)),
            "note": "main crossing = last downward crossing of var(log μ/ρ) − var(log ρ) above 100 keV, "
                    "R2 eligible rows, population moments with the stated weights"}
    common.write_json(OUT / "summary.json", summ)
    return summ


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1))
