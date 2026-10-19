"""Density bands of the B4-internal radionuclide prices.

The stage 3 data-conditioned density-error model (analysis3.density_model.RowModel, copula mode, ceiling
'unmeasured') is re-sampled for the stage 4a reference sets: R2, best_clean under the stage 4a
artefact flag, thickness basis, the eight Cs-137 / Co-60 Table C variants. Each draw scales every candidate
row's thickness by 1 + δ and recomputes both frontiers of each price. Candidates are the rows that can reach
the top 50 of any set under the model's δ bounds (density_model.candidates), so the frontiers are exact.
The band reported is the 5th–95th percentile over draws ("density band"); the measured-density point is the
headline (refsets, density = 'measured').
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import density_model as dm
from shieldmap.analysis4 import common, refsets

OUT = common.P4 / "S"
N_DRAWS = 2000
SEED = 20261012
MODE = "copula"
ELEMENTS = ("W", "Ir", "Os", "Bi", "Ta", "Hf", "Pt", "Au", "Re")


def run(n_draws: int = N_DRAWS) -> pd.DataFrame:
    paths.ensure(OUT)
    t0 = time.time()
    cols = refsets.columns()
    cols = cols[cols.scenario.isin(["CS137", "CO60"]) & cols.coherent].reset_index(drop=True)
    q = np.asarray(c3.profiles("thickness")[:, cols.variant_id.to_numpy()], dtype=np.float64)
    sets = refsets.practical_sets(elements=ELEMENTS, headroom=True)
    specs = {**refsets.marginal_specs(ELEMENTS), **{k: v for k, v in refsets.headroom_specs().items()
                                                    if k in ("RAD|F0", "B4|F0", "GAP(3.1)|F0", "B5|F0", "B6|F0")}}
    need = sorted({s for p in specs.values() for s in p})
    base = c3.base_mask(refsets.PRIMARY_RULE) & common.clean_mask()
    masks = {s: sets[s] & base for s in need}
    model = dm.RowModel()
    lo, hi = model.bounds(MODE)
    cand = dm.candidates(q, {("p", s): m for s, m in masks.items()}, lo, hi)
    qc = q[cand]
    mc = {s: m[cand] for s, m in masks.items()}
    rng = np.random.default_rng(SEED)
    vals = {lab: np.empty((n_draws, len(cols))) for lab in specs}
    for d in range(n_draws):
        dl = model.draw(cand, rng, MODE)
        qd = qc * (1.0 + dl[:, None])
        fr = {s: np.min(np.where(m[:, None], qd, np.inf), axis=0) for s, m in mc.items()}
        for lab, (num, den) in specs.items():
            vals[lab][d] = fr[num] / fr[den] - 1.0
    recs = []
    for lab, v in vals.items():
        for j, c in cols.iterrows():
            x = v[:, j]
            recs.append({"key": c.key, "scenario": c.scenario, "T_target": c.T_target, "label": lab,
                         "density_p05": float(np.quantile(x, 0.05)), "density_p50": float(np.quantile(x, 0.5)),
                         "density_p95": float(np.quantile(x, 0.95)), "n_draws": n_draws, "mode": MODE,
                         "n_candidates": int(cand.size)})
    out = pd.DataFrame(recs)
    out.to_csv(OUT / "density_bands.csv", index=False)
    common.write_json(OUT / "density_bands_meta.json", {"n_draws": n_draws, "seed": SEED, "mode": MODE,
                                                         "n_candidates": int(cand.size),
                                                         "seconds": round(time.time() - t0, 1)})
    return out


if __name__ == "__main__":
    print(run().head())
