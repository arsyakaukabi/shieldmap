"""The lead-free price at the Pb K-edge window against the B4-internal reference,
narrow beam and under the stage 3 first-order fluorescence correction.

Stage 3 (analysis3.edgewin) priced lead-free against F0. Its corrected F0 frontier (YbHgPb, e_hull 0.121 eV/atom)
is not in "B4 without PB", so that number does not carry over to the stage 4a reference. Here the same
certified routine (edgewin.price_cell: candidates in narrow-beam order, doubled until no row outside can enter
the statistic) is applied with base = R2 ∧ (B4 without PB), constraint = no Pb, areal mass, every grid node
85–95 keV, T = 0.1 and 0.01, models narrow / E₁ (primary) / E₂, and the phase4a artefact flag.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import edgewin as ew
from shieldmap.analysis4 import common, refsets

OUT = common.P4 / "S"


def run(verbose: bool = True) -> pd.DataFrame:
    from shieldmap.analysis3 import fluorescence as fl
    t0 = time.time()
    paths.ensure(OUT)
    u = c3.universe()
    e = c3.energy_keV()
    sets = refsets.practical_sets(headroom=False)
    base = c3.base_mask("R2") & sets["B4-PB"]
    km = ~ew._contains(82)
    art = common.artefact_mask()
    caches: dict = {}
    recs = []
    for j in np.flatnonzero((e >= ew.WINDOW_PB_KEV[0] - 1e-9) & (e <= ew.WINDOW_PB_KEV[1] + 1e-9)):
        for T in ew.T_LEVELS:
            for model in (None, *ew.MODELS):
                res = ew.price_cell(base, km, int(j), T, "areal", model, caches, art)
                for kind, v in res.items():
                    recs.append({"energy_keV": float(e[j]), "T": T, "model": model or "narrow", "kind": kind, **v,
                                 "base_formula": u.formula_pretty.iat[v["row_base"]],
                                 "constrained_formula": u.formula_pretty.iat[v["row_constrained"]],
                                 "base_mpid": u.material_id.iat[v["row_base"]],
                                 "constrained_mpid": u.material_id.iat[v["row_constrained"]]})
        if verbose:
            print(f"  [edge4] {e[j]:g} keV  {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame(recs)
    out["model_version"] = fl.MODEL_VERSION
    out.to_csv(OUT / "pb_window_b4.csv", index=False)
    return out


if __name__ == "__main__":
    print(run().groupby(["model", "kind"]).price.max())
