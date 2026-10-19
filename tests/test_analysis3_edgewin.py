"""K-edge windows under the fluorescence correction (analysis3.edgewin)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis3 import common, edgewin as E

P5 = common.P3 / "P" / "load_bearing_vs_energy.parquet"


@pytest.mark.skipif(not P5.exists(), reason="Step 5 output not present")
def test_narrow_load_bearing_reproduces_step5():
    """Narrow-beam (no fluorescence) prices from edgewin equal the continuum module's load-bearing prices."""
    d = pd.read_parquet(P5)
    d = d[(d.rule == "R2") & (d.basis == "thickness") & (d.baseline == "F0") & (d.element == "Ta")
          & (d.energy_keV == 67.5)].set_index("kind").price
    u = common.universe()
    art = common.artefact_flag(np.arange(len(u)))
    base = common.base_mask("R2")
    km = ~E._contains(73)
    res = E.price_cell(base, km, common.energy_index(67.5), 0.1, "thickness", None, {}, art)
    for kind in ("best", "median50", "best_clean"):
        if kind in d.index:
            # The continuum module evaluates the float32 Table C in its own order: agreement to float32 precision
            assert res[kind]["price"] == pytest.approx(float(d[kind]), rel=1e-7, abs=1e-12)
        assert res[kind]["certified"]


def test_certificate_grows_candidate_set_until_safe():
    """kind_value doubles N until the k-th corrected value is ≤ the first excluded narrow value."""
    class Fake:
        def get(self, rows):          # corrected attenuation 30 % lower for the 300 narrow-best rows
            mr = E._narrow_mu_rho(j)[rows]
            return np.where(np.isin(rows, top300), 0.7 * mr, mr)
    j = common.energy_index(100.0)
    base = common.base_mask("R2")
    art = np.zeros(base.size, bool)
    idx = np.flatnonzero(base)
    q = 1.0 / E._narrow_mu_rho(j)[idx]
    top300 = idx[np.argsort(q)[:300]]
    r = E.kind_value(base, "best", j, 0.1, "areal", Fake(), art)
    assert r["certified"] and r["n_candidates"] > 300
    assert r["row"] not in set(top300.tolist())
