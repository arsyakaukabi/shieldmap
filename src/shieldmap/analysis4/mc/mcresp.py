"""Air-kerma response for the MC tallies, identical to the map's response above the Ar K edge.

R(E) = E · (μ_en/ρ)_air (keV cm²/g):
* E ≥ 3.2029 keV: ``shieldmap.physics.response.response("air_kerma", E)`` (SRD 126, natural cubic spline in
  log-log), i.e. exactly the function Table C and the stage 3 fluorescence model use;
* 1 keV ≤ E < 3.2029 keV (only fluorescence lines below the Ar K edge, mostly M lines, and a negligible
  scattered continuum reach this range; the map never needs it): log-log linear interpolation of the same SRD
  126 air table between its nodes below the Ar K edge (the below-edge node is used at the edge). These photons
  are reported separately, because the map's first-order model omits them by construction (L_CUTOFF 3.2029 keV)
  and any cover or a few cm of air removes them.
"""
from __future__ import annotations

import numpy as np

from shieldmap import config
from shieldmap.physics import response as resp

AR_K = resp.AR_K_EDGE_KEV


def _below_table():
    rows = config.get("response.air_muen")
    e = np.array([float(r["energy_keV"]) for r in rows])
    muen = np.array([float(r["muen_rho"]) for r in rows])
    keep = e <= AR_K * (1 + 1e-6)
    e, muen = e[keep], muen[keep]
    # at the edge keep the first (below-edge) row
    _, first = np.unique(e, return_index=True)
    return e[first], muen[first]


def air_kerma(e_keV) -> np.ndarray:
    e = np.asarray(e_keV, dtype=float)
    out = np.zeros_like(e)
    hi = e >= AR_K
    if hi.any():
        out[hi] = resp.response("air_kerma", e[hi])
    lo = ~hi & (e >= 1.0)
    if lo.any():
        eb, mb = _below_table()
        out[lo] = e[lo] * np.exp(np.interp(np.log(e[lo]), np.log(eb), np.log(mb)))
    return out


def kerma_tab(n: int = 4000) -> dict:
    """Fine log grid of R(E) for OpenMC's EnergyFunctionFilter (log-log interpolation; cross-check tally)."""
    e = np.unique(np.concatenate([np.geomspace(1.0, AR_K * (1 - 1e-9), 200),
                                  np.geomspace(AR_K, 2.0e4, n)]))
    return {"energy_eV": (e * 1e3).tolist(), "R": air_kerma(e).tolist()}
