"""Detector-response functions R(E) on the energy grid.

air_kerma       R = E · (μ_en/ρ)_air      [PHYS]; μ_en/ρ from NIST SRD 126 (air, dry, near sea level)
fluence         R = 1
energy_fluence  R = E

(μ_en/ρ)_air between SRD 126 nodes is interpolated with a natural cubic spline in (ln E, ln μ_en/ρ)
over the nodes above the Ar K edge (3.2029 keV), i.e. the same scheme XCOM uses for smooth
quantities; the grid starts at 10 keV, so no edge is crossed. The log-log-linear alternative is
quantified in the report.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from shieldmap import config

AR_K_EDGE_KEV = 3.2029


def air_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(energy_keV, mu_rho, muen_rho) above the Ar K edge, ascending, de-duplicated."""
    rows = config.get("response.air_muen")
    e = np.array([float(r["energy_keV"]) for r in rows])
    mu = np.array([float(r["mu_rho"]) for r in rows])
    muen = np.array([float(r["muen_rho"]) for r in rows])
    # keep rows strictly above the edge, plus the above-edge row at the edge itself
    above = e > AR_K_EDGE_KEV * (1 + 1e-6)
    at_edge = np.where(np.isclose(e, AR_K_EDGE_KEV, rtol=1e-6))[0]
    keep = above.copy()
    if at_edge.size:
        keep[at_edge[-1]] = True                     # the second (above-edge) row
    e, mu, muen = e[keep], mu[keep], muen[keep]
    order = np.argsort(e, kind="stable")
    return e[order], mu[order], muen[order]


def air_muen_rho(e_keV: np.ndarray, method: str = "spline") -> np.ndarray:
    e_tab, _, muen = air_table()
    e_keV = np.asarray(e_keV, dtype=float)
    if np.any(e_keV < e_tab[0]) or np.any(e_keV > e_tab[-1]):
        raise ValueError("energy outside the SRD 126 air table")
    if method == "spline":
        sp = CubicSpline(np.log(e_tab), np.log(muen), bc_type="natural")
        return np.exp(sp(np.log(e_keV)))
    if method == "loglog_linear":
        return np.exp(np.interp(np.log(e_keV), np.log(e_tab), np.log(muen)))
    raise ValueError(method)


def response(name: str, e_keV: np.ndarray) -> np.ndarray:
    e_keV = np.asarray(e_keV, dtype=float)
    if name == "air_kerma":
        return e_keV * air_muen_rho(e_keV)
    if name == "fluence":
        return np.ones_like(e_keV)
    if name == "energy_fluence":
        return e_keV.copy()
    raise ValueError(f"unknown response {name!r}")
