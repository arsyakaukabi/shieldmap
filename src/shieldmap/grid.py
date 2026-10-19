"""The common photon energy grid (keV).

10–200 keV in 0.5 keV steps (bins of ±0.25 keV centred on each node, matching SpekPy at
dk = 0.25 keV), then ~150 log-spaced points to 15 MeV, plus the XCOM/SRD 126 standard energies
above 200 keV and the exact radionuclide lines. Edge energies are stored separately (Table A);
cross sections are evaluated *at* every node with XCOM's own scheme, so no value is ever
interpolated across an absorption edge.
"""
from __future__ import annotations

import numpy as np

from shieldmap import config


def key_lines_keV() -> list[float]:
    """Exact radionuclide gamma lines from the ENSDF source records (empty until they exist)."""
    lines = []
    for nuc in ("Cs-137", "Co-60"):
        try:
            for ln in config.get(f"radionuclides.{nuc}"):
                lines.append(float(ln["energy_keV"]))
        except (FileNotFoundError, KeyError):
            continue
    return lines


def energy_grid_keV(extra_lines: list[float] | None = None) -> np.ndarray:
    lo = config.get("energy_grid.linear_min_keV")
    mid = config.get("energy_grid.linear_max_keV")
    step = config.get("energy_grid.linear_step_keV")
    hi = config.get("energy_grid.log_max_keV")
    n_log = config.get("energy_grid.log_n_points")
    sep = config.get("energy_grid.min_separation_rel")

    n_lin = int(round((mid - lo) / step)) + 1
    linear = lo + step * np.arange(n_lin)
    log_pts = np.geomspace(mid, hi, n_log)[1:]          # 200 keV already in the linear part

    fixed = [e for e in config.get("energy_grid.anchor_energies_keV") if mid < e <= hi]
    fixed += [e for e in (key_lines_keV() if extra_lines is None else extra_lines) if mid < e <= hi]
    fixed = np.array(sorted(set(fixed)), dtype=float)
    if fixed.size:
        keep = np.array([np.min(np.abs(fixed / e - 1.0)) >= sep for e in log_pts])
        log_pts = log_pts[keep]
    grid = np.unique(np.concatenate([linear, log_pts, fixed]))
    return grid


def bin_edges_linear_keV() -> np.ndarray:
    """Edges of the ±step/2 bins around the linear-segment nodes (used for SpekPy mapping)."""
    lo = config.get("energy_grid.linear_min_keV")
    mid = config.get("energy_grid.linear_max_keV")
    step = config.get("energy_grid.linear_step_keV")
    n_lin = int(round((mid - lo) / step)) + 1
    return lo - step / 2 + step * np.arange(n_lin + 1)
