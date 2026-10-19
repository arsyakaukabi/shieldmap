"""Source spectra and the energy quadrature used for polyenergetic transmission.

X-ray tubes: SpekPy (W anode), IEC 61267 RQR qualities from IAEA TRS-457 Table 6.2 [STD]. For each
quality and anode angle [SWEEP], the total Al filtration is adjusted (Brent) until SpekPy's first
HVL equals the standard HVL; the match must be within ``xray_tubes.hvl_match_tolerance_rel``.

Quadrature: SpekPy's spectrum is piecewise constant on 0.25 keV cells
[9.75 + 0.25 i, 10.0 + 0.25 i]. Each cell is split at every absorption edge of every element
(Z = 1–100, XCOM edge energies) that falls inside it. Each piece takes the fluence of its share of
the cell, and μ is evaluated at the piece midpoint. A piece therefore never straddles an edge, and
no attenuation coefficient is used on the wrong side of an edge. The cells are aligned with the
common grid: two cells make up the 0.5 keV bin centred on each grid node. ``map_to_grid`` (the
plain grid mapping of) is kept for diagnostics. Fluence below 9.75 keV is dropped and its
share is recorded.

Radionuclides: ENSDF gamma lines [STD] with their absolute emission intensities per decay, at
energies that are also exact grid nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from shieldmap import config, grid, paths
from shieldmap.physics import response

CELL0_KEV = 9.75                     # lower edge of the first grid bin (10 keV − step/2)


@dataclass
class Spectrum:
    name: str                       # e.g. "RQR5_th12" or "Cs-137"
    kind: str                       # "tube" | "line"
    energy_keV: np.ndarray          # quadrature node energies
    fluence: np.ndarray             # photons per node (relative)
    meta: dict = field(default_factory=dict)
    lo_keV: np.ndarray | None = None   # piece bounds (tubes)
    hi_keV: np.ndarray | None = None

    def weights(self, resp: str) -> np.ndarray:
        return self.fluence * response.response(resp, self.energy_keV)


# ----------------------------------------------------------------------------------------------
# radionuclides
# ----------------------------------------------------------------------------------------------
def line_spectrum(nuclide: str) -> Spectrum:
    lines = config.get(f"radionuclides.{nuclide}")
    e_grid = grid.energy_grid_keV()
    energies, inten = [], []
    for ln in lines:
        e = float(ln["energy_keV"])
        if not np.isclose(e_grid, e, rtol=0, atol=1e-9).any():
            raise ValueError(f"{nuclide} line {e} keV is not an exact grid node")
        energies.append(e); inten.append(float(ln["intensity_per_decay"]))
    return Spectrum(name=nuclide, kind="line", energy_keV=np.array(energies), fluence=np.array(inten),
                    meta={"lines_keV": energies, "intensity_per_decay": inten,
                          "source": config.leaf(f"radionuclides.{nuclide}")["ref"]})


# ----------------------------------------------------------------------------------------------
# quadrature
# ----------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def all_edges_keV() -> np.ndarray:
    e = pd.read_parquet(paths.INTERIM / "elements" / "edges.parquet")["energy_keV"].to_numpy()
    return np.unique(e)


def quadrature(kvp: float, dk: float, subdivide: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pieces of the SpekPy cells from 9.75 keV to kvp, split at all element edges.

    Returns (cell index, lo, hi, mid). ``subdivide`` > 1 additionally cuts every cell into equal
    parts (convergence checks only).
    """
    n_cells = int(round((kvp - CELL0_KEV) / dk))
    edges = all_edges_keV()
    cells, lo, hi = [], [], []
    for i in range(n_cells):
        a = CELL0_KEV + dk * i; b = a + dk
        cuts = list(np.linspace(a, b, subdivide + 1))
        inner = edges[(edges > a + 1e-9) & (edges < b - 1e-9)]
        cuts = np.unique(np.concatenate([cuts, inner]))
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            cells.append(i); lo.append(x0); hi.append(x1)
    lo = np.array(lo); hi = np.array(hi)
    return np.array(cells), lo, hi, 0.5 * (lo + hi)


# ----------------------------------------------------------------------------------------------
# x-ray tubes
# ----------------------------------------------------------------------------------------------
def _spek(kvp: float, angle: float, al_mm: float):
    import spekpy as sp
    s = sp.Spek(kvp=kvp, th=angle, dk=config.get("xray_tubes.spekpy_dk_keV"),
                targ=config.get("xray_tubes.anode"))
    if al_mm > 0:
        s.filter("Al", al_mm)
    return s


def match_filtration(kvp: float, angle: float, hvl_target_mm: float) -> tuple[float, object]:
    """Total Al filtration (mm) for which SpekPy's first HVL (air kerma, in Al) equals the target."""
    def f(t):
        return _spek(kvp, angle, t).get_hvl1() - hvl_target_mm
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
        if hi > 64:
            raise RuntimeError(f"cannot reach HVL {hvl_target_mm} mm Al at {kvp} kV")
    t = brentq(f, 0.0, hi, xtol=1e-6, rtol=1e-10)
    return t, _spek(kvp, angle, t)


def _cells_from_spekpy(k: np.ndarray, phi: np.ndarray, dk: float, kvp: float) -> np.ndarray:
    """Photons per 0.25 keV cell, cell i = [9.75 + i dk, 9.75 + (i+1) dk]."""
    n_cells = int(round((kvp - CELL0_KEV) / dk))
    idx = np.round((k - CELL0_KEV - dk / 2) / dk).astype(int)
    if not np.allclose(CELL0_KEV + dk * (idx + 0.5), k, atol=1e-9):
        raise ValueError("SpekPy bin centres are not aligned with the grid cells")
    ok = (idx >= 0) & (idx < n_cells)
    out = np.zeros(n_cells)
    out[idx[ok]] = phi[ok]
    if np.any(phi[idx >= n_cells] > 0):
        raise ValueError("SpekPy fluence above kVp")
    return out


def dropped_fractions(k: np.ndarray, phi: np.ndarray) -> dict:
    below = k < CELL0_KEV
    kerma = phi * k * _air_muen_any(k)
    return {"dropped_fluence_fraction": float(phi[below].sum() / phi.sum()),
            "dropped_air_kerma_fraction": float(kerma[below].sum() / kerma.sum())}


def tube_spectrum(quality: dict, angle: float, subdivide: int = 1) -> Spectrum:
    kvp = float(quality["kV"]); target = float(quality["hvl1_mm_al"])
    dk = config.get("xray_tubes.spekpy_dk_keV")
    t_al, s = match_filtration(kvp, angle, target)
    hvl1 = float(s.get_hvl1()); hvl2 = float(s.get_hvl2())
    tol = config.get("xray_tubes.hvl_match_tolerance_rel")
    if abs(hvl1 / target - 1) > tol:
        raise RuntimeError(f"HVL match failed for {quality['name']} at {angle} deg")
    k, phi = s.get_spectrum(diff=False)          # photons per SpekPy bin
    k = np.asarray(k, dtype=float); phi = np.asarray(phi, dtype=float)
    cell_phi = _cells_from_spekpy(k, phi, dk, kvp)
    cell, lo, hi, mid = quadrature(kvp, dk, subdivide)
    flu = cell_phi[cell] * (hi - lo) / dk            # SpekPy fluence is uniform within a cell
    keep = flu > 0
    name = f"{quality['name'].replace(' ', '')}_th{angle:g}"
    meta = {"quality": quality["name"], "kV": kvp, "anode_angle_deg": angle,
            "total_al_filtration_mm": float(t_al), "hvl1_target_mm_al": target,
            "spekpy_hvl1_mm_al": hvl1, "spekpy_hvl2_mm_al": hvl2,
            "spekpy_homogeneity": hvl1 / hvl2, "standard_homogeneity": float(quality["homogeneity_h"]),
            "spekpy_mean_energy_keV": float(s.get_emean()),
            "spekpy_physics": s.state.model_parameters.physics,
            "spekpy_mu_data_source": s.state.model_parameters.mu_data_source,
            "n_cells": int(cell_phi.size), "n_pieces": int(keep.sum()), **dropped_fractions(k, phi)}
    sp = Spectrum(name=name, kind="tube", energy_keV=mid[keep], fluence=flu[keep], meta=meta,
                  lo_keV=lo[keep], hi_keV=hi[keep])
    sp._spek = s                                   # kept for diagnostics (not serialised)
    sp._k, sp._phi = k, phi
    return sp


def map_to_grid(k: np.ndarray, phi: np.ndarray, e_grid_keV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic only: SpekPy bins summed into the 0.5 keV bins centred on grid nodes."""
    edges = grid.bin_edges_linear_keV()
    step = config.get("energy_grid.linear_step_keV")
    inside = (k > edges[0]) & (k < edges[-1])
    pos = np.floor((k[inside] - edges[0]) / step).astype(int)
    acc = np.bincount(pos, weights=phi[inside], minlength=edges.size - 1)
    keep = acc > 0
    return np.arange(edges.size - 1)[keep], acc[keep]


def _air_muen_any(e_keV: np.ndarray) -> np.ndarray:
    """(μ_en/ρ)_air at any energy 1 keV–20 MeV, log-log linear on the correct side of the Ar K edge.
    Used only to report the air-kerma share dropped below the grid."""
    rows = config.get("response.air_muen")
    e = np.array([float(r["energy_keV"]) for r in rows]); m = np.array([float(r["muen_rho"]) for r in rows])
    edge = response.AR_K_EDGE_KEV
    first_edge = np.where(np.isclose(e, edge, rtol=1e-6))[0][0]
    eb, mb = e[:first_edge + 1], m[:first_edge + 1]
    ea, ma = e[first_edge + 1:], m[first_edge + 1:]
    out = np.empty_like(e_keV, dtype=float)
    lo_side = e_keV < edge
    out[lo_side] = np.exp(np.interp(np.log(e_keV[lo_side]), np.log(eb), np.log(mb)))
    out[~lo_side] = np.exp(np.interp(np.log(e_keV[~lo_side]), np.log(ea), np.log(ma)))
    return out


def all_spectra() -> list[Spectrum]:
    out = []
    for q in config.get("xray_tubes.qualities"):
        for angle in config.get("xray_tubes.anode_angle_deg"):
            out.append(tube_spectrum(q, float(angle)))
    for nuc in ("Cs-137", "Co-60"):
        out.append(line_spectrum(nuc))
    return out
