"""Mixture rule: μ/ρ(E) = Σ_i w_i (μ/ρ)_i(E), as one sparse × dense matrix product."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from shieldmap import paths

ELEM = paths.INTERIM / "elements"
PARTIAL_KEYS = ("coh", "incoh", "pe", "pair_nuc", "pair_el", "total_wcoh", "total_wocoh")


def load_element_xsec() -> dict[str, np.ndarray]:
    with np.load(ELEM / "xsec_grid.npz") as z:
        return {k: z[k] for k in z.files}


def mu_rho(mass_fractions: sps.csr_matrix | np.ndarray, elem_mu_rho: np.ndarray) -> np.ndarray:
    """(N × 100) mass fractions @ (100 × n_E) element μ/ρ → (N × n_E) compound μ/ρ (cm²/g)."""
    w = mass_fractions
    if sps.issparse(w):
        return np.asarray(w @ elem_mu_rho)
    return np.asarray(w) @ elem_mu_rho


def fractions_from_formula(formula: str) -> np.ndarray:
    """Mass-fraction row vector (100,) from a chemical formula, with pymatgen atomic masses."""
    from pymatgen.core import Composition
    comp = Composition(formula)
    w = np.zeros(100)
    for el in comp.elements:
        w[el.Z - 1] = comp.get_wt_fraction(el)
    return w


_XCOM_CACHE: dict = {}


def element_mu_rho_at(e_keV: np.ndarray, keys=("total_wcoh", "total_wocoh")) -> dict[str, np.ndarray]:
    """Element μ/ρ (100 × n, cm²/g) evaluated by the XCOM port directly at arbitrary energies."""
    if "els" not in _XCOM_CACHE:
        from shieldmap.ingest import xsec
        _XCOM_CACHE["els"] = xsec.load_elements(paths.RAW / "xcom" / "extracted" / "XCOM")
    els = _XCOM_CACHE["els"]
    e_ev = np.asarray(e_keV, dtype=float) * 1000.0
    out = {k: np.zeros((100, e_ev.size)) for k in keys}
    for z in range(1, 101):
        mr = els[z].mu_rho(e_ev)
        for k in keys:
            out[k][z - 1] = mr[k]
    return out
