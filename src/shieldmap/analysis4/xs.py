"""Cross-section library sensitivity (xraylib in place of XCOM). SENSITIVITY ONLY.

Nothing computed here is written to the map, to Table C or to any stage 1–3 file; outputs go to
``data/phase4a/xs/`` and ``reports/phase4a/step7_xs.md`` only. Regenerate with
``python -m shieldmap.analysis4.xs`` from ``shieldmap/`` (offline, deterministic; xraylib 4.3.0).

Libraries (element μ/ρ with coherent scattering, cm²/g; ``element_table``)
  xcom        stage 1 primary: the XCOM port (grid nodes: interim xsec_grid.npz; arbitrary energies:
              physics.mixture.element_mu_rho_at, exactly as Table C does at the spectral quadrature energies)
  xrl_total   xraylib CS_Total = CS_Photo + CS_Rayl + CS_Compt (xraylib's own interpolated tables), Z = 1–98;
              Es, Fm (Z = 99, 100) keep XCOM (xraylib raises). Its photo/Rayleigh/Compton terms reproduce XCOM to
              ~1e-4 for most Z (it is a re-tabulation of the same Scofield/Hubbell data), so this variant
              tests interpolation and the few elements where xraylib's tables differ (Mo, Tc, Po, Rn, Pu–Cf).
  xrl_kissel  PRIMARY xraylib variant: CS_Total_Kissel = CS_Photo_Total (Kissel partial-shell photoionization)
              + CS_Rayl + CS_Compt for Z ≥ KISSEL_ZMIN (26); CS_Total for Z < 26; XCOM for Z = 99, 100.
              Below Z = 26 xraylib 4.3.0's Kissel tables end inside 20–200 keV and their extrapolation
              collapses (CS_Photo_Total/CS_Photo → 0; e.g. O at 40 keV 0.03, S at 100 keV 0.44), so they are
              not used there; for Z ≥ 26 the Kissel tables cover the whole range (checked per run: the ratio
              to CS_Photo stays within KISSEL_CHECK on every node used, edge-adjacent nodes listed). Inside the
              Kissel "edge-step zones" (kissel_step_zones: a plateau just above a K/L edge ending in a non-physical
              2–7 % step, where Kissel's theoretical edge lies above xraylib's experimental one) CS_Total is used.
  xrl_kissel_raw  as xrl_kissel but keeping the edge-step zones (reported alongside; shows what the zones do).
  Pair production is absent from xraylib's totals; every energy used here is ≤ 200 keV, where it is zero.

Items
  (a) PB|B4 at every grid node 85–95 keV, areal mass, narrow beam: P = q*(B4)/q*(B4 without PB) − 1.
  (b) DIAG lanthanide advantage on the 36 primary DIAG variants (areal, air kerma, narrow beam):
      (best non-Pb textbook material) / (best R2 ∧ B4 lanthanide compound) − 1, every R2 ∧ B4 lanthanide row
      recomputed (no candidate restriction), T* fixed from XCOM Pb (as the map) and like-for-like (T*
      recomputed with the same library's Pb). Fluorescence-corrected: FIRST-ORDER ESTIMATE only — each row's
      Stage 3 corrected areal mass × (its narrow requirement under the library / under XCOM).
  (c) EXCL(X)|B4 for X ∈ {W, Ta, Hf}: P = q*(B4 ∧ no X)/q*(B4) − 1 at every grid node 20–200 keV; areal mass and
      thickness (at the stage 3 measured-density point, as the headline table; MP densities alongside).
Validation first: the XCOM path must reproduce the refsets bands (prices and frontier compounds), Table C
DIAG profiles (float32 storage precision) and the stage 3 textbook comparison; run() raises otherwise.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from shieldmap import config, paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis3 import continuum
from shieldmap.analysis4 import common, refsets

OUT = common.P4 / "xs"
REPORT = common.REPORTS4 / "step7_xs.md"
LIBS = ("xcom", "xrl_total", "xrl_kissel", "xrl_kissel_raw")
XRL_LIBS = ("xrl_kissel", "xrl_kissel_raw", "xrl_total")
KISSEL_LIBS = ("xrl_kissel", "xrl_kissel_raw")
PRIMARY_XRL = "xrl_kissel"
LIB_TEXT = {"xcom": "XCOM (map)", "xrl_total": "xraylib CS_Total",
            "xrl_kissel": "xraylib Kissel (CS_Total_Kissel, Z ≥ 26, edge-step zones → CS_Total)",
            "xrl_kissel_raw": "xraylib Kissel raw (CS_Total_Kissel, Z ≥ 26, edge-step zones kept)"}
KISSEL_ZMIN = 26
KISSEL_CHECK = (0.90, 1.10)          # Kissel/CS_Photo on the nodes used, Z ≥ 26 (edge-adjacent nodes reported)
STEP_SHELLS = ("K", "L1", "L2", "L3")
STEP_SCAN_KEV = (0.005, 1.0)         # scan step and range above each edge for the Kissel edge-step search
STEP_DROP = 0.01                     # a fall > 1 % within 5 eV above an edge is not physical (smooth E^-~3 is ≤ 0.2 %)
XRL_ZMAX = 98
E_MONO = (20.0, 200.0)
PB_WINDOW = (85.0, 95.0)
HEAD_WINDOW = (88.5, 90.5)           # the headline table's window (H55)
EDGE_ELEMENTS = ("W", "Ta", "Hf")
TEXTBOOK_EXCL = ("Pb", "PbO")
# validation tolerances (never loosened to make a check pass)
TOL_PRICE_ABS = 1e-6                 # absolute price difference vs refsets bands (float32 map vs float64 path)
TOL_PROFILE_REL = 2.5e-7             # Table C profiles are stored as float32 (2^-23 ≈ 1.2e-7 per rounding)
TOL_TSTAR_REL = 1e-12
TOL_ADV_ABS = 1e-6


# ---------------------------------------------------------------------------------------------
# element tables
# ---------------------------------------------------------------------------------------------
def _xrl():
    import xraylib
    return xraylib


def xraylib_provenance() -> dict:
    """Installed xraylib version and SHA-256 of its files, checked against config/sources/xraylib.yaml."""
    import yaml
    xr = _xrl()
    rec = yaml.safe_load((paths.ROOT / "config" / "sources" / "xraylib.yaml").read_text(encoding="utf-8"))
    d = Path(xr.__file__).parent
    got, ok = {}, True
    for name, sha in rec["package"]["files"].items():
        p = d / name
        got[name] = c3.sha256(p) if p.exists() else "missing"
        ok &= got[name] == sha
    return {"version": getattr(xr, "__version__", "?"), "expected_version": rec["package"]["version"],
            "files_sha256": got, "matches_source_record": bool(ok)}


def _xcom_at(e_keV: np.ndarray, grid_idx: np.ndarray | None) -> np.ndarray:
    if grid_idx is not None:
        return np.asarray(c3.element_xsec()["total_wcoh"][:, grid_idx], dtype=np.float64)
    from shieldmap.physics import mixture
    return mixture.element_mu_rho_at(np.asarray(e_keV, float), keys=("total_wcoh",))["total_wcoh"]


def element_table(lib: str, e_keV: np.ndarray, grid_idx: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """(100 × n) element μ/ρ with coherent (cm²/g) under ``lib`` and a record of every XCOM fallback.

    ``grid_idx`` (indices into the 545-node grid) makes the XCOM table the stage 1 grid table exactly."""
    e_keV = np.asarray(e_keV, dtype=float)
    xc = _xcom_at(e_keV, grid_idx)
    if lib == "xcom":
        return xc, {}
    if lib not in XRL_LIBS:
        raise ValueError(lib)
    xr = _xrl()
    out = xc.copy()
    fallback: dict[int, int] = {}
    zones = kissel_step_zones() if lib == "xrl_kissel" else {}
    for z in range(1, 101):
        if z > XRL_ZMAX:
            fallback[z] = int(e_keV.size); continue
        kissel = lib in KISSEL_LIBS and z >= KISSEL_ZMIN
        zz = zones.get(z, [])
        for j, E in enumerate(e_keV):
            use_k = kissel and not any(lo <= E < hi for lo, hi, _ in zz)
            try:
                out[z - 1, j] = xr.CS_Total_Kissel(z, float(E)) if use_k else xr.CS_Total(z, float(E))
            except ValueError:
                try:
                    out[z - 1, j] = xr.CS_Total(z, float(E))
                except ValueError:
                    out[z - 1, j] = xc[z - 1, j]
                    fallback[z] = fallback.get(z, 0) + 1
    return out, fallback


@lru_cache(maxsize=1)
def kissel_step_zones() -> dict[int, list[tuple[float, float, str]]]:
    """Per Z ≥ KISSEL_ZMIN: energy intervals [edge, E_step) just above the K/L1/L2/L3 edges in which xraylib's
    Kissel partial cross section is not tabulated data but a plateau that ends in a non-physical step down
    (> STEP_DROP within one 5 eV scan step; e.g. Ta K: +6 % over XCOM from 67.416 to 67.6 keV, then −0.7 %).
    The step sits where Kissel's own (theoretical) edge lies above xraylib's experimental edge energy. Inside a
    zone the primary Kissel variant uses CS_Total for that element (same rule as for Z < 26: Kissel only where
    its tables carry data)."""
    xr = _xrl()
    shells = {"K": xr.K_SHELL, "L1": xr.L1_SHELL, "L2": xr.L2_SHELL, "L3": xr.L3_SHELL}
    out: dict[int, list] = {}
    step, rng = STEP_SCAN_KEV
    for z in range(KISSEL_ZMIN, XRL_ZMAX + 1):
        for name in STEP_SHELLS:
            try:
                k = xr.EdgeEnergy(z, shells[name])
            except ValueError:
                continue
            if k <= 0:
                continue
            E = k + step * np.arange(1, int(round(rng / step)) + 1)
            v = np.full(E.size, np.nan)
            for i, e in enumerate(E):
                try:
                    v[i] = xr.CS_Photo_Partial(z, shells[name], float(e))
                except ValueError:
                    pass
            r = v[1:] / v[:-1] - 1.0
            if not np.isfinite(r).any():
                continue
            i = int(np.nanargmin(r))
            if r[i] < -STEP_DROP:
                out.setdefault(z, []).append((float(k), float(E[i + 1]), name))
    return out


def kissel_check(e_keV: np.ndarray) -> dict:
    """Kissel/CS_Photo photo ratio for Z ≥ KISSEL_ZMIN at the energies used; nodes outside KISSEL_CHECK."""
    xr = _xrl()
    zones = kissel_step_zones()
    bad, worst = [], (1.0, 0, 0.0)
    for z in range(KISSEL_ZMIN, XRL_ZMAX + 1):
        for E in np.asarray(e_keV, float):
            if any(lo <= E < hi for lo, hi, _ in zones.get(z, [])):
                continue                                  # CS_Total is used there by the primary variant
            r = xr.CS_Photo_Total(z, float(E)) / xr.CS_Photo(z, float(E))
            if abs(r - 1) > abs(worst[0] - 1):
                worst = (r, z, float(E))
            if not (KISSEL_CHECK[0] <= r <= KISSEL_CHECK[1]):
                k = xr.EdgeEnergy(z, xr.K_SHELL)
                bad.append({"Z": z, "energy_keV": float(E), "ratio": float(r), "K_edge_keV": float(k)})
    return {"n_outside": len(bad), "outside": bad, "worst": {"ratio": worst[0], "Z": worst[1], "energy_keV": worst[2]}}


def element_comparison(e_keV: np.ndarray, grid_idx: np.ndarray) -> pd.DataFrame:
    """Per element: median and max |μ/ρ_lib/μ/ρ_XCOM − 1| over the 20–200 keV nodes, and the energy of the max."""
    xc, _ = element_table("xcom", e_keV, grid_idx)
    sym = c3.elements_table()["symbol"].to_numpy()
    recs = []
    for lib in XRL_LIBS:
        t, _ = element_table(lib, e_keV, grid_idx)
        a = np.abs(t / xc - 1.0)
        for z in range(1, 101):
            j = int(np.argmax(a[z - 1]))
            recs.append({"lib": lib, "Z": z, "symbol": sym[z - 1], "median_abs_rel": float(np.median(a[z - 1])),
                         "max_abs_rel": float(a[z - 1, j]), "energy_of_max_keV": float(e_keV[j]),
                         "ratio_at_max": float(t[z - 1, j] / xc[z - 1, j])})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------------------------
# (a) and (c): single lines on the grid
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _mono_sets() -> dict:
    sets = refsets.practical_sets(elements=EDGE_ELEMENTS, headroom=False)
    keep = ["B4", "B4-PB"] + [f"B4&EXCL({el})" for el in EDGE_ELEMENTS]
    base = c3.base_mask(refsets.PRIMARY_RULE) & common.clean_mask()
    m = {k: sets[k] & base for k in keep}
    rows = np.flatnonzero(np.logical_or.reduce(list(m.values())))
    return {"rows": rows, "masks": {k: v[rows] for k, v in m.items()}}


def mono_specs() -> dict[str, tuple[str, str]]:
    out = {"PB|B4": ("B4", "B4-PB")}
    for el in EDGE_ELEMENTS:
        out[f"EXCL({el})|B4"] = (f"B4&EXCL({el})", "B4")
    return out


def mono_prices(lib: str) -> pd.DataFrame:
    """Frontiers and prices at the 20–200 keV grid nodes under ``lib`` (R2, best_clean, phase4a artefact flag)."""
    gi = continuum.grid_indices(*E_MONO)
    e = c3.energy_keV()[gi]
    tab, _ = element_table(lib, e, gi)
    s = _mono_sets()
    rows, masks = s["rows"], s["masks"]
    mr = np.asarray(c3.mass_fractions()[rows] @ tab)                      # (n_rows × n_E) μ/ρ, float64
    rho = c3.density()[rows]
    scale = refsets.measured_scale()[rows]
    q = {("areal", "any"): 1.0 / mr, ("thickness", "mp"): 1.0 / (mr * rho[:, None])}
    q[("thickness", "measured")] = q[("thickness", "mp")] * scale[:, None]
    u = c3.universe()
    form = u.formula_pretty.to_numpy(); mid = u.material_id.to_numpy()
    recs = []
    for (basis, den), Q in q.items():
        best = {}
        for name, m in masks.items():
            sub = np.where(m[:, None], Q, np.inf)
            j = np.argmin(sub, axis=0)                                       # first minimum = lowest row (as TopK)
            best[name] = (sub[j, np.arange(sub.shape[1])], rows[j])
        for label, (num, den_set) in mono_specs().items():
            qn, rn = best[num]; qd, rd = best[den_set]
            for k in range(e.size):
                recs.append({"lib": lib, "energy_keV": float(e[k]), "label": label, "basis": basis, "density": den,
                             "price": qn[k] / qd[k] - 1.0, "q_num": qn[k], "q_den": qd[k],
                             "delta_abs_T0.01": math.log(100.0) * (qn[k] - qd[k]),
                             "formula_num": form[rn[k]], "formula_den": form[rd[k]],
                             "mpid_num": mid[rn[k]], "mpid_den": mid[rd[k]]})
    return pd.DataFrame(recs)


def validate_mono(mx: pd.DataFrame) -> dict:
    """XCOM path vs refsets bands.parquet (headline R2 / coherent / best_clean, phase4a artefact flag)."""
    b = pd.read_parquet(common.P4 / "S" / "bands.parquet")
    b = b[(b.scenario == "MONO") & (b.artefact_def == "phase4a") & b.label.isin(list(mono_specs()))
          & (b.energy_keV >= E_MONO[0] - 1e-9) & (b.energy_keV <= E_MONO[1] + 1e-9)]
    b = b.assign(label=b.label.astype(str), basis=b.basis.astype(str), density=b.density.astype(str))
    m = mx.merge(b[["energy_keV", "label", "basis", "density", "price", "formula_num", "formula_den",
                    "mpid_num", "mpid_den"]], on=["energy_keV", "label", "basis", "density"], suffixes=("", "_s2"))
    d = (m.price - m.price_s2).abs()
    same = (m.mpid_num == m.mpid_num_s2) & (m.mpid_den == m.mpid_den_s2)
    # the 20–200 keV maxima of the EXCL filters equal the full-range (20 keV–10 MeV) headline maxima
    full = pd.read_parquet(common.P4 / "S" / "bands.parquet")
    full = full[(full.scenario == "MONO") & (full.artefact_def == "phase4a")]
    fm = {}
    for el in EDGE_ELEMENTS:
        for basis, den in (("areal", "any"), ("thickness", "measured"), ("thickness", "mp")):
            f = full[(full.label == f"EXCL({el})|B4") & (full.basis == basis) & (full.density == den)]
            s = mx[(mx.label == f"EXCL({el})|B4") & (mx.basis == basis) & (mx.density == den)]
            fm[f"EXCL({el})|B4 {basis}/{den}"] = {
                "full_range_max": float(f.price.max()),
                "full_range_argmax_keV": float(f.energy_keV.iloc[int(np.argmax(f.price.to_numpy()))]),
                "range_20_200_max": float(s.price.max())}
    full_ok = all(abs(x["full_range_max"] - x["range_20_200_max"]) <= TOL_PRICE_ABS for x in fm.values())
    return {"n_cells": int(len(m)), "n_expected": int(len(mx)), "max_abs_price_diff": float(d.max()),
            "excl_max_full_range_vs_20_200": fm, "excl_max_within_20_200": bool(full_ok),
            "n_frontier_compound_mismatch": int((~same).sum()),
            "mismatches": m[~same][["energy_keV", "label", "basis", "density", "formula_num", "formula_num_s2",
                                     "formula_den", "formula_den_s2"]].to_dict("records")[:20],
            "pass": bool(len(m) == len(mx) and d.max() <= TOL_PRICE_ABS and same.all() and full_ok)}


# ---------------------------------------------------------------------------------------------
# (b): DIAG lanthanide advantage
# ---------------------------------------------------------------------------------------------
def diag_variants() -> pd.DataFrame:
    v = c3.variants()
    v = v[(v.scenario == "DIAG") & v.primary].sort_values("variant_id").reset_index(drop=True)
    if len(v) != 36:
        raise RuntimeError(f"expected 36 primary DIAG variants, found {len(v)}")
    return v


@lru_cache(maxsize=1)
def diag_rows() -> dict:
    """R2 ∧ B4 lanthanide rows (stage 3 definition, families._b4_r2 ∧ lanthanide) and the textbook rows."""
    u = c3.universe()
    b4 = c3.base_mask("R2") & c3.bundle_mask("B4")
    lan = np.array([bool(set(x) & c3.LANTHANIDES) for x in u["elements"]])
    tb = pd.read_csv(c3.P3 / "C" / "textbook_by_variant.csv")
    t = tb.drop_duplicates("formula")[["formula", "material_id", "R2_material_id"]]
    t = t.assign(row=[c3.row_of(m) for m in t.material_id], R2_row=[c3.row_of(m) for m in t.R2_material_id])
    ln_rows = np.flatnonzero(b4 & lan)
    return {"ln_rows": ln_rows, "textbook": t.reset_index(drop=True),
            "n_b4_artefacts_phase4a": int((b4 & common.artefact_mask()).sum())}


@lru_cache(maxsize=64)
def _spec_table(lib: str, spectrum: str) -> np.ndarray:
    """Element μ/ρ (100 × K) at a spectrum's quadrature energies (evaluated there exactly, as Table C does)."""
    e, _ = c3.spectrum(spectrum)
    return element_table(lib, e)[0]


def pb_tstar(lib: str, v: pd.DataFrame) -> np.ndarray:
    """Air-kerma transmission of each variant's Pb sheet (ρ_Pb as Table C) under ``lib``."""
    from shieldmap.physics import transmission
    out = np.zeros(len(v))
    for i, r in enumerate(v.itertuples()):
        e, w = c3.spectrum_weights(r.spectrum, r.response)
        tab = _spec_table(lib, r.spectrum)
        mu = r.pb_density_g_cm3 * tab[81]
        out[i] = float(np.exp(transmission.log_transmission(mu[None, :], w, np.array([r.pb_thickness_cm]))[0]))
    return out


def diag_requirements(lib: str, rows: np.ndarray, like_for_like: bool = False, log=None) -> np.ndarray:
    """(rows × 36) narrow-beam areal mass (g/cm²) at each primary DIAG variant's target, Table C method:
    μ at the spectral quadrature energies, Brent solve of T(x) = T*, areal = x·ρ."""
    from shieldmap.physics import transmission
    v = diag_variants()
    xtol = config.get("solver.xtol_rel")
    W = c3.mass_fractions()[rows]
    rho = c3.density()[rows]
    ts = pb_tstar(lib, v) if like_for_like else v.T_target.to_numpy(float)
    out = np.zeros((rows.size, len(v)))
    for sp, g in v.groupby("spectrum", sort=True):
        e, w = c3.spectrum_weights(sp, "air_kerma")
        tab = _spec_table(lib, sp)
        mu = np.asarray(W @ tab) * rho[:, None]
        for j in g.index:
            x = transmission.solve_thickness(mu, w, ts[j], xtol)
            out[:, j] = x * rho
        if log:
            log(f"    [xs] {lib}{' lfl' if like_for_like else ''} {sp}")
    return out


def corrected_matrix(rows: np.ndarray) -> np.ndarray:
    """Stage 3 fluorescence-corrected areal mass (fixed-T, E₁, K + L) for ``rows`` × 36 variants, read from
    F/corrected_areal.parquet and C/corrected_extra.parquet (non-check rows), as families.diag_inputs does."""
    v = diag_variants()
    col = {int(x): j for j, x in enumerate(v.variant_id)}
    pos = {int(r): i for i, r in enumerate(rows)}
    Q = np.full((rows.size, len(v)), np.nan)
    fc = pd.read_parquet(c3.P3 / "F" / "corrected_areal.parquet", columns=["row", "variant_id", "q_eff_fixedT"])
    ex = pd.read_parquet(c3.P3 / "C" / "corrected_extra.parquet", columns=["row", "variant_id", "q_eff_fixedT", "is_check"])
    for df in (fc, ex[~ex.is_check]):
        df = df[df.row.isin(list(pos)) & df.variant_id.isin(list(col))]
        Q[df.row.map(pos).to_numpy(), df.variant_id.map(col).to_numpy()] = df.q_eff_fixedT.to_numpy()
    return Q


def advantage(q_ln: np.ndarray, q_tb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per variant: (advantage, argmin row position among ln rows, best textbook formula)."""
    tb = q_tb.drop(index=[f for f in TEXTBOOK_EXCL if f in q_tb.index])
    best_ln = np.nanmin(q_ln, axis=0)
    arg = np.nanargmin(q_ln, axis=0)
    best_tb = tb.min(axis=0).to_numpy()
    return best_tb / best_ln - 1.0, arg, tb.idxmin(axis=0).to_numpy()


def diag_item(log=None) -> tuple[pd.DataFrame, dict]:
    v = diag_variants()
    d = diag_rows()
    ln_rows, tbt = d["ln_rows"], d["textbook"]
    tb_rows = tbt.row.to_numpy()
    allrows = np.concatenate([ln_rows, tb_rows])
    nl = ln_rows.size
    u = c3.universe()
    res, checks = {}, {}
    for lib in LIBS:
        res[(lib, "fixed")] = diag_requirements(lib, allrows, False, log)
        if lib != "xcom":
            res[(lib, "lfl")] = diag_requirements(lib, allrows, True, log)
    # --- validation: XCOM path vs Table C, T*, stage 3 textbook comparison
    P = np.asarray(c3.profiles("areal")[:, v.variant_id.to_numpy()], dtype=np.float64)[allrows]
    rel = np.abs(res[("xcom", "fixed")] / P - 1.0)
    ts_x = pb_tstar("xcom", v)
    checks["tablec_max_rel"] = float(rel.max())
    checks["tstar_max_rel"] = float(np.max(np.abs(ts_x / v.T_target.to_numpy(float) - 1.0)))
    tbv = pd.read_csv(c3.P3 / "C" / "textbook_by_variant.csv")
    QE = corrected_matrix(ln_rows)
    if not np.isfinite(QE).all():
        raise RuntimeError("some R2 ∧ B4 lanthanide rows lack a Phase 3 corrected value")
    qe_tb = tbv[tbv.state == "after"].pivot(index="formula", columns="variant_id", values="q")[v.variant_id.to_numpy()]
    qn_tb_p3 = tbv[tbv.state == "before"].pivot(index="formula", columns="variant_id", values="q")[v.variant_id.to_numpy()]
    p3 = {}
    for state in ("before", "after"):
        t = tbv[(tbv.state == state) & ~tbv.formula.isin(TEXTBOOK_EXCL)].groupby("variant_id") \
            .ratio_to_best_B4_lanthanide.min().reindex(v.variant_id.to_numpy()).to_numpy() - 1.0
        p3[state] = t

    def tb_frame(Q):
        return pd.DataFrame(Q[nl:], index=tbt.formula.to_numpy(), columns=v.variant_id.to_numpy())

    recs = []
    adv = {}
    for lib in LIBS:
        for t in (("fixed",) if lib == "xcom" else ("fixed", "lfl")):
            Q = res[(lib, t)]
            a_n, arg_n, tb_n = advantage(Q[:nl], tb_frame(Q))
            ratio = Q / res[("xcom", "fixed")]                               # per-row narrow ratio lib / XCOM
            QEl = QE * ratio[:nl]
            qe_tb_l = qe_tb.loc[tbt.formula.to_numpy()].to_numpy() * ratio[nl:]
            a_c, arg_c, tb_c = advantage(QEl, pd.DataFrame(qe_tb_l, index=tbt.formula.to_numpy(),
                                                            columns=v.variant_id.to_numpy()))
            adv[(lib, t)] = {"narrow": a_n, "corrected": a_c}
            for j, r in v.iterrows():
                recs.append({"lib": lib, "tstar": "XCOM Pb (map)" if t == "fixed" else "like-for-like",
                             "variant_id": int(r.variant_id), "spectrum": r.spectrum, "quality": r.quality,
                             "anode_angle": int(r.anode_angle), "pb_mm": float(r.target_value),
                             "adv_narrow": a_n[j], "best_ln_narrow": u.formula_pretty.iat[ln_rows[arg_n[j]]],
                             "best_ln_narrow_mpid": u.material_id.iat[ln_rows[arg_n[j]]],
                             "best_textbook_narrow": tb_n[j],
                             "adv_corrected_estimate": a_c[j], "best_ln_corrected": u.formula_pretty.iat[ln_rows[arg_c[j]]],
                             "best_textbook_corrected": tb_c[j],
                             "ln_ratio_lib_over_xcom_min": float(np.min(ratio[:nl, j])),
                             "ln_ratio_lib_over_xcom_max": float(np.max(ratio[:nl, j]))})
    checks["adv_narrow_vs_phase3_max_abs"] = float(np.max(np.abs(adv[("xcom", "fixed")]["narrow"] - p3["before"])))
    checks["adv_corrected_vs_phase3_max_abs"] = float(np.max(np.abs(adv[("xcom", "fixed")]["corrected"] - p3["after"])))
    checks["textbook_narrow_vs_phase3_max_rel"] = float(np.max(np.abs(
        tb_frame(res[("xcom", "fixed")]).loc[qn_tb_p3.index].to_numpy() / qn_tb_p3.to_numpy() - 1.0)))
    checks["n_ln_rows"] = int(nl)
    checks["n_b4_artefacts_phase4a"] = d["n_b4_artefacts_phase4a"]
    checks["pass"] = bool(checks["tablec_max_rel"] <= TOL_PROFILE_REL and checks["tstar_max_rel"] <= TOL_TSTAR_REL
                          and checks["adv_narrow_vs_phase3_max_abs"] <= TOL_ADV_ABS
                          and checks["adv_corrected_vs_phase3_max_abs"] <= TOL_ADV_ABS
                          and checks["textbook_narrow_vs_phase3_max_rel"] <= TOL_PROFILE_REL)
    tst = {lib: pb_tstar(lib, v).tolist() for lib in XRL_LIBS}
    tst["xcom"] = v.T_target.tolist()
    checks["tstar"] = tst
    return pd.DataFrame(recs), checks


# ---------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------
def _pp(x: float) -> str:
    return f"{100 * x:+.2f} pp".replace("+-", "−").replace("-", "−")


def _pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f} %"


def mono_summary(mp: pd.DataFrame) -> pd.DataFrame:
    """One row per item × basis/density × lib: the headline statistic and the difference to XCOM."""
    recs = []

    def add(item, scen, basis, den, sub_by_lib, stat):
        x = sub_by_lib["xcom"]
        vx, ex_, fx = stat(x)
        for lib in XRL_LIBS:
            vl, el, fl = stat(sub_by_lib[lib])
            j = x.merge(sub_by_lib[lib], on="energy_keV", suffixes=("_x", "_l"))
            dd = j.price_l - j.price_x
            recs.append({"item": item, "scenario_or_energy": scen, "basis": basis, "density": den, "lib": lib,
                         "xcom_value": vx, "xraylib_value": vl, "delta": vl - vx,
                         "xcom_energy_keV": ex_, "xraylib_energy_keV": el, "xcom_frontier": fx, "xraylib_frontier": fl,
                         "nodewise_diff_min": float(dd.min()), "nodewise_diff_max": float(dd.max()),
                         "nodewise_diff_energy_of_maxabs": float(j.energy_keV.iloc[int(np.argmax(np.abs(dd.to_numpy())))])})

    def smax(s):
        i = s.price.idxmax(); r = s.loc[i]
        return float(r.price), float(r.energy_keV), f"{r.formula_den} → {r.formula_num}"

    for lo, hi, name in ((HEAD_WINDOW[0], HEAD_WINDOW[1], "88.5–90.5 keV (headline window, max)"),
                         (PB_WINDOW[0], PB_WINDOW[1], "85–95 keV (max)")):
        sub = {lib: mp[(mp.lib == lib) & (mp.label == "PB|B4") & (mp.basis == "areal")
                       & mp.energy_keV.between(lo - 1e-9, hi + 1e-9)] for lib in LIBS}
        add("PB|B4 areal window", name, "areal", "any", sub, smax)
    for el in EDGE_ELEMENTS:
        for basis, den in (("areal", "any"), ("thickness", "measured"), ("thickness", "mp")):
            sub = {lib: mp[(mp.lib == lib) & (mp.label == f"EXCL({el})|B4") & (mp.basis == basis) & (mp.density == den)]
                   for lib in LIBS}
            add(f"EXCL({el})|B4", "20–200 keV (max)", basis, den, sub, smax)
    return pd.DataFrame(recs)


def diag_summary(dv: pd.DataFrame) -> pd.DataFrame:
    recs = []
    x = dv[dv.lib == "xcom"].set_index("variant_id")
    for state, col in (("narrow", "adv_narrow"), ("corrected (estimate)", "adv_corrected_estimate")):
        for lib in XRL_LIBS:
            for tstar in ("XCOM Pb (map)", "like-for-like"):
                l = dv[(dv.lib == lib) & (dv.tstar == tstar)].set_index("variant_id")
                d = l[col] - x[col]
                for stat in ("median", "min", "max"):
                    vx = float(getattr(x[col], stat)()); vl = float(getattr(l[col], stat)())
                    recs.append({"item": f"DIAG advantage {state}", "scenario_or_energy": f"36 DIAG variants ({stat})",
                                 "basis": "areal", "density": "any", "lib": lib, "tstar": tstar,
                                 "xcom_value": vx, "xraylib_value": vl, "delta": vl - vx,
                                 "nodewise_diff_min": float(d.min()), "nodewise_diff_max": float(d.max()),
                                 "n_ge_10pct_xcom": int((x[col] >= 0.10).sum()), "n_ge_10pct_xraylib": int((l[col] >= 0.10).sum())})
    return pd.DataFrame(recs)


def summary_csv(ms: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    """data/phase4a/xs/summary.csv: item, scenario/variant or energy, basis, xcom_value, xraylib_value, diff, note."""
    recs = []
    for r in ms.itertuples():
        b = r.basis if r.basis == "areal" else f"thickness ({'measured-density point' if r.density == 'measured' else 'MP density'})"
        recs.append({"item": r.item, "scenario_or_energy": f"{r.scenario_or_energy}; XCOM at {r.xcom_energy_keV:g} keV, "
                     f"xraylib at {r.xraylib_energy_keV:g} keV", "basis": b, "xcom_value": r.xcom_value,
                     "xraylib_value": r.xraylib_value, "diff": r.delta,
                     "note": f"{LIB_TEXT[r.lib]}; frontier XCOM {r.xcom_frontier}, xraylib {r.xraylib_frontier}; "
                             f"node-wise Δ {r.nodewise_diff_min:+.4f} to {r.nodewise_diff_max:+.4f}"})
    for r in ds.itertuples():
        recs.append({"item": r.item, "scenario_or_energy": r.scenario_or_energy, "basis": "areal",
                     "xcom_value": r.xcom_value, "xraylib_value": r.xraylib_value, "diff": r.delta,
                     "note": f"{LIB_TEXT[r.lib]}; T* {r.tstar}; per-variant Δ {r.nodewise_diff_min:+.4f} to "
                             f"{r.nodewise_diff_max:+.4f}; variants ≥ 10 %: XCOM {r.n_ge_10pct_xcom}, xraylib {r.n_ge_10pct_xraylib}"
                             + ("; FIRST-ORDER ESTIMATE (corrected × xraylib/XCOM narrow ratio)" if "corrected" in r.item else "")})
    return pd.DataFrame(recs)


def headline_sensitivity(ms: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    """match_label → short string joined into the headline table by analysis4.headline."""
    out = []

    def band(vals):
        lo, hi = min(vals), max(vals)
        return _pp(lo) if abs(hi - lo) < 5e-5 else f"{_pp(lo)} to {_pp(hi)}"

    s = ms[(ms.item == "PB|B4 areal window") & ms.scenario_or_energy.str.startswith("88.5")].set_index("lib")
    k, kr, t = s.loc[PRIMARY_XRL], s.loc["xrl_kissel_raw"], s.loc["xrl_total"]
    w = ms[(ms.item == "PB|B4 areal window") & ms.scenario_or_energy.str.startswith("85")]
    out.append({"match_label": "PB|B4 areal window",
                "xs_sensitivity": f"xraylib (Kissel) {_pct(k.xraylib_value)} at {k.xraylib_energy_keV:g} keV "
                                  f"(XCOM {_pct(k.xcom_value)} at {k.xcom_energy_keV:g} keV), Δ {_pp(k.delta)}; "
                                  f"raw Kissel Δ {_pp(kr.delta)}; CS_Total Δ {_pp(t.delta)}; "
                                  f"node-wise 85–95 keV Δ band {band(list(w.nodewise_diff_min) + list(w.nodewise_diff_max))}"})
    for state, lab in (("narrow", "DIAG advantage narrow"), ("corrected (estimate)", "DIAG advantage corrected (estimate)")):
        d = ds[(ds.item == f"DIAG advantage {state}") & ds.scenario_or_energy.str.contains("median")]
        k = d[(d.lib == PRIMARY_XRL) & (d.tstar == "XCOM Pb (map)")].iloc[0]
        allv = list(d.nodewise_diff_min) + list(d.nodewise_diff_max)
        lfl = d[(d.lib == PRIMARY_XRL) & (d.tstar == "like-for-like")].iloc[0]
        raw = d[(d.lib == "xrl_kissel_raw") & (d.tstar == "XCOM Pb (map)")].iloc[0]
        tot = d[(d.lib == "xrl_total") & (d.tstar == "XCOM Pb (map)")].iloc[0]
        pre = "first-order estimate: " if "corrected" in state else ""
        out.append({"match_label": lab,
                    "xs_sensitivity": f"{pre}xraylib (Kissel) median {_pct(k.xraylib_value, 1)} (XCOM {_pct(k.xcom_value, 1)}), "
                                      f"Δ {_pp(k.delta)}; like-for-like T* Δ {_pp(lfl.delta)}; raw Kissel Δ {_pp(raw.delta)}; "
                                      f"CS_Total Δ {_pp(tot.delta)}; per-variant Δ band (all xraylib variants) {band(allv)}"})
    for el in EDGE_ELEMENTS:
        for basis, den in (("areal", "any"), ("thickness", "measured")):
            s = ms[(ms.item == f"EXCL({el})|B4") & (ms.basis == basis) & (ms.density == den)].set_index("lib")
            k, kr, t = s.loc[PRIMARY_XRL], s.loc["xrl_kissel_raw"], s.loc["xrl_total"]
            raw_txt = (f"raw Kissel {_pct(kr.xraylib_value)} at {kr.xraylib_energy_keV:g} keV (Δ {_pp(kr.delta)}, "
                       f"set by the non-physical Kissel step just above the K edge)" if abs(kr.delta - k.delta) > 5e-4
                       else f"raw Kissel Δ {_pp(kr.delta)}")
            out.append({"match_label": f"EXCL({el})|B4 {basis}",
                        "xs_sensitivity": f"xraylib (Kissel) max {_pct(k.xraylib_value)} at {k.xraylib_energy_keV:g} keV "
                                          f"(XCOM {_pct(k.xcom_value)} at {k.xcom_energy_keV:g} keV), Δ {_pp(k.delta)}; "
                                          f"{raw_txt}; CS_Total Δ {_pp(t.delta)}"
                                          + ("; measured-density point" if basis == "thickness" else "")})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------
def run(verbose: bool = True) -> dict:
    log = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    t0 = time.time()
    paths.ensure(OUT)
    prov = xraylib_provenance()
    gi = continuum.grid_indices(*E_MONO)
    e_grid = c3.energy_keV()[gi]
    ec = element_comparison(e_grid, gi)
    ec.to_csv(OUT / "element_ratio_20_200keV.csv", index=False)
    # Kissel coverage check on every energy used (grid nodes and DIAG quadrature energies)
    e_used = [e_grid] + [c3.spectrum(s)[0] for s in diag_variants().spectrum.unique()]
    kc = kissel_check(np.unique(np.concatenate(e_used)))
    fallbacks = {lib: {int(z): n for z, n in element_table(lib, e_grid, gi)[1].items()} for lib in XRL_LIBS}
    log(f"[xs] element tables ({time.time() - t0:.0f} s); Kissel nodes outside {KISSEL_CHECK}: {kc['n_outside']}")
    mp = pd.concat([mono_prices(lib) for lib in LIBS], ignore_index=True)
    mp.to_csv(OUT / "mono_prices.csv", index=False)
    vm = validate_mono(mp[mp.lib == "xcom"])
    log(f"[xs] mono done ({time.time() - t0:.0f} s); validation pass={vm['pass']}")
    if not vm["pass"]:
        common.write_json(OUT / "validation.json", {"mono": vm})
        raise RuntimeError(f"XCOM path does not reproduce the Step 2 bands: {vm}")
    dv, dchk = diag_item(log)
    dv.to_csv(OUT / "diag_advantage_by_variant.csv", index=False)
    log(f"[xs] DIAG done ({time.time() - t0:.0f} s); validation pass={dchk['pass']}")
    val = {"mono": vm, "diag": {k: v for k, v in dchk.items() if k != "tstar"}}
    common.write_json(OUT / "validation.json", val)
    if not dchk["pass"]:
        raise RuntimeError(f"XCOM path does not reproduce Table C / Phase 3 DIAG: {dchk}")
    ms = mono_summary(mp); ds = diag_summary(dv)
    ms.to_csv(OUT / "mono_summary.csv", index=False); ds.to_csv(OUT / "diag_summary.csv", index=False)
    sc = summary_csv(ms, ds); sc.to_csv(OUT / "summary.csv", index=False)
    hs = headline_sensitivity(ms, ds); hs.to_csv(OUT / "headline_sensitivity.csv", index=False)
    meta = {"xraylib": prov, "libs": LIB_TEXT, "primary_xraylib_variant": PRIMARY_XRL, "kissel_zmin": KISSEL_ZMIN,
            "kissel_check": {"range": KISSEL_CHECK, "n_outside": kc["n_outside"], "worst": kc["worst"],
                             "outside": kc["outside"][:50]},
            "kissel_step_zones": {int(z): v for z, v in kissel_step_zones().items()},
            "kissel_step_rule": {"shells": STEP_SHELLS, "scan_keV": STEP_SCAN_KEV, "drop": STEP_DROP},
            "xcom_fallback_elements_grid": fallbacks, "tstar": dchk["tstar"],
            "energies": {"mono_keV": list(E_MONO), "pb_window_keV": list(PB_WINDOW), "headline_window_keV": list(HEAD_WINDOW)},
            "seconds": round(time.time() - t0, 1)}
    common.write_json(OUT / "meta.json", meta)
    write_report(ms, ds, dv, val, meta, ec)
    log(f"[xs] done in {time.time() - t0:.0f} s")
    return {"validation": val, "meta": meta}


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def write_report(ms, ds, dv, val, meta, ec) -> None:
    L = []
    a = L.append
    a("# Phase 4a — Step 7: cross-section library sensitivity (xraylib vs XCOM)\n")
    a("**Sensitivity only.** No value computed in this step is written to the map, to Table C or to any Phase 1–3 "
      "file. Code `src/shieldmap/analysis4/xs.py` (`run()`), outputs `data/phase4a/xs/`. Primaries as the headline "
      "table: rule R2, with coherent, best_clean under the Phase 4a artefact flag, B4-internal prices, narrow beam.\n")
    p = meta["xraylib"]
    a("## Libraries\n")
    a(f"- xraylib {p['version']} (source record `config/sources/xraylib.yaml`, file SHA-256 match: "
      f"{p['matches_source_record']}).")
    a("- **XCOM** — the map (Phase 1 primary, with coherent).")
    a("- **xraylib CS_Total** (photo + Rayleigh + Compton from xraylib's own tables), Z = 1–98; Es and Fm keep XCOM. "
      "Its terms reproduce XCOM to ~1e-4 for almost every element (median |Δ| over 20–200 keV in "
      "`element_ratio_20_200keV.csv`), i.e. it is a re-tabulation of the same underlying data and tests "
      "interpolation plus the few elements where xraylib's tables differ (Mo K-edge region, Tc, Po, Rn, Pu–Cf).")
    a(f"- **xraylib Kissel** (primary xraylib variant): CS_Total_Kissel (Kissel partial-shell photoionization + "
      f"Rayleigh + Compton) for Z ≥ {KISSEL_ZMIN}, CS_Total below. In xraylib 4.3.0 the Kissel tables of the light "
      "elements end inside 20–200 keV and the extrapolation collapses (CS_Photo_Total/CS_Photo → 0: O 0.03 at 40 keV, "
      "S 0.44 at 100 keV, Mn 0.93 at 200 keV), so they are not used below Z = 26; for Z ≥ 26 the ratio to CS_Photo "
      f"stays within {KISSEL_CHECK} on every energy used except {meta['kissel_check']['n_outside']} node(s) "
      f"(worst {meta['kissel_check']['worst']['ratio']:.3f}, Z = {meta['kissel_check']['worst']['Z']} at "
      f"{meta['kissel_check']['worst']['energy_keV']:.4g} keV; list in `meta.json` — edge-straddling quadrature "
      "energies where XCOM/CS_Photo and Kissel place an edge differently, and the Mo K edge where xraylib's CS_Photo "
      "is the defective one).")
    zk = meta["kissel_step_zones"]
    ex = ", ".join(f"{c3.elements_table().symbol.iat[int(z) - 1]} {nm} {lo:.3f}–{hi:.3f}"
                   for z, zz in zk.items() for lo, hi, nm in zz
                   if nm == "K" and int(z) in (57, 64, 72, 73, 74, 75, 79, 80, 81))
    a("- **Kissel edge-step zones.** Just above many K and L edges xraylib's Kissel partial cross section runs on a "
      "plateau and then drops by 2–7 % within 5 eV, which is not physical (e.g. Ta K: +6.2 % over XCOM from the "
      "edge at 67.416 keV to 67.6 keV, −0.7 % beyond; W +7.3 % up to 69.73 keV; Tl +7 % up to 85.87 keV). The step "
      "marks where Kissel's own theoretical edge lies above xraylib's experimental edge energy, so inside the zone "
      "the value is not tabulated Kissel data. Zones are detected automatically (largest fall of CS_Photo_Partial "
      f"in a {STEP_SCAN_KEV[0] * 1000:g} eV scan over {STEP_SCAN_KEV[1]:g} keV above each K/L1/L2/L3 edge; zone if "
      f"> {100 * STEP_DROP:g} %): {sum(len(v) for v in zk.values())} zones for Z ≥ 26; K-edge examples: {ex}. "
      "The primary Kissel variant uses CS_Total inside the zones (same rule as for Z < 26: Kissel only where it "
      "carries data); the raw Kissel variant (`xrl_kissel_raw`) keeps them and is reported alongside, because the "
      "grid nodes 65.5 keV (Hf) and 67.5 keV (Ta) fall inside the zones.")
    a("- Pair production is absent from xraylib's totals; all energies here are ≤ 200 keV (no pair production).\n")
    a("Element-level differences at the K-edge regions that matter here (|μ/ρ_lib/μ/ρ_XCOM − 1|, 20–200 keV nodes):\n")
    a("| element | lib | median | max | at keV |")
    a("|---|---|---|---|---|")
    for s in ("Mo", "Sn", "Ce", "Pr", "Gd", "Er", "Hf", "Ta", "W", "Tl", "Pb", "Bi"):
        for lib in XRL_LIBS:
            r = ec[(ec.symbol == s) & (ec.lib == lib)].iloc[0]
            a(f"| {s} | {lib} | {r.median_abs_rel:.2e} | {r.max_abs_rel:.2e} | {r.energy_of_max_keV:g} |")
    a("")
    vm, vd = val["mono"], val["diag"]
    a("## Validation (XCOM path reproduces the map; checked before any xraylib number is used)\n")
    a(f"- Single lines 20–200 keV, PB|B4 and EXCL(W/Ta/Hf)|B4, areal + thickness (MP and measured-density point): "
      f"{vm['n_cells']} cells vs `data/phase4a/S/bands.parquet`; max |Δprice| = {vm['max_abs_price_diff']:.2e} "
      f"(tolerance {TOL_PRICE_ABS:g}); frontier-compound mismatches = {vm['n_frontier_compound_mismatch']}. "
      f"**{'PASS' if vm['pass'] else 'FAIL'}**.")
    a(f"- DIAG: {vd['n_ln_rows']} R2 ∧ B4 lanthanide rows + 10 textbook rows × 36 variants recomputed with the Table C "
      f"method; max relative difference to Table C `profiles('areal')` = {vd['tablec_max_rel']:.2e} (float32 storage; "
      f"tolerance {TOL_PROFILE_REL:g}); Pb T* recomputed = {vd['tstar_max_rel']:.1e}; narrow advantage vs Phase 3 "
      f"`textbook_by_variant.csv` max |Δ| = {vd['adv_narrow_vs_phase3_max_abs']:.1e}; corrected advantage (Phase 3 "
      f"corrected values reassembled) max |Δ| = {vd['adv_corrected_vs_phase3_max_abs']:.1e}. B4 artefact entries "
      f"under the Phase 4a flag: {vd['n_b4_artefacts_phase4a']}. **{'PASS' if vd['pass'] else 'FAIL'}**.\n")

    a("## (a) Pb K-edge mass window: PB|B4, areal mass, narrow beam\n")
    a("P = q*(B4)/q*(B4 without PB) − 1 at every grid node 85–95 keV.\n")
    a("| keV | XCOM | frontier (XCOM) | xraylib Kissel | Δ (pp) | frontier (Kissel) | raw Kissel Δ (pp) | xraylib CS_Total | Δ (pp) |")
    a("|---|---|---|---|---|---|---|---|---|")
    mp = pd.read_csv(OUT / "mono_prices.csv")
    w = mp[(mp.label == "PB|B4") & (mp.basis == "areal") & mp.energy_keV.between(PB_WINDOW[0] - 1e-9, PB_WINDOW[1] + 1e-9)]
    for E, g in w.groupby("energy_keV"):
        g = g.set_index("lib")
        x, k, kr, t = g.loc["xcom"], g.loc["xrl_kissel"], g.loc["xrl_kissel_raw"], g.loc["xrl_total"]
        a(f"| {E:g} | {_pct(x.price)} | {x.formula_den} → {x.formula_num} | {_pct(k.price)} | {100 * (k.price - x.price):+.3f} | "
          f"{k.formula_den} → {k.formula_num} | {100 * (kr.price - x.price):+.3f} | {_pct(t.price)} | {100 * (t.price - x.price):+.3f} |")
    a("")
    for r in ms[ms.item == "PB|B4 areal window"].itertuples():
        a(f"- {r.scenario_or_energy}, {LIB_TEXT[r.lib]}: XCOM {_pct(r.xcom_value)} at {r.xcom_energy_keV:g} keV, "
          f"xraylib {_pct(r.xraylib_value)} at {r.xraylib_energy_keV:g} keV, Δ {_pp(r.delta)}; node-wise Δ "
          f"{_pp(r.nodewise_diff_min)} to {_pp(r.nodewise_diff_max)}.")
    a("")
    a("## (b) DIAG lanthanide advantage (36 primary DIAG variants, areal, air kerma)\n")
    a("Advantage = (best non-Pb textbook material, Pb and PbO excluded) / (best R2 ∧ B4 lanthanide compound) − 1. "
      "Every R2 ∧ B4 lanthanide row is recomputed under each library (no candidate restriction, so no certification "
      "of a restriction is needed). T* fixed from XCOM Pb (as the map) and like-for-like (Pb T* recomputed with the same "
      "library). The corrected advantage is a **first-order estimate**: each row's Phase 3 corrected areal mass "
      "(E₁, K + L, fixed-T) × that row's narrow-beam ratio xraylib/XCOM; the escape model itself stays XCOM-based.\n")
    a("| quantity | library | T* | XCOM median (min–max) | xraylib median (min–max) | Δ median | per-variant Δ | variants ≥ 10 % (XCOM → xraylib) |")
    a("|---|---|---|---|---|---|---|---|")
    for item in ("DIAG advantage narrow", "DIAG advantage corrected (estimate)"):
        for lib in XRL_LIBS:
            for tstar in ("XCOM Pb (map)", "like-for-like"):
                d = ds[(ds.item == item) & (ds.lib == lib) & (ds.tstar == tstar)].set_index("scenario_or_energy")
                md, mn, mx = (d.loc[f"36 DIAG variants ({s})"] for s in ("median", "min", "max"))
                a(f"| {item.replace('DIAG advantage ', '')} | {lib} | {tstar} | {_pct(md.xcom_value, 1)} ({_pct(mn.xcom_value, 1)}–"
                  f"{_pct(mx.xcom_value, 1)}) | {_pct(md.xraylib_value, 1)} ({_pct(mn.xraylib_value, 1)}–{_pct(mx.xraylib_value, 1)}) | "
                  f"{_pp(md.delta)} | {_pp(md.nodewise_diff_min)} to {_pp(md.nodewise_diff_max)} | "
                  f"{md.n_ge_10pct_xcom} → {md.n_ge_10pct_xraylib} |")
    a("")
    x = dv[dv.lib == "xcom"].set_index("variant_id")
    k = dv[(dv.lib == PRIMARY_XRL) & (dv.tstar == "XCOM Pb (map)")].set_index("variant_id")
    ch_ln = int((x.best_ln_narrow_mpid != k.best_ln_narrow_mpid).sum())
    ch_tb = int((x.best_textbook_narrow != k.best_textbook_narrow).sum())
    a(f"- Narrow beam, Kissel, fixed T*: the best lanthanide compound changes in {ch_ln} of 36 variants, the best "
      f"textbook material in {ch_tb} of 36. Per-row narrow ratio xraylib/XCOM over the lanthanide rows: "
      f"{k.ln_ratio_lib_over_xcom_min.min():.4f}–{k.ln_ratio_lib_over_xcom_max.max():.4f}.")
    t = meta["tstar"]
    rel = np.array(t["xrl_kissel"]) / np.array(t["xcom"]) - 1
    a(f"- Like-for-like Pb T* (Kissel) / XCOM T*: {rel.min():+.2%} to {rel.max():+.2%} over the 36 variants.")
    worst = (k.adv_narrow - x.adv_narrow).abs().idxmax()
    a(f"- Largest per-variant narrow change: variant {worst} ({x.loc[worst, 'spectrum']}, {x.loc[worst, 'pb_mm']:g} mm): "
      f"{_pct(x.loc[worst, 'adv_narrow'], 1)} → {_pct(k.loc[worst, 'adv_narrow'], 1)}. Knife-edge RQR5 0.25 mm corrected "
      f"variants: " + ", ".join(f"{x.loc[i, 'spectrum']} {_pct(x.loc[i, 'adv_corrected_estimate'], 1)} → "
                                f"{_pct(k.loc[i, 'adv_corrected_estimate'], 1)}"
                                for i in x.index if x.loc[i, "quality"] == "RQR5" and abs(x.loc[i, "pb_mm"] - 0.25) < 1e-9) + ".\n")
    a("## (c) Element-exclusion filters W / Ta / Hf given B4 (single lines 20–200 keV)\n")
    a("P = q*(B4 ∧ no X)/q*(B4) − 1. Thickness at the Phase 3 measured-density point (as the headline table) and at "
      "MP densities; both are the same μ/ρ swap (the density scaling is library-independent). The XCOM maxima over "
      f"20–200 keV equal the headline maxima over 20 keV–10 MeV (checked: {val['mono']['excl_max_within_20_200']}).\n")
    a("| filter | basis | library | XCOM max (keV) | xraylib max (keV) | Δ | frontier XCOM → xraylib | node-wise Δ band |")
    a("|---|---|---|---|---|---|---|---|")
    for r in ms[ms.item.str.startswith("EXCL")].itertuples():
        b = r.basis if r.basis == "areal" else f"thickness ({r.density})"
        a(f"| {r.item} | {b} | {r.lib} | {_pct(r.xcom_value)} ({r.xcom_energy_keV:g}) | {_pct(r.xraylib_value)} "
          f"({r.xraylib_energy_keV:g}) | {_pp(r.delta)} | {r.xcom_frontier} / {r.xraylib_frontier} | "
          f"{_pp(r.nodewise_diff_min)} to {_pp(r.nodewise_diff_max)} (max at {r.nodewise_diff_energy_of_maxabs:g} keV) |")
    a("")
    a("## Bands for the headline table (`data/phase4a/xs/headline_sensitivity.csv`)\n")
    hs = pd.read_csv(OUT / "headline_sensitivity.csv")
    a("| match_label | xs_sensitivity |")
    a("|---|---|")
    for r in hs.itertuples():
        a(f"| {r.match_label} | {r.xs_sensitivity} |")
    a("")
    a("## Reading\n")
    a("- **xraylib CS_Total** shares XCOM's underlying data; every item moves by ≤ 0.02 pp. That is a check of "
      "interpolation, not an independent library.")
    a("- **xraylib Kissel** (independent photoionization) moves every headline item by less than 0.5 pp. "
      "No frontier compound changes in the 85–95 keV window or at the W/Ta/Hf maxima. Elsewhere on the 20–200 keV "
      "areal grid, the B4 frontier changes only at 21.5–24 keV, where xraylib's Mo μ/ρ (both variants) differs from "
      "XCOM's by up to 15–80 % (Mo K-edge region; Mo is the XCOM frontier there), and at 62 keV under Kissel "
      "(a near tie, YbH₂ → Tm). Every DIAG variant keeps an advantage ≥ 10 %, both narrow and "
      "corrected (estimate), and the knife-edge RQR5 0.25 mm corrected advantage stays at about 10.4–10.5 %. The "
      "W filter maximum moves from the 70 keV node to 71.5 keV, because Kissel's W μ/ρ just above the edge is about "
      "0.8 % lower than XCOM's.")
    a("- **The only large differences are edge artefacts.** With raw Kissel, EXCL(Ta) and EXCL(Hf) rise by 5–8 pp at "
      "the single grid node inside the 0.2 keV Kissel edge-step zone (67.5 keV, 65.5 keV). They vanish when the zone "
      "uses CS_Total. Single-line prices at nodes within a few hundred eV of a K edge therefore carry a "
      "library-interpolation uncertainty of several pp. A K-edge-window statement should quote the window, not "
      "the edge-adjacent node.\n")
    a("## Files\n")
    for f in ("summary.csv", "headline_sensitivity.csv", "mono_prices.csv", "mono_summary.csv",
              "diag_advantage_by_variant.csv", "diag_summary.csv", "element_ratio_20_200keV.csv", "validation.json",
              "meta.json"):
        a(f"- `data/phase4a/xs/{f}`")
    a("")
    paths.ensure(REPORT.parent)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["validation"], indent=1, default=str)[:4000])
