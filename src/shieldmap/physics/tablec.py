"""Table C — derived quantities, computed and never fetched (data/processed/).

Outputs
  energy_keV.npy                      common grid (keV)
  index.parquet                       row → material_id, formula_reduced, density
  mu_rho_wcoh.npy, mu_rho_wocoh.npy   float32 (N × n_E) mass attenuation coefficients (cm²/g)
  key_energies.parquet                per compound × key energy: μ/ρ, μ, HVL, TVL, MFP (both coherent
                                      definitions) and partial μ/ρ (coh, incoh, pe, pair_nuc, pair_el)
  envelope.parquet                    per energy: max μ/ρ over Z = 1–100; max μ over elemental solids
  spectra.parquet                     spectrum metadata (SpekPy matching, dropped fractions, lines)
  spectrum_bins.parquet               spectrum fluence per grid node
  profile_variants.parquet            one row per profile variant (source × target × response × coherent)
  profiles.parquet                    long: row × variant → thickness_req_cm, areal_mass_req_g_cm2,
                                      pb_relative_thickness, pb_relative_areal_mass

μ = ρ·μ/ρ, HVL = ln 2/μ, TVL = ln 10/μ, MFP = 1/μ are exact transforms; use the accessors below for
the full energy grid rather than storing them.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sps

from shieldmap import config, paths
from shieldmap.physics import mixture, spectra as spectra_mod, transmission

OUT = paths.PROCESSED
MP = paths.INTERIM / "mp"
CHUNK = 20000


# ----------------------------------------------------------------------------------------------
# accessors
# ----------------------------------------------------------------------------------------------
def load(coherent: bool = True, mmap: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(index, energy_keV, μ/ρ array) for the chosen coherent definition."""
    idx = pd.read_parquet(OUT / "index.parquet")
    e = np.load(OUT / "energy_keV.npy")
    arr = np.load(OUT / ("mu_rho_wcoh.npy" if coherent else "mu_rho_wocoh.npy"),
                  mmap_mode="r" if mmap else None)
    return idx, e, arr


def mu(index: pd.DataFrame, mu_rho: np.ndarray) -> np.ndarray:
    return np.asarray(mu_rho, dtype=np.float64) * index["density"].to_numpy()[:, None]


def hvl(mu_arr: np.ndarray) -> np.ndarray:
    return math.log(2.0) / mu_arr


def tvl(mu_arr: np.ndarray) -> np.ndarray:
    return math.log(10.0) / mu_arr


def mfp(mu_arr: np.ndarray) -> np.ndarray:
    return 1.0 / mu_arr


# ----------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------
def load_compounds() -> tuple[pd.DataFrame, sps.csr_matrix]:
    comp = pd.read_parquet(MP / "compounds.parquet")
    w = sps.load_npz(MP / "mass_fractions.npz").tocsr()
    ix = pd.read_parquet(MP / "mass_fractions_index.parquet")
    if not (ix["material_id"].to_numpy() == comp["material_id"].to_numpy()).all():
        raise ValueError("mass-fraction matrix rows do not match compounds.parquet order")
    if w.shape != (len(comp), 100):
        raise ValueError(f"mass-fraction matrix has shape {w.shape}")
    s = np.asarray(w.sum(axis=1)).ravel()
    if np.max(np.abs(s - 1)) > 1e-9:
        raise ValueError("mass fractions do not sum to 1")
    return comp, w


def lead_density() -> float:
    src = config.get("reference_material.lead_density_source")
    if src != "pymatgen_density_of_solid":
        raise ValueError(src)
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet")
    return float(el.loc[el.symbol == "Pb", "density_solid_exp"].iloc[0])


# ----------------------------------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------------------------------
def build_variants(spec_list) -> pd.DataFrame:
    rows = []
    for coh in config.get("coherent.include"):
        for resp in config.get("response.functions"):
            for sp in spec_list:
                if sp.kind == "tube":
                    targets = [("pb_mm", float(t)) for t in config.get("targets.diagnostic_pb_mm")]
                else:
                    targets = [("transmission", float(t)) for t in config.get("targets.radionuclide_transmission")]
                for ttype, tval in targets:
                    rows.append({"spectrum": sp.name, "source_kind": sp.kind, "target_type": ttype,
                                 "target_value": tval, "response": resp, "coherent": bool(coh)})
    v = pd.DataFrame(rows)
    v.insert(0, "variant_id", np.arange(len(v), dtype=np.int16))
    return v


# ----------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------
def build(chunk: int = CHUNK, limit: int | None = None, verbose: bool = True) -> dict:
    paths.ensure(OUT)
    xs = mixture.load_element_xsec()
    e = xs["energy_keV"]
    np.save(OUT / "energy_keV.npy", e)
    comp, W = load_compounds()
    if limit:
        comp, W = comp.iloc[:limit].reset_index(drop=True), W[:limit]
    n, n_e = len(comp), e.size
    dens = comp["density"].to_numpy(dtype=float)
    if not np.all(np.isfinite(dens) & (dens > 0)):
        raise ValueError("non-positive or missing densities in Table B")

    index = pd.DataFrame({"row": np.arange(n, dtype=np.int32), "material_id": comp["material_id"],
                          "formula_reduced": comp["formula_reduced"], "density": dens})
    index.to_parquet(OUT / "index.parquet", index=False)

    # ---- elemental reference envelope (unfiltered) ----
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet").sort_values("Z")
    sym = el["symbol"].to_numpy(); rho_el = el["density_solid_exp"].to_numpy(dtype=float)
    env = {"energy_keV": e}
    for tag, key in (("wcoh", "total_wcoh"), ("wocoh", "total_wocoh")):
        m = xs[key]
        env[f"max_mu_rho_{tag}"] = m.max(axis=0); env[f"argmax_mu_rho_{tag}"] = sym[m.argmax(axis=0)]
        has = np.isfinite(rho_el)
        mu_el = m[has] * rho_el[has, None]
        env[f"max_mu_{tag}"] = mu_el.max(axis=0); env[f"argmax_mu_{tag}"] = sym[has][mu_el.argmax(axis=0)]
    pd.DataFrame(env).to_parquet(OUT / "envelope.parquet", index=False)

    # ---- spectra, quadrature energies and variants ----
    spec_list = spectra_mod.all_spectra()
    smeta, sbins = [], []
    for sp in spec_list:
        smeta.append({"spectrum": sp.name, "kind": sp.kind, "n_nodes": int(sp.energy_keV.size),
                      "meta_json": json.dumps(sp.meta, default=float)})
        sbins.append(pd.DataFrame({"spectrum": sp.name, "energy_keV": sp.energy_keV,
                                   "lo_keV": sp.lo_keV if sp.lo_keV is not None else sp.energy_keV,
                                   "hi_keV": sp.hi_keV if sp.hi_keV is not None else sp.energy_keV,
                                   "fluence": sp.fluence}))
    pd.DataFrame(smeta).to_parquet(OUT / "spectra.parquet", index=False)
    pd.concat(sbins, ignore_index=True).to_parquet(OUT / "spectrum_bins.parquet", index=False)
    variants = build_variants(spec_list)
    spec_by = {sp.name: sp for sp in spec_list}
    e_spec = np.unique(np.concatenate([sp.energy_keV for sp in spec_list]))
    xs_spec = mixture.element_mu_rho_at(e_spec)
    spec_idx = {sp.name: np.searchsorted(e_spec, sp.energy_keV) for sp in spec_list}

    # Pb reference: T* per variant (Pb-thickness targets) and Pb thickness per variant
    rho_pb = lead_density()
    pb_row = np.zeros(100); pb_row[81] = 1.0
    t_star, x_pb = [], []
    for v in variants.itertuples():
        sp = spec_by[v.spectrum]
        mu_pb = rho_pb * xs_spec["total_wcoh" if v.coherent else "total_wocoh"][81, spec_idx[sp.name]]
        w = sp.weights(v.response)
        if v.target_type == "pb_mm":
            xp = v.target_value / 10.0
            ts = float(np.exp(transmission.log_transmission(mu_pb[None, :], w, np.array([xp]))[0]))
        else:
            ts = v.target_value
            xp = float(transmission.solve_thickness(mu_pb[None, :], w, ts,
                                                    config.get("solver.xtol_rel"))[0])
        t_star.append(ts); x_pb.append(xp)
    variants["T_target"] = t_star
    variants["pb_thickness_cm"] = x_pb
    variants["pb_density_g_cm3"] = rho_pb
    variants.to_parquet(OUT / "profile_variants.parquet", index=False)

    # ---- per-chunk mixture rule, key energies, profiles ----
    key_e = np.array(config.get("key_energies_keV"), dtype=float)
    key_i = np.array([int(np.where(np.isclose(e, k, rtol=0, atol=1e-9))[0][0]) for k in key_e])
    mm_w = np.lib.format.open_memmap(OUT / "mu_rho_wcoh.npy", mode="w+", dtype=np.float32, shape=(n, n_e))
    mm_wo = np.lib.format.open_memmap(OUT / "mu_rho_wocoh.npy", mode="w+", dtype=np.float32, shape=(n, n_e))
    prof_schema = pa.schema([("row", pa.int32()), ("variant_id", pa.int16()),
                             ("thickness_req_cm", pa.float32()), ("areal_mass_req_g_cm2", pa.float32()),
                             ("pb_relative_thickness", pa.float32()), ("pb_relative_areal_mass", pa.float32())])
    writer = pq.ParquetWriter(OUT / "profiles.parquet", prof_schema, compression="zstd")
    key_frames = []
    xtol = config.get("solver.xtol_rel")
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        Wc = W[s0:s1]; dc = dens[s0:s1]
        mr = {"wcoh": mixture.mu_rho(Wc, xs["total_wcoh"]), "wocoh": mixture.mu_rho(Wc, xs["total_wocoh"])}
        mm_w[s0:s1] = mr["wcoh"].astype(np.float32); mm_wo[s0:s1] = mr["wocoh"].astype(np.float32)
        # key energies
        part = {k: mixture.mu_rho(Wc, xs[k][:, key_i]) for k in ("coh", "incoh", "pe", "pair_nuc", "pair_el")}
        for j, (ek, ik) in enumerate(zip(key_e, key_i)):
            kf = {"row": np.arange(s0, s1, dtype=np.int32), "energy_keV": ek}
            for tag in ("wcoh", "wocoh"):
                m_r = mr[tag][:, ik]; m_ = m_r * dc
                kf[f"mu_rho_{tag}"] = m_r; kf[f"mu_{tag}"] = m_
                kf[f"hvl_cm_{tag}"] = math.log(2) / m_; kf[f"tvl_cm_{tag}"] = math.log(10) / m_
                kf[f"mfp_cm_{tag}"] = 1.0 / m_
            for k in part:
                kf[f"{k}_mu_rho"] = part[k][:, j]
            key_frames.append(pd.DataFrame(kf))
        # profiles (μ at the spectral quadrature energies)
        ms = {"wcoh": mixture.mu_rho(Wc, xs_spec["total_wcoh"]), "wocoh": mixture.mu_rho(Wc, xs_spec["total_wocoh"])}
        cols = {f: [] for f in prof_schema.names}
        for v in variants.itertuples():
            sp = spec_by[v.spectrum]
            mu_c = ms["wcoh" if v.coherent else "wocoh"][:, spec_idx[sp.name]] * dc[:, None]
            w = sp.weights(v.response)
            x = transmission.solve_thickness(mu_c, w, v.T_target, xtol)
            am = x * dc
            cols["row"].append(np.arange(s0, s1, dtype=np.int32))
            cols["variant_id"].append(np.full(s1 - s0, v.variant_id, dtype=np.int16))
            cols["thickness_req_cm"].append(x.astype(np.float32))
            cols["areal_mass_req_g_cm2"].append(am.astype(np.float32))
            cols["pb_relative_thickness"].append((x / v.pb_thickness_cm).astype(np.float32))
            cols["pb_relative_areal_mass"].append((am / (v.pb_thickness_cm * rho_pb)).astype(np.float32))
        writer.write_table(pa.table({k: np.concatenate(v) for k, v in cols.items()}, schema=prof_schema))
        if verbose:
            print(f"  table C: rows {s0}-{s1} of {n}", flush=True)
    writer.close()
    mm_w.flush(); mm_wo.flush(); del mm_w, mm_wo
    kdf = pd.concat(key_frames, ignore_index=True)
    kdf.insert(1, "material_id", index["material_id"].to_numpy()[kdf["row"].to_numpy()])
    kdf.to_parquet(OUT / "key_energies.parquet", index=False)
    return {"n_compounds": n, "n_energies": int(n_e), "n_variants": len(variants),
            "n_profile_rows": n * len(variants), "n_spectra": len(spec_list)}
