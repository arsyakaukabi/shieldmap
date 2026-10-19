"""Stage 4a the WSL side: OpenMC models, runs and record reduction for the Monte Carlo spot check.

Runs inside the micromamba env "openmc" (OpenMC 0.16.0, conda-forge; see wsl_setup.sh) with the official
OpenMC ENDF/B-VIII.0 HDF5 library, photon sublibrary only (config/sources/endfb_viii0_openmc.yaml).

Usage (from WSL; REPO = /mnt/c/Users/USER/Desktop/shield/shieldmap)::

    python openmc_cases.py endf  --repo $REPO            # ENDF mu/rho + radiative line energies -> JSON
    python openmc_cases.py run   --repo $REPO [--cases 6,1,3] [--deadline 2026-10-15T20:59]

Long runs are launched detached from Windows with Start-Process (see the Monte Carlo checkpoint).

Model (one OpenMC model per material x thickness)
------------------------------------------------
* geometry: slab 0 <= z <= t of the material, lateral planes x, y = +-5 cm REFLECTIVE (laterally infinite
  slab); void 1 cm in front (vacuum boundary, backscatter leaves) and a 1 cm void layer behind (vacuum
  boundary). Nothing behind the slab can scatter photons back.
* source: broad parallel beam at normal incidence, uniform over the 10 x 10 cm face at z = -0.5 cm, photons
  only, energies = the stage 1 quadrature pieces (or the line(s)) sampled with probability proportional to
  their fluence (analog).
* physics: photon transport (coherent, incoherent with Doppler broadening, photoelectric with full atomic
  relaxation from ENDF/B-VIII.0, i.e. K, L, M, ... radiative and non-radiative cascades, pair production);
  electrons and positrons by OpenMC's thick-target bremsstrahlung (TTB) approximation (electron_treatment =
  'ttb'), positrons annihilate into two 511 keV photons; photon energy cutoff 1 keV (OpenMC default).
* tallies: (i) surface-crossing records of every photon crossing the back face z = t (surf_source_write):
  energy, direction cosine u_z, weight -> reduced to records_*.npz; the air-kerma estimator is computed
  later on the Windows side as sum w R(E)/|u_z| with the map's SRD 126 response (compare.py);
  (ii) native cross-check tally: track-length photon flux in the back void layer times the air-kerma
  response (EnergyFunctionFilter, SRD 126 spline tabulated by export.py, log-log interpolation).
* statistics: each point is run as NB independent sub-runs (different seeds); the uncertainty is the
  standard error over sub-runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

NB = 10                      # minimum number of independent sub-runs per point
N_SUB_MAX = 5_000_000        # particles per sub-run at most (bounds the surface-crossing bank in memory)
SEED0 = 20261012
LATERAL = 5.0                # cm, half-width (reflective)
WORK = Path.home() / "mc4a" / "runs"
CROSS_SECTIONS = Path.home() / "endf" / "endfb-viii.0-hdf5" / "cross_sections.xml"
PHOTON_DIR = Path.home() / "endf" / "endfb-viii.0-hdf5" / "photon"


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cases(repo: Path, name: str = "cases.json") -> dict:
    return json.loads((repo / "data/phase4a/mc/inputs" / name).read_text())


def load_all_cases(repo: Path) -> dict:
    """cases.json plus diag_cases.json (diagnostics) when present: union of cases and materials."""
    c = load_cases(repo)
    p = repo / "data/phase4a/mc/inputs/diag_cases.json"
    if p.exists():
        d = json.loads(p.read_text())
        c = dict(c, cases=c["cases"] + d["cases"], materials={**c["materials"], **d["materials"]})
    return c


def seed_for(run_id: str, b: int) -> int:
    return SEED0 + (zlib.crc32(run_id.encode()) % 1_000_000) * 100 + b


# ----------------------------------------------------------------------------------------------
# ENDF quantities (analytic narrow beam with ENDF mu; radiative line energies)
# ----------------------------------------------------------------------------------------------
def _photon(el: str):
    import openmc.data
    return openmc.data.IncidentPhoton.from_hdf5(str(PHOTON_DIR / f"{el}.h5"))


def element_xs_barn(ph, e_eV: np.ndarray) -> dict:
    """Partial cross sections (b/atom) at e_eV, log-log interpolation on the element grid, as OpenMC's
    transport does (photon.cpp: all photon cross sections are interpolated in log-log)."""
    E = np.asarray(ph.energy if hasattr(ph, "energy") else ph[502].xs.x)
    out = {}
    for mt, name in ((502, "coherent"), (504, "incoherent"), (515, "pair_electron"), (517, "pair_nuclear")):
        if mt in ph.reactions:
            out[name] = _loglog(ph.reactions[mt].xs, e_eV)
    pe = np.zeros_like(e_eV, dtype=float)
    for mt, rx in ph.reactions.items():
        if 534 <= mt <= 572:
            pe += _loglog(rx.xs, e_eV)
    # transport uses the total photoelectric cross section (MT 522, dataset "photoelectric") for the total
    # and the subshell cross sections only to pick the shell; the subshell sum is kept as a check
    out["photoelectric"] = _loglog(ph.reactions[522].xs, e_eV) if 522 in ph.reactions else pe
    out["total"] = sum(out.values())
    out["photoelectric_subshell_sum"] = pe
    return out


def _loglog(xs, e_eV):
    x = np.asarray(xs.x, float); y = np.asarray(xs.y, float)
    e = np.asarray(e_eV, float)
    res = np.zeros_like(e)
    ok = (e >= x[0]) & (e <= x[-1])
    pos = y > 0
    xx, yy = x[pos], y[pos]
    if xx.size < 2:
        return res
    inside = ok & (e >= xx[0])
    res[inside] = np.exp(np.interp(np.log(e[inside]), np.log(xx), np.log(yy)))
    return res


def endf_tables(repo: Path) -> None:
    import openmc
    openmc.config["cross_sections"] = str(CROSS_SECTIONS)
    cases = load_all_cases(repo)
    out = {"library": str(CROSS_SECTIONS), "openmc_version": openmc.__version__, "materials": {}, "lines": {}}
    energies = set()
    for c in cases["cases"]:
        energies.update(c["source"]["energies_keV"])
    E = np.array(sorted(energies))
    els = set()
    for k, m in cases["materials"].items():
        mat = _material(m)
        dens = mat.get_element_atom_densities()             # atoms / b-cm
        tot = np.zeros_like(E); parts = {}
        for el, n in dens.items():
            ph = _photon(el); els.add(el)
            xs = element_xs_barn(ph, E * 1e3)
            tot += n * xs["total"]
            for kk, v in xs.items():
                parts.setdefault(kk, np.zeros_like(E)); parts[kk] += n * v
        out["materials"][k] = {"rho_used": m["rho_used"], "energies_keV": E.tolist(),
                               "mu_cm": tot.tolist(), "mu_rho_cm2_g": (tot / m["rho_used"]).tolist(),
                               "partials_mu_cm": {kk: v.tolist() for kk, v in parts.items()},
                               "element_atom_density_per_bcm": {el: float(n) for el, n in dens.items()}}
    for el in sorted(els):
        ph = _photon(el)
        lines = []
        rel = ph.atomic_relaxation
        if rel is not None:
            for sh, tr in rel.transitions.items():
                # columns: secondary, tertiary, energy (eV), probability; tertiary == None -> radiative
                for _, r in tr.iterrows():
                    if pd.isna(r["tertiary"]):
                        lines.append({"shell": sh, "from": str(r["secondary"]), "energy_eV": float(r["energy (eV)"]),
                                      "probability": float(r["probability"])})
            out.setdefault("binding_eV", {})[el] = {sh: float(v) for sh, v in rel.binding_energy.items()}
        out["lines"][el] = lines
    p = repo / "data/phase4a/mc/endf_reference.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
def _material(m: dict):
    import openmc
    mat = openmc.Material(name=m["key"])
    for el, w in m["mass_fractions"].items():
        mat.add_element(el, w, percent_type="wo")
    mat.set_density("g/cm3", m["rho_used"])
    return mat


def build_model(case: dict, run: dict, mat_def: dict, kerma_tab: dict, n_particles: int, seed: int,
                ssw_cap: int):
    import openmc
    openmc.reset_auto_ids()
    mat = _material(mat_def)
    t = float(run["t_cm"])
    xmin = openmc.XPlane(-LATERAL, boundary_type="reflective"); xmax = openmc.XPlane(LATERAL, boundary_type="reflective")
    ymin = openmc.YPlane(-LATERAL, boundary_type="reflective"); ymax = openmc.YPlane(LATERAL, boundary_type="reflective")
    zf = openmc.ZPlane(-1.0, boundary_type="vacuum")
    z0 = openmc.ZPlane(0.0)
    z1 = openmc.ZPlane(t)
    zb = openmc.ZPlane(t + 1.0, boundary_type="vacuum")
    lat = +xmin & -xmax & +ymin & -ymax
    front = openmc.Cell(name="front_void", region=lat & +zf & -z0)
    slab = openmc.Cell(name="slab", fill=mat, region=lat & +z0 & -z1)
    back = openmc.Cell(name="back_void", region=lat & +z1 & -zb)
    geom = openmc.Geometry([front, slab, back])
    src = openmc.IndependentSource()
    src.space = openmc.stats.CartesianIndependent(openmc.stats.Uniform(-LATERAL, LATERAL),
                                                  openmc.stats.Uniform(-LATERAL, LATERAL),
                                                  openmc.stats.Discrete([-0.5], [1.0]))
    src.angle = openmc.stats.Monodirectional((0.0, 0.0, 1.0))
    E = np.array(case["source"]["energies_keV"]) * 1e3
    src.energy = openmc.stats.Discrete(E, np.array(case["source"]["prob"]))
    src.particle = "photon"
    s = openmc.Settings()
    s.run_mode = "fixed source"
    s.source = src
    s.particles = int(n_particles)
    s.batches = 1
    s.photon_transport = True
    s.electron_treatment = "ttb"
    s.cutoff = {"energy_photon": 1000.0}
    s.seed = int(seed)
    s.surf_source_write = {"surface_ids": [z1.id], "max_particles": int(ssw_cap)}
    s.sourcepoint = {"write": False}
    s.output = {"tallies": False, "summary": False}
    # native cross-check tallies
    pf = openmc.ParticleFilter(["photon"])
    ef = openmc.EnergyFunctionFilter(np.array(kerma_tab["energy_eV"]), np.array(kerma_tab["R"]),
                                     interpolation="log-log")
    t1 = openmc.Tally(name="kerma_back_layer")
    t1.filters = [openmc.CellFilter([back]), pf, ef]
    t1.scores = ["flux"]
    t2 = openmc.Tally(name="flux_back_layer")
    t2.filters = [openmc.CellFilter([back]), pf]
    t2.scores = ["flux"]
    t3 = openmc.Tally(name="current_back_face")
    t3.filters = [openmc.SurfaceFilter([z1]), pf]
    t3.scores = ["current"]
    tallies = openmc.Tallies([t1, t2, t3])
    return openmc.model.Model(geometry=geom, materials=openmc.Materials([mat]), settings=s, tallies=tallies), z1.id


def reduce_surface_source(h5path: Path, out_npz: Path, surf_id: int) -> dict:
    import h5py
    with h5py.File(h5path, "r") as f:
        sb = f["source_bank"][()]
    names = sb.dtype.names
    ptype = sb["particle"]
    E = sb["E"].astype(np.float64)
    u = sb["u"]
    uz = u["z"] if u.dtype.names else u[:, 2]
    wgt = sb["wgt"].astype(np.float64)
    sid = sb["surf_id"]
    photon = ptype == 22 if np.any(ptype == 22) else ptype == 1
    keep = photon & (sid == surf_id)
    info = {"n_records": int(E.size), "n_photon": int(photon.sum()), "n_kept": int(keep.sum()),
            "n_other_particle": int((~photon).sum()), "n_backward": int((uz[keep] <= 0).sum()),
            "fields": list(names), "w_min": float(wgt.min()) if wgt.size else None,
            "w_max": float(wgt.max()) if wgt.size else None}
    np.savez_compressed(out_npz, E=E[keep], uz=uz[keep].astype(np.float64), wgt=wgt[keep])
    return info


def run_point(repo: Path, case: dict, run: dict, mat_def: dict, kerma_tab: dict, raw_root: Path,
              out_root: Path, threads: int, deadline: float | None) -> dict:
    rid = run["run_id"]
    wdir = WORK / rid
    rec_dir = raw_root / rid
    out_dir = out_root / rid
    for d in (wdir, rec_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    nb = max(NB, int(np.ceil(run["n_particles"] / N_SUB_MAX)))
    n_sub = int(np.ceil(run["n_particles"] / nb))
    log = []
    for b in range(nb):
        done = out_dir / f"b{b:02d}.json"
        if done.exists():
            continue
        if deadline and time.time() > deadline:
            return {"run_id": rid, "status": "deadline"}
        seed = seed_for(rid, b)
        cap = int(max(1_000_000, 0.5 * n_sub))      # observed crossings <= 5 % of histories; truncation flagged
        model, sid = build_model(case, run, mat_def, kerma_tab, n_sub, seed, cap)
        bdir = wdir / f"b{b:02d}"
        if bdir.exists():
            shutil.rmtree(bdir)
        bdir.mkdir(parents=True)
        model.export_to_model_xml(str(bdir / "model.xml"))
        t0 = time.time()
        env = dict(os.environ, OPENMC_CROSS_SECTIONS=str(CROSS_SECTIONS), OMP_NUM_THREADS=str(threads))
        p = subprocess.run(["openmc", "-s", str(threads)], cwd=bdir, env=env, capture_output=True, text=True)
        (bdir / "openmc.log").write_text(p.stdout + "\n" + p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"{rid} b{b}: openmc failed\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        secs = time.time() - t0
        info = reduce_surface_source(bdir / "surface_source.h5", rec_dir / f"b{b:02d}.npz", sid)
        info["truncated"] = info["n_records"] >= cap
        sp = next(bdir.glob("statepoint.*.h5"))
        nat = native_tallies(sp)
        shutil.copy(bdir / "model.xml", out_dir / f"b{b:02d}_model.xml")
        shutil.copy(sp, out_dir / f"b{b:02d}_statepoint.h5")
        rec = {"run_id": rid, "batch": b, "seed": seed, "n_particles": n_sub, "seconds": round(secs, 2),
               "threads": threads, "surf_id": sid, "ssw_cap": cap, "finished_utc": _now(), **info, "native": nat}
        done.write_text(json.dumps(rec, indent=1))
        (bdir / "surface_source.h5").unlink()
        log.append(rec)
        print(f"{_now()} {rid} b{b:02d} n={n_sub} {secs:.1f}s records={info['n_kept']} trunc={info['truncated']}",
              flush=True)
    return {"run_id": rid, "status": "done", "new_subruns": len(log)}


def native_tallies(sp_path: Path) -> dict:
    import openmc
    with openmc.StatePoint(str(sp_path)) as sp:
        out = {}
        for name in ("kerma_back_layer", "flux_back_layer", "current_back_face"):
            t = sp.get_tally(name=name)
            out[name] = float(np.asarray(t.mean).ravel()[0])
        out["runtime"] = {k: float(v) for k, v in sp.runtime.items()} if hasattr(sp, "runtime") else {}
    return out


def run_all(repo: Path, cases_sel, deadline: float | None, threads: int, inputs: str = "cases.json") -> None:
    cases = load_cases(repo, inputs)
    kt = cases["kerma_tab"]
    raw_root = repo / "data/raw/mc4a/records"
    out_root = repo / "data/phase4a/mc/runs"
    order = sorted(cases["cases"], key=lambda c: c["priority"])
    if cases_sel:
        order = [c for c in order if c["case"] in cases_sel]
        order.sort(key=lambda c: cases_sel.index(c["case"]))
    for c in order:
        # within a case, run the headline factor first for every material, then the brackets
        runs = sorted(c["runs"], key=lambda r: (abs(r["factor"] - 1.0) > 1e-9, r["factor"], r["material"]))
        for r in runs:
            res = run_point(repo, c, r, cases["materials"][r["material"]], kt, raw_root, out_root, threads, deadline)
            print(_now(), res, flush=True)
            if res["status"] == "deadline":
                print("DEADLINE REACHED", flush=True)
                return
    print(_now(), "ALL_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["endf", "run"])
    ap.add_argument("--repo", default="/mnt/c/Users/USER/Desktop/shield/shieldmap")
    ap.add_argument("--cases", default="")
    ap.add_argument("--deadline", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--inputs", default="cases.json", help="cases.json (handover cases) or diag_cases.json")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    if a.cmd == "endf":
        endf_tables(repo)
        return
    sel = [int(x) for x in a.cases.split(",") if x.strip()]
    dl = dt.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else None
    run_all(repo, sel, dl, a.threads, a.inputs)


if __name__ == "__main__":
    sys.exit(main())
