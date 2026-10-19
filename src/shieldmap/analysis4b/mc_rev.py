"""One more
Monte Carlo case, c7_rqr6, at the paper's reference beam.

Beam: RQR6, 12° anode angle, Pb target 0.35 mm (DIAG variant 13). Materials: Pb, Bi, Ce5AgSn3 (narrow-beam B4
frontier at variant 13), PrSbPd (first-order-corrected leader at variant 13, fig4b_advantage_by_variant.csv),
Ba2SmSbO6 (best AMB-ANION ∧ AMB-ICSD3 lanthanide compound at variant 13, e5_amb_lanthanide.csv) and the best
Sn/Ba/Gd elemental blend at variant 13 (e2_blend_by_variant.csv) as a homogeneous mixture at its mass fractions.

Everything reuses the stage 4a machinery (read only; nothing in src/shieldmap/analysis4 or data/phase4a is written):
  * case building     analysis4.mc.export (material_record, source_record, n_particles, FACTORS, _point_density)
                      and the per-case loop of export.build copied for one areal spectrum case (export.build is tied
                      to its own MATERIALS table and writes to data/phase4a);
  * transport         analysis4.mc.openmc_cases (build_model / run_point, unchanged model, seeds, sub-run policy),
                      called from the WSL side of this module with output roots under data/phase4b/mc_rev and
                      data/raw/mc4b;
  * reduction         analysis4.mc.compare (subrun_sums, points_table, requirements, _solve) with its module paths
                      temporarily pointed at the new folders; metrics as compare.metrics (pairs differ);
  * like-for-like     analysis4b.revision (_interp_T + compare._solve; joint bootstrap over all slabs).

Run (from shieldmap/):
  Windows   .venv\\Scripts\\python.exe -m shieldmap.analysis4b.mc_rev export
  WSL       (wrapper: src/shieldmap/analysis4b/mc_rev_wsl.sh endf | stubs | run [max_seconds])
            micromamba run -n openmc python src/shieldmap/analysis4b/mc_rev.py wsl-endf
            micromamba run -n openmc python src/shieldmap/analysis4b/mc_rev.py wsl-stubs   (neutron placeholders)
            micromamba run -n openmc python src/shieldmap/analysis4b/mc_rev.py wsl-run --max-seconds 480   (resumable)
  Windows   .venv\\Scripts\\python.exe -m shieldmap.analysis4b.mc_rev compare

Outputs: data/phase4b/mc_rev/ (inputs/, runs/ sub-run metadata, endf_reference.json, subrun_sums.csv, points.csv,
requirements.csv, metrics.csv, like_for_like.csv, chi2.json, summary.json, digest.md, records_manifest.json);
raw back-face records, statepoints and model XML under data/raw/mc4b/ (git-ignored).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CASE = {"case": 7, "id": "c7_rqr6", "priority": 7, "basis": "areal",
        "title": "RQR6 (80 kV, 12 deg) at the 0.35 mm Pb target (variant 13, reference beam): Ce5AgSn3, PrSbPd, "
                 "Ba2SmSbO6 and the Sn/Ba/Gd blend vs Bi vs Pb",
        "source": ("spectrum", "RQR6_th12"), "pb_mm": 0.35, "variant_id": 13}
MATS = {"Pb": "mp-20483", "Bi": "mp-23152", "Ce5AgSn3": "mp-1213880", "PrSbPd": "mp-1018899",
        "Ba2SmSbO6": "mp-6037", "SnBaGd_blend": None}
BLEND = "SnBaGd_blend"
PARTNERS = ("Ce5AgSn3", "PrSbPd", "Ba2SmSbO6", BLEND)
ESTIMATORS_LFL = ("T_tablec", "T_endf_narrow", "T_firstorder", "narrow", "cone1", "cone3", "fl", "fl_all", "broad",
                  "broad_hi")
N_BOOT = 4000
BOOT_SEED = 20261013

HERE = Path(__file__).resolve().parent


def _repo() -> Path:
    return HERE.parents[2]


def _dirs(repo: Path | None = None) -> dict:
    repo = repo or _repo()
    out = repo / "data" / "phase4b" / "mc_rev"
    raw = repo / "data" / "raw" / "mc4b"
    return {"out": out, "inputs": out / "inputs", "runs": out / "runs", "raw": raw, "rec": raw / "records",
            "raw_runs": raw / "runs"}


# ============================================================================================ Windows: export
def _verify_choices() -> dict:
    """Check the material choices against the revision outputs and Fig. 4 data; return the blend definition."""
    import pandas as pd
    from shieldmap import paths
    rev = paths.DATA / "phase4b" / "revision"
    b = pd.read_csv(rev / "e2_blend_by_variant.csv").set_index("variant_id").loc[13]
    ln = pd.read_csv(rev / "e5_amb_lanthanide.csv")
    ln = ln[(ln.tier == "AMB-ANION&ICSD3") & (ln.variant_id == 13)].iloc[0]
    f4 = pd.read_csv(paths.DATA / "phase4b" / "fig4b_advantage_by_variant.csv").set_index("variant_id").loc[13]
    checks = {"B4 frontier v13": (b.B4_frontier, "Ce5AgSn3"), "fig4b best_ln_narrow v13": (f4.best_ln_narrow, "Ce5AgSn3"),
              "fig4b best_ln_corrected v13": (f4.best_ln_corrected, "PrSbPd"),
              "AMB-ANION&ICSD3 lanthanide leader v13": (ln.leader, "Ba2SmSbO6"),
              "AMB-ANION&ICSD3 leader mp-id": (ln.leader_mpid, "mp-6037")}
    for k, (got, want) in checks.items():
        if got != want:
            raise RuntimeError(f"{k}: expected {want}, found {got}")
    if b.best_blend != b.best_ternary:
        raise RuntimeError("best blend at v13 is not the ternary")
    fr = {}
    for part in str(b.best_blend).split(" + "):
        el = part.rstrip("0123456789.")
        fr[el] = float(part[len(el):])
    if abs(sum(fr.values()) - 1) > 1e-9:
        raise RuntimeError(f"blend fractions do not sum to 1: {fr}")
    return {"fractions": fr, "label": str(b.best_blend), "q_blend_e2": float(b.q_blend), "q_B4_e2": float(b.q_B4),
            "q_Bi_e2": float(b.q_Bi), "blend_vs_B4_e2": float(b.blend_vs_B4), "blend_vs_Bi_e2": float(b.blend_vs_Bi),
            "T_target_e2": float(b.T_target), "ln_q_leader_e5": float(ln.q_leader), "ln_vs_Bi_e5": float(ln.vs_Bi),
            "ln_vs_B4_e5": float(ln.vs_B4), "fig4_adv_narrow": float(f4.adv_narrow),
            "fig4_adv_corrected": float(f4.adv_corrected_estimate), "fig4_best_textbook_narrow": str(f4.best_textbook_narrow),
            "fig4_best_textbook_corrected": str(f4.best_textbook_corrected)}


def blend_record(fr: dict) -> dict:
    """Homogeneous Sn/Ba/Gd mixture at the E2 mass fractions. Density: 1/ρ = Σ w_Z / ρ_Z with each element's
    measured-density point (export._point_density on its R2 element row). Areal-mass results do not depend on it."""
    import xraylib
    from shieldmap.analysis3 import common
    from shieldmap.analysis4.mc import export
    u = common.universe()
    base = common.base_mask("R2")
    nz = common.mass_fractions().shape[1]
    W = np.zeros(nz)
    el_rho, el_src, el_mp = {}, {}, {}
    for el, w in fr.items():
        r = np.flatnonzero(base & (u["formula_pretty"].to_numpy() == el))
        if r.size != 1:
            raise RuntimeError(f"{el}: {r.size} R2 rows")
        rho, src = export._point_density(np.array([int(r[0])]))
        el_rho[el], el_src[el], el_mp[el] = float(rho[0]), str(src[0]), str(u["material_id"].iloc[int(r[0])])
        W[xraylib.SymbolToAtomicNumber(el) - 1] = w
    rho = 1.0 / sum(w / el_rho[el] for el, w in fr.items())
    return {"key": BLEND, "row": -1, "material_id": "blend(" + "+".join(f"{el_mp[e]}" for e in fr) + ")",
            "formula": " + ".join(f"{e} {w:.3f}" for e, w in fr.items()) + " (mass fractions)",
            "theoretical": False, "e_hull": None, "rho_mp": None, "rho_used": rho,
            "rho_source": "inverse-density mixture of element measured-density points: "
                          + ", ".join(f"{e} {el_rho[e]:.4g} ({el_src[e]})" for e in fr),
            "mass_fractions": dict(fr), "W": W}


def build() -> tuple[dict, dict, dict]:
    """One areal spectrum case, exactly as export.build builds c3/c4 (loop body copied for this case only)."""
    import pandas as pd
    from shieldmap.analysis3 import common
    from shieldmap.analysis3 import fluorescence as fl
    from shieldmap.analysis4.mc import export
    choice = _verify_choices()
    mats = {k: (export.material_record(k, common.row_of(mp)) if mp else blend_record(choice["fractions"]))
            for k, mp in MATS.items()}
    c = dict(CASE, materials=list(MATS), factors=export.FACTORS)
    F = pd.read_parquet(common.P3 / "F" / "corrected_areal.parquet")
    fsum = json.loads((common.P3 / "F" / "summary.json").read_text(encoding="utf-8"))
    src = export.source_record(c["source"])
    e = np.array(src["energies_keV"]); phi = np.array(src["fluence"])
    Wm = np.vstack([mats[k]["W"] for k in c["materials"]])
    m_fl = fl.FluorescenceModel(Wm, e, phi)
    m_nb = fl.FluorescenceModel(Wm, e, phi, fluorescence=False)
    m_K = fl.FluorescenceModel(Wm, e, phi, l_shell=False)
    m_E2 = fl.FluorescenceModel(Wm, e, phi, escape="E2")
    m_nc = fl.FluorescenceModel(Wm, e, phi)
    for em in m_nc.emitters:
        em.b = m_nc.W[em.rows] @ fl._xs_at(em.energy)["total_wocoh"]
    T_target = float(fsum["pb"]["T_star"][str(c["variant_id"])])
    n = len(c["materials"])
    q_nb = m_nb.solve_narrow(np.full(n, T_target))
    q_fl = m_fl.solve_effective(np.full(n, T_target), lo=q_nb)
    q_nc = m_nc.solve_effective(np.full(n, T_target), lo=q_nb)
    # Pb's first-order transmission at its nominal areal mass (like-for-like first-order target)
    v = common.variants().set_index("variant_id")
    q_pb_nom = float(v.at[c["variant_id"], "pb_thickness_cm"] * v.at[c["variant_id"], "pb_density_g_cm3"])
    j_pb = c["materials"].index("Pb")
    T_pb_fo = float(m_fl.transmission(np.array([q_pb_nom]), idx=np.array([j_pb]))[0])
    q_fl_lfl = m_fl.solve_effective(np.full(n, T_pb_fo), lo=m_nb.solve_narrow(np.full(n, T_pb_fo)))
    runs, mref = [], {}
    for j, k in enumerate(c["materials"]):
        mat = mats[k]
        rho = mat["rho_used"]
        q0 = float(q_fl[j])
        r = {"key": k, "q_narrow_tablec_g_cm2": float(q_nb[j]), "q_firstorder_g_cm2": float(q_fl[j]),
             "q_firstorder_nocohlines_g_cm2": float(q_nc[j]), "q_firstorder_likeforlike_g_cm2": float(q_fl_lfl[j]),
             "t_narrow_tablec_cm": float(q_nb[j]) / rho, "q0_g_cm2": q0, "rho_used": rho, "points": []}
        if mat["row"] >= 0:
            f = F[(F.variant_id == c["variant_id"]) & (F.row == mat["row"])]
            if len(f):
                r["phase3_corrected_areal"] = {kk: float(f.iloc[0][kk]) for kk in
                                               ("q_tablec", "q_narrow", "q_eff_fixedT", "q_eff_likeforlike")}
        facs = [float(x) for x in c["factors"]]
        if float(q_nb[j]) / q0 < 0.85:
            for f2 in c["factors"]:
                x = round(float(q_nb[j]) * f2 / q0, 3)
                if all(abs(x / y - 1) > 0.03 for y in facs):
                    facs.append(x)
        a_j = m_nb.a[j]
        for fct in sorted(facs):
            q = q0 * fct
            Tf, Tp, Tfl, parts = m_fl.transmission(np.array([q]), idx=np.array([j]), parts=True)
            pt = {"factor": fct, "q_g_cm2": q, "t_cm": q / rho,
                  "T_tablec_narrow": float(m_nb.primary(np.array([q]), idx=np.array([j]))[0]),
                  "T_firstorder": float(Tf[0]), "T_firstorder_fluor": float(Tfl[0]),
                  "T_firstorder_fluor_by_Z": {int(z): float(vv[0]) for z, vv in parts.items()},
                  "T_firstorder_Konly": float(m_K.transmission(np.array([q]), idx=np.array([j]))[0]),
                  "T_firstorder_E2": float(m_E2.transmission(np.array([q]), idx=np.array([j]))[0]),
                  "T_firstorder_nocohlines": float(m_nc.transmission(np.array([q]), idx=np.array([j]))[0]),
                  "n_particles": export.n_particles(src, a_j, q)}
            r["points"].append(pt)
            runs.append({"run_id": f"{c['id']}__{k}__f{fct:.3f}", "case": c["case"], "material": k,
                         "factor": fct, "t_cm": q / rho, "n_particles": pt["n_particles"]})
        mref[k] = r
    case = {"case": c["case"], "id": c["id"], "title": c["title"], "priority": c["priority"], "basis": c["basis"],
            "T_target": T_target, "source": src, "materials": c["materials"], "pb_mm": c["pb_mm"],
            "variant_id": c["variant_id"], "runs": runs}
    refs = {c["id"]: {"T_target": T_target, "q_Pb_nominal": q_pb_nom, "T_Pb_firstorder_nominal": T_pb_fo,
                      "materials": mref}}
    mats_out = {k: {kk: vv for kk, vv in m.items() if kk != "W"} for k, m in mats.items()}
    return {"materials": mats_out, "cases": [case]}, refs, choice


def export_inputs() -> None:
    from shieldmap.analysis4.mc import export, mcresp
    d = _dirs()
    d["inputs"].mkdir(parents=True, exist_ok=True)
    cases, refs, choice = build()
    cases["air_response"] = export.air_response_table()
    cases["kerma_tab"] = mcresp.kerma_tab()
    (d["inputs"] / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    (d["inputs"] / "reference_values.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
    (d["inputs"] / "material_choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    c = cases["cases"][0]
    print(c["id"], "T* =", c["T_target"], "n energies", len(c["source"]["energies_keV"]))
    for r in c["runs"]:
        print(f"  {r['run_id']:40s} t={r['t_cm']:.5f} cm  n={r['n_particles']:.3g}")
    print("total particles", sum(r["n_particles"] for r in c["runs"]))
    for k, m in cases["materials"].items():
        ref = refs[c["id"]]["materials"][k]
        print(k, m["formula"], m["material_id"], round(m["rho_used"], 4), m["rho_source"],
              "q_nb", round(ref["q_narrow_tablec_g_cm2"], 5), "q_fl", round(ref["q_firstorder_g_cm2"], 5))


# ============================================================================================ WSL: transport
def _oc(repo: Path):
    sys.path.insert(0, str(repo / "src" / "shieldmap" / "analysis4" / "mc"))
    import openmc_cases                                  # Stage 4a model builder and runner (unchanged)
    return openmc_cases


def wsl_endf(repo: Path) -> None:
    """ENDF μ at the source energies and radiative line energies of every constituent (openmc_cases.endf_tables
    logic for this case; that function writes to data/phase4a)."""
    import openmc
    import pandas as pd
    oc = _oc(repo)
    d = _dirs(repo)
    openmc.config["cross_sections"] = str(oc.CROSS_SECTIONS)
    cases = json.loads((d["inputs"] / "cases.json").read_text())
    out = {"library": str(oc.CROSS_SECTIONS), "openmc_version": openmc.__version__, "materials": {}, "lines": {}}
    E = np.array(sorted(set(cases["cases"][0]["source"]["energies_keV"])))
    els = set()
    for k, m in cases["materials"].items():
        mat = oc._material(m)
        dens = mat.get_element_atom_densities()
        tot = np.zeros_like(E); parts = {}
        for el, nn in dens.items():
            ph = oc._photon(el); els.add(el)
            xs = oc.element_xs_barn(ph, E * 1e3)
            tot += nn * xs["total"]
            for kk, vv in xs.items():
                parts.setdefault(kk, np.zeros_like(E)); parts[kk] += nn * vv
        out["materials"][k] = {"rho_used": m["rho_used"], "energies_keV": E.tolist(), "mu_cm": tot.tolist(),
                               "mu_rho_cm2_g": (tot / m["rho_used"]).tolist(),
                               "partials_mu_cm": {kk: vv.tolist() for kk, vv in parts.items()},
                               "element_atom_density_per_bcm": {el: float(nn) for el, nn in dens.items()}}
    for el in sorted(els):
        ph = oc._photon(el)
        lines = []
        rel = ph.atomic_relaxation
        if rel is not None:
            for sh, tr in rel.transitions.items():
                for _, r in tr.iterrows():
                    if pd.isna(r["tertiary"]):
                        lines.append({"shell": sh, "from": str(r["secondary"]), "energy_eV": float(r["energy (eV)"]),
                                      "probability": float(r["probability"])})
            out.setdefault("binding_eV", {})[el] = {sh: float(v) for sh, v in rel.binding_energy.items()}
        out["lines"][el] = lines
    p = d["out"] / "endf_reference.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, sorted(els))


XS4B = Path.home() / "mc4b" / "xs"          # library view for this case (photon files and stage 4a neutron files linked)
STUB_TEMPLATE = "I135"                       # smallest existing incident-neutron file (0.9 MB); structure only


def wsl_stubs(repo: Path) -> None:
    """Neutron placeholder files for the nuclides of the new elements (Ce, Ag, Sn, Ba, Sm, Gd).

    OpenMC 0.16 reads an incident-neutron file for every nuclide of a material even in a photon-only fixed-source
    run; the neutron data are never sampled (the source emits photons only; no photonuclear physics in 0.16).
    The ENDF/B-VIII.0 archive was deleted after stage 4a extracted the photon library and the neutron files of the
    Stage 4a materials, and the decision rule forbids new downloads. So each missing nuclide gets a placeholder:
    the HDF5 structure of an existing neutron file (STUB_TEMPLATE) under the nuclide's own group name with its own
    Z, A, metastable = 0 and atomic_weight_ratio = atomic_mass/neutron mass (openmc.data, the values the Python
    side uses to expand elements; the only neutron-file quantity the photon run uses is the AWR, in the g/cm³ and
    wo → atom-density conversion). The verified library under ~/endf is not modified: a separate view ~/mc4b/xs
    links its photon/ directory and existing neutron files and holds the placeholders and a copy of
    cross_sections.xml. The narrow-beam MC vs ENDF-analytic check (atom densities from the Python side) tests it."""
    import h5py
    import openmc.data as od
    oc = _oc(repo)
    lib = Path(oc.CROSS_SECTIONS).parent
    cases = json.loads((_dirs(repo)["inputs"] / "cases.json").read_text())
    (XS4B / "neutron").mkdir(parents=True, exist_ok=True)
    if not (XS4B / "photon").exists():
        (XS4B / "photon").symlink_to(lib / "photon")
    shutil.copy(lib / "cross_sections.xml", XS4B / "cross_sections.xml")
    for f in (lib / "neutron").glob("*.h5"):
        dst = XS4B / "neutron" / f.name
        if not dst.exists():
            dst.symlink_to(f)
    els = sorted({el for m in cases["materials"].values() for el in m["mass_fractions"]})
    made = []
    tmpl = lib / "neutron" / f"{STUB_TEMPLATE}.h5"
    for el in els:
        for nuc, _ in od.isotopes(el):
            dst = XS4B / "neutron" / f"{nuc}.h5"
            if (lib / "neutron" / f"{nuc}.h5").exists():
                continue
            z, a, _m = od.zam(nuc)
            awr = od.atomic_mass(nuc) / od.NEUTRON_MASS
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            with h5py.File(tmpl, "r") as fs, h5py.File(dst, "w") as fd:
                for k, v in fs.attrs.items():
                    fd.attrs[k] = v
                fs.copy(fs[STUB_TEMPLATE], fd, name=nuc)
                g = fd[nuc]
                g.attrs["Z"] = np.int64(z); g.attrs["A"] = np.int64(a); g.attrs["metastable"] = np.int64(0)
                g.attrs["atomic_weight_ratio"] = np.float64(awr)
            made.append({"nuclide": nuc, "Z": z, "A": a, "atomic_weight_ratio": awr,
                         "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
    rec = {"library_view": str(XS4B), "verified_library": str(lib), "template": f"{STUB_TEMPLATE}.h5",
           "elements": els, "placeholders": made, "created_utc": oc._now()}
    (_dirs(repo)["out"] / "neutron_placeholders.json").write_text(json.dumps(rec, indent=1))
    print("placeholders:", [m["nuclide"] for m in made])


def wsl_run(repo: Path, max_seconds: float, threads: int) -> None:
    oc = _oc(repo)
    oc.CROSS_SECTIONS = XS4B / "cross_sections.xml"      # run_point reads the module global at call time
    if not oc.CROSS_SECTIONS.exists():
        raise RuntimeError("run wsl-stubs first")
    d = _dirs(repo)
    for k in ("rec", "runs", "raw_runs"):
        d[k].mkdir(parents=True, exist_ok=True)
    cases = json.loads((d["inputs"] / "cases.json").read_text())
    kt = cases["kerma_tab"]
    c = cases["cases"][0]
    deadline = time.time() + max_seconds
    t0 = time.time()
    runs = sorted(c["runs"], key=lambda r: (abs(r["factor"] - 1.0) > 1e-9, r["factor"], r["material"]))
    status = "ALL_DONE"
    for r in runs:
        res = oc.run_point(repo, c, r, cases["materials"][r["material"]], kt, d["rec"], d["runs"], threads, deadline)
        # statepoints and model XML are bulky: keep them with the raw records (git-ignored)
        src_dir = d["runs"] / r["run_id"]
        dst = d["raw_runs"] / r["run_id"]
        for f in list(src_dir.glob("b*_statepoint.h5")) + list(src_dir.glob("b*_model.xml")):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst / f.name))
        print(oc._now(), res, flush=True)
        if res["status"] == "deadline":
            status = "DEADLINE"
            break
    print(f"{status} wall {time.time() - t0:.0f}s", flush=True)


# ============================================================================================ Windows: reduction
@contextlib.contextmanager
def _compare_paths():
    """Point analysis4.mc.compare's module paths (MC, REC) at the new folders for the duration of a call."""
    from shieldmap.analysis4.mc import compare
    d = _dirs()
    old = compare.MC, compare.REC
    compare.MC, compare.REC = d["out"], d["rec"]
    try:
        yield compare
    finally:
        compare.MC, compare.REC = old


def _load():
    d = _dirs()
    cases = json.loads((d["inputs"] / "cases.json").read_text(encoding="utf-8"))
    refs = json.loads((d["inputs"] / "reference_values.json").read_text(encoding="utf-8"))
    endf = json.loads((d["out"] / "endf_reference.json").read_text(encoding="utf-8"))
    choice = json.loads((d["inputs"] / "material_choice.json").read_text(encoding="utf-8"))
    return cases, refs, endf, choice


def _pairs():
    out = []
    for x in PARTNERS:
        out.append(("Bi", x, f"Bi/{x} − 1"))
    for x in PARTNERS:
        out.append(("Pb", x, f"Pb/{x} − 1"))
    out += [("Pb", "Bi", "Pb/Bi − 1"), ("Ba2SmSbO6", "Ce5AgSn3", "Ba2SmSbO6/Ce5AgSn3 − 1"),
            (BLEND, "Ce5AgSn3", f"{BLEND}/Ce5AgSn3 − 1"), ("PrSbPd", "Ce5AgSn3", "PrSbPd/Ce5AgSn3 − 1")]
    return out


def metrics_fixed(cases, req):
    """compare.metrics for this case's pairs (X/Y − 1 of areal requirements at the fixed narrow-beam Pb target;
    SE by first-order propagation of the bootstrap SEs, as compare.metrics)."""
    import pandas as pd
    c = cases["cases"][0]
    out = []
    for a, b, label in _pairs():
        for est in req.estimator.unique():
            ra = req[(req.material == a) & (req.estimator == est)]
            rb = req[(req.material == b) & (req.estimator == est)]
            if ra.empty or rb.empty:
                continue
            x, y = float(ra.q_req_g_cm2.iloc[0]), float(rb.q_req_g_cm2.iloc[0])
            sx, sy = float(ra.q_req_se.iloc[0]), float(rb.q_req_se.iloc[0])
            v = x / y - 1.0
            se = (x / y) * math.hypot(sx / x, sy / y) if np.isfinite(sx) and np.isfinite(sy) else np.nan
            out.append({"case": c["case"], "case_id": c["id"], "target": "fixed", "metric": label, "num": a, "den": b,
                        "estimator": est, "basis": "areal", "value": v, "se": se,
                        "inside_bracket": bool(ra.inside_bracket.iloc[0] and rb.inside_bracket.iloc[0])})
    return pd.DataFrame(out)


def like_for_like(cases, refs, pts):
    """Each material's requirement at Pb's own transmission under the same estimator at the nominal 0.35 mm
    (revision._interp_T + compare._solve); joint bootstrap over the sub-run SEs of every slab of every material."""
    import pandas as pd
    from shieldmap.analysis4.mc import compare
    from shieldmap.analysis4b import revision
    c = cases["cases"][0]
    rf = refs[c["id"]]
    q_pb = float(rf["q_Pb_nominal"])
    mats = c["materials"]
    rng = np.random.default_rng(BOOT_SEED)
    P = {m: pts[pts.material == m].sort_values("q_g_cm2") for m in mats}
    reqs, mets = [], []
    for est in ESTIMATORS_LFL:
        has_se = est + "_se" in pts.columns

        def one(draw=None):
            T = {m: P[m][est].to_numpy(float) for m in mats}
            if draw is not None:
                T = {m: np.maximum(T[m] + draw[m] * P[m][est + "_se"].to_numpy(float), 1e-300) for m in mats}
            t_pb = revision._interp_T(P["Pb"].q_g_cm2.to_numpy(float), T["Pb"], q_pb)
            res = {m: compare._solve(P[m].q_g_cm2.to_numpy(float), T[m], t_pb) for m in mats}
            return t_pb, {m: v[0] for m, v in res.items()}, {m: v[1] for m, v in res.items()}
        t_pb, q, inside = one()
        if has_se:
            B = [one({m: rng.standard_normal(len(P[m])) for m in mats}) for _ in range(N_BOOT)]
            bt = np.array([b[0] for b in B]); bq = {m: np.array([b[1][m] for b in B]) for m in mats}
        else:
            bt, bq = None, None
        for m in mats:
            reqs.append({"case": c["case"], "case_id": c["id"], "target": "like-for-like", "material": m, "estimator": est,
                         "T_target": t_pb, "T_target_se": float(bt.std(ddof=1)) if bt is not None else np.nan,
                         "q_req_g_cm2": q[m], "q_req_se": float(bq[m].std(ddof=1)) if bq is not None else np.nan,
                         "t_req_cm": q[m] / rf["materials"][m]["rho_used"], "inside_bracket": bool(inside[m]),
                         "n_points": len(P[m]), "n_boot": N_BOOT if has_se else 0})
        for a, b, label in _pairs():
            xa = q_pb if a == "Pb" else q[a]
            v = xa / q[b] - 1
            if bq is not None:
                xb = q_pb if a == "Pb" else bq[a]
                se = float((xb / bq[b] - 1).std(ddof=1))
            else:
                se = np.nan
            mets.append({"case": c["case"], "case_id": c["id"], "target": "like-for-like", "metric": label, "num": a,
                         "den": b, "estimator": est, "basis": "areal", "value": v, "se": se,
                         "inside_bracket": bool(inside[b] and (a == "Pb" or inside[a])),
                         "T_Pb_same_estimator": t_pb, "q_Pb_nominal": q_pb})
    return pd.DataFrame(reqs), pd.DataFrame(mets)


def chi2(pts) -> dict:
    from scipy import stats
    z = pts.z_narrow_vs_endf.to_numpy(float)
    x2 = float((z ** 2).sum())
    n = int(z.size)
    return {"chi2_narrow_vs_endf": x2, "dof": n, "p_value": float(stats.chi2.sf(x2, n)),
            "max_abs_z": float(np.abs(z).max()), "n_beyond_2sigma": int((np.abs(z) > 2).sum()),
            "ratio_narrow_mc_tablec_min": float(pts.ratio_narrow_mc_tablec.min()),
            "ratio_narrow_mc_tablec_max": float(pts.ratio_narrow_mc_tablec.max()),
            "ratio_endf_tablec_min": float(pts.ratio_endf_tablec.min()),
            "ratio_endf_tablec_max": float(pts.ratio_endf_tablec.max()),
            "native_over_records_min": float(pts.native_over_records.min()),
            "native_over_records_max": float(pts.native_over_records.max()),
            "any_truncated": bool(pts.any_truncated.any())}


def records_manifest() -> dict:
    from shieldmap import paths
    d = _dirs()
    out = {}
    for p in sorted(d["rec"].rglob("*.npz")):
        out[str(p.relative_to(paths.ROOT)).replace("\\", "/")] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                                                   "bytes": p.stat().st_size}
    return out


def _fmt(x, se=None, pct=False, nd=4):
    if x is None or not np.isfinite(x):
        return "–"
    k = 100 if pct else 1
    s = f"{k * x:.2f}" if pct else f"{x:.{nd + 1}g}"
    if se is not None and np.isfinite(se):
        s += f" ± {k * se:.2f}" if pct else f" ± {se:.2g}"
    return s


def digest(cases, pts, req, met, lreq, lmet, x2) -> str:
    from shieldmap.analysis4.mc import compare
    base = compare.digest(cases, pts, req, met[met.target == "fixed"].drop(columns="target"))
    L = [base.replace("# MC spot check: digest (generated by compare.py; do not edit)",
                      "# MC case c7_rqr6: digest (generated by analysis4b.mc_rev; do not edit)"), "",
         f"Narrow MC vs ENDF analytic over the {x2['dof']} slabs: χ² = {x2['chi2_narrow_vs_endf']:.2f} / {x2['dof']} "
         f"(p = {x2['p_value']:.3f}); max |z| = {x2['max_abs_z']:.2f}; {x2['n_beyond_2sigma']} beyond 2σ. "
         f"MC narrow / Table C {x2['ratio_narrow_mc_tablec_min']:.4f}–{x2['ratio_narrow_mc_tablec_max']:.4f}.", "",
         "## Like-for-like target (Pb's own transmission under the same estimator at 0.3969 g/cm²)", "",
         "| estimator | T(Pb) | " + " | ".join(cases["cases"][0]["materials"]) + " |",
         "|" + "---|" * (2 + len(cases["cases"][0]["materials"]))]
    for est in ESTIMATORS_LFL:
        g = lreq[lreq.estimator == est]
        cells = [_fmt(g[g.material == m].q_req_g_cm2.iloc[0], g[g.material == m].q_req_se.iloc[0])
                 + ("" if g[g.material == m].inside_bracket.iloc[0] else "*") for m in cases["cases"][0]["materials"]]
        L.append(f"| {est} | {g.T_target.iloc[0]:.5f} | " + " | ".join(cells) + " |")
    L += ["", "Metrics, like-for-like (X/Y − 1, %):", "", "| metric | " + " | ".join(ESTIMATORS_LFL) + " |",
          "|" + "---|" * (1 + len(ESTIMATORS_LFL))]
    for lab in lmet.metric.unique():
        cells = []
        for est in ESTIMATORS_LFL:
            x = lmet[(lmet.metric == lab) & (lmet.estimator == est)].iloc[0]
            cells.append(_fmt(x.value, x.se, pct=True) + ("" if x.inside_bracket else "*"))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def compare_all() -> dict:
    import pandas as pd
    d = _dirs()
    cases, refs, endf, choice = _load()
    with _compare_paths() as compare:
        if d["rec"].exists() and any(d["rec"].rglob("*.npz")):
            s = compare.subrun_sums(cases, endf)
            s.to_csv(d["out"] / "subrun_sums.csv", index=False)
            (d["out"] / "records_manifest.json").write_text(json.dumps(records_manifest(), indent=1), encoding="utf-8")
        else:
            s = pd.read_csv(d["out"] / "subrun_sums.csv")
        pts = compare.points_table(cases, refs, endf, s)
        req = compare.requirements(cases, refs, pts)
    pts.to_csv(d["out"] / "points.csv", index=False)
    req.insert(2, "target", "fixed")
    lreq, lmet = like_for_like(cases, refs, pts)
    allreq = pd.concat([req, lreq], ignore_index=True)
    allreq.to_csv(d["out"] / "requirements.csv", index=False)
    met = metrics_fixed(cases, req)
    allmet = pd.concat([met, lmet], ignore_index=True)
    allmet.to_csv(d["out"] / "metrics.csv", index=False)
    x2 = chi2(pts)
    (d["out"] / "chi2.json").write_text(json.dumps(x2, indent=1), encoding="utf-8")
    summ = {"case_id": CASE["id"], "n_subruns": int(len(s)), "n_points": int(len(pts)),
            "total_particles": int(s.n_particles.sum()), "total_seconds_openmc_wall": float(s.seconds.sum()),
            "materials": cases["cases"][0]["materials"], "T_target_fixed": cases["cases"][0]["T_target"],
            "q_Pb_nominal": refs[CASE["id"]]["q_Pb_nominal"], "chi2": x2, "material_choice": choice,
            "blend_density_g_cm3": cases["materials"][BLEND]["rho_used"],
            "blend_density_rule": cases["materials"][BLEND]["rho_source"]}
    (d["out"] / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    (d["out"] / "digest.md").write_text(digest(cases, pts, req.drop(columns="target"), allmet, lreq, lmet, x2),
                                        encoding="utf-8")
    return summ


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "compare", "wsl-endf", "wsl-stubs", "wsl-run"])
    ap.add_argument("--repo", default="/mnt/c/Users/USER/Desktop/shield/shieldmap")
    ap.add_argument("--max-seconds", type=float, default=480.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        export_inputs()
    elif a.cmd == "compare":
        print(json.dumps(compare_all(), indent=1, default=str))
    elif a.cmd == "wsl-endf":
        wsl_endf(Path(a.repo))
    elif a.cmd == "wsl-stubs":
        wsl_stubs(Path(a.repo))
    else:
        wsl_run(Path(a.repo), a.max_seconds, a.threads)


if __name__ == "__main__":
    # as a script (WSL side) the package directory must not shadow third-party modules (e.g. 'tables')
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    main()
