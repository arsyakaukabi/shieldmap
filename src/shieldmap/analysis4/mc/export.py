"""Stage 4a the Windows side: export the Monte Carlo case definitions.

Run from ``shieldmap/``::

    .venv\\Scripts\\python.exe -m shieldmap.analysis4.mc.export

Writes ``data/phase4a/mc/inputs/cases.json`` (consumed by the WSL-side ``openmc_cases.py``) and
``data/phase4a/mc/inputs/reference_values.json`` (Table C / first-order values used by ``compare.py``).

Everything here is read from stage 1-3 outputs (read only):
* compositions: stage 1 mass fractions (``common.mass_fractions``), i.e. the rows the map uses;
* densities: the stage 3 measured-density point ρ_MP / (1 + δ_point) (``density_model._point_delta``:
  measured δ where it exists, cell median otherwise; read from ``data/phase3/T/delta_model_rows.parquet``);
  Pb, Bi, Tl, Ir are stage 1 element densities by this rule. Areal-mass results do not depend on density;
* sources: stage 1 quadrature pieces (``spectrum_bins.parquet``); a line source is its line(s);
* Table C narrow-beam transmission and the stage 3 first-order fluorescence transmission at the exact MC
  areal masses (``analysis3.fluorescence.FluorescenceModel``, primary model E1, K + L ≥ 3.2029 keV).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common
from shieldmap.analysis3 import fluorescence as fl
from shieldmap.physics import response as resp

OUT = paths.DATA / "phase4a" / "mc" / "inputs"
FACTORS = (0.8, 1.0, 1.2)                 # headline thickness and ±20 %
# radionuclides: 1.4 and 1.6 added so the broad-beam T = 0.01 thickness (build-up raises T, so it is THICKER
# than the narrow-beam one) is bracketed; 0.6 was added first by mistake (kept: cheap to run, and it maps B(t))
FACTORS_RADIONUCLIDE = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
TARGET_REL_ERR = 0.001                    # narrow-beam relative standard error aimed at per run
N_MIN, N_MAX = 10_000_000, 200_000_000

# materials (R2 rows the stage 3 outputs name; see the case docstrings in CASES)
MATERIALS = {
    "Pb": "mp-20483", "Bi": "mp-23152", "Ir": "mp-101", "Tl": None, "HfO2": None, "BiOF": None,
    "LaBi": "mp-22926", "PrSbPd": "mp-1018899", "Pr3OsI3": "mp-570779", "HfAu2": None, "La": None,
}
# Stage 3 row indices named in the outputs (used where the material_id was not printed in a report)
ROWS = {"Tl": 66, "HfAu2": 64805}

CASES = [
    # id, title, source, materials, basis, factors, extra
    {"case": 6, "id": "c6_cs137_pb", "title": "Textbook sanity: Pb at Cs-137 (Table C thickness, T = 0.01)",
     "source": ("line", "Cs-137"), "materials": ["Pb"], "basis": "thickness", "T_target": 0.01,
     "factors": FACTORS_RADIONUCLIDE, "priority": 0},
    {"case": 1, "id": "c1_cs137", "title": "Cs-137: Ir vs HfO2 vs BiOF (GAP | B4 transparency thickness price)",
     "source": ("line", "Cs-137"), "materials": ["Ir", "HfO2", "BiOF"], "basis": "thickness", "T_target": 0.01,
     "factors": FACTORS_RADIONUCLIDE, "priority": 1},
    {"case": 3, "id": "c3_rqr5", "title": "RQR5 (70 kV, 12 deg) at the 0.25 mm Pb target: LaBi vs Bi vs Pb",
     "source": ("spectrum", "RQR5_th12"), "pb_mm": 0.25, "variant_id": 3, "materials": ["LaBi", "Bi", "Pb", "PrSbPd"],
     "basis": "areal", "factors": FACTORS, "priority": 2},
    {"case": 2, "id": "c2_co60", "title": "Co-60: Ir vs HfO2 (GAP | B4 transparency thickness price)",
     "source": ("line", "Co-60"), "materials": ["Ir", "HfO2"], "basis": "thickness", "T_target": 0.01,
     "factors": FACTORS_RADIONUCLIDE, "priority": 3},
    {"case": 4, "id": "c4_rqr8", "title": "RQR8 (100 kV, 12 deg) at the 0.35 mm Pb target: Pr3OsI3 vs Bi vs Pb",
     "source": ("spectrum", "RQR8_th12"), "pb_mm": 0.35, "variant_id": 22, "materials": ["Pr3OsI3", "Bi", "Pb"],
     "basis": "areal", "factors": FACTORS, "priority": 4},
    {"case": 5, "id": "c5_89keV", "title": "89.5 keV mass window: Pb vs Tl vs HfAu2 (monoenergetic, T = 0.01)",
     "source": ("mono", 89.5), "materials": ["Pb", "Tl", "HfAu2"], "basis": "areal", "T_target": 0.01,
     "factors": FACTORS, "priority": 5},
]


def _row(key: str) -> int:
    if MATERIALS.get(key):
        return common.row_of(MATERIALS[key])
    if key in ROWS:
        return ROWS[key]
    raise KeyError(key)


def _rows_by_formula() -> dict:
    """Rows for materials named by formula in stage 3 (R2 representative = lowest-e_hull ICSD-matched entry)."""
    u = common.universe()
    out = {}
    base = common.base_mask("R2")
    for key, formula in (("HfO2", "HfO2"), ("BiOF", "BiOF"), ("Tl", "Tl"), ("La", "La")):
        r = np.flatnonzero(base & (u["formula_pretty"].to_numpy() == formula))
        if r.size != 1:
            raise RuntimeError(f"{formula}: {r.size} R2 rows")
        out[key] = int(r[0])
    return out


def _point_density(rows) -> tuple[np.ndarray, np.ndarray]:
    dm = pd.read_parquet(common.P3 / "T" / "delta_model_rows.parquet")
    dm = dm.set_index("row")
    u = common.universe()
    rho_mp = u["density"].to_numpy(float)[rows]
    d = dm.loc[rows]
    delta = np.where(d["source"].to_numpy() != "conditional", d["delta_meas"].to_numpy(float),
                     d["delta_p50"].to_numpy(float))
    return rho_mp / (1.0 + delta), d["source"].to_numpy()


def material_record(key: str, row: int) -> dict:
    u = common.universe()
    W = common.mass_fractions()[row].toarray().ravel()
    zs = np.flatnonzero(W > 0) + 1
    import xraylib
    rho_pt, src = _point_density(np.array([row]))
    return {"key": key, "row": int(row), "material_id": str(u["material_id"].iloc[row]),
            "formula": str(u["formula_pretty"].iloc[row]),
            "theoretical": bool(u["theoretical"].iloc[row]), "e_hull": float(u["energy_above_hull"].iloc[row]),
            "rho_mp": float(u["density"].iloc[row]), "rho_used": float(rho_pt[0]), "rho_source": str(src[0]),
            "mass_fractions": {xraylib.AtomicNumberToSymbol(int(z)): float(W[z - 1]) for z in zs},
            "W": W}


def source_record(src) -> dict:
    kind, name = src
    if kind in ("line", "spectrum"):
        e, phi = common.spectrum(name)
    else:
        e, phi = np.array([float(name)]), np.array([1.0])
    R = resp.response("air_kerma", e)
    keep = phi * R > 0
    e, phi, R = e[keep], phi[keep], R[keep]
    p = phi / phi.sum()
    return {"kind": kind, "name": str(name), "energies_keV": e.tolist(), "fluence": phi.tolist(),
            "prob": p.tolist(), "R_air_kerma": R.tolist(), "K0_per_photon": float((p * R).sum())}


def n_particles(src: dict, a: np.ndarray, q: float) -> int:
    """Particles so that the uncollided air-kerma estimator has ~TARGET_REL_ERR relative error."""
    p, R = np.array(src["prob"]), np.array(src["R_air_kerma"])
    s = p * np.exp(-a * q)
    m1, m2 = (s * R).sum(), (s * R * R).sum()
    n = m2 / m1 ** 2 / TARGET_REL_ERR ** 2
    return int(min(N_MAX, max(N_MIN, math.ceil(n / 1e5) * 1e5)))


# Diagnostic cases (added after case 3 showed MC escaping fluorescence 20-30 % above the first-order model for
# every emitter): monoenergetic 50 keV on La, Bi and LaBi at three areal masses, to separate production/escape
# (thin slab) from thickness-dependent terms. Not part of the case list; reported as diagnostics.
DIAG_CASES = [
    {"case": 101, "id": "d1_50keV_La", "title": "Diagnostic: La metal, 50 keV, three areal masses",
     "source": ("mono", 50.0), "materials": ["La"], "basis": "areal", "T_target": 0.1,
     "q_abs": (0.02, 0.10, 0.30), "priority": 101},
    {"case": 102, "id": "d2_50keV_Bi", "title": "Diagnostic: Bi metal, 50 keV, three areal masses",
     "source": ("mono", 50.0), "materials": ["Bi"], "basis": "areal", "T_target": 0.1,
     "q_abs": (0.02, 0.10, 0.30), "priority": 102},
    {"case": 103, "id": "d3_50keV_LaBi", "title": "Diagnostic: LaBi, 50 keV, three areal masses",
     "source": ("mono", 50.0), "materials": ["LaBi"], "basis": "areal", "T_target": 0.1,
     "q_abs": (0.02, 0.10, 0.30), "priority": 103},
]


def build(case_list=None) -> dict:
    case_list = CASES if case_list is None else case_list
    rows = {k: (_row(k) if (MATERIALS.get(k) or k in ROWS) else None) for k in MATERIALS}
    rows.update(_rows_by_formula())
    mats = {k: material_record(k, r) for k, r in rows.items()}
    F = pd.read_parquet(common.P3 / "F" / "corrected_areal.parquet")
    fsum = json.loads((common.P3 / "F" / "summary.json").read_text(encoding="utf-8"))
    mpi = pd.read_csv(common.P3 / "T" / "measured_point_increments.csv")
    cases, refs = [], {}
    for c in case_list:
        src = source_record(c["source"])
        e = np.array(src["energies_keV"]); phi = np.array(src["fluence"])
        Wm = np.vstack([mats[k]["W"] for k in c["materials"]])
        m_fl = fl.FluorescenceModel(Wm, e, phi)                          # primary first-order model
        m_nb = fl.FluorescenceModel(Wm, e, phi, fluorescence=False)      # Table C narrow beam
        m_K = fl.FluorescenceModel(Wm, e, phi, l_shell=False)            # sensitivity: K only
        m_E2 = fl.FluorescenceModel(Wm, e, phi, escape="E2")             # sensitivity: E2 kernel
        # attribution aid (not a stage 3 model): the primary model with the escaping lines attenuated WITHOUT
        # coherent scattering, i.e. Rayleigh-scattered fluorescence photons are not counted as removed
        m_nc = fl.FluorescenceModel(Wm, e, phi)
        for em in m_nc.emitters:
            em.b = m_nc.W[em.rows] @ fl._xs_at(em.energy)["total_wocoh"]
        if c["basis"] == "areal" and c["source"][0] == "spectrum":
            T_target = float(fsum["pb"]["T_star"][str(c["variant_id"])])
        else:
            T_target = float(c["T_target"])
        q_nb = m_nb.solve_narrow(np.full(len(c["materials"]), T_target))
        q_fl = m_fl.solve_effective(np.full(len(c["materials"]), T_target), lo=q_nb)
        q_nc = m_nc.solve_effective(np.full(len(c["materials"]), T_target), lo=q_nb)
        runs, mref = [], {}
        for j, k in enumerate(c["materials"]):
            mat = mats[k]
            rho = mat["rho_used"]
            # headline areal mass: thickness basis -> Table C narrow-beam requirement (T = 0.01) at the measured-
            # density point; DIAG/mono areal basis -> the first-order corrected requirement (the headline
            # quantity), so ±20 % brackets the narrow-beam one as well
            q0 = float(q_nb[j]) if c["basis"] == "thickness" else float(q_fl[j])
            a_j = m_nb.a[j]
            r = {"key": k, "q_narrow_tablec_g_cm2": float(q_nb[j]), "q_firstorder_g_cm2": float(q_fl[j]),
                 "q_firstorder_nocohlines_g_cm2": float(q_nc[j]),
                 "t_narrow_tablec_cm": float(q_nb[j]) / rho, "q0_g_cm2": q0, "rho_used": rho, "points": []}
            if c["basis"] == "areal" and c["source"][0] == "spectrum":
                f = F[(F.variant_id == c["variant_id"]) & (F.row == mat["row"])]
                if len(f):
                    r["phase3_corrected_areal"] = {kk: float(f.iloc[0][kk]) for kk in
                                                   ("q_tablec", "q_narrow", "q_eff_fixedT", "q_eff_likeforlike")}
            if c["basis"] == "thickness":
                sc = "CS137" if c["source"][1] == "Cs-137" else "CO60"
                hit = mpi[(mpi.scenario == sc) & np.isclose(mpi["T"], 0.01) & (mpi.density == "point")]
                for _, h in hit.iterrows():
                    for side, col in (("base", "t_base_cm"), ("constrained", "t_constrained_cm")):
                        if h[side] == mat["formula"]:
                            r.setdefault("phase3_measured_point_t_cm", float(h[col]))
            if "q_abs" in c:                       # diagnostic cases: absolute areal masses
                q0 = float(c["q_abs"][len(c["q_abs"]) // 2])
                c = dict(c, factors=[round(q / q0, 6) for q in c["q_abs"]])
                r["q0_g_cm2"] = q0
            facs = [float(f) for f in c["factors"]]
            if c["basis"] == "areal" and "q_abs" not in c and float(q_nb[j]) / q0 < 0.85:
                # the fluorescence-corrected requirement is far above the narrow-beam one (e.g. just above a K
                # edge): also bracket the narrow-beam requirement (x 0.8 / 1.0 / 1.2), merging points within 3 %
                for f2 in c["factors"]:
                    x = round(float(q_nb[j]) * f2 / q0, 3)
                    if all(abs(x / y - 1) > 0.03 for y in facs):
                        facs.append(x)
            for fct in sorted(facs):
                q = q0 * fct
                Tf, Tp, Tfl, parts = m_fl.transmission(np.array([q]), idx=np.array([j]), parts=True)
                pt = {"factor": fct, "q_g_cm2": q, "t_cm": q / rho,
                      "T_tablec_narrow": float(m_nb.primary(np.array([q]), idx=np.array([j]))[0]),
                      "T_firstorder": float(Tf[0]), "T_firstorder_fluor": float(Tfl[0]),
                      "T_firstorder_fluor_by_Z": {int(z): float(v[0]) for z, v in parts.items()},
                      "T_firstorder_Konly": float(m_K.transmission(np.array([q]), idx=np.array([j]))[0]),
                      "T_firstorder_E2": float(m_E2.transmission(np.array([q]), idx=np.array([j]))[0]),
                      "T_firstorder_nocohlines": float(m_nc.transmission(np.array([q]), idx=np.array([j]))[0]),
                      "n_particles": n_particles(src, a_j, q)}
                r["points"].append(pt)
                runs.append({"run_id": f"{c['id']}__{k}__f{fct:.3f}", "case": c["case"], "material": k,
                             "factor": fct, "t_cm": q / rho, "n_particles": pt["n_particles"]})
            mref[k] = r
        cases.append({"case": c["case"], "id": c["id"], "title": c["title"], "priority": c["priority"],
                      "basis": c["basis"], "T_target": T_target, "source": src, "materials": c["materials"],
                      "pb_mm": c.get("pb_mm"), "variant_id": c.get("variant_id"), "runs": runs})
        refs[c["id"]] = {"T_target": T_target, "materials": mref}
    mats_out = {k: {kk: vv for kk, vv in v.items() if kk != "W"} for k, v in mats.items()}
    return {"materials": mats_out, "cases": cases}, refs


def air_response_table() -> dict:
    """SRD 126 air table exactly as the map uses it (above the Ar K edge) plus the full table (for lines
    below 3.2029 keV, which the map never needs)."""
    e, mu, muen = resp.air_table()
    from shieldmap import config
    rows = config.get("response.air_muen")
    return {"above_ArK": {"energy_keV": e.tolist(), "muen_rho": muen.tolist()},
            "full": [{k: float(r[k]) for k in ("energy_keV", "mu_rho", "muen_rho")} for r in rows]}


def main(argv=None) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    diag = "--diag" in argv
    OUT.mkdir(parents=True, exist_ok=True)
    cases, refs = build(DIAG_CASES if diag else None)
    cases["air_response"] = air_response_table()
    from shieldmap.analysis4.mc import mcresp
    cases["kerma_tab"] = mcresp.kerma_tab()
    pre = "diag_" if diag else ""
    (OUT / f"{pre}cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    (OUT / f"{pre}reference_values.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
    for c in cases["cases"]:
        print(c["id"], c["T_target"], [(r["material"], round(r["t_cm"], 5), r["n_particles"]) for r in c["runs"]])
    for k, m in cases["materials"].items():
        print(k, m["formula"], m["material_id"], m["row"], round(m["rho_mp"], 4), round(m["rho_used"], 4), m["rho_source"])


if __name__ == "__main__":
    main()
