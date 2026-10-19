"""Stage 4a the Windows side: MC records -> summary tables. ONE COMMAND regenerates data/phase4a/mc/::

    .venv\\Scripts\\python.exe -m shieldmap.analysis4.mc.compare          (from shieldmap/)

Inputs
  data/phase4a/mc/inputs/cases.json, reference_values.json     (export.py)
  data/phase4a/mc/endf_reference.json                            (openmc_cases.py endf: ENDF μ, line energies)
  data/phase4a/mc/runs/<run_id>/bNN.json                         (per sub-run metadata + native tallies)
  data/raw/mc4a/records/<run_id>/bNN.npz                         (back-face crossing records: E, u_z, w)
     -> if the records are absent, the per-sub-run sums in data/phase4a/mc/subrun_sums.csv are used instead
Outputs (data/phase4a/mc/)
  subrun_sums.csv      per sub-run category sums (kerma estimators normalised to the unattenuated beam)
  points.csv           per material x thickness: MC estimators (mean, SE over sub-runs) vs Table C / ENDF /
                       first-order model
  requirements.csv     required areal mass / thickness at the case target under each estimator
  metrics.csv          case metrics: prices / advantages / build-up under each estimator
  summary.json         everything above keyed for the report, plus provenance
  records_manifest.json  SHA-256 of every record file

Estimators (air kerma at the back face of a laterally infinite slab, broad parallel beam at normal
incidence; each crossing photon contributes w · R(E) / |u_z|, R = E (μ_en/ρ)_air from SRD 126 as in the map):
  narrow     uncollided primary only (E equals a source energy and u_z = 1 to 1e-12): the MC counterpart of
             Table C's narrow-beam transmission
  cone1/3    every photon within 1° / 3° of the beam axis (small-acceptance collimated detector)
  fl         narrow + every fluorescence line photon (any generation, incl. secondary fluorescence and
             fluorescence after Compton vacancies; Rayleigh-scattered line photons included) with E ≥ 3.2029
             keV: the MC counterpart of the stage 3 first-order model (narrow primary + escaping K/L lines)
  fl_all     fl + lines between 1 and 3.2029 keV (mostly M lines; omitted by the first-order model)
  broad      every photon (≥ 1 keV): total air kerma incl. Compton/Rayleigh build-up, all fluorescence,
             bremsstrahlung (TTB) and annihilation photons
  broad_hi   broad restricted to E ≥ 3.2029 keV
Build-up B = broad / narrow at the same slab.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis4.mc import mcresp

MC = paths.DATA / "phase4a" / "mc"
REC = paths.DATA / "raw" / "mc4a" / "records"
AR_K = mcresp.AR_K
ANNIHILATION_EV = 510998.95
GRAZING_MU = 0.01
LN = math.log


def _load(pre: str = ""):
    cases = json.loads((MC / "inputs" / f"{pre}cases.json").read_text(encoding="utf-8"))
    refs = json.loads((MC / "inputs" / f"{pre}reference_values.json").read_text(encoding="utf-8"))
    endf = json.loads((MC / "endf_reference.json").read_text(encoding="utf-8"))
    return cases, refs, endf


def _line_table(endf: dict, elements) -> pd.DataFrame:
    rows = []
    for el in elements:
        for ln in endf["lines"].get(el, []):
            sh = ln["shell"]
            rows.append({"element": el, "shell": sh, "family": sh[0], "energy_eV": ln["energy_eV"]})
    return pd.DataFrame(rows)


def _classify(E, uz, src_eV, lines: pd.DataFrame):
    """Category codes: 0 uncollided, 1 primary energy scattered (coherent only), 2 line, 3 annihilation,
    4 other (Compton-scattered continuum, bremsstrahlung). Also the line family (K/L/M/N...) and element."""
    E = np.asarray(E); cat = np.full(E.size, 4, dtype=np.int8)
    srcs = np.sort(np.asarray(src_eV, float))
    i = np.clip(np.searchsorted(srcs, E), 1, srcs.size - 1) if srcs.size > 1 else np.zeros(E.size, int)
    near = np.minimum(np.abs(E - srcs[i - 1 if srcs.size > 1 else 0]), np.abs(E - srcs[i])) if srcs.size > 1 \
        else np.abs(E - srcs[0])
    src_match = near <= 1e-9 * E
    cat[src_match] = 1
    cat[src_match & (uz >= 1.0 - 1e-12)] = 0
    fam = np.full(E.size, "", dtype=object); elem = np.full(E.size, "", dtype=object)
    rest = ~src_match
    if len(lines):
        le = lines["energy_eV"].to_numpy()
        order = np.argsort(le); le_s = le[order]
        j = np.clip(np.searchsorted(le_s, E[rest]), 1, le_s.size - 1)
        dl = np.abs(E[rest] - le_s[j - 1]); dr = np.abs(E[rest] - le_s[j])
        jj = np.where(dl <= dr, j - 1, j); d = np.minimum(dl, dr)
        is_line = d <= 1e-7 * np.maximum(E[rest], 1.0)
        idx = np.flatnonzero(rest)[is_line]
        cat[idx] = 2
        src_rows = order[jj[is_line]]
        fam[idx] = lines["family"].to_numpy()[src_rows]
        elem[idx] = lines["element"].to_numpy()[src_rows]
    ann = (cat == 4) & (np.abs(E - ANNIHILATION_EV) < 1e-3)
    cat[ann] = 3
    return cat, fam, elem


def subrun_sums(cases, endf) -> pd.DataFrame:
    recs = []
    mats = cases["materials"]
    for c in cases["cases"]:
        src_eV = np.array(c["source"]["energies_keV"]) * 1e3
        K0 = float((np.array(c["source"]["prob"]) * mcresp.air_kerma(np.array(c["source"]["energies_keV"]))).sum())
        for r in c["runs"]:
            rid = r["run_id"]
            lines = _line_table(endf, mats[r["material"]]["mass_fractions"].keys())
            for meta_p in sorted((MC / "runs" / rid).glob("b*.json")):
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                npz = REC / rid / meta_p.name.replace(".json", ".npz")
                if not npz.exists():
                    continue
                z = np.load(npz)
                E, uz, w = z["E"], z["uz"], z["wgt"]
                ek = E / 1e3
                R = mcresp.air_kerma(ek)
                inv = 1.0 / np.abs(uz)
                invg = np.where(np.abs(uz) < GRAZING_MU, 2.0 / GRAZING_MU, inv)
                f = w * R * inv / (meta["n_particles"] * K0)
                fg = w * R * invg / (meta["n_particles"] * K0)
                cat, fam, elem = _classify(E, uz, src_eV, lines)
                hi = ek >= AR_K
                d = {"run_id": rid, "case": c["case"], "material": r["material"], "factor": r["factor"],
                     "t_cm": r["t_cm"], "batch": meta["batch"], "seed": meta["seed"], "n_particles": meta["n_particles"],
                     "n_records": int(E.size), "seconds": meta["seconds"], "truncated": meta["truncated"],
                     "narrow": f[cat == 0].sum(), "coh_primary": f[cat == 1].sum(),
                     "lines_hi": f[(cat == 2) & hi].sum(), "lines_lo": f[(cat == 2) & ~hi].sum(),
                     "annihilation": f[cat == 3].sum(), "other_hi": f[(cat == 4) & hi].sum(),
                     "other_lo": f[(cat == 4) & ~hi].sum(), "broad": f.sum(), "broad_hi": f[hi].sum(),
                     "broad_grazing_cut": fg.sum(),
                     "cone1": f[uz >= math.cos(math.radians(1.0))].sum(),
                     "cone3": f[uz >= math.cos(math.radians(3.0))].sum(),
                     "n_uncollided": int((cat == 0).sum()), "n_line": int((cat == 2).sum()),
                     # native track-length tally in the 1 cm back void layer: T = flux·R / (d · K0), d = 1 cm
                     "native_kerma_back_layer": meta["native"]["kerma_back_layer"] / K0
                     if "native" in meta else np.nan}
                for fm in ("K", "L", "M", "N"):
                    d[f"lines_{fm}_hi"] = f[(cat == 2) & hi & (fam == fm)].sum()
                    d[f"lines_{fm}_lo"] = f[(cat == 2) & ~hi & (fam == fm)].sum()
                for el in mats[r["material"]]["mass_fractions"]:
                    d[f"lines_hi_{el}"] = f[(cat == 2) & hi & (elem == el)].sum()
                    d[f"lines_all_{el}"] = f[(cat == 2) & (elem == el)].sum()
                recs.append(d)
    return pd.DataFrame(recs)


EST = ("narrow", "cone1", "cone3", "fl", "fl_all", "broad", "broad_hi", "broad_grazing_cut")


def _estimators(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["fl"] = s["narrow"] + s["lines_hi"]
    s["fl_all"] = s["fl"] + s["lines_lo"]
    return s


def _endf_narrow(endf, mat, src, t_cm) -> float:
    m = endf["materials"][mat]
    E = np.array(m["energies_keV"]); mu = np.array(m["mu_cm"])
    e = np.array(src["energies_keV"]); p = np.array(src["prob"])
    mu_s = np.interp(e, E, mu)                      # E contains every source energy exactly
    R = mcresp.air_kerma(e)
    return float((p * R * np.exp(-mu_s * t_cm)).sum() / (p * R).sum())


def points_table(cases, refs, endf, s: pd.DataFrame) -> pd.DataFrame:
    s = _estimators(s)
    out = []
    for c in cases["cases"]:
        rf = refs[c["id"]]
        for r in c["runs"]:
            g = s[s.run_id == r["run_id"]]
            if g.empty:
                continue
            mref = rf["materials"][r["material"]]
            pt = next(p for p in mref["points"] if abs(p["factor"] - r["factor"]) < 1e-9)
            d = {"case": c["case"], "case_id": c["id"], "material": r["material"], "factor": r["factor"],
                 "t_cm": r["t_cm"], "q_g_cm2": pt["q_g_cm2"], "n_subruns": len(g),
                 "n_particles": int(g.n_particles.sum()), "cpu_seconds": float(g.seconds.sum()),
                 "any_truncated": bool(g.truncated.any()),
                 "T_tablec": pt["T_tablec_narrow"], "T_endf_narrow": _endf_narrow(endf, r["material"], c["source"], r["t_cm"]),
                 "T_firstorder": pt["T_firstorder"], "T_firstorder_fluor": pt["T_firstorder_fluor"],
                 "T_firstorder_Konly": pt["T_firstorder_Konly"], "T_firstorder_E2": pt["T_firstorder_E2"],
                 "T_firstorder_nocohlines": pt.get("T_firstorder_nocohlines", np.nan)}
            for k in EST + ("lines_hi", "lines_lo", "coh_primary", "annihilation", "other_hi", "other_lo",
                            "lines_K_hi", "lines_L_hi", "lines_M_hi", "lines_M_lo", "lines_L_lo"):
                v = g[k].to_numpy(float)
                d[k] = v.mean()
                d[k + "_se"] = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else np.nan
            for col in g.columns:
                if col.startswith("lines_hi_") or col.startswith("lines_all_"):
                    d[col] = g[col].mean()
            d["native_over_records"] = float(g["native_kerma_back_layer"].mean() / g["broad"].mean()) \
                if g["broad"].mean() > 0 else np.nan
            d["z_narrow_vs_endf"] = (d["narrow"] - d["T_endf_narrow"]) / d["narrow_se"]
            d["z_narrow_vs_tablec"] = (d["narrow"] - d["T_tablec"]) / d["narrow_se"]
            d["ratio_narrow_mc_tablec"] = d["narrow"] / d["T_tablec"]
            d["ratio_endf_tablec"] = d["T_endf_narrow"] / d["T_tablec"]
            d["buildup"] = d["broad"] / d["narrow"]
            d["buildup_se"] = d["buildup"] * math.hypot(d["broad_se"] / d["broad"], d["narrow_se"] / d["narrow"])
            d["ratio_fl_mc_firstorder"] = d["fl"] / d["T_firstorder"]
            d["ratio_fluor_mc_firstorder"] = d["lines_hi"] / d["T_firstorder_fluor"] if d["T_firstorder_fluor"] > 0 else np.nan
            out.append(d)
    return pd.DataFrame(out)


def _solve(q, T, Tt):
    """q at which log T = log Tt: exact interpolation (polynomial of degree ≤ 2 in q through the nearest
    three points); outside the bracket the end segment is extrapolated linearly (flagged)."""
    q = np.asarray(q, float); y = np.log(np.asarray(T, float)); yt = LN(Tt)
    o = np.argsort(q); q, y = q[o], y[o]
    inside = y.min() <= yt <= y.max()
    if q.size >= 3:
        k = np.argsort(np.abs(y - yt))[:3]; k.sort()
        c = np.polyfit(q[k], y[k], 2)
        roots = np.roots([c[0], c[1], c[2] - yt])
        roots = roots[np.isreal(roots)].real
        lo, hi = q[k].min() - 0.5 * np.ptp(q), q[k].max() + 0.5 * np.ptp(q)
        roots = roots[(roots >= lo) & (roots <= hi)]
        if roots.size:
            return float(roots[np.argmin(np.abs(roots - np.interp(yt, y[::-1], q[::-1])))]), inside
    c = np.polyfit(q[:2] if yt > y[0] else q[-2:], y[:2] if yt > y[0] else y[-2:], 1)
    return float((yt - c[1]) / c[0]), inside


def requirements(cases, refs, pts: pd.DataFrame, n_boot: int = 2000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for c in cases["cases"]:
        Tt = refs[c["id"]]["T_target"]
        for mat in c["materials"]:
            g = pts[(pts.case_id == c["id"]) & (pts.material == mat)].sort_values("q_g_cm2")
            if len(g) < 2:
                continue
            rho = refs[c["id"]]["materials"][mat]["rho_used"]
            for est in ("T_tablec", "T_endf_narrow", "T_firstorder", "T_firstorder_nocohlines") + EST:
                q = g.q_g_cm2.to_numpy(); T = g[est].to_numpy(float)
                if not np.all(T > 0):
                    continue
                q0, inside = _solve(q, T, Tt)
                se = np.nan
                if est + "_se" in g:
                    s_ = g[est + "_se"].to_numpy(float)
                    bs = [_solve(q, np.maximum(T + rng.standard_normal(T.size) * s_, 1e-300), Tt)[0]
                          for _ in range(n_boot)]
                    se = float(np.std(bs, ddof=1))
                out.append({"case": c["case"], "case_id": c["id"], "material": mat, "estimator": est,
                            "T_target": Tt, "q_req_g_cm2": q0, "q_req_se": se, "t_req_cm": q0 / rho,
                            "t_req_se": se / rho, "inside_bracket": inside, "n_points": len(g)})
    return pd.DataFrame(out)


def metrics(cases, refs, req: pd.DataFrame) -> pd.DataFrame:
    """Case metrics as ratios of requirements, X/Y − 1, under every estimator (SE by first-order propagation
    of independent sub-run errors)."""
    pairs = {
        "c6_cs137_pb": [],
        "c1_cs137": [("BiOF", "Ir", "GAP(3.1) given B4 thickness price (Cs-137)"), ("HfO2", "Ir", "HfO2 vs Ir thickness"),
                     ("HfO2", "BiOF", "HfO2 vs BiOF thickness")],
        "c2_co60": [("HfO2", "Ir", "GAP(3.1) given B4 thickness price (Co-60)")],
        "c3_rqr5": [("Bi", "LaBi", "advantage Bi/LaBi − 1 (R3.1 knife edge)"), ("Pb", "LaBi", "Pb/LaBi − 1"),
                    ("Bi", "PrSbPd", "Bi/PrSbPd − 1"), ("PrSbPd", "LaBi", "PrSbPd/LaBi − 1"), ("Pb", "Bi", "Pb/Bi − 1")],
        "c4_rqr8": [("Bi", "Pr3OsI3", "advantage Bi/Pr3OsI3 − 1"), ("Pb", "Pr3OsI3", "Pb/Pr3OsI3 − 1"),
                    ("Pb", "Bi", "Pb/Bi − 1")],
        "c5_89keV": [("Tl", "Pb", "Tl/Pb − 1 (narrow-beam PB mass price)"), ("HfAu2", "Pb", "HfAu2/Pb − 1"),
                     ("HfAu2", "Tl", "HfAu2/Tl − 1")],
    }
    out = []
    if req.empty:
        return pd.DataFrame(out)
    for c in cases["cases"]:
        basis_col ="t_req_cm" if c["basis"] == "thickness" else "q_req_g_cm2"
        se_col = "t_req_se" if c["basis"] == "thickness" else "q_req_se"
        for a, b, label in pairs.get(c["id"], []):
            for est in req.estimator.unique():
                ra = req[(req.case_id == c["id"]) & (req.material == a) & (req.estimator == est)]
                rb = req[(req.case_id == c["id"]) & (req.material == b) & (req.estimator == est)]
                if ra.empty or rb.empty:
                    continue
                x, y = float(ra[basis_col].iloc[0]), float(rb[basis_col].iloc[0])
                sx, sy = float(ra[se_col].iloc[0]), float(rb[se_col].iloc[0])
                v = x / y - 1.0
                se = (x / y) * math.hypot(sx / x, sy / y) if np.isfinite(sx) and np.isfinite(sy) else np.nan
                out.append({"case": c["case"], "case_id": c["id"], "metric": label, "num": a, "den": b,
                            "estimator": est, "basis": c["basis"], "value": v, "se": se,
                            "inside_bracket": bool(ra.inside_bracket.iloc[0] and rb.inside_bracket.iloc[0])})
    return pd.DataFrame(out)


def records_manifest() -> dict:
    out = {}
    for p in sorted(REC.rglob("*.npz")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[str(p.relative_to(paths.ROOT)).replace("\\", "/")] = {"sha256": h, "bytes": p.stat().st_size}
    return out


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}g}"


def digest(cases, pts: pd.DataFrame, req: pd.DataFrame, met: pd.DataFrame) -> str:
    """Markdown digest of every table (data/phase4a/mc/digest.md); the report quotes from it."""
    L = ["# MC spot check: digest (generated by compare.py; do not edit)", ""]
    for c in sorted(cases["cases"], key=lambda c: c["priority"]):
        p = pts[pts.case_id == c["id"]] if len(pts) else pts
        if p is None or len(p) == 0:
            L += [f"## Case {c['case']}: {c['title']}", "", "not run", ""]
            continue
        L += [f"## Case {c['case']}: {c['title']}", "", f"Target T = {c['T_target']:.6g}; basis {c['basis']}.", "",
              "| material | factor | t (cm) | q (g/cm²) | T Table C | T ENDF analytic | MC narrow ± SE | z vs ENDF | MC/Table C "
              "| cone 1° | B = broad/narrow | T first-order | MC fl ± SE | MC fl / first-order | MC fl_all | MC broad ± SE | particles |",
              "|" + "---|" * 17]
        for _, r in p.sort_values(["material", "factor"]).iterrows():
            L.append(f"| {r.material} | {r.factor:.3f} | {r.t_cm:.5g} | {r.q_g_cm2:.5g} | {_f(r.T_tablec, 5)} | "
                     f"{_f(r.T_endf_narrow, 5)} | {_f(r.narrow, 5)} ± {_f(r.narrow_se, 2)} | {r.z_narrow_vs_endf:+.2f} | "
                     f"{r.ratio_narrow_mc_tablec:.4f} | {_f(r.cone1, 5)} | {r.buildup:.4f} ± {r.buildup_se:.4f} | "
                     f"{_f(r.T_firstorder, 5)} | {_f(r.fl, 5)} ± {_f(r.fl_se, 2)} | {r.ratio_fl_mc_firstorder:.4f} | "
                     f"{_f(r.fl_all, 5)} | {_f(r.broad, 5)} ± {_f(r.broad_se, 2)} | {r.n_particles:.3g} |")
        L.append("")
        lcols = [k for k in ("lines_K_hi", "lines_L_hi", "lines_M_hi", "lines_L_lo", "lines_M_lo", "coh_primary",
                             "other_hi", "other_lo", "annihilation") if k in p]
        L += ["Kerma components (normalised like T): " + ", ".join(lcols), "",
              "| material | factor | " + " | ".join(lcols) + " | first-order fluorescence |",
              "|" + "---|" * (len(lcols) + 3)]
        for _, r in p.sort_values(["material", "factor"]).iterrows():
            L.append(f"| {r.material} | {r.factor:.3f} | " + " | ".join(_f(r[k], 4) for k in lcols)
                     + f" | {_f(r.T_firstorder_fluor, 4)} |")
        L.append("")
        q = req[req.case_id == c["id"]] if len(req) else req
        if len(q):
            col = "t_req_cm" if c["basis"] == "thickness" else "q_req_g_cm2"
            secol = "t_req_se" if c["basis"] == "thickness" else "q_req_se"
            unit = "cm" if c["basis"] == "thickness" else "g/cm²"
            ests = ["T_tablec", "T_endf_narrow", "T_firstorder", "T_firstorder_nocohlines", "narrow", "cone1", "fl",
                    "fl_all", "broad"]
            L += [f"Required {('thickness' if c['basis'] == 'thickness' else 'areal mass')} ({unit}) at T = {c['T_target']:.6g}"
                  " (* = extrapolated outside the simulated bracket):", "",
                  "| material | " + " | ".join(ests) + " |", "|" + "---|" * (len(ests) + 1)]
            for mat in c["materials"]:
                cells = []
                for e in ests:
                    x = q[(q.material == mat) & (q.estimator == e)]
                    if x.empty:
                        cells.append("–"); continue
                    x = x.iloc[0]
                    s = f"{x[col]:.5g}" + (f" ± {x[secol]:.2g}" if np.isfinite(x[secol]) else "")
                    cells.append(s + ("" if x.inside_bracket else "*"))
                L.append(f"| {mat} | " + " | ".join(cells) + " |")
            L.append("")
        m = met[met.case_id == c["id"]] if len(met) else met
        if len(m):
            ests = ["T_tablec", "T_endf_narrow", "T_firstorder", "T_firstorder_nocohlines", "narrow", "fl", "fl_all",
                    "broad"]
            L += ["Metrics (X/Y − 1, %):", "", "| metric | " + " | ".join(ests) + " |", "|" + "---|" * (len(ests) + 1)]
            for lab in m.metric.unique():
                cells = []
                for e in ests:
                    x = m[(m.metric == lab) & (m.estimator == e)]
                    if x.empty:
                        cells.append("–"); continue
                    x = x.iloc[0]
                    s = f"{100 * x.value:.2f}" + (f" ± {100 * x.se:.2f}" if np.isfinite(x.se) else "")
                    cells.append(s + ("" if x.inside_bracket else "*"))
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
            L.append("")
    return "\n".join(L)


def main(argv=None) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    pre = "diag_" if "--diag" in argv else ""      # diagnostics: diag_cases.json -> diag_* outputs
    cases, refs, endf = _load(pre)
    if REC.exists() and any(REC.rglob("*.npz")):
        s = subrun_sums(cases, endf)
        s.to_csv(MC / f"{pre}subrun_sums.csv", index=False)
        if not pre:
            (MC / "records_manifest.json").write_text(json.dumps(records_manifest(), indent=1), encoding="utf-8")
    else:
        s = pd.read_csv(MC / f"{pre}subrun_sums.csv")
    pts = points_table(cases, refs, endf, s)
    pts.to_csv(MC / f"{pre}points.csv", index=False)
    req = requirements(cases, refs, pts)
    req.to_csv(MC / f"{pre}requirements.csv", index=False)
    met = metrics(cases, refs, req)
    met.to_csv(MC / f"{pre}metrics.csv", index=False)
    summ = {"n_subruns": int(len(s)), "n_points": int(len(pts)),
            "cases_with_points": sorted(set(int(x) for x in pts.case)) if len(pts) else [],
            "total_particles": int(s.n_particles.sum()) if len(s) else 0,
            "total_seconds": float(s.seconds.sum()) if len(s) else 0.0}
    (MC / f"{pre}summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    (MC / f"{pre}digest.md").write_text(digest(cases, pts, req, met), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if len(pts):
        print(pts[["case_id", "material", "factor", "T_tablec", "T_endf_narrow", "narrow", "narrow_se",
                   "z_narrow_vs_endf", "z_narrow_vs_tablec", "buildup", "T_firstorder", "fl", "fl_all"]].to_string())
    if len(met):
        print(met.to_string())


def run() -> None:
    """Entry point for `python -m shieldmap.analyze4a` (step mc): main cases, then the 50 keV diagnostics."""
    main([])
    if any(MC.rglob("diag_cases.json")):
        main(["--diag"])


if __name__ == "__main__":
    main()
