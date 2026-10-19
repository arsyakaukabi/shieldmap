"""Optional Monte Carlo case c8_cs137_irw, the transport check of the
one quoted non-zero radionuclide price that had none.

Price checked: scarcity proxy (crustal abundance ≥ 1 ppm) given B4 at Cs-137, thickness basis, T = 0.01: W replaces
Ir at the thickness frontier, +23.05 % (H11); with it the Pb-relative thickness of W, 0.664 (H67).

Source: Cs-137, the 661.657 keV line only (as the map). Materials: Ir (mp-101) and W (mp-91), elemental, at the
measured-density point densities the map uses (stage 1 element densities, Ir 22.65 and W 19.25 g/cm³), plus Pb
(mp-20483, 11.34 g/cm³) for the H67 ratio. Ir and Pb are NOT re-simulated: stage 4a cases c1_cs137 (Ir) and
c6_cs137_pb (Pb) ran them with the identical definition (same export.build path, densities, slab factors 0.6-1.6,
particles per slab, model builder, estimators, seeds by run id), which `export` verifies field by field before it
reuses their run ids; only the six W slabs are new.

Everything reuses the stage 4a/4b machinery (read only; nothing in src/shieldmap/analysis4*, data/phase4a or
data/phase4b is written):
  * case building     analysis4.mc.export.build, called with W (mp-91) added to its MATERIALS table for the call
                      (the table is restored afterwards); thickness basis, FACTORS_RADIONUCLIDE, n_particles;
  * transport         analysis4b.mc_rev.wsl_endf / wsl_stubs / wsl_run (hence analysis4.mc.openmc_cases.build_model
                      and run_point unchanged) with mc_rev's folders and library view pointed at data/phase4d/mc_c8,
                      data/raw/mc4d and ~/mc4d/xs for the call. The W neutron files are not in the kept library
                      (only the photon file W.h5 is), so W180/182/183/184/186 get mc_rev's placeholders
                      (structure of I135.h5, own Z, A, AWR; never sampled in a photon-only run);
  * reduction         analysis4.mc.compare (subrun_sums, points_table, requirements, digest) with its module paths
                      pointed at the c8 folders for W and at the stage 4a folders for the reused Ir and Pb records;
                      metrics as compare.metrics (thickness basis, first-order SE propagation; pairs differ);
                      χ² as analysis4b.mc_rev.chi2.

Run (from shieldmap/):
  Windows   .venv\\Scripts\\python.exe -m shieldmap.analysis4d.mc_c8 export
  WSL       (wrapper: src/shieldmap/analysis4d/mc_c8_wsl.sh endf | stubs | run [deadline ISO])
            micromamba run -n openmc python src/shieldmap/analysis4d/mc_c8.py wsl-endf
            micromamba run -n openmc python src/shieldmap/analysis4d/mc_c8.py wsl-stubs
            micromamba run -n openmc python src/shieldmap/analysis4d/mc_c8.py wsl-run --deadline 2026-10-16T23:49:00+00:00
  Windows   .venv\\Scripts\\python.exe -m shieldmap.analysis4d.mc_c8 compare
            .venv\\Scripts\\python.exe -m shieldmap.analysis4d.mc_c8 rows

Outputs: data/phase4d/mc_c8/ (inputs/, runs/ sub-run metadata, endf_reference.json, neutron_placeholders.json,
subrun_sums.csv, points.csv, requirements.csv, metrics.csv, chi2.json, summary.json, digest.md,
records_manifest.json); data/phase4d/rows_mc_c8.csv (rows H430+); raw back-face records, statepoints and model XML
under data/raw/mc4d/ (git-ignored).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

CASE = {"case": 8, "id": "c8_cs137_irw", "priority": 8, "basis": "thickness", "T_target": 0.01,
        "title": "Cs-137 (661.657 keV): W vs Ir (ABUND(1)|B4 thickness price, H11) and W vs Pb (H67)",
        "source": ("line", "Cs-137"), "materials": ["Ir", "W", "Pb"]}
W_MPID = "mp-91"
NEW = ("W",)
REUSE = {"Ir": "c1_cs137", "Pb": "c6_cs137_pb"}          # material -> stage 4a case that simulated it
PAIRS = [("W", "Ir", "price W/Ir − 1 (ABUND(1)|B4, thickness; H11)"),
         ("W", "Pb", "W/Pb − 1 (thickness; H67 ratio = value + 1)"),
         ("Ir", "Pb", "Ir/Pb − 1 (thickness)")]
PRICE_PP_TARGET = 2.0                                     # 1.96 SE of each price ≤ 2 percentage points
ROW0 = 430

HERE = Path(__file__).resolve().parent
XS4D = Path.home() / "mc4d" / "xs"                        # WSL library view (photon + neutron links + W placeholders)


def _repo() -> Path:
    return HERE.parents[2]


def _dirs(repo: Path | None = None) -> dict:
    repo = repo or _repo()
    out = repo / "data" / "phase4d" / "mc_c8"
    raw = repo / "data" / "raw" / "mc4d"
    return {"out": out, "inputs": out / "inputs", "runs": out / "runs", "raw": raw, "rec": raw / "records",
            "raw_runs": raw / "runs"}


def _p4a(repo: Path | None = None) -> dict:
    repo = repo or _repo()
    return {"mc": repo / "data" / "phase4a" / "mc", "rec": repo / "data" / "raw" / "mc4a" / "records"}


def _rev(repo: Path | None = None):
    """analysis4b.mc_rev: the package module on Windows; loaded from its file on the WSL side (no shieldmap there)."""
    try:
        from shieldmap.analysis4b import mc_rev
        return mc_rev
    except ImportError:
        p = (repo or _repo()) / "src" / "shieldmap" / "analysis4b" / "mc_rev.py"
        spec = importlib.util.spec_from_file_location("mc_rev", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


@contextlib.contextmanager
def _as_c8(rev):
    """Point mc_rev's folders and library view at the c8 ones for the duration of a call."""
    old = rev._dirs, rev.XS4B
    rev._dirs, rev.XS4B = _dirs, XS4D
    try:
        yield rev
    finally:
        rev._dirs, rev.XS4B = old


# ============================================================================================ Windows: export
@contextlib.contextmanager
def _export_with_w():
    from shieldmap.analysis4.mc import export
    old = dict(export.MATERIALS)
    export.MATERIALS["W"] = W_MPID
    try:
        yield export
    finally:
        export.MATERIALS.clear()
        export.MATERIALS.update(old)


def _same(a, b, rel=1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)
    return a == b


def _map_values() -> dict:
    import pandas as pd
    from shieldmap import paths
    h = pd.read_csv(paths.DATA / "phase4b" / "headline_table_v3.csv").set_index("id")
    return {k: {"value": float(h.at[k, "value"]), "statement": str(h.at[k, "statement"]),
                "delta_abs": float(h.at[k, "delta_abs"])} for k in ("H11", "H67")}


def build() -> tuple[dict, dict, dict]:
    """export.build for the c8 case; Ir and Pb slabs verified identical to stage 4a c1/c6 and their run ids reused."""
    from shieldmap import paths
    with _export_with_w() as export:
        c = dict(CASE, factors=export.FACTORS_RADIONUCLIDE)
        cases, refs = export.build([c])
    p4a_cases = json.loads((_p4a()["mc"] / "inputs" / "cases.json").read_text(encoding="utf-8"))
    p4a_refs = json.loads((_p4a()["mc"] / "inputs" / "reference_values.json").read_text(encoding="utf-8"))
    case = cases["cases"][0]
    mats = {k: cases["materials"][k] for k in case["materials"]}
    new_runs, reused, checks = [], [], {}
    for r in case["runs"]:
        m = r["material"]
        if m not in REUSE:
            new_runs.append(r)
            continue
        old_c = next(x for x in p4a_cases["cases"] if x["id"] == REUSE[m])
        old_r = [x for x in old_c["runs"] if x["material"] == m and abs(x["factor"] - r["factor"]) < 1e-12]
        if len(old_r) != 1:
            raise RuntimeError(f"{m} f{r['factor']}: no unique Phase 4a run in {REUSE[m]}")
        old_r = old_r[0]
        same = {"material": _same(mats[m], p4a_cases["materials"][m]),
                "t_cm": _same(r["t_cm"], old_r["t_cm"], 1e-12), "n_particles": r["n_particles"] == old_r["n_particles"],
                "source": _same(case["source"], old_c["source"], 1e-12),
                "T_target": case["T_target"] == old_c["T_target"]}
        ref_new = next(p for p in refs[case["id"]]["materials"][m]["points"] if abs(p["factor"] - r["factor"]) < 1e-12)
        ref_old = next(p for p in p4a_refs[REUSE[m]]["materials"][m]["points"]
                       if abs(p["factor"] - r["factor"]) < 1e-12)
        same["reference_point"] = _same(json.loads(json.dumps(ref_new)), ref_old, 1e-9)   # int Z keys -> str, as stored
        if not all(same.values()):
            raise RuntimeError(f"{m} f{r['factor']}: definition differs from Phase 4a {REUSE[m]}: {same}")
        checks[old_r["run_id"]] = same
        reused.append(dict(r, run_id=old_r["run_id"], reused_from=f"phase4a/{REUSE[m]}"))
    case["runs"] = new_runs                      # transported in c8 (W only)
    case["reused_runs"] = reused                 # Stage 4a sub-runs, identical definition
    cases = {"materials": mats, "cases": [case]}
    mv = _map_values()
    rf = refs[case["id"]]["materials"]
    t = {k: rf[k]["t_narrow_tablec_cm"] for k in rf}
    mv["tablec_at_mc_densities"] = {"price_W_Ir": t["W"] / t["Ir"] - 1, "ratio_W_Pb": t["W"] / t["Pb"],
                                    "t_cm": t, "delta_W_Ir_cm": t["W"] - t["Ir"]}
    mv["reuse_checks"] = checks
    mv["source_file"] = str((paths.DATA / "phase4b" / "headline_table_v3.csv").relative_to(paths.ROOT)).replace("\\", "/")
    return cases, refs, mv


def export_inputs() -> None:
    from shieldmap.analysis4.mc import export, mcresp
    d = _dirs()
    d["inputs"].mkdir(parents=True, exist_ok=True)
    cases, refs, mv = build()
    cases["air_response"] = export.air_response_table()
    cases["kerma_tab"] = mcresp.kerma_tab()
    p4a = json.loads((_p4a()["mc"] / "inputs" / "cases.json").read_text(encoding="utf-8"))
    if not (_same(cases["kerma_tab"], p4a["kerma_tab"]) and _same(cases["air_response"], p4a["air_response"])):
        raise RuntimeError("kerma table differs from Phase 4a")
    (d["inputs"] / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    (d["inputs"] / "reference_values.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
    (d["inputs"] / "map_values.json").write_text(json.dumps(mv, indent=1), encoding="utf-8")
    c = cases["cases"][0]
    for r in c["runs"] + c["reused_runs"]:
        print(f"  {r['run_id']:32s} t={r['t_cm']:.5f} cm  n={r['n_particles']:.3g}  {r.get('reused_from', 'NEW')}")
    print("new particles", sum(r["n_particles"] for r in c["runs"]))
    print("Table C at the MC densities:", json.dumps(mv["tablec_at_mc_densities"]), "map H11", mv["H11"]["value"],
          "H67", mv["H67"]["value"])


# ============================================================================================ WSL: transport
def wsl_endf(repo: Path) -> None:
    with _as_c8(_rev(repo)) as rev:
        rev.wsl_endf(repo)


def wsl_stubs(repo: Path) -> None:
    with _as_c8(_rev(repo)) as rev:
        rev.wsl_stubs(repo)


def wsl_run(repo: Path, deadline: str, threads: int) -> None:
    left = dt.datetime.fromisoformat(deadline).timestamp() - time.time()
    if left <= 0:
        print("DEADLINE already passed", flush=True)
        return
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} start, {left:.0f} s to the deadline {deadline}", flush=True)
    with _as_c8(_rev(repo)) as rev:
        rev.wsl_run(repo, left, threads)


# ============================================================================================ Windows: reduction
@contextlib.contextmanager
def _compare_paths(mc: Path, rec: Path):
    from shieldmap.analysis4.mc import compare
    old = compare.MC, compare.REC
    compare.MC, compare.REC = mc, rec
    try:
        yield compare
    finally:
        compare.MC, compare.REC = old


def _load():
    d = _dirs()
    cases = json.loads((d["inputs"] / "cases.json").read_text(encoding="utf-8"))
    refs = json.loads((d["inputs"] / "reference_values.json").read_text(encoding="utf-8"))
    endf = json.loads((d["out"] / "endf_reference.json").read_text(encoding="utf-8"))
    mv = json.loads((d["inputs"] / "map_values.json").read_text(encoding="utf-8"))
    return cases, refs, endf, mv


def _views(cases):
    c = cases["cases"][0]
    full = dict(cases, cases=[dict(c, runs=c["runs"] + c["reused_runs"])])
    new = dict(cases, cases=[dict(c, runs=c["runs"])])
    old = dict(cases, cases=[dict(c, runs=c["reused_runs"])])
    return full, new, old


def _has_records(rec: Path, runs) -> bool:
    return all((rec / r["run_id"]).exists() and any((rec / r["run_id"]).glob("*.npz")) for r in runs)


def subrun_sums(cases, endf):
    """Per sub-run sums: W from the c8 records, Ir and Pb from the stage 4a records (recomputed with the same
    compare.subrun_sums; checked against data/phase4a/mc/subrun_sums.csv)."""
    import pandas as pd
    d, a = _dirs(), _p4a()
    full, new, old = _views(cases)
    with _compare_paths(d["out"], d["rec"]) as cmp:
        s_new = cmp.subrun_sums(new, endf)
    s_new["origin"] = "phase4d/c8 (new)"
    p4a = pd.read_csv(a["mc"] / "subrun_sums.csv")
    ids = [r["run_id"] for r in cases["cases"][0]["reused_runs"]]
    p4a = p4a[p4a.run_id.isin(ids)]
    if _has_records(a["rec"], cases["cases"][0]["reused_runs"]):
        with _compare_paths(a["mc"], a["rec"]) as cmp:
            s_old = cmp.subrun_sums(old, endf)
        m = s_old.merge(p4a, on=["run_id", "batch"], suffixes=("", "_p4a"))
        chk = {k: float(np.max(np.abs(m[k] / m[k + "_p4a"] - 1))) for k in ("narrow", "broad", "lines_hi")}
        chk["n_matched"] = int(len(m)); chk["n_phase4a"] = int(len(p4a))
        src = "recomputed from data/raw/mc4a/records"
    else:
        s_old = p4a.copy()
        s_old["case"] = cases["cases"][0]["case"]
        chk = {"n_phase4a": int(len(p4a))}
        src = "data/phase4a/mc/subrun_sums.csv (records absent)"
    s_old["origin"] = [f"phase4a/{REUSE[m]} (reused)" for m in s_old.material]
    return pd.concat([s_new, s_old], ignore_index=True), {"reused_source": src, "check_vs_phase4a_csv": chk}


def metrics(cases, req):
    """compare.metrics for the c8 pairs (X/Y − 1 of thickness requirements at T = 0.01; SE by first-order
    propagation of the independent bootstrap SEs, as compare.metrics)."""
    import pandas as pd
    c = cases["cases"][0]
    out = []
    for a, b, label in PAIRS:
        for est in req.estimator.unique():
            ra = req[(req.material == a) & (req.estimator == est)]
            rb = req[(req.material == b) & (req.estimator == est)]
            if ra.empty or rb.empty:
                continue
            x, y = float(ra.t_req_cm.iloc[0]), float(rb.t_req_cm.iloc[0])
            sx, sy = float(ra.t_req_se.iloc[0]), float(rb.t_req_se.iloc[0])
            v = x / y - 1.0
            se = (x / y) * math.hypot(sx / x, sy / y) if np.isfinite(sx) and np.isfinite(sy) else np.nan
            out.append({"case": c["case"], "case_id": c["id"], "metric": label, "num": a, "den": b, "estimator": est,
                        "basis": "thickness", "value": v, "se": se,
                        "inside_bracket": bool(ra.inside_bracket.iloc[0] and rb.inside_bracket.iloc[0]),
                        "delta_cm": x - y})
    return pd.DataFrame(out)


def records_manifest() -> dict:
    with _as_c8(_rev()) as rev:
        return rev.records_manifest()


def _get(met, num, den, est):
    x = met[(met.num == num) & (met.den == den) & (met.estimator == est)].iloc[0]
    return float(x.value), float(x.se), bool(x.inside_bracket), float(x.delta_cm)


def compare_all() -> dict:
    import pandas as pd
    rev = _rev()
    d = _dirs()
    cases, refs, endf, mv = _load()
    full, _, _ = _views(cases)
    if d["rec"].exists() and any(d["rec"].rglob("*.npz")):
        s, reuse = subrun_sums(cases, endf)
        s.to_csv(d["out"] / "subrun_sums.csv", index=False)
        (d["out"] / "records_manifest.json").write_text(json.dumps(records_manifest(), indent=1), encoding="utf-8")
        (d["out"] / "reuse_check.json").write_text(json.dumps(reuse, indent=1), encoding="utf-8")
    else:
        s = pd.read_csv(d["out"] / "subrun_sums.csv")
        reuse = json.loads((d["out"] / "reuse_check.json").read_text(encoding="utf-8"))
    from shieldmap.analysis4.mc import compare
    pts = compare.points_table(full, refs, endf, s)
    pts.insert(3, "origin", [("phase4d/c8 (new)" if m in NEW else f"phase4a/{REUSE[m]} (reused)") for m in pts.material])
    pts.to_csv(d["out"] / "points.csv", index=False)
    req = compare.requirements(full, refs, pts)
    req.to_csv(d["out"] / "requirements.csv", index=False)
    met = metrics(full, req)
    met.to_csv(d["out"] / "metrics.csv", index=False)
    x2 = {"new_points": rev.chi2(pts[pts.material.isin(NEW)]), "all_points": rev.chi2(pts),
          "new_materials": list(NEW)}
    (d["out"] / "chi2.json").write_text(json.dumps(x2, indent=1), encoding="utf-8")
    B = {}
    for m in full["cases"][0]["materials"]:
        r = pts[(pts.material == m) & np.isclose(pts.factor, 1.0)].iloc[0]
        B[m] = {"t_cm": float(r.t_cm), "factor": 1.0, "narrow": float(r.narrow), "buildup": float(r.buildup),
                "buildup_se": float(r.buildup_se), "origin": r.origin}
    prices = {}
    for est in ("T_tablec", "T_endf_narrow", "narrow", "cone1", "fl", "fl_all", "broad"):
        v, se, ins, dl = _get(met, "W", "Ir", est)
        prices[est] = {"value": v, "se": se, "ci95_pp": 1.96 * 100 * se if np.isfinite(se) else None,
                       "inside_bracket": ins, "delta_cm": dl}
    ok = all(prices[e]["ci95_pp"] <= PRICE_PP_TARGET for e in ("narrow", "broad"))
    summ = {"case_id": CASE["id"], "n_subruns_new": int((s.origin.str.startswith("phase4d")).sum()),
            "n_subruns_reused": int((~s.origin.str.startswith("phase4d")).sum()), "n_points": int(len(pts)),
            "particles_new": int(s[s.origin.str.startswith("phase4d")].n_particles.sum()),
            "openmc_seconds_new": float(s[s.origin.str.startswith("phase4d")].seconds.sum()),
            "any_truncated": bool(s.truncated.any()), "materials": full["cases"][0]["materials"],
            "densities_g_cm3": {k: v["rho_used"] for k, v in cases["materials"].items()},
            "price_W_Ir": prices, "precision_target_pp": PRICE_PP_TARGET, "precision_target_met": bool(ok),
            "buildup_at_tablec_T001_slab": B, "chi2": x2, "map": mv, "reuse": reuse}
    (d["out"] / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    base = compare.digest(full, pts.drop(columns="origin"), req, met)
    L = [base.replace("# MC spot check: digest (generated by compare.py; do not edit)",
                      "# MC case c8_cs137_irw: digest (generated by analysis4d.mc_c8; do not edit)"), "",
         "Ir rows reuse Phase 4a c1_cs137 sub-runs and Pb rows reuse c6_cs137_pb (identical definition, checked by "
         "`mc_c8.build`); W is new.", "",
         f"Narrow MC vs ENDF analytic, new W slabs: χ² = {x2['new_points']['chi2_narrow_vs_endf']:.2f} / "
         f"{x2['new_points']['dof']} (p = {x2['new_points']['p_value']:.3f}); max |z| = "
         f"{x2['new_points']['max_abs_z']:.2f}. All {x2['all_points']['dof']} slabs: χ² = "
         f"{x2['all_points']['chi2_narrow_vs_endf']:.2f}.", "",
         "Build-up B = broad/narrow at the Table C T = 0.01 slab (factor 1.0): "
         + "; ".join(f"{m} {b['buildup']:.4f} ± {b['buildup_se']:.4f}" for m, b in B.items()), "",
         f"Price W/Ir − 1 (map H11 {100 * mv['H11']['value']:.2f} %): "
         + "; ".join(f"{e} {100 * p['value']:.2f} ± {100 * p['se']:.2f} %" for e, p in prices.items()
                     if np.isfinite(p['se'])) + "; "
         + "; ".join(f"{e} {100 * p['value']:.2f} %" for e, p in prices.items() if not np.isfinite(p['se'])), ""]
    (d["out"] / "digest.md").write_text("\n".join(L), encoding="utf-8")
    return summ


# ============================================================================================ rows H430+
def rows() -> list[dict]:
    import pandas as pd
    from shieldmap import paths
    d = _dirs()
    v3 = pd.read_csv(paths.DATA / "phase4b" / "headline_table_v3.csv", dtype=str, keep_default_na=False)
    summ = json.loads((d["out"] / "summary.json").read_text(encoding="utf-8"))
    met = pd.read_csv(d["out"] / "metrics.csv")
    mv, x2, B, P = summ["map"], summ["chi2"], summ["buildup_at_tablec_T001_slab"], summ["price_W_Ir"]
    rho = summ["densities_g_cm3"]
    case_txt = ("Monte Carlo (OpenMC 0.16.0, ENDF/B-VIII.0), case c8_cs137_irw (Cs-137, 661.657 keV line, T = 0.01, "
                "laterally infinite slab, normal-incidence parallel beam, air kerma at the back face)")
    common = {"block": "phase4d_mc", "quoted_in": "not quoted", "decision": "P4D", "v3_origin": "phase4d",
              "basis": "thickness", "artefact_reference": "False", "mc_status": "Monte Carlo result",
              "v3_note": "Phase 4d, MC case c8 (handover §3.3; reports/phase4d/mc_c8.md)",
              "density_source": f"measured-density point (Phase 1 element densities: Ir {rho['Ir']}, W {rho['W']}, "
                                f"Pb {rho['Pb']} g/cm³)"}
    ci = "Monte Carlo statistical 95 % interval (± 1.96 SE)"
    out = []

    def add(**kw):
        r = dict(common, id=f"H{ROW0 + len(out)}", **kw)
        r["statement_v3"] = r["statement"]
        out.append(r)

    est_txt = {"narrow": ("narrow-beam (uncollided) estimator", "Monte Carlo narrow beam (scored uncollided)"),
               "broad": ("broad-beam estimator (every photon: build-up, fluorescence, bremsstrahlung and "
                         "annihilation included)", "Monte Carlo broad beam (build-up included)"),
               "fl": ("fluorescence-inclusive estimator (uncollided + escaping lines ≥ 3.2029 keV)",
                      "Monte Carlo (fluorescence transported)")}
    for est in ("narrow", "broad", "fl"):
        p = P[est]
        others = [f"Table C (XCOM) at the MC slabs {100 * P['T_tablec']['value']:.2f} %",
                  f"map H11 {100 * mv['H11']['value']:.2f} %",
                  f"ENDF analytic narrow {100 * P['T_endf_narrow']['value']:.2f} %"]
        others += [f"MC {e} {100 * P[e]['value']:.2f} ± {100 * P[e]['se']:.2f} %" for e in ("narrow", "fl", "broad")
                   if e != est]
        add(dossier_result="R2.3", quantity="price", value=p["value"], value_pct=100 * p["value"], unit="%",
            band_lo_pct=100 * (p["value"] - 1.96 * p["se"]), band_hi_pct=100 * (p["value"] + 1.96 * p["se"]),
            band_type=ci, delta_abs=p["delta_cm"], delta_unit="cm", delta_note=f"t_W − t_Ir at T = 0.01 ({est})",
            statement=f"{case_txt}: price of the scarcity proxy (crustal abundance ≥ 1 ppm) given B4, thickness "
                      f"basis, W/Ir − 1 (Ir → W at the thickness frontier, H11), {est_txt[est][0]}",
            frontier_kind="best_clean", reference_set="B4-internal (ABUND(1)|B4)", reference_compounds="Ir → W",
            reference_mpids="mp-101 → mp-91", fluorescence_corrected=est_txt[est][1],
            sensitivity_values="; ".join(others), source_file="data/phase4d/mc_c8/metrics.csv",
            source_key=f"case_id=c8_cs137_irw; num=W; den=Ir; estimator={est}; column=value")
    for m in ("Ir", "W"):
        b = B[m]
        add(dossier_result="R-BU", quantity="value", value=b["buildup"], unit="ratio",
            band_lo_pct=b["buildup"] - 1.96 * b["buildup_se"], band_hi_pct=b["buildup"] + 1.96 * b["buildup_se"],
            band_type=ci + "; ratio, not %",
            statement=f"{case_txt}: build-up factor B = broad / narrow air kerma of {m} at its Table C T = 0.01 "
                      f"slab (t = {b['t_cm']:.4f} cm)",
            reference_set="n/a (not a price)", reference_compounds=m,
            reference_mpids={"Ir": "mp-101", "W": "mp-91"}[m], fluorescence_corrected="Monte Carlo broad / narrow",
            sensitivity_values=f"{b['origin']}; MC narrow T at this slab {b['narrow']:.5f}",
            source_file="data/phase4d/mc_c8/points.csv",
            source_key=f"case_id=c8_cs137_irw; material={m}; factor=1.000; column=buildup")
    for est in ("narrow", "broad"):
        v, se, ins, dl = _get(met, "W", "Pb", est)
        add(dossier_result="R2.1", quantity="value", value=v + 1, unit="ratio", band_lo_pct=v + 1 - 1.96 * se,
            band_hi_pct=v + 1 + 1.96 * se, band_type=ci + "; ratio, not %",
            delta_abs=dl, delta_unit="cm",
            delta_note=f"t_W − t_Pb at T = 0.01 ({est})",
            statement=f"{case_txt}: Pb-relative thickness t_W / t_Pb at T = 0.01 (H67), {est_txt[est][0]}",
            reference_set="n/a (not a price)", reference_compounds="W (mp-91) vs Pb (mp-20483)",
            reference_mpids="mp-91; mp-20483", fluorescence_corrected=est_txt[est][1],
            sensitivity_values=f"map H67 {mv['H67']['value']:.4f}; Table C at the MC slabs "
                               f"{mv['tablec_at_mc_densities']['ratio_W_Pb']:.4f}; Pb slabs reused from Phase 4a c6",
            source_file="data/phase4d/mc_c8/metrics.csv",
            source_key=f"case_id=c8_cs137_irw; num=W; den=Pb; estimator={est}; column=value (+1)")
    xn, xa = x2["new_points"], x2["all_points"]
    add(dossier_result="R-BU", quantity="value", value=xn["chi2_narrow_vs_endf"], unit="chi2",
        band_type=f"none (statistic; {xn['dof']} degrees of freedom)",
        statement=f"{case_txt}, consistency check: χ² of the narrow-beam (uncollided) MC transmission of W against "
                  f"the ENDF/B-VIII.0 analytic narrow-beam transmission over the {xn['dof']} new W slabs",
        reference_set="n/a (model check)", reference_compounds="W", reference_mpids="mp-91",
        fluorescence_corrected="Monte Carlo narrow beam (scored uncollided)",
        sensitivity_values=f"dof {xn['dof']}; p = {xn['p_value']:.2f}; max |z| {xn['max_abs_z']:.2f}; MC narrow / "
                           f"Table C {xn['ratio_narrow_mc_tablec_min']:.4f}–{xn['ratio_narrow_mc_tablec_max']:.4f}; "
                           f"with the reused Ir and Pb slabs χ² = {xa['chi2_narrow_vs_endf']:.2f} / {xa['dof']}",
        source_file="data/phase4d/mc_c8/chi2.json", source_key="new_points.chi2_narrow_vs_endf")
    df = pd.DataFrame(out).reindex(columns=v3.columns)
    df.to_csv(paths.DATA / "phase4d" / "rows_mc_c8.csv", index=False, encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "compare", "rows", "wsl-endf", "wsl-stubs", "wsl-run"])
    ap.add_argument("--repo", default="/mnt/c/Users/USER/Desktop/shield/shieldmap")
    ap.add_argument("--deadline", default="", help="ISO time; wsl-run stops before a new sub-run after it")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        export_inputs()
    elif a.cmd == "compare":
        print(json.dumps(compare_all(), indent=1, default=str))
    elif a.cmd == "rows":
        for r in rows():
            print(r["id"], r["value"], r.get("band_lo_pct", ""), r.get("band_hi_pct", ""), r["statement"][-90:])
    elif a.cmd == "wsl-endf":
        wsl_endf(Path(a.repo))
    elif a.cmd == "wsl-stubs":
        wsl_stubs(Path(a.repo))
    else:
        if not a.deadline:
            raise SystemExit("--deadline is required")
        wsl_run(Path(a.repo), a.deadline, a.threads)


if __name__ == "__main__":
    # as a script (WSL side) the package directory must not shadow third-party modules
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    main()
