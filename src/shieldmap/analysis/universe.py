"""Verify stage 1 outputs, build the analysis universe and the dense profile cache.

Read-only on stage 1: Tables A/B/C are only read. Outputs go to data/phase2/:

  universe.parquet         one row per Table B / Table C row (same order) with eligibility, the
                           reason for exclusion, membership in the representative rules R1/R2/R3,
                           and the attributes the constraints need
  profiles_thickness.npy   float32 (N × 264) = Table C `thickness_req_cm`, pivoted by variant_id
  profiles_areal.npy       float32 (N × 264) = Table C `areal_mass_req_g_cm2`
  universe_summary.json    counts for the report

Baseline eligibility removes database artefacts only: deprecated (already absent),
ρ < 1.0 g/cm³, null energy_above_hull, noble-gas compounds.

Representative rules, applied *within the eligible universe*:
  R1  lowest e_hull per reduced formula (Table B's rule; ties → lowest numeric id)
  R2  lowest e_hull among ICSD-matched (theoretical == False) entries; R1 if none
  R3  every eligible entry
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from shieldmap import paths
from shieldmap.ingest import fetch

OUT = paths.DATA / "phase2"
MIN_DENSITY = 1.0
N_VARIANTS = 264


def verify_manifest() -> dict:
    """SHA-256 of every stage 1 output listed in MANIFEST.json must match the file on disk."""
    man = json.loads((paths.PROCESSED / "MANIFEST.json").read_text(encoding="utf-8"))
    bad, checked = [], 0
    for o in man["outputs"]:
        p = paths.ROOT / o["path"]
        if not p.exists():
            bad.append((o["path"], "missing"))
            continue
        if fetch.sha256(p) != o["sha256"]:
            bad.append((o["path"], "sha256 mismatch"))
        checked += 1
    return {"checked": checked, "failures": bad, "build_finished_utc": man.get("build_finished_utc"),
            "config_sha256": man["config"].get("sha256_at_start")}


def _id_key(df: pd.DataFrame) -> pd.Series:
    num = pd.to_numeric(df["id_number"], errors="coerce").fillna(np.inf)
    return num


def build_universe() -> tuple[pd.DataFrame, dict]:
    tb = pd.read_parquet(paths.PROCESSED / "table_b_compounds.parquet")
    idx = pd.read_parquet(paths.PROCESSED / "index.parquet")
    if not (idx["material_id"].to_numpy() == tb["material_id"].to_numpy()).all():
        raise RuntimeError("Table B and Table C index disagree on row order")
    keep_cols = ["material_id", "formula_pretty", "formula_reduced", "chemsys", "nelements", "nsites",
                 "density", "volume", "energy_above_hull", "theoretical", "n_icsd_ids", "band_gap",
                 "is_metal", "spacegroup_number", "crystal_system", "deprecated", "is_representative",
                 "elements", "min_crustal_abundance_ppm", "abundance_weighted_log_ppm",
                 "contains_radioactive", "contains_noble_gas", "id_prefix", "id_number"]
    u = tb[keep_cols].copy()
    u.insert(0, "row", np.arange(len(u), dtype=np.int32))

    reasons = np.full(len(u), "", dtype=object)
    r_dep = u["deprecated"].fillna(False).to_numpy()
    r_rho = (u["density"] < MIN_DENSITY).to_numpy()
    r_eh = u["energy_above_hull"].isna().to_numpy()
    r_ng = u["contains_noble_gas"].to_numpy()
    for flag, name in ((r_dep, "deprecated"), (r_rho, "density<1"), (r_eh, "ehull_null"), (r_ng, "noble_gas")):
        reasons = np.where(flag, np.where(reasons == "", name, reasons + ";" + name), reasons)
    u["exclusion_reason"] = reasons
    u["eligible"] = reasons == ""

    # representative rules within the eligible universe
    e = u[u.eligible].copy()
    e["_id"] = _id_key(e)
    order = e.sort_values(["formula_reduced", "energy_above_hull", "_id"], kind="mergesort")
    r1 = order.groupby("formula_reduced", sort=False).head(1).index
    icsd = order[~order["theoretical"]]
    r2_icsd = icsd.groupby("formula_reduced", sort=False).head(1)
    no_icsd = set(order["formula_reduced"]) - set(r2_icsd["formula_reduced"])
    r2 = r2_icsd.index.union(order[order["formula_reduced"].isin(no_icsd)].groupby("formula_reduced", sort=False).head(1).index)
    u["in_R1"] = False; u.loc[r1, "in_R1"] = True
    u["in_R2"] = False; u.loc[r2, "in_R2"] = True
    u["in_R3"] = u["eligible"]
    u["R2_from_icsd"] = u["in_R2"] & ~u["theoretical"]

    # how R1 relates to Table B's own flag, and the artefact removals
    dens_cas = u[r_rho & ~r_dep].sort_values("density", ascending=False).head(1)
    summary = {
        "n_table_b": int(len(u)),
        "n_eligible": int(u.eligible.sum()),
        "removed": {"deprecated": int(r_dep.sum()), "density_lt_1": int(r_rho.sum()),
                    "ehull_null": int(r_eh.sum()), "noble_gas": int(r_ng.sum()),
                    "any": int((~u.eligible).sum())},
        "densest_density_casualty": dens_cas[["material_id", "formula_pretty", "density",
                                              "energy_above_hull"]].to_dict("records"),
        "n_formulas_eligible": int(e["formula_reduced"].nunique()),
        "n_R1": int(u.in_R1.sum()), "n_R2": int(u.in_R2.sum()), "n_R3": int(u.in_R3.sum()),
        "R2_from_icsd": int(u.R2_from_icsd.sum()),
        "R2_fallback_to_R1": int((u.in_R2 & u.theoretical).sum()),
        "R1_differs_from_R2_formulas": int(len(set(u.index[u.in_R1]) ^ set(u.index[u.in_R2])) // 2),
        "R1_differs_from_tableB_is_representative": int((u.in_R1 != (u.is_representative & u.eligible)).sum()),
        "formulas_whose_tableB_representative_is_ineligible": int(
            u[u.is_representative & ~u.eligible]["formula_reduced"].isin(e["formula_reduced"]).sum()),
    }
    return u, summary


def build_profile_cache(n_rows: int) -> dict:
    """Pivot the long Table C profiles into dense (N × 264) float32 arrays (a reshape, not new data)."""
    thick = np.lib.format.open_memmap(OUT / "profiles_thickness.npy", mode="w+", dtype=np.float32,
                                      shape=(n_rows, N_VARIANTS))
    areal = np.lib.format.open_memmap(OUT / "profiles_areal.npy", mode="w+", dtype=np.float32,
                                      shape=(n_rows, N_VARIANTS))
    thick[:] = np.nan; areal[:] = np.nan
    pf = pq.ParquetFile(paths.PROCESSED / "profiles.parquet")
    seen = 0
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["row", "variant_id", "thickness_req_cm", "areal_mass_req_g_cm2"])
        r = t.column("row").to_numpy(); v = t.column("variant_id").to_numpy()
        thick[r, v] = t.column("thickness_req_cm").to_numpy()
        areal[r, v] = t.column("areal_mass_req_g_cm2").to_numpy()
        seen += len(r)
    thick.flush(); areal.flush()
    n_nan = int(np.isnan(thick).sum() + np.isnan(areal).sum())
    if seen != n_rows * N_VARIANTS or n_nan:
        raise RuntimeError(f"profile cache incomplete: {seen} rows read, {n_nan} NaN cells")
    return {"profile_rows": seen}


def build() -> dict:
    paths.ensure(OUT)
    man = verify_manifest()
    if man["failures"]:
        raise SystemExit(f"Phase 1 outputs fail their manifest checksums: {man['failures']}")
    u, summary = build_universe()
    u.to_parquet(OUT / "universe.parquet", index=False)
    summary["manifest"] = man
    summary.update(build_profile_cache(len(u)))
    (OUT / "universe_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, default=str))
