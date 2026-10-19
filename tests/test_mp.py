"""Table B (Materials Project) checks on the built interim data. Offline; skipped if data are absent."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shieldmap import paths

OUT = paths.INTERIM / "mp"
FILES = ["compounds.parquet", "composition.parquet", "mass_fractions.npz",
         "mass_fractions_index.parquet"]

pytestmark = pytest.mark.skipif(not all((OUT / f).exists() for f in FILES),
                                reason="MP interim data not built (python -m shieldmap.ingest.mp)")


@pytest.fixture(scope="module")
def cmp() -> pd.DataFrame:
    return pd.read_parquet(OUT / "compounds.parquet")


@pytest.fixture(scope="module")
def comp() -> pd.DataFrame:
    return pd.read_parquet(OUT / "composition.parquet")


@pytest.fixture(scope="module")
def elements() -> pd.DataFrame:
    return pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet").set_index("symbol")


def test_unique_ids_and_order(cmp):
    assert cmp["material_id"].is_unique
    assert cmp["material_id_api"].is_unique
    key = list(zip(cmp["id_prefix"], cmp["id_number"]))
    assert key == sorted(key)


def test_ids_canonical(cmp):
    from emmet.core.mpid import AlphaID
    sample = cmp.sample(500, random_state=0) if len(cmp) > 500 else cmp
    for mid, api, num in zip(sample["material_id"], sample["material_id_api"], sample["id_number"]):
        a = AlphaID(api)
        assert a.string == mid and int(a) == num


def test_no_deprecated(cmp):
    assert not cmp["deprecated"].any()
    assert cmp["deprecated"].notna().all()


def test_mass_fractions_sum_to_one(comp, cmp):
    s = comp.groupby("material_id")["mass_fraction"].sum()
    assert len(s) == len(cmp)
    assert (np.abs(s - 1.0) < 1e-9).all()
    a = comp.groupby("material_id")["atom_fraction"].sum()
    assert (np.abs(a - 1.0) < 1e-9).all()
    assert (comp["mass_fraction"] > 0).all()
    assert comp["Z"].between(1, 100).all()


def test_one_representative_per_formula(cmp):
    reps = cmp[cmp["is_representative"]]
    assert reps["formula_reduced"].is_unique
    assert set(reps["formula_reduced"]) == set(cmp["formula_reduced"])


def test_representative_has_min_ehull(cmp):
    eh = cmp["energy_above_hull"]
    gmin = eh.groupby(cmp["formula_reduced"]).transform("min")
    reps = cmp[cmp["is_representative"]]
    rmin = gmin[reps.index]
    has = rmin.notna()
    # where the group has any non-null e_hull, the representative holds the minimum
    assert (reps.loc[has, "energy_above_hull"] == rmin[has]).all()
    # null-e_hull representatives only where the whole group is null
    assert reps.loc[~has, "energy_above_hull"].isna().all()
    # ties resolved to the lowest (prefix, id)
    tied = cmp[eh == gmin]
    first = tied.sort_values(["id_prefix", "id_number"]).drop_duplicates("formula_reduced")
    assert set(first["material_id"]) == set(reps.loc[has, "material_id"])


def test_element_flags(cmp, comp, elements):
    present = sorted(comp["symbol"].unique())
    for s in present:
        assert f"contains_{s}" in cmp.columns
    flagcols = [c for c in cmp.columns if c.startswith("contains_")
                and c not in ("contains_radioactive", "contains_noble_gas")]
    assert {c[len("contains_"):] for c in flagcols} == set(present)
    # contains_<X> agrees with the elements list
    for s in ("O", "Pb", "U", "H", "Bi"):
        if f"contains_{s}" in cmp.columns:
            assert (cmp[f"contains_{s}"] == cmp["elements"].map(lambda es, s=s: s in es)).all()
    if "contains_U" in cmp.columns:
        assert cmp.loc[cmp["contains_U"], "contains_radioactive"].all()
    radio = set(elements.index[elements["is_radioactive"]])
    noble = set(elements.index[elements["is_noble_gas"]])
    assert (cmp["contains_radioactive"] == cmp["elements"].map(lambda es: bool(set(es) & radio))).all()
    assert (cmp["contains_noble_gas"] == cmp["elements"].map(lambda es: bool(set(es) & noble))).all()
    bi_only = cmp[cmp["chemsys"] == "Bi"]
    assert len(bi_only) > 0
    assert not bi_only["contains_radioactive"].any()


def test_abundance_proxies(cmp, comp, elements):
    ab = elements["crustal_abundance_ppm"]
    missing = comp.assign(m=comp["symbol"].map(ab).isna()).groupby("material_id")["m"].any()
    missing = cmp["material_id"].map(missing)
    assert (cmp["min_crustal_abundance_ppm"].isna() == missing).all()
    assert (cmp["abundance_weighted_log_ppm"].isna() == missing).all()
    # recompute both for a sample
    sub = comp[comp["material_id"].isin(cmp.loc[~missing, "material_id"].sample(300, random_state=1))]
    a = sub["symbol"].map(ab)
    exp_min = a.groupby(sub["material_id"]).min()
    exp_w = (sub["mass_fraction"] * np.log10(a)).groupby(sub["material_id"]).sum()
    got = cmp.set_index("material_id")
    np.testing.assert_allclose(got.loc[exp_min.index, "min_crustal_abundance_ppm"], exp_min, rtol=0)
    np.testing.assert_allclose(got.loc[exp_w.index, "abundance_weighted_log_ppm"], exp_w,
                               rtol=1e-12, atol=1e-12)


def test_spot_check_si(cmp):
    si = cmp.set_index("material_id").loc["mp-149"]
    assert si["formula_reduced"] == "Si"
    assert si["density"] == pytest.approx(2.31, abs=0.01)
    assert not si["theoretical"]
    assert si["n_icsd_ids"] > 0
    assert si["spacegroup_number"] == 227


def test_sparse_matrix_matches_composition(cmp, comp):
    import scipy.sparse as sp
    W = sp.load_npz(OUT / "mass_fractions.npz")
    idx = pd.read_parquet(OUT / "mass_fractions_index.parquet")
    assert W.shape == (len(cmp), 100) and W.dtype == np.float64
    assert (idx["row"].to_numpy() == np.arange(len(cmp))).all()
    assert (idx["material_id"].to_numpy() == cmp["material_id"].to_numpy()).all()
    assert W.nnz == len(comp)
    assert np.all(np.abs(np.asarray(W.sum(axis=1)).ravel() - 1.0) < 1e-9)
    row_of = pd.Series(np.arange(len(cmp)), index=cmp["material_id"])
    r = comp["material_id"].map(row_of).to_numpy()
    vals = np.asarray(W[r, comp["Z"].to_numpy() - 1]).ravel()
    assert np.array_equal(vals, comp["mass_fraction"].to_numpy())


def test_theoretical_vs_icsd(cmp):
    # theoretical == False means matched to an ICSD structure; report-level consistency only
    assert cmp["theoretical"].notna().all()
    assert (cmp["n_icsd_ids"] == cmp["icsd_ids"].map(len)).all()


def test_source_record():
    from shieldmap.ingest import fetch
    rec = fetch.read_source("materials_project")
    assert rec is not None
    assert rec["data_class"] == "P2"
    assert rec["thermo_scheme"] in ("GGA_GGA+U_R2SCAN", "GGA_GGA+U", "R2SCAN")
    for f in rec["files"]:
        p = paths.ROOT / f["path"]
        if p.exists():
            assert fetch.sha256(p) == f["sha256"]
