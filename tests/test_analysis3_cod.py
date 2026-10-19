"""Stage 3 the analysis3/cod.py: COD filters, density, matcher selection rule, class fallback.

Fast synthetic tests only (no archive, no network). The fixture CIF is a synthetic rock-salt NaCl.
"""
import json

import numpy as np
import pandas as pd
import pytest

from shieldmap import paths
from shieldmap.analysis3 import cod

FIX = paths.FIXTURES / "phase3" / "nacl_synthetic.cif"
NAME = "cif/9/99/99/9999999.cif"
NACL = frozenset({"NaCl"})


@pytest.fixture(scope="module")
def cif() -> str:
    return FIX.read_text(encoding="utf-8")


def screen(text, formulas=NACL):
    return cod.screen_cif(NAME, text.encode(), formulas)


# ------------------------------------------------------------------------------------- parsing helpers
def test_num():
    assert cod.num("293(2)") == 293.0
    assert cod.num("1.63(5)E6") == 1.63
    assert cod.num("1.5E6") == 1.5e6
    assert cod.num("0.0410") == 0.041
    assert cod.num("?") is None and cod.num(".") is None and cod.num(None) is None
    assert cod.num("room temperature") is None


@pytest.mark.parametrize("line, expect", [
    ("_chemical_formula_sum 'Cl Na'\n", "Cl Na"),
    ("_chemical_formula_sum            Fe2O3\n", "Fe2O3"),
    ("_chemical_formula_sum\n;\nAl2 O3\n;\n", "Al2 O3"),
    ("_chemical_formula_sum \"Ge Mn2 O4\"\n", "Ge Mn2 O4"),
    ("_chemical_formula_sum ?\n", None),
])
def test_formula_sum_text(line, expect):
    assert cod.formula_sum_text("data_x\n" + line + "_cell_length_a 1\n") == expect


def test_reduced_formula_matches_table_b_convention():
    assert cod.reduced_formula("Ge Mn2 O4")[0] == "Mn2GeO4"
    assert cod.reduced_formula("Cl4 Na4")[0] == "NaCl"


# ------------------------------------------------------------------------------------- filters
def test_screen_kept(cif):
    r = screen(cif)
    assert r["reason"] == "kept"
    assert r["formula_reduced"] == "NaCl"
    assert r["r_factor"] == pytest.approx(0.031) and r["r_tag"] == "_refine_ls_r_factor_gt"
    assert r["temperature_K"] == 293.0
    assert "cif" in r


def test_screen_prefilters(cif):
    assert screen(cif, frozenset({"KCl"}))["reason"] == "not_in_mp_formula_set"
    assert screen(cif.replace("'Cl Na'", "'C2 H6 Cl Na'"), frozenset({"NaC2H6Cl"}))["reason"] == "organic"
    assert screen(cif.replace("_chemical_formula_sum            'Cl Na'\n", ""))["reason"] == "no_formula_sum"


@pytest.mark.parametrize("old, new, reason", [
    ("Na1 Na 0.0 0.0 0.0 1.0", "Na1 Na 0.0 0.0 0.0 0.5", "occupancy_not_unity"),
    ("Na1 Na 0.0 0.0 0.0 1.0", "Na1 Na 0.0 0.0 0.0 1.02", "occupancy_not_unity"),
    ("_diffrn_ambient_temperature      293(2)", "_diffrn_ambient_temperature      400(2)", "temperature_gt_350K"),
    ("_diffrn_ambient_temperature      293(2)", "_cell_measurement_temperature 351", "temperature_gt_350K"),
    ("_cod_database_code", "_diffrn_ambient_pressure 3e+06\n_cod_database_code", "high_pressure"),
    ("_cod_database_code", "_cell_measurement_pressure_gPa 2.5\n_cod_database_code", "high_pressure"),
    ("_cod_database_code", "_diffrn_ambient_environment 'diamond-anvil cell'\n_cod_database_code", "high_pressure"),
    ("_refine_ls_R_factor_gt           0.031", "_refine_ls_R_factor_gt 0.2", "r_factor_gt_0.1"),
    ("_refine_ls_R_factor_gt           0.031", "_refine_ls_R_factor_all 0.15", "r_factor_gt_0.1"),
    ("_cod_database_code", "_cod_struct_determination_method theoretical\n_cod_database_code", "theoretical_method"),
    ("_cod_database_code", "_cod_error_flag errors\n_cod_database_code", "cod_error_or_retraction"),
    ("_cod_database_code", "_cod_error_flag retracted\n_cod_database_code", "cod_error_or_retraction"),
])
def test_screen_quality_filters(cif, old, new, reason):
    assert old in cif
    r = screen(cif.replace(old, new))
    assert r["reason"] == reason
    assert reason in r["all_fails"]


def test_screen_disorder(cif):
    """A disorder_group column of placeholders ('.') means no disorder (kept); a real group value is dropped."""
    lines = cif.splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if ln == "_atom_site_occupancy":
            out.append("_atom_site_disorder_group")
    t_dot = "\n".join(ln + " ." if ln.endswith(" 1.0") and ln[:2] in ("Na", "Cl") else ln for ln in out) + "\n"
    assert screen(t_dot)["reason"] == "kept"
    t_grp = t_dot.replace("Cl4 Cl 0.5 0.5 0.5 1.0 .", "Cl4 Cl 0.5 0.5 0.5 1.0 2")
    assert screen(t_grp)["reason"] == "disorder_flag"


@pytest.mark.parametrize("old, new", [
    ("_diffrn_ambient_temperature      293(2)", "_diffrn_ambient_temperature      350"),      # boundary kept
    ("_refine_ls_R_factor_gt           0.031", "_refine_ls_R_factor_gt 0.100"),             # boundary kept
    ("_refine_ls_R_factor_gt           0.031", "_refine_ls_R_factor_gt ?"),                 # R not given: kept
    ("_cod_database_code", "_cell_measurement_pressure 101.3\n_cod_database_code"),          # ambient kPa
    ("_cod_database_code", "_cod_error_flag warnings\n_cod_database_code"),                  # warnings kept
    ("_diffrn_ambient_temperature      293(2)", "_diffrn_ambient_temperature      100(2)"),  # low T kept
])
def test_screen_boundaries_kept(cif, old, new):
    assert screen(cif.replace(old, new))["reason"] == "kept"


def test_screen_first_reason_order(cif):
    t = cif.replace("293(2)", "400").replace("0.031", "0.5")
    r = screen(t)
    assert r["reason"] == "temperature_gt_350K"
    assert r["all_fails"] == "temperature_gt_350K,r_factor_gt_0.1"


# ------------------------------------------------------------------------------------- density
def test_density_from_cif(cif):
    p = cod.parse_one(cif, "NaCl", 2.164)
    assert p["stage"] == "parsed_ok"
    expected = 4 * (22.98976928 + 35.453) / (6.02214076e23 * (5.64e-8) ** 3)
    assert p["rho_cod"] == pytest.approx(expected, rel=1e-4)
    assert abs(p["rho_crosscheck"]) < 1e-3


def test_density_crosscheck_and_composition(cif):
    assert cod.parse_one(cif, "NaCl", 2.164 * 1.05)["stage"] == "density_crosscheck_fail"
    assert cod.parse_one(cif, "NaCl", None)["stage"] == "parsed_ok"
    assert cod.parse_one(cif, "KCl", None)["stage"] == "composition_mismatch"
    assert cod.parse_one("data_x\n_cell_length_a 5\n", "NaCl", None)["stage"] in ("parse_failed", "not_one_structure")


# ------------------------------------------------------------------------------------- matching
def _rocksalt(a):
    from pymatgen.core import Lattice, Structure
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _cscl(a):
    from pymatgen.core import Lattice, Structure
    return Structure.from_spacegroup("Pm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_structure_matcher_defaults_with_prereduction(cif):
    p = cod.parse_one(cif, "NaCl", None)
    red = cod._reduce_task([("cod", p["structure_json"]),
                            ("mp_rs", json.dumps(_rocksalt(5.50).as_dict())),     # 2.5 % smaller cell
                            ("mp_cscl", json.dumps(_cscl(3.0).as_dict()))])
    d = {k: (s, n) for k, s, n in red}
    assert d["cod"][1] == d["mp_rs"][1] == 2 and d["mp_cscl"][1] == 2
    res = cod._match_task(([("mp_rs", d["mp_rs"][0]), ("mp_cscl", d["mp_cscl"][0])], [(1, d["cod"][0])]))
    assert dict(((m, ok) for m, _, ok in res)) == {"mp_rs": True, "mp_cscl": False}
    # pre-reduction + skip_structure_reduction gives the same answer as plain default fit()
    from pymatgen.analysis.structure_matcher import StructureMatcher
    assert bool(StructureMatcher().fit(_rocksalt(5.50), _rocksalt(5.64)))


def test_select_cod_rules():
    nan = float("nan")
    one = pd.DataFrame({"cod_id": [5], "r_factor": [nan], "rho_cod": [3.0]})
    assert cod.select_cod(one) == {"cod_id": 5, "rho_cod": 3.0, "rule": "single", "n_set": 1}
    lowest = pd.DataFrame({"cod_id": [1, 2, 3], "r_factor": [0.05, 0.03, nan], "rho_cod": [3.0, 3.1, 9.0]})
    s = cod.select_cod(lowest)
    assert (s["cod_id"], s["rho_cod"], s["rule"]) == (2, 3.1, "lowest_R")
    tied = pd.DataFrame({"cod_id": [1, 2, 3, 4], "r_factor": [0.03, 0.03, 0.03, 0.08],
                         "rho_cod": [3.0, 3.2, 3.1, 1.0]})
    s = cod.select_cod(tied)
    assert s["rule"] == "median_tied_R" and s["rho_cod"] == pytest.approx(3.1) and s["cod_id"] == 3
    assert s["n_set"] == 3
    nor = pd.DataFrame({"cod_id": [7, 8], "r_factor": [nan, nan], "rho_cod": [3.0, 3.2]})
    s = cod.select_cod(nor)
    assert s["rule"] == "median_no_R" and s["rho_cod"] == pytest.approx(3.1) and s["cod_id"] == 7


def test_ehull_bins():
    e = np.array([0.0, 1e-4, 0.025, 0.0251, 0.05, 0.1, 0.1001])
    assert list(cod.ehull_bin(e)) == ["0", "(0,0.025]", "(0,0.025]", "(0.025,0.05]", "(0.025,0.05]",
                                      "(0.05,0.1]", ">0.1"]


# ------------------------------------------------------------------------------------- class distributions
def test_class_distribution_fallback():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"primary_class": ["oxide"] * 40 + ["halide"] * 5,
                       "delta": np.r_[rng.normal(-0.01, 0.01, 40), np.full(5, 0.3)]})
    d, info = cod.class_distributions(df, min_n=30)
    assert len(d["oxide"]) == 40 and not info["oxide"]["uses_pooled"]
    assert info["halide"]["uses_pooled"] and info["halide"]["n_class"] == 5
    np.testing.assert_array_equal(d["halide"], df["delta"].to_numpy())
    np.testing.assert_array_equal(d["pooled"], df["delta"].to_numpy())
    assert info["f_electron"]["n_class"] == 0 and info["f_electron"]["uses_pooled"]
    assert set(d) == set(cod.common.PRIMARY_CLASS_ORDER) | {"pooled"}


def test_elemental_kind_and_band():
    from shieldmap.analysis import sensitivity
    el = sensitivity.bias_distribution("empirical")
    d = cod.class_delta_distribution("elemental")
    assert len(el) == 77
    for c in cod.common.PRIMARY_CLASS_ORDER:
        np.testing.assert_array_equal(d[c], el)
    lo, hi = cod.band("oxide", kind="elemental")
    assert lo == pytest.approx(np.percentile(el, 5)) and hi == pytest.approx(np.percentile(el, 95))
    with pytest.raises(ValueError):
        cod.class_delta_distribution("bogus")


def test_stats():
    s = cod.stats(np.array([-0.02, -0.01, 0.0, 0.01, 0.10]))
    assert s["n"] == 5 and s["median"] == 0.0 and s["frac_abs_gt_5pct"] == pytest.approx(0.2)
    assert s["median_ci95"][0] <= 0.0 <= s["median_ci95"][1]


@pytest.mark.skipif(not (cod.OUT / "delta_by_primary_class.parquet").exists(), reason="run() outputs absent")
def test_compound_kind_applies_the_robust_screen():
    d = pd.read_parquet(cod.OUT / "delta_by_primary_class.parquet")
    raw = cod.class_delta_distribution("compound_raw")["pooled"]
    scr = cod.class_delta_distribution("compound")["pooled"]
    old = cod.class_delta_distribution("compound_abs50")["pooled"]
    assert len(raw) == len(d)
    assert len(scr) == int((~d["excluded"]).sum()) and len(old) == int((~d["gross_abs50"]).sum())
    assert np.all(np.abs(d.loc[~d["excluded"], "z_robust"]) <= cod.SCREEN_K)
    assert np.all(np.abs(old) <= cod.GROSS_DELTA)
    for c in cod.common.PRIMARY_CLASS_ORDER:
        lo, hi = cod.band(c)
        assert lo < 0 < hi or lo < hi


# ------------------------------------------------------------------------------------- the hardening revision screen
def test_robust_z_is_log_symmetric():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 0.03, 2000)
    cls = np.array(["oxide"] * ref.size, dtype=object)
    up, down = np.log(1.5), np.log(1 / 1.5)
    z, med, sig = cod.robust_z(np.array([up, down]), np.array(["oxide", "oxide"], dtype=object), ref, cls)
    assert z[0] == pytest.approx(-z[1] + 2 * (0 - med[0]) / sig[0], abs=1e-12)    # symmetric about the median
    assert sig[0] == pytest.approx(1.4826 * np.median(np.abs(ref - np.median(ref))))
    z2, med2, _ = cod.robust_z(np.array([0.0]), np.array(["halide"], dtype=object), ref, cls)
    assert med2[0] == pytest.approx(np.median(ref))                                  # absent class → pooled


def test_screen_matches_flags_outliers_and_element_crosscheck(monkeypatch):
    rng = np.random.default_rng(2)
    n = 400
    m = pd.DataFrame({"delta": np.r_[rng.normal(-0.02, 0.02, n), -0.186, 0.6, -0.176, -0.007],
                      "primary_class": ["intermetallic"] * (n + 2) + ["intermetallic", "intermetallic"],
                      "nelements": [2] * (n + 2) + [1, 1],
                      "formula_reduced": ["AB"] * (n + 2) + ["Re", "Re"],
                      "rho_mp": [10.0] * (n + 2) + [20.873, 20.877]})
    m["rho_cod"] = m["rho_mp"] / (1 + m["delta"])
    ref = pd.DataFrame({"symbol": ["Re"], "density_solid_exp": [21.02], "rep_material_id": ["mp-x"], "rep_sg": [194]})
    monkeypatch.setattr(cod, "element_reference", lambda u=None: ref)
    out = cod.screen_matches(m, u=pd.DataFrame())
    assert out["excluded"].iat[n] and out["excluded"].iat[n + 1]                    # −18.6 % and +60 %
    assert not out["excluded"].iloc[:n].mean() > 0.01
    assert out["xcheck_fail"].iat[n + 2] and out["excluded"].iat[n + 2]              # COD Re at 25.3 g/cm3
    assert not out["xcheck_fail"].iat[n + 3]                                         # consistent Re
    assert out["gross_abs50"].iat[n + 1] and not out["gross_abs50"].iat[n]           # the old rule missed −18.6 %


def test_cluster_median_ci_widens_with_clusters():
    rng = np.random.default_rng(3)
    shifts = rng.normal(0, 0.02, 50)
    x = np.repeat(shifts, 20) + rng.normal(0, 0.001, 1000)
    g = np.repeat(np.arange(50), 20)
    lo, hi = cod.cluster_median_ci(x, g, n_boot=400)
    iid = cod.stats(x)["median_ci95"]
    assert lo <= np.median(x) <= hi
    assert (hi - lo) > 2 * (iid[1] - iid[0])


def test_tracked_checksums_mirror_source_yaml():
    import yaml
    c = yaml.safe_load((paths.CONFIG / "sources" / "cod_rev297631.yaml").read_text(encoding="utf-8"))
    assert c["files"][0]["sha256"] == cod.COD_SHA256 and c["files"][0]["bytes"] == cod.COD_SIZE
    assert c["files"][0]["md5"] == cod.COD_MD5
    m = yaml.safe_load((paths.CONFIG / "sources" / "mp_structures_cod.yaml").read_text(encoding="utf-8"))
    assert m["files"][0]["sha256"] == cod.MP_STRUCT_SHA256[m["database_version"]]


def test_cache_fingerprint_refuses_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(cod, "CACHE_FP", tmp_path / "fp.json")
    cod._cache_check("parse", "aaa")                          # adopted
    cod._cache_check("parse", "aaa")
    with pytest.raises(RuntimeError, match="stale"):
        cod._cache_check("parse", "bbb")
    assert cod._fingerprint(a=1, b=[2]) == cod._fingerprint(b=[2], a=1) != cod._fingerprint(a=2, b=[2])
