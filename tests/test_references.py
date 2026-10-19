"""Offline checks of the standards / reference data built by ``shieldmap.ingest.references``.

Nothing here touches the network. Tests that need a built output skip when it is absent; checks
against raw files run only where the (git-ignored) raw files are present.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shieldmap import config, paths
from shieldmap.ingest import fetch
from shieldmap.ingest import references as R

SOURCE_IDS = ["ensdf_cs137", "ensdf_co60", "rqr_trs457", "nist_srd126"]


def _source(sid: str) -> dict:
    p = paths.SOURCES / f"{sid}.yaml"
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return pd.read_csv(p, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------- source records & checksums

@pytest.mark.parametrize("sid", SOURCE_IDS)
def test_source_record_has_citation_and_checksums(sid):
    rec = _source(sid)
    assert rec["id"] == sid
    for key in ("title", "citation", "data_class", "retrieval_method", "files", "data"):
        assert rec.get(key), f"{sid}: missing {key}"
    for f in rec["files"] + rec.get("provenance_files", []):
        assert len(f["sha256"]) == 64 and int(f["sha256"], 16) >= 0
        assert f["url"].startswith("https://")
        assert f["bytes"] > 0 and f["retrieved_utc"]


@pytest.mark.parametrize("sid", SOURCE_IDS)
def test_raw_files_match_recorded_sha256(sid):
    rec = _source(sid)
    checked = 0
    for f in rec["files"] + rec.get("provenance_files", []):
        p = paths.ROOT / f["path"]
        if not p.exists():
            continue
        digest = fetch.sha256(p)
        if digest != f["sha256"]:
            # NIST pages carry a per-request injected script; the content hash must then match.
            assert "content_sha256" in f, f"{p}: sha256 mismatch"
            assert R.html_content_sha256(p) == f["content_sha256"], f"{p}: content changed"
        if "content_sha256" in f:
            assert R.html_content_sha256(p) == f["content_sha256"]
        checked += 1
    if not checked:
        pytest.skip(f"{sid}: raw files not present")


# ---------------------------------------------------------------- ENSDF

def test_ensdf_uncertainty_convention():
    assert R.ensdf_unc("5.8E-4", "8") == (Decimal("0.8E-4"), None)
    assert R.ensdf_unc("99.9826", "6") == (Decimal("0.0006"), None)
    assert R.ensdf_unc("661.657", "3") == (Decimal("0.003"), None)
    assert R.ensdf_unc("-0.0025", "22") == (Decimal("0.0022"), None)
    assert R.ensdf_unc("2.0E-6", "4") == (Decimal("0.4E-6"), None)
    assert R.ensdf_unc("1.0", "LT") == (None, "LT")
    assert R.ensdf_unc("1.0", "") == (None, None)


@pytest.mark.parametrize("sid,n_used", [("ensdf_cs137", 1), ("ensdf_co60", 2)])
def test_ensdf_lines_used(sid, n_used):
    rec = _source(sid)
    d = rec["data"]
    used = d["lines_used"]
    assert len(used) == n_used
    for u in used:
        assert set(u) == {"energy_keV", "energy_unc_keV", "intensity_per_decay", "intensity_unc"}
        assert 0 < u["intensity_per_decay"] <= 1
        assert 0 < u["intensity_unc"] < u["intensity_per_decay"]
        assert u["energy_unc_keV"] > 0
        # every used line is a G record of the dataset
        assert any(g["energy_keV"] == u["energy_keV"] and g["used"] for g in d["all_gamma_lines"])
    # decay-line energies used elsewhere in the configuration are ENSDF's
    keys = set(config.get("key_energies_keV"))
    assert {u["energy_keV"] for u in used} <= keys
    assert d["parent"]["level_energy_keV"] in ("0.0", "0")
    assert d["dataset"]["dsid"].split()[1:3] == ["B-", "DECAY"]
    # the excluded lines are negligible compared with the used ones
    assert d["other_gamma_lines"]["fraction_of_used_lines"] < 1e-3


def test_ensdf_reparse_of_stored_records():
    """Each stored 80-column G record re-parses to the stored values."""
    for sid in ("ensdf_cs137", "ensdf_co60"):
        d = _source(sid)["data"]
        norm = d["normalization"]
        mult = Decimal(str(norm["multiplier_RI_to_per_100_decays"]))
        mult_unc = Decimal(str(norm["multiplier_unc"]))
        reparsed = R._parse_gammas([g["record"] for g in d["all_gamma_lines"]], mult, mult_unc)
        for a, b in zip(reparsed, d["all_gamma_lines"]):
            for k in ("energy_keV", "energy_unc_keV", "ri", "ri_unc", "intensity_per_decay",
                      "intensity_unc"):
                assert a[k] == b[k], (sid, k, a[k], b[k])


def test_ensdf_expected_datasets():
    cs = _source("ensdf_cs137")["data"]
    co = _source("ensdf_co60")["data"]
    assert cs["dataset"]["nucid"] == "137BA" and cs["parent"]["nucid"] == "137CS"
    assert co["dataset"]["nucid"] == "60NI" and co["parent"]["nucid"] == "60CO"
    # 60Co ground state (years), not the 10.5 min isomer
    assert 5 < co["parent"]["half_life_years"] < 6
    assert 29 < cs["parent"]["half_life_years"] < 32


def test_radionuclide_config_resolves():
    cs = config.get("radionuclides.Cs-137")
    co = config.get("radionuclides.Co-60")
    assert [l["energy_keV"] for l in cs] == [661.657]
    assert sorted(l["energy_keV"] for l in co) == [1173.228, 1332.492]


# ---------------------------------------------------------------- RQR

def test_rqr_values():
    d = _source("rqr_trs457")["data"]
    assert [r["name"] for r in d["all"]] == [f"RQR {n}" for n in range(2, 11)]
    kv = [r["kV"] for r in d["all"]]
    hvl = [r["hvl1_mm_al"] for r in d["all"]]
    assert kv == sorted(kv) and hvl == sorted(hvl)
    got = {r["name"]: (r["kV"], r["hvl1_mm_al"], r["homogeneity_h"]) for r in d["used"]}
    assert got == {"RQR 5": (70, 2.58, 0.71), "RQR 6": (80, 3.01, 0.69),
                   "RQR 8": (100, 3.97, 0.68), "RQR 9": (120, 5.00, 0.68)}
    assert config.get("xray_tubes.qualities") == d["used"]


def test_rqr_reparse_pdf():
    if not R.TRS457_PDF.exists():
        pytest.skip("TRS-457 PDF not present")
    parsed = R.parse_rqr()
    assert parsed["pdf_page"] == 83 and parsed["printed_page"] == "71"
    assert parsed["used"] == _source("rqr_trs457")["data"]["used"]


# ---------------------------------------------------------------- SRD 126

def _check_monotone(df: pd.DataFrame, label: str):
    e = df["energy_MeV"].tolist()
    side = df["edge_side"].tolist()
    for i in range(1, len(e)):
        if side[i] == "above":
            assert side[i - 1] == "below" and e[i] == e[i - 1], (label, i)
        else:
            assert e[i] > e[i - 1], (label, i)


def test_srd126_fixture_rows():
    mu = _csv(R.FIXTURE_MU)
    rec = _source("nist_srd126")
    pages = rec["pages"]
    assert set(mu["material"]) == set(pages)
    for mat, g in mu.groupby("material", sort=False):
        assert len(g) == pages[mat]["rows"]
        assert g["energy_MeV"].iloc[0] == 0.001 and g["energy_MeV"].iloc[-1] == 20.0
        _check_monotone(g.reset_index(drop=True), mat)
        assert (g["muen_rho_cm2_g"] > 0).all()
        assert (g["muen_rho_cm2_g"] <= g["mu_rho_cm2_g"]).all()
        below = g[g["edge_side"] == "below"].reset_index(drop=True)
        above = g[g["edge_side"] == "above"].reset_index(drop=True)
        assert len(below) == len(above)
        assert (above["mu_rho_cm2_g"] > below["mu_rho_cm2_g"]).all()
        assert (below["edge_label"] == above["edge_label"]).all()
    el = mu[mu["kind"] == "element"]
    assert sorted(el["Z"].unique().tolist()) == [13, 26, 74, 82]
    # K edge of lead, as tabulated
    pb_k = mu[(mu["material"] == "Pb") & (mu["edge_label"] == "K")]
    assert pb_k["energy_MeV"].tolist() == [0.0880045, 0.0880045]


def test_srd126_materials():
    mat = _csv(R.FIXTURE_MAT)
    assert len(mat) == 16 and mat["material"].is_unique
    for r in mat.itertuples():
        assert r.density_g_cm3 > 0
        w = [float(x.split(":")[1]) for x in r.composition.split(";")]
        assert abs(sum(w) - 1) < 2e-6, r.material


def test_air_table():
    if not R.AIR_PARQUET.exists():
        pytest.skip("air_muen.parquet not built")
    air = pd.read_parquet(R.AIR_PARQUET)
    assert list(air.columns) == ["energy_keV", "mu_rho", "muen_rho", "edge_label", "edge_side"]
    assert air["energy_keV"].iloc[0] == 1.0 and air["energy_keV"].iloc[-1] == 20000.0
    assert len(air) == 38
    assert (air["muen_rho"] <= air["mu_rho"]).all()
    e = air["energy_keV"].tolist()
    for i in range(1, len(e)):
        assert e[i] > e[i - 1] or (e[i] == e[i - 1] and air["edge_side"].iloc[i] == "above")
    cfg = config.get("response.air_muen")
    assert [r["energy_keV"] for r in cfg] == e
    assert [r["muen_rho"] for r in cfg] == air["muen_rho"].tolist()


def test_srd126_reparse_raw_pages():
    rec = _source("nist_srd126")
    mu = _csv(R.FIXTURE_MU)
    for key, page, *_ in R.SRD126_MATERIALS:
        p = R.SRD126_DIR / R._raw_name(page)
        if not p.exists():
            pytest.skip("SRD 126 raw pages not present")
        rows = R.parse_coefficient_page(p)["rows"]
        g = mu[mu["material"] == key]
        assert [r["mu_rho"] for r in rows] == g["mu_rho_cm2_g"].tolist()
        assert [r["muen_rho"] for r in rows] == g["muen_rho_cm2_g"].tolist()
    assert rec["licence"] == "https://www.nist.gov/open/license"


# ---------------------------------------------------------------- configuration

def test_config_std_refs_resolve():
    for sid in SOURCE_IDS:
        _source(sid)
    ours = {"radionuclides.Cs-137", "radionuclides.Co-60", "xray_tubes.qualities",
            "response.air_muen"}
    problems = [p for p in config.validate() if p.split(":")[0] in ours]
    assert problems == []
