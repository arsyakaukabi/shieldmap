"""Restore the NIST SRD 126 transcriptions that the repository does not redistribute (needs network access).

    python scripts/restore_nist_srd126.py

NIST Standard Reference Database 126 (Hubbell & Seltzer, doi:10.18434/T4D01F) may not be reproduced without
permission, so the repository ships config/sources/nist_srd126.yaml without its data.air block (the air table that
config/default.yaml response.air_muen reads) and without the test fixtures tests/fixtures/srd126_*.csv. This script
downloads the 18 official HTML tables into data/raw/srd126/, checks each against the content checksum recorded in
config/sources/nist_srd126.yaml, parses them with shieldmap.ingest.references (as python -m shieldmap.build does),
and writes the air block, the fixtures and data/interim/references/air_muen.parquet. The restored YAML and fixtures
are verified against the SHA-256 of the files used for the paper, so the configuration and every result that
uses the air table regenerate exactly. Needed only for the full chain and the SRD 126 tests.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from decimal import Decimal

import pandas as pd
import yaml

from _common import ROOT, sha256_bytes, sha256_file, splice_yaml

META = {
 "yaml": "config/sources/nist_srd126.yaml",
 "yaml_sha256": "e52783bcce26c7b5d66e54890b924b7d6380226d4aff36825773b3425ce16577",
 "fixtures": {
  "tests/fixtures/srd126_mu_rho.csv": "a15f879432165a24456bb0921b5d639a4f9cdf36a4983d4a8d0cd345a8dec119",
  "tests/fixtures/srd126_materials.csv": "8c400db3c175a75d33e85ed42a2d43426b41c17e05a539d23e93ae371421e4b1"
 }
}
UA = "shieldmap-restore/1.0 (+https://github.com/arsyakaukabi/shieldmap)"


def fetch_pages(rec: dict) -> None:
    from shieldmap.ingest import references as R
    for f in rec["files"]:
        dest = ROOT / f["path"]
        if dest.exists() and R.html_content_sha256(dest) == f["content_sha256"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        time.sleep(0.5)
        if R.html_content_sha256(dest) != f["content_sha256"]:
            raise SystemExit(f"MISMATCH: content of {f['url']} differs from the recorded table")
        print(f"fetched {f['path']}")


def main() -> int:
    from shieldmap.ingest import references as R
    rec = yaml.safe_load((ROOT / META["yaml"]).read_text(encoding="utf-8"))
    fetch_pages(rec)
    pages = {key: R.parse_coefficient_page(R.SRD126_DIR / R._raw_name(page)) for key, page, *_ in R.SRD126_MATERIALS}
    tab1 = R.parse_tab1(R.SRD126_DIR / "tab1.html")
    tab2 = R.parse_tab2(R.SRD126_DIR / "tab2.html")
    mu_rows, mat_rows = [], []
    for key, page, kind, z, formula, heading, tab2_name in R.SRD126_MATERIALS:
        pg = pages[key]
        R._check_coefficients(key, pg["rows"])
        for r in pg["rows"]:
            mu_rows.append({"material": key, "kind": kind, "Z": z, "formula": formula, "energy_MeV": r["energy_MeV"],
                            "mu_rho_cm2_g": r["mu_rho"], "muen_rho_cm2_g": r["muen_rho"],
                            "edge_label": r["edge_label"], "edge_side": r["edge_side"], "source_page": page})
        if kind == "element":
            t1 = tab1[z]
            mat_rows.append({"material": key, "name": t1["name"], "formula": formula,
                             "density_g_cm3": t1["density_g_cm3"], "composition": f"{z}:1", "source_page": "tab1.html"})
        else:
            t2 = tab2[tab2_name]
            mat_rows.append({"material": key, "name": tab2_name, "formula": formula,
                             "density_g_cm3": t2["density_g_cm3"], "composition": t2["composition_str"],
                             "source_page": "tab2.html"})
    mu = pd.DataFrame(mu_rows)
    mu["Z"] = mu["Z"].astype("Int64")
    fixtures = {"tests/fixtures/srd126_mu_rho.csv": mu.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                "tests/fixtures/srd126_materials.csv":
                    pd.DataFrame(mat_rows).to_csv(index=False, lineterminator="\n").encode("utf-8")}
    air = [{"energy_keV": float(Decimal(r["energy_MeV_str"]) * 1000), "mu_rho": r["mu_rho"], "muen_rho": r["muen_rho"],
            "edge_label": r["edge_label"], "edge_side": r["edge_side"]} for r in pages["air"]["rows"]]
    bad = 0
    for rel, b in fixtures.items():
        if sha256_bytes(b) != META["fixtures"][rel]:
            print(f"MISMATCH: {rel}", file=sys.stderr)
            bad += 1
            continue
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / rel).write_bytes(b)
        print(f"restored {rel} (SHA-256 verified)")
    y = ROOT / META["yaml"]
    if sha256_file(y) != META["yaml_sha256"]:
        data = splice_yaml(META["yaml"], {"air": air})
        if sha256_bytes(data) != META["yaml_sha256"]:
            print(f"MISMATCH: {META['yaml']}", file=sys.stderr)
            bad += 1
        else:
            y.write_bytes(data)
            print(f"restored {META['yaml']} (data.air, SHA-256 verified)")
    else:
        print(f"ok (present): {META['yaml']}")
    R.AIR_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(air).to_parquet(R.AIR_PARQUET, index=False)
    print(f"wrote {R.AIR_PARQUET.relative_to(ROOT).as_posix()}")
    if not bad:
        print("ok: NIST SRD 126 air table and fixtures match the files used for the paper")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
