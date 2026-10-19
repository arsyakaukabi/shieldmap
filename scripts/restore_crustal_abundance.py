"""Restore the per-element crustal abundances that the repository does not redistribute.

    python scripts/restore_crustal_abundance.py

The values (CRC Handbook of Chemistry and Physics, 95th ed., section 14, as shipped by the mendeleev package) have
unclear reuse terms, so the repository ships config/sources/crustal_abundance.yaml without its data.abundance_ppm
block and the two element tables without their crustal_abundance_ppm column. This script reads the values from
mendeleev 1.3.0 (pinned in requirements.txt), checks mendeleev's elements.db by SHA-256, and writes the block and
the column back. Each restored file is verified against the SHA-256 of the file used for the paper.
"""
from __future__ import annotations

import io
import sys

import pandas as pd

from _common import ROOT, sha256_bytes, sha256_file, splice_yaml

META = {
 "yaml": "config/sources/crustal_abundance.yaml",
 "yaml_sha256": "c82c8cf44743b88ab3d9d8dc35f66befefeddf1ba8d3cb14a1028f8f5e6a7a7f",
 "elements_db_sha256": "8a6312b57603e22c58ab28005bb8cacfbff721bc859eea8ddc2b784c17ffd717",
 "z_max": 100,
 "tables": {
  "data/processed/table_a_elements.parquet": {
   "position": 4,
   "dtype": "float64",
   "sha256": "bd7ffb7cb8640b8bdc55cea9aa00c309c9d099966d1dcfe5075ddedb53bd3ffb"
  },
  "data/interim/elements/elements.parquet": {
   "position": 4,
   "dtype": "float64",
   "sha256": "bd7ffb7cb8640b8bdc55cea9aa00c309c9d099966d1dcfe5075ddedb53bd3ffb"
  }
 }
}


def values() -> dict:
    import mendeleev
    from mendeleev.fetch import fetch_table

    db = sha256_file(__import__("pathlib").Path(mendeleev.__file__).parent / "elements.db")
    if db != META["elements_db_sha256"]:
        raise SystemExit(f"MISMATCH: mendeleev elements.db {db}; install mendeleev==1.3.0 (requirements.txt)")
    tab = fetch_table("elements")[["atomic_number", "symbol", "abundance_crust"]]
    tab = tab[tab.atomic_number <= META["z_max"]]
    return {str(r.symbol): (None if pd.isna(r.abundance_crust) else float(r.abundance_crust)) for r in tab.itertuples()}


def main() -> int:
    v = values()
    bad = 0
    y = ROOT / META["yaml"]
    if sha256_file(y) != META["yaml_sha256"]:
        data = splice_yaml(META["yaml"], {"abundance_ppm": v})
        if sha256_bytes(data) != META["yaml_sha256"]:
            print(f"MISMATCH: {META['yaml']}", file=sys.stderr)
            bad += 1
        else:
            y.write_bytes(data)
            print(f"restored {META['yaml']} (data.abundance_ppm, SHA-256 verified)")
    else:
        print(f"ok (present): {META['yaml']}")
    for rel, rec in META["tables"].items():
        p = ROOT / rel
        t = pd.read_parquet(p)
        if "crustal_abundance_ppm" in t.columns:
            print(f"ok (present): {rel}")
            continue
        t.insert(rec["position"], "crustal_abundance_ppm", t["symbol"].map(v).astype("float64"))
        buf = io.BytesIO()
        t.to_parquet(buf, index=False)
        b = buf.getvalue()
        if sha256_bytes(b) != rec["sha256"]:
            print(f"MISMATCH: {rel} (pyarrow/pandas versions other than requirements.txt?)", file=sys.stderr)
            bad += 1
            continue
        p.write_bytes(b)
        print(f"restored crustal_abundance_ppm in {rel} (SHA-256 verified)")
    if not bad:
        print("ok: crustal abundances match the values used for the paper")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
