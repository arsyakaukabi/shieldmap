"""Headline table v4 (data/phase4d/headline_table_v4.csv).

Every row of data/phase4b/headline_table_v3.csv unchanged, plus the stage 4d rows H401+ from data/phase4d/rows_*.csv
(rows_anchors.csv: anchor-crystal check of the transparency filter; rows_mc_c8.csv: Monte Carlo case c8). A new row's
quoted_in / manuscript_id are filled from paper/build/number_usage.json, which paper/check_numbers.py writes; the
v3 rows keep the placement analysis4b.table_v3 writes. paper/check_numbers.py reads v4 when it exists.

    python -m shieldmap.analysis4d.table_v4
"""
from __future__ import annotations

import json

import pandas as pd

from shieldmap import paths

P4B = paths.ROOT / "data" / "phase4b"
P4D = paths.ROOT / "data" / "phase4d"
V3 = P4B / "headline_table_v3.csv"
V4 = P4D / "headline_table_v4.csv"
USAGE = paths.ROOT / "paper" / "build" / "number_usage.json"


def _placement(usage: dict, rid: str) -> tuple[str, str]:
    u = usage.get(rid, {})
    if not u:
        return "not quoted", ""
    docs = [d for d in ("main", "supplementary") if u.get(d)]
    return "; ".join(docs), "; ".join(w for d in docs for w in u[d])


def run() -> dict:
    v3 = pd.read_csv(V3, dtype=str, keep_default_na=False)
    new = []
    for f in sorted(P4D.glob("rows_*.csv")):
        r = pd.read_csv(f, dtype=str, keep_default_na=False)
        r["placement_source"] = "paper/build/number_usage.json"
        r["v3_origin"] = r["v3_origin"].where(r["v3_origin"] != "", "phase4d") if "v3_origin" in r else "phase4d"
        new.append(r)
    rows = pd.concat(new, ignore_index=True) if new else pd.DataFrame(columns=v3.columns)
    dup = sorted(set(rows["id"]) & set(v3["id"])) + sorted(rows["id"][rows["id"].duplicated()].tolist())
    if dup:
        raise ValueError(f"Phase 4d row ids collide: {dup}")
    usage = json.loads(USAGE.read_text(encoding="utf-8")) if USAGE.exists() else {}
    for i, rid in rows["id"].items():
        rows.loc[i, ["quoted_in", "manuscript_id"]] = _placement(usage, rid)
    out = pd.concat([v3, rows.reindex(columns=v3.columns, fill_value="")], ignore_index=True)
    P4D.mkdir(parents=True, exist_ok=True)
    out.to_csv(V4, index=False, encoding="utf-8")
    return {"v3_rows": len(v3), "phase4d_rows": len(rows), "v4_rows": len(out),
            "phase4d_quoted": int((rows["quoted_in"] != "not quoted").sum()) if len(rows) else 0}


if __name__ == "__main__":
    print(run())
