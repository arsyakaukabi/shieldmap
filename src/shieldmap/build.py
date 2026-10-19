"""Regenerate Tables A, B, C and the manifest:  python -m shieldmap.build [--refresh-mp] [--limit N]

Order: references (ENSDF lines, RQR, SRD 126 air; they fix the grid's key lines) → Table A
(elements, XCOM on the grid) → Table B (Materials Project snapshot; cached unless --refresh-mp)
→ Table C → MANIFEST.json. Raw retrievals are cached in data/raw/ and verified against the SHA-256
recorded in config/sources/.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from importlib.metadata import version

import pandas as pd

from shieldmap import config, paths
from shieldmap.ingest import fetch

PACKAGES = ["numpy", "scipy", "pandas", "pyarrow", "pymatgen", "mp-api", "emmet-core", "spekpy",
            "pyyaml", "pypdf"]


def _sha_listing(root, patterns=("*",)) -> list[dict]:
    out = []
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if p.is_file() and p.name != "MANIFEST.json":
                out.append({"path": p.relative_to(paths.ROOT).as_posix(), "bytes": p.stat().st_size,
                            "sha256": fetch.sha256(p)})
    return out


def _rows(p) -> int | None:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return None


def write_manifest(stats: dict, started: str, config_sha: str) -> dict:
    sources = {}
    for p in sorted(paths.SOURCES.glob("*.yaml")):
        rec = fetch.read_source(p.stem)
        sources[p.stem] = {k: rec.get(k) for k in ("title", "data_class", "retrieved_utc", "database_version",
                                                   "url", "files", "package") if rec.get(k) is not None}
        sources[p.stem]["record_sha256"] = fetch.sha256(p)
    outputs = _sha_listing(paths.PROCESSED) + _sha_listing(paths.INTERIM)
    for o in outputs:
        if o["path"].endswith(".parquet"):
            o["rows"] = _rows(paths.ROOT / o["path"])
    man = {
        "project": "shieldmap", "phase": 1,
        "build_started_utc": started, "build_finished_utc": fetch.utc_now(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {p: version(p) for p in PACKAGES},
        "config": {"path": "config/default.yaml", "sha256_at_start": config_sha,
                   "sha256_at_end": config.config_hash()},
        "sources": sources,
        "stats": stats,
        "outputs": outputs,
    }
    (paths.PROCESSED / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    return man


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-mp", action="store_true", help="re-query Materials Project")
    ap.add_argument("--limit", type=int, default=None, help="Table C for the first N compounds only (debug)")
    args = ap.parse_args(argv)
    started = fetch.utc_now()
    config_sha = config.config_hash()
    t0 = time.time()

    from shieldmap.ingest import references
    print("[1/5] reference data (ENSDF, RQR, SRD 126)", flush=True)
    references.build()
    config.raw.cache_clear()
    problems = config.validate()
    if problems:
        raise SystemExit("config policy violations:\n  " + "\n  ".join(problems))

    from shieldmap.ingest import elements
    print("[2/5] Table A (elements, XCOM on the grid)", flush=True)
    elements.build()

    from shieldmap.ingest import mp
    print("[3/5] Table B (Materials Project)", flush=True)
    mp.build(refresh=args.refresh_mp)

    from shieldmap.physics import tablec
    print("[4/5] Table C", flush=True)
    stats = tablec.build(limit=args.limit)

    # final copies of Tables A and B next to Table C
    paths.ensure(paths.PROCESSED)
    shutil.copy2(paths.INTERIM / "elements" / "elements.parquet", paths.PROCESSED / "table_a_elements.parquet")
    shutil.copy2(paths.INTERIM / "mp" / "compounds.parquet", paths.PROCESSED / "table_b_compounds.parquet")
    stats["n_elements"] = len(pd.read_parquet(paths.PROCESSED / "table_a_elements.parquet"))
    stats["limit"] = args.limit
    stats["elapsed_s"] = round(time.time() - t0, 1)

    print("[5/5] manifest", flush=True)
    if config.config_hash() != config_sha:
        print("WARNING: config/default.yaml changed during the build", flush=True)
    write_manifest(stats, started, config_sha)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
