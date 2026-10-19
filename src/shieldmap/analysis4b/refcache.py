"""Checksum manifest of the stage 4a reference caches.

data/raw/references/phase4a/ (ignored by git) holds the raw API responses the reference map is rebuilt from.
This writes data/phase4b/refcache_manifest.json (path, bytes, SHA-256 per file) and checks every file against
the entry for it in the verified stage 4a archive manifest (archives/phase4a-2026-10-13.manifest.json), so the
reference map rebuilds offline from that archive.
"""
from __future__ import annotations

import hashlib
import json

from shieldmap import paths
from shieldmap.analysis4b import patches

CACHE = paths.RAW / "references" / "phase4a"
ARCHIVE_MANIFEST = paths.REPO_ROOT / "archives" / "phase4a-2026-10-13.manifest.json"


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    files = sorted(p for p in CACHE.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    recs = [{"path": p.relative_to(paths.REPO_ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": _sha(p)} for p in files]
    arch = json.loads(ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    amap = {r["path"]: r["sha256"] for r in arch["files"]}
    missing = [r["path"] for r in recs if r["path"] not in amap]
    mismatch = [r["path"] for r in recs if r["path"] in amap and amap[r["path"]] != r["sha256"]]
    in_archive_only = sorted(p for p in amap if p.startswith("shieldmap/data/raw/references/phase4a/")
                             and p not in {r["path"] for r in recs})
    out = {"cache_dir": CACHE.relative_to(paths.REPO_ROOT).as_posix(), "file_count": len(recs),
           "total_bytes": sum(r["bytes"] for r in recs), "archive": arch["archive_name"] + ".zip",
           "archive_manifest": ARCHIVE_MANIFEST.relative_to(paths.REPO_ROOT).as_posix(),
           "all_in_archive_with_same_sha256": not missing and not mismatch and not in_archive_only,
           "missing_from_archive": missing, "sha256_mismatch": mismatch, "in_archive_only": in_archive_only,
           "restore": "unzip the Phase 4a archive at the repository root; the files land at the paths below",
           "files": recs}
    patches.common.write_json(patches.P4B / "refcache_manifest.json", out)
    return {k: v for k, v in out.items() if k != "files"}
