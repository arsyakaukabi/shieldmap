"""One-time retrieval of reference datasets (bulk download or official API only) and source records.

Every retrieved file is stored under ``data/raw/<source>/`` and described by a YAML record in
``config/sources/`` (citation, URL, retrieval date, SHA-256). Re-running a build reuses the cached
file and verifies its checksum against the record; it downloads only when the file is missing.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import time
import urllib.request
from pathlib import Path

import yaml

from shieldmap import paths

USER_AGENT = "shieldmap/0.1 (open photon-shielding map; one-time reference retrieval)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def download(url: str, dest: Path, source_id: str | None = None,
             retries: int = 4, timeout: float = 120.0) -> dict:
    """Download ``url`` to ``dest`` unless it already exists. Returns a file record.

    If ``source_id`` names an existing source record that lists this file, the (cached or new)
    file must match the recorded SHA-256, otherwise a ``ValueError`` is raised: a changed upstream
    file is never accepted silently. The original retrieval timestamp is carried over.
    """
    rel = Path(dest).resolve().relative_to(paths.ROOT).as_posix()
    prior = _prior_file_record(source_id, rel) if source_id else None
    expected_sha256 = prior.get("sha256") if prior else None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    retrieved = None
    if not dest.exists():
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                retrieved = utc_now()
                break
            except Exception as e:  # network errors are retried, then re-raised
                last_err = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_err}")
    digest = sha256(dest)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"checksum mismatch for {dest}: {digest} != {expected_sha256}")
    rec = {"path": rel, "url": url,
           "sha256": digest, "bytes": dest.stat().st_size}
    if retrieved:
        rec["retrieved_utc"] = retrieved
    elif prior and prior.get("retrieved_utc"):
        rec["retrieved_utc"] = prior["retrieved_utc"]
    return rec


def source_path(source_id: str) -> Path:
    return paths.SOURCES / f"{source_id}.yaml"


def read_source(source_id: str) -> dict | None:
    p = source_path(source_id)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_source(source_id: str, record: dict) -> Path:
    """Write a source record. Required keys: title, citation, data_class, retrieval_method."""
    for key in ("title", "citation", "data_class", "retrieval_method"):
        if key not in record:
            raise KeyError(f"source record {source_id!r} lacks {key!r}")
    record = {"id": source_id, **record}
    p = source_path(source_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True, width=100)
    return p


def _prior_file_record(source_id: str, rel_path: str) -> dict | None:
    """File entry for ``rel_path`` in an existing source record, if any."""
    rec = read_source(source_id)
    if not rec:
        return None
    for f in rec.get("files", []):
        if f.get("path") == rel_path:
            return f
    return None
