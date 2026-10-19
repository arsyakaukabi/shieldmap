"""List or download the raw upstream inputs that the repository does not redistribute, verified by checksum.

    python scripts/fetch_raw.py --list                 # every raw input: official URL, local path, checksum
    python scripts/fetch_raw.py [--only SOURCE_ID ...] [--large]

The records come from config/sources/*.yaml (the same records python -m shieldmap.build verifies). Files land at the
paths the pipeline expects (data/raw/...). The COD archive (18.5 GB) and the ENDF/B-VIII.0 library (3.4 GB) are
downloaded only with --large. NIST SRD 126 pages carry a per-request script token, so they are verified by their
content checksum (scripts removed, whitespace collapsed). Inputs without a public download URL are listed with the
way to obtain them: the Materials Project snapshot (API, MP_API_KEY in the environment; python -m shieldmap.build)
and the packages pinned in requirements.txt (mendeleev, pymatgen, SpekPy, xraylib).
"""
from __future__ import annotations

import argparse
import sys
import urllib.request

import yaml

from _common import ROOT, sha256_file

UA = "shieldmap-fetch-raw/1.0"
LARGE = {"cod_rev297631", "endfb_viii0_openmc"}


def records():
    for y in sorted((ROOT / "config" / "sources").glob("*.yaml")):
        rec = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        for f in rec.get("files") or []:
            yield rec.get("id", y.stem), f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--large", action="store_true")
    a = ap.parse_args()
    bad = 0
    for sid, f in records():
        if a.only and sid not in a.only:
            continue
        url, path = str(f.get("url") or ""), str(f.get("path") or "")
        check = f.get("content_sha256") or f.get("sha256") or ""
        direct = url.startswith("http") and path.startswith("data/")
        if a.list or not direct or (sid in LARGE and not a.large and not a.only):
            how = url or ("Materials Project API (MP_API_KEY), python -m shieldmap.build" if sid.startswith("m")
                          else "see config/sources/" + sid + ".yaml")
            print(f"{sid:22s} {path:50s} sha256={check[:16]}...  {how}")
            continue
        dst = ROOT / path
        from shieldmap.ingest.references import html_content_sha256
        verify = (lambda p: html_content_sha256(p)) if f.get("content_sha256") else sha256_file
        if dst.exists() and verify(dst) == check:
            print(f"ok (present)  {path}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading   {url}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        ok = verify(dst) == check
        bad += not ok
        print(f"{'ok' if ok else 'CHECKSUM DIFFERS':13s} {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
