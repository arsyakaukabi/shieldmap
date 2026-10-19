"""Restore data/phase2/profiles_thickness.npy (per-compound thickness profiles, 154,377 x 264 float32) from the
compressed parts shipped in data/phase2/ and verify it by SHA-256.

    python scripts/restore_map.py
"""
from __future__ import annotations

import lzma
import sys

import numpy as np

from _common import ROOT, sha256_bytes, sha256_file

META = {
 "file": "data/phase2/profiles_thickness.npy",
 "sha256": "147e9faa3917bd6fd12c171c2d97d2dae175dce768fc3028540e3dabaf727f96",
 "bytes": 163022240,
 "header_bytes": 128,
 "shuffle_width": 4,
 "codec": "lzma (xz container), preset 6",
 "parts": [
  {
   "name": "profiles_thickness.npy.xz.part01",
   "bytes": 45000000,
   "sha256": "0c908013e8c414cde7d77ca7b26e0feaf496d3951df9c70ac6ed247255dc8faa"
  },
  {
   "name": "profiles_thickness.npy.xz.part02",
   "bytes": 45000000,
   "sha256": "5fae8ebc0611401ff8565fb3e158199f7e3e9b61918eb6f306b53748559684cc"
  },
  {
   "name": "profiles_thickness.npy.xz.part03",
   "bytes": 815744,
   "sha256": "728934088d0b5ed146a40b0d47cf0b51fdfafae7f237fcfdaaac49985557fb87"
  }
 ]
}


def main() -> int:
    out = ROOT / META["file"]
    if out.exists() and sha256_file(out) == META["sha256"]:
        print(f"ok (present): {META['file']}")
        return 0
    comp = b""
    for p in META["parts"]:
        b = (ROOT / "data" / "phase2" / p["name"]).read_bytes()
        if sha256_bytes(b) != p["sha256"]:
            print(f"MISMATCH: {p['name']}", file=sys.stderr)
            return 1
        comp += b
    raw = lzma.decompress(comp)
    h, w = META["header_bytes"], META["shuffle_width"]
    body = np.frombuffer(raw, dtype=np.uint8, offset=h).reshape(w, -1).T.tobytes()
    data = raw[:h] + body
    if sha256_bytes(data) != META["sha256"]:
        print("MISMATCH: restored profiles_thickness.npy", file=sys.stderr)
        return 1
    out.write_bytes(data)
    print(f"restored {META['file']} ({len(data):,} bytes, SHA-256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
