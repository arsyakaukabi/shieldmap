"""Prepare a fresh clone for REPRODUCE.md: restore the per-compound thickness profiles and the crustal abundances
(both verified by SHA-256), and create the local folder that the headline-table step writes its Markdown view to.

    python scripts/prepare.py
"""
from __future__ import annotations

import sys

from _common import ROOT

import restore_crustal_abundance
import restore_map


def main() -> int:
    rc = restore_map.main() or restore_crustal_abundance.main()
    for d in ("reports/phase4b", "reports/phase4c/figures"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("ok: ready" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
