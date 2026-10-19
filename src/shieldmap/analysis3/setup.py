"""Provenance checks and environment record (data/phase3/step1_provenance.json)."""
from __future__ import annotations

import platform
from importlib import metadata

from shieldmap import paths
from shieldmap.analysis3 import common


def run() -> dict:
    paths.ensure(common.P3)
    p1 = common.verify_phase1()
    p2 = common.verify_phase2()
    if p1["failures"] or p2["failures"]:
        raise SystemExit(f"BLOCKER: Phase 1/2 checksums fail: {p1['failures']} {p2['failures']}")
    pk = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib", "pymatgen", "pymatgen-core", "mp-api",
                 "xraylib", "spekpy"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    out = {"phase1_manifest": p1, "phase2_archive_manifest": p2,
           "python": platform.python_version(), "packages": pk}
    common.write_json(common.P3 / "step1_provenance.json", out)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str))
