"""Provenance checks (stage 1 manifest, stage 2 and stage 3 archive manifests)."""
from __future__ import annotations

import platform
from importlib import metadata

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common


def run() -> dict:
    paths.ensure(common.P4)
    p1 = c3.verify_phase1()
    p2 = c3.verify_phase2()
    p3 = common.verify_phase3()
    if p1["failures"] or p2["failures"] or p3["failures"]:
        raise SystemExit(f"BLOCKER: Phase 1-3 checksums fail: {p1['failures']} {p2['failures']} {p3['failures']}")
    pk = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib", "pymatgen", "xraylib", "spekpy", "mendeleev"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    out = {"phase1_manifest": p1, "phase2_archive_manifest": p2, "phase3_archive_manifest": p3,
           "python": platform.python_version(), "packages": pk}
    common.write_json(common.P4 / "step1_provenance.json", out)
    return out


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, list) else len(vv))
                                                               for kk, vv in v.items()}) for k, v in r.items()},
                     indent=1, default=str))
