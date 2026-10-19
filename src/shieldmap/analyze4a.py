"""Regenerate every stage 4a output:  python -m shieldmap.analyze4a [--from STEP] [--only STEP ...]

Inputs: stages 1-3 (read only), the ENSDF and mendeleev data already used by stage 1, the cached reference API
responses under data/raw/references/phase4a/ (runs offline from them), and the saved OpenMC tallies
under data/phase4a/mc/ (comparison only; the transport runs themselves are on a Linux host).

Order:
  setup          provenance checks (stage 1 manifest, stage 2 and stage 3 archive manifests)
  refsets        frontiers and prices under the phase4a artefact flag; credible bands
  practical      natural-radioactivity flags, practical-set variants, abundance curve
  density4       density bands of the B4-internal radionuclide prices
  edge4          lead-free price at the Pb K-edge window against B4 (narrow and corrected)
  xs             xraylib cross-section sensitivity (optional module)
  mc             Monte Carlo comparison tables from saved tallies (optional module)
  references     reference map from cached API responses, offline (optional module)
  headline       headline table (csv and md)
  figures4       figure patches
  dossier4       results dossier v2
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

STEPS = ("setup", "refsets", "practical", "density4", "edge4", "xs", "mc", "references", "headline", "figures4",
         "dossier4")
OPTIONAL = {"xs": "shieldmap.analysis4.xs", "mc": "shieldmap.analysis4.mc.compare",
            "references": "shieldmap.analysis4.references"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args(argv)
    steps = args.only or STEPS[STEPS.index(args.start):]
    t0 = time.time()
    for s in steps:
        t = time.time()
        mod = OPTIONAL.get(s, f"shieldmap.analysis4.{s}")
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if s in OPTIONAL and e.name and mod.startswith(e.name):
                print(f"[skip] {s}: module {mod} not present", flush=True)
                continue
            raise
        print(f"[run] {s}", flush=True)
        m.run()
        print(f"[done] {s} {time.time() - t:.0f}s", flush=True)
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
