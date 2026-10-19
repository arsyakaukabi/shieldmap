"""Regenerate every stage 2 output from Tables A/B/C:  python -m shieldmap.analyze [--skip-universe]

Order:
  universe and dense profile cache (verifies the stage 1 manifest first)
  A1 landscape, A2 prices, A3 candidates
  A4 sensitivity and the headline table
Narrative reports (reports/phase2/*.md) cite the generated tables and figures; they are not rewritten.
"""
from __future__ import annotations

import argparse
import json
import time

from shieldmap import paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-universe", action="store_true",
                    help="reuse data/phase2/universe.parquet and the profile cache")
    args = ap.parse_args(argv)
    t0 = time.time()
    from shieldmap.analysis import universe
    if not args.skip_universe or not (paths.DATA / "phase2" / "universe.parquet").exists():
        print("[step 0] universe", flush=True)
        universe.build()
    from shieldmap.analysis import candidates, landscape, prices, sensitivity
    for name, mod in (("A1 landscape", landscape), ("A2 prices", prices), ("A3 candidates", candidates),
                      ("A4 sensitivity", sensitivity)):
        print(f"[run] {name}", flush=True)
        mod.run()
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
