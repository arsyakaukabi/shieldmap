"""Entry point: python -m shieldmap.analyze4b [--from STEP] [--only STEP ...]

Steps (in order): patches, table_v3, recheck, refcache, figures, tables. Steps whose module is absent are
skipped. The manuscript build (paper/build.py) runs after this and re-runs table_v3 with the number usage it
extracts from the manuscript.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

STEPS = {
    "patches": "shieldmap.analysis4b.patches",
    "table_v3": "shieldmap.analysis4b.table_v3",
    "recheck": "shieldmap.analysis4b.recheck",
    "refcache": "shieldmap.analysis4b.refcache",
    "figures": "shieldmap.figstyle.figures",
    "tables": "shieldmap.analysis4b.tables",
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=list(STEPS))
    ap.add_argument("--only", nargs="*", choices=list(STEPS))
    a = ap.parse_args(argv)
    names = list(STEPS)
    if a.start:
        names = names[names.index(a.start):]
    if a.only:
        names = [n for n in names if n in a.only]
    t0 = time.time()
    for n in names:
        try:
            mod = importlib.import_module(STEPS[n])
        except ModuleNotFoundError as e:
            if e.name and STEPS[n].startswith(e.name):
                print(f"[skip] {n} (module absent)", flush=True)
                continue
            raise
        t = time.time()
        print(f"[run] {n}", flush=True)
        mod.run()
        print(f"[done] {n} {time.time() - t:.0f}s", flush=True)
    print(json.dumps({"elapsed_s": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
