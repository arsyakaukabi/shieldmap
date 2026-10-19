"""Regenerate every stage 3 output:  python -m shieldmap.analyze3 [--from STEP] [--only STEP ...]

Inputs: stage 1 tables (data/processed, data/interim), the tracked stage 2 outputs (data/phase2) and its
git-ignored caches (universe.parquet, profile arrays; rebuilt from Table B/C if missing, without touching
any tracked stage 2 file), the COD snapshot (data/raw/cod) and the cached MP structures (data/raw/mp).
No network access is needed when those raw files exist.

Order:
  setup          provenance checks (stage 1 manifest, stage 2 archive manifest)
  cod            COD filter, structure matching, class-resolved density bias
  fluorescence   K-fluorescence escape correction
  frontiers      frontier kinds, artefact flags, class-band tiers, textbook ambient phases,
                       top-N and artefact-free sensitivity, marginal prices, absolute increments
  continuum      price curves vs energy, GAP inversion, PB anatomy, Shapley, load-bearing
  density_model: data-conditioned density-error model, tiers v2, bootstrap bands
  edgewin: K-edge-window claims under the fluorescence correction
  landscape3     variance decomposition with physics overlay, envelope window, polymorphs
  reweight: crossover energy under population reweighting
  families       DIAG families, class-band tier tables, B5 oxides
  dossier        dossier tables, final figure checks
Narrative step reports (reports/phase3/step*.md) cite generated files; they are not rewritten.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

STEPS = ("setup", "cod", "fluorescence", "frontiers", "continuum", "density_model", "edgewin", "landscape3",
         "reweight", "families", "dossier")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--only", nargs="+", choices=STEPS)
    args = ap.parse_args(argv)
    steps = args.only or STEPS[STEPS.index(args.start):]
    t0 = time.time()
    from shieldmap.analysis3 import common
    rebuilt = common.ensure_phase2_caches()
    if rebuilt:
        print(f"[phase 2] rebuilt ignored caches from Table B/C: {rebuilt}", flush=True)
    for s in steps:
        t = time.time()
        print(f"[run] {s}", flush=True)
        importlib.import_module(f"shieldmap.analysis3.{s}").run()
        print(f"[done] {s} {time.time() - t:.0f}s", flush=True)
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
