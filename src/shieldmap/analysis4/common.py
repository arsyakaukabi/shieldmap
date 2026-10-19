"""Shared stage 4a infrastructure: provenance checks, artefact flags, measured-density requirements.

Stages 1-3 are read only. This module builds on the stage 3 loaders (shieldmap.analysis3.common), which
re-export the stage 2 core; it never re-implements a stage 1-3 quantity.

Artefact definitions
  phase3     theoretical AND e_hull > 0.1 eV/atom (the conjunctive phase3 definition)
  phase4a    phase3 OR ρ_MP > 1.05 × 22.65 g/cm³ (the densest measured elemental solid, stage 1 Ir)
             OR (ICSD-tagged AND e_hull > 0.3 eV/atom)               PRIMARY
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common as c3

P4 = paths.DATA / "phase4a"
REPORTS4 = paths.ROOT / "reports" / "phase4a"
FIGURES4 = REPORTS4 / "figures"
P3_ARCHIVE_MANIFEST = paths.REPO_ROOT / "archives" / "phase3-2026-10-12.manifest.json"
P3_PREFIXES = ("shieldmap/data/phase3/", "shieldmap/src/shieldmap/analysis3/", "shieldmap/src/shieldmap/analyze3.py",
               "shieldmap/reports/phase3/", "shieldmap/config/sources/cod_rev297631.yaml",
               "shieldmap/config/sources/mp_structures_cod.yaml", "shieldmap/config/sources/xraylib.yaml")

RHO_MAX_MEASURED = 22.65        # g/cm³, stage 1 Ir: the densest measured elemental solid (Os/Ir)
RHO_FACTOR = 1.05               # artefact above RHO_FACTOR × RHO_MAX_MEASURED
ICSD_EHULL = 0.3                # eV/atom: ICSD-tagged entries further above the hull are artefacts
ARTEFACT_DEFS = ("phase4a", "phase3")
PRIMARY_ARTEFACT = "phase4a"

sha256 = c3.sha256
write_json = c3.write_json
universe = c3.universe


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------
def verify_archive_manifest(manifest_path, prefixes) -> dict:
    """Files under ``prefixes`` against a verified archive manifest (same EOL rule as stage 3)."""
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(man["files"]) + list(man["excluded_large_files"])
    bad, checked, eol = [], 0, []
    for f in entries:
        if not f["path"].startswith(prefixes) or "__pycache__" in f["path"]:
            continue
        p = paths.REPO_ROOT / f["path"]
        if not p.exists():
            bad.append((f["path"], "missing")); continue
        checked += 1
        if c3.sha256(p) == f["sha256"]:
            continue
        if p.suffix.lower() in c3.TEXT_SUFFIXES:
            if f["sha256"] in c3._text_digests(p) or c3._matches_git_blob(p, f["path"], man["source_commit"]):
                eol.append(f["path"]); continue
        bad.append((f["path"], "sha256 mismatch"))
    return {"checked": checked, "failures": bad, "matched_after_eol_normalisation": eol,
            "source_commit": man["source_commit"], "manifest": manifest_path.name}


def verify_phase3() -> dict:
    return verify_archive_manifest(P3_ARCHIVE_MANIFEST, P3_PREFIXES)


# ---------------------------------------------------------------------------------------------
# artefact flags
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def artefact_mask(definition: str = PRIMARY_ARTEFACT) -> np.ndarray:
    """True for artefact entries under ``definition`` (see module docstring)."""
    u = universe()
    theo = u["theoretical"].to_numpy(bool)
    eh = u["energy_above_hull"].to_numpy(float)
    conj = theo & (eh > c3.ARTEFACT_EHULL)
    if definition == "phase3":
        return conj
    if definition == "phase4a":
        dense = u["density"].to_numpy(float) > RHO_FACTOR * RHO_MAX_MEASURED
        icsd_far = ~theo & (eh > ICSD_EHULL)
        return conj | dense | icsd_far
    raise ValueError(definition)


def artefact_components() -> pd.DataFrame:
    """Per-row booleans of each disjunct of the phase4a artefact flag."""
    u = universe()
    theo = u["theoretical"].to_numpy(bool)
    eh = u["energy_above_hull"].to_numpy(float)
    return pd.DataFrame({"conjunctive": theo & (eh > c3.ARTEFACT_EHULL),
                         "dense": u["density"].to_numpy(float) > RHO_FACTOR * RHO_MAX_MEASURED,
                         "icsd_far": ~theo & (eh > ICSD_EHULL)})


def clean_mask(definition: str = PRIMARY_ARTEFACT) -> np.ndarray:
    return ~artefact_mask(definition)


def frontier(q: np.ndarray, mask: np.ndarray, kind: str = "best_clean",
             definition: str = PRIMARY_ARTEFACT) -> np.ndarray:
    """c3.frontier with the artefact-free kinds evaluated under ``definition``."""
    if kind in ("best_clean", "median50_clean"):
        return c3.core.frontier(q, np.asarray(mask, bool) & clean_mask(definition), kind.replace("_clean", ""))
    return c3.frontier(q, mask, kind)


def argbest_clean(q: np.ndarray, mask: np.ndarray, definition: str = PRIMARY_ARTEFACT) -> np.ndarray:
    return c3.argbest(q, np.asarray(mask, bool) & clean_mask(definition))
