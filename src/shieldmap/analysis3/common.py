"""Shared stage 3 infrastructure: provenance checks, loaders, frontiers, classes, figure style.

Every stage 3 module imports this. stage 1 (Tables A/B/C) and stage 2 (analysis/, data/phase2/) are
read only; this module re-exports the stage 2 core rather than re-implementing it.

Requirement conventions (as in stage 1/2)
  polyenergetic   Table C profiles via core.profiles(basis): thickness (cm) or areal mass (g/cm²)
  monoenergetic   thickness q = 1/μ (cm), areal q = 1/(μ/ρ) (g/cm²) at one grid energy; for a single
                  line the Table C requirement is ln(1/T)·q, so prices and ranks do not depend on T.

Frontier kinds
  best, 10th, median50   as stage 2 (core.frontier)
  best_icsd              best among ICSD-matched (theoretical == False) members of the set: EXP
                         applied to the baseline, reported as "what the physical world offers"
  best_clean             HEADLINE. Best member after removing artefact entries (theoretical and
                         e_hull > 0.1 eV/atom) from the set; applied to both sides of every price
  median50_clean         median of the top 50 of the artefact-free set (sensitivity)
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sps

from shieldmap import paths
from shieldmap.analysis import core
from shieldmap.analysis import universe as p2universe

P3 = paths.DATA / "phase3"
REPORTS3 = paths.ROOT / "reports" / "phase3"
FIGURES3 = REPORTS3 / "figures"
P2_ARCHIVE_MANIFEST = paths.REPO_ROOT / "archives" / "phase2-2026-10-12.manifest.json"

KINDS4 = ("best", "10th", "median50", "best_icsd")
KINDS6 = KINDS4 + ("best_clean", "median50_clean")
HEADLINE_KIND = "best_clean"  # one estimator for every cell and both bases
ARTEFACT_EHULL = 0.1          # eV/atom: a "best" frontier set by a theoretical entry above this is flagged

# re-exports (single source of truth stays in stage 2's core)
universe = core.universe
profiles = core.profiles
variants = core.variants
scenario_variants = core.scenario_variants
base_mask = core.base_mask
constraint_mask = core.constraint_mask
bundle_mask = core.bundle_mask
BUNDLES = core.BUNDLES
RULES = core.RULES
PRIMARY_RULE = core.PRIMARY_RULE
top_k = core.top_k
shapley = core.shapley


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_phase1() -> dict:
    """Stage 1 MANIFEST.json: SHA-256 of every Table A/B/C output."""
    return p2universe.verify_manifest()


P2_PREFIXES = ("shieldmap/data/phase2/", "shieldmap/src/shieldmap/analysis/", "shieldmap/src/shieldmap/analyze.py",
               "shieldmap/reports/phase2/")


TEXT_SUFFIXES = (".py", ".md", ".csv", ".json", ".log", ".txt", ".yaml", ".yml", ".toml", ".cfg")


def _text_digests(p) -> tuple[str, str]:
    """SHA-256 of a text file with LF and with CRLF line endings (git may check out either)."""
    lf = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()


def verify_phase2() -> dict:
    """Stage 2 artefacts against the verified stage 2 archive manifest (files and excluded large files).

    Binary files must match byte for byte. Text files match when their content matches with either
    line-ending convention: the manifest was taken from a Windows working tree, while a fresh checkout
    applies the repository's .gitattributes.
    """
    man = json.loads(P2_ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    entries = list(man["files"]) + list(man["excluded_large_files"])
    bad, checked, eol = [], 0, []
    for f in entries:
        if not f["path"].startswith(P2_PREFIXES) or "__pycache__" in f["path"]:
            continue
        p = paths.REPO_ROOT / f["path"]
        if not p.exists():
            bad.append((f["path"], "missing")); continue
        checked += 1
        if sha256(p) == f["sha256"]:
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            if f["sha256"] in _text_digests(p):
                eol.append(f["path"]); continue
            if _matches_git_blob(p, f["path"], man["source_commit"]):   # e.g. mixed line endings in a log
                eol.append(f["path"]); continue
        bad.append((f["path"], "sha256 mismatch"))
    return {"checked": checked, "failures": bad, "matched_after_eol_normalisation": eol,
            "source_commit": man["source_commit"]}


def _matches_git_blob(p, rel: str, commit: str) -> bool:
    """True when the file's LF-normalised content equals the committed blob at the manifest's commit."""
    import subprocess
    try:
        blob = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "show", f"{commit}:{rel}"],
                              capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return blob.replace(b"\r\n", b"\n") == p.read_bytes().replace(b"\r\n", b"\n")


def ensure_phase2_caches() -> list[str]:
    """Rebuild the git-ignored stage 2 caches (universe.parquet, profile arrays) if any is missing.

    Only the ignored caches are written; tracked stage 2 files are never touched. The caches
    are a reshape of stage 1 Table B/C and are verified against the stage 2 archive manifest in the setup step.
    """
    out = paths.DATA / "phase2"
    need = [n for n in ("universe.parquet", "profiles_thickness.npy", "profiles_areal.npy") if not (out / n).exists()]
    if not need:
        return []
    man = p2universe.verify_manifest()
    if man["failures"]:
        raise SystemExit(f"BLOCKER: Phase 1 outputs fail their manifest checksums: {man['failures']}")
    paths.ensure(out)
    u, _ = p2universe.build_universe()
    u.to_parquet(out / "universe.parquet", index=False)
    p2universe.build_profile_cache(len(u))
    return need


# ---------------------------------------------------------------------------------------------
# loaders (all row-aligned with Table B / Table C / universe)
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def energy_keV() -> np.ndarray:
    return np.load(paths.PROCESSED / "energy_keV.npy")


@lru_cache(maxsize=2)
def mu_rho(coherent: bool = True) -> np.ndarray:
    """(N × 545) float32 μ/ρ (cm²/g), memory-mapped."""
    return np.load(paths.PROCESSED / ("mu_rho_wcoh.npy" if coherent else "mu_rho_wocoh.npy"), mmap_mode="r")


@lru_cache(maxsize=1)
def density() -> np.ndarray:
    return universe()["density"].to_numpy(dtype=np.float64)


def energy_index(e_keV: float) -> int:
    e = energy_keV()
    i = int(np.argmin(np.abs(e - e_keV)))
    if abs(e[i] - e_keV) > 1e-6 * max(1.0, e_keV):
        raise ValueError(f"{e_keV} keV is not a grid node")
    return i


def mono_requirement(basis: str, coherent: bool = True, rows=None, e_idx=None) -> np.ndarray:
    """Monoenergetic requirement (rows × energies): thickness 1/μ (cm) or areal 1/(μ/ρ) (g/cm²)."""
    m = mu_rho(coherent)
    r = slice(None) if rows is None else rows
    c = slice(None) if e_idx is None else e_idx
    sub = np.asarray(m[r][:, c] if rows is not None else m[:, c], dtype=np.float64)
    if basis == "areal":
        return 1.0 / sub
    if basis == "thickness":
        d = density() if rows is None else density()[rows]
        return 1.0 / (sub * d[:, None])
    raise ValueError(basis)


@lru_cache(maxsize=1)
def mass_fractions() -> sps.csr_matrix:
    """(N × 100) mass fractions w_Z (column Z−1), row-aligned with Table B (order checked)."""
    w = sps.load_npz(paths.INTERIM / "mp" / "mass_fractions.npz").tocsr()
    ix = pd.read_parquet(paths.INTERIM / "mp" / "mass_fractions_index.parquet")
    if not (ix["material_id"].to_numpy() == universe()["material_id"].to_numpy()).all():
        raise RuntimeError("mass-fraction rows are not aligned with the universe")
    return w


@lru_cache(maxsize=1)
def element_xsec() -> dict:
    """Element partial μ/ρ on the 545-node grid (stage 1 interim; 100 × 545 each)."""
    with np.load(paths.INTERIM / "elements" / "xsec_grid.npz") as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=1)
def elements_table() -> pd.DataFrame:
    return pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet").sort_values("Z").reset_index(drop=True)


@lru_cache(maxsize=1)
def edges() -> pd.DataFrame:
    return pd.read_parquet(paths.INTERIM / "elements" / "edges.parquet")


@lru_cache(maxsize=16)
def spectrum(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(quadrature energies keV, fluence) of a stage 1 spectrum (spectrum_bins.parquet)."""
    sb = pd.read_parquet(paths.PROCESSED / "spectrum_bins.parquet")
    s = sb[sb.spectrum == name]
    if s.empty:
        raise KeyError(name)
    return s["energy_keV"].to_numpy(dtype=float), s["fluence"].to_numpy(dtype=float)


def spectrum_weights(name: str, response: str = "air_kerma") -> tuple[np.ndarray, np.ndarray]:
    from shieldmap.physics import response as resp
    e, phi = spectrum(name)
    return e, phi * resp.response(response, e)


def row_of(material_id: str) -> int:
    u = universe()
    r = np.flatnonzero(u["material_id"].to_numpy() == material_id)
    if r.size != 1:
        raise KeyError(material_id)
    return int(r[0])


# ---------------------------------------------------------------------------------------------
# frontiers (four kinds) and artefact flag
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def icsd_mask() -> np.ndarray:
    return ~universe()["theoretical"].to_numpy(dtype=bool)


@lru_cache(maxsize=1)
def clean_mask() -> np.ndarray:
    """True for every universe row that is not an artefact entry (see artefact_flag)."""
    return ~artefact_flag(np.arange(len(universe())))


def frontier(q: np.ndarray, mask: np.ndarray, kind: str = "best") -> np.ndarray:
    """core.frontier plus kinds 'best_icsd', 'best_clean' and 'median50_clean'."""
    if kind == "best_icsd":
        return core.frontier(q, np.asarray(mask, bool) & icsd_mask(), "best")
    if kind in ("best_clean", "median50_clean"):
        return core.frontier(q, np.asarray(mask, bool) & clean_mask(), kind.replace("_clean", ""))
    return core.frontier(q, mask, kind)


def price(q: np.ndarray, base: np.ndarray, constraint: np.ndarray, kind: str = "best") -> np.ndarray:
    """P = q*(F0 ∩ K)/q*(F0) − 1 for any frontier kind (both sides use the same kind)."""
    return frontier(q, base & constraint, kind) / frontier(q, base, kind) - 1.0


def argbest(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row index of the best member of mask for each column of q (n × V) or (n,)."""
    q2 = np.asarray(q, dtype=np.float64)
    one = q2.ndim == 1
    if one:
        q2 = q2[:, None]
    idx = np.flatnonzero(mask)
    sub = np.where(np.isfinite(q2[idx]), q2[idx], np.inf)
    out = idx[np.argmin(sub, axis=0)]
    return out[0] if one else out


def artefact_flag(rows) -> np.ndarray:
    """True where the row is a theoretical entry with e_hull > ARTEFACT_EHULL."""
    u = universe()
    r = np.asarray(rows)
    return (u["theoretical"].to_numpy(bool)[r] & (u["energy_above_hull"].to_numpy(float)[r] > ARTEFACT_EHULL))


def legacy_rule_kind(best_rows) -> np.ndarray:
    """The original frontiers-module per-cell rule (superseded and kept as a sensitivity):
    'median50' where the best-compound frontier is an artefact entry, else 'best'."""
    return np.where(artefact_flag(best_rows), "median50", "best")


# ---------------------------------------------------------------------------------------------
# compound classes (shared by the COD module density bias, the frontiers module tiers and the continuum-module bootstrap)
# ---------------------------------------------------------------------------------------------
LANTHANIDES = frozenset("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
ACTINIDES = frozenset("Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
MAGNETIC_3D = frozenset("V Cr Mn Fe Co Ni".split())
HALOGENS = frozenset("F Cl Br I".split())
CHALCOGENS = frozenset("S Se Te".split())
NONMETALS = frozenset("H B C N O F Si P S Cl As Se Br Te I He Ne Ar Kr Xe Rn".split())

# overlapping reporting classes (a compound can be in several)
REPORT_CLASSES = ("oxide", "halide", "chalcogenide", "intermetallic", "lanthanide", "3d_magnetic", "f_electron")
# exclusive bootstrap class: first match in this precedence order
PRIMARY_CLASS_ORDER = ("f_electron", "lanthanide", "3d_magnetic", "halide", "oxide", "chalcogenide",
                       "intermetallic", "other")


def report_classes(elements) -> set[str]:
    """Overlapping classes of one compound from its element list.

    oxide         contains O
    halide        contains F, Cl, Br or I
    chalcogenide  contains S, Se or Te
    intermetallic contains no non-metal or metalloid (H B C N O F Si P S Cl As Se Br Te I, noble gases)
    lanthanide    contains La–Lu
    3d_magnetic   contains V, Cr, Mn, Fe, Co or Ni
    f_electron    contains an actinide (5f; lanthanides are their own class)
    """
    s = set(elements)
    out = set()
    if "O" in s:
        out.add("oxide")
    if s & HALOGENS:
        out.add("halide")
    if s & CHALCOGENS:
        out.add("chalcogenide")
    if not (s & NONMETALS):
        out.add("intermetallic")
    if s & LANTHANIDES:
        out.add("lanthanide")
    if s & MAGNETIC_3D:
        out.add("3d_magnetic")
    if s & ACTINIDES:
        out.add("f_electron")
    return out


def primary_class(elements) -> str:
    """Exclusive class (precedence f_electron > lanthanide > 3d_magnetic > halide > oxide > chalcogenide >
    intermetallic > other)."""
    rc = report_classes(elements)
    for c in PRIMARY_CLASS_ORDER:
        if c in rc:
            return c
    return "other"


@lru_cache(maxsize=1)
def universe_primary_class() -> np.ndarray:
    return np.array([primary_class(e) for e in universe()["elements"]], dtype=object)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.2


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0,
                         "axes.linewidth": 0.6, "savefig.dpi": 300, "figure.dpi": 100,
                         "font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": SVG_HASHSALT})
    return plt


SVG_HASHSALT = "shieldmap-phase3"       # fixed SVG element ids, so regenerated SVGs are byte-stable


def savefig(fig, name: str) -> list[str]:
    """Write figures/<name>.png (300 dpi) and .svg; returns the paths relative to the repo root.

    The SVG carries no date and uses a fixed id salt, so a rerun reproduces it byte for byte.
    """
    import matplotlib
    paths.ensure(FIGURES3)
    out = []
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT}):
        for ext in ("png", "svg"):
            p = FIGURES3 / f"{name}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
            out.append(str(p.relative_to(paths.ROOT)).replace("\\", "/"))
    return out


def write_json(path, obj) -> None:
    paths.ensure(path.parent)
    path.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
