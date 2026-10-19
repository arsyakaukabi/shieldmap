"""Compound density bias of Materials Project (MP) against the Crystallography Open Database.

Purpose (single, restricted use of COD)
    COD (CC0) is used only to estimate the DFT-vs-experiment density bias on compounds,
    δ = ρ_MP/ρ_COD − 1 (same sign convention as stage 1's elemental rel_bias = ρ_MP/ρ_exp − 1).
    COD densities never replace MP densities in the map. They only calibrate the uncertainty band
    and the class-resolved δ distribution used by the continuum-module bootstrap.

Pipeline (``run()`` is idempotent; every stage is cached and needs no network once its raw input exists)
    download()              one bulk retrieval of the COD release archive by resumable HTTP range
                            requests into 128 MiB parts, which are concatenated and checked against
                            the published MD5. Writes data/raw/cod/SOURCE.json.
    filter_archive()        streams the .txz once (tarfile 'r|xz', never extracted). Steps:
                            (a) text pre-filter: inorganic (formula without both C and H), and the
                                reduced formula of _chemical_formula_sum must be in the set of reduced
                                formulas of the eligible universe. This is a tractability pre-filter;
                                the actual matching is structural;
                            (b) quality filters on the parsed CIF block: no theoretical method,
                                no COD error/retraction, all occupancies exactly 1, no disorder
                                assembly/group, T ≤ 350 K or unspecified, P ≤ 1 MPa or unspecified
                                with no diamond-anvil environment, R ≤ 0.1 where given.
                            Survivors (CIF text plus metadata) are cached in
                            data/interim/cod/cod_filtered.parquet, and the per-entry funnel in
                            screen_log.parquet.
    parse_cod()             pymatgen CifParser(primitive=False) → ordered structure. Requirements:
                            the composition must equal _chemical_formula_sum, and ρ must agree with
                            _exptl_crystal_density_diffrn within 1 % where that is given. Output:
                            data/interim/cod/cod_structures.parquet.
    fetch_mp_structures()   one official-API retrieval (mp-api) of the final structures of the
                            eligible MP entries whose reduced formula occurs among the parsed COD
                            entries. The density of every structure is checked against Table B
                            (|Δρ/ρ| ≤ 1e-6). Output: data/raw/mp/structures_cod_<db>.jsonl.gz
                            plus a .meta.json.
    match()                 pymatgen StructureMatcher with default tolerances, run per reduced
                            formula. Selection rule: one COD value per MP entry (see select_cod()).
    bias_tables()           overall statistics, by class, by e_hull bin, plus sensitivities and the
                            figure. Outputs go to data/phase3/D/.

Public API for downstream modules (the continuum module bootstrap)
    class_delta_distribution(kind="compound")   dict primary_class → δ samples (np.ndarray)
    band(cls)                                   (p05, p95) of δ for a primary class or "pooled"

CLI: ``python -m shieldmap.analysis3.cod [download|filter|parse|fetch|match|tables|all]``.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import threading
import time
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis3 import common

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# snapshot constants
# ---------------------------------------------------------------------------------------------
COD_URL = "https://www.crystallography.net/archives/2025/data/cod-rev297631-2025.02.07.txz"
COD_MD5_LIST_URL = "https://www.crystallography.net/archives/md5sum.dat"
COD_RELEASE_DATE = "2025-02-07"
COD_SVN_REVISION = 297631
COD_SIZE = 18_481_345_940
COD_MD5 = "10db38860747e2f8eeadd8e61d132d05"
COD_SHA256 = "569fae5dc969ed688aa62950d481be71fa6195c39a08b550085c14d169d8525c"   # config/sources/cod_rev297631.yaml
MP_STRUCT_SHA256 = {"2026.04.13": "60db58cfe151f5f91a863608a941d4d2daa74ac70167fc18cda9794a47696317"}  # mp_structures_cod.yaml
PART_SIZE = 128 * 1024 * 1024

RAW = paths.RAW / "cod"
PARTS = RAW / "parts"
ARCHIVE = RAW / "cod-rev297631-2025.02.07.txz"
SOURCE_JSON = RAW / "SOURCE.json"
INTERIM = paths.INTERIM / "cod"
FILTERED_CACHE = INTERIM / "cod_filtered.parquet"
SCREEN_LOG = INTERIM / "screen_log.parquet"
STRUCT_CACHE = INTERIM / "cod_structures.parquet"
MATCH_CACHE = INTERIM / "pair_matches.parquet"
MP_RAW = paths.RAW / "mp"
OUT = common.P3 / "D"

# ---------------------------------------------------------------------------------------------
# filter thresholds
# ---------------------------------------------------------------------------------------------
T_MAX_K = 350.0                 # keep T ≤ 350 K or unspecified
P_MAX_KPA = 1000.0              # keep P ≤ 1 MPa (CIF pressures are in kPa) or unspecified
R_MAX = 0.10                    # keep R ≤ 0.1 where given
OCC_TOL = 1e-9                  # occupancy must equal 1 (no partial occupancy; values ≠ 1 are dropped)
RHO_CROSSCHECK_TOL = 0.01       # |ρ_parsed/ρ_diffrn − 1| ≤ 1 % where _exptl_crystal_density_diffrn is given
MP_DENSITY_TOL = 1e-6           # |ρ(MP structure)/ρ(Table B) − 1| ≤ 1e-6
MIN_CLASS_N = 30                # class distributions with fewer matches fall back to the pooled one
GROSS_DELTA = 0.5               # Stage 3 original screen |δ| > 0.5 (asymmetric in log space); kept only as a
                                # sensitivity (kind="compound_abs50"), superseded by the robust screen below
SCREEN_K = 5.0                  # robust screen (the hardening revision): a pair is excluded from kind="compound" when
                                # |ln(1+δ) − median_c| > SCREEN_K · 1.4826·MAD_c within its primary class c
SCREEN_K_SENS = (8.0,)          # wider screen reported as a sensitivity (kind="compound_k8")
ELEMENT_XCHECK_TOL = 0.05       # unary COD match vs stage 1 experimental density: |ρ_COD/ρ_exp − 1| > 5 % marks
                                # the COD value as a data error when the MP entry itself is the ambient phase,
ELEMENT_PHASE_TOL = 0.10        # i.e. when |ρ_MP/ρ_exp − 1| ≤ 10 % (an MP polymorph far from the measured
                                # density, e.g. alpha-Sn against white tin, is not cross-checked)
N_WORKERS = 10
EHULL_BINS = (("0", None, 0.0), ("(0,0.025]", 0.0, 0.025), ("(0.025,0.05]", 0.025, 0.05),
              ("(0.05,0.1]", 0.05, 0.1), (">0.1", 0.1, None))
BOOT_SEED = 20261012
BOOT_N = 2000

TEMP_TAGS = ("_diffrn_ambient_temperature", "_cell_measurement_temperature")
PRESSURE_KPA_TAGS = ("_diffrn_ambient_pressure", "_cell_measurement_pressure", "_diffrn_ambient_pressure_gt")
PRESSURE_GPA_TAGS = ("_diffrn_ambient_pressure_gpa", "_cell_measurement_pressure_gpa")   # non-standard COD tags
R_TAGS = ("_refine_ls_r_factor_gt", "_refine_ls_r_factor_obs", "_refine_ls_r_factor_all", "_refine_ls_r_i_factor")
METHOD_TAGS = ("_cod_struct_determination_method", "_cod_structure_determination_method")
DISORDER_TAGS = ("_atom_site_disorder_assembly", "_atom_site_disorder_group")
TITLE_HP_RE = re.compile(r"high[- ]?pressure|\bGPa\b|\bkbar\b|compress|diamond[- ]anvil", re.I)
TITLE_HT_RE = re.compile(r"high[- ]?temperature", re.I)

# funnel stage names, in the order they are applied
STAGES = ("no_formula_sum", "formula_unparseable", "organic", "not_in_mp_formula_set", "cif_unreadable",
          "theoretical_method", "cod_error_or_retraction", "occupancy_not_unity", "disorder_flag",
          "temperature_gt_350K", "high_pressure", "r_factor_gt_0.1", "kept")
PARSE_STAGES = ("parse_failed", "not_one_structure", "parsed_disordered", "nonelement_species", "composition_mismatch",
                "density_crosscheck_fail", "parsed_ok")


# =============================================================================================
# 1. download
# =============================================================================================
def _part_len(k: int, total: int = COD_SIZE, chunk: int = PART_SIZE) -> int:
    return min(chunk, total - k * chunk)


def n_parts(total: int = COD_SIZE, chunk: int = PART_SIZE) -> int:
    return (total + chunk - 1) // chunk


def _download_parts(n_workers: int = 8) -> None:
    """Resumable parallel HTTP range download into PARTS/part_kkkk (part k = bytes [k·2^27, (k+1)·2^27))."""
    import requests

    PARTS.mkdir(parents=True, exist_ok=True)
    todo = list(range(n_parts()))
    lock = threading.Lock()

    def work():
        s = requests.Session()
        while True:
            with lock:
                if not todo:
                    return
                k = todo.pop(0)
            p = PARTS / f"part_{k:04d}"
            L = _part_len(k)
            fails = 0
            while True:
                have = p.stat().st_size if p.exists() else 0
                if have >= L:
                    break
                try:
                    r = s.get(COD_URL, headers={"Range": f"bytes={k * PART_SIZE + have}-{k * PART_SIZE + L - 1}"},
                              stream=True, timeout=60)
                    if r.status_code != 206:
                        fails += 1
                        time.sleep(3)
                    else:
                        with open(p, "ab") as f:
                            for b in r.iter_content(1 << 20):
                                f.write(b)
                except Exception:          # the server resets long connections: resume from the current size
                    fails += 1
                    time.sleep(1)
                    s = requests.Session()
                if fails > 500:
                    raise RuntimeError(f"part {k}: too many failed range requests")
            log.info("part %d complete", k)

    ts = [threading.Thread(target=work) for _ in range(n_workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def parts_complete() -> bool:
    return all((PARTS / f"part_{k:04d}").exists() and (PARTS / f"part_{k:04d}").stat().st_size == _part_len(k)
               for k in range(n_parts()))


def assemble() -> dict:
    """Verify part sizes and the published MD5 over the parts, then concatenate them in order.

    Disk-frugal: the MD5/SHA-256 are computed over the parts first, then each part is appended to
    the archive and deleted, so the peak footprint is about one archive. The assembled file is
    re-checked for size; its SHA-256 equals the digest of the concatenated parts.
    """
    sizes = {k: (PARTS / f"part_{k:04d}").stat().st_size if (PARTS / f"part_{k:04d}").exists() else -1
             for k in range(n_parts())}
    bad = {k: s for k, s in sizes.items() if s != _part_len(k)}
    if bad:
        raise RuntimeError(f"missing or wrong-size parts: {bad}")
    mtimes = [(PARTS / f"part_{k:04d}").stat().st_mtime for k in range(n_parts())]
    md5, sha = hashlib.md5(), hashlib.sha256()
    for k in range(n_parts()):
        with open(PARTS / f"part_{k:04d}", "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                md5.update(b)
                sha.update(b)
    if md5.hexdigest() != COD_MD5:
        raise RuntimeError(f"MD5 mismatch over parts: {md5.hexdigest()} != published {COD_MD5}")
    tmp = ARCHIVE.with_suffix(".txz.tmp")
    with open(tmp, "wb") as out:
        for k in range(n_parts()):
            with open(PARTS / f"part_{k:04d}", "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    out.write(b)
            out.flush()
            os.fsync(out.fileno())
            (PARTS / f"part_{k:04d}").unlink()
    if tmp.stat().st_size != COD_SIZE:
        raise RuntimeError(f"assembled size {tmp.stat().st_size} != {COD_SIZE}")
    tmp.replace(ARCHIVE)
    try:
        PARTS.rmdir()
    except OSError:
        pass
    rec = {
        "source": "Crystallography Open Database (COD)",
        "license": "CC0 1.0 (COD data); used only to estimate the MP density bias (handover Step 3)",
        "url": COD_URL,
        "md5_list_url": COD_MD5_LIST_URL,
        "release_date": COD_RELEASE_DATE,
        "svn_revision": COD_SVN_REVISION,
        "file": ARCHIVE.name,
        "size_bytes": COD_SIZE,
        "md5_published": COD_MD5,
        "md5": md5.hexdigest(),
        "sha256": sha.hexdigest(),
        "retrieval_first_part_finished_utc": _utc(min(mtimes)),
        "retrieval_last_part_finished_utc": _utc(max(mtimes)),
        "assembled_utc": _utc(time.time()),
        "method": ("one bulk retrieval by resumable parallel HTTP range requests (8 connections, 128 MiB parts; "
                   "the server resets long connections), MD5 verified over the ordered parts against the published "
                   "value, parts concatenated in order; no per-entry retrieval"),
        "note": "archive listed in md5sum.dat with the same ETag as cod-cifs-mysql.txz (DECISIONS P3-D-004)",
    }
    common.write_json(SOURCE_JSON, rec)
    return rec


def _utc(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).replace(microsecond=0).isoformat()


_VERIFIED: dict = {}


def verify_archive() -> dict:
    """Re-hash the local archive (SHA-256, once per process) against COD_SHA256 (review F-3).

    The recorded SOURCE.json fields are not trusted on their own: the bytes are hashed again. Returns a
    provenance record; raises if the archive is present but does not match.
    """
    if "archive" in _VERIFIED:
        return _VERIFIED["archive"]
    if not ARCHIVE.exists():
        return {"archive_present": False, "verified": False}
    if ARCHIVE.stat().st_size != COD_SIZE:
        raise RuntimeError(f"{ARCHIVE.name}: size {ARCHIVE.stat().st_size} != {COD_SIZE}")
    t0 = time.time()
    h = common.sha256(ARCHIVE)
    if h != COD_SHA256:
        raise RuntimeError(f"{ARCHIVE.name}: SHA-256 {h} != expected {COD_SHA256}")
    _VERIFIED["archive"] = {"archive_present": True, "verified": True, "sha256": h,
                            "_seconds": round(time.time() - t0, 1)}
    return _VERIFIED["archive"]


def download(n_workers: int = 8) -> Path:
    """Return the verified local archive, retrieving it once (resumable) if it is not present."""
    if ARCHIVE.exists() and ARCHIVE.stat().st_size == COD_SIZE and SOURCE_JSON.exists():
        rec = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
        if rec.get("md5") != COD_MD5:
            raise RuntimeError("SOURCE.json MD5 does not match the published MD5")
        verify_archive()
        return ARCHIVE
    if not parts_complete():
        _download_parts(n_workers)
    assemble()
    return ARCHIVE


# =============================================================================================
# 2. stream + filter
# =============================================================================================
class _PartsStream(io.RawIOBase):
    """Read the not-yet-assembled parts as one stream (optionally waiting for parts still downloading)."""

    def __init__(self, wait: bool = True, stall_s: float = 6 * 3600.0):
        self.k, self.f, self.wait, self.stall_s = 0, None, wait, stall_s
        self._open(0)

    def _open(self, k):
        p = PARTS / f"part_{k:04d}"
        last, t0 = -1, time.time()
        while True:
            size = p.stat().st_size if p.exists() else 0
            if size == _part_len(k):
                break
            if not self.wait:
                raise RuntimeError(f"part {k} incomplete")
            if size != last:
                last, t0 = size, time.time()
            elif time.time() - t0 > self.stall_s:
                raise RuntimeError(f"part {k} stalled")
            time.sleep(10)
        self.f = open(p, "rb")

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            d = self.f.read(len(b))
            if d:
                b[:len(d)] = d
                return len(d)
            self.f.close()
            self.k += 1
            if self.k >= n_parts():
                return 0
            self._open(self.k)


def _open_stream():
    if ARCHIVE.exists() and ARCHIVE.stat().st_size == COD_SIZE:
        return open(ARCHIVE, "rb"), "archive"
    if PARTS.exists():
        return io.BufferedReader(_PartsStream(wait=True), 1 << 22), "parts"
    raise FileNotFoundError("no COD archive or parts; run download()")


_SU = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def num(v) -> float | None:
    """CIF numeric value without its standard uncertainty: '293(2)' → 293.0; '?', '.', text → None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
        if v is None:
            return None
    m = _SU.match(str(v))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


_FORMULA_RE = re.compile(
    r"^_chemical_formula_sum[ \t]*(?:\r?\n[ \t]*)?(?:'([^'\r\n]*)'|\"([^\"\r\n]*)\"|;([^;]*?)\r?\n;|([^\s'\";][^\r\n]*))",
    re.M | re.I)


def formula_sum_text(text: str) -> str | None:
    m = _FORMULA_RE.search(text)
    if not m:
        return None
    v = next(g for g in m.groups() if g is not None).strip()
    return v if v and v not in ("?", ".") else None


def reduced_formula(formula: str) -> tuple[str, frozenset]:
    """pymatgen reduced formula (same convention as Table B formula_reduced) and the element set."""
    from pymatgen.core import Composition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Composition(formula.replace("'", " "))
    if c.num_atoms <= 0:
        raise ValueError("empty composition")
    return c.reduced_formula, frozenset(str(e) for e in c.elements)


def _lower_block(text: str) -> dict:
    from pymatgen.io.cif import CifFile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf_ = CifFile.from_str(text)
    blocks = list(cf_.data.values())
    if not blocks:
        raise ValueError("no data block")
    return {k.lower(): v for k, v in blocks[0].data.items()}, len(blocks)


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = " ".join(str(x) for x in v)
    s = str(v).strip()
    return None if s in ("", "?", ".") else s


def quality_screen(d: dict) -> tuple[str, dict]:
    """Apply the quality filters to a lower-cased CIF block dict. Returns (reason, metadata).

    reason is the first failing stage (STAGES order) or "kept". Metadata are recorded either way.
    """
    meta: dict = {}
    method = next((_str(d.get(t)) for t in METHOD_TAGS if _str(d.get(t))), None)
    meta["method"] = method
    err = _str(d.get("_cod_error_flag"))
    sev = [str(s).strip().lower() for s in _as_list(d.get("_cod_entry_issue_severity"))]
    meta["cod_error_flag"] = err
    meta["cod_issue_severities"] = ",".join(sorted(set(sev))) or None
    occ = [num(o) for o in _as_list(d.get("_atom_site_occupancy"))]
    occ = [o for o in occ if o is not None]
    meta["occ_min"] = min(occ) if occ else None
    meta["occ_max"] = max(occ) if occ else None
    dis = [str(x).strip() for t in DISORDER_TAGS for x in _as_list(d.get(t))]
    meta["disorder_flag"] = any(x not in (".", "?", "") for x in dis)
    temps = [num(d.get(t)) for t in TEMP_TAGS]
    temps = [t for t in temps if t is not None]
    meta["temperature_K"] = max(temps) if temps else None
    pk = [num(d.get(t)) for t in PRESSURE_KPA_TAGS] + [None if num(d.get(t)) is None else num(d.get(t)) * 1e6
                                                       for t in PRESSURE_GPA_TAGS]
    pk = [p for p in pk if p is not None]
    meta["pressure_kPa"] = max(pk) if pk else None
    env = _str(d.get("_diffrn_ambient_environment")) or ""
    meta["anvil_environment"] = bool(re.search(r"anvil|\bDAC\b", env, re.I))
    r_val, r_tag = None, None
    for t in R_TAGS:
        v = num(d.get(t))
        if v is not None:
            r_val, r_tag = v, t
            break
    meta["r_factor"], meta["r_tag"] = r_val, r_tag
    title = _str(d.get("_publ_section_title")) or ""
    meta["title_hp_marker"] = bool(TITLE_HP_RE.search(title))
    meta["title_ht_marker"] = bool(TITLE_HT_RE.search(title))
    meta["pressure_history"] = _str(d.get("_exptl_crystal_pressure_history"))
    meta["rho_diffrn"] = num(d.get("_exptl_crystal_density_diffrn"))
    meta["sg_number_cif"] = num(d.get("_space_group_it_number")) or num(d.get("_symmetry_int_tables_number"))
    meta["sg_hm_cif"] = _str(d.get("_space_group_name_h-m_alt")) or _str(d.get("_symmetry_space_group_name_h-m"))
    meta["journal_year"] = num(d.get("_journal_year"))

    fails = []
    if method and method.lower().strip() == "theoretical":
        fails.append("theoretical_method")
    if (err and err.lower() in ("errors", "retracted")) or any(s in ("error", "errors", "retraction") for s in sev):
        fails.append("cod_error_or_retraction")
    if occ and any(abs(o - 1.0) > OCC_TOL for o in occ):
        fails.append("occupancy_not_unity")
    if meta["disorder_flag"]:
        fails.append("disorder_flag")
    if temps and max(temps) > T_MAX_K:
        fails.append("temperature_gt_350K")
    if (pk and max(pk) > P_MAX_KPA) or meta["anvil_environment"]:
        fails.append("high_pressure")
    if r_val is not None and r_val > R_MAX:
        fails.append("r_factor_gt_0.1")
    meta["all_fails"] = ",".join(fails)
    return (fails[0] if fails else "kept"), meta


def screen_cif(name: str, raw: bytes, formulas) -> dict:
    """Full screen of one COD CIF (text pre-filter, then quality filters). Pure function (tests use it)."""
    cod_id = int(re.sub(r"\D", "", Path(name).stem) or -1)
    rec = {"cod_id": cod_id, "path": name, "reason": None, "formula_reduced": None}
    text = raw.decode("utf-8", errors="replace")
    fs = formula_sum_text(text)
    if fs is None:
        rec["reason"] = "no_formula_sum"
        return rec
    rec["formula_sum"] = fs
    try:
        fr, els = reduced_formula(fs)
    except Exception:
        rec["reason"] = "formula_unparseable"
        return rec
    rec["formula_reduced"] = fr
    if "C" in els and "H" in els:
        rec["reason"] = "organic"
        return rec
    if fr not in formulas:
        rec["reason"] = "not_in_mp_formula_set"
        return rec
    try:
        d, nblocks = _lower_block(text)
    except Exception:
        rec["reason"] = "cif_unreadable"
        return rec
    reason, meta = quality_screen(d)
    rec.update(meta)
    rec["n_blocks"] = nblocks
    rec["reason"] = reason
    if reason == "kept":
        rec["cif"] = text
    return rec


_FORMULAS: frozenset = frozenset()


def _init_screen(formulas):
    global _FORMULAS
    _FORMULAS = formulas
    warnings.simplefilter("ignore")


def _screen_batch(batch):
    return [screen_cif(n, b, _FORMULAS) for n, b in batch]


def mp_formula_set() -> frozenset:
    u = common.universe()
    return frozenset(u.loc[u["eligible"], "formula_reduced"].unique())


SHARD_CIFS = 25000                     # checkpoint granularity of the archive stream (CIF members)
SHARDS_BASE = INTERIM / "screen_shards"
CACHE_FP = INTERIM / "cache_fingerprints.json"


# ---- cache and shard fingerprints (review F-7) -----------------------------------------------
def _fingerprint(**kw) -> str:
    """Short hash of every input and parameter of a cached stage."""
    return hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _fp_filter(formulas, stop_after_cif_tree: bool = True) -> str:
    fh = hashlib.sha256("\n".join(sorted(formulas)).encode()).hexdigest()
    return _fingerprint(stage="filter", archive=ARCHIVE.name, size=COD_SIZE, sha256=COD_SHA256, formulas=fh,
                        t_max=T_MAX_K, p_max=P_MAX_KPA, r_max=R_MAX, occ=OCC_TOL, stages=STAGES,
                        shard=SHARD_CIFS, stop=stop_after_cif_tree)


def _pymatgen_version() -> str:
    from importlib.metadata import version
    return version("pymatgen")


def _fp_parse() -> str:
    return _fingerprint(stage="parse", upstream=_cache_fps().get("filter"), tol=RHO_CROSSCHECK_TOL,
                        shard=PARSE_SHARD_N, stages=PARSE_STAGES, pymatgen=_pymatgen_version())


def _fp_match(mp_sha: str) -> str:
    return _fingerprint(stage="match", upstream=_cache_fps().get("parse"), mp_structures=mp_sha,
                        mp_tol=MP_DENSITY_TOL, matcher="StructureMatcher() defaults", pymatgen=_pymatgen_version())


def _cache_fps() -> dict:
    return json.loads(CACHE_FP.read_text(encoding="utf-8")) if CACHE_FP.exists() else {}


def _cache_record(name: str, fp: str) -> None:
    rec = _cache_fps()
    rec[name] = fp
    common.write_json(CACHE_FP, rec)


def _cache_check(name: str, fp: str) -> None:
    """Refuse a cached stage built from different inputs; adopt a cache that predates fingerprinting."""
    rec = _cache_fps()
    if name in rec and rec[name] != fp:
        raise RuntimeError(f"stale COD '{name}' cache (fingerprint {rec[name]} != {fp}); rerun that stage "
                           f"with refresh=True")
    if name not in rec:
        _cache_record(name, fp)


def _remove_shards(d: Path) -> None:
    """Delete a shard directory once its final parquet exists (and its base directory if now empty)."""
    if d.exists():
        shutil.rmtree(d)
    base = d.parent
    if base.exists() and base != INTERIM and not any(base.iterdir()):
        base.rmdir()


def _done_shards(sdir: Path) -> int:
    k = 0
    while (sdir / f"screen_{k:03d}.parquet").exists() and (sdir / f"kept_{k:03d}.parquet").exists():
        k += 1
    return k


def _write_shard(sdir: Path, k: int, recs: list, kept: list) -> None:
    paths.ensure(sdir)
    for name, rows in (("kept", kept), ("screen", recs)):        # screen last: its presence marks completion
        tmp = sdir / f"{name}_{k:03d}.parquet.tmp"
        (pd.DataFrame(rows) if rows else pd.DataFrame({"cod_id": pd.Series([], dtype="int64")})).to_parquet(
            tmp, index=False, compression="zstd")
        tmp.replace(sdir / f"{name}_{k:03d}.parquet")


def filter_archive(workers: int = N_WORKERS, refresh: bool = False, batch_size: int = 400,
                   stop_after_cif_tree: bool = True) -> pd.DataFrame:
    """Stream the COD archive once and cache the surviving CIFs (data/interim/cod/cod_filtered.parquet).

    Resumable: results are checkpointed every SHARD_CIFS CIF members (in archive order) under
    data/interim/cod/screen_shards/. A restart re-decodes the stream but skips screening of the
    completed shards, and checks that the last skipped member equals the last checkpointed one.
    With stop_after_cif_tree, streaming stops at the first top-level member outside cif/ after the
    cif/ tree (tar writes each directory argument contiguously); that member's name is recorded.
    """
    formulas = mp_formula_set()
    fp = _fp_filter(formulas, stop_after_cif_tree)
    if FILTERED_CACHE.exists() and SCREEN_LOG.exists() and not refresh:
        _cache_check("filter", fp)
        return pd.read_parquet(FILTERED_CACHE)
    paths.ensure(INTERIM)
    SHARDS = SHARDS_BASE / fp                     # shards are keyed by every input and parameter
    if refresh and SHARDS.exists():
        shutil.rmtree(SHARDS)
    n_done = _done_shards(SHARDS)
    skip = n_done * SHARD_CIFS
    last_done_path = (pd.read_parquet(SHARDS / f"screen_{n_done - 1:03d}.parquet", columns=["path"])["path"].iloc[-1]
                      if n_done else None)
    log.info("resuming after %d completed shards (%d CIFs)", n_done, skip)
    fobj, kind = _open_stream()
    t0 = time.time()
    state = {"recs": [], "kept": [], "shard": n_done}
    n_bytes, i_cif = 0, 0
    non_cif = Counter()
    seen_cif, stopped_at = False, None
    with fobj, cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_screen,
                                      initargs=(formulas,)) as ex:
        pending: list = []

        def drain(limit):
            while len(pending) > limit:
                for r in pending.pop(0).result():
                    if r["reason"] == "kept":
                        state["kept"].append(r)
                        r = {k: v for k, v in r.items() if k != "cif"}
                    state["recs"].append({k: r.get(k) for k in ("cod_id", "path", "reason", "formula_reduced",
                                                                 "all_fails")})
                    if len(state["recs"]) == SHARD_CIFS:
                        _write_shard(SHARDS, state["shard"], state["recs"], state["kept"])
                        log.info("shard %d written (%.0f s)", state["shard"], time.time() - t0)
                        state.update(recs=[], kept=[], shard=state["shard"] + 1)

        tf = tarfile.open(fileobj=fobj, mode="r|xz")
        batch = []
        for m in tf:
            top = m.name.split("/")[0]
            if top == "cif":
                seen_cif = True
            elif seen_cif and stop_after_cif_tree:
                stopped_at = m.name          # tar stores a directory argument contiguously: the cif/ tree is complete
                break
            if not m.isfile():
                continue
            if not (m.name.startswith("cif/") and m.name.endswith(".cif")):
                non_cif[top] += 1
                continue
            i_cif += 1
            if i_cif <= skip:
                if i_cif == skip and m.name != last_done_path:
                    raise RuntimeError(f"stream order mismatch on resume: {m.name} != {last_done_path}")
                continue
            b = tf.extractfile(m).read()
            n_bytes += len(b)
            batch.append((m.name, b))
            if len(batch) >= batch_size:
                pending.append(ex.submit(_screen_batch, batch))
                batch = []
                drain(4 * workers)
        if batch:
            pending.append(ex.submit(_screen_batch, batch))
        drain(0)
        if state["recs"]:
            _write_shard(SHARDS, state["shard"], state["recs"], state["kept"])
    log.info("streamed %d CIFs (%d screened in this run, %.1f GB) in %.0f s", i_cif, i_cif - skip, n_bytes / 1e9,
             time.time() - t0)
    n_sh = _done_shards(SHARDS)
    screen = pd.concat([pd.read_parquet(SHARDS / f"screen_{k:03d}.parquet") for k in range(n_sh)], ignore_index=True)
    if len(screen) != i_cif or screen["path"].duplicated().any():
        raise RuntimeError(f"shards hold {len(screen)} rows for {i_cif} CIF members")
    screen = screen.sort_values("cod_id", kind="stable").reset_index(drop=True)
    screen.to_parquet(SCREEN_LOG, index=False)
    kparts = [pd.read_parquet(SHARDS / f"kept_{k:03d}.parquet") for k in range(n_sh)]
    kdf = pd.concat([k for k in kparts if len(k)], ignore_index=True)
    kdf = kdf.sort_values("cod_id", kind="stable").reset_index(drop=True)
    kdf.to_parquet(FILTERED_CACHE, index=False, compression="zstd")
    common.write_json(INTERIM / "filter_meta.json", {
        "source": kind, "archive": ARCHIVE.name, "n_cif": len(screen), "n_resumed_shards": n_done,
        "stop_after_cif_tree": stop_after_cif_tree, "stopped_at_member": stopped_at,
        "uncompressed_cif_bytes_screened_last_run": n_bytes,
        "non_cif_members_by_top_dir": dict(non_cif), "seconds_last_run": round(time.time() - t0, 1),
        "n_mp_formulas": len(formulas), "created_utc": _utc(time.time())})
    _cache_record("filter", fp)
    _remove_shards(SHARDS)
    return kdf


# =============================================================================================
# 3. parse COD structures
# =============================================================================================
def parse_one(text: str, formula_reduced: str, rho_diffrn: float | None) -> dict:
    """Parse one CIF into an ordered pymatgen structure and density. Pure function."""
    from pymatgen.io.cif import CifParser
    out = {"stage": None, "rho_cod": None, "nsites": None, "structure_json": None, "rho_crosscheck": None,
           "formula_parsed": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structs = CifParser.from_str(text).parse_structures(primitive=False, on_error="raise")
    except Exception:
        out["stage"] = "parse_failed"
        return out
    if len(structs) != 1:
        out["stage"] = "not_one_structure"
        return out
    s = structs[0]
    if not s.is_ordered:
        out["stage"] = "parsed_disordered"
        return out
    try:
        s.remove_oxidation_states()          # fails on dummy species (e.g. "M"), which cannot be matched to MP
        out["formula_parsed"] = s.composition.reduced_formula
    except Exception:
        out["stage"] = "nonelement_species"
        return out
    if s.composition.reduced_formula != formula_reduced:
        out["stage"] = "composition_mismatch"
        return out
    rho = float(s.density)
    out["rho_cod"], out["nsites"] = rho, len(s)
    if rho_diffrn is not None and rho_diffrn > 0 and np.isfinite(rho_diffrn):
        out["rho_crosscheck"] = rho / rho_diffrn - 1.0
        if abs(out["rho_crosscheck"]) > RHO_CROSSCHECK_TOL:
            out["stage"] = "density_crosscheck_fail"
            return out
    out["structure_json"] = json.dumps(s.as_dict(), separators=(",", ":"))
    out["stage"] = "parsed_ok"
    return out


def _parse_task(args):
    return [parse_one(*a) for a in args]


PARSE_SHARDS_BASE = INTERIM / "parse_shards"
PARSE_SHARD_N = 2000


def parse_cod(workers: int = N_WORKERS, refresh: bool = False) -> pd.DataFrame:
    """Parse the cached CIFs (resumable: shards of PARSE_SHARD_N entries under parse_shards/)."""
    fp = _fp_parse()
    if STRUCT_CACHE.exists() and not refresh:
        _cache_check("parse", fp)
        return pd.read_parquet(STRUCT_CACHE)
    f = pd.read_parquet(FILTERED_CACHE)
    PARSE_SHARDS = PARSE_SHARDS_BASE / fp
    if refresh and PARSE_SHARDS.exists():
        shutil.rmtree(PARSE_SHARDS)
    paths.ensure(PARSE_SHARDS)
    n_sh = (len(f) + PARSE_SHARD_N - 1) // PARSE_SHARD_N
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=warnings.simplefilter,
                                initargs=("ignore",)) as ex:
        for k in range(n_sh):
            dest = PARSE_SHARDS / f"parse_{k:04d}.parquet"
            if dest.exists() and not refresh:
                continue
            sub = f.iloc[k * PARSE_SHARD_N:(k + 1) * PARSE_SHARD_N]
            args = [(t, fr, (None if pd.isna(r) else float(r)))
                    for t, fr, r in zip(sub["cif"], sub["formula_reduced"], sub["rho_diffrn"])]
            res = []
            for r in ex.map(_parse_task, [args[i:i + 50] for i in range(0, len(args), 50)]):
                res.extend(r)
            out = pd.DataFrame(res)
            out.insert(0, "cod_id", sub["cod_id"].to_numpy())
            out.to_parquet(dest.with_suffix(".tmp"), index=False, compression="zstd")
            dest.with_suffix(".tmp").replace(dest)
            log.info("parse shard %d/%d", k + 1, n_sh)
    p = pd.concat([pd.read_parquet(PARSE_SHARDS / f"parse_{k:04d}.parquet") for k in range(n_sh)],
                  ignore_index=True)
    if not (p["cod_id"].to_numpy() == f["cod_id"].to_numpy()).all():
        raise RuntimeError("parse shards are not aligned with the filtered cache")
    out = pd.concat([f.drop(columns=["cif"]).reset_index(drop=True), p.drop(columns=["cod_id"])], axis=1)
    out.to_parquet(STRUCT_CACHE, index=False, compression="zstd")
    _cache_record("parse", fp)
    _remove_shards(PARSE_SHARDS)
    return out


# =============================================================================================
# 4. MP structures (one official-API retrieval)
# =============================================================================================
def _mp_struct_paths(db: str) -> tuple[Path, Path]:
    return MP_RAW / f"structures_cod_{db}.jsonl.gz", MP_RAW / f"structures_cod_{db}.meta.json"


def _mp_db_snapshot() -> str:
    from shieldmap.ingest import mp as ingest_mp
    snap = ingest_mp.existing_snapshot()
    return ingest_mp._db_version_of(snap)


def mp_targets(cod_parsed: pd.DataFrame) -> pd.DataFrame:
    u = common.universe()
    ok = set(cod_parsed.loc[cod_parsed["stage"] == "parsed_ok", "formula_reduced"])
    return u[u["eligible"] & u["formula_reduced"].isin(ok)]


def fetch_mp_structures(cod_parsed: pd.DataFrame | None = None, refresh: bool = False) -> pd.DataFrame:
    """Retrieve (once) and verify the MP structures needed for matching. Returns one row per target entry."""
    from shieldmap.ingest import mp as ingest_mp
    db = _mp_db_snapshot()
    dest, meta_p = _mp_struct_paths(db)
    cod_parsed = parse_cod() if cod_parsed is None else cod_parsed
    tgt = mp_targets(cod_parsed)
    b = pd.read_parquet(paths.PROCESSED / "table_b_compounds.parquet", columns=["material_id", "material_id_api"])
    api_of = dict(zip(b["material_id"], b["material_id_api"]))
    if not dest.exists() or refresh:
        from mp_api.client import MPRester
        ids_api = [api_of[m] for m in tgt["material_id"]]          # universe row order (deterministic)
        cdir = paths.ensure(MP_RAW / f"structures_cod_{db}_chunks")
        prog_p = cdir / "progress.json"
        prog = json.loads(prog_p.read_text(encoding="utf-8")) if prog_p.exists() else {}
        if prog.get("ids_sha256") != hashlib.sha256("\n".join(ids_api).encode()).hexdigest():
            for c in cdir.glob("chunk_*.jsonl.gz"):
                c.unlink()
            prog = {"ids_sha256": hashlib.sha256("\n".join(ids_api).encode()).hexdigest(),
                    "started_utc": _utc(time.time()), "n_requested": len(ids_api)}
            common.write_json(prog_p, prog)
        with MPRester(ingest_mp._api_key(), use_document_model=False, mute_progress_bars=True) as mpr:
            db_api = ingest_mp._retry(lambda: str(mpr.get_database_version()), "get database version")
            for i in range(0, len(ids_api), 1000):
                cp = cdir / f"chunk_{i // 1000:04d}.jsonl.gz"
                if cp.exists():
                    continue
                chunk = ids_api[i:i + 1000]
                docs = ingest_mp._retry(
                    lambda: mpr.materials.summary.search(material_ids=chunk, fields=["material_id", "structure"],
                                                         chunk_size=1000),
                    f"structure chunk {i}")
                with gzip.GzipFile(cp.with_suffix(".tmp"), "wb", mtime=0) as gz:
                    for d in docs:
                        gz.write((json.dumps({"material_id": str(d["material_id"]), "structure": d["structure"]},
                                             default=str, sort_keys=True) + "\n").encode())
                cp.with_suffix(".tmp").replace(cp)
                log.info("MP structures chunk %d (%d docs)", i // 1000, len(docs))
            db_after = str(mpr.get_database_version())
        lines = []
        for cp in sorted(cdir.glob("chunk_*.jsonl.gz")):
            with gzip.open(cp, "rt", encoding="utf-8") as g:
                lines.extend(g.read().splitlines())
        lines.sort(key=lambda ln: json.loads(ln)["material_id"])
        tmp = dest.with_suffix(".gz.part")
        with gzip.GzipFile(tmp, "wb", mtime=0) as gz:
            for ln in lines:
                gz.write((ln + "\n").encode())
        tmp.replace(dest)
        common.write_json(meta_p, {
            "source": "Materials Project", "endpoint": "https://api.materialsproject.org/materials/summary/",
            "client_call": "MPRester(use_document_model=False).materials.summary.search(material_ids=..., "
                           "fields=['material_id','structure'], chunk_size=1000), 1000 ids per request",
            "purpose": "final structures of eligible (R3) MP entries whose reduced formula occurs among the "
                       "filtered, parsed COD entries (Phase 3 Step 3 structural matching)",
            "table_b_database_version": db, "api_database_version": db_api, "api_database_version_after": db_after,
            "retrieval_started_utc": prog["started_utc"], "retrieval_finished_utc": _utc(time.time()),
            "n_requested": len(ids_api), "n_returned": len(lines),
            "packages": _pkg_versions(), "file": dest.name, "sha256": common.sha256(dest)})
        for cp in cdir.glob("*"):
            cp.unlink()
        cdir.rmdir()
    verify_mp_structures(dest, meta_p, db)
    rows = []
    from pymatgen.core import Structure
    canon = {v: k for k, v in api_of.items()}
    dens = dict(zip(tgt["material_id"], tgt["density"]))
    with gzip.open(dest, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            mid_api = d["material_id"]
            mid = canon.get(mid_api) or ingest_mp.parse_id(mid_api)[2]
            if mid not in dens:
                continue
            s = Structure.from_dict(d["structure"])
            rel = s.density / dens[mid] - 1.0
            rows.append({"material_id": mid, "rho_struct": float(s.density), "rho_table_b": float(dens[mid]),
                         "density_rel_diff": rel, "density_ok": abs(rel) <= MP_DENSITY_TOL,
                         "structure_json": json.dumps(d["structure"], separators=(",", ":"))})
    out = pd.DataFrame(rows)
    missing = sorted(set(tgt["material_id"]) - set(out["material_id"]))
    common.write_json(INTERIM / "mp_structures_check.json", {
        "n_targets": len(tgt), "n_retrieved": len(out), "n_missing": len(missing), "missing": missing[:200],
        "n_density_mismatch": int((~out["density_ok"]).sum()),
        "density_mismatch_ids": out.loc[~out["density_ok"], "material_id"].tolist()[:500],
        "max_abs_rel_diff_ok": float(out.loc[out["density_ok"], "density_rel_diff"].abs().max()),
        "tolerance": MP_DENSITY_TOL})
    return out


def verify_mp_structures(dest: Path, meta_p: Path, db: str) -> str:
    """SHA-256 of the cached MP structures file against its .meta.json and the tracked value (review F-3)."""
    h = common.sha256(dest)
    rec = json.loads(meta_p.read_text(encoding="utf-8")).get("sha256") if meta_p.exists() else None
    if rec is not None and h != rec:
        raise RuntimeError(f"{dest.name}: SHA-256 {h} != .meta.json {rec}")
    if db in MP_STRUCT_SHA256 and h != MP_STRUCT_SHA256[db]:
        raise RuntimeError(f"{dest.name}: SHA-256 {h} != tracked {MP_STRUCT_SHA256[db]}")
    return h


def _pkg_versions() -> dict:
    from importlib.metadata import version
    return {p: version(p) for p in ("mp-api", "emmet-core", "pymatgen")}


# =============================================================================================
# 5. structural matching
# =============================================================================================
def _reduce_task(items):
    """(key, structure_json) → (key, reduced structure dict, n reduced sites)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    warnings.simplefilter("ignore")
    out = []
    for key, sj in items:
        try:
            s = Structure.from_dict(json.loads(sj))
            r = StructureMatcher._get_reduced_structure(s, True, True)   # what fit() does with the defaults
            out.append((key, r.as_dict(), len(r)))
        except Exception:
            out.append((key, None, None))
    return out


def _match_task(args):
    """Fit every (MP, COD) pair of one formula / reduced-site-count group with default StructureMatcher."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    warnings.simplefilter("ignore")
    mp_items, cod_items = args
    sm = StructureMatcher()                     # default tolerances: ltol 0.2, stol 0.3, angle 5°, scale=True
    cods = [(cid, Structure.from_dict(d)) for cid, d in cod_items]
    res = []
    for mid, d in mp_items:
        s = Structure.from_dict(d)
        for cid, c in cods:
            try:
                ok = bool(sm.fit(s, c, skip_structure_reduction=True))
            except Exception:
                ok = None
            res.append((mid, cid, ok))
    return res


def _check_skipped_task(pairs):
    """Run full default fit() (no pre-reduction) on pairs skipped by the site-count rule (verification)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    warnings.simplefilter("ignore")
    sm = StructureMatcher()
    out = []
    for a, b in pairs:
        try:
            out.append(bool(sm.fit(Structure.from_dict(json.loads(a)), Structure.from_dict(json.loads(b)))))
        except Exception:
            out.append(None)
    return out


def select_cod(cands: pd.DataFrame) -> dict:
    """Selection rule: one COD value per MP entry from its structurally matching COD entries.

    cands columns: cod_id, r_factor (NaN if not given), rho_cod.
    1. If any matched entry reports R, keep those with the lowest R. If exactly one, it is the match
       ("lowest_R").
    2. If several share that lowest R ("median_tied_R"), or none reports R ("median_no_R"), the
       matched density is the median ρ over that equivalent set. The cod_id is the member closest to
       the median (lowest cod_id on ties).
    A single matched entry is "single".
    """
    c = cands.sort_values("cod_id")
    n = len(c)
    if n == 0:
        raise ValueError("no candidates")
    if n == 1:
        return {"cod_id": int(c["cod_id"].iloc[0]), "rho_cod": float(c["rho_cod"].iloc[0]), "rule": "single",
                "n_set": 1}
    r = c["r_factor"].to_numpy(dtype=float)
    if np.isfinite(r).any():
        rmin = np.nanmin(r)
        s = c[np.isclose(r, rmin, rtol=0, atol=1e-12)]
        if len(s) == 1:
            return {"cod_id": int(s["cod_id"].iloc[0]), "rho_cod": float(s["rho_cod"].iloc[0]),
                    "rule": "lowest_R", "n_set": 1}
        rule = "median_tied_R"
    else:
        s, rule = c, "median_no_R"
    med = float(np.median(s["rho_cod"].to_numpy(dtype=float)))
    i = int(np.argmin(np.abs(s["rho_cod"].to_numpy(dtype=float) - med)))
    return {"cod_id": int(s["cod_id"].iloc[i]), "rho_cod": med, "rule": rule, "n_set": len(s)}


RED_SHARDS_BASE = INTERIM / "reduce_shards"
MATCH_SHARDS_BASE = INTERIM / "match_shards"


def _sharded_map(fn, items: list, shard_dir: Path, prefix: str, per_shard: int, per_task: int, workers: int,
                 columns: list[str]) -> pd.DataFrame:
    """Map fn over items in task chunks with per-shard parquet checkpoints (resumable, deterministic order)."""
    paths.ensure(shard_dir)
    n_sh = (len(items) + per_shard - 1) // per_shard
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for k in range(n_sh):
            dest = shard_dir / f"{prefix}_{k:05d}.parquet"
            if dest.exists():
                continue
            sub = items[k * per_shard:(k + 1) * per_shard]
            res = []
            for r in ex.map(fn, [sub[i:i + per_task] for i in range(0, len(sub), per_task)]):
                res.extend(r)
            pd.DataFrame(res, columns=columns).to_parquet(dest.with_suffix(".tmp"), index=False)
            dest.with_suffix(".tmp").replace(dest)
            log.info("%s shard %d/%d", prefix, k + 1, n_sh)
    return pd.concat([pd.read_parquet(shard_dir / f"{prefix}_{k:05d}.parquet") for k in range(n_sh)],
                     ignore_index=True)


def _reduce_task_flat(items):
    return [(src, str(key), (None if d is None else json.dumps(d, separators=(",", ":"))), n)
            for (src, key), d, n in _reduce_task(items)]


def _match_task_flat(tasks):
    out = []
    for t in tasks:
        out.extend(_match_task(t))
    return out


def match(workers: int = N_WORKERS, refresh: bool = False, n_verify_skipped: int = 1000) -> pd.DataFrame:
    """Structural matching (resumable cached pair table) → one row per matched MP entry (matches.parquet)."""
    cod = parse_cod()
    cod = cod[cod["stage"] == "parsed_ok"].reset_index(drop=True)
    mps = fetch_mp_structures(cod)
    mps_ok = mps[mps["density_ok"]].reset_index(drop=True)
    u = common.universe()
    db = _mp_db_snapshot()
    fp = _fp_match(verify_mp_structures(*_mp_struct_paths(db), db))
    RED_SHARDS, MATCH_SHARDS = RED_SHARDS_BASE / fp, MATCH_SHARDS_BASE / fp
    if refresh:
        for d in (RED_SHARDS, MATCH_SHARDS):
            if d.exists():
                shutil.rmtree(d)
        MATCH_CACHE.unlink(missing_ok=True)
    if MATCH_CACHE.exists():
        _cache_check("match", fp)
    else:
        t0 = time.time()
        items = ([(("mp", m), sj) for m, sj in zip(mps_ok["material_id"], mps_ok["structure_json"])]
                 + [(("cod", str(int(c))), sj) for c, sj in zip(cod["cod_id"], cod["structure_json"])])
        red_df = _sharded_map(_reduce_task_flat, items, RED_SHARDS, "reduce", 5000, 50, workers,
                              ["src", "key", "reduced_json", "n_reduced"])
        log.info("reduced %d structures (%.0f s)", len(red_df), time.time() - t0)
        form_mp = dict(zip(u["material_id"], u["formula_reduced"]))
        form_cod = {str(int(c)): f for c, f in zip(cod["cod_id"], cod["formula_reduced"])}
        red_df["formula_reduced"] = [form_mp[k] if s_ == "mp" else form_cod[k]
                                     for s_, k in zip(red_df["src"], red_df["key"])]
        ok = red_df[red_df["reduced_json"].notna()]
        groups: dict = {}
        for s_, k, d, n, f in zip(ok["src"], ok["key"], ok["reduced_json"], ok["n_reduced"], ok["formula_reduced"]):
            g = groups.setdefault((f, int(n)), {"mp": [], "cod": []})
            g[s_].append((k if s_ == "mp" else int(k), json.loads(d)))
        tasks = []
        for (f, n) in sorted(groups):
            g = groups[(f, n)]
            if not g["mp"] or not g["cod"]:
                continue
            step = max(1, 2000 // max(1, len(g["cod"])))
            for i in range(0, len(g["mp"]), step):
                tasks.append((g["mp"][i:i + step], g["cod"]))
        n_pairs = sum(len(a_) * len(b_) for a_, b_ in tasks)
        log.info("%d match tasks, %d pairs", len(tasks), n_pairs)
        pairs = _sharded_map(_match_task_flat, tasks, MATCH_SHARDS, "match", 200, 1, workers,
                             ["material_id", "cod_id", "fit"])
        # verification of the exact site-count shortcut on a random sample of skipped same-formula pairs
        rng = np.random.default_rng(BOOT_SEED)
        by_f: dict = {}
        for s_, k, n, f in zip(ok["src"], ok["key"], ok["n_reduced"], ok["formula_reduced"]):
            by_f.setdefault(f, {"mp": [], "cod": []})[s_].append((k, int(n)))
        skipped = [(m, int(c)) for f, g in sorted(by_f.items()) for m, nm in g["mp"] for c, nc in g["cod"] if nm != nc]
        sample = [skipped[i] for i in sorted(rng.choice(len(skipped), size=min(n_verify_skipped, len(skipped)),
                                                          replace=False))] if skipped else []
        sj_mp = dict(zip(mps_ok["material_id"], mps_ok["structure_json"]))
        sj_cod = dict(zip(cod["cod_id"].astype(int), cod["structure_json"]))
        vchunks = [[(sj_mp[m], sj_cod[c]) for m, c in sample[i:i + 20]] for i in range(0, len(sample), 20)]
        vres = []
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_check_skipped_task, vchunks):
                vres.extend(r)
        common.write_json(INTERIM / "match_meta.json", {
            "n_structures_reduced": len(red_df), "n_reduce_failed": int(red_df["reduced_json"].isna().sum()),
            "n_mp_structures_density_ok": len(mps_ok), "n_cod_parsed_ok": len(cod),
            "n_groups_formula_x_reduced_sites": len(groups), "n_tasks": len(tasks), "n_pairs_fitted": n_pairs,
            "n_pairs_fit_error": int(pairs["fit"].isna().sum()), "n_pairs_true": int((pairs["fit"] == True).sum()),  # noqa: E712
            "n_pairs_skipped_site_count": len(skipped), "n_skipped_verified": len(vres),
            "n_skipped_verified_true": int(sum(v is True for v in vres)),
            "n_skipped_verified_error": int(sum(v is None for v in vres)),
            "matcher": "pymatgen StructureMatcher() defaults (ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, "
                       "scale=True, attempt_supercell=False, allow_subset=False); structures pre-reduced once with "
                       "StructureMatcher._get_reduced_structure(primitive, niggli) and fitted with "
                       "skip_structure_reduction=True; pairs whose reduced cells have different site counts "
                       "cannot match under these settings and are not fitted (verified on a random sample)",
            "seconds_last_run": round(time.time() - t0, 1)})
        pairs.to_parquet(MATCH_CACHE, index=False)
        _cache_record("match", fp)
        _remove_shards(RED_SHARDS)
        _remove_shards(MATCH_SHARDS)
    pairs = pd.read_parquet(MATCH_CACHE)
    return _assemble_matches(pairs, cod, mps_ok, u)


def _assemble_matches(pairs, cod, mps_ok, u) -> pd.DataFrame:
    hit = pairs[pairs["fit"] == True]  # noqa: E712  (None = fit error, excluded)
    cmeta = cod.set_index(cod["cod_id"].astype(int))
    ui = u.set_index("material_id")
    rows = []
    for mid, g in hit.groupby("material_id", sort=True):
        ids = sorted(int(c) for c in g["cod_id"])
        cands = pd.DataFrame({"cod_id": ids, "r_factor": cmeta.loc[ids, "r_factor"].astype(float).to_numpy(),
                              "rho_cod": cmeta.loc[ids, "rho_cod"].astype(float).to_numpy()})
        sel = select_cod(cands)
        r = ui.loc[mid]
        cm = cmeta.loc[sel["cod_id"]]
        rho_mp = float(r["density"])
        els = list(r["elements"])
        rc = common.report_classes(els)
        row = {"row": int(r["row"]), "material_id": mid, "formula_reduced": r["formula_reduced"],
               "nelements": int(r["nelements"]), "cod_id": sel["cod_id"], "selection_rule": sel["rule"],
               "n_cod_equivalent": len(ids), "n_cod_selected_set": sel["n_set"],
               "cod_ids_equivalent": ",".join(map(str, ids)),
               "rho_mp": rho_mp, "rho_cod": sel["rho_cod"], "delta": rho_mp / sel["rho_cod"] - 1.0,
               "rho_cod_median_all": float(np.median(cands["rho_cod"])),
               "delta_median_all": rho_mp / float(np.median(cands["rho_cod"])) - 1.0,
               "e_hull": float(r["energy_above_hull"]), "theoretical": bool(r["theoretical"]),
               "in_R2": bool(r["in_R2"]), "is_representative": bool(r["is_representative"]),
               "primary_class": common.primary_class(els),
               **{f"is_{c}": (c in rc) for c in common.REPORT_CLASSES},
               "sg_mp": int(r["spacegroup_number"]),
               "sg_cod_cif": None if pd.isna(cm["sg_number_cif"]) else int(cm["sg_number_cif"]),
               "cod_r_factor": None if pd.isna(cm["r_factor"]) else float(cm["r_factor"]),
               "cod_temperature_K": None if pd.isna(cm["temperature_K"]) else float(cm["temperature_K"]),
               "cod_title_hp_marker": bool(cm["title_hp_marker"]), "cod_title_ht_marker": bool(cm["title_ht_marker"]),
               "is_metal": (None if pd.isna(r["is_metal"]) else bool(r["is_metal"]))}
        rows.append(row)
    m = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    m["ehull_bin"] = ehull_bin(m["e_hull"].to_numpy())
    m = screen_matches(m, u)
    paths.ensure(OUT)
    m.to_parquet(OUT / "matches.parquet", index=False)
    return m


def robust_z(L: np.ndarray, cls: np.ndarray, ref_L: np.ndarray, ref_cls: np.ndarray) -> tuple:
    """z = (L − median_c)/(1.4826·MAD_c) with class statistics from the reference population.

    L = ln(1 + δ) is symmetric in log space: ρ_MP/ρ_COD = 1.5 and 1/1.5 are equally far from 0.
    Classes absent from the reference use the pooled reference statistics. Returns (z, med, sigma).
    """
    L = np.asarray(L, float); cls = np.asarray(cls, dtype=object)
    ref_L = np.asarray(ref_L, float); ref_cls = np.asarray(ref_cls, dtype=object)
    pm = float(np.median(ref_L)); ps = 1.4826 * float(np.median(np.abs(ref_L - pm)))
    med = np.full(L.size, pm); sig = np.full(L.size, ps)
    for c in np.unique(cls):
        x = ref_L[ref_cls == c]
        if x.size >= MIN_CLASS_N:
            mc = float(np.median(x)); sc = 1.4826 * float(np.median(np.abs(x - mc)))
            med[cls == c] = mc; sig[cls == c] = sc
    return (L - med) / sig, med, sig


def element_reference(u: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage 1 experimental density per element with the space group of its representative MP entry."""
    u = common.universe() if u is None else u
    el = common.elements_table()
    rep = u[(u["nelements"] == 1) & u["is_representative"]]
    rep = rep.assign(symbol=[e[0] for e in rep["elements"]])
    out = el[["symbol", "density_solid_exp"]].merge(
        rep[["symbol", "material_id", "spacegroup_number"]].rename(
            columns={"material_id": "rep_material_id", "spacegroup_number": "rep_sg"}), on="symbol", how="inner")
    return out[out["density_solid_exp"].notna()].reset_index(drop=True)


def screen_matches(m: pd.DataFrame, u: pd.DataFrame | None = None) -> pd.DataFrame:
    """Robust log-symmetric outlier screen plus the elemental cross-check (the hardening revision).

    Columns added: L = ln(1+δ); z_robust (class median/MAD from the ≥ 2-element matches); excluded_k5 and
    excluded_k8 (|z| > SCREEN_K, > 8); gross_abs50 (the original |δ| > 0.5 rule); for unary matches
    xcheck_rel (ρ_COD/ρ_exp − 1 against the stage 1 experimental density, computed when the MP entry is
    plausibly the ambient phase, |ρ_MP/ρ_exp − 1| ≤ ELEMENT_PHASE_TOL) and xcheck_fail (|xcheck_rel| >
    ELEMENT_XCHECK_TOL); excluded = excluded_k5 or xcheck_fail (the screen used for kind="compound" and for
    measured densities).
    """
    m = m.copy()
    m["L"] = np.log1p(m["delta"].to_numpy(float))
    ref = m["nelements"] >= 2
    z, _, _ = robust_z(m["L"], m["primary_class"], m.loc[ref, "L"], m.loc[ref, "primary_class"])
    m["z_robust"] = z
    m["excluded_k5"] = np.abs(z) > SCREEN_K
    m["excluded_k8"] = np.abs(z) > 8.0
    m["gross_abs50"] = m["delta"].abs() > GROSS_DELTA
    er = element_reference(u).set_index("symbol")
    xr = np.full(len(m), np.nan)
    for i, (f, n, rmp, rho) in enumerate(zip(m["formula_reduced"], m["nelements"], m["rho_mp"], m["rho_cod"])):
        if n == 1 and f in er.index:
            rexp = float(er.at[f, "density_solid_exp"])
            if abs(rmp / rexp - 1.0) <= ELEMENT_PHASE_TOL:
                xr[i] = rho / rexp - 1.0
    m["xcheck_rel"] = xr
    m["xcheck_fail"] = np.abs(np.nan_to_num(xr, nan=0.0)) > ELEMENT_XCHECK_TOL
    m["excluded"] = m["excluded_k5"] | m["xcheck_fail"]
    return m


def ehull_bin(e: np.ndarray) -> np.ndarray:
    out = np.empty(len(e), dtype=object)
    for name, lo, hi in EHULL_BINS:
        if lo is None:
            sel = e <= hi
        elif hi is None:
            sel = e > lo
        else:
            sel = (e > lo) & (e <= hi)
        out[sel] = name
    return out


# =============================================================================================
# 6. statistics, tables, figure
# =============================================================================================
def stats(x, ci: bool = True, seed: int = BOOT_SEED) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    q = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    out = {"n": n, "median": q[3], "mean": float(x.mean()), "std": float(x.std(ddof=1)) if n > 1 else None,
           "q25": q[2], "q75": q[4], "iqr": q[4] - q[2], "p05": q[1], "p95": q[5], "p01": q[0], "p99": q[6],
           "min": float(x.min()), "max": float(x.max()),
           "frac_abs_gt_2pct": float(np.mean(np.abs(x) > 0.02)), "frac_abs_gt_5pct": float(np.mean(np.abs(x) > 0.05)),
           "frac_abs_gt_10pct": float(np.mean(np.abs(x) > 0.10)), "frac_positive": float(np.mean(x > 0))}
    if ci and n >= 5:
        rng = np.random.default_rng(seed)
        meds = np.median(x[rng.integers(0, n, size=(BOOT_N, n))], axis=1)
        out["median_ci95"] = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def cluster_median_ci(x, groups, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> list[float]:
    """95 % percentile CI of the median when whole clusters (formulas, COD entries) are resampled."""
    x = np.asarray(x, float)
    codes, _ = pd.factorize(pd.Series(groups).astype(str))
    ok = np.isfinite(x)
    x, codes = x[ok], codes[ok]
    order = np.argsort(x, kind="stable")
    xs, cs = x[order], codes[order]
    G = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[cs].astype(float)
        cw = np.cumsum(w)
        meds[b] = xs[np.searchsorted(cw, 0.5 * cw[-1])]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


SUBSETS_FOR_RANGE = ("cod_R_given", "cod_R_not_given", "cod_T_250_350K", "cod_T_below_250K", "cod_T_unspecified",
                     "metals", "non_metals")


def headline(m: pd.DataFrame) -> pd.DataFrame:
    """Headline population: matched eligible (R3) MP compounds (≥ 2 elements)."""
    return m[m["nelements"] >= 2]


def _phase1_elemental() -> np.ndarray:
    from shieldmap.analysis import sensitivity
    return sensitivity.bias_distribution("empirical")


def bias_tables(m: pd.DataFrame | None = None) -> dict:
    m = pd.read_parquet(OUT / "matches.parquet") if m is None else m
    if "excluded" not in m.columns:
        m = screen_matches(m)
    h = headline(m)
    pops = {"R3_compounds (headline)": h, "R2_compounds": h[h["in_R2"]],
            "ICSD-matched (theoretical=False)": h[~h["theoretical"]], "theoretical=True": h[h["theoretical"]],
            f"R3_compounds_screened_robust_k{SCREEN_K:g}": h[~h["excluded"]],
            "R3_compounds_screened_robust_k8": h[~h["excluded_k8"]],
            "R3_compounds_screened_abs_delta_le_0.5 (original)": h[~h["gross_abs50"]],
            "R3_including_unaries": m, "unaries_only": m[m["nelements"] == 1]}
    overall = {k: stats(v["delta"]) for k, v in pops.items()}
    hd = overall["R3_compounds (headline)"]
    hd["median_ci95_note"] = "sampling only: iid percentile bootstrap over pairs"
    hd["median_ci95_cluster_formula"] = cluster_median_ci(h["delta"], h["formula_reduced"])
    hd["median_ci95_cluster_cod_entry"] = cluster_median_ci(h["delta"], h["cod_id"])
    elemental = stats(_phase1_elemental())
    overall["phase1_elemental"] = elemental
    common.write_json(OUT / "bias_overall.json", overall)

    rows = []
    for pop_name, pop in (("R3", h), ("R2", h[h["in_R2"]])):
        rows.append({"population": pop_name, "class_kind": "all", "class": "all", **stats(pop["delta"])})
        for c in common.REPORT_CLASSES:
            rows.append({"population": pop_name, "class_kind": "report (overlapping)", "class": c,
                         **stats(pop.loc[pop[f"is_{c}"], "delta"])})
        for c in common.PRIMARY_CLASS_ORDER:
            rows.append({"population": pop_name, "class_kind": "primary (exclusive)", "class": c,
                         **stats(pop.loc[pop["primary_class"] == c, "delta"])})
    by_class = pd.DataFrame(rows)
    by_class["median_ci95"] = by_class["median_ci95"].apply(lambda v: json.dumps(v) if isinstance(v, list) else None)
    by_class.to_csv(OUT / "bias_by_class.csv", index=False)

    rows = []
    for pop_name, pop in (("R3", h), ("R2", h[h["in_R2"]])):
        for name, _, _ in EHULL_BINS:
            for th, sub in (("all", pop), ("ICSD-matched", pop[~pop["theoretical"]]),
                            ("theoretical", pop[pop["theoretical"]])):
                rows.append({"population": pop_name, "ehull_bin": name, "subset": th,
                             **stats(sub.loc[sub["ehull_bin"] == name, "delta"])})
    by_e = pd.DataFrame(rows)
    by_e["median_ci95"] = by_e["median_ci95"].apply(lambda v: json.dumps(v) if isinstance(v, list) else None)
    by_e.to_csv(OUT / "bias_by_ehull.csv", index=False)

    d = h[["material_id", "row", "primary_class", "delta", "z_robust", "excluded", "excluded_k8",
           "gross_abs50"]].sort_values("row").reset_index(drop=True)
    d.to_parquet(OUT / "delta_by_primary_class.parquet", index=False)
    _class_cache.cache_clear()

    bands = {"kind": "compound", "definition": "90 % interval (p05, p95) of δ = ρ_MP/ρ_COD − 1 per primary class; "
             f"classes with n < {MIN_CLASS_N} use the pooled distribution; kind 'compound' excludes pairs with "
             f"|ln(1+δ) − class median| > {SCREEN_K:g} robust SD (1.4826·MAD); 'compound_k8' uses 8 robust SD; "
             f"'compound_abs50' the original |δ| > {GROSS_DELTA} rule; 'compound_raw' keeps every pair",
             "min_class_n": MIN_CLASS_N, "screen_k": SCREEN_K, "gross_delta": GROSS_DELTA,
             "n_excluded": {"compound": int(h["excluded"].sum()), "compound_k8": int(h["excluded_k8"].sum()),
                            "compound_abs50": int(h["gross_abs50"].sum()), "compound_raw": 0},
             "classes": {}, "classes_k8": {}, "classes_abs50": {}, "classes_raw": {}}
    for kind, key in (("compound", "classes"), ("compound_k8", "classes_k8"), ("compound_abs50", "classes_abs50"),
                      ("compound_raw", "classes_raw")):
        for c, i in class_delta_info(kind).items():
            lo, hi = band(c, kind)
            bands[key][c] = {"p05": lo, "p95": hi, "median": float(np.median(_class_cache(kind, MIN_CLASS_N)[0][c])),
                             "n_class": i["n_class"], "uses_pooled": i["uses_pooled"]}
    el = _phase1_elemental()
    bands["phase1_elemental"] = {"p05": float(np.percentile(el, 5)), "p95": float(np.percentile(el, 95)), "n": len(el)}
    common.write_json(OUT / "band.json", bands)

    sens = {
        "selection_rule_median_of_all_matches": stats(h["delta_median_all"]),
        "excluding_title_high_pressure_marker": stats(h.loc[~h["cod_title_hp_marker"], "delta"]),
        "excluding_title_high_temperature_marker": stats(h.loc[~h["cod_title_ht_marker"], "delta"]),
        "cod_T_250_350K": stats(h.loc[h["cod_temperature_K"].between(250, 350), "delta"]),
        "cod_T_below_250K": stats(h.loc[h["cod_temperature_K"] < 250, "delta"]),
        "cod_T_unspecified": stats(h.loc[h["cod_temperature_K"].isna(), "delta"]),
        "cod_R_given": stats(h.loc[h["cod_r_factor"].notna(), "delta"]),
        "cod_R_not_given": stats(h.loc[h["cod_r_factor"].isna(), "delta"]),
        "same_space_group_cif": stats(h.loc[h["sg_cod_cif"] == h["sg_mp"], "delta"]),
        "metals": stats(h.loc[h["is_metal"] == True, "delta"]),  # noqa: E712
        "non_metals": stats(h.loc[h["is_metal"] == False, "delta"]),  # noqa: E712
        "one_per_cod_entry_R3": stats(h.sort_values("e_hull").drop_duplicates("cod_id")["delta"]),
    }
    meds = {k: sens[k]["median"] for k in SUBSETS_FOR_RANGE}
    sens["subset_median_range"] = {"subsets": list(SUBSETS_FOR_RANGE), "min": min(meds.values()),
                                   "max": max(meds.values()), "argmin": min(meds, key=meds.get),
                                   "argmax": max(meds, key=meds.get)}
    common.write_json(OUT / "sensitivity.json", sens)
    hd["subset_median_range"] = [sens["subset_median_range"]["min"], sens["subset_median_range"]["max"]]
    common.write_json(OUT / "bias_overall.json", overall)
    return {"overall": overall, "by_class": by_class, "by_ehull": by_e, "bands": bands, "sensitivity": sens}


# =============================================================================================
# 7. public API (the continuum module bootstrap)
# =============================================================================================
@lru_cache(maxsize=4)
def _class_cache(kind: str, min_n: int):
    if kind == "elemental":
        el = _phase1_elemental()
        d = {c: el.copy() for c in common.PRIMARY_CLASS_ORDER}
        d["pooled"] = el.copy()
        info = {c: {"n_class": len(el), "uses_pooled": False, "n_used": len(el)} for c in d}
        return d, info
    if kind not in ("compound", "compound_k8", "compound_abs50", "compound_raw"):
        raise ValueError(kind)
    p = OUT / "delta_by_primary_class.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run shieldmap.analysis3.cod.run()")
    df = pd.read_parquet(p)
    col = {"compound": "excluded", "compound_k8": "excluded_k8", "compound_abs50": "gross_abs50"}.get(kind)
    if col is not None:
        df = df[~df[col].astype(bool)]
    return class_distributions(df, min_n)


def class_distributions(df: pd.DataFrame, min_n: int = MIN_CLASS_N) -> tuple[dict, dict]:
    """Pure helper: (dict class → δ array, info) with the pooled fallback for classes with n < min_n."""
    pooled = df["delta"].to_numpy(dtype=float)
    d, info = {}, {}
    for c in common.PRIMARY_CLASS_ORDER:
        x = df.loc[df["primary_class"] == c, "delta"].to_numpy(dtype=float)
        use_pooled = len(x) < min_n
        d[c] = pooled.copy() if use_pooled else x
        info[c] = {"n_class": int(len(x)), "uses_pooled": bool(use_pooled), "n_used": int(len(d[c]))}
    d["pooled"] = pooled.copy()
    info["pooled"] = {"n_class": int(len(pooled)), "uses_pooled": False, "n_used": int(len(pooled))}
    return d, info


def class_delta_distribution(kind: str = "compound", min_n: int = MIN_CLASS_N) -> dict[str, np.ndarray]:
    """δ samples per exclusive primary class (keys: common.PRIMARY_CLASS_ORDER plus "pooled").

    δ = ρ_MP/ρ_ref − 1, the same sign as stage 1 rel_bias. A density closer to experiment is therefore
    ρ_MP/(1 + δ), and the thickness requirement scales by (1 + δ) (areal mass is invariant).

    kind="compound" (recommended, default): empirical COD-matched δ of the headline population, i.e.
        matched eligible (R3) MP compounds with ≥ 2 elements and one COD value per MP entry, minus the
        pairs outside the robust log-symmetric screen (|ln(1+δ) − class median| > SCREEN_K robust SD;
        see screen_matches). A class with fewer than ``min_n`` (default 30) matches returns the pooled
        distribution.
    kind="compound_k8": the same with an 8-robust-SD screen (sensitivity).
    kind="compound_abs50": the original stage 3 screen |δ| > 0.5 (sensitivity).
    kind="compound_raw": the same without any screen.
    kind="elemental": the stage 1 elemental distribution (77 elements) for every class. This is the
        fallback used if the COD calibration is unavailable.
    Returned arrays are copies; draw from them with replacement.
    """
    d, _ = _class_cache(kind, min_n)
    return {k: v.copy() for k, v in d.items()}


def class_delta_info(kind: str = "compound", min_n: int = MIN_CLASS_N) -> dict:
    """Per class: n_class (matches in the class), uses_pooled (fallback applied), n_used."""
    return _class_cache(kind, min_n)[1]


def band(cls: str = "pooled", kind: str = "compound", min_n: int = MIN_CLASS_N) -> tuple[float, float]:
    """90 % interval (p05, p95) of δ for a primary class (or "pooled"), after the pooled fallback."""
    x = _class_cache(kind, min_n)[0][cls]
    return float(np.percentile(x, 5)), float(np.percentile(x, 95))


# =============================================================================================
# 8. funnel, figure, summary, run
# =============================================================================================
def filter_counts() -> dict:
    s = pd.read_parquet(SCREEN_LOG)
    meta = json.loads((INTERIM / "filter_meta.json").read_text(encoding="utf-8"))
    vc = s["reason"].value_counts()
    funnel, remaining = [], len(s)
    for st in STAGES:
        n = int(vc.get(st, 0))
        if st == "kept":
            funnel.append({"stage": "kept after quality filters", "n_remaining": remaining})
        else:
            remaining -= n
            funnel.append({"stage": st, "n_dropped": n, "n_remaining": remaining})
    p = pd.read_parquet(STRUCT_CACHE, columns=["cod_id", "stage", "formula_reduced", "rho_crosscheck",
                                               "title_hp_marker", "r_tag", "r_factor"])
    pvc = p["stage"].value_counts()
    k = pd.read_parquet(FILTERED_CACHE, columns=["cod_id", "r_tag", "temperature_K", "pressure_kPa", "rho_diffrn",
                                                 "cod_error_flag", "title_hp_marker", "title_ht_marker"])
    out = {
        "archive": ARCHIVE.name, "stream_source": meta.get("source"), "n_cif_in_archive": int(len(s)),
        "non_cif_members_by_top_dir": meta.get("non_cif_members_by_top_dir"),
        "n_mp_eligible_reduced_formulas": meta.get("n_mp_formulas"),
        "funnel": funnel,
        "parse": {st: int(pvc.get(st, 0)) for st in PARSE_STAGES},
        "kept_r_tag_counts": {str(a): int(b) for a, b in k["r_tag"].fillna("not given").value_counts().items()},
        "kept_temperature_given": int(k["temperature_K"].notna().sum()),
        "kept_pressure_given": int(k["pressure_kPa"].notna().sum()),
        "kept_cod_warning_flag": int((k["cod_error_flag"].fillna("").str.lower() == "warnings").sum()),
        "kept_title_hp_marker": int(k["title_hp_marker"].sum()),
        "kept_title_ht_marker": int(k["title_ht_marker"].sum()),
        "kept_rho_diffrn_given": int(k["rho_diffrn"].notna().sum()),
        "parsed_ok_distinct_formulas": int(p.loc[p["stage"] == "parsed_ok", "formula_reduced"].nunique()),
        "crosscheck_abs_rel_diff_parsed_ok": stats(p.loc[p["stage"] == "parsed_ok", "rho_crosscheck"].abs().dropna(),
                                                   ci=False),
    }
    mm = INTERIM / "match_meta.json"
    if mm.exists():
        out["matching"] = json.loads(mm.read_text(encoding="utf-8"))
    mc = INTERIM / "mp_structures_check.json"
    if mc.exists():
        x = json.loads(mc.read_text(encoding="utf-8"))
        out["mp_structures"] = {k2: v for k2, v in x.items() if k2 not in ("missing", "density_mismatch_ids")}
    return out


CLASS_LABEL = {"oxide": "oxide", "halide": "halide", "chalcogenide": "chalcogenide", "intermetallic": "intermetallic",
               "lanthanide": "lanthanide-containing", "3d_magnetic": "3d-magnetic", "f_electron": "f-electron (actinide)",
               "other": "other"}


def figure(m: pd.DataFrame | None = None) -> list[str]:
    """Figure 7: δ by compound class (COD) vs the stage 1 elemental distribution; (b) by e_hull bin."""
    plt = common.style()
    m = pd.read_parquet(OUT / "matches.parquet") if m is None else m
    h = headline(m)
    el = _phase1_elemental()
    rows = [("Phase 1 elements (exp. ref.)", el, "el"), (f"COD compounds, all", h["delta"].to_numpy(), "all")]
    for c in common.REPORT_CLASSES:
        rows.append((CLASS_LABEL[c], h.loc[h[f"is_{c}"], "delta"].to_numpy(), "cls"))
    erows = []
    for name, _, _ in EHULL_BINS:
        erows.append((name if name != "0" else "0 (on hull)", h.loc[h["ehull_bin"] == name, "delta"].to_numpy()))

    colors = {"el": "#eb6834", "all": "#2a78d6", "cls": "#2a78d6", "eh": "#2a78d6"}   # dataviz reference slots 1-2
    allx = [100 * np.asarray(r[1], float) for r in rows] + [100 * np.asarray(r[1], float) for r in erows]
    p5s = [np.percentile(x, 5) for x in allx if len(x)]
    p95s = [np.percentile(x, 95) for x in allx if len(x)]
    XL = (float(np.floor(min(p5s) - 1.0)), float(np.ceil(max(p95s) + 1.0)))   # every p5–p95 whisker fits
    fig, axes = plt.subplots(1, 2, figsize=(common.DOUBLE_COL_IN, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})

    def draw(ax, data, labels, kinds):
        ys = np.arange(len(data))[::-1]
        for y, x, k in zip(ys, data, kinds):
            x = 100 * np.asarray(x, float)
            if len(x) == 0:
                continue
            p5, p25, p50, p75, p95 = np.percentile(x, [5, 25, 50, 75, 95])
            col = colors[k]
            ax.plot([p5, p95], [y, y], color=col, lw=1.0, solid_capstyle="butt")
            ax.add_patch(plt.Rectangle((p25, y - 0.28), p75 - p25, 0.56, facecolor=col, alpha=0.35, edgecolor=col,
                                       lw=0.8))
            ax.plot([p50, p50], [y - 0.32, y + 0.32], color=col, lw=1.8)
            ax.text(XL[1] + 0.3, y, f"n={len(x):,}", va="center", ha="left", fontsize=7, color="#333333")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="#999999", lw=0.6, zorder=0)
        ax.set_xlim(*XL)
        ax.set_ylim(-0.7, len(data) - 0.3)
        ax.set_xlabel("δ = ρ_MP/ρ_ref − 1 (%)")
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    draw(axes[0], [r[1] for r in rows], [r[0] for r in rows], [r[2] for r in rows])
    top = len(rows) - 1
    axes[0].axhspan(top - 0.5, top + 0.5, color="#fbe9e1", zorder=-1, lw=0)        # reference row, set apart
    axes[0].axhline(top - 0.5, color="#555555", lw=0.8)
    axes[0].axhline(top - 1.5, color="#cccccc", lw=0.5, ls=":")
    axes[0].set_title("(a) by class; box IQR, whiskers 5th–95th percentile", loc="left")
    draw(axes[1], [r[1] for r in erows], [r[0] for r in erows], ["eh"] * len(erows))
    axes[1].set_ylabel("e_hull bin (eV/atom)")
    axes[1].set_title("(b) COD compounds by e_hull", loc="left")
    n_cls = sum(len(r[1]) for r in rows if r[2] == "cls")
    fig.text(0.01, -0.01, f"Reference: COD (structure-matched compounds, eligible R3, ≥ 2 elements, no outlier "
             f"screen; n = {len(h):,}) or, top row, Phase 1 experimental element densities.\nClasses overlap: a "
             f"compound can belong to several, so the class n sum to {n_cls:,} > {len(h):,}. The axis spans every "
             f"5th–95th percentile whisker; values beyond the whiskers are not drawn.",
             fontsize=7, color="#444444", ha="left", va="top")
    fig.tight_layout(w_pad=3.0)
    out = common.savefig(fig, "fig7_density_bias_by_class")
    plt.close(fig)

    # diagnostic: COD-recomputed vs CIF-declared density, and δ histogram
    p = pd.read_parquet(STRUCT_CACHE, columns=["stage", "rho_crosscheck"])
    fig, ax = plt.subplots(1, 2, figsize=(common.DOUBLE_COL_IN, 2.4))
    x = 100 * p["rho_crosscheck"].dropna().to_numpy()
    ax[0].hist(np.clip(x, -5, 5), bins=np.linspace(-5, 5, 101), color="#2a78d6")
    ax[0].axvline(-100 * RHO_CROSSCHECK_TOL, color="#555555", lw=0.8, ls="--")
    ax[0].axvline(100 * RHO_CROSSCHECK_TOL, color="#555555", lw=0.8, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("ρ_parsed/ρ_diffrn(CIF) − 1 (%) [clipped ±5]")
    ax[0].set_ylabel("COD entries")
    ax[0].set_title("(a) COD density cross-check", loc="left")
    xh = 100 * h["delta"].to_numpy()
    ax[1].hist(np.clip(xh, -15, 15), bins=np.linspace(-15, 15, 121), color="#2a78d6", alpha=0.75,
               label=f"COD compounds (n={len(xh):,})", density=True)
    ax[1].hist(np.clip(100 * el, -15, 15), bins=np.linspace(-15, 15, 61), color="#eb6834", alpha=0.55,
               label=f"Phase 1 elements (n={len(el)})", density=True)
    ax[1].set_xlabel("δ (%) [clipped ±15]")
    ax[1].set_ylabel("density")
    ax[1].legend(frameon=False)
    ax[1].set_title("(b) δ distributions", loc="left")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    out += common.savefig(fig, "D_crosscheck_and_delta_hist")
    plt.close(fig)
    return out


def summary(m: pd.DataFrame, tables: dict, figs: list[str]) -> dict:
    h = headline(m)
    n = len(h)
    use_compound = n >= 1000
    o = tables["overall"]
    s = {
        "step": "Phase 3 Step 3: compound density bias via COD",
        "cod_snapshot": json.loads(SOURCE_JSON.read_text(encoding="utf-8")) if SOURCE_JSON.exists() else None,
        "headline_population": "matched eligible (R3) MP entries with ≥ 2 elements, one COD value per MP entry",
        "n_matched_headline": n,
        "n_matched_R2": int(h["in_R2"].sum()),
        "n_matched_including_unaries": int(len(m)),
        "n_distinct_cod_entries_selected": int(h["cod_id"].nunique()),
        "n_distinct_formulas": int(h["formula_reduced"].nunique()),
        "selection_rules": {str(k): int(v) for k, v in h["selection_rule"].value_counts().items()},
        "overall": o["R3_compounds (headline)"],
        "headline_statement": "median δ with its sampling-only 95 % CI (iid pairs), cluster CIs (by formula, "
                              "by COD entry), and the range of the subset medians (R given/not, T bins, "
                              "metals/non-metals)",
        "elemental_crosscheck": {"n_unary_checked": int(m["xcheck_rel"].notna().sum()),
                                 "n_fail": int(m["xcheck_fail"].sum()),
                                 "failed": m.loc[m["xcheck_fail"], ["material_id", "formula_reduced", "rho_mp",
                                                                    "rho_cod", "xcheck_rel"]].to_dict("records"),
                                 "tolerance": ELEMENT_XCHECK_TOL},
        "overall_R2": o["R2_compounds"],
        "phase1_elemental": o["phase1_elemental"],
        "recommendation": {
            "rule": "compound-based band iff the headline match count ≥ 1,000 (handover Step 3.4)",
            "use": "compound" if use_compound else "elemental",
            "class_delta_distribution_kind": "compound" if use_compound else "elemental",
            "pooled_band_p05_p95": band("pooled", "compound") if n else None,
            "pooled_band_p05_p95_raw": band("pooled", "compound_raw") if n else None,
            "outlier_screen": f"|ln(1+δ) − class median| > {SCREEN_K:g} robust SD excluded from "
                              f"kind='compound' ({int(h['excluded'].sum())} of {n} pairs); 8 robust SD: "
                              f"{int(h['excluded_k8'].sum())}; original |δ| > {GROSS_DELTA}: "
                              f"{int(h['gross_abs50'].sum())}",
            "pooled_band_p05_p95_k8": band("pooled", "compound_k8") if n else None,
            "pooled_band_p05_p95_abs50": band("pooled", "compound_abs50") if n else None,
        },
        "class_bands": {c: band(c, "compound") for c in list(common.PRIMARY_CLASS_ORDER) + ["pooled"]} if n else None,
        "class_info": class_delta_info("compound") if n else None,
        "figures": figs,
    }
    common.write_json(OUT / "summary.json", s)
    return s


def run(workers: int = N_WORKERS) -> dict:
    """Regenerate all COD-module outputs from the cached archive/structures (network only if raw files are missing)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths.ensure(OUT)
    t0 = time.time()
    arch = verify_archive()              # re-hashes the archive bytes (review F-3); absent archive → recorded
    if not (FILTERED_CACHE.exists() and SCREEN_LOG.exists()):
        download()                       # no-op if the verified archive exists; else fills parts, verifies, assembles
    filter_archive(workers)              # cached: checks the cache fingerprint only
    parse_cod(workers)
    m = match(workers)
    fc = filter_counts()
    timing = {"archive_sha256_seconds": arch.pop("_seconds", None) if arch.get("verified") else None}
    if "matching" in fc:
        timing["matching_seconds_last_run"] = fc["matching"].pop("seconds_last_run", None)
    fc["archive_verification"] = {k: v for k, v in arch.items() if not k.startswith("_")}
    fc["matched_mp_entries_including_unaries"] = int(len(m))
    fc["matched_mp_compounds_headline"] = int(len(headline(m)))
    common.write_json(OUT / "filter_counts.json", fc)
    k = pd.read_parquet(STRUCT_CACHE).drop(columns=["structure_json"])
    k.to_parquet(OUT / "cod_filtered.parquet", index=False)
    t = bias_tables(m)
    figs = figure(m)
    s = summary(m, t, figs)
    timing["run_seconds"] = round(time.time() - t0, 1)
    common.write_json(OUT / "_timing.json", timing)
    return s


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if cmd == "download":
        print(download())
    elif cmd == "assemble":
        print(json.dumps(assemble(), indent=1))
    elif cmd == "filter":
        print(len(filter_archive(refresh="--refresh" in sys.argv)))
    elif cmd == "parse":
        print(parse_cod(refresh="--refresh" in sys.argv)["stage"].value_counts())
    elif cmd == "fetch":
        print(fetch_mp_structures()["density_ok"].value_counts())
    elif cmd == "match":
        print(len(match()))
    elif cmd == "all":
        print(json.dumps(run(), indent=1, default=str)[:4000])
    else:
        raise SystemExit(f"unknown command {cmd}")
