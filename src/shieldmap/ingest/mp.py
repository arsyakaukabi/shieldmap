"""Table B — Materials Project compounds (database of record, class P2).

One retrieval of the MP summary collection through the official API (``mp-api`` MPRester), stored
verbatim as gzip JSONL under ``data/raw/mp/``, then turned into Table B (``config/DATA_CONTRACT.md``).

Raw snapshot (data/raw/mp/):
  summary_<db>.jsonl.gz          one MP summary document per line, exactly as returned (deprecated=False)
  summary_<db>.meta.json         database version, UTC timestamps, query, fields, counts, package versions
  thermo_check_<db>.json         empirical check of the thermo scheme behind summary.energy_above_hull

Interim outputs (data/interim/mp/):
  compounds.parquet              Table B, one row per non-deprecated entry, deterministic order
  composition.parquet            long table: material_id, Z, symbol, mass_fraction, atom_fraction
  mass_fractions.npz             scipy.sparse CSR float64 (N x 100), column = Z - 1, rows = compounds order
  mass_fractions_index.parquet   row, material_id

Material identifiers: since emmet-core 0.8x the API returns IDs in MP's "AlphaID" form
(``mp-aaaaaaft`` = mp-149). ``material_id`` holds MP's canonical display string
(``emmet.core.mpid.AlphaID(id).string``, i.e. ``mp-149`` for every legacy ID); the string exactly as
returned by the API is kept in ``material_id_api``; ``id_number`` is the integer value.

CLI: ``python -m shieldmap.ingest.mp [--refresh]``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.ingest import fetch as _fetch

log = logging.getLogger(__name__)

SOURCE_ID = "materials_project"
RAW = paths.RAW / "mp"
OUT = paths.INTERIM / "mp"
ELEMENTS = paths.INTERIM / "elements" / "elements.parquet"
Z_MAX = 100

FIELDS = [
    "material_id", "formula_pretty", "chemsys", "composition", "composition_reduced", "elements",
    "nelements", "density", "volume", "nsites", "energy_above_hull", "formation_energy_per_atom",
    "is_stable", "theoretical", "database_IDs", "band_gap", "is_gap_direct", "is_metal", "symmetry",
    "deprecated",
]
QUERY = {"deprecated": False}          # the only filter; include_gnome left at the client default (True)
CHUNK_SIZE = 1000
THERMO_TYPES = ["GGA_GGA+U_R2SCAN", "GGA_GGA+U", "R2SCAN"]
THERMO_SAMPLE_N = 200
THERMO_SAMPLE_SEED = 20261011
THERMO_TOL = 1e-6                       # eV/atom: identical numbers expected if the scheme matches
RETRIES = 4


# --------------------------------------------------------------------------------------------- utils

def _api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv(paths.REPO_ROOT / ".env", override=False)   # existing environment variables win
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY not set (environment or ../.env)")
    return key


def _retry(func, what: str, retries: int = RETRIES):
    last = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:  # transient network/server failures are retried, then re-raised
            last = e
            wait = 15 * 2 ** attempt
            log.warning("%s failed (attempt %d/%d): %s; retrying in %ds",
                        what, attempt + 1, retries, type(e).__name__, wait)
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries} attempts: {last}")


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(paths.ROOT).as_posix()


def parse_id(material_id: str) -> tuple[str, int, str]:
    """(prefix, integer value, canonical MP display string) of an MP identifier (legacy or AlphaID)."""
    from emmet.core.mpid import AlphaID
    a = AlphaID(str(material_id))
    return a._prefix, int(a), a.string


def existing_snapshot() -> Path | None:
    snaps = sorted(RAW.glob("summary_*.jsonl.gz"))
    return snaps[-1] if snaps else None


def _db_version_of(snapshot: Path) -> str:
    return snapshot.name[len("summary_"):-len(".jsonl.gz")]


def read_raw(snapshot: Path | None = None) -> list[dict]:
    snapshot = snapshot or existing_snapshot()
    if snapshot is None:
        raise FileNotFoundError("no MP snapshot in data/raw/mp/; run fetch() first")
    with gzip.open(snapshot, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------------------------- fetch

def fetch(refresh: bool = False) -> Path:
    """Retrieve the MP summary snapshot once (official API); reuse the cached file unless ``refresh``."""
    snap = existing_snapshot()
    if snap is not None and not refresh:
        db = _db_version_of(snap)
        if not (RAW / f"thermo_check_{db}.json").exists():
            _thermo_check(db, read_raw(snap))
        return snap

    from mp_api.client import MPRester

    RAW.mkdir(parents=True, exist_ok=True)
    with MPRester(_api_key(), use_document_model=False, mute_progress_bars=True) as mpr:
        db = _retry(lambda: str(getattr(mpr, "db_version", None) or mpr.get_database_version()),
                    "get database version")
        started = _fetch.utc_now()
        n_expected = _retry(lambda: int(mpr.materials.summary.count(dict(QUERY))), "summary count")
        try:
            n_deprecated = _retry(lambda: int(mpr.materials.summary.count({"deprecated": True})),
                                  "deprecated count")
        except Exception:  # optional information only
            n_deprecated = None
        log.info("MP %s: %d non-deprecated summary docs expected", db, n_expected)
        docs = _retry(lambda: mpr.materials.summary.search(**QUERY, fields=FIELDS,
                                                           chunk_size=CHUNK_SIZE),
                      "summary search")
        finished = _fetch.utc_now()
        db_after = str(getattr(mpr, "db_version", None) or mpr.get_database_version())

    if db_after != db:
        raise RuntimeError(f"MP database version changed during retrieval: {db} -> {db_after}")
    ids = [d["material_id"] for d in docs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate material_id in retrieved MP documents")
    if len(docs) != n_expected:
        raise RuntimeError(f"retrieved {len(docs)} docs, count query said {n_expected}")
    if any(d.get("deprecated") is not False for d in docs):
        raise RuntimeError("deprecated document returned despite deprecated=False filter")

    dest = RAW / f"summary_{db}.jsonl.gz"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with gzip.GzipFile(tmp, "wb", mtime=0) as gz:          # mtime=0: byte-reproducible gzip header
        for d in docs:
            gz.write((json.dumps(d, default=str, sort_keys=True, ensure_ascii=False) + "\n").encode())
    tmp.replace(dest)

    meta = {
        "source": "Materials Project",
        "endpoint": "https://api.materialsproject.org/materials/summary/",
        "client_call": "MPRester(use_document_model=False).materials.summary.search",
        "database_version": db,
        "retrieval_started_utc": started,
        "retrieval_finished_utc": finished,
        "query": dict(QUERY, include_gnome=True),
        "fields": FIELDS,
        "chunk_size": CHUNK_SIZE,
        "n_docs": len(docs),
        "n_docs_count_query": n_expected,
        "n_deprecated_count_query": n_deprecated,
        "packages": {p: version(p) for p in ("mp-api", "emmet-core", "pymatgen")},
        "file": dest.name,
        "sha256": _fetch.sha256(dest),
    }
    (RAW / f"summary_{db}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _thermo_check(db, docs)
    return dest


def _thermo_check(db: str, docs: list[dict]) -> dict:
    """Compare summary.energy_above_hull with the thermo collection for a fixed random sample."""
    from mp_api.client import MPRester

    rng = np.random.default_rng(THERMO_SAMPLE_SEED)
    ids_sorted = sorted(d["material_id"] for d in docs)
    sample = sorted(rng.choice(ids_sorted, size=min(THERMO_SAMPLE_N, len(ids_sorted)),
                               replace=False).tolist())
    summary_eh = {d["material_id"]: d.get("energy_above_hull") for d in docs}
    result = {"database_version": db, "retrieved_utc": _fetch.utc_now(),
              "sample_seed": THERMO_SAMPLE_SEED, "sample_n": len(sample), "tolerance_eV_atom": THERMO_TOL,
              "sample_ids": sample, "by_thermo_type": {}}
    with MPRester(_api_key(), use_document_model=False, mute_progress_bars=True) as mpr:
        for tt in THERMO_TYPES:
            rows = _retry(lambda: mpr.materials.thermo.search(
                material_ids=sample, thermo_types=[tt],
                fields=["material_id", "thermo_type", "energy_above_hull"]), f"thermo {tt}")
            got = {}
            for r in rows:
                got[str(r["material_id"])] = r.get("energy_above_hull")
            compared = agree = 0
            max_abs = 0.0
            for mid in sample:
                a, b = summary_eh.get(mid), got.get(mid)
                if a is None or b is None:
                    continue
                compared += 1
                diff = abs(float(a) - float(b))
                max_abs = max(max_abs, diff)
                agree += diff <= THERMO_TOL
            result["by_thermo_type"][tt] = {
                "n_returned": len(rows), "n_compared": compared, "n_agree": agree,
                "agree_fraction": (agree / compared) if compared else None,
                "max_abs_diff_eV_atom": max_abs if compared else None,
                "values": {str(r["material_id"]): r.get("energy_above_hull") for r in rows},
            }
    best = max(THERMO_TYPES, key=lambda t: (result["by_thermo_type"][t]["n_agree"],
                                            -THERMO_TYPES.index(t)))
    result["inferred_scheme"] = best
    p = RAW / f"thermo_check_{db}.json"
    p.write_text(json.dumps(result, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------------- build

def _elements_table() -> pd.DataFrame:
    if not ELEMENTS.exists():
        raise FileNotFoundError(f"{ELEMENTS} missing; build Table A first")
    return pd.read_parquet(ELEMENTS).set_index("symbol")


def build(refresh: bool = False) -> pd.DataFrame:
    import scipy.sparse as sp
    from pymatgen.core import Composition, Element

    snap = fetch(refresh=refresh)
    db = _db_version_of(snap)
    meta = json.loads((RAW / f"summary_{db}.meta.json").read_text(encoding="utf-8"))
    thermo = json.loads((RAW / f"thermo_check_{db}.json").read_text(encoding="utf-8"))
    docs = read_raw(snap)
    if meta["sha256"] != _fetch.sha256(snap):
        raise ValueError(f"checksum mismatch for {snap} against its meta record")
    _verify_against_source_record(snap)

    el = _elements_table()
    mass_of: dict[str, float] = {}
    z_of: dict[str, int] = {}

    rows, comp_rows = [], []
    formula_mismatch = []
    # pymatgen warns that noble gases have no Pauling electronegativity (used only to order symbols
    # in formula strings); the warning carries no data issue and is silenced here only.
    warnings.filterwarnings("ignore", message="No Pauling electronegativity")
    for d in docs:
        prefix, num, mid = parse_id(d["material_id"])
        comp = Composition(d["composition"])
        syms = []
        for e in comp.elements:
            if not isinstance(e, Element):
                raise ValueError(f"{mid}: non-element species {e!r} in composition")
            if e.Z > Z_MAX:
                raise ValueError(f"{mid}: element {e.symbol} (Z={e.Z}) beyond Z={Z_MAX}")
            syms.append(e.symbol)
            mass_of.setdefault(e.symbol, float(e.atomic_mass))
            z_of.setdefault(e.symbol, int(e.Z))
        n_atoms = comp.num_atoms
        wsum = 0.0
        for s in syms:
            w = float(comp.get_wt_fraction(s))
            wsum += w
            comp_rows.append((mid, z_of[s], s, w, float(comp[s]) / n_atoms))
        if abs(wsum - 1.0) >= 1e-9:
            raise AssertionError(f"{mid}: sum of mass fractions = {wsum!r}")
        reduced = comp.reduced_formula
        if Composition(d["composition_reduced"]).reduced_formula != reduced:
            formula_mismatch.append(mid)
        if sorted(syms) != sorted(d.get("elements") or []):
            raise ValueError(f"{mid}: composition elements {syms} != MP elements {d.get('elements')}")
        sym = d.get("symmetry") or {}
        icsd = list((d.get("database_IDs") or {}).get("icsd") or [])
        rows.append({
            "material_id": mid,
            "material_id_api": str(d["material_id"]),
            "id_prefix": prefix,
            "id_number": num,
            "formula_pretty": d.get("formula_pretty"),
            "formula_reduced": reduced,
            "chemsys": d.get("chemsys"),
            "nelements": d.get("nelements"),
            "nsites": d.get("nsites"),
            "density": d.get("density"),
            "volume": d.get("volume"),
            "energy_above_hull": d.get("energy_above_hull"),
            "formation_energy_per_atom": d.get("formation_energy_per_atom"),
            "is_stable": d.get("is_stable"),
            "theoretical": d.get("theoretical"),
            "icsd_ids": [str(x) for x in icsd],
            "n_icsd_ids": len(icsd),
            "band_gap": d.get("band_gap"),
            "is_gap_direct": d.get("is_gap_direct"),
            "is_metal": d.get("is_metal"),
            "spacegroup_number": sym.get("number"),
            "crystal_system": None if sym.get("crystal_system") is None else str(sym["crystal_system"]),
            "deprecated": d.get("deprecated"),
            "elements": sorted(syms, key=lambda s: z_of[s]),
        })
    if formula_mismatch:
        raise ValueError(f"{len(formula_mismatch)} entries where composition and composition_reduced "
                         f"give different reduced formulas, e.g. {formula_mismatch[:5]}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["id_prefix", "id_number"], kind="mergesort").reset_index(drop=True)
    if not df["material_id"].is_unique:
        raise ValueError("material_id not unique after canonicalisation")
    if df["deprecated"].ne(False).any():
        raise ValueError("deprecated entries present")

    for c in ("density", "volume", "energy_above_hull", "formation_energy_per_atom", "band_gap"):
        df[c] = df[c].astype("float64")
    for c in ("nelements", "nsites", "spacegroup_number"):
        df[c] = df[c].astype("Int64")
    df["id_number"] = df["id_number"].astype("int64")
    df["n_icsd_ids"] = df["n_icsd_ids"].astype("int64")
    for c in ("is_stable", "theoretical", "is_gap_direct", "is_metal", "deprecated"):
        df[c] = df[c].astype("boolean")

    # is_representative: per formula_reduced, lowest e_hull (nulls last), ties -> lowest (prefix, id)
    order = df.sort_values(["formula_reduced", "energy_above_hull", "id_prefix", "id_number"],
                           na_position="last", kind="mergesort")
    rep_idx = order.drop_duplicates("formula_reduced", keep="first").index
    df["is_representative"] = False
    df.loc[rep_idx, "is_representative"] = True

    # element flags and abundance proxies from Table A
    present = sorted(z_of, key=lambda s: z_of[s])
    elsets = df["elements"].map(set)
    flags = {f"contains_{s}": elsets.map(lambda es, s=s: s in es).astype(bool) for s in present}
    radio = {s for s in present if bool(el.loc[s, "is_radioactive"])}
    noble = {s for s in present if bool(el.loc[s, "is_noble_gas"])}
    flags["contains_radioactive"] = elsets.map(lambda es: bool(es & radio)).astype(bool)
    flags["contains_noble_gas"] = elsets.map(lambda es: bool(es & noble)).astype(bool)

    comp_df = pd.DataFrame(comp_rows, columns=["material_id", "Z", "symbol", "mass_fraction",
                                               "atom_fraction"])
    comp_df["Z"] = comp_df["Z"].astype("int64")
    abund = el["crustal_abundance_ppm"]
    comp_df["_ab"] = comp_df["symbol"].map(abund).astype("float64")
    g = comp_df.groupby("material_id", sort=False)
    any_missing = g["_ab"].apply(lambda s: bool(s.isna().any()))
    min_ab = g["_ab"].min()
    wlog = (comp_df["mass_fraction"] * np.log10(comp_df["_ab"])).groupby(comp_df["material_id"]).sum()
    min_ab[any_missing] = np.nan
    wlog[any_missing] = np.nan
    flags["min_crustal_abundance_ppm"] = df["material_id"].map(min_ab).astype("float64")
    flags["abundance_weighted_log_ppm"] = df["material_id"].map(wlog).astype("float64")
    df = pd.concat([df, pd.DataFrame(flags)], axis=1)
    comp_df = comp_df.drop(columns="_ab")

    # long composition table and sparse matrix, both in compounds order
    row_of = pd.Series(np.arange(len(df)), index=df["material_id"])
    comp_df["_row"] = comp_df["material_id"].map(row_of).astype("int64")
    comp_df = comp_df.sort_values(["_row", "Z"], kind="mergesort").reset_index(drop=True)
    W = sp.csr_matrix((comp_df["mass_fraction"].to_numpy(np.float64),
                       (comp_df["_row"].to_numpy(), comp_df["Z"].to_numpy() - 1)),
                      shape=(len(df), Z_MAX), dtype=np.float64)
    W.sort_indices()
    if not np.all(np.abs(np.asarray(W.sum(axis=1)).ravel() - 1.0) < 1e-9):
        raise AssertionError("sparse mass-fraction rows do not sum to 1")
    comp_df = comp_df.drop(columns="_row")

    cols = ["material_id", "material_id_api", "id_prefix", "id_number", "formula_pretty",
            "formula_reduced", "chemsys", "nelements", "nsites", "density", "volume",
            "energy_above_hull", "formation_energy_per_atom", "is_stable", "theoretical", "icsd_ids",
            "n_icsd_ids", "band_gap", "is_gap_direct", "is_metal", "spacegroup_number",
            "crystal_system", "deprecated", "is_representative", "elements"]
    df = df[cols + [c for c in df.columns if c not in cols]]

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "compounds.parquet", index=False)
    comp_df.to_parquet(OUT / "composition.parquet", index=False)
    sp.save_npz(OUT / "mass_fractions.npz", W, compressed=True)
    pd.DataFrame({"row": np.arange(len(df), dtype=np.int64),
                  "material_id": df["material_id"]}).to_parquet(OUT / "mass_fractions_index.parquet",
                                                                index=False)
    write_source_record(snap, meta, thermo)
    return df


def _raw_files(snap: Path) -> list[Path]:
    db = _db_version_of(snap)
    return [snap, RAW / f"summary_{db}.meta.json", RAW / f"thermo_check_{db}.json"]


def _verify_against_source_record(snap: Path) -> None:
    """A cached raw file must match the SHA-256 recorded in config/sources (never accepted silently)."""
    rec = _fetch.read_source(SOURCE_ID)
    if not rec:
        return
    recorded = {f["path"]: f["sha256"] for f in rec.get("files", [])}
    for p in _raw_files(snap):
        rel = _rel(p)
        if rel in recorded and recorded[rel] != _fetch.sha256(p):
            raise ValueError(f"checksum mismatch for {rel} against config/sources/{SOURCE_ID}.yaml")


def write_source_record(snap: Path, meta: dict, thermo: dict) -> Path:
    db = meta["database_version"]
    scheme = thermo["inferred_scheme"]
    ev = {t: {k: v for k, v in r.items() if k != "values"} for t, r in thermo["by_thermo_type"].items()}
    return _fetch.write_source(SOURCE_ID, {
        "title": f"Materials Project summary collection, database version {db}",
        "citation": ("A. Jain, S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, "
                     "D. Gunter, D. Skinner, G. Ceder, K. A. Persson, \"Commentary: The Materials "
                     "Project: A materials genome approach to accelerating materials innovation\", "
                     f"APL Mater. 1, 011002 (2013), doi:10.1063/1.4812323. Database version {db}."),
        "data_class": "P2",
        "retrieval_method": "official API (mp-api MPRester.materials.summary.search)",
        "url": "https://api.materialsproject.org",
        "database_version": db,
        "retrieved_utc": meta["retrieval_finished_utc"],
        "retrieval_started_utc": meta["retrieval_started_utc"],
        "packages": meta["packages"],
        "query": {"filter": meta["query"], "fields": meta["fields"], "chunk_size": meta["chunk_size"]},
        "n_docs": meta["n_docs"],
        "n_deprecated_count_query": meta.get("n_deprecated_count_query"),
        "thermo_scheme": scheme,
        "thermo_scheme_evidence": {
            "method": ("summary.energy_above_hull compared with materials.thermo.search(material_ids, "
                       "thermo_types=[t]) for a fixed random sample of material_ids"),
            "sample_seed": thermo["sample_seed"], "sample_n": thermo["sample_n"],
            "tolerance_eV_atom": thermo["tolerance_eV_atom"],
            "checked_utc": thermo["retrieved_utc"],
            "by_thermo_type": ev,
        },
        "material_id_note": ("API returns AlphaID strings (mp-aaaaaaft); material_id in Table B is "
                             "emmet.core.mpid.AlphaID(id).string (mp-149); raw string kept in "
                             "material_id_api."),
        "licence": "CC BY 4.0 (Materials Project data)",
        "files": [{"path": _rel(p), "sha256": _fetch.sha256(p), "bytes": p.stat().st_size}
                  for p in _raw_files(snap)],
    })


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Materials Project ingest -> Table B")
    ap.add_argument("--refresh", action="store_true", help="re-query the API even if a snapshot exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build(refresh=args.refresh)
    print(f"Table B: {len(df)} rows, {int(df['is_representative'].sum())} representative -> "
          f"{_rel(OUT / 'compounds.parquet')}")


if __name__ == "__main__":
    main()
