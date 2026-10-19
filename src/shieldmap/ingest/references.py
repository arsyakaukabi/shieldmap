"""Standards and reference data: ENSDF decay lines, IEC 61267 RQR qualities, NIST SRD 126 tables.

Every value here is read from a file retrieved once from its publisher and archived under
``data/raw/<source>/``. The SHA-256 of each file is recorded in ``config/sources/<id>.yaml``.
No value is typed in by hand. The only literals are the expected RQR values, which are used as
an assertion against the parsed PDF text, and loose plausibility windows for the decay data.

Sources and outputs
-------------------
* NNDC ENSDF bulk distribution (``ensdf_261001.zip``). The 137Cs and 60Co beta-minus decay
  datasets are parsed in the ENSDF 80-column format → ``config/sources/ensdf_cs137.yaml`` and
  ``config/sources/ensdf_co60.yaml`` (``data.lines_used`` feeds ``radionuclides.*``).
* IAEA TRS-457 (2007) PDF, Table 6.2 (IEC 61267 RQR series) → ``config/sources/rqr_trs457.yaml``
  (``data.used`` feeds ``xray_tubes.qualities``).
* NIST SRD 126 (Hubbell & Seltzer) published HTML tables →
  ``config/sources/nist_srd126.yaml`` (``data.air`` feeds ``response.air_muen``),
  ``data/interim/references/air_muen.parquet``, ``tests/fixtures/srd126_mu_rho.csv``,
  ``tests/fixtures/srd126_materials.csv``.

Run ``python -m shieldmap.ingest.references``. The build is idempotent: cached raw files are
reused and verified against the recorded checksums.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
import zipfile
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from shieldmap import paths
from shieldmap.ingest import fetch

# --------------------------------------------------------------------------------------------
# common helpers
# --------------------------------------------------------------------------------------------


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(paths.ROOT).as_posix()


def _mtime_utc(p: Path) -> str:
    t = _dt.datetime.fromtimestamp(Path(p).stat().st_mtime, tz=_dt.timezone.utc)
    return t.replace(microsecond=0).isoformat()


def _complete(rec: dict, dest: Path) -> dict:
    """Make sure a file record carries a retrieval time.

    A file that already exists but has no source record yet (the first build after a manual
    retrieval) gets its retrieval time from the file's modification time. That is when it was
    written by :func:`fetch.download`.
    """
    if "retrieved_utc" not in rec:
        rec["retrieved_utc"] = _mtime_utc(dest)
        rec["retrieved_utc_from"] = "file mtime (no prior record)"
    return rec


def _provenance_file(source_id: str, dest: Path, url: str, note: str) -> dict | None:
    """Record for a file kept only as provenance evidence. The build never re-fetches it."""
    rel = _rel(dest)
    prior = fetch.read_source(source_id) or {}
    for f in prior.get("provenance_files", []):
        if f.get("path") == rel:
            return f
    if not dest.exists():
        return None
    return {"path": rel, "url": url, "sha256": fetch.sha256(dest), "bytes": dest.stat().st_size,
            "retrieved_utc": _mtime_utc(dest), "note": note}


def _f(x: Decimal | None) -> float | None:
    return None if x is None else float(str(x))


# --------------------------------------------------------------------------------------------
# 1. ENSDF decay data
# --------------------------------------------------------------------------------------------

ENSDF_URL = "https://www.nndc.bnl.gov/ensdfarchivals/distributions/dist26/ensdf_261001.zip"
ENSDF_FILES_JSON = "https://www.nndc.bnl.gov/ensdfarchivals/distributions/files.json"
ENSDF_ZIP = paths.RAW / "ensdf" / "ensdf_261001.zip"
ENSDF_DOI = "10.18139/nndc.ensdf/1845010"

# The decay lines named (nominal energies). ENSDF's own energies are used and
# a mismatch is flagged.
NUCLIDES = {
    "cs137": {"label": "Cs-137", "parent": "137CS", "daughter": "137BA", "member": "ensdf.137",
              "nominal_keV": [661.657],
              # plausibility only (never used as values)
              "t_half_years": (29.0, 32.0),
              "plausible_intensity": [(0.80, 0.90)]},
    "co60": {"label": "Co-60", "parent": "60CO", "daughter": "60NI", "member": "ensdf.060",
             "nominal_keV": [1173.228, 1332.492],
             "t_half_years": (5.2, 5.35),
             "plausible_intensity": [(0.99, 1.0), (0.99, 1.0)]},
}
_UNC_FLAGS = {"LT", "GT", "LE", "GE", "AP", "CA", "SY"}
_NUM = re.compile(r"^([+-]?)(\d*)(?:\.(\d*))?(?:[Ee]([+-]?\d+))?$")
_T_UNITS_Y = {"Y": Decimal(1), "D": Decimal(1) / Decimal("365.2422"),
              "H": Decimal(1) / Decimal("8765.8128"), "M": Decimal(1) / Decimal("525948.768"),
              "S": Decimal(1) / Decimal("31556925.97")}


def _col(line: str, a: int, b: int) -> str:
    """Columns a..b (1-indexed, inclusive) of an 80-column ENSDF record, stripped."""
    return line.ljust(80)[a - 1:b].strip()


def ensdf_value(s: str) -> Decimal | None:
    s = s.strip()
    if not s or not _NUM.match(s) or s in {"+", "-", "."}:
        return None
    return Decimal(s)


def ensdf_unc(value: str, unc: str) -> tuple[Decimal | None, str | None]:
    """ENSDF uncertainty: an integer in units of the last digit of ``value``.

    Returns (absolute uncertainty, flag). Non-numeric entries (LT, GT, AP, CA, SY, asymmetric
    ``+a-b``) give ``(None, raw)``.
    """
    value, unc = value.strip(), unc.strip()
    if not unc:
        return None, None
    if unc in _UNC_FLAGS or not unc.isdigit():
        return None, unc
    m = _NUM.match(value)
    if not m:
        return None, unc
    decimals = len(m.group(3) or "")
    exp = int(m.group(4) or 0)
    return Decimal(int(unc)).scaleb(exp - decimals), None


def _record_type(line: str) -> tuple[str, str, str]:
    """(continuation char col 6, comment/parent char col 7, record type col 8)."""
    line = line.ljust(80)
    return line[5], line[6], line[7]


def split_datasets(text: str) -> list[list[str]]:
    """Split an ENSDF file into datasets. Each dataset ends with a blank END record."""
    out, cur = [], []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(line.rstrip("\r\n"))
    if cur:
        out.append(cur)
    return out


def _parse_history(ds: list[str]) -> list[dict]:
    hist, buf = [], None
    for line in ds:
        cont, c7, rt = _record_type(line)
        if rt == "H" and c7 == " ":
            body = line.ljust(80)[9:80].rstrip()
            if cont == " ":
                if buf is not None:
                    hist.append(buf)
                buf = body
            else:
                buf = (buf or "") + body
    if buf is not None:
        hist.append(buf)
    recs = []
    for h in hist:
        d = {}
        for part in h.split("$"):
            if "=" in part:
                k, _, v = part.partition("=")
                d[k.strip()] = v.strip()
        recs.append(d)
    return recs


def _parse_parent(line: str) -> dict:
    t_str, t_unc = _col(line, 40, 49), _col(line, 50, 55)
    m = re.match(r"^([\d.Ee+-]+)\s*([A-Z]+)$", t_str)
    t_years = None
    if m and m.group(2) in _T_UNITS_Y:
        t_years = Decimal(m.group(1)) * _T_UNITS_Y[m.group(2)]
    return {"nucid": _col(line, 1, 5), "level_energy_keV": _col(line, 10, 19),
            "level_energy_unc": _col(line, 20, 21), "jpi": _col(line, 22, 39),
            "half_life": t_str, "half_life_unc": t_unc,
            "half_life_years": None if t_years is None else float(f"{t_years:.6g}"),
            "q_keV": _col(line, 65, 74), "q_unc": _col(line, 75, 76), "record": line.rstrip()}


def _parse_normalization(ds: list[str]) -> dict:
    n_line = pn_line = None
    for line in ds:
        cont, c7, rt = _record_type(line)
        if cont == " " and rt == "N" and c7 == " ":
            if n_line is not None:
                raise ValueError("more than one N record in dataset (multiple parents?)")
            n_line = line
        elif cont == " " and rt == "N" and c7 == "P":
            pn_line = line
    if n_line is None:
        raise ValueError("dataset has no N (normalization) record")
    fields = {}
    for name, (a, b), (ua, ub) in [("NR", (10, 19), (20, 21)), ("NT", (22, 29), (30, 31)),
                                   ("BR", (32, 39), (40, 41)), ("NB", (42, 49), (50, 55)),
                                   ("NP", (56, 62), (63, 64))]:
        v, u = _col(n_line, a, b), _col(n_line, ua, ub)
        unc, flag = ensdf_unc(v, u)
        fields[name] = {"value": v, "unc": u, "value_num": ensdf_value(v), "unc_num": unc,
                        "unc_flag": flag}
    pn = None
    if pn_line is not None:
        pn = {}
        for name, (a, b), (ua, ub) in [("NRBR", (10, 19), (20, 21)), ("NTBR", (22, 29), (30, 31)),
                                       ("NBBR", (42, 49), (50, 55)), ("NP", (56, 62), (63, 64))]:
            v, u = _col(pn_line, a, b), _col(pn_line, ua, ub)
            unc, flag = ensdf_unc(v, u)
            pn[name] = {"value": v, "unc": u, "value_num": ensdf_value(v), "unc_num": unc,
                        "unc_flag": flag}
        pn["option"] = _col(pn_line, 78, 78)
        pn["comment_flag"] = _col(pn_line, 77, 77)
    return {"N": fields, "PN": pn, "N_record": n_line.rstrip(),
            "PN_record": None if pn_line is None else pn_line.rstrip()}


def _gamma_multiplier(norm: dict) -> tuple[Decimal, Decimal, str]:
    """Factor converting RI to photons per 100 decays of the parent, its absolute uncertainty,
    and a description of the rule used.

    ENSDF rule: %Iγ = RI × NR × BR. NR converts RI to per-100-decays-of-the-branch and BR converts
    to per-100-parent-decays. If the PN record gives NRBR explicitly, it is used instead.
    Uncertainties are combined in quadrature (relative), ignoring correlations.
    """
    pn = norm["PN"]
    if pn and pn["NRBR"]["value_num"] is not None:
        f = pn["NRBR"]["value_num"]
        u = pn["NRBR"]["unc_num"] or Decimal(0)
        if pn["NRBR"]["unc_flag"]:
            raise ValueError(f"non-numeric NRBR uncertainty {pn['NRBR']['unc_flag']!r}")
        return f, u, "PN NRBR"
    nr, br = norm["N"]["NR"], norm["N"]["BR"]
    if nr["value_num"] is None:
        raise ValueError("NR is blank: absolute gamma intensities are not defined")
    br_v = br["value_num"] if br["value_num"] is not None else Decimal(1)
    for k in (nr, br):
        if k["unc_flag"]:
            raise ValueError(f"non-numeric normalization uncertainty {k['unc_flag']!r}")
    f = nr["value_num"] * br_v
    rel2 = sum(((k["unc_num"] or 0) / k["value_num"]) ** 2
               for k in (nr, br) if k["value_num"] not in (None, 0))
    u = (f * Decimal(rel2).sqrt()) if rel2 else Decimal(0)
    return f, u, "N NR x BR" + ("" if br["value_num"] is not None else " (BR blank = 1)")


def _parse_gammas(ds: list[str], mult: Decimal, mult_unc: Decimal) -> list[dict]:
    out = []
    for line in ds:
        cont, c7, rt = _record_type(line)
        if not (cont == " " and c7 == " " and rt == "G"):
            continue
        e_s, de_s = _col(line, 10, 19), _col(line, 20, 21)
        ri_s, dri_s = _col(line, 22, 29), _col(line, 30, 31)
        e, de = ensdf_value(e_s), ensdf_unc(e_s, de_s)
        ri, dri = ensdf_value(ri_s), ensdf_unc(ri_s, dri_s)
        if e is None:
            raise ValueError(f"non-numeric gamma energy in record {line!r}")
        inten = inten_unc = None
        if ri is not None:
            inten = ri * mult / 100                                         # per decay
            if dri[0] is not None:
                if mult_unc == 0:
                    inten_unc = dri[0] * mult / 100
                else:
                    rel = ((dri[0] / ri) ** 2 + (mult_unc / mult) ** 2).sqrt()
                    inten_unc = Decimal(f"{inten * rel:.3g}")
        out.append({
            "energy_keV": _f(e), "energy_unc_keV": _f(de[0]), "energy_unc_flag": de[1],
            "ri": _f(ri), "ri_unc": _f(dri[0]), "ri_unc_flag": dri[1],
            "multipolarity": _col(line, 32, 41) or None,
            "intensity_per_decay": _f(inten), "intensity_unc": _f(inten_unc),
            "record": line.rstrip(),
        })
    return out


def find_decay_dataset(text: str, parent: str, daughter: str) -> list[str]:
    """The single β⁻-decay dataset of the parent's ground state (parent level energy 0)."""
    hits = []
    for ds in split_datasets(text):
        head = ds[0].ljust(80)
        if head[0:5].strip() != daughter or head[5:9].strip():
            continue
        dsid = _col(head, 10, 39)
        if not dsid.startswith(f"{parent} B- DECAY"):
            continue
        p_lines = [l for l in ds if _record_type(l)[0] == " " and _record_type(l)[1] == " "
                   and _record_type(l)[2] == "P"]
        if len(p_lines) != 1:
            continue
        if ensdf_value(_col(p_lines[0], 10, 19)) == 0:
            hits.append(ds)
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {parent} B- decay dataset (ground-state parent) "
                         f"for daughter {daughter}, found {len(hits)}")
    return hits[0]


def parse_decay(nuc: str, zip_path: Path = ENSDF_ZIP) -> dict:
    spec = NUCLIDES[nuc]
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(spec["member"]).decode("latin-1")
    ds = find_decay_dataset(text, spec["parent"], spec["daughter"])
    head = ds[0]
    p_line = next(l for l in ds if _record_type(l)[0] == " " and _record_type(l)[1] == " "
                  and _record_type(l)[2] == "P")
    parent = _parse_parent(p_line)
    norm = _parse_normalization(ds)
    mult, mult_unc, rule = _gamma_multiplier(norm)
    gammas = _parse_gammas(ds, mult, mult_unc)

    lo, hi = spec["t_half_years"]
    if parent["half_life_years"] is None or not lo <= parent["half_life_years"] <= hi:
        raise ValueError(f"{nuc}: parent half-life {parent['half_life']} implausible")

    used, used_idx = [], set()
    for nominal, (plo, phi) in zip(spec["nominal_keV"], spec["plausible_intensity"]):
        cand = [i for i, g in enumerate(gammas) if abs(g["energy_keV"] - nominal) < 0.5]
        if len(cand) != 1:
            raise ValueError(f"{nuc}: expected one G record near {nominal} keV, found {len(cand)}")
        g = gammas[cand[0]]
        if g["intensity_per_decay"] is None or g["intensity_unc"] is None:
            raise ValueError(f"{nuc}: line {g['energy_keV']} keV lacks a numeric intensity")
        if not plo <= g["intensity_per_decay"] <= phi:
            raise ValueError(f"{nuc}: {g['energy_keV']} keV intensity {g['intensity_per_decay']} "
                             f"outside plausibility window [{plo}, {phi}]")
        used_idx.add(cand[0])
        g["energy_matches_nominal"] = g["energy_keV"] == nominal
        used.append({"energy_keV": g["energy_keV"], "energy_unc_keV": g["energy_unc_keV"],
                     "intensity_per_decay": g["intensity_per_decay"],
                     "intensity_unc": g["intensity_unc"]})
    for i, g in enumerate(gammas):
        g["used"] = i in used_idx

    others = [g for i, g in enumerate(gammas) if i not in used_idx]
    other_sum = sum((Decimal(str(g["intensity_per_decay"])) for g in others
                     if g["intensity_per_decay"] is not None), Decimal(0))
    other_unc2 = sum((Decimal(str(g["intensity_unc"])) ** 2 for g in others
                      if g["intensity_unc"] is not None), Decimal(0))
    used_sum = sum((Decimal(str(u["intensity_per_decay"])) for u in used), Decimal(0))

    history = _parse_history(ds)
    full = [h for h in history if h.get("TYP") == "FUL"]
    def _clean(d: dict) -> dict:
        return {k: (_f(v) if isinstance(v, Decimal) else v) for k, v in d.items()}
    norm_out = {"N": {k: _clean(v) for k, v in norm["N"].items()},
                "PN": None if norm["PN"] is None else
                {k: (_clean(v) if isinstance(v, dict) else v) for k, v in norm["PN"].items()},
                "N_record": norm["N_record"], "PN_record": norm["PN_record"],
                "rule": rule, "multiplier_RI_to_per_100_decays": _f(mult),
                "multiplier_unc": _f(mult_unc)}
    return {
        "dataset": {
            "member": spec["member"], "nucid": _col(head, 1, 5), "dsid": _col(head, 10, 39),
            "dsref": _col(head, 40, 65), "pub": _col(head, 66, 74), "date": _col(head, 75, 80),
            "id_record": head.rstrip(), "history": history,
            "evaluators": full[-1].get("AUT") if full else None,
            "evaluation_citation": full[-1].get("CIT") if full else None,
            "literature_cutoff": next((h["CUT"] for h in history if "CUT" in h), None),
            "n_records": len(ds),
        },
        "parent": parent,
        "normalization": norm_out,
        "all_gamma_lines": gammas,
        "lines_used": used,
        "other_gamma_lines": {
            "count": len(others),
            "intensity_sum_per_decay": float(other_sum),
            "intensity_sum_unc": float(f"{other_unc2.sqrt():.3g}") if other_unc2 else 0.0,
            "fraction_of_used_lines": float(f"{other_sum / used_sum:.4g}"),
        },
    }


def build_ensdf() -> dict:
    out = {}
    for nuc, spec in NUCLIDES.items():
        sid = f"ensdf_{nuc}"
        rec = _complete(fetch.download(ENSDF_URL, ENSDF_ZIP, source_id=sid), ENSDF_ZIP)
        prov = _provenance_file(
            sid, paths.RAW / "ensdf" / "files.json", ENSDF_FILES_JSON,
            "NNDC distribution index at retrieval time; shows ensdf_261001.zip as the latest "
            "distribution. Provenance only: never re-fetched by the build (it changes upstream).")
        data = parse_decay(nuc)
        ds = data["dataset"]
        flags = [f"{g['energy_keV']} keV differs from nominal handover energy"
                 for g in data["all_gamma_lines"] if g.get("used") and not g["energy_matches_nominal"]]
        record = {
            "title": f"ENSDF {ds['dsid']} (daughter {ds['nucid']}), gamma lines and absolute "
                     "intensities",
            "citation": (f"Evaluated Nuclear Structure Data File (ENSDF), National Nuclear Data "
                         f"Center, Brookhaven National Laboratory; distribution "
                         f"{ENSDF_ZIP.name} (NNDC, 2026-10-01; file dated 2026-10-02), member "
                         f"{ds['member']}, dataset '{ds['dsid']}' (NUCID {ds['nucid']}), "
                         f"evaluation by {ds['evaluators']}, {ds['evaluation_citation']} "
                         f"[ENSDF PUB '{ds['pub']}', date {ds['date']}]. "
                         f"doi:{ENSDF_DOI}."),
            "doi": ENSDF_DOI,
            "data_class": "P1",
            "retrieval_method": ("bulk download of the full NNDC ENSDF distribution (zip), parsed "
                                 "locally (ENSDF 80-column format); no web form or API"),
            "licence": "ENSDF is produced by the US Nuclear Data Program and is publicly available "
                       "(NNDC); cite the dataset and DOI.",
            "files": [rec],
            **({"provenance_files": [prov]} if prov else {}),
            "notes": [
                "Absolute emission intensity per decay = RI x NR x BR / 100 (ENSDF normalization "
                "rule; PN NRBR overrides when present). Uncertainties combined in quadrature, "
                "correlations ignored.",
                f"lines_used = the handover lines {spec['nominal_keV']} keV, at ENSDF's own energies."
                + (" FLAG: " + "; ".join(flags) if flags else " ENSDF energies equal the nominal values."),
                "Only ENSDF G (gamma) records are included. X-rays (e.g. Ba K x-rays following "
                "137Cs decay, Ni K x-rays following 60Co decay) are not G records and are not "
                "included (see reports/phase1/wave1_C_references.md).",
            ],
            "data": data,
        }
        fetch.write_source(sid, record)
        out[nuc] = record
    return out


# --------------------------------------------------------------------------------------------
# 2. IEC 61267 RQR qualities from IAEA TRS-457 Table 6.2
# --------------------------------------------------------------------------------------------

TRS457_URL = "https://www-pub.iaea.org/MTCD/Publications/PDF/TRS457_web.pdf"
TRS457_PDF = paths.RAW / "trs457" / "TRS457_web.pdf"
RQR_USED = ["RQR 5", "RQR 6", "RQR 8", "RQR 9"]
# Read from the PDF in the probe. Used only as an assertion on the parsed text.
RQR_EXPECTED = {"RQR 5": (70, 2.58, 0.71), "RQR 6": (80, 3.01, 0.69),
                "RQR 8": (100, 3.97, 0.68), "RQR 9": (120, 5.00, 0.68)}
_RQR_ROW = re.compile(r"^RQR\s+(\d+)(\*?)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$")


def parse_rqr(pdf_path: Path = TRS457_PDF) -> dict:
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if "TABLE 6.2" in t and "SERIES RQR" in t:
            pages.append((i, t))
    if len(pages) != 1:
        raise ValueError(f"Table 6.2 (RQR) found on {len(pages)} PDF pages, expected 1")
    idx, text = pages[0]
    lines = [l.strip() for l in text.splitlines()]
    printed_page = lines[0] if lines and lines[0].isdigit() else None
    start = next(i for i, l in enumerate(lines) if l.startswith("TABLE 6.2"))
    rows, raw_rows = [], []
    for l in lines[start:]:
        m = _RQR_ROW.match(l)
        if m:
            raw_rows.append(l)
            rows.append({"name": f"RQR {int(m.group(1))}", "kV": int(m.group(3)),
                         "hvl1_mm_al": float(m.group(4)), "homogeneity_h": float(m.group(5)),
                         "reference_quality": m.group(2) == "*"})
    names = [r["name"] for r in rows]
    if names != [f"RQR {n}" for n in range(2, 11)]:
        raise ValueError(f"unexpected RQR rows parsed: {names}")
    star = next((i for i, l in enumerate(lines) if l.startswith("*")), None)
    footnote = None if star is None else " ".join(l for l in lines[star:] if l)
    by_name = {r["name"]: r for r in rows}
    for name, (kv, hvl, h) in RQR_EXPECTED.items():
        r = by_name[name]
        if (r["kV"], r["hvl1_mm_al"], r["homogeneity_h"]) != (kv, hvl, h):
            raise ValueError(f"{name}: parsed {r} differs from the values read in step 0 "
                             f"{(kv, hvl, h)}; investigate the PDF text")
    used = [{k: by_name[n][k] for k in ("name", "kV", "hvl1_mm_al", "homogeneity_h")}
            for n in RQR_USED]
    meta = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    return {"pdf_page_index0": idx, "pdf_page": idx + 1, "printed_page": printed_page,
            "pdf_pages_total": len(reader.pages), "pdf_metadata": meta,
            "table_title": "TABLE 6.2. CHARACTERIZATION OF RADIATION QUALITY SERIES RQR",
            "columns": ["Radiation quality", "X ray tube voltage (kV)", "First HVL (mm Al)",
                        "Homogeneity coefficient (h)"],
            "raw_rows": raw_rows, "footnote": footnote, "all": rows, "used": used}


def build_rqr() -> dict:
    sid = "rqr_trs457"
    rec = _complete(fetch.download(TRS457_URL, TRS457_PDF, source_id=sid), TRS457_PDF)
    parsed = parse_rqr()
    record = {
        "title": "IEC 61267 RQR radiation qualities (W anode), IAEA TRS-457 Table 6.2",
        "citation": ("International Atomic Energy Agency, Dosimetry in Diagnostic Radiology: An "
                     "International Code of Practice, Technical Reports Series No. 457, IAEA, "
                     "Vienna (2007), Table 6.2 'Characterization of radiation quality series RQR' "
                     f"(printed p. {parsed['printed_page']}, PDF p. {parsed['pdf_page']}). Values "
                     "originate from IEC 61267:2005, Medical diagnostic X-ray equipment - Radiation "
                     "conditions for use in the determination of characteristics."),
        "data_class": "STD",
        "retrieval_method": "official open document (IAEA publication PDF), table text extracted "
                            "with pypdf",
        "licence": "(c) IAEA 2007. The table rows are cited factual data and are not "
                   "redistributed in bulk.",
        "files": [rec],
        "extraction": {k: parsed[k] for k in ("pdf_page", "printed_page", "pdf_pages_total",
                                              "pdf_metadata", "table_title", "columns",
                                              "raw_rows", "footnote")},
        "notes": ["'*' on RQR 5 in the table marks the reference radiation quality for "
                  "unattenuated beams in general radiography (table footnote); it is stored "
                  "as reference_quality in data.all.",
                  "used = RQR 5, 6, 8, 9 (handover section 5.1)."],
        "data": {"all": parsed["all"], "used": parsed["used"]},
    }
    fetch.write_source(sid, record)
    return record


# --------------------------------------------------------------------------------------------
# 3. NIST SRD 126 (Hubbell & Seltzer) published tables
# --------------------------------------------------------------------------------------------

SRD126_BASE = "https://physics.nist.gov/PhysRefData/XrayMassCoef/"
SRD126_DIR = paths.RAW / "srd126"
SRD126_DOI = "10.18434/T4D01F"
# material key, page path, kind, Z, formula, expected page heading, Table 2 name (as printed, incl. [sic])
SRD126_MATERIALS = [
    ("air", "ComTab/air.html", "mixture", None, None, "Air, Dry (Near Sea Level)",
     "Air, Dry (near sea level)"),
    ("Al", "ElemTab/z13.html", "element", 13, "Al", "Aluminum", None),
    ("Fe", "ElemTab/z26.html", "element", 26, "Fe", "Iron", None),
    ("W", "ElemTab/z74.html", "element", 74, "W", "Tungsten", None),
    ("Pb", "ElemTab/z82.html", "element", 82, "Pb", "Lead", None),
    ("H2O", "ComTab/water.html", "compound", None, "H2O", "Water, Liquid", "Water, Liquid"),
    ("CaF2", "ComTab/fluoride.html", "compound", None, "CaF2", "Calcium Fluoride",
     "Calcium Fluoride"),
    ("CsI", "ComTab/cesium.html", "compound", None, "CsI", "Cesium Iodide", "Cesium Iodide"),
    ("LiF", "ComTab/lithiumflu.html", "compound", None, "LiF", "Lithium Fluride",
     "Lithium Fluride"),
    ("GaAs", "ComTab/gallium.html", "compound", None, "GaAs", "Gallium Arsenide",
     "Gallium Arsenide"),
    ("CdTe", "ComTab/telluride.html", "compound", None, "CdTe", "Cadmium Telluride",
     "Cadmium Telluride"),
    ("HgI2", "ComTab/mercuric.html", "compound", None, "HgI2", "Mercuric Iodide",
     "Mercuric Iodide"),
    ("Gd2O2S", "ComTab/gadolinium.html", "compound", None, "Gd2O2S", "Gadolinium Oxysulfide",
     "Gadolinium Oxysulfide"),
    ("CaSO4", "ComTab/calcium.html", "compound", None, "CaSO4", "Calcium Sulfate",
     "Calcium Sulfate"),
    ("Li2B4O7", "ComTab/lithium.html", "compound", None, "Li2B4O7", "Lithium Tetraborate",
     "Lithium Tetraborate"),
    ("MgB4O7", "ComTab/magnesium.html", "compound", None, "MgB4O7", "Magnesium Tetraborate",
     "Magnesium Tetroborate"),
]
SRD126_CONST_PAGES = ["tab1.html", "tab2.html"]
_SCI = re.compile(r"^\d\.\d+E[+-]\d\d$")
_PRE_ROW = re.compile(r"^\s*(.*?)\s*(\d\.\d+E[+-]\d\d)\s+(\d\.\d+E[+-]\d\d)\s+(\d\.\d+E[+-]\d\d)\s*$")
FIXTURE_MU = paths.FIXTURES / "srd126_mu_rho.csv"
FIXTURE_MAT = paths.FIXTURES / "srd126_materials.csv"
AIR_PARQUET = paths.INTERIM / "references" / "air_muen.parquet"


def _raw_name(page: str) -> str:
    return page.replace("/", "_")


def html_content_sha256(path: Path) -> str:
    """SHA-256 of the page with all <script> blocks removed and whitespace runs collapsed.

    The NIST server (behind Cloudflare) injects a script with a per-request token, so the raw
    bytes differ between retrievals of an unchanged table. This hash identifies the content.
    """
    t = Path(path).read_bytes().decode("latin-1")
    t = re.sub(r"(?is)<script\b.*?</script>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha256(t.encode("latin-1")).hexdigest()


def _fetch_srd126(page: str, prior_files: dict) -> dict:
    dest = SRD126_DIR / _raw_name(page)
    rel = _rel(dest)
    prior = prior_files.get(rel)
    rec = _complete(fetch.download(SRD126_BASE + page, dest), dest)
    rec["content_sha256"] = html_content_sha256(dest)
    if prior:
        if rec["sha256"] == prior["sha256"]:
            return prior
        if rec["content_sha256"] == prior.get("content_sha256"):
            # Re-retrieved page that differs only in the injected script: same table content.
            return {**prior, "note": "local copy re-retrieved on "
                    f"{rec['retrieved_utc']}; raw bytes differ only in the server-injected "
                    "script (content_sha256 identical)"}
        raise ValueError(f"SRD 126 page {page} content changed: {rec['content_sha256']} != "
                         f"{prior.get('content_sha256')}")
    return rec


class _TableParser(HTMLParser):
    """Collect <tr>/<td> rows, the <title>, bold headings and <pre> text with the stdlib parser."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._stack: list[list[list[str]]] = []
        self.title = ""
        self.bold: list[str] = []
        self.pre: list[str] = []
        self._in = {"title": 0, "b": 0, "pre": 0, "script": 0}
        self._bold_buf: list[str] = []
        self._pre_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._in:
            self._in[tag] += 1
            if tag == "b":
                self._bold_buf = []
            if tag == "pre":
                self._pre_buf = []
        if tag == "tr":
            self._stack.append([])
        elif tag in ("td", "th") and self._stack:
            self._stack[-1].append([])
        elif tag == "br":
            self._data(" ")

    def handle_endtag(self, tag):
        if tag in self._in and self._in[tag]:
            self._in[tag] -= 1
            if tag == "b":
                self.bold.append(re.sub(r"\s+", " ", "".join(self._bold_buf)).strip())
            if tag == "pre":
                self.pre.append("".join(self._pre_buf))
        if tag == "tr" and self._stack:
            row = self._stack.pop()
            self.rows.append([re.sub(r"\s+", " ", "".join(c)).strip() for c in row])

    def _data(self, data):
        if self._stack and self._stack[-1]:
            self._stack[-1][-1].append(data)

    def handle_data(self, data):
        if self._in["script"]:
            return
        if self._in["title"]:
            self.title += data
        if self._in["b"]:
            self._bold_buf.append(data)
        if self._in["pre"]:
            self._pre_buf.append(data)
        self._data(data)


def _parse_html(path: Path) -> _TableParser:
    p = _TableParser()
    p.feed(Path(path).read_bytes().decode("latin-1"))
    p.close()
    return p


def parse_coefficient_page(path: Path) -> dict:
    """Energy (MeV), μ/ρ, μ_en/ρ (cm²/g) rows of an SRD 126 element or compound page.

    The HTML table is parsed and cross-checked string-for-string against the ASCII (<pre>)
    version on the same page. At an absorption edge the table repeats the energy: the first row
    is below the edge and the second, labelled row (e.g. ``K``, ``L1`` or ``18 K`` for Z = 18)
    is above. Both rows carry the label, with ``edge_side`` below/above.
    """
    p = _parse_html(path)
    table = []
    for cells in p.rows:
        cells = [c for c in cells if c]
        if len(cells) == 3 and all(_SCI.match(c) for c in cells):
            table.append(("", *cells))
        elif len(cells) == 4 and all(_SCI.match(c) for c in cells[1:]) and not _SCI.match(cells[0]):
            table.append((cells[0], *cells[1:]))
    ascii_rows = []
    for block in p.pre:
        for line in block.splitlines():
            m = _PRE_ROW.match(line)
            if m:
                ascii_rows.append((re.sub(r"\s+", " ", m.group(1)).strip(), *m.group(2, 3, 4)))
    if not table:
        raise ValueError(f"{path}: no coefficient rows found")
    if table != ascii_rows:
        raise ValueError(f"{path}: HTML table and ASCII table disagree")
    rows = []
    for i, (label, e, mu, muen) in enumerate(table):
        rows.append({"energy_MeV_str": e, "energy_MeV": float(e), "mu_rho": float(mu),
                     "muen_rho": float(muen), "edge_label": label or None, "edge_side": None})
        if label:
            if i == 0 or table[i - 1][1] != e or table[i - 1][0]:
                raise ValueError(f"{path}: edge row {label} at {e} MeV lacks its below-edge twin")
            rows[i]["edge_side"] = "above"
            rows[i - 1]["edge_side"] = "below"
            rows[i - 1]["edge_label"] = label
    heading = next((b for b in p.bold if b and b not in ("Energy",)), "")
    return {"title": re.sub(r"\s+", " ", p.title).strip(), "heading": heading, "rows": rows}


def parse_tab2(path: Path) -> dict:
    """Material constants and composition (Z: mass fraction) from SRD 126 Table 2."""
    p = _parse_html(path)
    out = {}
    for cells in p.rows:
        cells = [c for c in cells if c]
        if len(cells) != 5 or not re.match(r"^\d\.\d+E[+-]\d\d$", cells[3]):
            continue
        name, z_a, i_ev, dens, comp = cells
        pairs = re.findall(r"(\d+)\s*:\s*(\d*\.\d+)", comp)
        if not pairs:
            raise ValueError(f"tab2: no composition for {name!r}")
        out[name] = {"z_over_a": float(z_a), "I_eV": float(i_ev), "density_g_cm3": float(dens),
                     "composition": [(int(z), float(w)) for z, w in pairs],
                     "composition_str": ";".join(f"{int(z)}:{w}" for z, w in pairs)}
    return out


def parse_tab1(path: Path) -> dict:
    """Element constants (Z, symbol, name, Z/A, I, density) from SRD 126 Table 1."""
    p = _parse_html(path)
    out = {}
    for cells in p.rows:
        cells = [c for c in cells if c]
        if len(cells) == 6 and cells[0].isdigit() and re.match(r"^\d\.\d+E[+-]\d\d$", cells[5]):
            z = int(cells[0])
            out[z] = {"symbol": cells[1], "name": cells[2], "z_over_a": float(cells[3]),
                      "I_eV": float(cells[4]), "density_g_cm3": float(cells[5])}
    return out


def _check_coefficients(key: str, rows: list[dict]) -> None:
    e = [r["energy_MeV"] for r in rows]
    for i in range(1, len(e)):
        dup = rows[i]["edge_side"] == "above" and rows[i - 1]["edge_side"] == "below"
        if dup:
            if e[i] != e[i - 1] or not rows[i]["mu_rho"] > rows[i - 1]["mu_rho"]:
                raise ValueError(f"{key}: inconsistent edge pair at {e[i]} MeV")
        elif not e[i] > e[i - 1]:
            raise ValueError(f"{key}: energies not increasing at row {i}")
    if e[0] != 1e-3 or e[-1] != 20.0:
        raise ValueError(f"{key}: table spans {e[0]}–{e[-1]} MeV, expected 0.001–20 MeV")
    for r in rows:
        if not 0 < r["muen_rho"] <= r["mu_rho"]:
            raise ValueError(f"{key}: muen_rho > mu_rho at {r['energy_MeV']} MeV")


def build_srd126() -> dict:
    sid = "nist_srd126"
    prior = fetch.read_source(sid) or {}
    prior_files = {f["path"]: f for f in prior.get("files", [])}
    files, pages = [], {}
    for key, page, *_ in SRD126_MATERIALS:
        files.append(_fetch_srd126(page, prior_files))
        pages[key] = parse_coefficient_page(SRD126_DIR / _raw_name(page))
    for page in SRD126_CONST_PAGES:
        files.append(_fetch_srd126(page, prior_files))
    tab1 = parse_tab1(SRD126_DIR / "tab1.html")
    tab2 = parse_tab2(SRD126_DIR / "tab2.html")
    prov = _provenance_file(
        sid, SRD126_DIR / "cover.html", SRD126_BASE + "cover.html",
        "SRD 126 table of contents (now a www.nist.gov page). Retrieved for the database "
        "identification: 'NIST Standard Reference Database 126', 'Last Update to Data Content: "
        "July 2004', 'Online: May 1996', DOI 10.18434/T4D01F. Provenance only: never re-fetched "
        "by the build (the page is regenerated upstream).")

    mu_rows, mat_rows = [], []
    for key, page, kind, z, formula, heading, tab2_name in SRD126_MATERIALS:
        pg = pages[key]
        want = heading if z is None else f"{heading} Z = {z}"      # element pages also show Z
        if heading.split(" (")[0] not in pg["title"] or pg["heading"] != want:
            raise ValueError(f"{page}: title {pg['title']!r} / heading {pg['heading']!r} "
                             f"do not match {heading!r}")
        _check_coefficients(key, pg["rows"])
        for r in pg["rows"]:
            mu_rows.append({"material": key, "kind": kind, "Z": z, "formula": formula,
                            "energy_MeV": r["energy_MeV"], "mu_rho_cm2_g": r["mu_rho"],
                            "muen_rho_cm2_g": r["muen_rho"], "edge_label": r["edge_label"],
                            "edge_side": r["edge_side"], "source_page": page})
        if kind == "element":
            t1 = tab1[z]
            if t1["name"] != heading:
                raise ValueError(f"tab1 Z={z}: name {t1['name']!r} != {heading!r}")
            mat_rows.append({"material": key, "name": t1["name"], "formula": formula,
                             "density_g_cm3": t1["density_g_cm3"], "composition": f"{z}:1",
                             "source_page": "tab1.html"})
        else:
            t2 = tab2[tab2_name]
            s = sum(w for _, w in t2["composition"])
            if abs(s - 1) > 2e-6:
                raise ValueError(f"tab2 {tab2_name}: mass fractions sum to {s}")
            mat_rows.append({"material": key, "name": tab2_name, "formula": formula,
                             "density_g_cm3": t2["density_g_cm3"],
                             "composition": t2["composition_str"], "source_page": "tab2.html"})

    mu_df = pd.DataFrame(mu_rows)
    mu_df["Z"] = mu_df["Z"].astype("Int64")
    paths.ensure(paths.FIXTURES)
    mu_df.to_csv(FIXTURE_MU, index=False, lineterminator="\n")
    pd.DataFrame(mat_rows).to_csv(FIXTURE_MAT, index=False, lineterminator="\n")

    air = [{"energy_keV": float(Decimal(r["energy_MeV_str"]) * 1000),
            "mu_rho": r["mu_rho"], "muen_rho": r["muen_rho"],
            "edge_label": r["edge_label"], "edge_side": r["edge_side"]}
           for r in pages["air"]["rows"]]
    paths.ensure(AIR_PARQUET.parent)
    pd.DataFrame(air).to_parquet(AIR_PARQUET, index=False)

    record = {
        "title": "NIST SRD 126: X-ray mass attenuation and mass energy-absorption coefficients "
                 "(air; validation elements and compounds; material constants)",
        "citation": ("J. H. Hubbell and S. M. Seltzer, Tables of X-Ray Mass Attenuation "
                     "Coefficients and Mass Energy-Absorption Coefficients (version 1.4), NIST "
                     "Standard Reference Database 126, National Institute of Standards and "
                     "Technology, Gaithersburg, MD (online 1996, last data update July 2004), "
                     f"doi:{SRD126_DOI}. Originally published as NISTIR 5632 (1995)."),
        "doi": SRD126_DOI,
        "version": {"version": "1.4", "version_source": "NIST recommended citation; the version "
                    "number is not printed on the retrieved table pages",
                    "last_data_update": "July 2004", "online_since": "May 1996",
                    "date_source": "cover.html (provenance_files)"},
        "data_class": "P1",
        "retrieval_method": ("official published HTML tables, each page retrieved once and "
                             "archived; parsed with the Python stdlib html.parser (HTML table "
                             "cross-checked against the ASCII table on the same page). No bulk "
                             "file exists (data.nist.gov ark:/88434/mds01fkz7n holds only a DOI "
                             "link); the XCOM web form is never used. See DECISIONS D-004."),
        "licence": "https://www.nist.gov/open/license",
        "checksum_note": ("sha256 is of the archived raw bytes. The server injects a script with "
                          "a per-request token, so a re-retrieval of an unchanged page has "
                          "different raw bytes. content_sha256 (scripts removed, whitespace "
                          "collapsed; shieldmap.ingest.references.html_content_sha256) is stable "
                          "and is what a rebuild verifies."),
        "files": files,
        **({"provenance_files": [prov]} if prov else {}),
        "pages": {key: {"page": page, "title": pages[key]["title"],
                        "heading": pages[key]["heading"], "rows": len(pages[key]["rows"]),
                        "kind": kind, "Z": z, "formula": formula, "tab2_name": tab2_name}
                  for key, page, kind, z, formula, _, tab2_name in SRD126_MATERIALS},
        "outputs": {"air_parquet": _rel(AIR_PARQUET), "fixture_mu_rho": _rel(FIXTURE_MU),
                    "fixture_materials": _rel(FIXTURE_MAT)},
        "notes": [
            "Units: energy in keV (data.air) / MeV (fixture); mu_rho and muen_rho in cm2/g.",
            "Edge rows: the energy is repeated; edge_side 'below' is the first row and 'above' "
            "the labelled second row. Compound/mixture labels carry Z (e.g. '18 K' = argon K edge).",
            "Formulas in the fixtures are assigned by shieldmap from the NIST material names; "
            "Table 2 names are kept as printed (incl. 'Lithium Fluride' [sic], 'Magnesium "
            "Tetroborate' [sic]).",
        ],
        "data": {"air": air},
    }
    fetch.write_source(sid, record)
    return record


# --------------------------------------------------------------------------------------------


def build() -> dict:
    out = {"ensdf": build_ensdf(), "rqr": build_rqr(), "srd126": build_srd126()}
    return out


def _summary(out: dict) -> str:
    lines = []
    for nuc, rec in out["ensdf"].items():
        d = rec["data"]
        lines.append(f"{nuc}: {d['dataset']['dsid']} ({d['dataset']['nucid']})")
        for u in d["lines_used"]:
            lines.append(f"  {u['energy_keV']} +/- {u['energy_unc_keV']} keV: "
                         f"{u['intensity_per_decay']} +/- {u['intensity_unc']} /decay")
        o = d["other_gamma_lines"]
        lines.append(f"  other G lines: {o['count']}, sum {o['intensity_sum_per_decay']:.4g}/decay")
    for r in out["rqr"]["data"]["used"]:
        lines.append(f"{r['name']}: {r['kV']} kV, HVL1 {r['hvl1_mm_al']} mm Al, h {r['homogeneity_h']}")
    air = out["srd126"]["data"]["air"]
    lines.append(f"air: {len(air)} rows, {air[0]["energy_keV"]}-{air[-1]['energy_keV']} keV")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_summary(build()))
