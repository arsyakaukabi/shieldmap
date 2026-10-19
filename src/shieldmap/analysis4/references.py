"""Reference reconciliation - offline build (and CLI entry to the fetch mode).

    python -m shieldmap.analysis4.references fetch [--only ID|GROUP ...] [--retry-failed]   # network
    python -m shieldmap.analysis4.references build                                           # offline

``run()`` is the offline build used by ``python -m shieldmap.analyze4a``. It never touches the network
(``refs_fetch`` and ``requests`` are not imported). Inputs:

* ``config/phase4a/reference_queries.yaml``        - the query plan (what was asked, and why);
* ``config/phase4a/reference_classifications.yaml`` - the reading-based judgements, each citing raw files;
* ``data/raw/references/phase4a/api/*.json``         - cached API requests/responses (endpoint, query string,
  UTC timestamp, HTTP status, raw response);
* ``data/raw/references/phase4a/fulltext/*``         - open full texts with ``.meta.json`` sidecars (URL, SHA-256);
* ``../phase0/references.bib`` / ``overlap_table.md`` - the library (read only);
* ``data/phase4a/headline_table.csv`` (optional)     - headline ids per dossier result (CSV ``headline_rows``) and the coverage section.

Outputs: ``data/phase4a/reference_map.csv``, ``data/phase4a/references_phase4a.bib``,
``reports/phase4a/reference_map.md`` and ``reports/phase4a/queries_phase4a.md``.

Every identifier in the map is checked against its cached Crossref/DataCite record (HTTP 200, DOI equal to the
cited DOI); every file named as reading evidence must exist, and full texts must match the SHA-256 in their
sidecar. Any failure raises; nothing unverified enters the map or the bib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import unicodedata
from pathlib import Path

import yaml

from shieldmap import paths

RAW_REFS = paths.RAW / "references" / "phase4a"
API_DIR = RAW_REFS / "api"
FT_DIR = RAW_REFS / "fulltext"
PLAN = paths.CONFIG / "phase4a" / "reference_queries.yaml"
CLASSIFICATIONS = paths.CONFIG / "phase4a" / "reference_classifications.yaml"
PHASE0 = paths.REPO_ROOT / "phase0"
OUT_CSV = paths.DATA / "phase4a" / "reference_map.csv"
OUT_BIB = paths.DATA / "phase4a" / "references_phase4a.bib"
OUT_MD = paths.ROOT / "reports" / "phase4a" / "reference_map.md"
OUT_QUERIES = paths.ROOT / "reports" / "phase4a" / "queries_phase4a.md"
HEADLINE = paths.DATA / "phase4a" / "headline_table.csv"

CLASSES = ("confirms known", "quantifies known", "new")
DEPTHS = ("full-text", "abstract-only", "metadata-only")
FIRST_NEW_KEY = 84  # Ends at Shield083

CSV_COLUMNS = [
    "row_id", "result_id", "kind", "dossier_result", "statement", "classification", "prior_bib_keys", "prior_dois",
    "closest_prior",
    "reading_depth", "closest_prior_statement", "searched_queries", "positioning_sentence",
    "phase3_framing_change", "headline_rows", "raw_search_files", "raw_reading_files",
    "raw_verification_files", "verification",
]


class ReferenceBuildError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- helpers
def _rel(p: Path) -> str:
    return p.resolve().relative_to(paths.REPO_ROOT.resolve()).as_posix()


def _load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _api(qid: str) -> dict:
    p = API_DIR / f"{qid}.json"
    if not p.exists():
        raise ReferenceBuildError(f"missing cached API record {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_title(t: str) -> str:
    t = html.unescape(re.sub(r"<[^>]+>", " ", t or ""))
    t = unicodedata.normalize("NFKD", t)
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _clean(t: str) -> str:
    t = re.sub(r"</sub>\s+(?=[A-Z][a-z]?\s*<sub>)", "</sub>", t or "")  # join formula tokens (Lu2Hf2O7)
    t = re.sub(r"\s*<(sub|sup)>\s*([^<]*?)\s*</\1>", r"\2", t)  # chemical formulae, ordinals
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    return re.sub(r"\s+", " ", t).strip()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fulltext_files(qid: str, unreadable: dict) -> list[Path]:
    meta = FT_DIR / f"{qid}.meta.json"
    if not meta.exists():
        raise ReferenceBuildError(f"missing full-text sidecar {meta}")
    side = json.loads(meta.read_text(encoding="utf-8"))
    if qid in unreadable:
        raise ReferenceBuildError(f"{qid} is listed as unreadable but cited as reading evidence")
    if side.get("http_status") != 200 or not side.get("file"):
        raise ReferenceBuildError(f"{qid}: full text not retrieved (status {side.get('http_status')})")
    f = FT_DIR / side["file"]
    if _sha256(f) != side["sha256"]:
        raise ReferenceBuildError(f"{qid}: SHA-256 mismatch for {f.name}")
    out = [meta, f]
    if side.get("text_file"):
        t = FT_DIR / side["text_file"]
        if _sha256(t) != side["text_sha256"]:
            raise ReferenceBuildError(f"{qid}: SHA-256 mismatch for {t.name}")
        out.append(t)
    return out


def _token_files(tok: str, unreadable: dict) -> list[Path]:
    """Resolve a provenance token from the classification file to concrete files."""
    if tok == "phase0":
        return [PHASE0 / "references.csv", PHASE0 / "overlap_table.md"]
    if tok.startswith("phase0:"):
        return [PHASE0 / tok.split(":", 1)[1]]
    if tok.startswith("config:"):
        return [paths.SOURCES / f"{tok.split(':', 1)[1]}.yaml"]
    if (FT_DIR / f"{tok}.meta.json").exists():
        return _fulltext_files(tok, unreadable)
    p = API_DIR / f"{tok}.json"
    if not p.exists():
        raise ReferenceBuildError(f"provenance token {tok!r} resolves to no cached file")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("http_status") != 200:
        raise ReferenceBuildError(f"{tok}: cached request has HTTP status {rec.get('http_status')}")
    return [p]


def _record_title(rec: dict) -> str | None:
    j = rec.get("response_json") or {}
    if rec["kind"] == "openalex_work":
        return j.get("title")
    if rec["kind"] == "s2_paper":
        return j.get("title")
    if rec["kind"] == "crossref_work":
        return (j.get("message", {}).get("title") or [None])[0]
    return None


# ------------------------------------------------------------------------------------ verified metadata
def _meta_crossref(rec: dict, doi: str) -> dict:
    m = rec["response_json"]["message"]
    if m.get("DOI", "").lower() != doi.lower():
        raise ReferenceBuildError(f"Crossref record DOI {m.get('DOI')} != cited {doi}")
    year = None
    for k in ("issued", "published-print", "published-online", "approved", "created"):
        dp = (m.get(k) or {}).get("date-parts") or [[None]]
        if dp and dp[0] and dp[0][0]:
            year = dp[0][0]
            break
    authors = []
    for a in m.get("author", []) or []:
        if a.get("family"):
            authors.append({"family": a["family"].strip(), "given": (a.get("given") or "").strip()})
        elif a.get("name"):
            authors.append({"literal": a["name"].strip()})
    return {
        "source": "crossref", "doi": m.get("DOI"), "type": m.get("type"),
        "title": _clean((m.get("title") or [""])[0]), "authors": authors, "year": year,
        "container": _clean((m.get("container-title") or [""])[0]) if m.get("container-title") else "",
        "volume": m.get("volume", ""), "number": m.get("issue", ""), "pages": m.get("page", ""),
        "publisher": m.get("publisher", ""),
        "standards_body": (m.get("standards-body") or {}).get("name", ""),
    }


def _meta_datacite(rec: dict, doi: str) -> dict:
    a = rec["response_json"]["data"]["attributes"]
    if a.get("doi", "").lower() != doi.lower():
        raise ReferenceBuildError(f"DataCite record DOI {a.get('doi')} != cited {doi}")
    authors = []
    for c in a.get("creators", []) or []:
        if c.get("familyName"):
            authors.append({"family": c["familyName"], "given": c.get("givenName", "")})
        elif c.get("name"):
            authors.append({"literal": c["name"]})
    return {
        "source": "datacite", "doi": a.get("doi"), "type": "dataset",
        "title": _clean((a.get("titles") or [{}])[0].get("title", "")), "authors": authors,
        "year": a.get("publicationYear"), "container": "", "volume": "", "number": "", "pages": "",
        "publisher": a.get("publisher") if isinstance(a.get("publisher"), str) else
        (a.get("publisher") or {}).get("name", ""), "standards_body": "",
    }


def verify_reference(key: str, ref: dict, unreadable: dict) -> dict:
    for f in ("doi", "verify", "group", "depth", "read", "found_by", "states", "role"):
        if f not in ref:
            raise ReferenceBuildError(f"reference {key}: missing field {f}")
    if ref["depth"] not in DEPTHS:
        raise ReferenceBuildError(f"reference {key}: bad depth {ref['depth']}")
    rec = _api(ref["verify"])
    if rec.get("http_status") != 200:
        raise ReferenceBuildError(f"reference {key}: verification record status {rec.get('http_status')}")
    meta = _meta_crossref(rec, ref["doi"]) if rec["kind"] == "crossref_work" else _meta_datacite(rec, ref["doi"])
    if not meta["title"]:
        raise ReferenceBuildError(f"reference {key}: verified record has no title")
    read_files: list[Path] = []
    for tok in ref["read"]:
        fs = _token_files(tok, unreadable)
        read_files += fs
        if fs and fs[0].parent == API_DIR:
            t = _record_title(json.loads(fs[0].read_text(encoding="utf-8")))
            if t and _norm_title(t)[:30] != _norm_title(meta["title"])[:30] and ref["read"].index(tok) == 0:
                raise ReferenceBuildError(f"reference {key}: read record title {t!r} does not match {meta['title']!r}")
    if ref["depth"] == "full-text" and not any(p.parent == FT_DIR for p in read_files):
        raise ReferenceBuildError(f"reference {key}: depth full-text but no full text in `read`")
    found_files: list[Path] = []
    for tok in ref["found_by"]:
        found_files += _token_files(tok, unreadable)
    return {"key": key, **ref, "meta": meta, "verify_file": API_DIR / f"{ref['verify']}.json",
            "read_files": read_files, "found_files": found_files}


# ------------------------------------------------------------------------------------------------- bib
def _phase0_bib() -> tuple[str, dict[str, str]]:
    text = (PHASE0 / "references.bib").read_text(encoding="utf-8")
    doi2key = {}
    for m in re.finditer(r"@\w+\{(Shield\d+),(.*?)\n\}", text, flags=re.S):
        d = re.search(r"\bdoi = \{([^}]*)\}", m.group(2))
        if d:
            doi2key[d.group(1).lower()] = m.group(1)
    return text, doi2key


# Project terminology in project-written `note` fields of the entries (titles are never touched).
# Own references.bib is not modified; only the merged stage 4a copy is.
_P0_NOTE_TERMS = [
    ("proxy-cost input rather than compound purchase price", "price-proxy input rather than compound purchase price"),
    ("dose-equivalent rate, weight and cost,", "dose-equivalent rate, weight and a monetary objective,"),
    ("with weight/cost objectives", "with weight and monetary objectives"),
    ("neutron flux, dose, cost and weight", "neutron flux, dose, a monetary objective and weight"),
    ("Lead-equivalence measurement methods", "Measurement methods for Pb-relative ratings"),
    ("joint-constraint cost attribution", "joint-constraint price attribution"),
    ("constraint-cost Shapley games", "constraint-price Shapley games"),
]


def _p0_note_terminology(text: str) -> str:
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("note = {"):
            for old, new in _P0_NOTE_TERMS:
                line = line.replace(old, new)
            if re.search(r"\bcost|lead.equivalen|non-toxic", line, flags=re.I):
                raise ReferenceBuildError(f"unhandled terminology in a Phase 0 note: {line[:120]}")
        out.append(line)
    return "\n".join(out)


def _bib_escape(s: str) -> str:
    s = str(s).replace("\\", "/")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    return s.replace("{", "(").replace("}", ")")


def _bib_authors(authors: list[dict]) -> str:
    out = []
    for a in authors:
        if "literal" in a:
            out.append("{" + _bib_escape(a["literal"]) + "}")
        else:
            out.append(_bib_escape(f"{a['family']}, {a['given']}".strip().rstrip(",")))
    return " and ".join(out)


_TYPE = {"journal-article": "article", "book-chapter": "incollection", "book": "book", "report": "techreport",
         "proceedings-article": "inproceedings", "posted-content": "misc", "standard": "misc", "dataset": "misc"}


def _bib_entry(key: str, r: dict) -> str:
    m = r["meta"]
    etype = _TYPE.get(m["type"], "misc")
    f = [("title", "{" + _bib_escape(m["title"]) + "}")]
    if m["authors"]:
        f.append(("author", _bib_authors(m["authors"])))
    if m["year"]:
        f.append(("year", str(m["year"])))
    if m["container"]:
        f.append(({"article": "journal", "incollection": "booktitle", "inproceedings": "booktitle"}.get(etype, "howpublished"),
                  _bib_escape(m["container"])))
    if etype == "techreport":
        f.append(("institution", _bib_escape(m["publisher"] or "unknown")))
    for k in ("volume", "pages"):
        if m[k]:
            f.append((k, _bib_escape(m[k])))
    if m["publisher"]:
        f.append(("publisher", _bib_escape(m["publisher"])))
    if m["number"]:
        f.append(("number", _bib_escape(m["number"])))
    f.append(("doi", m["doi"].lower() if m["source"] == "crossref" else m["doi"]))
    f.append(("url", "https://doi.org/" + (m["doi"].lower() if m["source"] == "crossref" else m["doi"])))
    f.append(("keywords", f"group-{r['group']}"))
    extra = []
    if m["type"] == "posted-content":
        extra.append("Preprint (not peer reviewed).")
    if m["standards_body"]:
        extra.append(f"Standards body: {m['standards_body']}; Crossref gives the approval date as the year.")
    if not m["authors"]:
        extra.append(f"The {m['source'].capitalize()} record lists no author; none added.")
    note = (f"Reading depth: {r['depth']}. Role: {r['role']} Phase 4a Step 6 reference reconciliation; identifier "
            f"verified via {m['source']} ({_rel(r['verify_file'])}).")
    if extra:
        note += " " + " ".join(extra)
    f.append(("note", _bib_escape(note)))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f)
    return f"@{etype}{{{key},\n{body}\n}}\n"


# ---------------------------------------------------------------------------------------------- queries
def _hits(rec: dict) -> str:
    j = rec.get("response_json")
    k = rec.get("kind")
    st = rec.get("http_status")
    if st != 200:
        return "-"
    if k in ("openalex_search", "openalex_filter"):
        return str((j or {}).get("meta", {}).get("count", ""))
    if k == "crossref_search":
        msg = (j or {}).get("message", {})
        return f"{msg.get('total-results', '')} (ranked; {len(msg.get('items', []))} returned)"
    if k == "s2_search":
        return str((j or {}).get("total", ""))
    if k == "arxiv_search":
        m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", rec.get("response_text", ""))
        return m.group(1) if m else ""
    if k == "fulltext":
        return f"{rec.get('bytes', '')} bytes"
    if k == "doi_handle":
        return "1" if (j or {}).get("responseCode") == 1 else "0"
    return "1"


def _query_label(q: dict) -> str:
    for k in ("query", "filter", "work", "doi", "url", "arxiv"):
        if q.get(k):
            return f"{k}: {q[k]}"
    return ""


def write_queries_md(plan: list[dict], out: Path) -> None:
    lines = ["# Phase 4a Step 6 - query log", "",
             "Generated by `shieldmap.analysis4.references.run()` from `config/phase4a/reference_queries.yaml` and the "
             "cached responses in `data/raw/references/phase4a/api/` (one JSON per query: endpoint, exact query "
             "string with any API key removed, UTC timestamp, HTTP status, raw response). Hits: OpenAlex `meta.count`; "
             "Crossref `total-results` (relevance-ranked, so large and loose) and the number of items returned; "
             "Semantic Scholar `total`; identifier lookups 1 when resolved; full texts the byte size.", "",
             "| # | query id | group | kind | query | purpose | HTTP | hits | UTC |", "|---|---|---|---|---|---|---|---|---|"]
    for i, q in enumerate(plan, 1):
        p = API_DIR / f"{q['id']}.json"
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            st, hits, ts = rec.get("http_status"), _hits(rec), rec.get("timestamp_utc", "")
        else:
            st, hits, ts = "not fetched", "-", ""
        lab = _query_label(q).replace("|", "\\|")
        pur = (q.get("purpose") or "").replace("|", "\\|")
        lines.append(f"| {i} | `{q['id']}` | {q['group']} | {q['kind']} | {lab} | {pur} | {st} | {hits} | {ts} |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------------------------------- build
def _load_plan() -> list[dict]:
    plan = _load_yaml(PLAN)
    out = []
    for grp in plan["groups"]:
        for q in grp["queries"]:
            q = dict(q)
            q.setdefault("purpose", grp.get("purpose", ""))
            q["group"] = grp["name"]
            out.append(q)
    return out


def _headline_ids_by_dossier(headline: Path | None) -> dict[str, list[str]]:
    """Headline-table ids per dossier result, derived at build time (ids shift when rows are added)."""
    out: dict[str, list[str]] = {}
    if headline is None or not Path(headline).exists():
        return out
    with open(headline, encoding="utf-8") as f:
        for h in csv.DictReader(f):
            out.setdefault(h.get("dossier_result", ""), []).append(h.get("id", ""))
    return out


def _short_prior(rs: list[dict]) -> str:
    """Short citation text with DOI, for joins by the dossier generator."""
    parts = []
    for r in rs:
        a = r["meta"]["authors"]
        first = (a[0].get("family") or a[0].get("literal", "").split(",")[0]) if a else (r["meta"]["publisher"] or "anon.")
        parts.append(f"{first}{' et al.' if len(a) > 1 else ''} {r['meta']['year']}, doi:{r['doi'].lower()}")
    return "; ".join(parts)


def build(out_csv: Path = OUT_CSV, out_bib: Path = OUT_BIB, out_md: Path = OUT_MD,
          out_queries: Path = OUT_QUERIES, headline: Path | None = HEADLINE) -> dict:
    cls = _load_yaml(CLASSIFICATIONS)
    plan = _load_plan()
    unreadable = cls.get("fulltext_unreadable", {}) or {}
    for qid in unreadable:
        if not (FT_DIR / f"{qid}.meta.json").exists():
            raise ReferenceBuildError(f"unreadable full text {qid} has no sidecar")

    refs = {k: verify_reference(k, v, unreadable) for k, v in cls["references"].items()}
    cited = ({k for row in cls["rows"] for k in row["prior"]} | {k for m in cls["methods"] for k in m["refs"]}
             | {k for h in cls.get("high_risk", []) for k in h["refs"]})
    uncited = sorted(set(refs) - cited)
    if uncited:
        raise ReferenceBuildError(f"references defined but never cited (remove them): {uncited}")
    p0_text, p0_doi2key = _phase0_bib()
    p0_text = _p0_note_terminology(p0_text)
    # keys: key if the DOI is in the library; else the pinned `bib_key` (stable across rebuilds,
    # ); else the next free number after every key already used
    pinned = [r["bib_key"] for r in refs.values() if r.get("bib_key")]
    if len(pinned) != len(set(pinned)) or set(pinned) & set(p0_doi2key.values()):
        raise ReferenceBuildError("pinned bib keys clash with each other or with Phase 0 keys")
    next_key = max([FIRST_NEW_KEY - 1] + [int(k[6:]) for k in pinned]) + 1
    for k, r in refs.items():
        p0 = p0_doi2key.get(r["doi"].lower())
        if p0:
            r["bib_key"], r["new"] = p0, False
        elif r.get("bib_key"):
            r["new"] = True
        else:
            r["bib_key"], r["new"] = f"Shield{next_key:03d}", True
            next_key += 1

    def ref_list(keys):
        for k in keys:
            if k not in refs:
                raise ReferenceBuildError(f"unknown reference key {k}")
        return [refs[k] for k in keys]

    hl_by_dossier = _headline_ids_by_dossier(headline)
    rows_out = []
    seen = set()
    for row in cls["rows"]:
        if row["id"] in seen:
            raise ReferenceBuildError(f"duplicate row {row['id']}")
        seen.add(row["id"])
        if row["classification"] not in CLASSES:
            raise ReferenceBuildError(f"row {row['id']}: bad classification {row['classification']}")
        rs = ref_list(row["prior"])
        if row["classification"] != "new" and not rs:
            raise ReferenceBuildError(f"row {row['id']}: {row['classification']} needs at least one prior reference")
        if row["classification"] != "new" and all(r["depth"] == "metadata-only" for r in rs):
            raise ReferenceBuildError(f"row {row['id']}: classification rests only on metadata-only references")
        search_files = []
        for tok in row["searched"]:
            search_files += _token_files(tok, unreadable)
        kind = "result" if row["id"].startswith("R") and row["id"][1:2].isdigit() else "cross-cutting"
        rows_out.append({
            "row_id": row["id"], "result_id": row["dossier"] if kind == "result" else "CROSS:" + row["id"], "kind": kind, "dossier_result": row["dossier"], "statement": row["statement"],
            "classification": row["classification"],
            "prior_bib_keys": "; ".join(r["bib_key"] for r in rs),
            "prior_dois": "; ".join(r["doi"].lower() for r in rs),
            "closest_prior": _short_prior(rs),
            "reading_depth": "; ".join(f"{r['bib_key']}:{r['depth']}" for r in rs),
            "closest_prior_statement": _clean(row["closest"]),
            "searched_queries": "; ".join(row["searched"]),
            "positioning_sentence": _clean(row["positioning"]),
            "phase3_framing_change": _clean(str(row.get("phase3_change", "none"))),
            "headline_rows": "; ".join(hl_by_dossier.get(row["dossier"], [])),
            "raw_search_files": "; ".join(dict.fromkeys(_rel(p) for p in search_files)),
            "raw_reading_files": "; ".join(dict.fromkeys(_rel(p) for r in rs for p in r["read_files"])),
            "raw_verification_files": "; ".join(dict.fromkeys(_rel(r["verify_file"]) for r in rs)),
            "verification": "all identifiers verified (Crossref/DataCite record, HTTP 200, DOI match)" if rs else "no identifier cited",
        })
    for m in cls["methods"]:
        rs = ref_list(m["refs"])
        rows_out.append({
            "row_id": m["id"], "result_id": "METHOD:" + m["method"].split(" (")[0], "kind": "method", "dossier_result": "", "statement": m["method"],
            "classification": "method",
            "prior_bib_keys": "; ".join(r["bib_key"] for r in rs),
            "prior_dois": "; ".join(r["doi"].lower() for r in rs),
            "closest_prior": _short_prior(rs),
            "reading_depth": "; ".join(f"{r['bib_key']}:{r['depth']}" for r in rs),
            "closest_prior_statement": "", "searched_queries": "; ".join(dict.fromkeys(t for r in rs for t in r["found_by"])),
            "positioning_sentence": "", "phase3_framing_change": _clean(m.get("note", "")), "headline_rows": "",
            "raw_search_files": "; ".join(dict.fromkeys(_rel(p) for r in rs for p in r["found_files"])),
            "raw_reading_files": "; ".join(dict.fromkeys(_rel(p) for r in rs for p in r["read_files"])),
            "raw_verification_files": "; ".join(dict.fromkeys(_rel(r["verify_file"]) for r in rs)),
            "verification": "all identifiers verified (Crossref/DataCite record, HTTP 200, DOI match)",
        })

    dossier_ids = {"R1.1", "R1.2", "R1.3", "R1.4", "R2.0", "R2.1", "R2.2", "R2.3", "R2.4", "R2.5", "R2.6", "R2.7",
                   "R2.8", "R3.1", "R3.2", "R3.3", "R3.4", "R3.5", "R4.1", "R4.2", "R4.3"}
    covered = {r["dossier_result"] for r in rows_out if r["kind"] == "result"}
    missing = dossier_ids - covered
    if missing:
        raise ReferenceBuildError(f"dossier results without a map row: {sorted(missing)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(rows_out)
    out_csv.write_text(buf.getvalue(), encoding="utf-8", newline="")

    # bib: all entries verbatim, then the new verified entries
    new = sorted((r for r in refs.values() if r["new"]), key=lambda r: r["bib_key"])
    bib = ("% Phase 4a merged library: all Phase 0 entries (project-written note fields terminology-edited,\n"
           "% decision P4A-S6b-D7), then the Phase 4a additions.\n\n" + p0_text.rstrip("\n")
           + "\n\n% ---- Phase 4a additions (Step 6 reference reconciliation): verified identifiers only ----\n\n")
    bib +="\n".join(_bib_entry(r["bib_key"], r) for r in new)
    out_bib.parent.mkdir(parents=True, exist_ok=True)
    out_bib.write_text(bib, encoding="utf-8", newline="\n")

    out_queries.parent.mkdir(parents=True, exist_ok=True)
    write_queries_md(plan, out_queries)
    summary = _write_md(cls, refs, rows_out, plan, out_md, headline)
    return summary


# --------------------------------------------------------------------------------------------- markdown
def _cite(r: dict) -> str:
    m = r["meta"]
    a = m["authors"]
    first = (a[0].get("family") or a[0].get("literal", "").split(",")[0]) if a else (m["publisher"] or "anon.")
    etal = " et al." if len(a) > 2 else (f" and {a[1].get('family') or a[1].get('literal')}" if len(a) == 2 else "")
    return f"{first}{etal} {m['year']} [{r['bib_key']}; doi:{r['doi'].lower()}]"


def _write_md(cls, refs, rows, plan, out_md: Path, headline: Path | None) -> dict:
    res = [r for r in rows if r["kind"] == "result"]
    counts = {c: sum(1 for r in res if r["classification"] == c) for c in CLASSES}
    cross = [r for r in rows if r["kind"] == "cross-cutting"]
    meth = [r for r in rows if r["kind"] == "method"]
    n_new_refs = sum(1 for r in refs.values() if r["new"])
    new_keys = [r["bib_key"] for r in refs.values() if r["new"]]
    depth_counts = {d: sum(1 for r in refs.values() if r["depth"] == d) for d in DEPTHS}
    L = ["# Phase 4a Step 6 - reference map", "",
         "Generated offline by `shieldmap.analysis4.references.run()` from the cached API responses and open full texts "
         "in `data/raw/references/phase4a/` and the reading judgements in `config/phase4a/reference_classifications.yaml`. "
         "Machine-readable version: `data/phase4a/reference_map.csv` (one row per result, sub-result, cross-cutting caveat and "
         "method, each naming the raw files its classification rests on). Bibliography: `data/phase4a/references_phase4a.bib`. "
         "Query log: `reports/phase4a/queries_phase4a.md`.", "",
         "**Classes.** *confirms known*: a work we read states the same claim (direction and order of magnitude). "
         "*quantifies known*: a work we read states the mechanism or the qualitative claim, not our quantity or scale. "
         "*new*: no prior statement found in the searches named in the row (this is a statement about those searches, "
         "not proof of absence). Reading depth is per reference: full-text, abstract-only or metadata-only.", "",
         f"**Counts (dossier result rows, {len(res)} rows covering R1.1-R4.3):** confirms known {counts['confirms known']}, "
         f"quantifies known {counts['quantifies known']}, new {counts['new']}. Cross-cutting caveat rows: {len(cross)} "
         f"({', '.join(r['row_id'] + ' ' + r['classification'] for r in cross)}). Method rows: {len(meth)}.", "",
         f"**References:** {len(refs)} verified works cited ({len(refs) - n_new_refs} already in the Phase 0 library, "
         f"{n_new_refs} new, keys {min(new_keys)}-{max(new_keys)} with gaps where earlier candidates were withdrawn); reading depth "
         f"full-text {depth_counts['full-text']}, abstract-only {depth_counts['abstract-only']}, metadata-only "
         f"{depth_counts['metadata-only']}. API requests in the plan: {len(plan)}.", ""]

    L += ["## High-risk items (checked first)", ""]
    for h in cls.get("high_risk", []):
        rr = ", ".join(_cite(refs[k]) for k in h["refs"])
        L += [f"- **{h['item']}.** {_clean(h['finding'])} Rows: {', '.join(h['rows'])}. References: {rr}.", ""]

    L += ["## Phase 3 framing that must be weakened", ""]
    for r in rows:
        ch = r["phase3_framing_change"]
        if r["kind"] != "method" and ch and not ch.lower().startswith(("none", "see ")):
            L.append(f"- **{r['row_id']}** ({r['classification']}): {ch}")
    L.append("")

    L += ["## Summary table", "", "| row | dossier | classification | closest prior (verified) | positioning sentence for the paper |",
          "|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] == "method":
            continue
        keys = [k for k in r["prior_bib_keys"].split("; ") if k]
        by_key = {v["bib_key"]: v for v in refs.values()}
        pri = "; ".join(_cite(by_key[k]) + f" ({by_key[k]['depth']})" for k in keys) or "none found"
        L.append(f"| {r['row_id']} | {r['dossier_result']} | **{r['classification']}** | {pri} | {r['positioning_sentence']} |")
    L.append("")

    L += ["## Per-row detail", ""]
    for r in rows:
        if r["kind"] == "method":
            continue
        L += [f"### {r['row_id']} - {r['classification']}", "",
              f"- **Statement (dossier {r['dossier_result']}).** {r['statement']}",
              f"- **Closest prior statement.** {r['closest_prior_statement']}",
              f"- **Prior references.** {r['prior_dois'] or 'none'} (reading depth: {r['reading_depth'] or '-'})",
              f"- **Searched.** {r['searched_queries']}",
              f"- **Positioning sentence.** {r['positioning_sentence']}",
              f"- **Phase 3 framing.** {r['phase3_framing_change']}",
              f"- **Headline rows (derived).** {r['headline_rows'] or '-'}",
              f"- **Raw files.** reading: {r['raw_reading_files'] or '-'}; verification: {r['raw_verification_files'] or '-'}; "
              f"searches: {r['raw_search_files']}", ""]

    L += ["## Method and data citations", "", "| row | method | verified references |", "|---|---|---|"]
    by_key = {v["bib_key"]: v for v in refs.values()}
    for r in meth:
        keys = [k for k in r["prior_bib_keys"].split("; ") if k]
        note = f" Note: {r['phase3_framing_change']}" if r["phase3_framing_change"] else ""
        L.append(f"| {r['row_id']} | {r['statement']} | {'; '.join(_cite(by_key[k]) for k in keys)}.{note} |")
    L.append("")

    L += ["## UNVERIFIED (excluded from the map and the bib)", ""]
    for u in cls.get("unverified", []):
        L.append(f"- **{u['item']}** - {u['reason']} Searched: {', '.join(u['searched']) or 'not searched'}.")
    L += ["", "## Open full texts requested but not readable (not used)", ""]
    for k, v in (cls.get("fulltext_unreadable") or {}).items():
        L.append(f"- `{k}`: {v}")
    L.append("")

    # headline coverage: map rows sharing the headline row's dossier_result (ids derived at build time)
    cov_rows = []
    if headline is not None and Path(headline).exists():
        with open(headline, encoding="utf-8") as f:
            hl = list(csv.DictReader(f))
        by_dossier = {}
        for r in rows:
            if r["kind"] == "result":
                by_dossier.setdefault(r["dossier_result"], []).append(r["row_id"])
        L += ["## Headline-table coverage", "",
              f"Read from `{_rel(Path(headline))}` at build time ({len(hl)} rows). The CSV column `headline_rows` is derived "
              "the same way: a headline row is covered by every map row with the same `dossier_result`.", "",
              "| headline | block | dossier_result | statement | covered by map rows |", "|---|---|---|---|---|"]
        for h in hl:
            hid, dr = h.get("id", ""), h.get("dossier_result", "")
            cov = by_dossier.get(dr, [])
            cov_rows.append((hid, dr, cov))
            st = (h.get("statement") or "").replace("|", "/")
            L.append(f"| {hid} | {h.get('block', '')} | {dr} | {st} | {', '.join(cov) or '**NOT COVERED**'} |")
        L.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf-8", newline="\n")
    return {"counts": counts, "n_refs": len(refs), "n_new_refs": n_new_refs, "depths": depth_counts,
            "uncovered_headlines": [c[0] for c in cov_rows if not c[2]]}


def run() -> dict:
    """Offline build (no network): regenerate the reference map CSV/MD, the bib and the query log."""
    s = build()
    print(f"reference map: {s['counts']}; {s['n_refs']} verified refs ({s['n_new_refs']} new); "
          f"uncovered headline rows: {s['uncovered_headlines'] or 'none'}")
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="network: execute the query plan, writing raw caches (resumable)")
    f.add_argument("--only", nargs="+", help="query ids or group names")
    f.add_argument("--retry-failed", action="store_true")
    sub.add_parser("build", help="offline: regenerate outputs from the caches")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        from shieldmap.analysis4 import refs_fetch  # network code is imported only here
        refs_fetch.fetch_all(args.only, args.retry_failed)
    else:
        run()


if __name__ == "__main__":
    main()
