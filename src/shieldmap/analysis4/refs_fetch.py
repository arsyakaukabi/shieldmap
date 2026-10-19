"""The reference reconciliation, fetch mode: bibliographic API requests and open full texts, cached with provenance.

This module is the ONLY part of the reference reconciliation that touches the network. The offline build
(``shieldmap.analysis4.references.run``) never imports it.

Every request listed in ``config/phase4a/reference_queries.yaml`` is executed once and its raw response is
saved under ``data/raw/references/phase4a/api/<query_id>.json`` together with the endpoint, the exact query
string (API key stripped), the UTC timestamp and the HTTP status. Open full texts are saved under
``data/raw/references/phase4a/fulltext/`` with a sidecar ``<name>.json`` holding the URL, SHA-256 and
retrieval metadata (and, for PDFs, an extracted ``.txt`` with its own SHA-256 recorded in the sidecar).

Fetching is resumable: a query whose cache file exists with HTTP status 200 (or 404, which is a valid
answer for identifier checks) is skipped. Run:

    python -m shieldmap.analysis4.references fetch [--only ID ...] [--retry-failed]

Credentials: ``OPENALEX_API_KEY`` is read from the repository ``.env`` (or the environment) and is never
written anywhere; ``OPENALEX_MAILTO`` is not used. Semantic Scholar, Crossref, DataCite, doi.org and arXiv
are queried without keys, with back-off on HTTP 429/5xx.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shieldmap import paths

RAW_REFS = paths.RAW / "references" / "phase4a"
API_DIR = RAW_REFS / "api"
FT_DIR = RAW_REFS / "fulltext"
PLAN = paths.CONFIG / "phase4a" / "reference_queries.yaml"

UA = "shieldmap-phase4a-reference-reconciliation/1.0 (research; no-mailto)"
OA_SELECT = ("id,doi,title,display_name,publication_year,type,authorships,primary_location,open_access,"
             "best_oa_location,abstract_inverted_index,cited_by_count,referenced_works,ids")
S2_FIELDS = "title,year,externalIds,abstract,authors,venue,openAccessPdf,citationCount"
_MIN_GAP = {"openalex": 0.15, "crossref": 1.0, "s2": 3.5, "arxiv": 3.5, "datacite": 0.5, "doi_org": 0.5,
            "europepmc": 0.5, "fulltext": 1.0}
_last: dict[str, float] = {}
_KEY_EXHAUSTED = False  # set once the keyed OpenAlex pool answers 429; later requests go anonymous


def _env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    env = paths.REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v or None
    return None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _throttle(service: str) -> None:
    gap = _MIN_GAP.get(service, 1.0)
    dt = time.time() - _last.get(service, 0.0)
    if dt < gap:
        time.sleep(gap - dt)
    _last[service] = time.time()


def _request(service: str, endpoint: str, params: dict | None, *, secret_params: dict | None = None,
             accept: str = "application/json", binary: bool = False, tries: int = 6):
    """GET with back-off. Returns (status, final_url_sanitized, headers, body bytes)."""
    import requests  # network-only import

    params = dict(params or {})
    full = dict(params)
    full.update(secret_params or {})
    delay = 5.0
    last_exc = None
    for _ in range(tries):
        _throttle(service)
        try:
            r = requests.get(endpoint, params=full or None, headers={"User-Agent": UA, "Accept": accept},
                             timeout=90, allow_redirects=True)
        except Exception as exc:  # noqa: BLE001 - network errors are recorded, then retried
            last_exc = exc
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            time.sleep(min(float(ra), 120.0) if ra and ra.isdigit() else delay)  # never wait hours
            delay = min(delay * 2, 120)
            continue
        url = r.url
        for k in (secret_params or {}):
            url = re.sub(rf"([?&]){re.escape(k)}=[^&]*&?", r"\1", url).rstrip("&?")
        return r.status_code, url, dict(r.headers), r.content
    if last_exc is not None:
        return -1, endpoint, {}, repr(last_exc).encode()
    return 429, endpoint, {}, b""


def _save_api(qid: str, rec: dict) -> Path:
    API_DIR.mkdir(parents=True, exist_ok=True)
    p = API_DIR / f"{qid}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _endpoint(q: dict) -> tuple[str, str, dict, dict, str]:
    """(service, endpoint, params, secret_params, accept) for a plan entry."""
    kind = q["kind"]
    key = None if _KEY_EXHAUSTED else _env_key("OPENALEX_API_KEY")
    oa_secret = {"api_key": key} if key else {}
    if kind == "openalex_search":
        p = {"search": q["query"], "per-page": q.get("per_page", 25), "select": OA_SELECT}
        if q.get("filter"):
            p["filter"] = q["filter"]
        if q.get("sort"):
            p["sort"] = q["sort"]
        return "openalex", "https://api.openalex.org/works", p, oa_secret, "application/json"
    if kind == "openalex_filter":
        p = {"filter": q["filter"], "per-page": q.get("per_page", 50), "select": OA_SELECT}
        if q.get("sort"):
            p["sort"] = q["sort"]
        return "openalex", "https://api.openalex.org/works", p, oa_secret, "application/json"
    if kind == "openalex_work":
        ident = q["work"]
        if ident.lower().startswith("10."):
            ident = "https://doi.org/" + ident
        return ("openalex", "https://api.openalex.org/works/" + urllib.parse.quote(ident, safe=":/"),
                {"select": OA_SELECT}, oa_secret, "application/json")
    if kind == "crossref_search":
        p = {"query.bibliographic": q["query"], "rows": q.get("rows", 20)}
        if q.get("filter"):
            p["filter"] = q["filter"]
        return "crossref", "https://api.crossref.org/works", p, {}, "application/json"
    if kind == "crossref_work":
        return ("crossref", "https://api.crossref.org/works/" + urllib.parse.quote(q["doi"], safe=""), {}, {},
                "application/json")
    if kind == "datacite_doi":
        return ("datacite", "https://api.datacite.org/dois/" + urllib.parse.quote(q["doi"], safe=""), {}, {},
                "application/json")
    if kind == "doi_handle":
        return ("doi_org", "https://doi.org/api/handles/" + q["doi"], {}, {}, "application/json")
    if kind == "s2_search":
        return ("s2", "https://api.semanticscholar.org/graph/v1/paper/search",
                {"query": q["query"], "limit": q.get("limit", 20), "fields": S2_FIELDS}, {}, "application/json")
    if kind == "s2_paper":
        return ("s2", "https://api.semanticscholar.org/graph/v1/paper/DOI:" + q["doi"],
                {"fields": S2_FIELDS}, {}, "application/json")
    if kind == "arxiv_search":
        return ("arxiv", "http://export.arxiv.org/api/query",
                {"search_query": q["query"], "max_results": q.get("max_results", 20)}, {}, "application/atom+xml")
    if kind == "arxiv_id":
        return ("arxiv", "http://export.arxiv.org/api/query", {"id_list": q["arxiv"]}, {}, "application/atom+xml")
    if kind == "europepmc_search":
        return ("europepmc", "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                {"query": q["query"], "format": "json", "pageSize": q.get("rows", 25), "resultType": "core"}, {},
                "application/json")
    raise ValueError(f"unknown kind {kind}")


def fetch_query(q: dict, retry_failed: bool = False) -> Path | None:
    qid = q["id"]
    out = API_DIR / f"{qid}.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("http_status") in (200, 404) or not retry_failed:
            return out
    if q["kind"] == "fulltext":
        return fetch_fulltext(q)
    service, endpoint, params, secret, accept = _endpoint(q)
    ts = _utc()
    status, url, headers, body = _request(service, endpoint, params, secret_params=secret, accept=accept,
                                          tries=3 if secret else 6)
    key_fallback = False
    if status == 429 and service == "openalex" and secret:
        # keyed pool exhausted: one anonymous re-send (no key, no mailto), recorded in the cache
        status, url, headers, body = _request(service, endpoint, params, secret_params={}, accept=accept, tries=4)
        secret, key_fallback = {}, True
        global _KEY_EXHAUSTED
        _KEY_EXHAUSTED = True
    rec = {
        "query_id": qid,
        "kind": q["kind"],
        "purpose": q.get("purpose", ""),
        "service": service,
        "endpoint": endpoint,
        "query_string": urllib.parse.urlencode(params),
        "params": params,
        "url": url,
        "api_key_used": bool(secret),
        "api_key_fallback": key_fallback,
        "timestamp_utc": ts,
        "http_status": status,
        "content_type": headers.get("Content-Type", headers.get("content-type", "")),
    }
    text = body.decode("utf-8", errors="replace")
    try:
        rec["response_json"] = json.loads(text)
    except ValueError:
        rec["response_text"] = text
    return _save_api(qid, rec)


def fetch_fulltext(q: dict) -> Path:
    """Download one open full text (PDF/HTML/XML) for reading; sidecar holds URL and SHA-256."""
    FT_DIR.mkdir(parents=True, exist_ok=True)
    qid = q["id"]
    ts = _utc()
    status, url, headers, body = _request("fulltext", q["url"], None, accept=q.get("accept", "*/*"))
    ctype = headers.get("Content-Type", headers.get("content-type", ""))
    ext = ".pdf" if (body[:4] == b"%PDF") else (".xml" if "xml" in ctype else ".html")
    fpath = FT_DIR / f"{qid}{ext}"
    side = {"query_id": qid, "kind": "fulltext", "purpose": q.get("purpose", ""), "doi": q.get("doi", ""),
            "url": q["url"], "final_url": url, "timestamp_utc": ts, "http_status": status, "content_type": ctype,
            "file": fpath.name if status == 200 else None}
    if status == 200:
        fpath.write_bytes(body)
        side["sha256"] = hashlib.sha256(body).hexdigest()
        side["bytes"] = len(body)
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                import io
                rd = PdfReader(io.BytesIO(body))
                txt = "\n\n".join((pg.extract_text() or "") for pg in rd.pages)
                tpath = FT_DIR / f"{qid}.txt"
                tpath.write_text(txt, encoding="utf-8")
                side["text_file"] = tpath.name
                side["text_sha256"] = hashlib.sha256(tpath.read_bytes()).hexdigest()
                side["pages"] = len(rd.pages)
            except Exception as exc:  # noqa: BLE001
                side["text_extraction_error"] = repr(exc)
    # the sidecar doubles as the api-level record so the plan's fetch status is visible in one place
    (FT_DIR / f"{qid}.meta.json").write_text(json.dumps(side, ensure_ascii=False, indent=1), encoding="utf-8")
    return _save_api(qid, {**side, "service": "fulltext", "endpoint": q["url"], "query_string": "",
                           "params": {}, "note": "full text stored under fulltext/; see sidecar .meta.json"})


def load_plan() -> list[dict]:
    with open(PLAN, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    out = []
    for grp in plan["groups"]:
        for q in grp["queries"]:
            q = dict(q)
            q.setdefault("purpose", grp.get("purpose", ""))
            q["group"] = grp["name"]
            out.append(q)
    ids = [q["id"] for q in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate query ids: {sorted(dup)}")
    return out


def fetch_all(only: list[str] | None = None, retry_failed: bool = False, verbose: bool = True) -> None:
    plan = load_plan()
    for q in plan:
        if only and q["id"] not in only and q.get("group") not in only:
            continue
        p = fetch_query(q, retry_failed=retry_failed)
        if verbose and p is not None:
            rec = json.loads(p.read_text(encoding="utf-8"))
            print(f"{q['id']:<40} {rec.get('http_status')}", flush=True)
