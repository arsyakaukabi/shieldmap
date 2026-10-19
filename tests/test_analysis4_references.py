"""The reference map must regenerate from the cached API responses with networking disabled.

Skips when the git-ignored raw caches (data/raw/references/phase4a/) are absent, as the other raw-file tests do.
"""
from __future__ import annotations

import socket

import pytest

from shieldmap.analysis4 import references as R


def _no_network(*args, **kwargs):
    raise OSError("network access is disabled in this test")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    yield


def test_offline_build_reproduces_committed_csv(tmp_path, offline):
    if not R.API_DIR.exists() or not any(R.API_DIR.glob("*.json")):
        pytest.skip("raw reference caches not present")
    if not R.OUT_CSV.exists():
        pytest.skip("committed reference_map.csv not present")
    out = {k: tmp_path / n for k, n in (("out_csv", "m.csv"), ("out_bib", "r.bib"), ("out_md", "m.md"),
                                        ("out_queries", "q.md"))}
    summary = R.build(**out)  # headline_rows are derived from data/phase4a/headline_table.csv
    assert out["out_csv"].read_bytes() == R.OUT_CSV.read_bytes()
    assert out["out_bib"].read_bytes() == R.OUT_BIB.read_bytes()
    assert sum(summary["counts"].values()) > 0


def test_bib_keeps_all_phase0_entries(offline):
    if not R.OUT_BIB.exists():
        pytest.skip("references_phase4a.bib not built")
    import re
    p0 = set(re.findall(r"@\w+\{(Shield\d+),", (R.PHASE0 / "references.bib").read_text(encoding="utf-8")))
    merged = re.findall(r"@\w+\{(Shield\d+),", R.OUT_BIB.read_text(encoding="utf-8"))
    assert p0 <= set(merged)
    assert len(merged) == len(set(merged))


def test_classification_rows_use_allowed_classes():
    import yaml
    cls = yaml.safe_load(R.CLASSIFICATIONS.read_text(encoding="utf-8"))
    for row in cls["rows"]:
        assert row["classification"] in R.CLASSES, row["id"]
        assert row["searched"], f"{row['id']} names no searches"
