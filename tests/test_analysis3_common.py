"""Stage 3 shared infrastructure (analysis3/common.py)."""
import math

import numpy as np
import pytest

from shieldmap.analysis3 import common as c


def test_classes_precedence():
    assert c.report_classes(["Fe", "O"]) == {"oxide", "3d_magnetic"}
    assert c.primary_class(["Fe", "O"]) == "3d_magnetic"
    assert c.primary_class(["Gd", "Fe", "O"]) == "lanthanide"
    assert c.primary_class(["U", "Gd"]) == "f_electron"
    assert c.primary_class(["Cs", "I"]) == "halide"
    assert c.primary_class(["W"]) == "intermetallic"
    assert c.primary_class(["Zn", "S"]) == "chalcogenide"
    assert c.primary_class(["B", "N"]) == "other"
    assert c.primary_class(["Ba", "S", "O"]) == "oxide"


def test_best_icsd_frontier_synthetic(monkeypatch):
    q = np.array([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(c, "icsd_mask", lambda: np.array([False, False, True, True]))
    m = np.ones(4, bool)
    assert c.frontier(q, m, "best") == 1.0
    assert c.frontier(q, m, "best_icsd") == 3.0
    assert c.price(q, m, np.array([True, False, False, True]), "best_icsd") == pytest.approx(4 / 3 - 1)


def test_mono_requirement_matches_table_c_cs137():
    """For a single line, Table C thickness = ln(1/T)/μ(661.657 keV) exactly (up to float32)."""
    v = c.variants()
    vid = int(v[(v.spectrum == "Cs-137") & v.primary & np.isclose(v.T_target, 0.01)]["variant_id"].iloc[0])
    rows = np.arange(0, 150000, 7919)
    qm = c.mono_requirement("thickness", True, rows=rows, e_idx=[c.energy_index(661.657)])[:, 0]
    qt = np.asarray(c.profiles("thickness")[rows, vid], dtype=float)
    np.testing.assert_allclose(qt, math.log(100) * qm, rtol=2e-6)


def test_artefact_flag_puNp3():
    r = c.row_of("mp-1186824")
    assert c.artefact_flag([r])[0]


def test_text_digests_accept_either_line_ending(tmp_path):
    import hashlib
    body = b"a\nb\n"
    f = tmp_path / "x.md"
    f.write_bytes(body.replace(b"\n", b"\r\n"))
    lf, crlf = c._text_digests(f)
    assert lf == hashlib.sha256(body).hexdigest()
    assert crlf == hashlib.sha256(body.replace(b"\n", b"\r\n")).hexdigest()


def test_clean_kinds_synthetic(monkeypatch):
    q = np.array([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(c, "clean_mask", lambda: np.array([False, True, True, True]))
    m = np.ones(4, bool)
    assert c.frontier(q, m, "best_clean") == 2.0
    assert c.price(q, m, np.array([True, False, True, True]), "best_clean") == pytest.approx(3 / 2 - 1)
    assert c.HEADLINE_KIND == "best_clean"


def test_clean_mask_excludes_artefacts():
    r = c.row_of("mp-1186824")                      # PuNp3, theoretical, e_hull ≈ 0.46
    assert not c.clean_mask()[r]
    assert c.clean_mask().sum() == len(c.universe()) - c.artefact_flag(np.arange(len(c.universe()))).sum()
