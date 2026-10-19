"""Stage 4a core checks: artefact flag, natural-radioactivity flag, reproduction of stage 3, headline table rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shieldmap import paths
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common

S = common.P4 / "S"
needs_s = pytest.mark.skipif(not (S / "bands.parquet").exists(), reason="run python -m shieldmap.analyze4a first")


def test_artefact_flag_definition():
    u = c3.universe()
    art = common.artefact_mask("phase4a")
    mid = dict(zip(u.material_id, art))
    assert mid["mp-107"]          # Pu, 26.6 g/cm³, ICSD-tagged, e_hull 0.55
    assert mid["mp-613989"]       # Pu, 24.3 g/cm³ (density cap)
    assert not mid["mp-101"]      # Ir
    assert not mid["mp-11534"]    # alpha-Np (measured, ICSD, on the hull)
    old = common.artefact_mask("phase3")
    assert (old & ~art).sum() == 0                                  # the new flag only adds entries
    b4 = c3.bundle_mask("B4")
    assert not (b4 & art).any()                                     # B4 never contains an artefact entry


def test_natural_activity_threshold_and_moderate_set():
    from shieldmap.analysis4 import natrad
    act, iso, a_k = natrad.activities()
    assert abs(a_k - 31.7) / 31.7 < 0.01                            # natural K, about 31.7 Bq/g
    added = set(act.loc[act.moderate & ~act.strict, "element"])
    assert added == {"Rb", "Sm", "Lu", "Re"}
    assert not bool(act.loc[act.element == "K", "moderate"].iloc[0])  # "exceeds" is strict
    named = set(act.loc[act.named & ~act.strict, "element"])
    assert named == {"La", "Lu", "Re"}


@needs_s
def test_measured_point_reproduces_phase3():
    """RAD against F0 at the measured-density point (best_clean, R2) equals stage 3's measured-point increment
    table, which used the same density point and the phase3 artefact flag."""
    b = pd.read_parquet(S / "bands.parquet")
    p3 = pd.read_csv(paths.DATA / "phase3" / "T" / "measured_point_increments.csv")
    for key, vid in (("CS137|v38", 38), ("CO60|v42", 42)):
        mine = b[(b.key == key) & (b.label == "RAD|F0") & (b.basis == "thickness") & (b.density == "measured")
                 & (b.artefact_def == "phase3")].price.iloc[0]
        ref = p3[(p3.variant_id == vid) & (p3.constraint == "RAD") & (p3.baseline == "F0") & (p3.density == "point")].price.iloc[0]
        assert abs(mine - ref) < 1e-9


@needs_s
def test_headline_table_rules():
    t = pd.read_csv(common.P4 / "headline_table.csv")
    h = t[t.block == "headline"]
    assert len(h) > 50
    assert not h.artefact_reference.astype(bool).any()               # no headline row rests on an artefact
    assert h.source_file.notna().all() and (h.source_file.str.len() > 0).all()
    assert h.sensitivity_axes.notna().all()
    price = h[h.unit == "%"]
    assert price.band_lo_pct.notna().all() and price.band_hi_pct.notna().all()
    assert (price.band_lo_pct <= price.value_pct + 1e-9).all() and (price.value_pct <= price.band_hi_pct + 1e-9).all()
    for col in ("statement", "sensitivity_values"):
        txt = " ".join(t[col].fillna("").astype(str)).lower()
        for w in (" cost", "lead equivalen", "non-toxic"):
            assert w not in txt
