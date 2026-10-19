"""A2 prices: the evaluation loop reproduces core.price / core.top_k on the real map (skipped without data)."""
import numpy as np
import pytest

from shieldmap.analysis import core, prices

pytestmark = pytest.mark.skipif(not (core.P2 / "universe.parquet").exists(), reason="Phase 2 universe not built")


def test_labels():
    assert prices.label("STAB", 0.05) == "STAB(0.05)"
    assert prices.label("EXCL", "W") == "EXCL-W"
    assert prices.label("PB") == "PB"
    assert len(prices.ABUND_FINE) == 37
    assert np.isclose(prices.ABUND_FINE, 1.0).any() and np.isclose(prices.ABUND_FINE, 0.01).any()


def test_evaluate_matches_core():
    cols = [13, 38, 42]
    q = np.ascontiguousarray(np.asarray(core.profiles("thickness"))[:, cols])
    sets = [s for s in prices._sets_single_bundle() if s[1] in ("B4", "GAP(3.1)", "PB")]
    res = prices._evaluate(q, sets, rules=("R2",))
    base = core.base_mask("R2")
    for r, s in zip([r for r in res], [s for s in sets for _ in core.KINDS]):
        ref = core.price(q, base, s[6], r["kind"])
        np.testing.assert_allclose(r["price"], ref, rtol=0, atol=1e-12)
        if r["kind"] == "best":
            for j in range(len(cols)):
                assert r["frontier_row"][j] == core.top_k(q[:, j], base & s[6], 1)[0]


def test_mono_matches_table_c_cs137():
    """Single line: Table C thickness at T is ln(1/T)/μ, so q_mono(661.657) × ln(1/T) equals it."""
    mono = prices.mono_requirements()["thickness"][:, prices.KEY_E.index(661.657)]
    v = core.variants()
    j = 38
    T = float(v.T_target[j])
    tc = np.asarray(core.profiles("thickness"))[:, j]
    ok = core.base_mask("R2")
    np.testing.assert_allclose(mono[ok] * np.log(1 / T), tc[ok], rtol=2e-6)
