"""Frontiers with six kinds, tiers, textbook ambient phases, the hardening revision tables."""
import numpy as np

from shieldmap.analysis3 import common, frontiers


def test_frontier_all_matches_core(monkeypatch):
    rng = np.random.default_rng(1)
    n = 400
    q = rng.lognormal(size=(n, 5))
    icsd = rng.random(n) < 0.5
    clean = rng.random(n) < 0.8
    monkeypatch.setattr(common, "icsd_mask", lambda: icsd)
    monkeypatch.setattr(common, "clean_mask", lambda: clean)
    mask = rng.random(n) < 0.7
    f = frontiers._frontier_all(q, mask)
    for k in ("best", "10th", "median50", "best_icsd", "best_clean", "median50_clean"):
        np.testing.assert_allclose(f[k], common.frontier(q, mask, k))
    np.testing.assert_allclose(f["best_icsd"], common.frontier(q, mask & icsd, "best"))
    np.testing.assert_allclose(f["best_clean"], common.frontier(q, mask & clean, "best"))
    np.testing.assert_array_equal(f["best_row"], common.argbest(q, mask))
    np.testing.assert_array_equal(f["best_clean_row"], common.argbest(q, mask & clean))


def test_median_topn_and_marginal_masks():
    v = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert frontiers._median_topn(v, 1) == 1.0
    assert frontiers._median_topn(v, 4) == 2.5
    assert np.isnan(frontiers._median_topn(v, 6))
    b4 = common.bundle_mask("B4")
    for x in ("RAD", "PB", "STAB", "EXP"):
        m = frontiers._named_mask(f"B4-{x}")
        assert (m | ~b4).all()                       # B4 ⊆ B4 without X
        assert m.sum() > b4.sum()


def test_tiers_synthetic():
    q = np.array([1.00, 1.05, 1.30, 1.31, 2.0, 0.5])
    mask = np.array([True, True, True, True, True, False])
    lo = np.full(6, -0.10); hi = np.full(6, 0.10)
    t = frontiers.tiers(q, mask, lo, hi, max_tier=3)
    # U* = 1.10; lower bounds 0.90, 0.945, 1.17 … → tier 1 = {0, 1}
    assert list(t) == [1, 1, 2, 2, 3, 0]


def test_tiers_respects_class_band():
    q = np.array([1.0, 1.15])
    t = frontiers.tiers(q, np.ones(2, bool), np.array([-0.02, -0.15]), np.array([0.02, 0.15]), 2)
    assert list(t) == [1, 1]          # 1.15·0.85 = 0.98 ≤ 1.02


def test_textbook_ambient_phases():
    tp = frontiers.textbook_phases().set_index(["formula", "spacegroup_number"])
    assert tp.loc[("Sn", 141), "material_id"] == "mp-84"          # beta-Sn, not R2's alpha-Sn
    assert not tp.loc[("Sn", 141), "same_as_R2"]
    assert tp.loc[("W", 229), "same_as_R2"]
