"""Stage 3 the continuum: synthetic fixtures with known answers (no Table B/C access)."""
import math

import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis import core
from shieldmap.analysis3 import continuum as C


def _brute(q, member):
    out = {k: core.frontier(q, member, k) for k in C.KINDS3}
    rows = []
    for j in range(q.shape[1]):
        t = core.top_k(q[:, j], member & np.isfinite(q[:, j]), 1)
        rows.append(t[0] if t.size else -1)
    out["best_row"] = np.array(rows)
    return out


@pytest.mark.parametrize("k", [30, 120, 5000])
def test_topk_walk_equals_brute_force(k):
    rng = np.random.default_rng(0)
    n, c = 700, 9
    q = rng.lognormal(size=(n, c))
    q[rng.random((n, c)) < 0.01] = np.inf                       # a few non-finite requirements
    base = rng.random(n) < 0.8
    tk = C.TopK(q, base, k)
    icsd = rng.random(n) < 0.5
    for dens in (1.0, 0.6, 0.1, 0.03, 0.005, 0.0):
        mem = rng.random(n) < dens
        got = tk.frontiers(mem, C.KINDS3, icsd)
        ref = _brute(q, base & mem)
        for kk in C.KINDS3:
            np.testing.assert_allclose(got[kk], ref[kk], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(got["best_row"], ref["best_row"])
        np.testing.assert_allclose(got["best_icsd"], core.frontier(q, base & mem & icsd, "best"), equal_nan=True)
    # the full baseline (member None)
    got = tk.frontiers(None)
    ref = _brute(q, base)
    for kk in C.KINDS3:
        np.testing.assert_allclose(got[kk], ref[kk], equal_nan=True)


def test_topk_element_exclusion_shortcut_vs_brute_force():
    """EXCL-X via the walk (K far smaller than the set) equals the price computed from scratch."""
    rng = np.random.default_rng(3)
    n, c, n_el = 2000, 6, 12
    q = rng.lognormal(sigma=0.5, size=(n, c))
    contains = rng.random((n, n_el)) < rng.uniform(0.02, 0.9, n_el)   # some elements in most compounds
    base = rng.random(n) < 0.9
    b4 = rng.random(n) < 0.3
    for bl in (base, base & b4):
        tk = C.TopK(q, bl, 25)
        f0 = tk.frontiers(None)
        for x in range(n_el):
            m = ~contains[:, x]
            fx = tk.frontiers(m)
            for kind in C.KINDS3:
                ref = core.price(q, bl, m, kind)
                np.testing.assert_allclose(fx[kind] / f0[kind] - 1, ref, rtol=1e-12, equal_nan=True)
    assert tk.n_fallback > 0                                     # the fallback path was exercised


def test_members_equals_sorted_rows():
    rng = np.random.default_rng(5)
    q = rng.lognormal(size=(300, 4))
    base = rng.random(300) < 0.9
    mem = rng.random(300) < 0.2
    tk = C.TopK(q, base, 20)
    got = tk.members(mem, 10)
    for j in range(4):
        np.testing.assert_array_equal(got[:, j], core.top_k(q[:, j], base & mem, 10))


def test_frontier_vs_energy_known_answer():
    """Three compounds: q_A = E, q_B = 2, q_C = 3. The price of excluding A is 2/min(E, 2) − 1; excluding C adds nothing."""
    E = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    q = np.vstack([E, np.full_like(E, 2.0), np.full_like(E, 3.0)])
    tk = C.TopK(q, np.ones(3, bool), 2)
    f0 = tk.frontiers(None, ("best",))
    pa = tk.frontiers(np.array([False, True, True]), ("best",))["best"] / f0["best"] - 1
    np.testing.assert_allclose(pa, 2 / np.minimum(E, 2) - 1)
    pc = tk.frontiers(np.array([True, True, False]), ("best",))["best"] / f0["best"] - 1
    np.testing.assert_allclose(pc, 0.0)
    np.testing.assert_array_equal(f0["best_row"], [0, 0, 0, 0, 1, 1])


def _game(fn, players):
    import itertools
    v = {}
    for r in range(len(players) + 1):
        for s in itertools.combinations(players, r):
            v[frozenset(s)] = fn(frozenset(s))
    return v


def test_shapley_additive_game():
    P = ["a", "b", "c", "d"]
    w = {"a": np.array([1.0, 0.0]), "b": np.array([2.0, 1.0]), "c": np.array([0.0, 0.5]), "d": np.array([0.3, 0.0])}
    v = _game(lambda s: sum((w[p] for p in s), np.zeros(2)), P)
    phi = C.shapley_vec(P, v)
    for p in P:
        np.testing.assert_allclose(phi[p], w[p], atol=1e-12)
    np.testing.assert_allclose(C.shapley_interaction(P, v, "a", "b"), 0.0, atol=1e-12)


def test_shapley_substitutes_and_complements():
    P = ["PB", "RAD", "STAB", "EXP"]
    # substitutes: STAB and EXP remove the same phase → value 1 if either is present
    v = _game(lambda s: float(bool(s & {"STAB", "EXP"})), P)
    phi = C.shapley_vec(P, v)
    assert phi["STAB"] == pytest.approx(0.5) and phi["EXP"] == pytest.approx(0.5)
    assert phi["PB"] == pytest.approx(0.0) and phi["RAD"] == pytest.approx(0.0)
    assert C.shapley_interaction(P, v, "STAB", "EXP") == pytest.approx(-1.0)
    # complements: value only when both are present
    v2 = _game(lambda s: float({"STAB", "EXP"} <= s), P)
    assert C.shapley_interaction(P, v2, "STAB", "EXP") == pytest.approx(1.0)
    # efficiency: Σφ = v(N) − v(∅) for a non-trivial game
    v3 = _game(lambda s: math.log(1 + len(s) ** 2 + 0.3 * ("RAD" in s)), P)
    phi3 = C.shapley_vec(P, v3)
    assert sum(phi3.values()) == pytest.approx(v3[frozenset(P)] - v3[frozenset()])


def test_bootstrap_scaling_convention():
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    d = np.array([0.1, -0.2])
    np.testing.assert_allclose(C.true_requirement(q, d), [[1.1, 2.2], [2.4, 3.2]])
    # a common δ leaves every price unchanged
    rng = np.random.default_rng(0)
    qq = rng.lognormal(size=(200, 3))
    m = rng.random(200) < 0.3
    p0 = core.price(qq, np.ones(200, bool), m, "best")
    p1 = core.price(C.true_requirement(qq, np.full(200, 0.07)), np.ones(200, bool), m, "best")
    np.testing.assert_allclose(p0, p1, rtol=1e-12)
    # δ that only makes one row's material denser in reality (δ < 0) lowers its requirement
    assert C.true_requirement(np.array([2.0]), np.array([-0.5]))[0] == pytest.approx(1.0)


def test_draw_deltas_modes():
    cls = np.array(["x"] * 500 + ["y"] * 300)
    dist = {"x": np.array([-0.1, 0.0, 0.1]), "y": np.array([0.5, 0.7])}
    rng = np.random.default_rng(1)
    d = C.draw_deltas(cls, dist, rng, "iid")
    assert set(np.unique(d[:500])) <= {-0.1, 0.0, 0.1} and set(np.unique(d[500:])) <= {0.5, 0.7}
    assert np.unique(d[:500]).size == 3                          # independent draws use the whole pool
    d = C.draw_deltas(cls, dist, rng, "class_shift")
    assert np.unique(d[:500]).size == 1 and np.unique(d[500:]).size == 1
    assert d[0] in dist["x"] and d[500] in dist["y"]
    # mixed: per-row mean and variance equal the class pool's (exact expectation, checked by enumeration)
    x = dist["x"]; mu = x.mean()
    vals = [mu + math.sqrt(C.MIXED_RHO) * (s - mu) + math.sqrt(1 - C.MIXED_RHO) * (e - mu) for s in x for e in x]
    assert np.mean(vals) == pytest.approx(mu)
    assert np.var(vals) == pytest.approx(np.var(x))
    d = C.draw_deltas(cls, dist, rng, "mixed")
    assert np.unique(d[:500]).size <= 3


def test_headline_select():
    kinds = ["best", "median50", "best_clean"]
    df = pd.DataFrame({"e": [1, 1, 1, 2, 2, 2], "kind": kinds * 2, "price": [0.5, 0.1, 0.3, 0.2, 0.3, 0.25],
                       "artefact_any": [True, True, True, False, False, False]})
    h = C.headline_select(df, ["e"]).set_index("e")
    assert (h.headline_kind == "best_clean").all()
    assert h.loc[1, "price_headline"] == pytest.approx(0.3) and h.loc[2, "price_headline"] == pytest.approx(0.25)
    # the superseded per-cell rule is kept as a sensitivity column
    assert h.loc[1, "legacy_rule_kind"] == "median50" and h.loc[1, "price_legacy_rule"] == pytest.approx(0.1)
    assert h.loc[2, "legacy_rule_kind"] == "best" and h.loc[2, "price_legacy_rule"] == pytest.approx(0.2)
