"""Stage 3 the analysis3/density_model.py: density-error model, tiers v2, bootstrap bands v2.

Synthetic tests of the pieces plus data-gated consistency checks of the delivered outputs.
"""
import math

import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis3 import density_model as dm


# ------------------------------------------------------------------------------------- pieces
def test_icc_recovers_a_known_group_correlation():
    rng = np.random.default_rng(0)
    G, n, rho = 3000, 4, 0.3
    s = rng.standard_normal(G)
    z = (math.sqrt(rho) * np.repeat(s, n) + math.sqrt(1 - rho) * rng.standard_normal(G * n))
    out = dm._icc(z, np.repeat(np.arange(G), n))
    assert out["icc"] == pytest.approx(rho, abs=0.03)
    assert dm._icc(rng.standard_normal(G * n), np.repeat(np.arange(G), n))["icc"] == pytest.approx(0.0, abs=0.03)


def test_pool90_is_the_smallest_covering_set():
    wins = {10: 50, 11: 30, 12: 15, 13: 5}
    assert dm.pool(wins, 100, 0.9) == [10, 11, 12]
    assert dm.pool({1: 100}, 100, 0.9) == [1]
    assert dm.pool({1: 89, 2: 1, 3: 10}, 100, 0.9) == [1, 3]


def _cells_frame(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "nelements": 2, "excluded": False,
        "theoretical": np.r_[np.zeros(n - 150, bool), np.ones(150, bool)],
        "rho_mp": np.r_[rng.uniform(1, 5, n - 150), rng.uniform(1, 5, 150)],
        "primary_class": ["oxide"] * (n - 150) + ["halide"] * 150,
        "e_hull": np.r_[np.zeros(n - 250), np.full(100, 0.2), np.zeros(150)],
        "delta": rng.normal(-0.02, 0.03, n)})


def test_cells_fall_back_from_specific_to_general():
    cells, tab = dm.calibration_cells(_cells_frame())
    b = int(dm._bin(np.array([2.0]), dm.BINS_ICSD)[0])
    k = dm._cell_key(cells, False, 2.0, "oxide", 0.0)
    assert k == ("icsd", b, "oxide", "stable") and cells[k].size >= dm.MIN_CELL
    k = dm._cell_key(cells, False, 2.0, "halide", 0.0)            # no ICSD halides → general stable cell
    assert k[2] == "*" and k[0] == "icsd"
    k = dm._cell_key(cells, True, 2.0, "oxide", 0.0)              # theoretical entries never use ICSD cells
    assert k[0] == "theo"
    k = dm._cell_key(cells, False, 50.0, "oxide", 0.0)            # empty bin: still a key (the bin cell)
    assert k == ("icsd", len(dm.BINS_ICSD) - 2, "*", "*")
    assert set(tab.columns) >= {"status", "rho_bin", "class", "stability", "n"}


class _Mini(dm.RowModel):
    """RowModel with synthetic arrays (no data files)."""

    def __init__(self, rho_copula=0.4):
        self.n = 6
        self.measured = np.array([True, False, False, False, False, False])
        self.delta_meas = np.array([0.05, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.resid = np.sort(np.array([-0.002, 0.0, 0.002]))
        x = np.sort(np.random.default_rng(5).normal(-0.03, 0.04, 4000))
        self.cell_arrays = [x]
        self.cell_id = np.zeros(6, int)
        self.chemsys_code = np.array([0, 1, 1, 2, 2, 3])
        self.rho = rho_copula
        self.floor = np.array([-1, -1, -1, -1, -1, 0.0])          # row 5 sits at the density ceiling
        self.cls = np.array(["oxide"] * 6, dtype=object)
        self.class_dist = {}


def test_draws_respect_marginals_measured_rows_and_floor():
    m = _Mini()
    rng = np.random.default_rng(7)
    D = np.array([m.draw(np.arange(6), rng, "copula") for _ in range(4000)])
    assert np.all(np.abs(D[:, 0] - 0.05) <= 1.06 * 0.002)                   # measured: δ_meas + residual
    x = m.cell_arrays[0]
    for q in (0.1, 0.5, 0.9):                                                    # copula keeps the marginal
        assert np.quantile(D[:, 3], q) == pytest.approx(np.quantile(x, q), abs=0.004)
    assert np.all(D[:, 5] >= 0.0)                                                # floor: ρ_true ≤ ρ_max
    r_same = np.corrcoef(D[:, 1], D[:, 2])[0, 1]                                 # same chemical system
    r_diff = np.corrcoef(D[:, 2], D[:, 3])[0, 1]
    assert r_same == pytest.approx(0.4, abs=0.08) and abs(r_diff) < 0.06
    Di = np.array([m.draw(np.arange(6), rng, "iid") for _ in range(2000)])
    assert abs(np.corrcoef(Di[:, 1], Di[:, 2])[0, 1]) < 0.07


def test_candidate_pruning_is_exact():
    rng = np.random.default_rng(3)
    q = rng.uniform(1, 2, (300, 2))
    lo = np.full(300, -0.1); hi = np.full(300, 0.1)
    masks = {"a": np.ones(300, bool)}
    cand = dm.candidates(q, masks, lo, hi, n=5)
    for _ in range(300):                                   # any δ within the bounds keeps the top 5 inside cand
        d = rng.uniform(-0.1, 0.1, 300)
        top = np.argsort(q * (1 + d[:, None]), axis=0)[:5]
        assert np.isin(top, cand).all()


def test_top_and_kinds():
    q = np.arange(60, dtype=float)[::-1, None] * np.array([[1.0, 2.0]])
    v, p = dm._top(q)
    k = dm._kinds(v)
    assert k["best"].tolist() == [0.0, 0.0] and k["10th"].tolist() == [9.0, 18.0]
    assert k["median50"].tolist() == [24.5, 49.0]
    assert p[0].tolist() == [59, 59]


# ------------------------------------------------------------------------------------- outputs
_OUT = dm.OUT / "summary.json"


@pytest.mark.skipif(not _OUT.exists(), reason="density_model.run() outputs absent")
def test_outputs_consistent():
    pt = pd.read_csv(dm.OUT / "bootstrap_prices.csv")
    ok = pt.dropna(subset=["p05", "median", "p95"])
    assert (ok.p05 <= ok["median"] + 1e-12).all() and (ok["median"] <= ok.p95 + 1e-12).all()
    assert set(pt["mode"]) >= {"copula", "iid", "class_iid", "class_iid_abs50", "copula_k8", "copula_rawresid"}
    assert (pt.n_draws >= 5000).all()
    ts = pd.read_csv(dm.OUT / "tier_summary.csv")
    for c in [c for c in ts.columns if c.startswith("coverage")]:
        assert ts[c].between(0, 1).all()
    assert (ts.n_tier1 >= 1).all() and (ts.pool90_copula >= 1).all()
    tr = pd.read_parquet(dm.OUT / "tiers.parquet")
    assert (tr.q_lo <= tr.q_mid + 1e-12).all() and (tr.q_mid <= tr.q_hi + 1e-12).all()
    t1 = tr[tr.tier == 1]
    for key, g in t1.groupby(["scenario", "variant_id", "rule", "set"]):          # tier rule holds
        rest = tr[(tr.scenario == key[0]) & (tr.variant_id == key[1]) & (tr.rule == key[2]) & (tr.set == key[3])]
        assert g.q_lo.max() <= rest.q_hi.min() + 1e-12


def test_measured_rows_are_two_sided_and_curves_reproduce_point():
    """The ceiling does not bind for measured rows in the primary mode (Ir's interval is two-sided),
    the new sensitivity modes are present, and the measured-point curve at 661.657 keV equals the hardening-revision point."""
    if not (dm.OUT / "measured_point_curves.csv").exists():
        pytest.skip("run python -m shieldmap.analysis3.density_model first")
    tr = pd.read_parquet(dm.OUT / "tiers.parquet")
    ir = tr[(tr.scenario == "CS137") & (tr.rule == "R2") & (tr.set == "B4") & (tr.formula_pretty == "Ir")]
    assert len(ir) and (ir.q_lo < ir.q_mid).all()
    pt = pd.read_csv(dm.OUT / "bootstrap_prices.csv")
    assert {"copula_noceiling", "copula_ceilall"} <= set(pt["mode"])
    cur = pd.read_csv(dm.OUT / "measured_point_curves.csv")
    c = cur[(cur.constraint == "B4") & np.isclose(cur.energy_keV, 661.657)].iloc[0]
    assert c.base_point == "Np" and c.constrained_point == "Ir"
    assert c.factor_1pP == pytest.approx((1 + c.price_point) / (1 + c.price_nominal), rel=1e-9)
    assert 0.17 < c.price_point < 0.20 and 0.27 < c.price_nominal < 0.29
