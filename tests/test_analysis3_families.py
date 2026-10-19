"""Fast synthetic tests for stage 3 the analysis3/families.py: family grouping, within-x % counts,
family statistics, rank stability and tier listing. No universe data is loaded."""
import numpy as np
import pandas as pd
import pytest

from shieldmap.analysis3 import families as C

Z = {"H": 1, "O": 8, "F": 9, "Pd": 46, "Sn": 50, "Sb": 51, "I": 53, "La": 57, "Ce": 58, "Pr": 59, "Gd": 64,
     "W": 74, "Ir": 77, "Bi": 83}


def test_family_keys_binary_and_ternary():
    k = C.family_keys(["Sb", "Pr", "Pd"], Z)
    assert k["ln"] == "Pr"
    assert k["partners"] == "Pd-Sb"                  # Z order
    assert k["fine"] == "Pr | Pd-Sb"
    assert k["principal"] == "Sb"                     # heaviest partner (Z 51 > 46)
    assert k["coarse"] == "Pr | Sb"


def test_family_keys_multi_lanthanide_and_lanthanide_only():
    k = C.family_keys(["La", "Ce", "Ir"], Z)
    assert k["ln"] == "La+Ce" and k["coarse"] == "La+Ce | Ir"
    k = C.family_keys(["Gd"], Z)
    assert k["partners"] == "" and k["principal"] == "none" and k["fine"] == "Gd | none"


def test_family_keys_non_lanthanide_is_none():
    assert C.family_keys(["W", "O"], Z) is None


def test_family_frame_groups_and_drops():
    rows = np.array([10, 11, 12, 13])
    els = [["Gd", "O"], ["O", "Gd", "H"], ["Bi"], ["Gd", "W", "O"]]
    f = C.family_frame(rows, els, Z)
    assert list(f.row) == [10, 11, 13]               # Bi dropped
    assert list(f.coarse) == ["Gd | O", "Gd | O", "Gd | W"]
    assert list(f.fine) == ["Gd | O", "Gd | H-O", "Gd | O-W"]
    assert f.ln.nunique() == 1


def test_within_counts_inclusive_boundary_and_nan():
    q = np.array([1.0, 1.02, 1.0201, 1.05, 1.0501, np.nan])
    c = C.within_counts(q, 1.0)
    assert c[0.02] == 2                               # 1.0 and exactly 1.02
    assert c[0.05] == 4


def test_family_stats_known_answer():
    fam = pd.Series(["A", "A", "A", "B", "B", "C"])
    q = np.array([1.00, 1.03, 1.10, 1.01, 1.04, 1.20])
    st = C.family_stats(fam, q, frontier=1.0).set_index("family")
    assert st.loc["A", "n"] == 3 and st.loc["A", "best"] == 1.00 and st.loc["A", "median"] == 1.03
    assert st.loc["A", "n_within_2pct"] == 1 and st.loc["A", "n_within_5pct"] == 2
    assert st.loc["B", "n_within_2pct"] == 1 and st.loc["B", "n_within_5pct"] == 2
    assert st.loc["C", "n_within_5pct"] == 0
    assert st.loc["B", "best_ratio"] == pytest.approx(1.01)
    assert list(st["rank"].loc[["A", "B", "C"]]) == [1, 2, 3]


def test_family_stats_ignores_nan_members():
    st = C.family_stats(pd.Series(["A", "A"]), np.array([np.nan, 2.0]), frontier=2.0).set_index("family")
    assert st.loc["A", "n"] == 1 and st.loc["A", "n_within_2pct"] == 1


def test_rank_stability():
    rk = pd.DataFrame({"v1": [1, 2, 15], "v2": [3, 1, 12], "v3": [1, 11, 2]}, index=["A", "B", "C"])
    s = C.rank_stability(rk, top=10)
    assert s.loc["A", "rank_median"] == 1 and s.loc["A", "rank_max"] == 3
    assert s.loc["A", "n_variants_top10"] == 3 and s.loc["A", "n_variants_rank1"] == 2
    assert s.loc["B", "n_variants_top10"] == 2 and s.loc["C", "n_variants_top10"] == 1


def test_tier_listing_known_answer():
    # q with a ±10 % band: tier 1 = rows whose q·0.9 ≤ min q·1.1 = 1.1
    q = np.array([1.0, 1.2, 1.25, 1.5, 3.0, 0.5])
    mask = np.array([True, True, True, True, True, False])   # row 5 (best) outside the set
    lo = np.full(6, -0.10)
    hi = np.full(6, 0.10)
    t = C.tier_listing(q, mask, lo, hi, max_tier=3)
    tiers = dict(zip(t.row, t.tier))
    assert tiers[0] == 1 and tiers[1] == 1            # 1.2·0.9 = 1.08 ≤ 1.1
    assert tiers[2] == 2 and tiers[3] == 2            # tier 2: U* = 1.25·1.1 = 1.375; 1.5·0.9 = 1.35
    assert tiers[4] == 3
    assert 5 not in tiers
    assert t.loc[t.tier == 1, "U_star"].iloc[0] == pytest.approx(1.1)
    assert t.loc[t.tier == 2, "U_star"].iloc[0] == pytest.approx(1.375)
    assert list(t.row) == [0, 1, 2, 3, 4]             # sorted by tier then row (no order by q inside a tier)


def test_tier_listing_class_specific_band():
    q = np.array([1.0, 1.15])
    lo = np.array([-0.05, -0.20])                     # second row's class has a wide lower band
    hi = np.array([0.05, 0.05])
    t = C.tier_listing(q, np.ones(2, bool), lo, hi)
    assert list(t.tier) == [1, 1]                     # 1.15·0.8 = 0.92 ≤ 1.05


def test_no_external_glass_reference_in_code():
    """Reproducibility review F-4: the web-sourced glass-density range is not embedded anywhere."""
    import inspect
    assert not hasattr(C, "GLASS_REF")
    assert "SCHOTT" not in inspect.getsource(C)
