"""Shared stage 2 analysis core: universe, scenarios, constraints, frontier and price.

Every stage 2 module imports this; nothing re-implements the frontier or the price.

Definitions
  requirement q_S,b(m)   Table C `thickness_req_cm` (b = "thickness") or `areal_mass_req_g_cm2`
                         (b = "areal"); lower is better
  frontier q*(F)         kind "best":     min over m ∈ F
                         kind "10th":     10th-smallest over F
                         kind "median50": median of the 50 smallest over F
                         (NaN when F has fewer members than the kind needs)
  price P(K)             q*(F0 ∩ K) / q*(F0) − 1   (a fraction; ×100 for percent)

All functions are vectorised over profile variants: a requirement matrix has shape (n, V).
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from shieldmap import paths

P2 = paths.DATA / "phase2"
REPORTS = paths.ROOT / "reports" / "phase2"
FIGURES = REPORTS / "figures"
KINDS = ("best", "10th", "median50")
BASES = ("thickness", "areal")
RULES = ("R1", "R2", "R3")
PRIMARY_RULE = "R2"

# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def universe() -> pd.DataFrame:
    """One row per Table B / Table C row (same order); see analysis/universe.py."""
    return pd.read_parquet(P2 / "universe.parquet")


@lru_cache(maxsize=2)
def profiles(basis: str) -> np.ndarray:
    """(N × 264) float32 requirement matrix, memory-mapped; columns are Table C variant_id."""
    if basis not in BASES:
        raise ValueError(basis)
    name = "profiles_thickness.npy" if basis == "thickness" else "profiles_areal.npy"
    return np.load(P2 / name, mmap_mode="r")


@lru_cache(maxsize=1)
def variants() -> pd.DataFrame:
    """Table C profile variants with scenario labels.

    scenario: DIAG (RQR tubes, Pb-thickness targets), CS137, CO60
    primary:  air-kerma response and with-coherent
    """
    v = pd.read_parquet(paths.PROCESSED / "profile_variants.parquet").copy()
    v["scenario"] = np.select([v.source_kind == "tube", v.spectrum == "Cs-137", v.spectrum == "Co-60"],
                              ["DIAG", "CS137", "CO60"], "")
    v["quality"] = v.spectrum.str.extract(r"^(RQR\d+)")[0]
    v["anode_angle"] = pd.to_numeric(v.spectrum.str.extract(r"_th(\d+)")[0], errors="coerce")
    v["primary"] = (v.response == "air_kerma") & v.coherent
    return v


def scenario_variants(scenario: str, primary_only: bool = True) -> pd.DataFrame:
    v = variants()
    v = v[v.scenario == scenario]
    return v[v.primary] if primary_only else v


def default_basis(scenario: str) -> str:
    """Primary basis: areal mass for DIAG, thickness for the radionuclides."""
    return "areal" if scenario == "DIAG" else "thickness"


def element_list(u: pd.DataFrame | None = None) -> list[str]:
    u = universe() if u is None else u
    return sorted({s for lst in u.loc[u.eligible, "elements"] for s in lst})


@lru_cache(maxsize=1)
def _contains_matrix() -> tuple[np.ndarray, tuple[str, ...]]:
    """(N × n_el) boolean containment matrix from Table B's per-element flags."""
    tb = pd.read_parquet(paths.PROCESSED / "table_b_compounds.parquet",
                         columns=None)
    cols = [c for c in tb.columns if c.startswith("contains_") and c not in
            ("contains_radioactive", "contains_noble_gas")]
    return tb[cols].to_numpy(dtype=bool), tuple(c[len("contains_"):] for c in cols)


def contains(element: str) -> np.ndarray:
    mat, syms = _contains_matrix()
    if element not in syms:
        return np.zeros(mat.shape[0], dtype=bool)
    return mat[:, syms.index(element)]


# ---------------------------------------------------------------------------------------------
# candidate sets and constraints (boolean masks over all N rows)
# ---------------------------------------------------------------------------------------------
def base_mask(rule: str = PRIMARY_RULE, u: pd.DataFrame | None = None) -> np.ndarray:
    """Baseline-eligible universe under a representative rule (F0)."""
    u = universe() if u is None else u
    if rule not in RULES:
        raise ValueError(rule)
    return u[f"in_{rule}"].to_numpy(dtype=bool)


def constraint_mask(code: str, param=None, u: pd.DataFrame | None = None) -> np.ndarray:
    """Rows that SATISFY a constraint. Unknown attribute values fail the constraint."""
    u = universe() if u is None else u
    if code == "PB":
        return ~_contains_from(u, "Pb")
    if code == "RAD":
        return ~u["contains_radioactive"].to_numpy(dtype=bool)
    if code == "STAB":
        return (u["energy_above_hull"] <= float(param)).to_numpy()
    if code == "EXP":
        return ~u["theoretical"].to_numpy(dtype=bool)
    if code == "EXCL":
        return ~_contains_from(u, str(param))
    if code == "GAP":
        return (u["band_gap"] >= float(param)).fillna(False).to_numpy(dtype=bool)
    if code == "ABUND":
        return (u["min_crustal_abundance_ppm"] >= float(param)).fillna(False).to_numpy(dtype=bool)
    raise ValueError(f"unknown constraint {code!r}")


def _contains_from(u: pd.DataFrame, element: str) -> np.ndarray:
    if "elements" in u.columns and u is not universe():
        return np.array([element in lst for lst in u["elements"]])
    return contains(element)


BUNDLES: dict[str, list[tuple[str, object]]] = {
    "B0": [],
    "B1": [("RAD", None)],
    "B2": [("RAD", None), ("PB", None)],
    "B3": [("RAD", None), ("PB", None), ("STAB", 0.05)],
    "B4": [("RAD", None), ("PB", None), ("STAB", 0.05), ("EXP", None)],
    "B5": [("RAD", None), ("PB", None), ("STAB", 0.05), ("EXP", None), ("GAP", 3.1)],
    "B6": [("RAD", None), ("PB", None), ("STAB", 0.05), ("EXP", None), ("ABUND", 1.0)],
}


def bundle_mask(name: str, u: pd.DataFrame | None = None) -> np.ndarray:
    u = universe() if u is None else u
    m = np.ones(len(u), dtype=bool)
    for code, param in BUNDLES[name]:
        m &= constraint_mask(code, param, u)
    return m


# ---------------------------------------------------------------------------------------------
# frontier and price
# ---------------------------------------------------------------------------------------------
def _k_needed(kind: str) -> int:
    return {"best": 1, "10th": 10, "median50": 50}[kind]


def frontier(q: np.ndarray, mask: np.ndarray, kind: str = "best") -> np.ndarray:
    """Frontier value(s) of the candidate set ``mask`` for requirement matrix q (n × V) or (n,)."""
    q2 = np.asarray(q)
    one = q2.ndim == 1
    if one:
        q2 = q2[:, None]
    sub = np.asarray(q2[np.asarray(mask, dtype=bool)], dtype=np.float64)
    sub = np.where(np.isfinite(sub), sub, np.inf)
    k = _k_needed(kind)
    out = np.full(sub.shape[1], np.nan)
    if sub.shape[0] >= k:
        if kind == "best":
            out = sub.min(axis=0)
        elif kind == "10th":
            out = np.partition(sub, 9, axis=0)[9]
        elif kind == "median50":
            top = np.sort(np.partition(sub, 49, axis=0)[:50], axis=0)
            out = 0.5 * (top[24] + top[25])
        else:
            raise ValueError(kind)
        out = np.where(np.isfinite(out), out, np.nan)
    return out[0] if one else out


def price(q: np.ndarray, base: np.ndarray, constraint: np.ndarray, kind: str = "best") -> np.ndarray:
    """P(K) = q*(F0 ∩ K) / q*(F0) − 1 (fraction)."""
    return frontier(q, base & constraint, kind) / frontier(q, base, kind) - 1.0


def top_k(q: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Row indices of the k smallest requirements within mask (q one column), best first; ties by row."""
    q = np.asarray(q, dtype=np.float64)
    idx = np.flatnonzero(mask)
    order = np.lexsort((idx, q[idx]))
    return idx[order[:k]]


def ranks(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dense array of 1-based ranks within mask (NaN outside); ties broken by row index."""
    q = np.asarray(q, dtype=np.float64)
    out = np.full(q.shape[0], np.nan)
    idx = np.flatnonzero(mask)
    order = idx[np.lexsort((idx, q[idx]))]
    out[order] = np.arange(1, order.size + 1)
    return out


# ---------------------------------------------------------------------------------------------
# attribution, Pareto, TOPSIS
# ---------------------------------------------------------------------------------------------
def shapley(players: list[str], value: Callable[[frozenset], float]) -> dict[str, float]:
    """Exact Shapley values for a small game; value(∅) is subtracted implicitly."""
    n = len(players)
    cache = {}

    def v(s):
        if s not in cache:
            cache[s] = value(s)
        return cache[s]

    phi = {}
    for p in players:
        others = [x for x in players if x != p]
        tot = 0.0
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for s in itertools.combinations(others, r):
                s = frozenset(s)
                tot += w * (v(s | {p}) - v(s))
        phi[p] = tot
    return phi


def pareto_layers(a: np.ndarray, b: np.ndarray, max_layers: int = 3,
                  minimize: tuple[bool, bool] = (True, False)) -> np.ndarray:
    """Non-dominated sorting in 2-D. Returns layer 1..max_layers, or 0 for deeper points.

    Default: minimise a (requirement), maximise b (e.g. abundance or band gap). NaN → never ranked.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    x = a if minimize[0] else -a
    y = b if minimize[1] else -b
    layer = np.zeros(a.size, dtype=int)
    alive = np.isfinite(x) & np.isfinite(y)
    for L in range(1, max_layers + 1):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        order = idx[np.lexsort((y[idx], x[idx]))]          # by x asc, then y asc
        best_y = np.inf
        front = []
        for i in order:
            if y[i] < best_y:                                # strictly better in y than all with x ≤
                front.append(i); best_y = y[i]
            elif y[i] == best_y and front and x[front[-1]] == x[i]:
                front.append(i)                              # exact duplicate point
        front = np.array(front, dtype=int)
        layer[front] = L
        alive[front] = False
    return layer


def topsis(matrix: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """TOPSIS closeness (higher is better). matrix (n × c) with finite values; vector normalisation."""
    m = np.asarray(matrix, dtype=float)
    norm = np.sqrt((m ** 2).sum(axis=0)); norm[norm == 0] = 1.0
    v = m / norm * np.asarray(weights, dtype=float)
    ideal = np.where(benefit, v.max(axis=0), v.min(axis=0))
    anti = np.where(benefit, v.min(axis=0), v.max(axis=0))
    dp = np.sqrt(((v - ideal) ** 2).sum(axis=1)); dm = np.sqrt(((v - anti) ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = dm / (dp + dm)
    return np.where(np.isfinite(c), c, 0.0)


def simplex_grid(n: int, step: float = 0.1) -> np.ndarray:
    k = int(round(1 / step))
    pts = [c for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]
    return np.array(pts, dtype=float) / k


def percent(x: float | np.ndarray) -> float | np.ndarray:
    return 100.0 * np.asarray(x)


def summarize_range(values: Iterable[float]) -> dict:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "min": np.nan, "median": np.nan, "max": np.nan}
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}
