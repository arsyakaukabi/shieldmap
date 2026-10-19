"""Data-conditioned density-error model, tiers v2 and bootstrap price bands v2.

Why (review_statistics.md findings 1, 3, 4, 9, 10, 15; review_referee.md major 8, 12)
    The frontiers-module tiers and the continuum-module bootstrap gave every compound its class's COD δ distribution. That
    distribution is dominated by light compounds, δ depends on density within a class, and measured
    densities that exist for most frontier compounds were ignored. This module replaces that model.

Convention (unchanged): δ = ρ_MP/ρ_true − 1, so the true thickness requirement at fixed μ/ρ is
q_true = q_MP·(1 + δ). Areal mass is density-invariant and is not modelled. MP densities stay the map's
densities: this model only describes their uncertainty (COD calibrates it, never replaces ρ_MP).

Error model per universe row m (``RowModel``)
    measured, element   unary MP entry with the space group of the element's representative entry and a
                        Stage 1 experimental density: δ_m = ρ_MP/ρ_exp − 1 (stage 1 data).
    measured, COD       MP entry structurally matched to COD (the COD module) and not excluded by the COD-module
                        screen (robust 5-SD log screen, elemental cross-check): δ_m = the COD module δ.
                        Both measured kinds carry a residual r drawn from the COD replicate spread (log
                        deviations of structurally equivalent COD entries from their set median):
                        1 + δ = (1 + δ_m)·exp(r).
    conditional         every other row: δ drawn from the screened COD δ of its cell. Cells: ICSD-matched
                        entries by ρ_MP bin × primary class × stability (e_hull ≤ 0.05 eV/atom or not),
                        theoretical entries by a coarser ρ_MP bin × stability with their own (wider)
                        COD-matched theoretical distribution. A cell with fewer than MIN_CELL pairs
                        falls back, in order, to bin × stability, bin × class, bin.
    Physical floor: no draw puts a true density above the largest stage 1 experimental element density
    (δ ≥ ρ_MP/ρ_max − 1); MP entries denser than that (DFT artefacts) are moved to at most ρ_max.
    Correlation: Gaussian copula with one shared shock per chemical system (MP chemsys),
    z_m = √ρ_g·s_g + √(1−ρ_g)·e_m, δ_m = F_cell⁻¹(Φ(z_m)); ρ_g is the intraclass correlation of the COD
    normal scores by chemical system (estimated here). The copula keeps every cell's empirical shape.
    Modes: "copula" (primary), "iid" (no correlation), "copula_k8" (8-SD COD-module screen),
    "copula_rawresid" (replicate residual without the outlier screen), and two references reproducing the
    Class model (iid draws from the class distribution for every row, measured or not):
    "class_iid" (current screen) and "class_iid_abs50" (the original |δ| ≤ 0.5 screen, as delivered).

Tiers v2 (rule with per-compound intervals)
    interval [q(1+lo_m), q(1+hi_m)], (lo_m, hi_m) = 5th/95th percentile of the row's δ model; tier 1 =
    members whose lower bound ≤ the smallest upper bound; tier 2 repeats on the rest. Thickness basis,
    Cs-137 and Co-60 primary variants, rules R1/R2/R3, bundles B0/B4/B5/B6 plus B6 with ABUND ≥ 0.5 and
    ≥ 2 ppm. Calibration (N_DRAWS draws per mode): coverage of tier 1 (draws whose true best is in
    tier 1), P(best) per member, and pool90 = the smallest set of most frequent winners covering ≥ 90 %
    of draws.

Bootstrap price bands v2 (thickness, R2 primary Cs-137 and Co-60 variants)
    RAD, B4, GAP(3.1), B5, B6 against F0 for the kinds best_clean (headline), median50,
    median50_clean, best_icsd, best, 10th and the original per-cell rule (median50 where the best
    compound on either side is an artefact entry, else best). Bands are forward-simulated sensitivity
    bands (5th–95th percentile of the price over draws of the true densities), not confidence intervals.

Outputs: data/phase3/T/; figures fig6_tiers, T_bootstrap. Run: python -m shieldmap.analysis3.density_model
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from shieldmap import paths
from shieldmap.analysis3 import common

OUT = common.P3 / "T"
D_DIR = common.P3 / "D"
N_DRAWS = 5000
SEED = 20261012
MODES = ("copula", "iid", "class_iid", "class_iid_abs50")
SENS_MODES = ("copula",)                          # sensitivity reruns: "copula_k8" (8-SD screen),
                                                  # "copula_rawresid" (unscreened replicate residual),
                                                  # "copula_noceiling" (no density ceiling) and
                                                  # "copula_ceilall" (ceiling on measured rows too)
CURVE_CONSTRAINTS = ("RAD", "B4", "GAP(3.1)", "B5", "B6", "ABUND(1)")
CURVE_MARGINAL = (("GAP(3.1)|B4", "B4", "B5"), ("ABUND(1)|B4", "B4", "B6"))
LN100 = float(np.log(100.0))
MIN_CELL = 100                                    # class cells within a density bin need ≥ 100 COD pairs
BINS_ICSD = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, np.inf)
BINS_THEO = (0.0, 4.0, 8.0, np.inf)
SCENARIOS = ("CS137", "CO60")
RULES = ("R1", "R2", "R3")
TIER_SETS = ("B0", "B4", "B5", "B6", "B6@ABUND0.5", "B6@ABUND2")
PRICE_CONSTRAINTS = ("RAD", "B4", "GAP(3.1)", "B5", "B6")
PRICE_KINDS = ("best_clean", "median50", "median50_clean", "best_icsd", "best", "10th", "rule_original")
TOPN = 50
DISPLAY_T = 0.01                                  # variant shown in fig6 (tiers are T-independent for Cs-137)


# =============================================================================================
# the error model
# =============================================================================================
def _bin(x: np.ndarray, edges) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(edges[1:-1]), x, side="right"), 0, len(edges) - 2)


def _bin_label(edges, i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    return f"({lo:g},{hi:g}]" if np.isfinite(hi) else f">{lo:g}"


def _matches() -> pd.DataFrame:
    m = pd.read_parquet(D_DIR / "matches.parquet")
    if "excluded" not in m.columns:
        from shieldmap.analysis3 import cod
        m = cod.screen_matches(m)
    return m


def replicate_residuals(m: pd.DataFrame | None = None, k: float | None = None) -> np.ndarray:
    """log deviations ln(ρ_i/median) of structurally equivalent COD entries of one MP entry (sets ≥ 2).

    Uses the screened matches only; each set contributes all of its members. With k, values farther than
    k robust SD (1.4826·MAD) from the pooled median are dropped: they are inconsistent members of a set
    (mis-scaled cells), not the uncertainty of the selected density.
    """
    m = _matches() if m is None else m
    cf = pd.read_parquet(D_DIR / "cod_filtered.parquet", columns=["cod_id", "rho_cod"])
    rho = dict(zip(cf["cod_id"].astype(int), cf["rho_cod"].astype(float)))
    out = []
    sel = m[(~m["excluded"]) & (m["n_cod_equivalent"] >= 2)]
    for ids in sel["cod_ids_equivalent"]:
        r = np.array([rho[int(i)] for i in str(ids).split(",") if int(i) in rho], float)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size >= 2:
            out.append(np.log(r / np.median(r)))
    x = np.concatenate(out) if out else np.zeros(1)
    if k is not None:                        # same log-symmetric robust screen as the COD module
        sd = 1.4826 * float(np.median(np.abs(x - np.median(x))))
        x = x[np.abs(x - np.median(x)) <= k * sd]
    return x


STAB_SPLIT = 0.05                                 # eV/atom; e_hull ≤ 0.05 ("stable") vs above


def _stab(e) -> str:
    return "stable" if float(e) <= STAB_SPLIT else "unstable"


def _cell_candidates(theoretical: bool, b: int, cls: str, stab: str) -> list[tuple]:
    """Cell keys from the most to the least specific (the first with data is used)."""
    if theoretical:
        return [("theo", b, "*", stab), ("theo", b, "*", "*")]
    return [("icsd", b, cls, stab), ("icsd", b, "*", stab), ("icsd", b, cls, "*"), ("icsd", b, "*", "*")]


def calibration_cells(m: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Screened COD δ per cell (status, ρ_MP bin, class | "*", stability | "*"); cells need ≥ MIN_CELL pairs,
    except the (status, bin, "*", "*") cells, which always exist."""
    h = m[(m["nelements"] >= 2) & (~m["excluded"])].copy()
    h["stab"] = [_stab(e) for e in h["e_hull"]]
    cells, recs = {}, []
    for status, edges, sub in (("icsd", BINS_ICSD, h[~h["theoretical"]]), ("theo", BINS_THEO, h[h["theoretical"]])):
        bi = _bin(sub["rho_mp"].to_numpy(float), edges)
        for b in range(len(edges) - 1):
            sb = sub[bi == b]
            cells[(status, b, "*", "*")] = np.sort(sb["delta"].to_numpy(float))
            classes = ("*",) + (tuple(common.PRIMARY_CLASS_ORDER) if status == "icsd" else ())
            for c in classes:
                for st in ("stable", "unstable", "*"):
                    if c == "*" and st == "*":
                        continue
                    sel = sb
                    if c != "*":
                        sel = sel[sel["primary_class"] == c]
                    if st != "*":
                        sel = sel[sel["stab"] == st]
                    x = sel["delta"].to_numpy(float)
                    if x.size >= MIN_CELL:
                        cells[(status, b, c, st)] = np.sort(x)
    for k, x in cells.items():
        edges = BINS_ICSD if k[0] == "icsd" else BINS_THEO
        recs.append({"status": k[0], "rho_bin": _bin_label(edges, k[1]), "class": k[2], "stability": k[3],
                     "n": int(x.size),
                     **({f"p{q:02d}": float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95)} if x.size else {})})
    return cells, pd.DataFrame(recs)


def _cell_key(cells: dict, theoretical: bool, rho: float, cls: str, e_hull: float) -> tuple:
    edges = BINS_THEO if theoretical else BINS_ICSD
    b = int(_bin(np.array([rho]), edges)[0])
    for k in _cell_candidates(theoretical, b, cls, _stab(e_hull)):
        if k in cells:
            return k
    raise KeyError((theoretical, b, cls))


def measured_deltas(m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows with a measured density.

    1. Unary MP entries that are plausibly the measured (ambient) phase of their element get the stage 1
       experimental density: same space group as the element's representative entry and
       |ρ_MP/ρ_exp − 1| ≤ cod.ELEMENT_PHASE_TOL (α-Sn, for example, is not given white tin's density).
    2. Every other MP entry with a screened COD structural match gets the COD-module δ (polymorph-specific).
    """
    from shieldmap.analysis3 import cod
    u = common.universe()
    m = _matches() if m is None else m
    recs = {}
    er = cod.element_reference(u).set_index("symbol")
    un = u[u["nelements"] == 1]
    for r, els, sg, rho in zip(un["row"], un["elements"], un["spacegroup_number"], un["density"]):
        s_ = els[0]
        if s_ not in er.index or int(er.at[s_, "rep_sg"]) != int(sg):
            continue
        d = float(rho) / float(er.at[s_, "density_solid_exp"]) - 1.0
        if abs(d) <= cod.ELEMENT_PHASE_TOL:
            recs[int(r)] = {"row": int(r), "delta_meas": d, "source": "phase1_element"}
    ok = m[~m["excluded"]]
    for r, d in zip(ok["row"], ok["delta"]):
        if int(r) not in recs:
            recs[int(r)] = {"row": int(r), "delta_meas": float(d), "source": "cod"}
    return pd.DataFrame(list(recs.values())).sort_values("row").reset_index(drop=True)


def chemsys_icc(m: pd.DataFrame, cells: dict) -> dict:
    """Intraclass correlation of COD normal scores by chemical system (and, for comparison, by class)."""
    u = common.universe()
    h = m[(m["nelements"] >= 2) & (~m["excluded"])].copy()
    h["chemsys"] = u["chemsys"].to_numpy()[h["row"].to_numpy()]
    z = np.empty(len(h))
    for i, (th, rho, c, eh, d) in enumerate(zip(h["theoretical"], h["rho_mp"], h["primary_class"], h["e_hull"],
                                                 h["delta"])):
        x = cells[_cell_key(cells, bool(th), float(rho), c, float(eh))]
        lo = np.searchsorted(x, d, side="left"); hi = np.searchsorted(x, d, side="right")
        z[i] = ndtri((0.5 * (lo + hi) + 0.5) / (x.size + 1))
    h["z"] = z

    def icc(groups) -> dict:
        return _icc(h["z"].to_numpy(), np.asarray(groups))

    out = {"chemsys": icc(h["chemsys"]), "primary_class": icc(h["primary_class"])}
    # bootstrap over chemical systems for a CI of the chemsys ICC
    rng = np.random.default_rng(SEED)
    sizes = h.groupby("chemsys").size()
    multi = sizes[sizes >= 2].index.to_numpy()
    hh = h[h["chemsys"].isin(multi)]
    grp = {c: g["z"].to_numpy() for c, g in hh.groupby("chemsys")}
    vals = []
    keys = list(grp)
    for _ in range(200):
        pick = rng.integers(0, len(keys), len(keys))
        zz = [grp[keys[i]] for i in pick]
        n_i = np.array([z_.size for z_ in zz], float); means = np.array([z_.mean() for z_ in zz])
        allz = np.concatenate(zz); grand = allz.mean(); N = n_i.sum(); k = n_i.size
        msb = float((n_i * (means - grand) ** 2).sum()) / (k - 1)
        msw = float(sum(((z_ - z_.mean()) ** 2).sum() for z_ in zz)) / (N - k)
        n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
        vals.append((msb - msw) / (msb + (n0 - 1) * msw))
    out["chemsys"]["ci95_bootstrap_chemsys"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def _icc(z: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects intraclass correlation (unbalanced ANOVA estimator), groups with ≥ 2 members."""
    df = pd.DataFrame({"z": z, "g": groups})
    df = df[df.groupby("g")["z"].transform("size") >= 2]
    g = df.groupby("g")["z"]
    n_i = g.size().to_numpy(float); means = g.mean().to_numpy()
    grand = float(df["z"].mean()); N = n_i.sum(); k = n_i.size
    msb = float((n_i * (means - grand) ** 2).sum()) / (k - 1)
    msw = float(((df["z"] - g.transform("mean")) ** 2).sum()) / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    return {"icc": float((msb - msw) / (msb + (n0 - 1) * msw)), "n_groups": int(k), "n_rows": int(N), "n0": float(n0)}


class RowModel:
    """δ model of every universe row (see module docstring)."""

    def __init__(self, screen: str = "k5", resid_screen: bool = True, ceiling: str = "unmeasured"):
        """screen: "k5" (primary; the COD module robust 5-SD screen + elemental cross-check) or "k8" (8 SD).
        resid_screen: apply the same screen to the replicate residual pool (False: keep every member).
        ceiling: "unmeasured" (primary: ρ_max for unmeasured rows, ρ_max·exp(−r_min) for measured rows),
        "all" (ρ_max for every row, as first delivered; it made Ir's own interval one-sided, because ρ_max is Ir's
        measured density) or "none"."""
        from shieldmap.analysis3 import cod
        u = common.universe()
        self.n = len(u)
        self.screen = screen
        m = _matches()
        if screen == "k8":
            m = m.assign(excluded=m["excluded_k8"] | m["xcheck_fail"])
        elif screen != "k5":
            raise ValueError(screen)
        self.cells, self.cell_table = calibration_cells(m)
        from shieldmap.analysis3 import cod as _cod
        self.resid_raw_n = int(replicate_residuals(m).size)
        self.resid = np.sort(replicate_residuals(m, (_cod.SCREEN_K if screen == "k5" else 8.0) if resid_screen else None))
        meas = measured_deltas(m)
        self.source = np.full(self.n, "conditional", dtype=object)
        self.delta_meas = np.full(self.n, np.nan)
        self.source[meas["row"].to_numpy()] = meas["source"].to_numpy()
        self.delta_meas[meas["row"].to_numpy()] = meas["delta_meas"].to_numpy()
        self.measured = self.source != "conditional"
        cls = common.universe_primary_class()
        self.cls = cls
        theo = u["theoretical"].to_numpy(bool); rho = u["density"].to_numpy(float)
        eh = u["energy_above_hull"].to_numpy(float)
        keys = [_cell_key(self.cells, bool(t), float(r), c, float(e)) for t, r, c, e in zip(theo, rho, cls, eh)]
        # physical floor: no true density exceeds the densest stage 1 experimental element density
        self.rho_max = float(common.elements_table()["density_solid_exp"].max())
        self.floor = rho / self.rho_max - 1.0
        self.ceiling = ceiling
        if ceiling == "unmeasured":
            # Measured rows keep their screened replicate residual on both sides: their ceiling is raised to
            # ρ_max·exp(−r_min), r_min the smallest screened residual, so it never binds in the primary mode
            # (ρ_meas ≤ ρ_max). It still caps unscreened-residual draws (copula_rawresid) at that density.
            r_min = float(np.min(replicate_residuals(m, _cod.SCREEN_K)))
            floor_meas = rho * math.exp(r_min) / self.rho_max - 1.0
            self.floor = np.where(self.measured, floor_meas, self.floor)
        elif ceiling == "none":
            self.floor = np.full(self.n, -np.inf)
        elif ceiling != "all":
            raise ValueError(ceiling)
        self.cell_keys = sorted(set(keys), key=str)
        kid = {k: i for i, k in enumerate(self.cell_keys)}
        self.cell_id = np.array([kid[k] for k in keys])
        self.cell_arrays = [self.cells[k] for k in self.cell_keys]
        self.chemsys_code, self.chemsys_names = pd.factorize(u["chemsys"])
        icc = chemsys_icc(m, self.cells)
        self.icc = icc
        self.rho = max(0.0, float(icc["chemsys"]["icc"]))
        # the class model (reference modes): current screen, and the original |δ| ≤ 0.5 screen
        self.class_dist = {}
        for mode, kind in (("class_iid", "compound"), ("class_iid_abs50", "compound_abs50")):
            cd = cod.class_delta_distribution(kind)
            self.class_dist[mode] = {c: np.sort(np.asarray(cd[c], float)) for c in common.PRIMARY_CLASS_ORDER}
        # per-row 90 % interval and support
        self.lo = np.empty(self.n); self.hi = np.empty(self.n); self.med = np.empty(self.n)
        self.sup_lo = np.empty(self.n); self.sup_hi = np.empty(self.n)
        rq = np.percentile(self.resid, [5, 50, 95])
        for i, x in enumerate(self.cell_arrays):
            ix = (self.cell_id == i) & ~self.measured
            p5, p50, p95 = np.percentile(x, [5, 50, 95])
            self.lo[ix], self.med[ix], self.hi[ix] = p5, p50, p95
            self.sup_lo[ix], self.sup_hi[ix] = x[0], x[-1]
        mm = self.measured
        base = 1.0 + self.delta_meas[mm]
        self.lo[mm] = base * math.exp(rq[0]) - 1; self.med[mm] = base * math.exp(rq[1]) - 1
        self.hi[mm] = base * math.exp(rq[2]) - 1
        self.sup_lo[mm] = base * math.exp(self.resid[0]) - 1; self.sup_hi[mm] = base * math.exp(self.resid[-1]) - 1
        self._clip_intervals()
    def _clip_intervals(self) -> None:
        for a in (self.lo, self.med, self.hi, self.sup_lo, self.sup_hi):
            np.maximum(a, self.floor, out=a)

    def bounds(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        if mode in self.class_dist:
            d = self.class_dist[mode]
            return (np.array([d[c][0] for c in self.cls]), np.array([d[c][-1] for c in self.cls]))
        return self.sup_lo, self.sup_hi

    def draw(self, rows: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
        """One draw of δ for the given rows."""
        rows = np.asarray(rows)
        out = np.empty(rows.size)
        if mode in self.class_dist:
            cl = self.cls[rows]
            for c in np.unique(cl):
                ix = np.flatnonzero(cl == c)
                x = self.class_dist[mode][c]
                out[ix] = x[rng.integers(0, x.size, ix.size)]
            return out                                  # reference model: reproduced as it was (no floor)
        meas = self.measured[rows]
        if meas.any():
            r = self.resid[rng.integers(0, self.resid.size, int(meas.sum()))]
            out[meas] = (1.0 + self.delta_meas[rows[meas]]) * np.exp(r) - 1.0
        cr = rows[~meas]
        if cr.size:
            e = rng.standard_normal(cr.size)
            if mode == "copula" and self.rho > 0:
                g, inv = np.unique(self.chemsys_code[cr], return_inverse=True)
                s = rng.standard_normal(g.size)
                z = math.sqrt(self.rho) * s[inv] + math.sqrt(1.0 - self.rho) * e
            elif mode in ("copula", "iid"):
                z = e
            else:
                raise ValueError(mode)
            uu = ndtr(z)
            vals = np.empty(cr.size)
            cid = self.cell_id[cr]
            for i in np.unique(cid):
                ix = np.flatnonzero(cid == i)
                x = self.cell_arrays[i]
                vals[ix] = x[np.minimum((uu[ix] * x.size).astype(np.int64), x.size - 1)]
            out[~meas] = vals
        return np.maximum(out, self.floor[rows])

    def table(self) -> pd.DataFrame:
        u = common.universe()
        return pd.DataFrame({"row": np.arange(self.n), "material_id": u["material_id"].to_numpy(),
                             "source": self.source, "delta_meas": self.delta_meas,
                             "cell": [str(self.cell_keys[i]) for i in self.cell_id],
                             "delta_p05": self.lo, "delta_p50": self.med, "delta_p95": self.hi,
                             "primary_class": self.cls})


# =============================================================================================
# sets, columns, frontiers
# =============================================================================================
def tier_masks() -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for rule in RULES:
        base = common.base_mask(rule)
        for b in TIER_SETS:
            if b.startswith("B6@ABUND"):
                a = float(b.split("ABUND")[1])
                m = common.bundle_mask("B4") & common.constraint_mask("ABUND", a)
            else:
                m = common.bundle_mask(b)
            out[(rule, b)] = base & m
    return out


def price_masks() -> dict[str, np.ndarray]:
    """R2: F0 and F0 ∩ K for the price constraints."""
    base = common.base_mask("R2")
    out = {"F0": base}
    for k in PRICE_CONSTRAINTS:
        if k.startswith("B"):
            m = common.bundle_mask(k)
        elif k.startswith("GAP"):
            m = common.constraint_mask("GAP", 3.1)
        else:
            m = common.constraint_mask(k)
        out[k] = base & m
    return out


def columns() -> pd.DataFrame:
    recs = []
    for sc in SCENARIOS:
        for _, r in common.scenario_variants(sc).iterrows():
            recs.append({"scenario": sc, "variant_id": int(r.variant_id), "T": float(r.target_value)})
    return pd.DataFrame(recs)


def _top(qd: np.ndarray, n: int = TOPN) -> tuple[np.ndarray, np.ndarray]:
    """(sorted values, positions) of the n smallest entries per column."""
    k = min(n, qd.shape[0])
    if qd.shape[0] > k:
        part = np.argpartition(qd, k - 1, axis=0)[:k]
    else:
        part = np.repeat(np.arange(qd.shape[0])[:, None], qd.shape[1], axis=1)
    vals = np.take_along_axis(qd, part, axis=0)
    o = np.argsort(vals, axis=0, kind="stable")
    return np.take_along_axis(vals, o, axis=0), np.take_along_axis(part, o, axis=0)


def _kinds(vals: np.ndarray) -> dict:
    k = vals.shape[0]
    nan = np.full(vals.shape[1], np.nan)
    return {"best": vals[0] if k else nan, "10th": vals[9] if k >= 10 else nan,
            "median50": 0.5 * (vals[24] + vals[25]) if k >= 50 else nan}


def candidates(q: np.ndarray, masks: dict, lo: np.ndarray, hi: np.ndarray, n: int = TOPN) -> np.ndarray:
    """Rows that can enter the n best of any mask under any δ in [lo, hi] (exact pruning)."""
    keep = np.zeros(q.shape[0], bool)
    for m in masks.values():
        idx = np.flatnonzero(m & np.isfinite(q).all(axis=1))
        if idx.size == 0:
            continue
        upper = q[idx] * (1.0 + hi[idx, None])
        kk = min(n, idx.size)
        tau = np.partition(upper, kk - 1, axis=0)[kk - 1]                       # per column
        lower = q[idx] * (1.0 + lo[idx, None])
        keep[idx[(lower <= tau[None, :]).any(axis=1)]] = True
    return np.flatnonzero(keep)


# =============================================================================================
# tiers v2
# =============================================================================================
def tiers(q: np.ndarray, mask: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_tier: int = 3) -> np.ndarray:
    """Rule with per-row bounds (frontiers.tiers, restated here for per-row arrays)."""
    from shieldmap.analysis3 import frontiers
    return frontiers.tiers(q, mask, lo, hi, max_tier)


def pool(win_counts: dict, n_draws: int, level: float = 0.9) -> list[int]:
    """Smallest set of most frequent winners whose frequencies sum to ≥ level."""
    items = sorted(win_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out, acc = [], 0
    for r, c in items:
        if acc >= level * n_draws:
            break
        out.append(int(r)); acc += c
    return out


# =============================================================================================
# the simulation
# =============================================================================================
def simulate(model: RowModel, n_draws: int = N_DRAWS, modes=MODES, seed: int = SEED, verbose: bool = True,
             label_suffix: str = "", nominal_from: dict | None = None) -> dict:
    cols = columns()
    qall = np.asarray(common.profiles("thickness"))[:, cols["variant_id"].to_numpy()].astype(float)
    tm = tier_masks()
    pm = price_masks()
    art = common.artefact_flag(np.arange(model.n))
    icsd = common.icsd_mask()
    pset = {}
    for k, m in pm.items():
        pset[(k, "all")] = m; pset[(k, "clean")] = m & ~art; pset[(k, "icsd")] = m & icsd
    # class-band (the frontiers module) tiers for comparison, recomputed with the current COD-module band
    from shieldmap.analysis3 import frontiers
    bands, band_src = frontiers.class_band()
    cb_lo = np.array([bands[c][0] for c in model.cls]); cb_hi = np.array([bands[c][1] for c in model.cls])
    # nominal tiers (new model and class band); a sensitivity run reuses the primary run's tier-1 sets
    nom_t1, nom_cb_t1, tier_recs = {}, {}, []
    for (rule, b), m in tm.items():
        for j, cr in cols.iterrows():
            t = tiers(qall[:, j], m, model.lo, model.hi, 3)
            nom_t1[(rule, b, j)] = np.flatnonzero(t == 1)
            nom_cb_t1[(rule, b, j)] = np.flatnonzero(tiers(qall[:, j], m, cb_lo, cb_hi, 1) == 1)
            for r in np.flatnonzero(t):
                tier_recs.append({"scenario": cr.scenario, "variant_id": cr.variant_id, "T": cr["T"], "rule": rule,
                                  "set": b, "tier": int(t[r]), "row": int(r), "q_cm": qall[r, j],
                                  "q_lo": qall[r, j] * (1 + model.lo[r]), "q_mid": qall[r, j] * (1 + model.med[r]),
                                  "q_hi": qall[r, j] * (1 + model.hi[r])})
    # nominal prices (MP densities) and the measured-centred point
    def prices(qd: np.ndarray, rows: np.ndarray | None = None) -> dict:
        res = {}
        fr = {}
        for key, m in pset.items():
            mm = m if rows is None else m[rows]
            idx = np.flatnonzero(mm)
            v, p = _top(qd[idx])
            kd = _kinds(v)
            kd["best_row"] = (np.arange(model.n) if rows is None else rows)[idx[p[0]]]
            fr[key] = kd
        for k in PRICE_CONSTRAINTS:
            a, c, i = fr[(k, "all")], fr[(k, "clean")], fr[(k, "icsd")]
            a0, c0, i0 = fr[("F0", "all")], fr[("F0", "clean")], fr[("F0", "icsd")]
            r = {"best": a["best"] / a0["best"] - 1, "10th": a["10th"] / a0["10th"] - 1,
                 "median50": a["median50"] / a0["median50"] - 1,
                 "best_clean": c["best"] / c0["best"] - 1, "median50_clean": c["median50"] / c0["median50"] - 1,
                 "best_icsd": i["best"] / i0["best"] - 1}
            flag = art[a0["best_row"]] | art[a["best_row"]]
            r["rule_original"] = np.where(flag, r["median50"], r["best"])
            r["flag"] = flag
            r["best_row_clean_F0"] = c0["best_row"]; r["best_row_clean_K"] = c["best_row"]
            res[k] = r
        return res

    if nominal_from is not None:
        nom_t1, nom_cb_t1 = nominal_from["nom_t1"], nominal_from["nom_cb_t1"]
    nominal = prices(qall)
    point_delta = np.where(model.measured, model.delta_meas, model.med)
    point = prices(qall * (1.0 + point_delta[:, None]))
    # simulation
    acc = {}
    t0 = time.time()
    for mi, mode in enumerate(modes):
        rng = np.random.default_rng(seed + 101 * mi)
        lo_s, hi_s = model.bounds(mode)
        allm = {**{("t",) + k: v for k, v in tm.items()}, **{("p",) + k: v for k, v in pset.items()}}
        cand = candidates(qall, allm, lo_s, hi_s)
        qc = qall[cand]
        tmc = {k: v[cand] for k, v in tm.items()}
        pdraw = {k: {kd: np.empty((n_draws, len(cols))) for kd in PRICE_KINDS + ("flag",)} for k in PRICE_CONSTRAINTS}
        fwin = {}                                   # which compounds set the best_clean frontiers in the draws
        wins = {(k, j): {} for k in tm for j in range(len(cols))}
        cover = {(k, j): 0 for k in tm for j in range(len(cols))}
        cover_cb = {(k, j): 0 for k in tm for j in range(len(cols))}
        nt1 = {k: set(v.tolist()) for k, v in nom_t1.items()}
        ncb = {k: set(v.tolist()) for k, v in nom_cb_t1.items()}
        for d in range(n_draws):
            dl = model.draw(cand, rng, mode)
            qd = qc * (1.0 + dl[:, None])
            pr = prices(qd, cand)
            for k in PRICE_CONSTRAINTS:
                for kd in PRICE_KINDS + ("flag",):
                    pdraw[k][kd][d] = pr[k][kd]
                for side in ("best_row_clean_F0", "best_row_clean_K"):
                    for j, r_ in enumerate(np.asarray(pr[k][side]).tolist()):
                        key_ = (k, side, j, int(r_))
                        fwin[key_] = fwin.get(key_, 0) + 1
            for k, m in tmc.items():
                idx = np.flatnonzero(m)
                am = cand[idx[np.argmin(qd[idx], axis=0)]]
                for j in range(len(cols)):
                    w = int(am[j])
                    wd = wins[(k, j)]
                    wd[w] = wd.get(w, 0) + 1
                    cover[(k, j)] += w in nt1[(k[0], k[1], j)]
                    cover_cb[(k, j)] += w in ncb[(k[0], k[1], j)]
            if verbose and (d + 1) % 500 == 0:
                print(f"  [T] {mode} {d + 1}/{n_draws}  {time.time() - t0:.0f} s  (candidates {cand.size})",
                      flush=True)
        acc[mode + label_suffix] = {"prices": pdraw, "wins": wins, "cover": cover, "cover_cb": cover_cb, "n_candidates": int(cand.size),
                                    "frontier_wins": fwin}
    return {"cols": cols, "nominal": nominal, "point": point, "acc": acc, "nom_t1": nom_t1, "nom_cb_t1": nom_cb_t1,
            "tier_recs": pd.DataFrame(tier_recs), "band_source": band_src, "n_draws": n_draws,
            "modes": [m_ + label_suffix for m_ in modes],
            "seconds": time.time() - t0}


# =============================================================================================
# tables
# =============================================================================================
def price_table(sim: dict) -> pd.DataFrame:
    cols, recs = sim["cols"], []
    for mode in sim["modes"]:
        pdr = sim["acc"][mode]["prices"]
        for k in PRICE_CONSTRAINTS:
            for kd in PRICE_KINDS:
                x = pdr[k][kd]
                for j, cr in cols.iterrows():
                    c = x[:, j]
                    recs.append({"mode": mode, "scenario": cr.scenario, "variant_id": cr.variant_id, "T": cr["T"],
                                 "constraint": k, "kind": kd, "nominal": float(sim["nominal"][k][kd][j]),
                                 "measured_point": float(sim["point"][k][kd][j]),
                                 "p05": float(np.nanquantile(c, 0.05)), "median": float(np.nanmedian(c)),
                                 "p95": float(np.nanquantile(c, 0.95)), "n_draws": int(np.isfinite(c).sum()),
                                 "flag_flip_rate": float(np.mean(pdr[k]["flag"][:, j] != sim["nominal"][k]["flag"][j]))
                                 if kd == "rule_original" else np.nan,
                                 "nominal_flag": bool(sim["nominal"][k]["flag"][j]) if kd == "rule_original" else None})
    df = pd.DataFrame(recs)
    df["width"] = df["p95"] - df["p05"]
    w = df.pivot_table(index=["scenario", "variant_id", "constraint", "kind"], columns="mode", values="width")
    ratio = (w["copula"] / w["iid"]).rename("width_ratio_copula_vs_iid").reset_index()
    return df.merge(ratio, on=["scenario", "variant_id", "constraint", "kind"], how="left")


def frontier_win_table(sim: dict) -> pd.DataFrame:
    """How often each compound sets the best_clean F0 / F0∩K frontier in the draws (explains the bands)."""
    u = common.universe()
    cols, recs = sim["cols"], []
    for mode in sim["modes"]:
        for (k, side, j, r), c in sim["acc"][mode]["frontier_wins"].items():
            recs.append({"mode": mode, "scenario": cols.loc[j, "scenario"], "variant_id": cols.loc[j, "variant_id"],
                         "constraint": k, "side": "F0" if side.endswith("F0") else "F0∩K", "row": r,
                         "formula": u.formula_pretty.iat[r], "material_id": u.material_id.iat[r],
                         "share_of_draws": c / sim["n_draws"]})
    return pd.DataFrame(recs).sort_values(["mode", "scenario", "variant_id", "constraint", "side", "share_of_draws"],
                                          ascending=[True, True, True, True, True, False])


def tier_tables(sim: dict, model: RowModel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u = common.universe()
    cols, nd = sim["cols"], sim["n_draws"]
    tr = sim["tier_recs"].copy()
    tm_keys = list(sim["nom_t1"].keys())
    srecs, precs = [], []
    pb = {}
    for (rule, b, j) in tm_keys:
        cr = cols.loc[j]
        rec = {"scenario": cr.scenario, "variant_id": cr.variant_id, "T": cr["T"], "rule": rule, "set": b,
               "n_tier1": int(len(sim["nom_t1"][(rule, b, j)])),
               "n_tier1_class_band": int(len(sim["nom_cb_t1"][(rule, b, j)]))}
        for mode in sim["modes"]:
            a = sim["acc"][mode]
            w = a["wins"][((rule, b), j)]
            p90 = pool(w, nd, 0.9)
            rec[f"coverage_{mode}"] = a["cover"][((rule, b), j)] / nd
            rec[f"coverage_class_band_tier1_{mode}"] = a["cover_cb"][((rule, b), j)] / nd
            rec[f"pool90_{mode}"] = len(p90)
            rec[f"n_distinct_winners_{mode}"] = len(w)
            top = max(w.items(), key=lambda kv: kv[1])
            rec[f"top_winner_{mode}"] = u.formula_pretty.iat[top[0]]
            rec[f"top_winner_p_{mode}"] = top[1] / nd
            for r, c in w.items():
                pb[(rule, b, j, r, mode)] = c / nd
            if mode == "copula":
                for rank, r in enumerate(p90):
                    precs.append({"scenario": cr.scenario, "variant_id": cr.variant_id, "rule": rule, "set": b,
                                  "rank": rank + 1, "row": r, "material_id": u.material_id.iat[r],
                                  "formula": u.formula_pretty.iat[r], "p_best": w[r] / nd,
                                  "in_tier1": r in set(sim["nom_t1"][(rule, b, j)].tolist())})
        srecs.append(rec)
    jof = {int(v): j for j, v in zip(cols.index, cols.variant_id)}
    for mode in sim["modes"]:
        tr[f"p_best_{mode}"] = [pb.get((ru, b, jof[int(v)], int(r), mode), 0.0)
                                for ru, b, v, r in zip(tr["rule"], tr["set"], tr["variant_id"], tr["row"])]
    tr = tr.merge(u[["row", "material_id", "formula_pretty", "density", "energy_above_hull", "theoretical",
                     "band_gap", "min_crustal_abundance_ppm", "elements"]], on="row", how="left")
    tr["elements"] = tr["elements"].apply(lambda e: "-".join(e))
    tr["delta_source"] = model.source[tr["row"].to_numpy()]
    tr["delta_meas"] = model.delta_meas[tr["row"].to_numpy()]
    tr["primary_class"] = model.cls[tr["row"].to_numpy()]
    return tr, pd.DataFrame(srecs), pd.DataFrame(precs)


def measured_coverage(model: RowModel, tr: pd.DataFrame, sim: dict) -> dict:
    """Share of tier members and of frontier-setting rows that have a measured density."""
    out = {}
    t1 = tr[(tr["tier"] == 1) & (tr["rule"] == "R2")]
    for (sc, b), g in t1.groupby(["scenario", "set"]):
        g = g.drop_duplicates("row")
        out[f"tier1_{sc}_{b}"] = {"n": int(len(g)), "n_measured": int((g["delta_source"] != "conditional").sum()),
                                  "by_source": g["delta_source"].value_counts().to_dict()}
    rows = set()
    for k in PRICE_CONSTRAINTS:
        for key in ("best_row_clean_F0", "best_row_clean_K"):
            rows.update(np.asarray(sim["nominal"][k][key]).tolist())
    rows = sorted(rows)
    u = common.universe()
    out["best_clean_frontier_rows_R2"] = [{"row": r, "formula": u.formula_pretty.iat[r],
                                           "material_id": u.material_id.iat[r], "source": model.source[r],
                                           "delta_meas": None if not np.isfinite(model.delta_meas[r])
                                           else float(model.delta_meas[r])} for r in rows]
    return out


# =============================================================================================
# figures
# =============================================================================================
def _fmt(f: str) -> str:
    import re
    return re.sub(r"(\d+)", r"$_{\1}$", f)


def fig_tiers(tr: pd.DataFrame) -> list[str]:
    plt = common.style()
    sets = ("B4", "B5", "B6")
    titles = {"B4": "B4", "B5": "B5 (B4 + band gap ≥ 3.1 eV)",
              "B6": "B6 (B4 + scarcity proxy (crustal abundance) ≥ 1 ppm)"}
    sc_lab = {"CS137": "Cs-137 (662 keV)", "CO60": "Co-60"}
    fig, axes = plt.subplots(2, 3, figsize=(common.DOUBLE_COL_IN, 6.4))
    for i, sc in enumerate(SCENARIOS):
        for j, b in enumerate(sets):
            ax = axes[i, j]
            g = tr[(tr.scenario == sc) & (tr.rule == "R2") & (tr.set == b) & np.isclose(tr["T"], DISPLAY_T)
                   & (tr.tier <= 2)].sort_values(["tier", "q_mid"])
            t1 = g[g.tier == 1]
            g = pd.concat([t1, g[g.tier == 2].head(max(0, 12 - len(t1)))])
            ys = np.arange(len(g))[::-1]
            for y, (_, r) in zip(ys, g.iterrows()):
                col = "#2a78d6" if r.tier == 1 else "#9aa5b1"
                ax.plot([r.q_lo, r.q_hi], [y, y], color=col, lw=2.2, solid_capstyle="butt")
                ax.plot([r.q_cm], [y], marker="|", color="#222222", ms=6, mew=1.0)
                ax.plot([r.q_mid], [y], marker="o", color=col, ms=3.2)
                if r.tier == 1:
                    ax.text(max(r.q_hi, r.q_cm), y, f"  {100 * r.p_best_copula:.0f} %", va="center", ha="left", fontsize=7,
                            color="#333333")
            ax.set_yticks(ys)
            ax.set_yticklabels([_fmt(f) + ("" if s != "conditional" else " *")
                                for f, s in zip(g.formula_pretty, g.delta_source)], fontsize=7)
            if len(t1) < len(g):
                ax.axhline(ys[len(t1) - 1] - 0.5, color="#bbbbbb", lw=0.6, ls="--")
            xmin, xmax = min(g.q_lo.min(), g.q_cm.min()), max(g.q_hi.max(), g.q_cm.max())
            ax.set_xlim(xmin - 0.04 * (xmax - xmin), xmax + 0.28 * (xmax - xmin))
            ax.set_title(f"{sc_lab[sc]}: {titles[b]}", loc="left", fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.grid(axis="x", color="#eeeeee", lw=0.5)
            if i == 1:
                ax.set_xlabel("narrow-beam thickness for T = 0.01 (cm)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#2a78d6", lw=2.2, label="could be the thinnest (tier 1): 90 % range of true thickness"),
         Line2D([], [], color="#9aa5b1", lw=2.2, label="next group (tier 2)"),
         Line2D([], [], color="#222222", marker="|", lw=0, ms=6, label="thickness with the MP density"),
         Line2D([], [], color="#2a78d6", marker="o", lw=0, ms=3.2, label="most likely true thickness")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.045))
    fig.text(0.01, -0.075, "% = share of simulated density draws in which the compound is the thinnest of the set. "
             "* = no measured density (range from COD-calibrated statistics for similar compounds); others use a "
             "measured density (COD crystal structure or Phase 1 element). B4 = lead-free, no radioactive element, "
             "e_hull ≤ 0.05 eV/atom, ICSD-matched. Rule R2.", fontsize=7, color="#444444",
             ha="left", va="top", wrap=True)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    out = common.savefig(fig, "fig6_tiers")
    plt.close(fig)
    return out


def fig_bootstrap(pt: pd.DataFrame) -> list[str]:
    plt = common.style()
    kinds = ("best_clean", "median50_clean", "median50", "best_icsd", "rule_original")
    klab = {"best_clean": "best (artefacts removed)", "median50_clean": "median of top 50 (artefacts removed)",
            "median50": "median of top 50", "best_icsd": "best ICSD-matched", "rule_original": "original per-cell rule"}
    sel = pt[np.isclose(pt["T"], DISPLAY_T)]
    fig, axes = plt.subplots(len(PRICE_CONSTRAINTS), 2, figsize=(common.DOUBLE_COL_IN, 8.0), sharey=True)
    mcol = {"copula": "#2a78d6", "iid": "#eb6834"}
    for i, k in enumerate(PRICE_CONSTRAINTS):
        for j, sc in enumerate(SCENARIOS):
            ax = axes[i, j]
            for y, kd in enumerate(kinds[::-1]):
                for dy, mode in ((0.17, "copula"), (-0.17, "iid")):
                    r = sel[(sel.scenario == sc) & (sel.constraint == k) & (sel.kind == kd) & (sel["mode"] == mode)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.plot([100 * r.p05, 100 * r.p95], [y + dy, y + dy], color=mcol[mode], lw=2.0,
                            solid_capstyle="butt")
                    ax.plot([100 * r["median"]], [y + dy], marker="o", color=mcol[mode], ms=2.8)
                r = sel[(sel.scenario == sc) & (sel.constraint == k) & (sel.kind == kd) & (sel["mode"] == "copula")]
                if not r.empty:
                    ax.plot([100 * r.iloc[0].nominal], [y], marker="|", color="#222222", ms=9, mew=1.2)
            ax.set_yticks(range(len(kinds)))
            ax.set_yticklabels([klab[k_] for k_ in kinds[::-1]], fontsize=7)
            ax.set_title(f"{'Cs-137' if sc == 'CS137' else 'Co-60'}: {k}", loc="left", fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", color="#eeeeee", lw=0.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if i == len(PRICE_CONSTRAINTS) - 1:
                ax.set_xlabel("thickness price (%), narrow beam, T = 0.01")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=mcol["copula"], lw=2, label="5th–95th percentile, densities correlated within a "
                                                           "chemical system (primary)"),
         Line2D([], [], color=mcol["iid"], lw=2, label="5th–95th percentile, independent densities"),
         Line2D([], [], color="#222222", marker="|", lw=0, ms=9, label="price with MP densities")]
    fig.legend(handles=h, loc="lower center", ncol=1, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(h_pad=0.8)
    out = common.savefig(fig, "T_bootstrap")
    plt.close(fig)
    return out


# =============================================================================================
# run
# =============================================================================================
def run(n_draws: int = N_DRAWS, figures: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    paths.ensure(OUT)
    model = RowModel()
    rt = model.table()
    rt.to_parquet(OUT / "delta_model_rows.parquet", index=False)
    model.cell_table.to_csv(OUT / "calibration_cells.csv", index=False)
    t_model = time.time() - t0
    sim = simulate(model, n_draws=n_draws, verbose=verbose)
    model8 = RowModel("k8")
    sim8 = simulate(model8, n_draws=n_draws, modes=SENS_MODES, verbose=verbose, label_suffix="_k8", nominal_from=sim)
    sim["acc"].update(sim8["acc"]); sim["modes"] += sim8["modes"]
    sim["point_k8"] = sim8["point"]; sim["seconds"] += sim8["seconds"]
    model_rr = RowModel("k5", resid_screen=False)
    simr = simulate(model_rr, n_draws=n_draws, modes=SENS_MODES, verbose=verbose, label_suffix="_rawresid",
                    nominal_from=sim)
    sim["acc"].update(simr["acc"]); sim["modes"] += simr["modes"]; sim["seconds"] += simr["seconds"]
    for lab, ceil in (("_noceiling", "none"), ("_ceilall", "all")):
        mc = RowModel("k5", ceiling=ceil)
        sc_ = simulate(mc, n_draws=n_draws, modes=SENS_MODES, verbose=verbose, label_suffix=lab, nominal_from=sim)
        sim["acc"].update(sc_["acc"]); sim["modes"] += sc_["modes"]; sim["seconds"] += sc_["seconds"]
    mp_curves = measured_point_curves(model)
    mp_curves.to_csv(OUT / "measured_point_curves.csv", index=False)
    mp_inc = measured_point_increments(model)
    mp_inc.to_csv(OUT / "measured_point_increments.csv", index=False)
    pt = price_table(sim)
    pt.to_csv(OUT / "bootstrap_prices.csv", index=False)
    fw = frontier_win_table(sim)
    fw.to_csv(OUT / "best_clean_frontier_draws.csv", index=False)
    tr, ts, pools = tier_tables(sim, model)
    tr.to_parquet(OUT / "tiers.parquet", index=False)
    tr[tr.tier == 1].to_csv(OUT / "tier1_members.csv", index=False)
    ts.to_csv(OUT / "tier_summary.csv", index=False)
    pools.to_csv(OUT / "pool90_members.csv", index=False)
    figs = (fig_tiers(tr) + fig_bootstrap(pt)) if figures else []
    summ = _summary(model, sim, pt, tr, ts, pools, figs, fw)
    summ["measured_point"] = _measured_point_summary(mp_curves, mp_inc)
    common.write_json(OUT / "summary.json", summ)
    common.write_json(OUT / "_timing.json", {"model_seconds": round(t_model, 1),
                                              "simulation_seconds": round(sim["seconds"], 1),
                                              "total_seconds": round(time.time() - t0, 1)})
    return summ


def _point_delta(model: RowModel) -> np.ndarray:
    """δ at the measured-density point: measured where available, cell median otherwise (as in simulate())."""
    return np.where(model.measured, model.delta_meas, model.med)


def _curve_masks() -> dict[str, np.ndarray]:
    from shieldmap.analysis3 import frontiers
    base = common.base_mask("R2") & common.clean_mask()
    out = {"F0": base}
    for k in CURVE_CONSTRAINTS:
        out[k] = base & frontiers._named_mask(k)
    return out


def measured_point_curves(model: RowModel, chunk: int = 64) -> pd.DataFrame:
    """Thickness basis, R2, with coherent, single lines on the continuum-module grid (20 keV–10 MeV): the best_clean price
    and frontier compounds with MP densities (nominal) and at the measured-density point, where every row's
    thickness is scaled by 1 + δ_point and the frontiers are recomputed. Also the marginal prices GAP | B4 and
    ABUND | B4 and the F0 thickness for T = 0.01."""
    from shieldmap.analysis3 import continuum
    gi = continuum.grid_indices()
    e = common.energy_keV()[gi]
    masks = _curve_masks()
    idx = {k: np.flatnonzero(m) for k, m in masks.items()}
    allrows = np.flatnonzero(np.logical_or.reduce(list(masks.values())))
    pos = {k: np.searchsorted(allrows, v) for k, v in idx.items()}
    scale = 1.0 + _point_delta(model)[allrows]
    meas = model.measured
    u = common.universe()
    form = u["formula_pretty"].to_numpy()
    recs = []
    for c0 in range(0, gi.size, chunk):
        cols = gi[c0:c0 + chunk]
        qn = common.mono_requirement("thickness", True, rows=allrows, e_idx=cols)
        qp = qn * scale[:, None]
        best = {}
        for k, p_ in pos.items():
            for lab, q in (("nominal", qn), ("point", qp)):
                sub_ = q[p_]
                a = np.argmin(sub_, axis=0)
                best[(k, lab)] = (sub_[a, np.arange(cols.size)], allrows[p_[a]])
        for j in range(cols.size):
            for k in CURVE_CONSTRAINTS + tuple(m[0] for m in CURVE_MARGINAL):
                b0, kk = ("F0", k) if "|" not in k else next((m[1], m[2]) for m in CURVE_MARGINAL if m[0] == k)
                r = {"energy_keV": float(e[c0 + j]), "constraint": k, "baseline": b0}
                for lab in ("nominal", "point"):
                    q0, r0 = best[(b0, lab)][0][j], best[(b0, lab)][1][j]
                    qk, rk = best[(kk, lab)][0][j], best[(kk, lab)][1][j]
                    r[f"price_{lab}"] = qk / q0 - 1.0
                    r[f"t_base_cm_{lab}"] = LN100 * q0; r[f"dt_cm_{lab}"] = LN100 * (qk - q0)
                    r[f"base_{lab}"] = form[r0]; r[f"constrained_{lab}"] = form[rk]
                    r[f"base_measured_{lab}"] = bool(meas[r0]); r[f"constrained_measured_{lab}"] = bool(meas[rk])
                r["factor_1pP"] = (1.0 + r["price_point"]) / (1.0 + r["price_nominal"])
                recs.append(r)
    return pd.DataFrame(recs)


def measured_point_increments(model: RowModel) -> pd.DataFrame:
    """Cs-137 / Co-60 profile variants (R2, every T): best_clean thickness frontiers and increments (cm) with MP
    densities and at the measured-density point, for the curve constraints and the two marginals."""
    cols = columns()
    q = np.asarray(common.profiles("thickness"))[:, cols["variant_id"].to_numpy()].astype(float)
    masks = _curve_masks()
    scale = 1.0 + _point_delta(model)
    form = common.universe()["formula_pretty"].to_numpy()
    recs = []
    for lab, qq in (("nominal", q), ("point", q * scale[:, None])):
        best = {}
        for k, m in masks.items():
            idx = np.flatnonzero(m)
            a = np.argmin(qq[idx], axis=0)
            best[k] = (qq[idx[a], np.arange(len(cols))], idx[a])
        for j, cr in cols.iterrows():
            for k in CURVE_CONSTRAINTS + tuple(m[0] for m in CURVE_MARGINAL):
                b0, kk = ("F0", k) if "|" not in k else next((m[1], m[2]) for m in CURVE_MARGINAL if m[0] == k)
                q0, r0 = best[b0][0][j], best[b0][1][j]
                qk, rk = best[kk][0][j], best[kk][1][j]
                recs.append({"scenario": cr.scenario, "variant_id": int(cr.variant_id), "T": float(cr["T"]),
                             "constraint": k, "baseline": b0, "density": lab, "t_base_cm": q0, "t_constrained_cm": qk,
                             "dt_cm": qk - q0, "price": qk / q0 - 1.0, "base": form[r0], "constrained": form[rk],
                             "base_measured": bool(model.measured[r0]), "constrained_measured": bool(model.measured[rk])})
    return pd.DataFrame(recs)


def _measured_point_summary(cur: pd.DataFrame, inc: pd.DataFrame) -> dict:
    out = {}
    for k, d in cur.groupby("constraint"):
        i = int(d.price_nominal.to_numpy().argmax())
        r = d.iloc[i]
        out[f"curve|{k}"] = {
            "max_nominal_pct": 100 * float(r.price_nominal), "energy_keV_at_max_nominal": float(r.energy_keV),
            "point_at_that_energy_pct": 100 * float(r.price_point),
            "max_point_pct": 100 * float(d.price_point.max()),
            "energy_keV_at_max_point": float(d.energy_keV.to_numpy()[d.price_point.to_numpy().argmax()]),
            "share_energies_both_frontiers_measured_point": float((d.base_measured_point & d.constrained_measured_point).mean()),
            "factor_1pP_range": [float(d.factor_1pP.min()), float(d.factor_1pP.max())],
            "key": {f"{E:g}": {"nominal_pct": 100 * float(x.price_nominal), "point_pct": 100 * float(x.price_point),
                              "dt_cm_T0.01_nominal": float(x.dt_cm_nominal), "dt_cm_T0.01_point": float(x.dt_cm_point),
                              "base_point": x.base_point, "constrained_point": x.constrained_point}
                    for E in (46.5, 100.0, 200.0, 300.0, 661.657, 1250.0)
                    for x in [d.iloc[int(np.abs(d.energy_keV.to_numpy() - E).argmin())]]}}
    t = inc[np.isclose(inc["T"], 0.01)]
    for (sc, k, lab), d in t.groupby(["scenario", "constraint", "density"]):
        out.setdefault(f"increments_T0.01|{sc}|{k}", {})[lab] = {
            "dt_cm_range": [float(d.dt_cm.min()), float(d.dt_cm.max())], "t_base_cm_range": [float(d.t_base_cm.min()), float(d.t_base_cm.max())],
            "price_pct_range": [100 * float(d.price.min()), 100 * float(d.price.max())],
            "base": sorted(set(d.base)), "constrained": sorted(set(d.constrained))}
    return out


def frontier_source_shares(fw: pd.DataFrame, model: RowModel) -> list[dict]:
    """Share of draws in which each best_clean frontier is set by a compound with a measured density."""
    f = fw.assign(measured=model.measured[fw["row"].to_numpy()])
    g = f.groupby(["mode", "scenario", "variant_id", "constraint", "side"])
    out = g.apply(lambda d: float(d.loc[d.measured, "share_of_draws"].sum()), include_groups=False)
    return out.rename("share_measured").reset_index().to_dict("records")


def _summary(model, sim, pt, tr, ts, pools, figs, fw=None) -> dict:
    u = common.universe()
    elig = u["eligible"].to_numpy(bool)
    src = pd.Series(model.source[elig]).value_counts().to_dict()
    head = pt[(pt["mode"] == "copula")].copy()
    key = head[np.isclose(head["T"], DISPLAY_T)]
    bands = {}
    for (sc, k, kd), g in key.groupby(["scenario", "constraint", "kind"]):
        r = g.iloc[0]
        bands.setdefault(sc, {}).setdefault(k, {})[kd] = {
            "nominal": round(100 * r.nominal, 2), "measured_point": round(100 * r.measured_point, 2),
            "p05": round(100 * r.p05, 2), "median": round(100 * r["median"], 2), "p95": round(100 * r.p95, 2),
            "width_ratio_copula_vs_iid": None if not np.isfinite(r.width_ratio_copula_vs_iid)
            else round(float(r.width_ratio_copula_vs_iid), 3)}
    iid = pt[(pt["mode"] == "iid") & np.isclose(pt["T"], DISPLAY_T)]
    for (sc, k, kd), g in iid.groupby(["scenario", "constraint", "kind"]):
        r = g.iloc[0]
        bands[sc][k][kd]["iid_p05_p95"] = [round(100 * r.p05, 2), round(100 * r.p95, 2)]
    flips = pt[(pt.kind == "rule_original")].groupby(["mode", "scenario", "constraint"])["flag_flip_rate"].agg(
        ["min", "max"]).reset_index().to_dict("records")
    wr = pt[(pt["mode"] == "copula") & pt.kind.isin(["best_clean", "median50", "median50_clean", "best_icsd"])]
    t1 = {}
    r2 = tr[(tr.rule == "R2") & (tr.tier == 1) & np.isclose(tr["T"], DISPLAY_T)]
    for (sc, b), g in r2.groupby(["scenario", "set"]):
        t1[f"{sc}/{b}"] = [f"{f} ({100 * p:.0f} %)" for f, p in
                           g.sort_values("p_best_copula", ascending=False)[["formula_pretty", "p_best_copula"]]
                           .itertuples(index=False)]
    tsd = ts[np.isclose(ts["T"], DISPLAY_T)]
    return {
        "step": "Phase 3 Step 10 (Agent T): data-conditioned density-error model, tiers v2, bootstrap bands v2",
        "convention": "δ = ρ_MP/ρ_true − 1; q_true = q_MP (1 + δ); thickness basis only",
        "n_draws": sim["n_draws"], "seed": SEED, "modes": sim["modes"],
        "rows_by_delta_source_eligible": {str(k): int(v) for k, v in src.items()},
        "residual_replicate_spread": {"n": int(model.resid.size), "n_before_screen": model.resid_raw_n,
                                      "p05_p50_p95": [float(x) for x in np.percentile(model.resid, [5, 50, 95])],
                                      "p90_abs": float(np.percentile(np.abs(model.resid), 90))},
        "icc": model.icc, "copula_rho": model.rho,
        "cells": model.cell_table.to_dict("records"),
        "class_band_source_for_comparison": sim["band_source"],
        "n_candidates": {m: sim["acc"][m]["n_candidates"] for m in sim["modes"]},
        "tier1_R2_T0.01": t1,
        "tier_summary_R2_T0.01": tsd[tsd.rule == "R2"].to_dict("records"),
        "tier1_size_range_all_rules_variants": ts.groupby(["scenario", "set"]).n_tier1.agg(["min", "max"])
        .reset_index().to_dict("records"),
        "price_bands_T0.01_pct": bands,
        "flag_flip_rate_rule_original": flips,
        "width_ratio_copula_vs_iid_range": [float(wr.width_ratio_copula_vs_iid.min()),
                                            float(wr.width_ratio_copula_vs_iid.max())],
        "measured_coverage": measured_coverage(model, tr, sim),
        "best_clean_frontier_set_by_measured_density_T0.01": [
            r for r in (frontier_source_shares(fw, model) if fw is not None else [])
            if r["mode"] == "copula" and r["variant_id"] in
            set(sim["cols"].loc[np.isclose(sim["cols"]["T"], DISPLAY_T), "variant_id"].astype(int))],
        "figures": figs,
    }


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DRAWS
    print(json.dumps(run(n_draws=n), indent=1, default=str)[:6000])
