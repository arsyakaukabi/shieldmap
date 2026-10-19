"""A4: sensitivity analysis and the headline-number table.

Density-bias propagation. Each compound's density is scaled by (1 + δ), with δ drawn
i.i.d. from the empirical stage 1 elemental bias distribution: 77 values of ρ_MP/ρ_exp − 1 in
reports/phase1/data/step4_density_bias.csv, derived from Tables A and B. There are 200 bootstrap
draws with a fixed seed.

At fixed μ/ρ, transmission depends only on ρx. Required thickness therefore scales exactly as
1/(1 + δ), and areal mass is unchanged. Areal-mass frontiers and rankings are exactly invariant to
density errors; only the thickness basis is perturbed.

The other axes (coherent definition, response, anode angle, Pb target, T target, representative
rule, frontier kind) are read from the price table data/phase2/A2/prices.parquet.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import core

OUT = core.P2 / "A4"
N_DRAWS = 200
SEED = 20261012


CORE_CUT = 0.10   # "core" variant: drop the 7 outliers with identified causes (|δ| > 10 %: Sn, Ac, Se, As, P, Ce, Yb)


def bias_distribution(variant: str = "empirical") -> np.ndarray:
    d = pd.read_csv(paths.ROOT / "reports" / "phase1" / "data" / "step4_density_bias.csv")
    x = d["rel_bias"].to_numpy(dtype=float)
    if variant == "empirical":
        return x
    if variant == "core":
        return x[np.abs(x) <= CORE_CUT]
    raise ValueError(variant)


def density_bootstrap(bundles=("B0", "B4", "B5", "B6"), top=10, variant: str = "empirical"
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap the thickness basis under density perturbation, for each scenario's primary variants.

    Returns (draw-level table, summary table).
    """
    rng = np.random.default_rng(SEED)
    delta = bias_distribution(variant)
    u = core.universe()
    base = core.base_mask(core.PRIMARY_RULE)
    rows_base = np.flatnonzero(base)
    q_all = core.profiles("thickness")
    recs = []
    for sc in ("CS137", "CO60", "DIAG"):
        v = core.scenario_variants(sc)
        vids = v["variant_id"].to_numpy()
        q = np.asarray(q_all[np.ix_(rows_base, vids)], dtype=np.float64)        # (n_base × V)
        masks = {b: core.bundle_mask(b)[rows_base] for b in bundles}
        nominal = {}
        for b, m in masks.items():
            nominal[b] = {
                "q": core.frontier(q, np.ones(len(rows_base), bool) & m, "best"),
                "top": [frozenset(rows_base[core.top_k(q[:, j], m, top)]) for j in range(q.shape[1])],
                "top1": [rows_base[core.top_k(q[:, j], m, 1)][0] for j in range(q.shape[1])],
            }
        q0 = core.frontier(q, np.ones(len(rows_base), bool), "best")
        for d in range(N_DRAWS):
            s = 1.0 / (1.0 + rng.choice(delta, size=len(rows_base), replace=True))
            qd = q * s[:, None]
            q0d = core.frontier(qd, np.ones(len(rows_base), bool), "best")
            for b, m in masks.items():
                qb = core.frontier(qd, m, "best")
                for j, vid in enumerate(vids):
                    ordered = rows_base[core.top_k(qd[:, j], m, top)]
                    topd = frozenset(ordered)
                    recs.append({"distribution": variant, "scenario": sc, "variant_id": int(vid), "bundle": b, "draw": d,
                                 "price": qb[j] / q0d[j] - 1.0,
                                 "nominal_price": nominal[b]["q"][j] / q0[j] - 1.0,
                                 "top_overlap": len(topd & nominal[b]["top"][j]) / top,
                                 "top_changed": topd != nominal[b]["top"][j],
                                 "top1_changed": ordered[0] != nominal[b]["top1"][j]})
    draws = pd.DataFrame(recs)
    g = draws.groupby(["distribution", "scenario", "bundle"])
    summ = g.agg(nominal_price_min=("nominal_price", "min"), nominal_price_max=("nominal_price", "max"),
                 price_p05=("price", lambda x: np.quantile(x, 0.05)),
                 price_median=("price", "median"),
                 price_p95=("price", lambda x: np.quantile(x, 0.95)),
                 frac_top10_changed=("top_changed", "mean"),
                 mean_top10_overlap=("top_overlap", "mean"),
                 min_top10_overlap=("top_overlap", "min"),
                 frac_top1_changed=("top1_changed", "mean")).reset_index()
    return draws, summ


def top_packing(top=10) -> pd.DataFrame:
    """How tightly the nominal top-10 is packed: (q_10 / q_1 − 1) per scenario × bundle × primary variant."""
    base = core.base_mask(core.PRIMARY_RULE)
    recs = []
    for sc in ("CS137", "CO60", "DIAG"):
        for basis in core.BASES:
            q = core.profiles(basis)
            for b in ("B0", "B4", "B5", "B6"):
                m = base & core.bundle_mask(b)
                for vid in core.scenario_variants(sc)["variant_id"]:
                    t = core.top_k(q[:, vid], m, top)
                    recs.append({"scenario": sc, "basis": basis, "bundle": b, "variant_id": int(vid),
                                 "q1": float(q[t[0], vid]), "q10": float(q[t[-1], vid]),
                                 "gap_10_vs_1": float(q[t[-1], vid] / q[t[0], vid] - 1)})
    return pd.DataFrame(recs)


def run() -> dict:
    paths.ensure(OUT)
    parts = [density_bootstrap(variant=v) for v in ("empirical", "core")]
    draws = pd.concat([p[0] for p in parts], ignore_index=True)
    summ = pd.concat([p[1] for p in parts], ignore_index=True)
    top_packing().to_csv(OUT / "top10_packing.csv", index=False)
    draws.to_parquet(OUT / "density_bootstrap_draws.parquet", index=False)
    summ.to_csv(OUT / "density_bootstrap_summary.csv", index=False)
    info = {"n_draws": N_DRAWS, "seed": SEED, "bias_values": int(bias_distribution().size),
            "core_bias_values": int(bias_distribution("core").size), "core_cut_abs": CORE_CUT,
            "basis": "thickness (areal mass is exactly invariant to density scaling)"}
    (OUT / "density_bootstrap_meta.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    head = write_headline()
    return {"summary": summ, "headline": head}


if __name__ == "__main__":
    print(run()["summary"].to_string(index=False))


# ---------------------------------------------------------------------------------------------
# headline table (A4.1, A4.3): each headline price with its range along every sensitivity axis
# ---------------------------------------------------------------------------------------------
HEADLINE_CONSTRAINTS = ["PB", "RAD", "STAB(0.05)", "EXP", "GAP(3.1)", "ABUND(1)", "B1", "B2", "B3", "B4", "B5", "B6"]


def _rng(x: pd.Series) -> str:
    x = 100 * x.dropna()
    if x.empty:
        return "—"
    lo, hi = x.min(), x.max()
    return f"{lo:.1f}" if abs(hi - lo) < 0.05 else f"{lo:.1f} … {hi:.1f}"


def headline_table() -> pd.DataFrame:
    p = pd.read_parquet(core.P2 / "A2" / "prices.parquet")
    p = p[(p.baseline == "F0") & p.constraint.isin(HEADLINE_CONSTRAINTS)]
    v = core.variants().set_index("variant_id")
    p = p.join(v[["response", "coherent", "anode_angle", "quality", "target_value"]], on="variant_id")
    boot = pd.read_csv(OUT / "density_bootstrap_summary.csv")
    rows = []
    for sc in ("DIAG", "CS137", "CO60"):
        basis = core.default_basis(sc)
        other = "thickness" if basis == "areal" else "areal"
        s = p[p.scenario == sc]
        prim = s[s.primary & (s.rule == "R2") & (s.kind == "best")]
        for c in HEADLINE_CONSTRAINTS:
            pc = prim[(prim.constraint == c) & (prim.basis == basis)]
            sc_all = s[s.constraint == c]
            ax = {
                "headline_median_pct": 100 * pc.price.median(),
                "sweep": _rng(pc.price),
                "coherent_off": _rng(sc_all[(sc_all.basis == basis) & (sc_all.rule == "R2") & (sc_all.kind == "best")
                                            & (sc_all.response == "air_kerma") & ~sc_all.coherent].price),
                "response_fluence_energy": _rng(sc_all[(sc_all.basis == basis) & (sc_all.rule == "R2")
                                                       & (sc_all.kind == "best") & sc_all.coherent
                                                       & (sc_all.response != "air_kerma")].price),
                "rule_R1_R3": _rng(sc_all[(sc_all.basis == basis) & sc_all.primary & (sc_all.kind == "best")
                                          & sc_all.rule.isin(["R1", "R3"])].price),
                "kind_10th_median50": _rng(sc_all[(sc_all.basis == basis) & sc_all.primary & (sc_all.rule == "R2")
                                                  & sc_all.kind.isin(["10th", "median50"])].price),
                "all_axes": _rng(sc_all[(sc_all.basis == basis)].price),
                f"other_basis_{other}": _rng(sc_all[(sc_all.basis == other) & sc_all.primary & (sc_all.rule == "R2")
                                                    & (sc_all.kind == "best")].price),
            }
            if sc == "DIAG":
                for q in ("RQR5", "RQR6", "RQR8", "RQR9"):
                    ax[f"DIAG_{q}"] = _rng(pc[pc.quality == q].price)
            if c in ("B4", "B5", "B6"):
                for dist in ("empirical", "core"):
                    b = boot[(boot.distribution == dist) & (boot.scenario == sc) & (boot.bundle == c)]
                    if len(b):
                        ax[f"density_{dist}_p05_p95_thickness"] = (f"{100*b.price_p05.iloc[0]:.1f} … "
                                                                   f"{100*b.price_p95.iloc[0]:.1f}")
            rows.append({"scenario": sc, "basis": basis, "constraint": c, **ax})
    return pd.DataFrame(rows)


def write_headline() -> pd.DataFrame:
    h = headline_table()
    h.to_csv(OUT / "headline.csv", index=False)
    lines = ["# A4 headline table (generated by `sensitivity.write_headline`)", "",
             "Prices in % extra requirement. Headline: rule R2, frontier `best`, primary variants (air kerma, with "
             "coherent), primary basis (DIAG areal mass; CS137/CO60 thickness); median across the primary sweep. "
             "Every other column is a range (min … max) along one sensitivity axis; `all_axes` covers every Table C "
             "variant × rule × frontier kind on the primary basis. Density-bootstrap ranges are 5th–95th percentiles "
             "on the thickness basis (areal mass is invariant).", ""]
    cols = [c for c in h.columns if c not in ("scenario", "basis")]
    for sc in ("DIAG", "CS137", "CO60"):
        d = h[h.scenario == sc].dropna(axis=1, how="all")
        cs = [c for c in cols if c in d.columns]
        lines += [f"## {sc} ({d.basis.iloc[0]})", "", "| " + " | ".join(cs) + " |",
                  "|" + "---|" * len(cs)]
        for r in d.itertuples(index=False):
            vals = [getattr(r, c) for c in cs]
            vals = ["—" if (isinstance(x, float) and x != x) or str(x) == "nan" else (f"{x:.1f}" if isinstance(x, float) else str(x)) for x in vals]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    (core.REPORTS / "A4_tables.md").write_text("\n".join(lines), encoding="utf-8")
    return h
