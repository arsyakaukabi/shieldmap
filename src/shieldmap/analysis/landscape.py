"""Analysis A1: the landscape of photon attenuation over the compound map (RQ1).

Regenerate everything with ``python -c "from shieldmap.analysis import landscape; landscape.run()"``.
Idempotent: every output is overwritten from Tables A/B/C (and the stage 2 universe) only.

Outputs
  data/phase2/A1/
    quantiles_key_energies.{parquet,csv}  per key energy: quantiles of μ/ρ and μ over the R2 map + envelope
    variance_decomposition.{parquet,csv}  per energy × rule × coherent × subset: var(log μ) split into
                                          var(log ρ), var(log μ/ρ), 2 cov and their shares
    crossings.csv                         energies where composition share = density share
    envelope_distance_summary.parquet     per energy: quantiles of the two envelope ratios, counts > 1
    envelope_ratio_key_energies.parquet   per R2 compound (Table C row) × key energy: the three ratios (float32)
    envelope_top20.csv                    top-20 non-radioactive R2 compounds by each ratio at 60/661.657/1250 keV
    envelope_mu_exceed.csv                eligible entries whose μ exceeds the max elemental-solid μ
    polymorph_groups.parquet              per formula with ≥ 3 eligible polymorphs: μ/ρ and ρ spreads
    spot_check.json                       μ, μ/ρ recomputed vs key_energies.parquet
    summary.json                          every number quoted in reports/phase2/A1_report.md
  reports/phase2/figures/A1_histograms.png, A1_variance_share.png, A1_envelope_distance.png,
                         A1_polymorph_density.png

Definitions
  x = log10 ρ, y(E) = log10 (μ/ρ)(E), log10 μ(E) = x + y.
  var(log μ) = var(x) + var(y) + 2 cov(x, y) across the compounds of a subset (population moments).
  density share = var(x)/var(log μ); composition share = var(y)/var(log μ); cov share = 2cov/var(log μ).
  The three shares sum to 1; the cov share can be negative.
  Subsets: "all" (the map), "top10_rho" (top 10 % by density; not conditioned on the outcome),
  "top10_mu" and "top1_mu" (top 10 % / 1 % by μ at that energy; conditioned on the outcome, which
  induces a negative cov(x, y) by selection on x + y).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import core

OUT = core.P2 / "A1"
FIG = core.FIGURES
KEY_ENERGIES = (20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0, 500.0, 661.657, 1000.0, 1250.0,
                2000.0, 5000.0, 10000.0)
ENV_ENERGIES = (60.0, 661.657, 1250.0)
CO60_LINES = (1173.228, 1332.492)
CS137_LINE = 661.657
SUBSETS = ("all", "top10_rho", "top10_mu", "top1_mu")
COHS = ("wcoh", "wocoh")
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
BLOCK = 48

# dataviz reference palette (light mode, categorical slots 1-3; ≤ 3 series per chart)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
def energies() -> np.ndarray:
    return np.load(paths.PROCESSED / "energy_keV.npy")


def energy_index(values, e: np.ndarray | None = None) -> np.ndarray:
    """Grid indices of the requested energies; every value must be an exact grid node."""
    e = energies() if e is None else e
    idx = []
    for v in values:
        hit = np.flatnonzero(e == float(v))
        if hit.size != 1:
            raise AssertionError(f"{v} keV is not an exact grid node")
        idx.append(int(hit[0]))
    return np.array(idx)


def envelope() -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / "envelope.parquet")


def nonradioactive_envelope() -> pd.DataFrame:
    """The same envelope restricted to elements with a stable isotope (from Table A cross sections).

    Also returns the recomputed unfiltered envelope so it can be checked against envelope.parquet.
    """
    xs = np.load(paths.INTERIM / "elements" / "xsec_grid.npz")
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet").sort_values("Z")
    sym = el["symbol"].to_numpy()
    rho = el["density_solid_exp"].to_numpy(dtype=float)
    rad = el["is_radioactive"].to_numpy(dtype=bool)
    out = {"energy_keV": xs["energy_keV"]}
    for tag, key in (("wcoh", "total_wcoh"), ("wocoh", "total_wocoh")):
        m = xs[key]
        for lab, keep in (("all", np.ones_like(rad)), ("nonrad", ~rad)):
            mm = m[keep]
            out[f"{lab}_max_mu_rho_{tag}"] = mm.max(axis=0)
            out[f"{lab}_argmax_mu_rho_{tag}"] = sym[keep][mm.argmax(axis=0)]
            has = keep & np.isfinite(rho)
            mu = m[has] * rho[has, None]
            out[f"{lab}_max_mu_{tag}"] = mu.max(axis=0)
            out[f"{lab}_argmax_mu_{tag}"] = sym[has][mu.argmax(axis=0)]
    return pd.DataFrame(out)


def diag_mean_energies() -> pd.Series:
    s = pd.read_parquet(paths.PROCESSED / "spectra.parquet")
    s = s[s.kind == "tube"]
    return pd.Series([json.loads(m)["spekpy_mean_energy_keV"] for m in s.meta_json], index=s.spectrum.values)


def iter_blocks(coh: str, rows: np.ndarray, block: int = BLOCK):
    """Yield (energy slice, log10 μ/ρ block (len(rows) × k), float64) over the full grid."""
    arr = np.load(paths.PROCESSED / f"mu_rho_{coh}.npy")          # 336 MB, in memory once
    n_e = arr.shape[1]
    try:
        for j0 in range(0, n_e, block):
            j1 = min(n_e, j0 + block)
            blk = arr[:, j0:j1][rows].astype(np.float64)
            yield slice(j0, j1), np.log10(blk), blk
    finally:
        del arr


# ---------------------------------------------------------------------------------------------
# variance decomposition
# ---------------------------------------------------------------------------------------------
def decompose(x: np.ndarray, Y: np.ndarray) -> dict:
    """Population-moment decomposition of var(x + y) for each column of Y (n × k)."""
    x = np.asarray(x, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    vx = float((xc ** 2).mean())
    vy = (Yc ** 2).mean(axis=0)
    cv = (xc[:, None] * Yc).mean(axis=0)
    vmu = vx + vy + 2 * cv
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"n": np.full(Y.shape[1], x.size), "var_logmu": vmu, "var_logrho": np.full(Y.shape[1], vx),
                "var_logmurho": vy, "cov": cv, "share_density": vx / vmu, "share_composition": vy / vmu,
                "share_cov": 2 * cv / vmu, "corr": cv / np.sqrt(vx * vy)}


def _decompose_top_mu(x: np.ndarray, Y: np.ndarray, frac: float) -> dict:
    n = x.size
    k = max(2, int(math.ceil(frac * n)))
    parts = []
    for j in range(Y.shape[1]):
        lm = x + Y[:, j]
        idx = np.argpartition(lm, n - k)[n - k:]
        parts.append(decompose(x[idx], Y[idx, j]))
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def find_crossings(e: np.ndarray, d: np.ndarray, edges: np.ndarray | None = None) -> list[dict]:
    """Zero crossings of d (= composition share − density share) on the grid, linear in log10 E.

    direction "comp->dens": composition dominates below, density above (d goes + → −).
    edge: an absorption edge (K or L1–3 of an element in the map) lies between the two nodes.
    """
    out = []
    le = np.log10(e)
    for i in range(d.size - 1):
        a, b = d[i], d[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0 or (a > 0) == (b > 0):
            continue
        t = a / (a - b)
        ec = 10 ** (le[i] + t * (le[i + 1] - le[i]))
        rec = {"E_cross_keV": float(ec), "E_lo_keV": float(e[i]), "E_hi_keV": float(e[i + 1]),
               "direction": "comp->dens" if a > 0 else "dens->comp", "d_lo": float(a), "d_hi": float(b),
               "edge_between_nodes": ""}
        if edges is not None and len(edges):
            hit = edges[(edges.energy_keV > e[i]) & (edges.energy_keV < e[i + 1])]
            rec["edge_between_nodes"] = ";".join(f"{s} {sh}" for s, sh in zip(hit.symbol, hit.shell))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------
# main pass over the μ/ρ arrays
# ---------------------------------------------------------------------------------------------
def _main_pass(u: pd.DataFrame, e: np.ndarray, key_idx: np.ndarray) -> dict:
    elig = np.flatnonzero(u.eligible.to_numpy())
    rho = u.density.to_numpy(dtype=np.float64)[elig]
    x = np.log10(rho)
    rule_masks = {r: u[f"in_{r}"].to_numpy(dtype=bool)[elig] for r in core.RULES}
    env = envelope()
    env_mr = env.max_mu_rho_wcoh.to_numpy()
    env_mu = env.max_mu_wcoh.to_numpy()
    env_mu_nr = nonradioactive_envelope().nonrad_max_mu_wcoh.to_numpy()
    nonrad = ~u.contains_radioactive.to_numpy()[elig]
    b4 = core.bundle_mask("B4", u)[elig]
    # per eligible entry: where μ exceeds the max μ of every non-radioactive elemental solid
    nr_max_ratio = np.zeros(elig.size); nr_n_e = np.zeros(elig.size, dtype=int)
    nr_e_lo = np.full(elig.size, np.nan); nr_e_hi = np.full(elig.size, np.nan)
    nr_e_at_max = np.full(elig.size, np.nan)

    dec_rows = []
    key_cols = {}                     # coh -> (n_elig × n_key) μ/ρ at key energies
    env_summ = []
    env_exceed_cnt = {}
    r2 = rule_masks["R2"]
    hist_vals = {}                    # (energy) -> (log μ/ρ, log μ) over R2
    for coh in COHS:
        kc = np.empty((elig.size, key_idx.size), dtype=np.float64)
        for sl, L, blk in iter_blocks(coh, elig):
            js = np.arange(sl.start, sl.stop)
            # key-energy columns
            for kk, j in enumerate(key_idx):
                if sl.start <= j < sl.stop:
                    kc[:, kk] = blk[:, j - sl.start]
            for rule, m in rule_masks.items():
                xr, Yr = x[m], L[m]
                thr = np.quantile(xr, 0.9)
                top_rho = xr >= thr
                res = {"all": decompose(xr, Yr), "top10_rho": decompose(xr[top_rho], Yr[top_rho]),
                       "top10_mu": _decompose_top_mu(xr, Yr, 0.10), "top1_mu": _decompose_top_mu(xr, Yr, 0.01)}
                for sub, d in res.items():
                    df = pd.DataFrame(d)
                    df.insert(0, "energy_keV", e[js]); df.insert(1, "rule", rule)
                    df.insert(2, "coherent", coh == "wcoh"); df.insert(3, "subset", sub)
                    dec_rows.append(df)
            if coh == "wcoh":
                mu = blk * rho[:, None]
                rmr = blk / env_mr[js]
                rmu = mu / env_mu[js]
                rnr = mu / env_mu_nr[js]
                over = rnr > 1
                nr_n_e += over.sum(axis=1)
                for t, j in enumerate(js):
                    hit = over[:, t]
                    nr_e_lo[hit & np.isnan(nr_e_lo)] = e[j]
                    nr_e_hi[hit] = e[j]
                bm = rnr.max(axis=1); ba = js[rnr.argmax(axis=1)]
                upd = bm > nr_max_ratio
                nr_max_ratio[upd] = bm[upd]; nr_e_at_max[upd] = e[ba[upd]]
                for rule in ("R2", "R3"):
                    m = rule_masks[rule]
                    qs = np.quantile(rmr[m], [0.5, 0.9, 0.99, 0.999], axis=0)
                    qm = np.quantile(rmu[m], [0.5, 0.9, 0.99, 0.999], axis=0)
                    for t, j in enumerate(js):
                        env_summ.append({"energy_keV": e[j], "rule": rule,
                                         "ratio_mu_rho_median": qs[0, t], "ratio_mu_rho_q90": qs[1, t],
                                         "ratio_mu_rho_q99": qs[2, t], "ratio_mu_rho_q999": qs[3, t],
                                         "ratio_mu_rho_max": rmr[m, t].max(),
                                         "ratio_mu_median": qm[0, t], "ratio_mu_q90": qm[1, t],
                                         "ratio_mu_q99": qm[2, t], "ratio_mu_q999": qm[3, t],
                                         "ratio_mu_max": rmu[m, t].max(),
                                         "n_mu_ratio_gt1": int((rmu[m, t] > 1).sum()),
                                         "n_mu_ratio_gt1_nonrad": int(((rmu[m, t] > 1) &
                                             ~u.contains_radioactive.to_numpy()[elig][m]).sum()),
                                         "n_nonrad_gt_nonrad_env": int((rnr[m, t] > 1)[nonrad[m]].sum()),
                                         "n_B4_gt_nonrad_env": int((rnr[m, t] > 1)[b4[m]].sum()),
                                         "ratio_mu_nonrad_env_max_nonrad": float(rnr[m, t][nonrad[m]].max()),
                                         "n": int(m.sum())})
        key_cols[coh] = kc
    dec = pd.concat(dec_rows, ignore_index=True)
    ue = u.iloc[elig]
    sel = nonrad & (nr_n_e > 0)
    beat = pd.DataFrame({"material_id": ue.material_id.values[sel], "formula_pretty": ue.formula_pretty.values[sel],
                         "density": ue.density.values[sel], "energy_above_hull": ue.energy_above_hull.values[sel],
                         "theoretical": ue.theoretical.values[sel], "in_R2": rule_masks["R2"][sel],
                         "in_B4": b4[sel], "max_ratio_mu_nonrad_env": nr_max_ratio[sel],
                         "energy_at_max_keV": nr_e_at_max[sel], "first_energy_keV": nr_e_lo[sel],
                         "last_energy_keV": nr_e_hi[sel], "n_grid_energies": nr_n_e[sel]}
                        ).sort_values("max_ratio_mu_nonrad_env", ascending=False)
    return {"elig": elig, "rho": rho, "rule_masks": rule_masks, "dec": dec, "key_cols": key_cols,
            "env_summary": pd.DataFrame(env_summ), "beat_nonrad_env": beat}


# ---------------------------------------------------------------------------------------------
# A1.1 histograms and quantiles
# ---------------------------------------------------------------------------------------------
def quantile_table(mr: np.ndarray, mu: np.ndarray, e_key, env: pd.DataFrame, nonrad: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, ek in enumerate(e_key):
        er = env[env.energy_keV == ek].iloc[0]
        nr = nonrad[nonrad.energy_keV == ek].iloc[0]
        rec = {"energy_keV": ek, "n": mr.shape[0]}
        for q in QUANTILES:
            rec[f"mu_rho_q{q:g}"] = float(np.quantile(mr[:, k], q))
        for q in QUANTILES:
            rec[f"mu_q{q:g}"] = float(np.quantile(mu[:, k], q))
        rec.update({"env_max_mu_rho": er.max_mu_rho_wcoh, "env_argmax_mu_rho": er.argmax_mu_rho_wcoh,
                    "env_max_mu": er.max_mu_wcoh, "env_argmax_mu": er.argmax_mu_wcoh,
                    "env_nonrad_max_mu_rho": nr.nonrad_max_mu_rho_wcoh,
                    "env_nonrad_argmax_mu_rho": nr.nonrad_argmax_mu_rho_wcoh,
                    "env_nonrad_max_mu": nr.nonrad_max_mu_wcoh, "env_nonrad_argmax_mu": nr.nonrad_argmax_mu_wcoh,
                    "frac_mu_rho_within_10pct_env": float((mr[:, k] >= 0.9 * er.max_mu_rho_wcoh).mean()),
                    "frac_mu_within_50pct_env": float((mu[:, k] >= 0.5 * er.max_mu_wcoh).mean()),
                    "n_mu_gt_env": int((mu[:, k] > er.max_mu_wcoh).sum())})
        rows.append(rec)
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_histograms(mr, mu, e_key, env, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(7, 4, figsize=(15, 21))
    for k, ek in enumerate(e_key):
        er = env[env.energy_keV == ek].iloc[0]
        for q, (vals, envv, envel, lab, col) in enumerate((
                (mr[:, k], er.max_mu_rho_wcoh, er.argmax_mu_rho_wcoh, "μ/ρ (cm²/g)", C1),
                (mu[:, k], er.max_mu_wcoh, er.argmax_mu_wcoh, "μ (cm⁻¹)", C2))):
            ax = axes[k // 2, 2 * (k % 2) + q]
            _style(ax)
            lv = np.log10(vals)
            lo, hi = lv.min(), max(lv.max(), np.log10(envv))
            pad = 0.03 * (hi - lo)
            ax.hist(lv, bins=np.linspace(lo - pad, hi + pad, 90), color=col, histtype="stepfilled",
                    alpha=0.85, edgecolor=col, lw=0.5)
            ax.axvline(np.log10(envv), color=INK, lw=1.2, ls="--")
            ax.text(np.log10(envv), 0.97, f" envelope\n ({envel})", transform=ax.get_xaxis_transform(),
                    ha="right", va="top", fontsize=7, color=INK)
            ax.set_yscale("log")
            ek_s = f"{ek:g}"
            ax.set_title(f"{ek_s} keV · log₁₀ {lab.split(' ')[0]}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel(f"log₁₀ {lab}", fontsize=8, color=INK2)
            if q == 0 and k % 2 == 0:
                ax.set_ylabel("compounds per bin", fontsize=8, color=INK2)
    fig.suptitle("A1 · Distribution of μ/ρ and μ over the R2 map (with coherent; n = "
                 f"{mr.shape[0]:,}); dashed = elemental envelope (max μ/ρ over Z = 1–100; "
                 "max μ over elemental solids)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# A1.2 figure
# ---------------------------------------------------------------------------------------------
def fig_variance_share(dec: pd.DataFrame, cross: pd.DataFrame, diag_e: pd.Series, path):
    """(a), (b): shares of var(log μ). (c), (d): outcome-conditioned tails, where the shares leave [0, 1]
    (selection on log μ makes var(log μ) small and cov strongly negative), so the ratio
    var(log μ/ρ) / var(log ρ) is drawn instead on a log axis; a crossing is ratio = 1 in both forms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    titles = {"all": "(a) whole R2 map: shares of var(log μ)",
              "top10_rho": "(b) top 10 % by density (not outcome-conditioned): shares",
              "top10_mu": "(c) top 10 % by μ at each energy (outcome-conditioned): variance ratio",
              "top1_mu": "(d) top 1 % by μ at each energy (outcome-conditioned): variance ratio"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.4), sharex=True)
    d_all = dec.assign(ratio=dec.var_logmurho / dec.var_logrho)
    for ax, sub in zip(axes.flat, SUBSETS):
        _style(ax)
        d = d_all[d_all.subset == sub]
        prim = d[(d.rule == "R2") & d.coherent].sort_values("energy_keV")
        e = prim.energy_keV.to_numpy()
        ax.axvspan(diag_e.min(), diag_e.max(), color="#d9d8d3", alpha=0.6, lw=0)
        for ev in (CS137_LINE,) + CO60_LINES:
            ax.axvline(ev, color=MUTED, lw=0.8, ls=":")
        cr = cross[(cross.subset == sub) & (cross.rule == "R2") & cross.coherent]
        if sub in ("all", "top10_rho"):
            for col, c in (("share_density", C1), ("share_composition", C2), ("share_cov", C3)):
                piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values=col)
                ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=c, alpha=0.18, lw=0)
                ax.plot(e, prim[col], color=c, lw=2 if col != "share_cov" else 1.4)
            ycr = lambda ec: np.interp(np.log10(ec), np.log10(e), prim.share_density)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_ylabel("share of var(log₁₀ μ)", fontsize=9, color=INK2)
        else:
            piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values="ratio")
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=C2, alpha=0.18, lw=0)
            ax.plot(e, prim.ratio, color=C2, lw=2)
            ax.axhline(1, color=INK, lw=1, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(min(0.03, float(piv.min().min()) * 0.8), 5)
            ax.text(11, 1.08, "composition varies more ↑", fontsize=7.5, color=INK2, va="bottom")
            ax.text(11, 0.93, "density varies more ↓", fontsize=7.5, color=INK2, va="top")
            ycr = lambda ec: 1.0
            ax.set_ylabel("var(log μ/ρ) ÷ var(log ρ)", fontsize=9, color=INK2)
        for _, r in cr.iterrows():
            ax.plot(r.E_cross_keV, ycr(r.E_cross_keV), "o", ms=7, mfc="white", mec=INK, mew=1.4, zorder=5)
        main = cr[cr.direction == "comp->dens"]
        if len(main):
            ec = main.E_cross_keV.iloc[-1]
            ax.annotate(f"{ec:.0f} keV", (ec, ycr(ec)), xytext=(10, 10), textcoords="offset points",
                        fontsize=8.5, color=INK)
        ax.set_xscale("log")
        ax.set_title(titles[sub], fontsize=9.5, color=INK, loc="left")
        ylo, yhi = ax.get_ylim()
        ax.text(np.sqrt(diag_e.min() * diag_e.max()), yhi, "DIAG\n(RQR mean E)", ha="center", va="top",
                fontsize=7, color=INK2)
        ax.text(CS137_LINE, yhi, "Cs-137 ", ha="right", va="top", fontsize=7, color=INK2, rotation=90)
        ax.text(CO60_LINES[1], yhi, " Co-60", ha="left", va="top", fontsize=7, color=INK2, rotation=90)
    for ax in axes[1]:
        ax.set_xlabel("photon energy (keV, log scale)", fontsize=9, color=INK2)
    handles = [Line2D([], [], color=C1, lw=2, label="density share  var(log ρ)/var(log μ)"),
               Line2D([], [], color=C2, lw=2, label="composition share  var(log μ/ρ)/var(log μ)  [(c),(d): ratio to var(log ρ)]"),
               Line2D([], [], color=C3, lw=1.4, label="2 cov(log ρ, log μ/ρ)/var(log μ)"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="crossing (composition = density)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("A1 · Composition vs density across compounds (lines: R2, with coherent; bands: range over "
                 "R1/R2/R3 × with/without coherent)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# A1.3 envelope distance
# ---------------------------------------------------------------------------------------------
def fig_envelope(ratios: pd.DataFrame, env_summary: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[2]; _style(ax)
    es = env_summary[env_summary.rule == "R2"].sort_values("energy_keV")
    for col, c, lab in (("n_nonrad_gt_nonrad_env", C1, "non-radioactive R2 compounds"),
                        ("n_B4_gt_nonrad_env", C2, "… also in bundle B4 (no Pb, e_hull ≤ 0.05, ICSD)")):
        y = es[col].to_numpy().astype(float)
        ax.step(es.energy_keV, y, where="mid", color=c, lw=1.8, label=lab)
    pos = es[es.n_nonrad_gt_nonrad_env > 0].energy_keV
    lo, hi = (pos.min(), pos.max()) if len(pos) else (40, 80)
    ax.set_xlim(lo - 5, hi + 5)
    ax.text(0.02, 0.97, f"zero at every other grid energy (10 keV – 15 MeV)", transform=ax.transAxes,
            fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel("photon energy (keV, linear; 0.5 keV grid)", fontsize=9, color=INK2)
    ax.set_ylabel("compounds with μ > max μ of every\nnon-radioactive elemental solid", fontsize=9, color=INK2)
    ax.set_title("(c) compounds above the non-radioactive envelope", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 0.92))
    for ax, col, lab in ((axes[0], "ratio_mu_rho", "(μ/ρ) ÷ max elemental μ/ρ"),
                         (axes[1], "ratio_mu", "μ ÷ max elemental-solid μ (all elements)")):
        _style(ax)
        allv = np.log10(ratios[col].to_numpy())
        bins = np.linspace(allv.min() - 0.02, max(allv.max(), 0.0) + 0.02, 120)
        for ek, c in zip(ENV_ENERGIES, (C1, C2, C3)):
            v = np.log10(ratios.loc[ratios.energy_keV == ek, col].to_numpy())
            ax.hist(v, bins=bins, histtype="step", lw=1.8, color=c, label=f"{ek:g} keV")
        ax.axvline(0, color=INK, lw=1.0, ls="--")
        ax.text(0, 0.98, " envelope", transform=ax.get_xaxis_transform(), fontsize=8, va="top", color=INK)
        ax.set_yscale("log")
        ax.set_xlabel(f"log₁₀ [{lab}]", fontsize=9, color=INK2)
        ax.set_ylabel("R2 compounds per bin", fontsize=9, color=INK2)
    axes[0].set_title("(a) mass attenuation relative to the elemental envelope", fontsize=10, loc="left", color=INK)
    axes[1].set_title("(b) linear attenuation relative to the elemental-solid envelope", fontsize=10, loc="left",
                      color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("A1 · Distance to the elemental envelope over the R2 map (with coherent)", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def envelope_tables(u, elig, key_cols_w, key_e, rule_masks, env, nre):
    """Per-compound ratios at the key energies (R2), top-20 non-radioactive, μ-ratio > 1 entries (R3)."""
    rho = u.density.to_numpy()[elig]
    ue = u.iloc[elig].reset_index(drop=True)
    recs = []
    for k, ek in enumerate(key_e):
        er = env[env.energy_keV == ek].iloc[0]
        nr = nre[nre.energy_keV == ek].iloc[0]
        mr = key_cols_w[:, k]
        recs.append(pd.DataFrame({"row": ue.row.values, "material_id": ue.material_id.values, "energy_keV": ek,
                                  "mu_rho": mr, "mu": mr * rho,
                                  "ratio_mu_rho": mr / er.max_mu_rho_wcoh, "ratio_mu": mr * rho / er.max_mu_wcoh,
                                  "ratio_mu_nonrad_env": mr * rho / nr.nonrad_max_mu_wcoh,
                                  "in_R2": rule_masks["R2"]}))
    allr = pd.concat(recs, ignore_index=True)
    cols = ["material_id", "formula_pretty", "density", "energy_above_hull", "theoretical", "band_gap",
            "contains_radioactive", "elements"]
    meta = ue[cols].copy()
    meta["elements"] = meta["elements"].map(lambda s: "-".join(s))
    allr = allr.merge(meta, on="material_id", how="left")
    r2 = allr[allr.in_R2].drop(columns="in_R2")
    top = []
    for ek in ENV_ENERGIES:
        sub = r2[(r2.energy_keV == ek) & ~r2.contains_radioactive]
        for col in ("ratio_mu_rho", "ratio_mu"):
            t = sub.sort_values([col, "material_id"], ascending=[False, True]).head(20).copy()
            t.insert(0, "rank", np.arange(1, len(t) + 1)); t.insert(0, "ratio", col)
            top.append(t)
    top = pd.concat(top, ignore_index=True)
    exceed = allr[(allr.ratio_mu > 1) & allr.energy_keV.isin(ENV_ENERGIES)].sort_values(
        ["energy_keV", "ratio_mu"], ascending=[True, False])
    return r2, top, exceed


# ---------------------------------------------------------------------------------------------
# A1.4 polymorphs
# ---------------------------------------------------------------------------------------------
def polymorph_groups(u, elig, key_cols, key_e, rule_masks) -> pd.DataFrame:
    ue = u.iloc[elig].reset_index(drop=True)
    ue["_i"] = np.arange(len(ue))
    cnt = ue.groupby("formula_reduced")["_i"].transform("size")
    g = ue[cnt >= 3]
    kw = key_cols["wcoh"]
    # Table C's own stored μ (key_energies.parquet, 7 energies) — an independent product ρ·μ/ρ
    ke = pd.read_parquet(paths.PROCESSED / "key_energies.parquet", columns=["row", "energy_keV", "mu_wcoh"])
    ke = ke[ke.row.isin(g.row)]
    stored = ke.pivot(index="row", columns="energy_keV", values="mu_wcoh")
    rows = []
    for f, grp in g.groupby("formula_reduced", sort=True):
        i = grp._i.to_numpy()
        rho = grp.density.to_numpy()
        st = stored.loc[grp.row.to_numpy()].to_numpy() / rho[:, None]       # stored μ ÷ ρ
        st_spread = float(((st.max(0) - st.min(0)) / st.mean(0)).max())
        eh = grp.energy_above_hull.to_numpy()
        th = grp.theoretical.to_numpy(dtype=bool)
        mr = kw[i]                                   # (n_poly × n_key)
        mrw = key_cols["wocoh"][i]
        rel = (mr.max(0) - mr.min(0)) / mr.mean(0)
        relw = (mrw.max(0) - mrw.min(0)) / mrw.mean(0)
        mu = mr * rho[:, None]
        # μ through ρ only: μ_i / (ρ_i · mean μ/ρ) − 1
        mu_dev = np.abs(mu / (rho[:, None] * mr.mean(0)) - 1).max()
        rec = {"formula_reduced": f, "n_poly": len(i), "rho_min": rho.min(), "rho_max": rho.max(),
               "rho_spread": rho.max() / rho.min() - 1, "rho_cv": rho.std() / rho.mean(),
               "max_rel_spread_mu_rho_wcoh": float(rel.max()), "max_rel_spread_mu_rho_wocoh": float(relw.max()),
               "n_key_energies_mu_rho_identical": int((rel == 0).sum()),
               "max_abs_dev_mu_from_rho_scaling": float(mu_dev),
               "max_rel_spread_stored_mu_over_rho": st_spread,
               "mu_spread_661": float(mu[:, list(key_e).index(661.657)].max() / mu[:, list(key_e).index(661.657)].min() - 1),
               "rel_spread_mu_rho_60": float(rel[list(key_e).index(60.0)]),
               "ehull_max": eh.max(), "n_icsd": int((~th).sum())}
        for tag, keep in (("eh005", eh <= 0.05), ("eh010", eh <= 0.10), ("icsd", ~th)):
            rec[f"n_{tag}"] = int(keep.sum())
            if keep.sum() >= 2:
                rr = rho[keep]
                rec[f"rho_spread_{tag}"] = rr.max() / rr.min() - 1
                rec[f"rho_cv_{tag}"] = rr.std() / rr.mean()
            else:
                rec[f"rho_spread_{tag}"] = np.nan
                rec[f"rho_cv_{tag}"] = np.nan
        # density gain available over the R2 representative
        r2 = rule_masks["R2"][i]
        rho_r2 = rho[r2][0] if r2.any() else np.nan
        rec["rho_R2"] = rho_r2
        rec["gain_max_over_R2"] = rho.max() / rho_r2 - 1
        k05 = eh <= 0.05
        rec["gain_max_eh005_over_R2"] = rho[k05].max() / rho_r2 - 1 if k05.any() else np.nan
        # density of the densest polymorph relative to the ground state, and its e_hull
        j0 = np.argmin(eh); jd = np.argmax(rho)
        rec["densest_over_lowest_ehull"] = rho[jd] / rho[j0] - 1
        rec["ehull_of_densest"] = eh[jd]
        rows.append(rec)
    return pd.DataFrame(rows)


def _dist(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {"n": int(v.size), "q05": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q90": q[4],
            "q95": q[5], "q99": q[6], "max": float(v.max()), "frac_gt_5pct": float((v > 0.05).mean()),
            "frac_gt_10pct": float((v > 0.10).mean()), "frac_gt_20pct": float((v > 0.20).mean())}


def fig_polymorph(pg: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    floor = 1e-19
    # (a) density spread distributions
    ax = axes[0]; _style(ax)
    bins = np.logspace(-4, 1, 70)
    for col, c, lab in (("rho_spread", C1, "all polymorphs"), ("rho_spread_eh005", C2, "e_hull ≤ 0.05 eV/atom"),
                        ("rho_spread_icsd", C3, "ICSD-matched only")):
        v = pg[col].dropna().to_numpy()
        ax.hist(np.clip(v, 1e-4, None), bins=bins, histtype="step", lw=1.8, color=c,
                label=f"{lab} (n = {v.size:,}; median {100*np.median(v):.1f} %)")
    ax.set_xscale("log")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_xlabel("density spread across polymorphs, ρ_max/ρ_min − 1 (first bin: ≤ 10⁻⁴)", fontsize=9, color=INK2)
    ax.set_ylabel("formulas per bin", fontsize=9, color=INK2)
    ax.set_title("(a) how much density varies across polymorphs", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    # (b) μ/ρ spread vs ρ spread: Table C μ/ρ (float32) and Table C stored μ ÷ ρ (float64)
    ax = axes[1]; _style(ax)
    y = pg.rel_spread_mu_rho_60.to_numpy()
    zero = y == 0
    ys = pg.max_rel_spread_stored_mu_over_rho.to_numpy()
    zs = ys == 0
    ax.scatter(pg.rho_spread[~zs], ys[~zs], s=6, color=C1, alpha=0.6, lw=0,
               label=f"stored μ ÷ ρ, max over 7 energies (> 0: {(~zs).sum():,} formulas)")
    ax.scatter(pg.rho_spread[zero], np.full(zero.sum(), floor), s=6, color=C2, alpha=0.6, lw=0,
               label=f"μ/ρ at 60 keV: exactly 0 for {zero.sum():,} of {len(y):,} formulas (drawn at 10⁻¹⁹)")
    ax.axhline(np.finfo(np.float32).eps, color=INK, lw=1, ls="--")
    ax.text(2e-4, np.finfo(np.float32).eps * 2, "float32 machine ε (1.2 × 10⁻⁷)", fontsize=7.5, color=INK)
    ax.axhline(np.finfo(np.float64).eps, color=MUTED, lw=1, ls=":")
    ax.text(2e-4, np.finfo(np.float64).eps * 2, "float64 machine ε (2.2 × 10⁻¹⁶)", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 10); ax.set_ylim(3e-20, 1e-2)
    ax.set_xlabel("density spread ρ_max/ρ_min − 1", fontsize=9, color=INK2)
    ax.set_ylabel("relative spread across polymorphs, (max − min)/mean", fontsize=9, color=INK2)
    ax.set_title("(b) μ/ρ is identical across polymorphs", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    # (c) μ spread vs ρ spread at 661.657 keV
    ax = axes[2]; _style(ax)
    ax.scatter(pg.rho_spread, pg.mu_spread_661, s=6, color=C1, alpha=0.6, lw=0)
    lim = (1e-5, 10)
    ax.plot(lim, lim, color=INK, lw=1, ls="--")
    ax.text(2e-4, 6e-4, "y = x", fontsize=8, color=INK, rotation=38)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("density spread ρ_max/ρ_min − 1", fontsize=9, color=INK2)
    ax.set_ylabel("μ spread at 661.657 keV, μ_max/μ_min − 1", fontsize=9, color=INK2)
    ax.set_title("(c) μ varies only through ρ", fontsize=10, loc="left", color=INK)
    fig.suptitle(f"A1 · Structure acts only through density: {len(pg):,} formulas with ≥ 3 eligible "
                 "polymorphs (R3 universe)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------
def spot_check(u, elig, key_cols, key_e, n: int = 300, seed: int = 20261012) -> dict:
    ke = pd.read_parquet(paths.PROCESSED / "key_energies.parquet",
                         columns=["row", "energy_keV", "mu_rho_wcoh", "mu_wcoh", "mu_rho_wocoh", "mu_wocoh"])
    rng = np.random.default_rng(seed)
    pick = rng.choice(elig.size, size=n, replace=False)
    rows = elig[pick]
    pos = {r: p for p, r in zip(pick, rows)}
    rho = u.density.to_numpy()
    out = {}
    for ek in sorted(set(ke.energy_keV.unique()) & set(key_e)):
        k = list(key_e).index(ek)
        s = ke[(ke.energy_keV == ek) & ke.row.isin(rows)]
        p = np.array([pos[r] for r in s.row])
        mr_w = key_cols["wcoh"][p, k]; mr_wo = key_cols["wocoh"][p, k]
        out[f"{ek:g}"] = {
            "n": int(len(s)),
            "max_rel_err_mu_rho_wcoh": float(np.max(np.abs(mr_w / s.mu_rho_wcoh.to_numpy() - 1))),
            "max_rel_err_mu_wcoh": float(np.max(np.abs(mr_w * rho[s.row] / s.mu_wcoh.to_numpy() - 1))),
            "max_rel_err_mu_rho_wocoh": float(np.max(np.abs(mr_wo / s.mu_rho_wocoh.to_numpy() - 1))),
            "max_rel_err_mu_wocoh": float(np.max(np.abs(mr_wo * rho[s.row] / s.mu_wocoh.to_numpy() - 1)))}
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _save(df: pd.DataFrame, name: str, csv: bool = True):
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    if csv:
        df.to_csv(OUT / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------
def run(verbose: bool = True) -> dict:
    paths.ensure(OUT); paths.ensure(FIG)
    u = core.universe()
    e = energies()
    key_idx = energy_index(KEY_ENERGIES, e)
    energy_index(CO60_LINES + (CS137_LINE,), e)
    env = envelope()
    nre = nonradioactive_envelope()
    env_check = {
        "max_abs_diff_mu_rho_wcoh": float(np.abs(nre.all_max_mu_rho_wcoh - env.max_mu_rho_wcoh).max()),
        "max_abs_diff_mu_wcoh": float(np.abs(nre.all_max_mu_wcoh - env.max_mu_wcoh).max()),
        "argmax_identical": bool((nre.all_argmax_mu_wcoh == env.argmax_mu_wcoh).all()
                                 and (nre.all_argmax_mu_rho_wcoh == env.argmax_mu_rho_wcoh).all())}
    if verbose:
        print("envelope recomputation:", env_check)

    mp = _main_pass(u, e, key_idx)
    elig, rule_masks, dec = mp["elig"], mp["rule_masks"], mp["dec"]
    _save(dec, "variance_decomposition")

    # crossings
    edges = pd.read_parquet(paths.INTERIM / "elements" / "edges.parquet")
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet")[["Z", "symbol"]]
    edges = edges.merge(el, on="Z")
    present = set(core.element_list(u))
    edges = edges[edges.shell.isin(["K", "L1", "L2", "L3"]) & edges.symbol.isin(present)]
    cr = []
    for (rule, coh, sub), d in dec.groupby(["rule", "coherent", "subset"]):
        d = d.sort_values("energy_keV")
        for c in find_crossings(d.energy_keV.to_numpy(),
                                (d.share_composition - d.share_density).to_numpy(), edges):
            c.update({"rule": rule, "coherent": coh, "subset": sub})
            cr.append(c)
    cross = pd.DataFrame(cr)[["subset", "rule", "coherent", "E_cross_keV", "direction", "E_lo_keV", "E_hi_keV",
                              "d_lo", "d_hi", "edge_between_nodes"]]
    cross.to_csv(OUT / "crossings.csv", index=False)

    # A1.1 quantiles + histograms (R2, wcoh)
    r2 = rule_masks["R2"]
    rho_e = mp["rho"]
    mr_key = mp["key_cols"]["wcoh"][r2]
    mu_key = mr_key * rho_e[r2][:, None]
    qt = quantile_table(mr_key, mu_key, KEY_ENERGIES, env, nre)
    _save(qt, "quantiles_key_energies")
    fig_histograms(mr_key, mu_key, KEY_ENERGIES, env, FIG / "A1_histograms.png")

    # A1.2 figure
    diag_e = diag_mean_energies()
    fig_variance_share(dec, cross, diag_e, FIG / "A1_variance_share.png")

    # A1.3 envelope distance
    env_summary = mp["env_summary"]
    env_summary.to_parquet(OUT / "envelope_distance_summary.parquet", index=False)
    ratios, top20, exceed = envelope_tables(u, elig, mp["key_cols"]["wcoh"], KEY_ENERGIES, rule_masks, env, nre)
    beat = mp["beat_nonrad_env"]
    beat.to_csv(OUT / "nonrad_above_nonrad_envelope.csv", index=False)
    slim = ratios[["row", "energy_keV", "ratio_mu_rho", "ratio_mu",
                   "ratio_mu_nonrad_env"]].copy()
    for c in ("ratio_mu_rho", "ratio_mu", "ratio_mu_nonrad_env"):
        slim[c] = slim[c].astype(np.float32)
    slim.to_parquet(OUT / "envelope_ratio_key_energies.parquet", index=False)
    top20.to_csv(OUT / "envelope_top20.csv", index=False)
    exceed.to_csv(OUT / "envelope_mu_exceed.csv", index=False)
    fig_envelope(ratios[ratios.energy_keV.isin(ENV_ENERGIES)], env_summary, FIG / "A1_envelope_distance.png")

    # A1.4 polymorphs
    pg = polymorph_groups(u, elig, mp["key_cols"], KEY_ENERGIES, rule_masks)
    pg.to_parquet(OUT / "polymorph_groups.parquet", index=False)
    fig_polymorph(pg, FIG / "A1_polymorph_density.png")

    sc = spot_check(u, elig, mp["key_cols"], KEY_ENERGIES)
    (OUT / "spot_check.json").write_text(json.dumps(sc, indent=1, default=_json_default), encoding="utf-8")

    summary = build_summary(dec, cross, qt, env, nre, env_summary, ratios, top20, exceed, pg, diag_e, sc,
                            env_check, beat)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    if verbose:
        print(json.dumps(summary["crossings_main"], indent=1, default=_json_default))
    return summary


def build_summary(dec, cross, qt, env, nre, env_summary, ratios, top20, exceed, pg, diag_e, sc, env_check, beat):
    s: dict = {"envelope_recomputation_check": env_check, "spot_check": sc,
               "diag_mean_energy_keV": {"min": float(diag_e.min()), "max": float(diag_e.max()),
                                        "per_spectrum": diag_e.to_dict()}}
    # shares at the scenario / key energies
    at = {}
    for ek in KEY_ENERGIES + CO60_LINES:
        d = dec[np.isclose(dec.energy_keV, ek)]
        at[f"{ek:g}"] = {
            f"{sub}|{rule}|{'wcoh' if coh else 'wocoh'}": {
                "share_density": r.share_density, "share_composition": r.share_composition,
                "share_cov": r.share_cov, "corr": float(r["corr"]), "n": int(r["n"])}
            for (sub, rule, coh), r in d.set_index(["subset", "rule", "coherent"]).iterrows()}
    s["shares_at_energies"] = at
    # DIAG band mean shares (R2 wcoh)
    band = dec[(dec.energy_keV >= diag_e.min()) & (dec.energy_keV <= diag_e.max()) & (dec.rule == "R2") & dec.coherent]
    s["shares_in_diag_band_R2_wcoh"] = {sub: {"share_composition_min": g.share_composition.min(),
                                              "share_composition_max": g.share_composition.max(),
                                              "share_density_min": g.share_density.min(),
                                              "share_density_max": g.share_density.max()}
                                        for sub, g in band.groupby("subset")}
    # crossings: every series' list, and the robustness range of the main (last comp->dens) crossing
    main = {}
    for sub in SUBSETS:
        c = cross[cross.subset == sub]
        per = {}
        for (rule, coh), g in c.groupby(["rule", "coherent"]):
            per[f"{rule}|{'wcoh' if coh else 'wocoh'}"] = g[["E_cross_keV", "direction", "edge_between_nodes"]].to_dict("records")
        lastc = c[c.direction == "comp->dens"].groupby(["rule", "coherent"]).E_cross_keV.max()
        main[sub] = {"all_series": per,
                     "last_comp_to_dens_by_series": {f"{r}|{'wcoh' if h else 'wocoh'}": v for (r, h), v in lastc.items()},
                     "last_comp_to_dens_range_keV": [float(lastc.min()), float(lastc.max())] if len(lastc) else None}
    s["crossings_main"] = main
    # sign of the composition − density difference at high energy (does density stay ahead?)
    hi = dec[(dec.rule == "R2") & dec.coherent & (dec.energy_keV >= 1000)]
    s["max_share_composition_above_1MeV_R2_wcoh"] = {sub: float(g.share_composition.max())
                                                    for sub, g in hi.groupby("subset")}
    s["quantiles_key_energies"] = qt.to_dict("records")
    # envelope distance
    ed = {}
    for ek in ENV_ENERGIES:
        r = ratios[ratios.energy_keV == ek]
        er = env[env.energy_keV == ek].iloc[0]
        nr = nre[nre.energy_keV == ek].iloc[0]
        ed[f"{ek:g}"] = {
            "envelope": {"max_mu_rho": er.max_mu_rho_wcoh, "argmax_mu_rho": er.argmax_mu_rho_wcoh,
                         "max_mu": er.max_mu_wcoh, "argmax_mu": er.argmax_mu_wcoh,
                         "nonrad_max_mu_rho": nr.nonrad_max_mu_rho_wcoh, "nonrad_argmax_mu_rho": nr.nonrad_argmax_mu_rho_wcoh,
                         "nonrad_max_mu": nr.nonrad_max_mu_wcoh, "nonrad_argmax_mu": nr.nonrad_argmax_mu_wcoh},
            "ratio_mu_rho": _dist(r.ratio_mu_rho), "ratio_mu": _dist(r.ratio_mu),
            "n_R2_mu_ratio_gt1": int((r.ratio_mu > 1).sum()),
            "n_R2_mu_ratio_gt1_nonrad": int(((r.ratio_mu > 1) & ~r.contains_radioactive).sum()),
            "n_R3_mu_ratio_gt1": int((exceed.energy_keV == ek).sum()),
            "n_R3_mu_ratio_gt1_nonrad": int(((exceed.energy_keV == ek) & ~exceed.contains_radioactive).sum()),
            "top5_mu_rho_nonrad": top20[(top20.energy_keV == ek) & (top20.ratio == "ratio_mu_rho")].head(5)[
                ["material_id", "formula_pretty", "ratio_mu_rho", "density", "energy_above_hull"]].to_dict("records"),
            "top5_mu_nonrad": top20[(top20.energy_keV == ek) & (top20.ratio == "ratio_mu")].head(5)[
                ["material_id", "formula_pretty", "ratio_mu", "density", "energy_above_hull"]].to_dict("records")}
    s["envelope_distance"] = ed
    s["n_mu_ratio_gt1_any_energy"] = {
        rule: {"max_over_energies": int(g.n_mu_ratio_gt1.max()), "min_over_energies": int(g.n_mu_ratio_gt1.min()),
               "nonrad_max": int(g.n_mu_ratio_gt1_nonrad.max())}
        for rule, g in env_summary.groupby("rule")}
    es2 = env_summary[env_summary.rule == "R2"]
    pos = es2[es2.n_nonrad_gt_nonrad_env > 0]
    b2 = beat[beat.in_R2]
    s["nonrad_above_nonrad_envelope"] = {
        "energies_with_any_R2_keV": [float(pos.energy_keV.min()), float(pos.energy_keV.max())] if len(pos) else None,
        "n_energies_with_any_R2": int(len(pos)),
        "energies_list_R2": pos.energy_keV.tolist(),
        "max_count_R2": int(es2.n_nonrad_gt_nonrad_env.max()),
        "energy_of_max_count_R2": float(es2.loc[es2.n_nonrad_gt_nonrad_env.idxmax(), "energy_keV"]),
        "max_count_B4": int(es2.n_B4_gt_nonrad_env.max()),
        "n_distinct_R2_compounds": int(len(b2)), "n_distinct_R2_B4_compounds": int(b2.in_B4.sum()),
        "n_distinct_R3_entries": int(len(beat)),
        "top_R2": b2.head(15).to_dict("records"),
        "top_R2_B4": b2[b2.in_B4].head(10).to_dict("records"),
        "counts_at_key_energies_R2": {f"{ek:g}": int(es2.loc[np.isclose(es2.energy_keV, ek), "n_nonrad_gt_nonrad_env"].iloc[0])
                                      for ek in KEY_ENERGIES},
    }
    # polymorphs
    s["polymorphs"] = {
        "n_formulas_ge3": int(len(pg)), "n_entries": int(pg.n_poly.sum()),
        "n_poly_distribution": pg.n_poly.describe().to_dict(),
        "max_rel_spread_mu_rho_wcoh_all": float(pg.max_rel_spread_mu_rho_wcoh.max()),
        "max_rel_spread_mu_rho_wocoh_all": float(pg.max_rel_spread_mu_rho_wocoh.max()),
        "frac_formulas_mu_rho_exactly_identical_all_key": float((pg.n_key_energies_mu_rho_identical == len(KEY_ENERGIES)).mean()),
        "max_abs_dev_mu_from_rho_scaling": float(pg.max_abs_dev_mu_from_rho_scaling.max()),
        "max_rel_spread_stored_mu_over_rho_7_energies": float(pg.max_rel_spread_stored_mu_over_rho.max()),
        "frac_stored_mu_over_rho_spread_exactly_0": float((pg.max_rel_spread_stored_mu_over_rho == 0).mean()),
        "rho_spread_all": _dist(pg.rho_spread), "rho_cv_all": _dist(pg.rho_cv),
        "rho_spread_eh005": _dist(pg.rho_spread_eh005), "rho_cv_eh005": _dist(pg.rho_cv_eh005),
        "rho_spread_eh010": _dist(pg.rho_spread_eh010),
        "rho_spread_icsd": _dist(pg.rho_spread_icsd), "rho_cv_icsd": _dist(pg.rho_cv_icsd),
        "gain_max_over_R2": _dist(pg.gain_max_over_R2), "gain_max_eh005_over_R2": _dist(pg.gain_max_eh005_over_R2),
        "densest_over_lowest_ehull": _dist(pg.densest_over_lowest_ehull),
        "ehull_of_densest": _dist(pg.ehull_of_densest),
        "frac_densest_is_lowest_ehull": float((pg.densest_over_lowest_ehull == 0).mean()),
        "spearman_rho_spread_vs_ehull_max": float(pg[["rho_spread", "ehull_max"]].corr("spearman").iloc[0, 1]),
        "largest_spreads": pg.sort_values("rho_spread", ascending=False).head(10)[
            ["formula_reduced", "n_poly", "rho_min", "rho_max", "rho_spread", "ehull_max", "rho_spread_eh005"]].to_dict("records"),
        "largest_spreads_eh005": pg.sort_values("rho_spread_eh005", ascending=False).head(10)[
            ["formula_reduced", "n_poly", "n_eh005", "rho_spread_eh005", "rho_min", "rho_max"]].to_dict("records"),
    }
    return s


if __name__ == "__main__":
    run()
