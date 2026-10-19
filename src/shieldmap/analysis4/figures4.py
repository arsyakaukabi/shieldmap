"""Figure patches (analysis quality; journal formatting is in stage 4b).

Regenerate with ``python -m shieldmap.analysis4.figures4`` (or ``analyze4a --only figures4``). Every figure is
written to ``reports/phase4a/figures/`` as PNG (300 dpi) and SVG (no date, fixed id salt "shieldmap-phase4a",
so a rerun reproduces the SVG byte for byte). stage 3 code and figures are read only: the stage 3 figure
functions are copied here and adapted, and their inputs are read from the saved stage 3 data files.

Figures and the open items they patch
  fig1_variance_physics      stage 3 Fig. 1 panels a-f; the 235 keV crossing line is labelled
  fig1s_z_exponents          stage 3 Fig. 1 panels g-h (textbook Z-exponents), moved to a supplementary figure
  fig2_price_curves          re-based on the stage 4a primaries: B4-internal marginal prices, best_clean,
                             phase4a artefact flag, rule R2, with coherent; thickness at the measured-density
                             point and at MP densities, areal mass alongside; credible bands with one legend
                             swatch per shaded colour
  fig2s_price_curves_headroom  the same against F0 ("headroom", appendix reference set)
  fig3_gap_inversion         stage 3 Fig. 3 with the panel (c) legend moved outside the plotting area
  fig4_load_bearing          stage 3 Fig. 4 (narrow beam) with the fluorescence-corrected cells (E1) marked
  fig4s_load_bearing_b4internal  element-exclusion filter prices on the stage 4a primaries (B4-internal)
  fig5_diag_plateau          stage 3 Fig. 5 with legible variant tick labels
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.analysis import core
from shieldmap.analysis3 import common as c3
from shieldmap.analysis4 import common

OUT = common.FIGURES4
P3 = c3.P3
S4A = common.P4 / "S"
SVG_HASHSALT = "shieldmap-phase4a"

# palette and helpers as in stage 3 (landscape3 / continuum / families)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"thickness": "#2a78d6", "areal": "#eb6834"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASIS_LABEL = {"thickness": "thickness", "areal": "areal mass"}
LINES = {"Cs-137": 661.657, "Co-60": 1250.0}
E_LO, E_HI = 20.0, 10000.0
XT = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
XTL = ["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k"]
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def style():
    plt = c3.style()
    plt.rcParams["svg.hashsalt"] = SVG_HASHSALT
    return plt


def savefig(fig, name: str) -> list[str]:
    """Write reports/phase4a/figures/<name>.png (300 dpi) and .svg; returns paths relative to shieldmap/.

    Same settings as stage 3's ``analysis3.common.savefig`` (no SVG date, fixed id salt), stage 4a folder."""
    import matplotlib
    paths.ensure(OUT)
    out = []
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASHSALT}):
        for ext in ("png", "svg"):
            p = OUT / f"{name}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
            out.append(str(p.relative_to(paths.ROOT)).replace("\\", "/"))
    return out


def _fmt(f: str) -> str:
    return f.translate(_SUB)


def _ax_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _recess(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="major", color="#e4e3df", lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, color="#8a8984")


def _logx(ax, lo=E_LO, hi=E_HI):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(XT)
    ax.set_xticklabels(XTL)
    ax.minorticks_off()


# =============================================================================================
# Fig. 1: variance physics (a-f) and the supplementary Z-exponents (g-h)
# =============================================================================================
def fig1_inputs() -> dict:
    """The inputs of stage 3 landscape3.run_item1, read from data/phase3/L (no recomputation)."""
    from shieldmap.analysis import landscape as p2
    L = P3 / "L"
    return {"dec": pd.read_parquet(L / "variance_decomposition.parquet"),
            "cross": pd.read_csv(L / "crossings.csv"),
            "sh": pd.read_parquet(L / "partial_shares_R2.parquet"),
            "zx": pd.read_parquet(L / "z_exponents.parquet"),
            "diag_e": p2.diag_mean_energies()}


def _e_cross(cross: pd.DataFrame) -> float:
    xc = cross[(cross.subset == "all") & (cross.rule == "R2") & cross.coherent & (cross.direction == "comp->dens")]
    return float(xc.E_cross_keV.iloc[-1])


def _label_cross(ax, e_cross: float, y: float = 0.98, ha: str = "right"):
    """Label the dotted crossing line (open item: the 235 keV line in stage 3 Fig. 1 e-h had no label)."""
    ax.axvline(e_cross, color=MUTED, lw=0.6, ls=":")
    ax.annotate(f"{e_cross:.0f} keV", xy=(e_cross, y), xycoords=("data", "axes fraction"),
                xytext=(-1.5 if ha == "right" else 1.5, 0), textcoords="offset points", ha=ha, va="top",
                fontsize=6, color=INK2, rotation=90)


def fig1_variance_physics(dec, cross, sh, diag_e) -> list[str]:
    """Stage 3 Fig. 1 (landscape3.fig_variance_physics) panels a-f; g-h moved to fig1s_z_exponents."""
    from shieldmap.analysis import landscape as p2
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(3, 2, figsize=(c3.DOUBLE_COL_IN, 7.4), sharex=True)
    d_all = dec.assign(ratio=dec.var_logmurho / dec.var_logrho)
    e_cross = _e_cross(cross)
    CR = "#6250d6"
    titles = {"all": "(a) R2 map: shares of var(log μ)", "top10_rho": "(b) densest 10 %: shares",
              "top10_mu": "(c) best 10 % by μ: var(log μ/ρ)/var(log ρ)",
              "top1_mu": "(d) best 1 % by μ: var(log μ/ρ)/var(log ρ)"}
    for ax, sub in zip(axes.flat[:4], p2.SUBSETS):
        _ax_style(ax)
        d = d_all[d_all.subset == sub]
        prim = d[(d.rule == "R2") & d.coherent].sort_values("energy_keV")
        e = prim.energy_keV.to_numpy()
        ax.axvspan(diag_e.min(), diag_e.max(), color="#d9d8d3", alpha=0.5, lw=0)
        cr = cross[(cross.subset == sub) & (cross.rule == "R2") & cross.coherent]
        if sub in ("all", "top10_rho"):
            for col, c in (("share_density", C1), ("share_composition", C2), ("share_cov", C3)):
                piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values=col)
                ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=c, alpha=0.2, lw=0)
                ax.plot(e, prim[col], color=c, lw=1.2)
            if sub == "all":
                t = sh.sort_values("energy_keV")
                ax.fill_between(t.energy_keV, t.share_pe_q25, t.share_pe_q75, color=INK, alpha=0.10, lw=0)
                ax.plot(t.energy_keV, t.share_pe_q50, color=INK, lw=1.0, ls="--")
            ycr = lambda ec, p=prim: np.interp(np.log10(ec), np.log10(p.energy_keV), p.share_density)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_ylim(-0.1 if sub == "all" else -0.2, 1.05)
            ax.set_ylabel("share")
        else:
            piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values="ratio")
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=CR, alpha=0.2, lw=0)
            ax.plot(e, prim.ratio, color=CR, lw=1.2)
            ax.axhline(1, color=INK, lw=0.8, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(min(0.03, float(piv.min().min()) * 0.8), 5)
            ycr = lambda ec: 1.0
            ax.set_ylabel("variance ratio")
        for _, r in cr.iterrows():
            ax.plot(r.E_cross_keV, ycr(r.E_cross_keV), "o", ms=4, mfc="white", mec=INK, mew=0.9, zorder=5)
        main = cr[cr.direction == "comp->dens"]
        lab = cr if sub in ("top10_mu", "top1_mu") else main.tail(1)
        for k, (_, r) in enumerate(lab.iterrows()):
            ec = r.E_cross_keV
            ax.annotate(f"{ec:.0f}", (ec, ycr(ec)), xytext=(3, 5 if k % 2 == 0 else -9), textcoords="offset points",
                        fontsize=6, color=INK)
        ax.set_xscale("log")
        ax.set_title(titles[sub], loc="left")
    # (e) variances
    ax = axes[2, 0]; _ax_style(ax)
    t = sh.sort_values("energy_keV")
    ax.plot(t.energy_keV, t.var_logmurho, color=C2, lw=1.2, label="var(log μ/ρ)")
    ax.plot(t.energy_keV, t.var_logmurho_no_pe, color=C2, lw=1.0, ls=":", label="… without photoelectric")
    ax.plot(t.energy_keV, t.var_pe_term, color=INK, lw=0.9, ls="--", label="var(−log(1 − f_pe)): pe term")
    ax.axhline(float(t.var_logrho.iloc[0]), color=C1, lw=1.2, label="var(log ρ)")
    _label_cross(ax, e_cross, y=0.98, ha="right")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_ylim(5e-5, 4.0)
    ax.set_ylabel("variance (log₁₀ units²)")
    ax.set_title("(e) composition spread falls to the density spread", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    # (f) partial shares
    ax = axes[2, 1]; _ax_style(ax)
    for k, c, lab in (("pe", C2, "photoelectric"), ("incoh", C1, "Compton"), ("coh", C3, "coherent"),
                      ("pair", C4, "pair")):
        ax.fill_between(t.energy_keV, t[f"share_{k}_q25"], t[f"share_{k}_q75"], color=c, alpha=0.2, lw=0)
        ax.plot(t.energy_keV, t[f"share_{k}_q50"], color=c, lw=1.2, label=lab)
    _label_cross(ax, e_cross, y=0.98, ha="left")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of μ/ρ (R2 median, IQR)")
    ax.set_title("(f) interaction shares over the R2 map", loc="left")
    ax.legend(frameon=False, loc="center left", fontsize=6)
    for ax in axes[2]:
        ax.set_xlabel("photon energy (keV)")
    handles = [Line2D([], [], color=C1, lw=1.2, label="density share var(log ρ)/var(log μ)"),
               Line2D([], [], color=C2, lw=1.2, label="composition share var(log μ/ρ)/var(log μ)"),
               Line2D([], [], color=C3, lw=1.2, label="covariance share 2cov/var(log μ) (can be < 0)"),
               Line2D([], [], color=CR, lw=1.2, label="(c, d) variance ratio var(log μ/ρ)/var(log ρ)"),
               Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK, label="crossing (keV)"),
               Line2D([], [], color=INK, lw=1.0, ls="--", label="(a) photoelectric share of μ/ρ, R2 median (band: IQR)"),
               Patch(color="#d9d8d3", alpha=0.5, lw=0, label="energy range of the diagnostic tube spectra"),
               Line2D([], [], color=MUTED, lw=0.8, ls=":",
                      label=f"(e, f) {e_cross:.0f} keV: last crossing in (a), composition share = density share")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=6.3)
    fig.text(0.0, 0.0, "Shares sum to 1; where the covariance share is negative a single share can exceed 1. "
             "Bands: rules R1/R2/R3 × coherent on/off. Population: eligible Materials Project entries (see the "
             "reweighting test, data/phase3/RW).\nThe Z-exponents (Phase 3 Fig. 1 g–h, textbook physics) are in "
             "the supplementary figure fig1s_z_exponents.", fontsize=6, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=0.6, w_pad=0.8)
    out = savefig(fig, "fig1_variance_physics")
    plt.close(fig)
    return out


def fig1s_z_exponents(zx, cross) -> list[str]:
    """Supplementary: stage 3 Fig. 1 g-h (per-atom and per-mass Z-exponents), with the crossing labelled."""
    from shieldmap.analysis3.landscape3 import Z_RANGE
    plt = style()
    e_cross = _e_cross(cross)
    fig, axes = plt.subplots(1, 2, figsize=(c3.DOUBLE_COL_IN, 2.9), sharex=True)
    for ax, kind, ttl in ((axes[0], "atom", "(a) per-atom Z-exponent n: σ ∝ Zⁿ"),
                          (axes[1], "mass", "(b) per-mass Z-exponent n′: μ/ρ ∝ Zⁿ′")):
        _ax_style(ax)
        z = zx.sort_values("energy_keV")
        ax.plot(z.energy_keV, z[f"n_{kind}_pe_Kactive"], color=C2, lw=1.2, label="photoelectric (K-active)")
        ax.plot(z.energy_keV, z[f"n_{kind}_pe"], color=C2, lw=0.8, ls=":", label="photoelectric (all Z)")
        ax.plot(z.energy_keV, z[f"n_{kind}_incoh"], color=C1, lw=1.2, label="Compton")
        ax.plot(z.energy_keV, z[f"n_{kind}_coh"], color=C3, lw=1.0, label="coherent")
        pz = z[z.energy_keV >= 2000]
        ax.plot(pz.energy_keV, pz[f"n_{kind}_pair"], color=C4, lw=1.0, label="pair")
        _label_cross(ax, e_cross, y=0.98, ha="right")       # 235 keV: the Fig. 1 (a) crossing
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_xscale("log")
        ax.set_ylabel("exponent")
        ax.set_title(ttl + f" (Z {Z_RANGE[0]}–{Z_RANGE[1]})", loc="left")
        ax.set_xlabel("photon energy (keV)")
    axes[0].set_ylim(-0.3, 6.4); axes[1].set_ylim(-0.8, 4.9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)
    out = savefig(fig, "fig1s_z_exponents")
    plt.close(fig)
    return out


# =============================================================================================
# Fig. 2: price curves on the stage 4a primaries
# =============================================================================================
FIG2_PANELS = [("RAD|B4", "RAD (non-radioactive)"), ("ABUND(1)|B4", "ABUND(1 ppm), scarcity proxy"),
               ("GAP(3.1)|B4", "GAP(3.1 eV)"), ("PB|B4", "PB (lead-free)"), ("STAB|B4", "STAB(0.05 eV/atom)"),
               ("EXP|B4", "EXP (ICSD-matched)")]
FIG2_GROUPS = [("RAD|B4", "ABUND(1)|B4"), ("GAP(3.1)|B4",), ("PB|B4", "STAB|B4", "EXP|B4")]
FIG2S_PANELS = [("PB|F0", "PB: lead-free"), ("STAB(0.05)|F0", "STAB(0.05 eV/atom)"), ("EXP|F0", "EXP: ICSD-matched"),
                ("RAD|F0", "RAD (= B1): non-radioactive"), ("ABUND(1)|F0", "ABUND(1 ppm), scarcity proxy"),
                ("B4|F0", "B4 = RAD + PB + STAB + EXP"), ("B6|F0", "B6 = B4 + ABUND(1)"),
                ("GAP(3.1)|F0", "GAP(3.1 eV)"), ("B5|F0", "B5 = B4 + GAP(3.1)")]
FIG2S_GROUPS = [("PB|F0", "STAB(0.05)|F0", "EXP|F0"), ("RAD|F0", "ABUND(1)|F0", "B4|F0", "B6|F0"),
                ("GAP(3.1)|F0", "B5|F0")]
# (basis, density) → line style; the band is drawn for the primary density of each basis
CURVES = (("thickness", "measured", "-", 1.15, "thickness, measured-density point (primary)"),
          ("thickness", "mp", (0, (3, 1.6)), 0.9, "thickness, MP densities"),
          ("areal", "any", "-", 1.15, "areal mass (density-independent)"))
BAND_ALPHA = {"thickness": 0.14, "areal": 0.24}


def fig2_inputs(reference: str = "B4-internal", artefact_def: str = common.PRIMARY_ARTEFACT) -> pd.DataFrame:
    cols = ["label", "basis", "density", "energy_keV", "price", "band_lo", "band_hi", "reference", "artefact_def",
            "scenario"]
    b = pd.read_parquet(S4A / "bands.parquet", columns=cols)
    b = b[(b.scenario == "MONO") & (b.reference == reference) & (b.artefact_def == artefact_def)]
    for c in ("label", "basis", "density"):
        b[c] = b[c].astype(str)
    return b.drop(columns=["scenario", "reference", "artefact_def"]).reset_index(drop=True)


def identical_panels(b: pd.DataFrame, labels) -> list[tuple[str, str]]:
    """Pairs of labels whose lines AND bands agree on every basis × density × energy (to be merged)."""
    out = []
    key = ["basis", "density", "energy_keV"]
    g = {lab: b[b.label == lab].sort_values(key).reset_index(drop=True) for lab in labels}
    for i, a in enumerate(labels):
        for c in labels[i + 1:]:
            x, y = g[a], g[c]
            if len(x) == len(y) and (x[key].to_numpy() == y[key].to_numpy()).all() and all(
                    np.allclose(x[v].to_numpy(float), y[v].to_numpy(float), atol=1e-12, equal_nan=True)
                    for v in ("price", "band_lo", "band_hi")):
                out.append((a, c))
    return out


def _fig2(b: pd.DataFrame, panels, groups, name: str, ncols: int, title: str, note: str) -> list[str]:
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    nrows = math.ceil(len(panels) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(c3.DOUBLE_COL_IN, 2.15 * nrows + 1.25), sharex=True,
                            squeeze=False, layout="constrained")
    ylim = {}
    for grp in groups:
        h = b[b.label.isin(grp)]
        top = float(100 * np.nanmax(np.r_[h.band_hi.to_numpy(float), h.price.to_numpy(float)]))
        top = max(top, 1.0)
        for c in grp:
            ylim[c] = (-0.03 * top, 1.05 * top)
    for ax, (lab, ttl) in zip(axs.flat, panels):
        _recess(ax)
        for E in LINES.values():
            ax.axvline(E, color="#b9b8b2", lw=0.5, ls=":", zorder=0)
        for basis, den, ls, lw, _ in CURVES:
            h = b[(b.label == lab) & (b.basis == basis) & (b.density == den)].sort_values("energy_keV")
            if den in ("measured", "any"):
                ax.fill_between(h.energy_keV, 100 * h.band_lo, 100 * h.band_hi, color=COL[basis],
                                alpha=BAND_ALPHA[basis], lw=0)
            ax.plot(h.energy_keV, 100 * h.price, color=COL[basis], lw=lw, ls=ls)
        ax.set_title(ttl, loc="left", fontsize=7.2)
        ax.axhline(0, color="#52514e", lw=0.5)
        ax.set_ylim(*ylim[lab])
        _logx(ax)
        for nm, E in LINES.items():
            ax.text(E * 0.95, 0.97, nm, transform=ax.get_xaxis_transform(), fontsize=6, color="#8a8984",
                    ha="right", va="top", rotation=90)
    for ax in axs.flat[len(panels):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("price (%)")
    for ax in axs[-1]:
        ax.set_xlabel("photon energy (keV)")
        ax.xaxis.set_tick_params(labelbottom=True)
    hs = [Line2D([], [], color=COL[bs], lw=lw, ls=ls, label=lab) for bs, _, ls, lw, lab in CURVES]
    hs += [Patch(facecolor=COL["thickness"], alpha=BAND_ALPHA["thickness"], lw=0,
                 label="credible band, thickness (measured-density point)"),
           Patch(facecolor=COL["areal"], alpha=BAND_ALPHA["areal"], lw=0, label="credible band, areal mass")]
    leg = fig.legend(handles=hs, loc="outside upper left", ncol=3, frameon=False, fontsize=6.3,
                     handlelength=2.2, columnspacing=1.2, title=title, title_fontsize=7.5)
    leg._legend_box.align = "left"                     # title flush left above the entries
    fig.supxlabel(note, x=0.0, ha="left", fontsize=6, color=INK2)   # inside the constrained layout
    out = savefig(fig, name)
    plt.close(fig)
    return out


def fig2_price_curves(b: pd.DataFrame) -> tuple[list[str], list]:
    labs = [p for p, _ in FIG2_PANELS]
    same = identical_panels(b, labs)
    panels = [p for p in FIG2_PANELS if p[0] not in {c for _, c in same}]   # merge identical panels (none expected)
    for a, c in same:
        panels = [(p, t + f" (= {c})" if p == a else t) for p, t in panels]
    note = ("Price of X inside the practical set B4 (= RAD ∧ PB ∧ STAB(0.05) ∧ EXP): q*(B4 ∧ X)/q*(B4) − 1 for GAP and "
            "ABUND; q*(B4)/q*(B4 without X) − 1 for RAD, PB, STAB, EXP. Line: best non-artefact compound (Phase 4a "
            "artefact flag),\nrule R2, with coherent, narrow beam. Band: rules R1/R2/R3 × kinds {best_clean, "
            "best_icsd, median of top 50 clean} × coherent on/off, at the same density source. PB and STAB are 0 on "
            "the thickness basis at\nevery energy (only their bands differ); no two panels are identical. "
            "Data: data/phase4a/S/bands.parquet.")
    out = _fig2(b, panels, FIG2_GROUPS, "fig2_price_curves", 3,
                "Marginal prices inside the practical set B4 (Phase 4a primary reference set)", note)
    return out, same


def fig2s_price_curves_headroom(b: pd.DataFrame) -> list[str]:
    note = ("Headroom (appendix reference set): P = q*(F0 ∧ K)/q*(F0) − 1 against the unconstrained artefact-free "
            "frontier F0 (Phase 4a artefact flag). Line: best_clean, rule R2, with coherent, narrow beam.\n"
            "Band as in Fig. 2. B1 = RAD; B2 = B1 + PB and B3 = B2 + STAB are not shown. "
            "Data: data/phase4a/S/bands.parquet (reference = headroom).")
    return _fig2(b, FIG2S_PANELS, FIG2S_GROUPS, "fig2s_price_curves_headroom", 3,
                 "Headroom prices against F0 (supplementary; not the headline reference set)", note)


# =============================================================================================
# Fig. 3: GAP inversion (legend of panel c outside the plotting area)
# =============================================================================================
def fig3_gap_inversion(gi: pd.DataFrame) -> list[str]:
    """Stage 3 continuum.fig_gap_inversion; panel (c) legend moved below the axes, no y headroom needed."""
    plt = style()
    g = gi[gi.constraint == "GAP(3.1)"].sort_values("energy_keV")
    fig, axs = plt.subplots(2, 2, figsize=(c3.DOUBLE_COL_IN, 5.9), sharex=True, layout="constrained")
    ax = axs[0, 0]
    _recess(ax)
    for basis, col, lab in (("thickness", "P_thickness", "thickness"), ("areal", "P_mass", "areal mass")):
        ax.fill_between(g.energy_keV, 100 * g[f"{col}_band_lo"], 100 * g[f"{col}_band_hi"], color=COL[basis],
                        alpha=0.16, lw=0)
        ax.plot(g.energy_keV, 100 * g[f"{col}_headline"], color=COL[basis], lw=1.1, label=f"{lab}, GAP(3.1)")
        for other, ls in (("GAP(2.5)", (0, (1, 1))), ("GAP(3.5)", (0, (4, 1.5)))):
            o = gi[gi.constraint == other].sort_values("energy_keV")
            ax.plot(o.energy_keV, 100 * o[f"{col}_headline"], color=COL[basis], lw=0.5, ls=ls)
    ax.set_ylabel("price (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_title("a  GAP price (thin: g = 2.5 dotted, 3.5 dashed)", loc="left", fontsize=7.5)
    ax = axs[0, 1]
    _recess(ax)
    ax.plot(g.energy_keV, g.inversion_clean, color="#52514e", lw=1.1, label="best non-artefact compound")
    ax.plot(g.energy_keV, g.inversion_median50_clean, color="#52514e", lw=0.6, ls="--",
            label="median of top 50 (artefact-free)")
    ax.axhline(1, color="#8a8984", lw=0.5)
    ax.set_ylabel("(1 + P_thickness) / (1 + P_mass)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("b  inversion factor (same kind on both bases)", loc="left", fontsize=7.5)
    ax = axs[1, 0]
    _recess(ax)
    for sname, col, lab in (("F0_clean", "#52514e", "F0"), ("GAP(3.1)_clean", CAT[2], "F0 ∩ GAP(3.1)")):
        ax.fill_between(g.energy_keV, g[f"rho_top50_p10_{sname}_thk"], g[f"rho_top50_p90_{sname}_thk"], color=col,
                        alpha=0.15, lw=0)
        ax.plot(g.energy_keV, g[f"rho_top50_median_{sname}_thk"], color=col, lw=1.0, label=f"{lab}: top-50 median")
    ax.plot(g.energy_keV, g.rho_thk_F0_clean, color="#52514e", lw=0.6, ls="--", label="F0: best compound")
    ax.plot(g.energy_keV, g.rho_thk_K_clean, color=CAT[2], lw=0.6, ls="--",
            label="GAP(3.1): best compound (ThO₂ at high E: radioactive)")
    ax.plot(g.energy_keV, g.rho_best_F0_thk, color="#b9b8b2", lw=0.6, ls=":",
            label="F0 best incl. artefact entries")
    ax.set_ylabel("density (g/cm³)")
    top = float(np.nanmax(g[["rho_top50_p90_F0_clean_thk", "rho_top50_p90_GAP(3.1)_clean_thk", "rho_thk_F0_clean",
                             "rho_thk_K_clean", "rho_best_F0_thk"]].to_numpy(float)))
    ax.set_ylim(0, 1.05 * top)
    hc, lc_ = ax.get_legend_handles_labels()
    fig.legend(hc, lc_, loc="outside lower center", ncol=3, frameon=False, fontsize=6.3, handlelength=1.8,
               columnspacing=1.2, title="panel c", title_fontsize=6.3)
    ax.set_title("c  thickness-frontier densities, artefact entries removed\n    (shade: top-50 p10–p90)",
                 loc="left", fontsize=7.5)
    ax.set_xlabel("photon energy (keV)")
    ax = axs[1, 1]
    _recess(ax)
    ax.plot(g.energy_keV, g.inversion_clean, color="#52514e", lw=1.1, label="inversion factor")
    ax.plot(g.energy_keV, g.density_factor_clean, color=CAT[2], lw=0.9, label="density factor ρ(c₀)/ρ(c_K)")
    ax.plot(g.energy_keV, g.mismatch_factor_clean, color=COL["areal"], lw=0.9, label="frontier-mismatch factor")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3, 4])
    ax.set_yticklabels(["0.5", "0.7", "1", "1.5", "2", "3", "4"])
    ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.axhline(1, color="#8a8984", lw=0.5)
    ax.set_title("d  accounting identity: inversion = density × mismatch", loc="left", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_xlabel("photon energy (keV)")
    for a in axs.flat:
        _logx(a)
        a.xaxis.set_tick_params(labelbottom=True)
    fig.suptitle("GAP(3.1) price against F0 (Phase 3 definitions: headroom reference, Phase 3 artefact flag, "
                 "MP densities); the B4-internal GAP price is in Fig. 2", x=0.0, ha="left", fontsize=7)
    out = savefig(fig, "fig3_gap_inversion")
    plt.close(fig)
    return out


# =============================================================================================
# Fig. 4: load-bearing elements, narrow beam, with the fluorescence-corrected cells marked
# =============================================================================================
def fig4_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    eh = pd.read_parquet(P3 / "P" / "load_bearing_headline.parquet")
    lc = pd.read_csv(P3 / "E" / "load_bearing_corrected.csv")
    return eh, lc


def corrected_cells(eh: pd.DataFrame, lc: pd.DataFrame, model: str = "E1", kind: str = "best_clean") -> pd.DataFrame:
    """Per element × baseline: the corrected cells (thickness basis), their energies, the corrected price range
    over those energies and T = 0.1/0.01, and the check that the file's narrow-beam value equals the heat map."""
    nb = lc[(lc.model == "narrow") & (lc.kind == kind)]
    hm = eh[eh.basis == "thickness"].set_index(["element", "baseline", "energy_keV"]).price_headline
    diff = np.abs(nb.price.to_numpy() - hm.reindex(list(zip(nb.element, nb.baseline, nb.energy_keV))).to_numpy())
    cc = lc[(lc.model == model) & (lc.kind == kind)]
    g = cc.groupby(["element", "baseline"])
    out = g.agg(energies=("energy_keV", lambda s: sorted(set(map(float, s)))),
                n_energies=("energy_keV", "nunique")).reset_index()
    # as dossier R2.7: maximum over the sampled energies at each T, range over T = 0.1 / 0.01
    tmax = cc.groupby(["element", "baseline", "T"]).price.max().groupby(["element", "baseline"])
    out = out.merge(tmax.min().rename("corr_min").reset_index(), on=["element", "baseline"]) \
             .merge(tmax.max().rename("corr_max").reset_index(), on=["element", "baseline"])
    nbg = nb.groupby(["element", "baseline"]).price.agg(["min", "max"]).rename(
        columns={"min": "narrow_min", "max": "narrow_max"}).reset_index()
    out = out.merge(nbg, on=["element", "baseline"])
    out.attrs["max_abs_diff_narrow_vs_heatmap"] = float(np.nanmax(diff))
    return out


def fig4_load_bearing(eh: pd.DataFrame, cells: pd.DataFrame, thresh: float = 0.02) -> tuple[list[str], list[str]]:
    """Stage 3 continuum.fig_load_bearing (narrow beam) + the E1-corrected cells marked on the thickness panels."""
    plt = style()
    import matplotlib
    from matplotlib.colors import PowerNorm
    from matplotlib.lines import Line2D
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("white")
    et = c3.elements_table()
    zmap = dict(zip(et["symbol"], et["Z"]))
    mx = eh.groupby("element").price_headline.max()
    els = sorted(mx.index[mx >= thresh].tolist(), key=lambda s: zmap.get(s, 999))
    E = np.sort(eh.energy_keV.unique())
    edges = np.concatenate([[E[0]], np.sqrt(E[1:] * E[:-1]), [E[-1]]])
    vmax = 100 * float(eh.price_headline.max())
    fig, axs = plt.subplots(2, 2, figsize=(c3.DOUBLE_COL_IN, 0.12 * len(els) * 2 + 1.7), sharex=True,
                            sharey=True, layout="constrained")
    im = None
    shown = []
    for i, basis in enumerate(core.BASES):
        for j, bl in enumerate(("F0", "F0&B4")):
            ax = axs[i, j]
            z = (eh[(eh.basis == basis) & (eh.baseline == bl)]
                 .pivot(index="element", columns="energy_keV", values="price_headline").reindex(index=els, columns=E))
            zz = 100 * z.to_numpy()
            zz = np.ma.masked_where(~(zz > 0), zz)
            im = ax.pcolormesh(edges, np.arange(len(els) + 1), zz, cmap=cmap,
                               norm=PowerNorm(0.5, vmin=0, vmax=vmax), shading="flat", rasterized=True)
            _logx(ax)
            ax.set_yticks(np.arange(len(els)) + 0.5)
            ax.set_yticklabels(els, fontsize=6)
            ax.set_ylim(len(els), 0)
            ax.set_title(f"{BASIS_LABEL[basis]}, against {'F0' if bl == 'F0' else 'F0 ∩ B4'}", loc="left", fontsize=7.5)
            ax.tick_params(length=2)
            for E0 in LINES.values():
                ax.axvline(E0, color="#8a8984", lw=0.4, ls=":")
            if i == 1:
                ax.set_xlabel("photon energy (keV)")
            if basis != "thickness":
                continue
            for r in cells[cells.baseline == bl].itertuples():
                if r.element not in els:
                    continue
                y = els.index(r.element) + 0.5
                for e0 in r.energies:
                    ax.plot(e0, y, marker="D", ms=3.4, mfc="#e34948", mec="white", mew=0.5, zorder=6)
                em = max(r.energies)
                rng = f"{100 * r.corr_min:.1f}–{100 * r.corr_max:.1f} %" if r.corr_max - r.corr_min > 5e-4 \
                    else f"{100 * r.corr_max:.1f} %"
                ax.annotate(f"{rng}", (em, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=5.6, color="#a3141d", zorder=7,
                            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
                shown.append((r.element, bl))
    cb = fig.colorbar(im, ax=axs, shrink=0.45, pad=0.01, aspect=30)
    cb.set_label("element-exclusion filter price (%), best non-artefact\ncompound, rule R2, narrow beam; "
                 "square-root colour scale, white = 0")
    h = [Line2D([], [], ls="none", marker="D", ms=4, mfc="#e34948", mec="white",
                label="fluorescence-corrected cells (thickness; E₁ model, K + L lines, best_clean; Phase 3 data/phase3/E): "
                      "marker at each sampled energy; label = corrected price, maximum over the sampled energies, "
                      "range over T = 0.1 / 0.01 (as dossier R2.7)")]
    fig.legend(handles=h, loc="outside lower left", frameon=False, fontsize=6.3)
    fig.suptitle(f"Narrow beam. {len(els)} elements whose filter price reaches ≥ {100 * thresh:g} % at some energy "
                 f"(of {eh.element.nunique()} tested); under the fluorescence correction the U maximum falls to "
                 f"≤ 4 % (red labels; dossier R2.7)", fontsize=7.5)
    out = savefig(fig, "fig4_load_bearing")
    plt.close(fig)
    return out, sorted(set(shown))


def fig4s_load_bearing_b4internal(thresh: float = 0.02) -> tuple[list[str], list[str]]:
    """Supplementary: EXCL(el)|B4 on the stage 4a primaries (B4-internal, best_clean, phase4a flag, R2, coh)."""
    plt = style()
    import matplotlib
    from matplotlib.colors import PowerNorm
    b = pd.read_parquet(S4A / "bands.parquet", columns=["label", "basis", "density", "energy_keV", "price",
                                                         "reference", "artefact_def", "scenario"])
    b = b[(b.scenario == "MONO") & (b.reference == "B4-internal") & (b.artefact_def == common.PRIMARY_ARTEFACT)
          & b.label.astype(str).str.startswith("EXCL(")].copy()
    b["element"] = b.label.astype(str).str.slice(5).str.split(")").str[0]
    b["basis"] = b.basis.astype(str); b["density"] = b.density.astype(str)
    et = c3.elements_table()
    zmap = dict(zip(et["symbol"], et["Z"]))
    mx = b.groupby("element").price.max()
    els = sorted(mx.index[mx >= thresh].tolist(), key=lambda s: zmap.get(s, 999))
    E = np.sort(b.energy_keV.unique())
    edges = np.concatenate([[E[0]], np.sqrt(E[1:] * E[:-1]), [E[-1]]])
    vmax = 100 * float(b.price.max())
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("white")
    panels = (("thickness", "measured", "thickness, measured density (primary)"),
              ("thickness", "mp", "thickness, MP densities"), ("areal", "any", "areal mass"))
    fig, axs = plt.subplots(1, 3, figsize=(c3.DOUBLE_COL_IN, 0.13 * len(els) + 1.5), sharey=True,
                            layout="constrained")
    im = None
    for ax, (basis, den, ttl) in zip(axs, panels):
        z = b[(b.basis == basis) & (b.density == den)].pivot(index="element", columns="energy_keV", values="price") \
            .reindex(index=els, columns=E)
        zz = np.ma.masked_where(~(100 * z.to_numpy() > 0), 100 * z.to_numpy())
        im = ax.pcolormesh(edges, np.arange(len(els) + 1), zz, cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=vmax),
                           shading="flat", rasterized=True)
        _logx(ax)
        ax.set_xticks([20, 100, 1000, 10000]); ax.set_xticklabels(["20", "100", "1k", "10k"])
        ax.set_yticks(np.arange(len(els)) + 0.5)
        ax.set_yticklabels(els, fontsize=6)
        ax.set_ylim(len(els), 0)
        ax.set_title(ttl, loc="left", fontsize=7)
        ax.tick_params(length=2)
        for E0 in LINES.values():
            ax.axvline(E0, color="#8a8984", lw=0.4, ls=":")
        ax.set_xlabel("photon energy (keV)")
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01, aspect=30)
    cb.set_label("element-exclusion filter price within B4 (%),\nq*(B4 ∧ no el)/q*(B4) − 1; sqrt scale, white = 0")
    fig.suptitle(f"Narrow beam, Phase 4a primaries (best_clean, Phase 4a artefact flag, R2, with coherent): "
                 f"{len(els)} of {b.element.nunique()} priced elements reach ≥ {100 * thresh:g} %", fontsize=7)
    out = savefig(fig, "fig4s_load_bearing_b4internal")
    plt.close(fig)
    return out, els


# =============================================================================================
# Fig. 5: DIAG plateau (legible variant tick labels)
# =============================================================================================
DISPLAY_DIAG = ("RQR6_th12", 0.35)
WITHIN = (0.02, 0.05)
TEXTBOOK10 = ("W", "Bi", "Sn", "Sb2O3", "BaSO4", "Gd2O3", "WO3", "Bi2O3", "PbO", "Pb")
C_BEFORE, C_AFTER, C_TB, INK5 = "#8c8c8c", "#2a78d6", "#eb6834", "#333333"


def fig5_inputs() -> dict:
    """Stage 3 families.run inputs to fig5 from data/phase3/C and F (no corrected solve is run).

    Only the display variant's corrected column is assembled: refsets' values (F/corrected_areal.parquet) plus
    the rows stage 3 computed itself (C/corrected_extra.parquet, check rows excluded), as in
    families.diag_inputs. The corrected B4 frontier is checked against C/plateau_by_variant.csv."""
    from shieldmap.analysis3 import families as f3
    from shieldmap.analysis3 import fluorescence as F
    C = P3 / "C"
    fams = pd.read_parquet(C / "families_by_variant.parquet")
    tbv = pd.read_csv(C / "textbook_by_variant.csv")
    pv = pd.read_csv(C / "plateau_by_variant.csv")
    vdf = F._diag_variants("air_kerma").reset_index(drop=True)
    vids = vdf["variant_id"].to_numpy()
    j = int(np.flatnonzero((vdf.spectrum == DISPLAY_DIAG[0]).to_numpy() & np.isclose(vdf.pb_mm, DISPLAY_DIAG[1]))[0])
    vid = int(vids[j])
    u = c3.universe()
    b4 = c3.base_mask("R2") & c3.bundle_mask("B4")
    lan = np.array([bool(set(e) & c3.LANTHANIDES) for e in u["elements"]])
    qe = np.full(len(u), np.nan)
    fc = pd.read_parquet(P3 / "F" / "corrected_areal.parquet", columns=["row", "variant_id", "q_eff_fixedT"])
    fc = fc[fc.variant_id == vid]
    qe[fc.row.to_numpy()] = fc.q_eff_fixedT.to_numpy()
    ex = pd.read_parquet(C / "corrected_extra.parquet")
    ex = ex[(ex.variant_id == vid) & ~ex.is_check]
    qe[ex.row.to_numpy()] = ex.q_eff_fixedT.to_numpy()
    f_after = float(np.nanmin(np.where(b4, qe, np.nan)))
    ref = float(pv[(pv.state == "after") & (pv.variant_id == vid)].frontier_g_cm2.iloc[0])
    if abs(f_after / ref - 1) > 1e-9:
        raise RuntimeError(f"fig5: corrected B4 frontier {f_after} differs from plateau_by_variant {ref}")
    fam = f3.family_frame(np.flatnonzero(b4 & lan), u["elements"].to_numpy()[b4 & lan])
    return {"fams": fams, "tbv": tbv, "pv": pv, "vdf": vdf, "vids": vids, "j": j, "vid": vid, "qe": qe,
            "f_after": f_after, "fam": fam}


def _fam_label(name: str, n: int) -> str:
    ln, part = name.split(" | ")
    return f"{_fmt(ln)}–{_fmt(part)} (n = {n})"            # subscripts only in formulas, not in the count


def fig5_diag_plateau(d: dict, n_fam: int = 12) -> list[str]:
    """Stage 3 families.fig5; panel (c) variant ticks re-labelled hierarchically (anode angle, then T*, then
    quality) at ≥ 6 pt, counts no longer subscripted, legends ≥ 6 pt."""
    from matplotlib.lines import Line2D
    plt = style()
    fams, tbv, pv = d["fams"], d["tbv"], d["pv"]
    ZOOM_HI = 1.10
    fig = plt.figure(figsize=(c3.DOUBLE_COL_IN, 6.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.62)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.45)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.45, 0.85], wspace=0.42)   # (c) wider: 36 variants
    ax1 = fig.add_subplot(top[0, 0]); ax1b = fig.add_subplot(top[0, 1])
    ax2 = fig.add_subplot(bot[0, 0]); ax3 = fig.add_subplot(bot[0, 1])
    vid = d["vid"]
    title_v = f"{DISPLAY_DIAG[0].replace('_th', ' ')}°, T* of {DISPLAY_DIAG[1]} mm Pb"
    # (a)
    fa = fams[(fams.level == "coarse") & (fams.variant_id == vid)]
    aft = fa[fa.state == "after"].nsmallest(n_fam, "best").set_index("family")
    bef = fa[fa.state == "before"].set_index("family").reindex(aft.index)
    fam = d["fam"]
    ylab = []
    for y, name in enumerate(aft.index):
        rows = fam.row[fam.coarse == name].to_numpy()
        ra = d["qe"][rows] / d["f_after"]
        ra = ra[ra <= ZOOM_HI]
        ax1.plot(ra, np.full(ra.size, y), "|", color=C_AFTER, ms=5, alpha=0.45, mew=0.6)
        b = float(bef.loc[name, "best_ratio"])
        if b <= ZOOM_HI:
            ax1.plot(b, y, "o", mfc="white", mec=C_BEFORE, ms=4.2, mew=0.9)
        else:
            ax1.annotate(f"{b:.2f}", (ZOOM_HI, y), xytext=(-2, 0), textcoords="offset points", ha="right",
                         va="center", fontsize=6, color=C_BEFORE)
        ax1.plot(aft.loc[name, "best_ratio"], y, "o", color=C_AFTER, ms=4.2)
        ylab.append(_fam_label(name, int(aft.loc[name, "n"])))
    for x in WITHIN:
        ax1.axvline(1 + x, color=INK5, lw=0.5, ls=":")
    ax1.axvline(1.0, color=INK5, lw=0.6)
    ax1.set_yticks(range(len(ylab)))
    ax1.set_yticklabels(ylab, fontsize=6.5)
    ax1.set_ylim(len(ylab) - 0.4, -0.6)
    ax1.set_xlim(0.995, ZOOM_HI)
    ax1.set_xlabel("areal mass / B4 frontier (same state)")
    ax1.set_title(f"(a) plateau families (Ln–principal partner), {title_v}", loc="left", fontsize=7)
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec=C_BEFORE, label="family best, narrow beam"),
         Line2D([], [], marker="o", ls="none", color=C_AFTER, label="family best, corrected"),
         Line2D([], [], marker="|", ls="none", color=C_AFTER, label=f"members ≤ {ZOOM_HI:.2f}, corrected"),
         Line2D([], [], color=INK5, lw=0.5, ls=":", label="2 % and 5 % above the frontier")]
    ax1.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6.2, frameon=False)
    ax1.grid(axis="x", color="#eeeeee", lw=0.5)
    # (b)
    tb = tbv[tbv.variant_id == vid].pivot(index="formula", columns="state", values="ratio_to_B4_frontier")
    tb = tb.loc[list(TEXTBOOK10)].sort_values("after")
    for y, f in enumerate(tb.index):
        ax1b.plot([tb.loc[f, "before"], tb.loc[f, "after"]], [y, y], color="#dddddd", lw=0.8, zorder=0)
        ax1b.plot(tb.loc[f, "before"], y, "D", mfc="white", mec=C_BEFORE, ms=3.6, mew=0.9)
        ax1b.plot(tb.loc[f, "after"], y, "D", color=C_TB, ms=3.6)
    ax1b.axvline(1.0, color=INK5, lw=0.6)
    ax1b.axvline(float(aft.best_ratio.min()), color=C_AFTER, lw=0.6, ls="--")
    ax1b.set_yticks(range(len(tb)))
    ax1b.set_yticklabels([_fmt(f) + (" (β)" if f == "Sn" else "") for f in tb.index], fontsize=6.5)
    ax1b.set_ylim(len(tb) - 0.4, -0.6)
    ax1b.set_xlabel("areal mass / B4 frontier (same state)")
    ax1b.set_title("(b) textbook materials, same variant", loc="left", fontsize=7)
    ax1b.set_xlim(0.97, float(np.nanmax(tb.values)) * 1.04)
    h = [Line2D([], [], marker="D", ls="none", mfc="white", mec=C_BEFORE, label="narrow beam"),
         Line2D([], [], marker="D", ls="none", color=C_TB, label="corrected"),
         Line2D([], [], color=C_AFTER, lw=0.6, ls="--", label="best family, corrected")]
    ax1b.legend(handles=h, loc="upper right", fontsize=6.2, frameon=True, framealpha=0.95, edgecolor="#dddddd")
    ax1b.grid(axis="x", color="#eeeeee", lw=0.5)
    # (c) plateau breadth per variant, hierarchical x labels
    vd = d["vdf"].sort_values(["quality", "pb_mm", "anode_angle"])
    order = vd.index.to_numpy()
    vids = d["vids"]
    x = np.arange(len(vids))
    zero_after = []
    for state, kw in (("before", dict(mfc="white", mec=C_BEFORE)), ("after", dict(color=C_AFTER))):
        p = pv[pv.state == state].set_index("variant_id").loc[vids[order]]
        ax2.plot(x, p["n_coarse_families_within_2pct"], "o", ms=3.2, ls="none", mew=0.8, **kw)
        if state == "after":
            zero_after = [k for k, n in enumerate(p["n_coarse_families_within_2pct"].to_numpy()) if n == 0]
    for k in zero_after:
        ax2.plot(k, 0, "x", color="#c0392b", ms=4, mew=0.9)
    nq = len(vd.quality.unique())
    per_q = len(x) // nq
    for k in range(1, nq):
        ax2.axvline(per_q * k - 0.5, color="#d0d0d0", lw=0.6, zorder=0)
    for k in range(len(x) // 3):
        if k % 3:
            ax2.axvline(3 * k - 0.5, color="#eeeeee", lw=0.4, zorder=0)
    pbs = vd.pb_mm.to_numpy()
    angs = sorted(set(int(a) for a in vd.anode_angle))
    ax2.set_xticks(x, minor=True)                      # one minor tick per variant (anode angle)
    ax2.set_xticks(np.arange(1, len(x), 3))           # T* label at the centre of each block of three angles
    ax2.set_xticklabels([f"{pbs[k - 1]:.2f}" for k in range(1, len(x), 3)], fontsize=6.3)
    ax2.tick_params(axis="x", which="minor", length=2)
    ax2.tick_params(axis="x", which="major", length=0, pad=3)
    ymax = float(pv.n_coarse_families_within_2pct.max())
    for k, q in enumerate(sorted(vd.quality.unique())):
        ax2.text(per_q * k + (per_q - 1) / 2, 1.01, q, ha="center", va="bottom", fontsize=6.5,
                 transform=ax2.get_xaxis_transform())
    ax2.set_ylim(-0.6, ymax * 1.45 + 1)
    ax2.set_xlim(-0.8, len(x) - 0.2)
    ax2.set_ylabel("families within 2 %")
    ax2.set_xlabel(f"T* (mm Pb); within each block anode angle {'°, '.join(map(str, angs))}° (left to right)")
    ax2.set_title("(c) families within 2 % of the B4 frontier", loc="left", fontsize=7, pad=11)
    h = [Line2D([], [], marker="o", ls="none", mfc="white", mec=C_BEFORE, label="narrow beam"),
         Line2D([], [], marker="o", ls="none", color=C_AFTER, label="corrected"),
         Line2D([], [], marker="x", ls="none", color="#c0392b", label=f"no family within 2 % after correction "
                                                                      f"({len(zero_after)} variants)")]
    ax2.legend(handles=h, fontsize=6.2, frameon=False, loc="upper left", ncol=1)
    # (d)
    tt = tbv.groupby(["formula", "state"]).ratio_to_B4_frontier.agg(["min", "median", "max"])
    order_tb = tt.xs("after", level="state").sort_values("median").index
    for i, f in enumerate(order_tb):
        for k, (state, col) in enumerate((("before", C_BEFORE), ("after", C_TB))):
            r = tt.loc[(f, state)]
            yy = i + (k - 0.5) * 0.32
            ax3.plot([r["min"], r["max"]], [yy, yy], color=col, lw=1.6, solid_capstyle="round")
            ax3.plot(r["median"], yy, "|", color=INK5, ms=4, mew=0.8)
    ax3.set_yticks(range(len(order_tb)))
    ax3.set_yticklabels([_fmt(f) for f in order_tb], fontsize=6.5)
    ax3.invert_yaxis()
    ax3.axvline(1.0, color=INK5, lw=0.6)
    ax3.set_xlabel("textbook areal mass / B4 frontier (36 variants)")
    ax3.set_title("(d) textbook materials: range over 36 variants", loc="left", fontsize=7, pad=11)
    h = [Line2D([], [], color=C_BEFORE, lw=1.6, label="narrow beam"),
         Line2D([], [], color=C_TB, lw=1.6, label="corrected"),
         Line2D([], [], marker="|", ls="none", color=INK5, label="median")]
    ax3.legend(handles=h, fontsize=6.2, frameon=True, framealpha=0.95, edgecolor="#dddddd", loc="upper right")
    ax3.grid(axis="x", color="#eeeeee", lw=0.5)
    out = savefig(fig, "fig5_diag_plateau")
    plt.close(fig)
    return out


# =============================================================================================
def run() -> dict:
    """Regenerate every stage 4a figure (idempotent; reads saved stage 3 and stage 4a data only)."""
    paths.ensure(OUT)
    figs, meta = {}, {}
    i1 = fig1_inputs()
    figs["fig1_variance_physics"] = fig1_variance_physics(i1["dec"], i1["cross"], i1["sh"], i1["diag_e"])
    figs["fig1s_z_exponents"] = fig1s_z_exponents(i1["zx"], i1["cross"])
    meta["fig1_crossing_keV"] = _e_cross(i1["cross"])
    b4i = fig2_inputs("B4-internal")
    figs["fig2_price_curves"], same = fig2_price_curves(b4i)
    meta["fig2_identical_panel_pairs"] = [list(p) for p in same]
    figs["fig2s_price_curves_headroom"] = fig2s_price_curves_headroom(fig2_inputs("headroom"))
    figs["fig3_gap_inversion"] = fig3_gap_inversion(pd.read_csv(P3 / "P" / "gap_inversion.csv"))
    eh, lc = fig4_inputs()
    cells = corrected_cells(eh, lc)
    figs["fig4_load_bearing"], shown = fig4_load_bearing(eh, cells)
    meta["fig4_corrected_cells"] = cells.to_dict("records")
    meta["fig4_corrected_cells_marked"] = [list(s) for s in shown]
    meta["fig4_max_abs_diff_narrow_file_vs_heatmap"] = cells.attrs["max_abs_diff_narrow_vs_heatmap"]
    figs["fig4s_load_bearing_b4internal"], meta["fig4s_elements"] = fig4s_load_bearing_b4internal()
    figs["fig5_diag_plateau"] = fig5_diag_plateau(fig5_inputs())
    res = {"figures": figs, "meta": meta}
    common.write_json(OUT / "figures4.json", res)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["meta"], indent=1, default=str)[:3000])
