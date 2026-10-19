"""Publication figures: main Fig. 1-4, graphical abstract, supplementary S1-S7.

Entry point: ``python -m shieldmap.analyze4b --only figures`` -> ``run()``. Outputs go to ``paper/figures``
(PDF with embedded TrueType fonts, TIFF 600 dpi LZW, PNG 300 dpi), the printed-number register to
``paper/figures/figure_numbers.json``; the checks (CVD, greyscale, fonts, contact sheet) to
``reports/phase4b/figures`` (``shieldmap.figstyle.checks``).

All styling goes through ``shieldmap.figstyle.helpers`` (H) and ``shieldmap.mplstyle``. Data come from the
Files in ``data/phase4b`` and, read only, from stage 3 / stage 4a outputs (loaders of
``shieldmap.analysis4.figures4`` reused). No new analysis.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from shieldmap import paths
from shieldmap.figstyle import helpers as H

ROOT = paths.ROOT
OUT = ROOT / "paper" / "figures"
REP = ROOT / "reports" / "phase4b" / "figures"
D4B = ROOT / "data" / "phase4b"
D4A = ROOT / "data" / "phase4a"
D3 = ROOT / "data" / "phase3"

# ---------------------------------------------------------------------------------------------
# printed-number register (CONVENTIONS.md tags)
# ---------------------------------------------------------------------------------------------
DEFS = {
    # names shared with paper/defs.yaml where the quantity exists there (pb_k_edge, cs137_line)
    "pb_k_edge": (H.E_PB_K, "keV", "Pb K-edge energy (reference marker; = defs.yaml pb_k_edge)"),
    "cs137_line": (H.E_CS137, "keV", "Cs-137 line, marked at 661.7 keV (defs.yaml cs137_line = 661.657)"),
    "co60_mean": (H.E_CO60, "keV", "mean of the Co-60 lines (defs.yaml co60_lines), the single Co-60 marker"),
    "zero": (0.0, "%", "zero price / zero advantage reference"),
    "identity_one": (1.0, "", "the 1 in 1 + price (Fig. 3b identity 1 + P = density factor x composition factor)"),
    "frontier": (0.0, "%", "B4 frontier (0 % above itself)"),
    "diag_pb_targets": (None, "mm Pb", "DIAG Pb targets T* = 0.25, 0.35, 0.50 mm"),
    "abs_delta_screen": (0.2, "", "Phase 3 polymorph screen: a member with |delta| > 0.2"),
    "spread_floor": (1e-4, "", "plot floor of the polymorph spreads (values below drawn at 1e-4)"),
    "ehull_bins": (None, "eV/atom", "Phase 3 e_hull bins (cod.EHULL_BINS): 0, 0-0.025, 0.025-0.05, 0.05-0.1, > 0.1"),
    "excl_display_threshold": (2.0, "%", "Fig. S2 shows elements whose exclusion price reaches >= 2 % somewhere"),
    "amb_icsd_min": (3.0, "ICSD ids", "ambient-chemistry tier AMB-ICSD3: at least 3 ICSD ids (revision E5)"),
    "mc_ci_z": (1.96, "", "Monte Carlo statistical interval ± 1.96 SE (= defs.yaml mc_ci_z)"),
    "anode_angles": (None, "deg", "anode angles 10, 12, 16 degrees"),
}


class Numbers:
    """Every number printed inside a figure, with its source tag, checked against headline_table_v3.csv."""

    def __init__(self):
        self.ht = pd.read_csv(D4B / "headline_table_v3.csv").set_index("id")
        self.rows: dict[str, list[dict]] = {}

    def value(self, tag: str) -> float:
        if tag.startswith("def:"):
            return float(DEFS[tag[4:]][0])
        hid, _, kind = tag.partition(":")
        r = self.ht.loc[hid]
        pct = r.unit == "%"                     # as paper/check_numbers.py
        if kind == "":
            return float(r.value_pct if pct else r.value)
        if kind in ("lo", "hi"):
            return float(r[f"band_{kind}_pct"])
        if kind == "delta":
            return float(r.delta_abs)
        if kind == "delta_mm":
            return float(r.delta_abs) * (10.0 if r.delta_unit == "cm" else 1.0)
        if kind == "pct_over_1":
            return 100.0 * (float(r.value) - 1.0)
        if kind in ("lo_over_1", "hi_over_1"):
            return 100.0 * (float(r[f"band_{kind[:2]}_pct"]) - 1.0)
        raise KeyError(tag)

    @staticmethod
    def matches(printed: str, q: float) -> bool:
        s = printed.replace(H.MINUS, "-").replace(H.THIN, "").replace(",", "")
        dec = len(s.split(".")[1]) if "." in s else 0
        p = float(s)
        r = float(Decimal(repr(q)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))
        if abs(r - p) < 1e-9:
            return True
        return abs(float(f"{q:.2g}") - p) < 1e-9

    def fmt(self, fig: str, tag: str, decimals: int, where: str) -> str:
        """Format the tagged quantity for printing and register it."""
        q = self.value(tag)
        s = H.fmt_num(float(Decimal(repr(q)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)),
                      decimals)
        self.add(fig, s, tag, where)
        return s

    def add(self, fig: str, printed: str, tag: str, where: str):
        q = self.value(tag) if not (tag.startswith("def:") and DEFS[tag[4:]][0] is None) else None
        ok = True if q is None else self.matches(printed, q)
        self.rows.setdefault(fig, []).append({"printed": printed, "tag": tag, "where": where,
                                              "source_value": q, "match": ok})
        if not ok:
            raise ValueError(f"{fig}: printed {printed} does not match {tag} = {q}")

    def write(self):
        out = {"note": "Every number printed inside a figure (annotations, labels), with its source tag "
                       "(paper/CONVENTIONS.md). Tick labels and axis limits are not listed. Checked by "
                       "shieldmap.figstyle.figures.Numbers (half-up rounding or two significant figures).",
               "defs": {k: {"value": v[0], "unit": v[1], "meaning": v[2]} for k, v in DEFS.items()},
               "figures": self.rows}
        (OUT / "figure_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


NUM: Numbers | None = None
META: dict = {}


def _num() -> Numbers:
    global NUM
    if NUM is None:
        NUM = Numbers()
    return NUM


def _save(fig, name: str, **kw):
    files = H.export(fig, name, OUT, **kw)
    META.setdefault(name, {})["files"] = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    META[name]["size_mm"] = list(fig._mm)
    META[name]["min_font_pt_artists"] = H.min_font_size(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return files


# =============================================================================================
# Fig. 2: price of each requirement inside the practical set vs energy
# =============================================================================================
FIG2_PANELS = (("PB|B4", "Lead-free"), ("STAB|B4", "Stable"), ("EXP|B4", "Observed"),
               ("GAP(3.1)|B4", "Transparent"), ("ABUND(1)|B4", "Scarcity proxy"))


def noh_from_kev() -> float:
    """Switch point of the areal curves: the lowest energy at which NH3.H2O is a reference."""
    from shieldmap.analysis4b import patches
    return float(patches.noh_from_kev())


def fig2_curve(d: pd.DataFrame, label: str, basis: str) -> pd.DataFrame:
    """Thickness: measured-density point. Areal: primary set below the switch point (800 keV), H-excluded
    variant at and above it."""
    s = d[(d.label == label) & (d.basis == basis)]
    if basis == "thickness":
        s = s[s.density == "measured"]
    else:
        e0 = noh_from_kev()
        s = pd.concat([s[(s.variant == "primary") & (s.energy_keV < e0)],
                       s[(s.variant == "noH") & (s.energy_keV >= e0)]])
    return s.sort_values("energy_keV")


def h5no_range(set_name: str) -> tuple[float, float] | None:
    """Single-line energy range over which NH3.H2O (MP H5NO) is the areal best_clean frontier of ``set_name``
    (R2, coherent, phase4a artefact flag) in the primary set; read from data/phase4a/S/frontiers.parquet."""
    from shieldmap.analysis3 import common as c3
    f = pd.read_parquet(D4A / "S" / "frontiers.parquet", columns=["col", "basis", "rule", "set", "artefact_def",
                                                                   "kind", "row"],
                        filters=[("set", "==", set_name), ("basis", "==", "areal"), ("rule", "==", "R2"),
                                 ("artefact_def", "==", "phase4a"), ("kind", "==", "best_clean")])
    cols = pd.read_csv(D4A / "S" / "columns.csv")
    cols = cols[(cols.scenario == "MONO") & cols.coherent]
    f = f.merge(cols[["col", "energy_keV"]], on="col")
    u = c3.universe()
    f["formula"] = u["formula_reduced"].to_numpy()[f.row.to_numpy()]
    e = f.loc[f.formula == "H5NO", "energy_keV"]
    return (float(e.min()), float(e.max())) if len(e) else None


def _price_panel(ax, d, label, ylim, show_x=True):
    for basis, col in (("thickness", H.THICKNESS), ("areal", H.AREAL)):
        c = fig2_curve(d, label, basis)
        H.band(ax, c.energy_keV, 100 * c.band_lo, 100 * c.band_hi, col, hatched=(basis == "areal"))
        ax.plot(c.energy_keV, 100 * c.price, color=col, lw=H.LW_DATA, zorder=3)
    H.zero_line(ax)
    ax.set_ylim(*ylim)
    H.log_energy_axis(ax, labels=show_x, xlabel=show_x)
    H.hgrid(ax)


def fig2():
    d = pd.read_csv(D4B / "fig2_prices.csv")
    fig = H.figure(H.DOUBLE, 112)
    L, R, GAP1 = 14.0, 4.0, 7.0
    w1 = (H.DOUBLE - L - R - 2 * GAP1) / 3
    GAP2 = 14.0
    w2 = (H.DOUBLE - L - R - GAP2) / 2
    hgt, b2, b1 = 40.0, 10.0, 66.0
    axs = [H.add_axes_mm(fig, L + k * (w1 + GAP1), b1, w1, hgt) for k in range(3)]
    axs += [H.add_axes_mm(fig, L + k * (w2 + GAP2), b2, w2, hgt) for k in range(2)]
    top3 = 100 * max(float(fig2_curve(d, lab, bs)[["band_hi", "price"]].to_numpy().max())
                     for lab, _ in FIG2_PANELS[:3] for bs in ("thickness", "areal"))
    ylims, yticks = {}, {}
    for lab, _ in FIG2_PANELS[:3]:
        ylims[lab] = (-0.6, np.ceil(top3 / 5) * 5)
        yticks[lab] = np.arange(0, ylims[lab][1] + 0.1, 5)
    for lab, _ in FIG2_PANELS[3:]:
        t = 100 * max(float(fig2_curve(d, lab, bs)[["band_hi", "price"]].to_numpy().max())
                      for bs in ("thickness", "areal"))
        step = 50 if t > 100 else 10
        ylims[lab] = (-0.03 * t, np.ceil(t / step) * step)
        yticks[lab] = np.arange(0, ylims[lab][1] + 0.1, step)
    for k, (ax, (lab, name)) in enumerate(zip(axs, FIG2_PANELS)):
        _price_panel(ax, d, lab, ylims[lab], show_x=True)
        ax.set_yticks(yticks[lab])
        if k in (0, 2):
            ax.set_xlabel("")
        H.ref_markers(ax, label=(k in (0, 3)))
        H.note(ax, 0.02, 0.97, name, va="top", color=H.INK, fontsize=H.FS_LABEL,
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        if k in (1, 2):
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("Price (%)")
        H.panel_letter(ax, "abcde"[k], dx_mm=(11.5 if k in (0, 3) else 3.5) if k != 4 else 11.5)
    # direct labels on the transparency panel (thickness / areal mass key for the whole figure)
    ax = axs[3]
    c = fig2_curve(d, "GAP(3.1)|B4", "thickness")
    e0 = 300.0
    y0 = float(np.interp(np.log(e0), np.log(c.energy_keV), 100 * c.price))
    H.direct_label(ax, e0, y0, "Thickness", H.THICKNESS, dx=0, dy=4, ha="center", va="bottom")
    c = fig2_curve(d, "GAP(3.1)|B4", "areal")
    e1 = 3000.0
    y1 = float(np.interp(np.log(e1), np.log(c.energy_keV), 100 * c.price))
    H.direct_label(ax, e1, y1, "Areal mass", H.AREAL, dx=0, dy=4, ha="center", va="bottom")
    # compact key on panel a (the first-row panels share it)
    ax = axs[0]
    for j, (txt, col) in enumerate((("Thickness", H.THICKNESS), ("Areal mass", H.AREAL))):
        y = 0.62 - 0.11 * j
        ax.plot([0.72, 0.79], [y, y], transform=ax.transAxes, color=col, lw=H.LW_DATA, clip_on=False)
        H.direct_label(ax, 0.79, y, txt, col, dx=1.5, coords="axes fraction")
    META["Fig2"] = {"ylim_shared_rows1": ylims["PB|B4"], "data": "data/phase4b/fig2_prices.csv"}
    _save(fig, "Fig2")


# =============================================================================================
# Fig. 1: (a) pipeline schematic, (b) composition vs density share of var(log mu)
# =============================================================================================
def _box(ax, x, y, w, h, title, body=None, fill=H.BOX_FILL, edge="none", ls="-"):
    """Flat schematic box (mm coordinates): bold title at the top, secondary-ink body below."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=edge,
                                lw=0.6 if edge != "none" else 0, ls=ls))
    top = y + h - 2.0
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=H.FS_LABEL, color=H.INK, fontweight="bold",
            linespacing=1.15)
    if body:
        nt = title.count("\n") + 1
        ax.text(x + w / 2, top - 3.3 * nt - 1.6, body, ha="center", va="top",
                fontsize=H.FS_CHEM if "$" in body else H.FS_ANNOT, color=H.INK2, linespacing=1.3)


def _arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=5.5, color=H.ARROW, lw=0.7, shrinkA=0,
                                shrinkB=0, linestyle=ls))


def fig1_schematic(ax, W, Hh):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, W)
    ax.set_ylim(0, Hh)
    ax.axis("off")
    bw, gap = 20.5, 4.0
    xin, win = 0.5, 26.5
    # inputs
    _box(ax, xin, 40.5, win, 14, "XCOM", "elemental\ncross-sections")
    _box(ax, xin, 22.5, win, 14, "Materials Project", "compositions,\nDFT densities")
    _box(ax, xin, 3.5, win, 14, "COD", "measured densities\n(density check)", fill="white", edge=H.SPINE,
         ls=(0, (3, 2)))
    _arrow(ax, xin + win / 2, 17.5, xin + win / 2, 22.3)
    # chain
    x1 = xin + win + gap
    yb0, hh = 12.0, 36.0
    yb = yb0 + hh / 2
    stages = [("Mixture rule", r"$\mu/\rho = \Sigma\, w_i\,(\mu/\rho)_i$"),
              ("Narrow-beam\ntransmission", "per spectrum:\nsingle lines,\nCs-137, Co-60,\ntube spectra"),
              ("Practical-set\nfilters", "B4: non-radioactive,\nlead-free, stable,\nobserved")]
    xs = []
    for k, (t, b) in enumerate(stages):
        x = x1 + k * (bw + gap)
        xs.append(x)
        _box(ax, x, yb0, bw, hh, t, b)
    _arrow(ax, xin + win, 47.5, xs[0], yb + 5)
    _arrow(ax, xin + win, 29.5, xs[0], yb - 5)
    for k in range(len(xs) - 1):
        _arrow(ax, xs[k] + bw, yb, xs[k + 1], yb)
    # prices
    xp = xs[-1] + bw + gap
    wp = W - xp - 0.5
    _box(ax, xp, yb0, wp, hh, "Prices", "of each\nrequirement\ninside B4")
    for j, (txt, col) in enumerate((("thickness", H.THICKNESS), ("areal mass", H.AREAL))):
        yc = yb0 + 10.5 - 6.0 * j
        ax.add_patch(FancyBboxPatch((xp + 1.2, yc - 2.3), wp - 2.4, 4.6, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=col, ec="none"))
        ax.text(xp + wp / 2, yc, txt, ha="center", va="center", fontsize=H.FS_ANNOT, color="white",
                fontweight="bold")
    _arrow(ax, xs[-1] + bw, yb, xp, yb)


def fig1():
    from shieldmap.analysis4 import figures4 as f4
    i1 = f4.fig1_inputs()
    dec, cross = i1["dec"], i1["cross"]
    fig = H.figure(H.DOUBLE, 64)
    Wa = 122.0
    axa = H.add_axes_mm(fig, 0, 2, Wa, 58)
    fig1_schematic(axa, Wa, 58)
    H.panel_letter(axa, "a", x_mm=0.5, dy_mm=-2.5)
    ax = H.add_axes_mm(fig, 139, 10, 47, 46)
    d = dec[dec.subset == "all"]
    prim = d[(d.rule == "R2") & d.coherent].sort_values("energy_keV")
    n = _num()
    xc = cross[(cross.subset == "all") & (cross.rule == "R2") & cross.coherent & (cross.direction == "comp->dens")]
    e_cross = float(xc.E_cross_keV.iloc[-1])
    lo, hi = n.value("H79:lo"), n.value("H79:hi")
    if abs(e_cross - n.value("H79")) > 1e-6:
        raise ValueError("Fig1: crossing differs from H79")
    ax.axvspan(lo, hi, color=H.REF, alpha=0.35, lw=0, zorder=0)
    ax.axvline(e_cross, color=H.INK2, lw=H.LW_REF, zorder=1)
    for col, c, name in (("share_composition", H.COMPOSITION, "Composition"), ("share_density", H.DENSITY, "Density")):
        piv = d.pivot_table(index="energy_keV", columns=["rule", "coherent"], values=col)
        H.band(ax, piv.index, piv.min(axis=1), piv.max(axis=1), c)
        ax.plot(prim.energy_keV, prim[col], color=c, lw=H.LW_DATA, zorder=3,
                ls=H.DENSITY_LS if col == "share_density" else "-")
    ax.plot(prim.energy_keV, prim.share_cov, color=H.INK2, lw=0.7, ls=(0, (1, 1.2)), zorder=2)
    H.zero_line(ax)
    H.log_energy_axis(ax)
    ax.set_ylim(-0.12, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_ylabel("Share of var(log μ)")
    H.hgrid(ax)
    # direct labels
    def yat(col, e):
        return float(np.interp(np.log(e), np.log(prim.energy_keV), prim[col]))
    H.direct_label(ax, 22, 0.60, "Composition", H.COMPOSITION, dx=0, dy=0, ha="left", va="bottom")
    H.direct_label(ax, 560, yat("share_density", 560), "Density", H.DENSITY, dx=6, dy=-2, ha="left",
                   va="top")
    H.direct_label(ax, 2500, yat("share_cov", 2500), "Covariance", H.INK2, dx=0, dy=3, ha="center", va="bottom")
    s = n.fmt("Fig1b", "H79", 0, "crossover energy label")
    ax.annotate(f"{s} keV", xy=(e_cross, 0.98), xycoords=("data", "axes fraction"), xytext=(-3, 0),
                textcoords="offset points", ha="right", va="top", fontsize=H.FS_ANNOT, color=H.INK)
    ax.annotate("crossover of\nthe population", xy=(e_cross, 0.90), xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points", ha="right", va="top", fontsize=H.FS_ANNOT,
                color=H.INK2, linespacing=1.15)
    H.panel_letter(ax, "b", dx_mm=11.0)
    META["Fig1"] = {"crossing_keV": e_cross, "band": [lo, hi], "data": "data/phase3/L (figures4.fig1_inputs)"}
    _save(fig, "Fig1")


# =============================================================================================
# Fig. 3: transparency is a space price
# =============================================================================================
MC_STYLE = {"narrow": dict(marker="o", mfc=H.MC, mec="white", mew=0.4, ms=H.MS + 0.3),
            "broad": dict(marker="D", mfc="white", mec=H.MC, mew=0.8, ms=H.MS - 0.2),
            "fl": dict(marker="s", mfc=H.MC, mec="white", mew=0.4, ms=H.MS)}
MC_NAME = {"narrow": "Monte Carlo, narrow beam", "broad": "Monte Carlo, broad beam",
           "fl": "Monte Carlo, fluorescence-inclusive"}


def fig3():
    g = pd.read_csv(D4B / "fig3_gap_b4internal.csv")
    mc = pd.read_csv(D4B / "fig3_mc_points.csv")
    m = g[g.scenario == "MONO"].sort_values("energy_keV")
    n = _num()
    fig = H.figure(H.ONE_HALF, 64)
    axa = H.add_axes_mm(fig, 12.5, 10, 53, 47)
    axb = H.add_axes_mm(fig, 84, 10, 50, 47)
    # (a) both bases with credible bands, MC points at Cs-137 and Co-60
    ax = axa
    if (m.mass_variant == "noH").any() and abs(m[m.mass_variant == "noH"].energy_keV.min() - noh_from_kev()) > 1e-6:
        raise ValueError("Fig3: mass_variant switch differs from patches.noh_from_kev()")
    for col, c in (("P_thickness", H.THICKNESS), ("P_mass", H.AREAL)):
        H.band(ax, m.energy_keV, 100 * m[f"{col}_lo"], 100 * m[f"{col}_hi"], c, hatched=(col == "P_mass"))
        ax.plot(m.energy_keV, 100 * m[col], color=c, lw=H.LW_DATA, zorder=3)
    H.zero_line(ax)
    H.log_energy_axis(ax)
    ax.set_ylim(-6, 225)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("Price of transparency (%)")
    H.hgrid(ax)
    H.ref_markers(ax)
    E = {"c1_cs137": H.E_CS137, "c2_co60": H.E_CO60}
    for est, fx in (("narrow", 0.90), ("broad", 1.11)):
        for case, e in E.items():
            r = mc[(mc.case_id == case) & (mc.estimator == est)].iloc[0]
            ax.plot(e * fx, 100 * r.price, ls="none", zorder=5, **MC_STYLE[est])
    # direct labels and the two headline numbers at Cs-137
    def yat(col, e):
        return float(np.interp(np.log(e), np.log(m.energy_keV), 100 * m[col]))
    H.direct_label(ax, 150, yat("P_thickness", 150), "Thickness", H.THICKNESS, dx=0, dy=-4, ha="center", va="top")
    H.direct_label(ax, 200, yat("P_mass", 200), "Areal mass", H.AREAL, dx=0, dy=10, ha="center", va="bottom")
    s9 = n.fmt("Fig3a", "H09", 0, "GAP thickness price at Cs-137 (label)")
    s10 = n.fmt("Fig3a", "H10", 1, "GAP areal-mass price at Cs-137 (label)")
    # values at Cs-137: the labels end on the Cs-137 marker line
    H.direct_label(ax, H.E_CS137, yat("P_thickness", H.E_CS137), f"{s9} %", H.THICKNESS, dx=-1.5, dy=8,
                   ha="right", va="bottom")
    H.direct_label(ax, H.E_CS137, yat("P_mass", H.E_CS137), f"{s10} %", H.AREAL, dx=-1.5, dy=4, ha="right",
                   va="bottom")
    # MC key (frameless, in the empty upper right)
    for j, est in enumerate(("narrow", "broad")):
        y = 0.42 - 0.085 * j
        ax.plot([0.70], [y], transform=ax.transAxes, ls="none", clip_on=False, **MC_STYLE[est])
        H.note(ax, 0.73, y, ("MC narrow" if est == "narrow" else "MC broad"), color=H.INK, fontsize=H.FS_ANNOT)
    H.panel_letter(ax, "a", dx_mm=11.0)
    # (b) decomposition
    ax = axb
    ax.set_yscale("log")
    ax.plot(m.energy_keV, 1 + m.P_thickness, color=H.THICKNESS, lw=1.25, zorder=4)
    ax.plot(m.energy_keV, m.density_factor, color=H.DENSITY, lw=H.LW_DATA, zorder=3, ls=H.DENSITY_LS)
    ax.plot(m.energy_keV, m.composition_factor, color=H.COMPOSITION, lw=H.LW_DATA, zorder=3)
    H.zero_line(ax, 1.0)
    H.log_energy_axis(ax)
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.set_ylim(0.45, 4.2)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 4]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["0.5", "1", "2", "3", "4"])
    ax.set_ylabel("Factor (log scale)")
    H.hgrid(ax)
    H.ref_markers(ax)

    def yb(col, e):
        v = (1 + m.P_thickness) if col == "prod" else m[col]
        return float(np.exp(np.interp(np.log(e), np.log(m.energy_keV), np.log(v))))
    H.direct_label(ax, 10000, yb("prod", 10000), "1 + price", H.THICKNESS, dx=0, dy=5, ha="right",
                   va="bottom")
    n.add("Fig3b", "1", "def:identity_one", "label '1 + price'")
    H.direct_label(ax, 3000, yb("density_factor", 3000), "Density factor", H.DENSITY, dx=0, dy=-4, ha="center",
                   va="top")
    H.direct_label(ax, 3000, yb("composition_factor", 3000), "Composition factor", H.COMPOSITION, dx=0, dy=4,
                   ha="center", va="bottom")
    # the Cs-137 decomposition: density x composition
    gc = g[g.scenario == "CS137"]
    cs = gc.loc[(gc.P_thickness - n.value("H09") / 100).abs().idxmin()]     # the headline profile row (T = 0.01)
    if not (abs(cs.density_factor - n.value("H150")) < 1e-9 and abs(cs.composition_factor - n.value("H151")) < 1e-9):
        raise ValueError("Fig3b: Cs-137 factors differ from H150/H151")
    sd = n.fmt("Fig3b", "H150", 2, "density factor at Cs-137")
    sc = n.fmt("Fig3b", "H151", 2, "composition factor at Cs-137")
    for v, c in ((cs.density_factor, H.DENSITY), (cs.composition_factor, H.COMPOSITION)):
        ax.plot([H.E_CS137], [v], ls="none", marker="o", ms=H.MS, mfc="white", mec=H.text_color(c), mew=0.7,
                zorder=6)
    # the two circled values, written once in the empty lower right
    H.note(ax, 0.97, 0.05, f"Cs-137: {sd} × {sc}", ha="right", va="bottom", color=H.INK,
           bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    H.panel_letter(ax, "b", dx_mm=11.0)
    META["Fig3"] = {"data": ["data/phase4b/fig3_gap_b4internal.csv (scenario MONO)", "data/phase4b/fig3_mc_points.csv"],
                    "mc_shown": ["narrow", "broad"], "mc_x_offset_factor": [0.90, 1.11]}
    _save(fig, "Fig3")


# =============================================================================================
# Fig. 4: diagnostic frontier
# =============================================================================================
TEXTBOOK10 = ("W", "Bi", "Sn", "Sb2O3", "BaSO4", "Gd2O3", "WO3", "Bi2O3", "PbO", "Pb")
NARROW_MK = dict(marker="o", ms=H.MS + 0.2, mec="white", mew=0.4, ls="none")
FIRST_MK = dict(marker="o", ms=H.MS, mfc="white", mec=H.MODEL, mew=0.9, ls="none")


def _tb_label(f: str) -> str:
    return "β-Sn" if f == "Sn" else H.chem(f)


def fig4():
    fa = pd.read_csv(D4B / "fig4a_families_v13.csv")
    tb = pd.read_csv(D4B / "fig4a_textbook_v13.csv")
    adv = pd.read_csv(D4B / "fig4b_advantage_by_variant.csv")
    mc = pd.read_csv(D4B / "fig4b_mc_points.csv")
    n = _num()
    blend = pd.read_csv(D4B / "revision" / "e2_blend_by_variant.csv").set_index("variant_id")
    amb = pd.read_csv(D4B / "revision" / "e5_amb_lanthanide.csv")
    fig = H.figure(H.DOUBLE, 110)
    # ---------------- (a)
    ax = H.add_axes_mm(fig, 17, 10, 58, 89)
    ln = fa[(fa.level == "ln") & ~fa.family.str.contains(r"\+")]
    nb = ln[ln.state == "before"].set_index("family").best_ratio
    co = ln[ln.state == "after"].set_index("family").best_ratio
    fams = nb.sort_values().index.tolist()
    t = tb.pivot(index="formula", columns="state", values="ratio_to_B4_frontier").loc[list(TEXTBOOK10)]
    t = t.sort_values("before")
    # ambient-chemistry tier: best fully oxidised lanthanide compound with >= 3 ICSD ids, v13
    a13 = amb[(amb.tier == "AMB-ANION&ICSD3") & (amb.variant_id == 13)].iloc[0]
    if abs(100 * a13.vs_B4 - n.value("H214")) > 1e-9:
        raise ValueError("Fig4a: ambient-tier lanthanide value differs from H214")
    rows = [("ln", f) for f in fams] + [("amb", "Ln, fully oxidised")] + [("tb", f) for f in t.index]
    ypos, y = [], 0.0
    for k, (kind, _) in enumerate(rows):
        if k > 0 and kind != rows[k - 1][0]:
            y += 1.6 if kind == "tb" else 0.4
        ypos.append(y)
        y += 1.0
    for (kind, f), yy in zip(rows, ypos):
        if kind == "amb":
            ax.plot(100 * a13.vs_B4, yy, marker="D", ms=H.MS, mfc=H.LANTHANIDE, mec="white", mew=0.4, ls="none",
                    zorder=4)
            continue
        if kind == "ln":
            a, b, col = nb[f], co[f], H.LANTHANIDE
        else:
            a, b = t.loc[f, "before"], t.loc[f, "after"]
            col = H.BI if f == "Bi" else H.TEXTBOOK
        ax.plot([100 * (a - 1), 100 * (b - 1)], [yy, yy], color=H.GRID, lw=1.0, zorder=1, solid_capstyle="butt")
        ax.plot(100 * (b - 1), yy, zorder=3, **FIRST_MK)
        ax.plot(100 * (a - 1), yy, mfc=col, zorder=4, **NARROW_MK)
    i_amb = rows.index(("amb", "Ln, fully oxidised"))
    H.note(ax, 100 * a13.vs_B4 + 3, ypos[i_amb], "Ln, fully oxidised (best, ≥ 3 ICSD entries)",
           coords="data", color=H.text_color(H.LANTHANIDE))
    n.add("Fig4a", "3", "def:amb_icsd_min", "ambient-tier row note (>= 3 ICSD entries)")
    # elemental blend reference line: narrow beam only
    xb_ = 100 * float(blend.loc[13, "blend_vs_B4"])
    if abs(xb_ - n.value("H180")) > 1e-9:
        raise ValueError("Fig4a: blend value differs from H180")
    ax.axvline(xb_, color=H.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=2)
    sbl = n.fmt("Fig4a", "H180", 2, "elemental blend vs the B4 frontier, v13, narrow beam (label)")
    ax.annotate(f"elemental blend (narrow beam), {sbl} %", xy=(xb_, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-1, 2.5), textcoords="offset points", ha="left", va="bottom", fontsize=H.FS_ANNOT,
                color=H.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f if k == "ln" else ("" if k == "amb" else _tb_label(f)) for k, f in rows], fontsize=H.FS_CHEM)
    for lab, (kind, f) in zip(ax.get_yticklabels(), rows):
        lab.set_color(H.text_color(H.LANTHANIDE) if kind in ("ln", "amb")
                      else (H.text_color(H.BI) if f == "Bi" else H.INK))
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ypos[-1] + 0.8, -2.2)
    ax.set_xlim(-3, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(0, color=H.INK2, lw=H.LW_REF, zorder=0)
    H.vgrid(ax)
    ax.set_xlabel("Areal mass above the B4 frontier (%)")
    n.add("Fig4a", "0", "def:frontier", "x = 0: B4 frontier")
    i_tb = rows.index(("tb", t.index[0]))
    H.note(ax, 1.5, -1.2, "Lanthanide families (best member)", coords="data", va="center", fontsize=H.FS_ANNOT,
           color=H.text_color(H.LANTHANIDE))
    H.note(ax, 1.5, ypos[i_tb] - 1.15, "Textbook materials", coords="data", va="center", fontsize=H.FS_ANNOT,
           color=H.INK2)
    # which textbook material is closest depends on the estimator: say so on the rows
    i_bi, i_sn = rows.index(("tb", "Bi")), rows.index(("tb", "Sn"))
    xb = 100 * (max(t.loc["Bi", "before"], t.loc["Bi", "after"]) - 1)
    xs = 100 * (max(t.loc["Sn", "before"], t.loc["Sn", "after"]) - 1)
    H.note(ax, xb + 4, ypos[i_bi], "closest, fluorescence model", coords="data", color=H.text_color(H.BI))
    H.note(ax, xs + 4, ypos[i_sn], "closest, narrow beam", coords="data", color=H.INK2)
    # key (upper right, empty region): filled = narrow beam (teal or grey), open = first-order model only
    yy = 2.0
    ax.plot([49], [yy], mfc=H.LANTHANIDE, **NARROW_MK)
    ax.plot([52], [yy], mfc=H.TEXTBOOK, **NARROW_MK)
    H.note(ax, 55, yy, "Narrow beam", coords="data", color=H.INK)
    ax.plot([52], [yy + 1.3], **FIRST_MK)
    H.note(ax, 55, yy + 1.3, "First-order fluorescence\n(model only)", coords="data", color=H.INK, va="top",
           linespacing=1.15)
    ax.texts[-1].set_position((55, yy + 0.95))
    H.panel_letter(ax, "a", x_mm=1.0, dy_mm=4.5)
    # ---------------- (b)
    ax = H.add_axes_mm(fig, 97, 70, 89, 22)
    q_order = ["RQR5", "RQR6", "RQR8", "RQR9"]
    adv = adv.copy()
    adv["qi"] = adv.quality.map({q: i for i, q in enumerate(q_order)})
    pbs = sorted(adv.pb_mm.unique())
    angs = sorted(adv.anode_angle.unique())
    adv["pi"] = adv.pb_mm.map({p: i for i, p in enumerate(pbs)})
    adv["ai"] = adv.anode_angle.map({a: i for i, a in enumerate(angs)})
    QW = 3 * 3 + 2 * 0.8 + 2.4
    adv["x"] = adv.qi * QW + adv.pi * 3.8 + adv.ai
    ax.plot(adv.x, 100 * adv.adv_corrected_estimate, zorder=3, **FIRST_MK)
    ax.plot(adv.x, 100 * adv.adv_narrow, mfc=H.LANTHANIDE, zorder=4, **NARROW_MK)
    # elemental blend on the same y quantity: advantage of the best blend over the best textbook
    # material, narrow beam. In every variant the best B4 lanthanide compound is the B4 frontier (checked), so
    # q_tb / q_blend - 1 = (1 + adv_narrow) / (1 + blend_vs_B4) - 1 exactly (no new computation).
    bb = blend.reindex(adv.variant_id)
    if not (bb.B4_frontier.to_numpy() == adv.best_ln_narrow.to_numpy()).all():
        raise ValueError("Fig4b: best lanthanide compound is not the B4 frontier in every variant")
    adv["blend_adv"] = (1 + adv.adv_narrow.to_numpy()) / (1 + bb.blend_vs_B4.to_numpy()) - 1
    for (qi, pi), g in adv.groupby(["qi", "pi"]):
        g = g.sort_values("ai")
        ax.plot(g.x, 100 * g.blend_adv, color=H.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=5)
    xm = adv.set_index("variant_id").x
    mc_vids = sorted(set(mc.variant_id.unique()) | {13})          # 13: case c7 (revision)
    # carets: the two variants simulated by Monte Carlo (panel c, same compound pair)
    ax.plot([xm[v] for v in mc_vids], [44.0] * len(mc_vids), ls="none", marker="v", ms=3.2, mfc=H.MC, mec="none",
            zorder=6, clip_on=False)
    H.zero_line(ax, 0.0, color=H.INK2, lw=0.8)
    n.add("Fig4b", "0", "def:zero", "zero line (direction of the advantage)")
    ax.set_ylim(-2, 46)
    ax.set_yticks([0, 20, 40])
    ax.set_ylabel("Advantage over best\ntextbook material (%)")
    H.hgrid(ax)
    ctr = [adv[(adv.qi == qi) & (adv.pi == pi)].x.mean() for qi in range(4) for pi in range(len(pbs))]
    ax.set_xticks(ctr)
    ax.set_xticklabels([f"{p:.2f}" for _ in range(4) for p in pbs], fontsize=H.FS_ANNOT)
    n.add("Fig4b", "0.25/0.35/0.50", "def:diag_pb_targets", "x tick labels (Pb targets)")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(adv.x.min() - 1.2, adv.x.max() + 1.2)
    for qi, q in enumerate(q_order):
        xc = adv[adv.qi == qi].x.mean()
        ax.annotate(q, xy=(xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=H.FS_ANNOT, color=H.INK)
        if qi:
            ax.axvline(adv[adv.qi == qi].x.min() - 1.2, color=H.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Pb target (mm), by beam quality; anode angles 10°, 12°, 16° left to right", labelpad=11)
    n.add("Fig4b", "10°/12°/16°", "def:anode_angles", "x label (anode-angle order within each group)")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], mfc=H.LANTHANIDE, **NARROW_MK), Line2D([], [], **FIRST_MK),
               Line2D([], [], color=H.INK, lw=0.8, ls=(0, (3, 1.6))),
               Line2D([], [], ls="none", marker="v", ms=3.2, mfc=H.MC, mec="none")],
              ["Lanthanide compound, narrow beam", "First-order fluorescence (model only)",
               "Elemental blend (narrow beam)", "Simulated in c"],
              loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, handletextpad=0.3, columnspacing=1.2,
              borderaxespad=0, fontsize=H.FS_ANNOT)
    H.panel_letter(ax, "b", dx_mm=11.5, dy_mm=11.5)
    # ---------------- (c) every Monte Carlo pair, all estimators on the same pair (review must-fix 2; revision
    # items 3 and O1-O3): one row per pair, grouped by beam; MC like-for-like filled, fixed target open
    ax = H.add_axes_mm(fig, 125, 11, 61, 33)
    met = pd.read_csv(D4A / "mc" / "metrics.csv")
    lfl = pd.read_csv(D4B / "revision" / "e3_like_for_like.csv")
    m7 = pd.read_csv(D4B / "mc_rev" / "metrics.csv")
    LFL_ROWS = {("c3_rqr5", "fl"): "H189", ("c3_rqr5", "broad"): "H190", ("c4_rqr8", "fl"): "H191",
                ("c4_rqr8", "broad"): "H192", ("c3_rqr5", "narrow"): "H193", ("c4_rqr8", "narrow"): "H194"}
    C7_ROWS = {"Ce5AgSn3": ("H260", "H261", "H262", "H263"), "PrSbPd": ("H264", "H265", "H266", "H267"),
               "Ba2SmSbO6": ("H268", "H269", "H270", "H271"), "SnBaGd_blend": ("H272", "H273", "H274", "H275")}
    MC_EST = {"narrow": "o", "fl": "s", "broad": "D"}
    ANALYTIC_MK = dict(marker="o", ms=H.MS, mfc=H.TEXTBOOK, mec="white", mew=0.4, ls="none")
    rows_c = []                               # (label, is_lanthanide, {est: fixed}, {est: (lfl, se)}, beam)
    for vid in sorted(mc.variant_id.unique()):
        s = mc[mc.variant_id == vid].set_index("estimator")
        case = s.case_id.iloc[0]
        m4 = met[(met.case_id == case) & met.metric.str.startswith("advantage")].set_index("estimator")
        for e in ("T_tablec", "T_firstorder", *MC_EST):
            if abs(float(m4.loc[e, "value"]) - float(s.loc[e, "advantage"])) > 1e-9:
                raise ValueError(f"Fig4c: {case} {e} differs between fig4b_mc_points and metrics.csv")
        lc = lfl[lfl.case_id == case].set_index("estimator")
        for e in MC_EST:
            if abs(100 * float(lc.loc[e, "adv_lfl"]) - n.value(LFL_ROWS[(case, e)])) > 1e-9:
                raise ValueError(f"Fig4c: {case} {e} like-for-like differs from {LFL_ROWS[(case, e)]}")
        q = adv.set_index("variant_id").loc[vid, "quality"]
        ln_el = "La" if s.lanthanide.iloc[0].startswith("La") else s.lanthanide.iloc[0][:2]
        fixed = {e: float(s.loc[e, "advantage"]) for e in ("T_tablec", "T_firstorder", *MC_EST)}
        like = {e: (float(lc.loc[e, "adv_lfl"]), float(lc.loc[e, "adv_lfl_se"])) for e in MC_EST}
        rows_c.append((f"{ln_el} compound, {q}", True, fixed, like, q))
    # case c7 at the reference beam (variant 13): frontier compound, Pr compound, blend, fully oxidised compound
    for mat, lab, is_ln in (("Ce5AgSn3", "Ce compound, RQR6", True), ("PrSbPd", "Pr compound, RQR6", True),
                            ("SnBaGd_blend", "Sn/Ba/Gd blend, RQR6", False), ("Ba2SmSbO6", "Ln, fully oxidised, RQR6", True)):
        mm = m7[m7.metric == f"Bi/{mat} − 1"]
        fx = mm[mm.target == "fixed"].set_index("estimator")
        lk = mm[mm.target == "like-for-like"].set_index("estimator")
        tags = C7_ROWS[mat]
        for val, tag in ((lk.loc["fl", "value"], tags[0]), (lk.loc["broad", "value"], tags[1]),
                         (fx.loc["fl", "value"], tags[2]), (fx.loc["broad", "value"], tags[3])):
            if abs(100 * float(val) - n.value(tag)) > 1e-9:
                raise ValueError(f"Fig4c: c7 {mat} differs from {tag}")
        fixed = {e: float(fx.loc[e, "value"]) for e in ("T_tablec", "T_firstorder", *MC_EST)}
        like = {e: (float(lk.loc[e, "value"]), float(lk.loc[e, "se"])) for e in MC_EST}
        rows_c.append((lab, is_ln, fixed, like, "RQR6"))
    order = ["RQR5", "RQR6", "RQR8"]
    rows_c.sort(key=lambda r: order.index(r[4]))
    yc, y = [], 0.0
    for k, r in enumerate(rows_c):
        if k and r[4] != rows_c[k - 1][4]:
            y += 0.5
        yc.append(y)
        y += 1.0
    OFF = {"T_tablec": -0.30, "T_firstorder": -0.15, "narrow": 0.0, "fl": 0.15, "broad": 0.30}
    for (lab, is_ln, fixed, like, q), yy in zip(rows_c, yc):
        ax.plot(100 * fixed["T_tablec"], yy + OFF["T_tablec"], zorder=5, **ANALYTIC_MK)
        ax.plot(100 * fixed["T_firstorder"], yy + OFF["T_firstorder"], zorder=5, **FIRST_MK)
        for e, mk in MC_EST.items():
            xl, se = like[e]
            xf = fixed[e]
            ax.plot([100 * xf, 100 * xl], [yy + OFF[e]] * 2, color=H.TEXTBOOK_LIGHT, lw=0.4, zorder=2)
            ax.plot(100 * xf, yy + OFF[e], ls="none", marker=mk, ms=H.MS - 0.5, mfc="white", mec=H.MC, mew=0.6,
                    zorder=4)
            ax.errorbar(100 * xl, yy + OFF[e], xerr=100 * 1.96 * se, fmt="none", ecolor=H.MC, elinewidth=0.6,
                        capsize=0.8, capthick=0.6, zorder=5)
            ax.plot(100 * xl, yy + OFF[e], ls="none", marker=mk, ms=H.MS - 0.2, mfc=H.MC, mec="white", mew=0.4,
                    zorder=6)
    ax.set_yticks(yc)
    ax.set_yticklabels([r[0] for r in rows_c], fontsize=H.FS_ANNOT)
    for tl, r in zip(ax.get_yticklabels(), rows_c):
        tl.set_color(H.text_color(H.LANTHANIDE) if r[1] else H.INK)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(yc[-1] + 0.6, -0.6)
    ax.set_xlim(-10, 46)
    ax.set_xticks([-10, 0, 10, 20, 30, 40])
    H.vgrid(ax)
    ax.axvline(0, color=H.INK2, lw=0.8, zorder=1)
    ax.set_xlabel("Advantage over Bi, same pair (%)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], **ANALYTIC_MK), Line2D([], [], **FIRST_MK)]
    hs += [Line2D([], [], ls="none", marker=mk, ms=H.MS - 0.2, mfc=H.MC, mec="white", mew=0.4)
           for mk in MC_EST.values()]
    ax.legend(hs, ["Narrow beam (analytic)", "First-order (model only)", "MC narrow", "MC fluorescence-incl.",
                   "MC broad"], loc="lower left", bbox_to_anchor=(-0.45, 1.13), ncol=3, handletextpad=0.2,
              columnspacing=0.9, borderaxespad=0, fontsize=H.FS_ANNOT)
    fig.text(186 / H.DOUBLE, (11 + 33 + 0.8) / fig._mm[1],
             "MC filled: like-for-like Pb target; open: fixed target; bars ± 1.96 SE, statistical only",
             ha="right", va="bottom", fontsize=H.FS_ANNOT, color=H.INK2)
    n.add("Fig4c", "1.96", "def:mc_ci_z", "error-bar key (± 1.96 SE)")
    n.add("Fig4c", "0", "def:zero", "x = 0 line (Bi as light as the compound)")
    H.panel_letter(ax, "c", x_mm=89.0, dy_mm=12.5)
    META["Fig4"] = {"display_a": "narrow beam (filled, entity colour) and first-order corrected (open grey); "
                                 "elemental-blend line (narrow beam, H180); ambient-tier row (H214)",
                    "families_level": "ln (single lanthanide)", "n_families": len(fams),
                    "b": "elemental blend over the best textbook material = (1 + adv_narrow)/(1 + blend_vs_B4) - 1; "
                         "carets at the simulated variants 3, 13, 22",
                    "c": "rows = MC pairs at RQR5 (v3), RQR6 (v13, case c7: Ce, Pr, blend, oxide), RQR8 (v22); "
                         "like-for-like filled (H189-H194, H260/H261/H264/H265/H268/H269/H272/H273), fixed target "
                         "open (H262/H263/...); analytic narrow and first-order rows fixed-target",
                    "data": ["data/phase4b/fig4a_families_v13.csv", "fig4a_textbook_v13.csv",
                             "fig4b_advantage_by_variant.csv", "fig4b_mc_points.csv", "data/phase4a/mc/metrics.csv",
                             "data/phase4b/revision/e2_blend_by_variant.csv",
                             "data/phase4b/revision/e3_like_for_like.csv",
                             "data/phase4b/revision/e5_amb_lanthanide.csv", "data/phase4b/mc_rev/metrics.csv"]}
    _save(fig, "Fig4")


# =============================================================================================
# Graphical abstract (Elsevier: >= 1328 x 531 px, readable at that size)
# =============================================================================================
GA_W_MM, GA_H_MM = 168.656, 67.462        # 6.64 x 2.656 in: 1328 x 531 px at 200 dpi, 2656 x 1062 at 400 dpi


def graphical_abstract():
    n = _num()
    fig = H.figure(GA_W_MM, GA_H_MM)
    big, mid = 13.0, 10.0
    # left: the two messages
    fig.text(5 / GA_W_MM, 0.80, "Lead-free, stable and", fontsize=big, color=H.INK, fontweight="bold",
             va="baseline")
    fig.text(5 / GA_W_MM, 0.80 - 0.105, "observed are free.", fontsize=big, color=H.INK, fontweight="bold",
             va="baseline")
    y2 = 0.42
    fig.text(5 / GA_W_MM, y2, "Transparency is paid", fontsize=big, color=H.INK, fontweight="bold", va="baseline")
    t1 = fig.text(5 / GA_W_MM, y2 - 0.105, "in ", fontsize=big, color=H.INK, fontweight="bold", va="baseline")
    t2 = fig.text(0, 0, "space", fontsize=big, color=H.text_color(H.THICKNESS), fontweight="bold", va="baseline")
    t3 = fig.text(0, 0, ", not in ", fontsize=big, color=H.INK, fontweight="bold", va="baseline")
    t4 = fig.text(0, 0, "mass", fontsize=big, color=H.text_color(H.AREAL), fontweight="bold", va="baseline")
    t5 = fig.text(0, 0, ".", fontsize=big, color=H.INK, fontweight="bold", va="baseline")
    fig.set_dpi(400)                                      # measure at the export resolution
    r = fig.canvas.get_renderer()
    x = t1.get_window_extent(r).x1
    yb = t1.get_position()[1]
    W = fig.bbox.width
    for tt in (t2, t3, t4, t5):
        tt.set_position((x / W, yb))
        x = tt.get_window_extent(r).x1
    fig.text(5 / GA_W_MM, 0.08, "Price of each requirement inside the practical set, Cs-137 (narrow beam)",
             fontsize=8.0, color=H.INK2, va="baseline")
    # right: paired bars
    ax = H.add_axes_mm(fig, 104, 12, 62, 48)
    rows = [("Lead-free", "H03", "H04"), ("Stable", "H05", "H06"), ("Observed", "H07", "H08"),
            ("Transparent", "H09", "H10"), ("Scarcity proxy", "H11", "H12")]
    for i, (name, ht, hm) in enumerate(rows):
        vt, vm = n.value(ht), n.value(hm)
        y = i
        ax.barh(y - 0.19, vt, height=0.34, color=H.THICKNESS, lw=0)
        ax.barh(y + 0.19, vm, height=0.34, color=H.AREAL, lw=0)
        ax.text(-4, y, name, ha="right", va="center", fontsize=mid, color=H.INK)
        if vt == 0 and vm == 0:
            st = n.fmt("GraphicalAbstract", ht, 0, f"{name}: thickness price")
            n.fmt("GraphicalAbstract", hm, 0, f"{name}: areal-mass price")
            ax.text(3, y, f"{st} %", ha="left", va="center", fontsize=mid, color=H.INK2)
        else:
            main = name == "Transparent"
            st = n.fmt("GraphicalAbstract", ht, 0, f"{name}, thickness price")
            sm = n.fmt("GraphicalAbstract", hm, 1 if vm < 10 else 0, f"{name}, areal-mass price")   # 2 s.f.
            ax.text(vt + 3, y - 0.19, f"+{st} %" + (" thickness" if main else ""), ha="left", va="center",
                    fontsize=mid, color=H.text_color(H.THICKNESS), fontweight="bold" if main else "normal")
            ax.text(vm + 3, y + 0.19, f"+{sm} %" + (" mass" if main else ""), ha="left", va="center",
                    fontsize=mid, color=H.text_color(H.AREAL), fontweight="bold" if main else "normal")
    ax.axvline(0, color=H.INK2, lw=0.8)
    ax.set_xlim(0, 330)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.axis("off")
    files = H.export(fig, "GraphicalAbstract", OUT, formats=("pdf", "tif", "png"), exact_px=(2656, 1062))
    from PIL import Image
    with Image.open(OUT / "GraphicalAbstract.png") as im:
        META["GraphicalAbstract"] = {"png_px": list(im.size), "size_mm": [GA_W_MM, GA_H_MM],
                                     "files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in files],
                                     "min_font_pt_artists": H.min_font_size(fig)}
    import matplotlib.pyplot as plt
    plt.close(fig)


# =============================================================================================
# Supplementary figures
# =============================================================================================
def figS1():
    """COD vs MP density bias by class (stage 3 Fig. 7 restyled; data/phase3/D/matches.parquet)."""
    from shieldmap.analysis3 import cod
    from shieldmap.analysis3 import common as c3
    n = _num()
    m = pd.read_parquet(D3 / "D" / "matches.parquet")
    h = cod.headline(m)
    if abs(float(h.delta.median()) * 100 - n.value("H87")) > 1e-9:
        raise ValueError("FigS1: median delta differs from H87")
    el = np.asarray(cod._phase1_elemental(), float)
    rows = [("Elements (Phase 1, measured)", el, "ref"), ("All compounds", h.delta.to_numpy(), "all")]
    rows += [(cod.CLASS_LABEL[c].capitalize() if c != "3d_magnetic" else "3d-magnetic",
              h.loc[h[f"is_{c}"], "delta"].to_numpy(), "cls") for c in c3.REPORT_CLASSES]
    def _bin(nm, lo, hi):
        if lo is None:
            return "0 (on hull)"
        if hi is None:
            return f"> {lo:g}"
        return f"{lo:g}–{hi:g}"
    erows = [(_bin(nm, lo, hi), h.loc[h.ehull_bin == nm, "delta"].to_numpy(), "cls") for nm, lo, hi in cod.EHULL_BINS]
    _num().add("FigS1b", "0/0.025/0.05/0.1", "def:ehull_bins", "e_hull bin labels (Phase 3 cod.EHULL_BINS)")
    allx = [100 * np.asarray(r[1], float) for r in rows + erows]
    XL = (float(np.floor(min(np.percentile(x, 5) for x in allx) - 1)),
          float(np.ceil(max(np.percentile(x, 95) for x in allx) + 1)))
    fig = H.figure(H.DOUBLE, 66)
    axa = H.add_axes_mm(fig, 40, 10, 66, 50)
    axb = H.add_axes_mm(fig, 143, 10, 43, 50)
    col = {"ref": H.TEXTBOOK, "all": H.DENSITY, "cls": H.DENSITY}

    def draw(ax, data):
        ys = np.arange(len(data))[::-1]
        for y, (lab, x, k) in zip(ys, data):
            x = 100 * np.asarray(x, float)
            p5, p25, p50, p75, p95 = np.percentile(x, [5, 25, 50, 75, 95])
            c = col[k]
            ax.plot([p5, p95], [y, y], color=c, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((p25, y - 0.28), p75 - p25, 0.56, fc=c,
                                                                     alpha=0.35 if k != "all" else 0.55, lw=0,
                                                                     zorder=2))
            ax.plot([p50, p50], [y - 0.32, y + 0.32], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in data], fontsize=H.FS_ANNOT)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=H.INK2, lw=H.LW_REF, zorder=1)
        ax.set_xlim(*XL)
        ax.set_ylim(-0.7, len(data) - 0.3)
        ax.set_xlabel("δ = ρ$_{MP}$ / ρ$_{ref}$ − 1 (%)", fontsize=H.FS_LABEL)
        H.vgrid(ax)
        return ys
    ys = draw(axa, rows)
    axa.axhline(ys[0] - 0.5, color=H.SPINE, lw=0.5)
    med = n.fmt("FigS1a", "H87", 1, "median delta, all compounds")
    p95 = float(np.percentile(100 * h.delta, 95))
    axa.annotate(f"median {med} %", xy=(p95, ys[1]), xytext=(4, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=H.FS_ANNOT, color=H.text_color(H.DENSITY))
    draw(axb, erows)
    axb.set_ylabel("Energy above hull (eV/atom)", labelpad=4)
    H.panel_letter(axa, "a", x_mm=1.0)
    H.panel_letter(axb, "b", x_mm=112.0)
    META["FigS1"] = {"data": "data/phase3/D/matches.parquet (cod.headline) + Phase 1 elemental distribution",
                     "xlim": XL}
    _save(fig, "FigS1")


def _heat_cmap(color):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("w2c", ["#FFFFFF", color])
    cm.set_bad("white")
    return cm


def figS2(thresh: float = 0.02):
    """Element-exclusion filter prices inside B4 (B4-internal; figures4.fig4s_load_bearing_b4internal data)."""
    from matplotlib.colors import PowerNorm
    from shieldmap.analysis3 import common as c3
    from shieldmap.analysis4 import common
    b = pd.read_parquet(common.P4 / "S" / "bands.parquet", columns=["label", "basis", "density", "energy_keV",
                                                                     "price", "reference", "artefact_def",
                                                                     "scenario"])
    b = b[(b.scenario == "MONO") & (b.reference == "B4-internal") & (b.artefact_def == common.PRIMARY_ARTEFACT)
          & b.label.astype(str).str.startswith("EXCL(")].copy()
    b["element"] = b.label.astype(str).str.slice(5).str.split(")").str[0]
    b["basis"] = b.basis.astype(str)
    b["density"] = b.density.astype(str)
    b = b[((b.basis == "thickness") & (b.density == "measured")) | (b.basis == "areal")]
    zmap = dict(zip(c3.elements_table()["symbol"], c3.elements_table()["Z"]))
    mx = b.groupby("element").price.max()
    els = sorted(mx.index[mx >= thresh].tolist(), key=lambda s: zmap.get(s, 999))
    E = np.sort(b.energy_keV.unique())
    edges = np.concatenate([[E[0]], np.sqrt(E[1:] * E[:-1]), [E[-1]]])
    hgt = 2.7 * len(els)                                  # 2.7 mm per element row (6 pt labels)
    fig = H.figure(H.DOUBLE, hgt + 22)
    W = (H.DOUBLE - 12 - 6 - 2 * 18) / 2
    axs = [H.add_axes_mm(fig, 12 + k * (W + 18), 12, W, hgt) for k in range(2)]
    nh3 = h5no_range("B4")
    for k, (ax, basis, col, lab) in enumerate(zip(axs, ("thickness", "areal"), (H.THICKNESS, H.AREAL),
                                                  ("Thickness price (%), measured density",
                                                   "Areal-mass price (%)"))):
        z = b[b.basis == basis].pivot(index="element", columns="energy_keV", values="price").reindex(index=els,
                                                                                                    columns=E)
        zz = np.ma.masked_where(~(100 * z.to_numpy() > 0), 100 * z.to_numpy())
        vmax = float(np.nanmax(zz))
        im = ax.pcolormesh(edges, np.arange(len(els) + 1), zz, cmap=_heat_cmap(col),
                           norm=PowerNorm(0.5, vmin=0, vmax=vmax), shading="flat", rasterized=True)
        H.log_energy_axis(ax)
        ax.set_yticks(np.arange(len(els)) + 0.5)
        ax.set_yticklabels(els, fontsize=H.FS_TICK)
        ax.set_ylim(len(els), 0)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.spines["left"].set_visible(False)
        H.ref_markers(ax, label=(k == 0), zorder=3)          # above the heat map; labels on panel a only
        if basis == "areal" and nh3:
            # half a grid step either side of the first/last affected single line
            H.region_hatch(ax, nh3[0] / 1.015, nh3[1] * 1.015, "NH$_3$·H$_2$O\nfrontier\n(primary set)",
                           y_label=0.5, fontsize=H.FS_CHEM)
            META.setdefault("FigS2", {})["nh3_range_keV"] = list(nh3)
        bb = ax.get_position()
        cax = fig.add_axes((bb.x1 + 2 / H.DOUBLE, bb.y0, 2.2 / H.DOUBLE, bb.height))
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=H.FS_TICK, length=1.5)
        cb.set_label(lab, fontsize=H.FS_ANNOT)
        H.panel_letter(ax, "ab"[k], dx_mm=8.0)
    META.setdefault("FigS2", {}).update({"elements": els, "threshold": thresh, "data": "data/phase4a/S/bands.parquet (B4-internal, "
                     "EXCL(el)|B4, phase4a artefact flag)", "scale": "square-root colour scale, white = 0"})
    _save(fig, "FigS2")


FIGS3_PANELS = (("PB|F0", "Lead-free"), ("STAB(0.05)|F0", "Stable"), ("EXP|F0", "Observed"),
                ("RAD|F0", "Non-radioactive"), ("ABUND(1)|F0", "Scarcity proxy"), ("B4|F0", "B4 (all four)"),
                ("B6|F0", "B6 = B4 + scarcity proxy"), ("GAP(3.1)|F0", "Transparent"),
                ("B5|F0", "B5 = B4 + transparent"))
FIGS3_GROUPS = (("PB|F0", "STAB(0.05)|F0", "EXP|F0"), ("RAD|F0", "ABUND(1)|F0", "B4|F0", "B6|F0"),
                ("GAP(3.1)|F0", "B5|F0"))


def figS3():
    """Headroom prices against the unconstrained frontier F0, including RAD (stage 4a Fig. 2s restyled)."""
    from shieldmap.analysis4 import figures4 as f4
    b = f4.fig2_inputs("headroom")
    b = b[((b.basis == "thickness") & (b.density == "measured")) | (b.basis == "areal")]
    fig = H.figure(H.DOUBLE, 150)
    L, R, gx, gy = 14.0, 4.0, 7.0, 12.0
    w = (H.DOUBLE - L - R - 2 * gx) / 3
    hh = 36.0
    ylim = {}
    for grp in FIGS3_GROUPS:
        top = 100 * float(np.nanmax(b[b.label.isin(grp)][["band_hi", "price"]].to_numpy(float)))
        step = 50 if top > 100 else (10 if top > 20 else 5)
        for c in grp:
            ylim[c] = (-0.03 * top, np.ceil(top / step) * step, step)
    for k, (lab, name) in enumerate(FIGS3_PANELS):
        i, j = divmod(k, 3)
        ax = H.add_axes_mm(fig, L + j * (w + gx), 10 + (2 - i) * (hh + gy), w, hh)
        for basis, col in (("thickness", H.THICKNESS), ("areal", H.AREAL)):
            c = b[(b.label == lab) & (b.basis == basis)].sort_values("energy_keV")
            H.band(ax, c.energy_keV, 100 * c.band_lo, 100 * c.band_hi, col, hatched=(basis == "areal"))
            ax.plot(c.energy_keV, 100 * c.price, color=col, lw=H.LW_DATA, zorder=3)
        H.zero_line(ax)
        lo, hi, step = ylim[lab]
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.arange(0, hi + 0.1, step))
        H.log_energy_axis(ax, xlabel=(i == 2 and j == 1))
        H.hgrid(ax)
        H.ref_markers(ax, label=(k == 0))
        # areal values set by NH3.H2O in the primary set: where it is the frontier of the constrained set or of F0
        num_set = "F0&" + lab.split("|")[0]
        rngs = [r for r in (h5no_range(num_set), h5no_range("F0")) if r]
        if rngs:
            rng = (min(r[0] for r in rngs), max(r[1] for r in rngs))
            H.region_hatch(ax, rng[0] / 1.015, rng[1] * 1.015,
                           "NH$_3$·H$_2$O" if k == 0 else None, y_label=0.62, fontsize=H.FS_CHEM)
            META.setdefault("FigS3", {}).setdefault("nh3_range_keV", {})[lab] = list(rng)
        H.note(ax, 0.02, 0.97, name, ha="left", va="top", color=H.INK,
               fontsize=H.FS_LABEL, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        if j == 0:
            ax.set_ylabel("Headroom price (%)")
        H.panel_letter(ax, "abcdefghi"[k], dx_mm=11.5 if j == 0 else 5.0)
        if k == 7:
            c = b[(b.label == lab) & (b.basis == "thickness")].sort_values("energy_keV")
            y0 = float(np.interp(np.log(300), np.log(c.energy_keV), 100 * c.price))
            H.direct_label(ax, 300, y0, "Thickness", H.THICKNESS, dx=0, dy=4, ha="center", va="bottom")
            c = b[(b.label == lab) & (b.basis == "areal")].sort_values("energy_keV")
            y1 = float(np.interp(np.log(3000), np.log(c.energy_keV), 100 * c.price))
            H.direct_label(ax, 3000, y1, "Areal mass", H.AREAL, dx=0, dy=4, ha="center", va="bottom")
    META.setdefault("FigS3", {}).update({"data": "data/phase4a/S/bands.parquet (reference = headroom, phase4a flag)",
                                         "thickness": "measured-density point",
                                         "areal": "primary set (no H-excluded variant for F0)"})
    _save(fig, "FigS3")


def figS4():
    """Z-exponents behind the crossover (stage 4a Fig. 1s; data/phase3/L/z_exponents.parquet)."""
    from shieldmap.analysis3.landscape3 import Z_RANGE
    from shieldmap.analysis4 import figures4 as f4
    n = _num()
    i1 = f4.fig1_inputs()
    z = i1["zx"].sort_values("energy_keV")
    e_cross = f4._e_cross(i1["cross"])
    fig = H.figure(H.ONE_HALF, 62)
    axs = [H.add_axes_mm(fig, 12, 10, 53, 46), H.add_axes_mm(fig, 82, 10, 53, 46)]
    series = (("pe_Kactive", H.PHOTOELECTRIC, "-", "Photoelectric (K-active)"),
              ("pe", H.PHOTOELECTRIC, (0, (1, 1.2)), "Photoelectric (all Z)"),
              ("incoh", H.COMPTON, "-", "Compton"), ("coh", H.COHERENT, "-", "Coherent"),
              ("pair", H.PAIR, "-", "Pair"))
    s = n.fmt("FigS4", "H79", 0, "crossover energy label (both panels)")
    for k, (ax, kind) in enumerate(zip(axs, ("atom", "mass"))):
        for key, col, ls, lab in series:
            zz = z[z.energy_keV >= 2000] if key == "pair" else z
            ax.plot(zz.energy_keV, zz[f"n_{kind}_{key}"], color=col, ls=ls, lw=H.LW_DATA, zorder=3)
        ax.axvline(e_cross, color=H.INK2, lw=H.LW_REF, zorder=1)
        ax.annotate(f"{s} keV", xy=(e_cross, 0.08 if kind == "atom" else 0.02), xycoords=("data", "axes fraction"),
                    xytext=(2, 0),
                    textcoords="offset points", ha="left", va="bottom", fontsize=H.FS_ANNOT, color=H.INK)
        H.zero_line(ax)
        H.log_energy_axis(ax)
        H.hgrid(ax)
        ax.set_ylabel("Per-atom Z-exponent of σ" if kind == "atom" else "Per-mass Z-exponent of μ/ρ",
                      fontsize=H.FS_LABEL)
        H.panel_letter(ax, "ab"[k], dx_mm=10.5)
    axs[0].set_ylim(-0.3, 6.4)
    axs[1].set_ylim(-0.8, 4.9)
    # direct labels on both panels (review should-fix 10)
    def yat(key, e, kind):
        zz = z[z.energy_keV >= 2000] if key == "pair" else z
        return float(np.interp(np.log(e), np.log(zz.energy_keV), zz[f"n_{kind}_{key}"]))
    for ax, kind in zip(axs, ("atom", "mass")):
        H.direct_label(ax, 22, yat("pe_Kactive", 22, kind), "Photoelectric", H.PHOTOELECTRIC, dx=0, dy=6,
                       ha="left", va="bottom")
        H.direct_label(ax, 45, yat("pe", 45, kind), "all Z", H.PHOTOELECTRIC, dx=3, dy=-2, ha="left", va="top")
        up = kind == "atom"                            # in b the Compton curve sits just under the zero line
        H.direct_label(ax, 3000, yat("incoh", 3000, kind), "Compton", H.COMPTON, dx=0, dy=3 if up else -3,
                       ha="center", va="bottom" if up else "top")
        H.direct_label(ax, 60, yat("coh", 60, kind), "Coherent", H.COHERENT, dx=0, dy=-3, ha="center", va="top")
        H.direct_label(ax, 6000, yat("pair", 6000, kind), "Pair", H.PAIR, dx=0, dy=3, ha="center", va="bottom")
    META["FigS4"] = {"data": "data/phase3/L/z_exponents.parquet", "Z_range": list(Z_RANGE),
                     "dotted": "photoelectric, all Z (K-inactive included)"}
    _save(fig, "FigS4")


def figS5():
    """Polymorph density spread, DFT vs measured (H81; data/phase3/L/polymorph_cod.csv)."""
    n = _num()
    pt = pd.read_csv(D3 / "L" / "polymorph_cod.csv")
    if abs(float(pt.spread_dft.median()) - float(n.ht.loc["H81", "value"])) > 1e-12:
        raise ValueError("FigS5: median DFT spread differs from H81")
    fig = H.figure(H.SINGLE, 92)
    ax = H.add_axes_mm(fig, 13, 11, 72, 68)
    floor = 1e-4
    x = np.clip(pt.spread_cod.to_numpy(), floor, None)
    y = np.clip(pt.spread_dft.to_numpy(), floor, None)
    ok = pt.pairs_discordant.to_numpy() == 0
    bad = pt.max_abs_delta.to_numpy() > 0.2
    ax.scatter(x[ok & ~bad], y[ok & ~bad], s=5, color=H.DENSITY, lw=0, alpha=0.6, zorder=3,
               label="Ordering preserved")
    ax.scatter(x[~ok & ~bad], y[~ok & ~bad], s=6, color=H.INK2, lw=0, alpha=0.85, zorder=4,
               label="Ordering changed")
    ax.scatter(x[bad], y[bad], s=7, facecolor="none", edgecolor=H.TEXTBOOK, lw=0.5, zorder=4,
               label="A member with |δ| > 0.2")
    n.add("FigS5", "0.2", "def:abs_delta_screen", "legend: |delta| > 0.2")
    lim = (floor * 0.8, 3.5)
    ax.plot(lim, lim, color=H.INK2, lw=H.LW_REF, ls=(0, (3, 2)), zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    for axis in (ax.xaxis, ax.yaxis):
        H.plain_log_ticks(axis, [1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xlabel("Measured spread (COD), ρ$_{max}$/ρ$_{min}$ − 1", fontsize=H.FS_LABEL)
    ax.set_ylabel("DFT spread (MP), ρ$_{max}$/ρ$_{min}$ − 1", fontsize=H.FS_LABEL)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0, handletextpad=0.2,
              columnspacing=1.0, markerscale=1.6, fontsize=H.FS_ANNOT)
    H.note(ax, 0.97, 0.03, "values below 0.0001\ndrawn at 0.0001", ha="right", va="bottom",
           fontsize=H.FS_ANNOT, color=H.INK2)
    n.add("FigS5", "0.0001", "def:spread_floor", "plot floor note")
    META["FigS5"] = {"data": "data/phase3/L/polymorph_cod.csv", "floor": floor}
    _save(fig, "FigS5")


def figS6():
    """Cross-section library sensitivity near edges: xraylib vs XCOM (stage 4a xs) and ENDF vs XCOM."""
    from shieldmap.physics import mixture
    er = pd.read_csv(D4A / "xs" / "element_ratio_20_200keV.csv")
    mp = pd.read_csv(D4A / "xs" / "mono_prices.csv")
    endf = json.loads((D4A / "mc" / "endf_reference.json").read_text(encoding="utf-8"))
    fig = H.figure(H.DOUBLE, 64)
    axs = [H.add_axes_mm(fig, 14, 11, 48, 47), H.add_axes_mm(fig, 76, 11, 50, 47),
           H.add_axes_mm(fig, 141, 11, 45, 47)]
    # (a) element-level deviation
    ax = axs[0]
    for lib, kw, lab in (("xrl_kissel", dict(marker="o", ms=3.0, mfc=H.INK2, mec="none"), "xraylib (Kissel)"),
                         ("xrl_total", dict(marker="o", ms=3.0, mfc="none", mec=H.TEXTBOOK, mew=0.6),
                          "xraylib (CS_Total)")):
        s = er[er.lib == lib]
        ax.plot(s.Z, 100 * s.max_abs_rel, ls="none", label=lab, zorder=3, **kw)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 400)
    ax.set_xlim(0, 101)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("Atomic number Z")
    ax.set_ylabel("Max |xraylib/XCOM − 1|, 20–200 keV (%)", fontsize=H.FS_LABEL)
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1, 1, 10, 100]))
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    ax.yaxis.set_minor_locator(NullLocator())
    H.hgrid(ax)
    ax.legend(loc="lower right", handletextpad=0.1, fontsize=H.FS_ANNOT)
    H.panel_letter(ax, "a", dx_mm=12.0)
    # (b) prices near the W/Ta/Hf and Pb K edges: XCOM vs xraylib
    ax = axs[1]
    sel = mp[(mp.basis == "areal") & (mp.energy_keV >= 60) & (mp.energy_keV <= 100)]
    peaks = {}
    for lab in ("EXCL(Hf)|B4", "EXCL(Ta)|B4", "EXCL(W)|B4", "PB|B4"):
        for lib, ls in (("xcom", "-"), ("xrl_kissel", (0, (2.5, 1.5)))):
            s = sel[(sel.label == lab) & (sel.lib == lib)].sort_values("energy_keV")
            ax.plot(s.energy_keV, 100 * s.price, color=H.AREAL if lib == "xcom" else H.text_color(H.AREAL),
                    ls=ls, lw=H.LW_DATA, zorder=3 if lib == "xcom" else 4)
            if lib == "xcom":
                i = s.price.idxmax()
                peaks[lab] = (float(s.loc[i, "energy_keV"]), 100 * float(s.loc[i, "price"]))
    names = {"EXCL(Hf)|B4": "Hf excluded", "EXCL(Ta)|B4": "Ta excluded", "EXCL(W)|B4": "W excluded",
             "PB|B4": "lead-free"}
    # labels placed clear of the curves, with thin leaders to the XCOM peaks
    pos = {"EXCL(Hf)|B4": (61.0, 4.05, "left"), "EXCL(Ta)|B4": (67.0, 4.75, "left"),
           "EXCL(W)|B4": (73.5, 3.9, "left"), "PB|B4": (90.0, 3.4, "center")}
    for lab, (e, p) in peaks.items():
        x, y, ha = pos[lab]
        ax.annotate(names[lab], xy=(e, p), xytext=(x, y), textcoords="data", ha=ha, va="bottom",
                    fontsize=H.FS_ANNOT, color=H.INK,
                    arrowprops=dict(arrowstyle="-", color=H.INK2, lw=0.4, shrinkA=0, shrinkB=1.5))
    H.zero_line(ax)
    ax.set_xlim(60, 100)
    ax.set_ylim(-0.2, 5.4)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("Price, areal mass (%)")
    H.ref_markers(ax, which=("Pb K",), sides={"Pb K": -1})
    H.hgrid(ax)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=H.AREAL, lw=H.LW_DATA), Line2D([], [], color=H.text_color(H.AREAL), lw=0.9,
                                                                    ls=(0, (2.5, 1.5)))],
              ["XCOM", "xraylib (Kissel)"], loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0, fontsize=H.FS_ANNOT)
    H.panel_letter(ax, "b", dx_mm=10.0)
    # (c) ENDF/B-VIII.0 (Monte Carlo library) vs XCOM, mu/rho of the simulated materials
    ax = axs[2]
    mats = (("Pb", "-"), ("Bi", (0, (3, 1.5))), ("Tl", (0, (1, 1.2))), ("Ir", (0, (4, 1.5, 1, 1.5))))
    endf_ratio = {}
    for mat, ls in mats:
        e = np.asarray(endf["materials"][mat]["energies_keV"], float)
        mu = np.asarray(endf["materials"][mat]["mu_rho_cm2_g"], float)
        k = (e >= 60) & (e <= 100)
        w = mixture.fractions_from_formula(mat)
        xc = w @ mixture.element_mu_rho_at(e[k])["total_wcoh"]
        r = mu[k] / xc - 1
        endf_ratio[mat] = float(np.nanmax(np.abs(r)))
        # nodes where the two libraries put the K edge on different sides are off scale: the curve is broken
        # there and the node is marked by a triangle at the top, 1 mm right of the edge
        big = np.abs(r) > 0.08
        rr = np.where(big, np.nan, 100 * r)
        ax.plot(e[k], rr, color=H.MC, ls=ls, lw=1.0, zorder=3, label=mat)
        for ee in e[k][big]:
            ax.plot([ee + 0.8], [7.2], ls="none", marker="^", ms=3.0, mfc=H.MC, mec="none", zorder=5,
                    clip_on=False)
    H.zero_line(ax)
    ax.set_ylim(-8, 8)
    H.note(ax, 0.03, 0.80, "Triangle: K-edge node\noff scale", va="top", fontsize=H.FS_ANNOT, color=H.INK2)
    ax.set_xlim(60, 100)
    ax.set_xlabel("Photon energy (keV)")
    ax.set_ylabel("ENDF/B-VIII.0 / XCOM − 1, μ/ρ (%)", fontsize=H.FS_LABEL)
    H.ref_markers(ax, which=("Pb K",), sides={"Pb K": -1})
    H.hgrid(ax)
    ax.legend(loc="lower left", fontsize=H.FS_ANNOT, handlelength=2.2, ncol=4, columnspacing=0.8,
              handletextpad=0.3)
    H.panel_letter(ax, "c", dx_mm=12.0)
    META["FigS6"] = {"data": ["data/phase4a/xs/element_ratio_20_200keV.csv", "data/phase4a/xs/mono_prices.csv",
                              "data/phase4a/mc/endf_reference.json (vs XCOM via physics.mixture)"],
                     "endf_vs_xcom_max_abs_rel_60_100keV": endf_ratio}
    _save(fig, "FigS6")


def figS7():
    """Monte Carlo setup validation (Pb at Cs-137, case c6) and build-up factors (data/phase4a/mc/points.csv)."""
    n = _num()
    p = pd.read_csv(D4A / "mc" / "points.csv")
    fig = H.figure(H.DOUBLE, 64)
    axs = [H.add_axes_mm(fig, 14, 11, 47, 47), H.add_axes_mm(fig, 78, 11, 45, 47), H.add_axes_mm(fig, 139, 11, 47, 47)]
    pb = p[p.case_id == "c6_cs137_pb"].sort_values("t_cm")
    # (a) transmission vs thickness
    ax = axs[0]
    for col, ls, c, lab in (("T_tablec", "-", H.TEXTBOOK, "Analytic, XCOM (map)"),
                            ("T_endf_narrow", (0, (3, 1.5)), H.INK2, "Analytic, ENDF/B-VIII.0")):
        # the analytic transmissions at the simulated thicknesses, joined (no fit)
        ax.plot(pb.t_cm, pb[col], color=c, ls=ls, lw=H.LW_DATA, zorder=2, label=lab)
    ax.plot(pb.t_cm, pb.narrow, ls="none", zorder=4, label="MC, narrow beam", **MC_STYLE["narrow"])
    ax.set_yscale("log")
    H.plain_log_ticks(ax.yaxis, [0.001, 0.01, 0.1])
    ax.set_xlim(1.9, 6.2)
    ax.set_xlabel("Pb thickness (cm)")
    ax.set_ylabel("Air-kerma transmission")
    H.hgrid(ax)
    ax.legend(loc="lower left", fontsize=H.FS_ANNOT, handlelength=1.8)
    H.panel_letter(ax, "a", dx_mm=12.0)
    # (b) MC narrow / ENDF analytic - 1 with 1.96 SE
    ax = axs[1]
    # marker shape = compound (as in c); fill = energy (filled Cs-137, open Co-60); Pb black, others grey
    SHAPE = {"Pb": "o", "Ir": "s", "HfO2": "^", "BiOF": "D"}
    XOFF = {("c6_cs137_pb", "Pb"): -0.045, ("c1_cs137", "Ir"): -0.015, ("c1_cs137", "HfO2"): 0.0,
            ("c1_cs137", "BiOF"): 0.015, ("c2_co60", "Ir"): 0.03, ("c2_co60", "HfO2"): 0.045}
    for (case, mat), dx in XOFF.items():
        s = p[(p.case_id == case) & (p.material == mat)]
        r = 100 * (s.narrow / s.T_endf_narrow - 1)
        e = 100 * 1.96 * s.narrow_se / s.T_endf_narrow
        is_pb, co = mat == "Pb", case == "c2_co60"
        c = H.MC if is_pb else H.TEXTBOOK
        ax.errorbar(s.factor + dx, r, yerr=e, fmt="none", ecolor=H.MC if is_pb else H.TEXTBOOK_LIGHT,
                    elinewidth=0.6, capsize=0, zorder=3 if is_pb else 2)
        ax.plot(s.factor + dx, r, ls="none", marker=SHAPE[mat], ms=3.0, mfc="white" if co else c, mec=c, mew=0.6,
                zorder=4 if is_pb else 3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker=SHAPE[m], ms=3.0, mfc=H.MC if m == "Pb" else H.TEXTBOOK,
                 mec=H.MC if m == "Pb" else H.TEXTBOOK) for m in SHAPE]
    hs.append(Line2D([], [], ls="none", marker="s", ms=3.0, mfc="white", mec=H.TEXTBOOK, mew=0.6))
    ax.legend(hs, ["Pb", "Ir", "HfO$_2$", "BiOF", "open: Co-60"], loc="lower left", ncol=3, fontsize=H.FS_CHEM,
              handletextpad=0.1, columnspacing=0.8)
    H.zero_line(ax)
    ax.set_ylim(-1.3, 1.0)
    ax.set_xlim(0.5, 1.7)
    ax.set_xlabel("Thickness / headline thickness")
    ax.set_ylabel("MC narrow / analytic (ENDF) − 1 (%)")
    H.hgrid(ax)
    H.panel_letter(ax, "b", dx_mm=11.0)
    # (c) build-up factors B = broad / narrow
    ax = axs[2]
    curves = (("c6_cs137_pb", "Pb", "o", "-"), ("c1_cs137", "Ir", "s", "-"), ("c1_cs137", "BiOF", "D", "-"),
              ("c1_cs137", "HfO2", "^", "-"), ("c2_co60", "Ir", "s", (0, (3, 1.5))),
              ("c2_co60", "HfO2", "^", (0, (3, 1.5))))
    for case, mat, mk, ls in curves:
        s = p[(p.case_id == case) & (p.material == mat)].sort_values("factor")
        B = s.broad / s.narrow
        is_pb = mat == "Pb"
        ax.plot(s.factor, B, color=H.MC if is_pb else H.TEXTBOOK, ls=ls, lw=1.0, marker=mk, ms=3.0,
                mfc="white" if case == "c2_co60" else (H.MC if is_pb else H.TEXTBOOK),
                mec=H.MC if is_pb else H.TEXTBOOK, mew=0.6, zorder=4 if is_pb else 3)
        lab = H.chem(mat) + (", Co-60" if case == "c2_co60" else "")
        dy = {("c6_cs137_pb", "Pb"): -4, ("c1_cs137", "BiOF"): 0.5, ("c1_cs137", "Ir"): 4.5,
              ("c1_cs137", "HfO2"): -2, ("c2_co60", "Ir"): 2}.get((case, mat), 0)
        ax.annotate(lab, xy=(s.factor.iloc[-1], B.iloc[-1]), xytext=(3, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=H.FS_CHEM, color=H.INK if is_pb else H.INK2)
    s = p[(p.case_id == "c6_cs137_pb") & np.isclose(p.factor, 1.0)].iloc[0]
    b124 = float(s.broad / s.narrow)
    if abs(b124 - n.value("H124")) > 1e-6:
        raise ValueError("FigS7: Pb build-up at the headline thickness differs from H124")
    sb = n.fmt("FigS7c", "H124", 2, "Pb build-up factor at the headline thickness (Cs-137)")
    ax.annotate(f"Pb: B = {sb}", xy=(1.0, b124), xytext=(4, -5), textcoords="offset points", ha="left", va="top",
                fontsize=H.FS_ANNOT, color=H.INK)
    ax.axvline(1.0, color=H.REF, lw=H.LW_REF, ls=(0, (3, 2)), zorder=0)
    n.add("FigS7c", "1", "def:identity_one", "x = 1: headline thickness (vertical line)")
    ax.set_xlim(0.5, 2.05)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("Thickness / headline thickness")
    ax.set_ylabel("Build-up factor B (broad / narrow)")
    H.hgrid(ax)
    H.panel_letter(ax, "c", dx_mm=11.0)
    META["FigS7"] = {"data": ["data/phase4a/mc/points.csv", "data/phase4a/mc/digest.md"],
                     "B_Pb_headline": b124}
    _save(fig, "FigS7")


# =============================================================================================
def run(only: list[str] | None = None) -> dict:
    H.use()
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    funcs = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "Fig4": fig4, "GraphicalAbstract": graphical_abstract,
             "FigS1": figS1, "FigS2": figS2, "FigS3": figS3, "FigS4": figS4, "FigS5": figS5, "FigS6": figS6,
             "FigS7": figS7}
    for name, f in funcs.items():
        if only and name not in only:
            continue
        print(f"[fig] {name}", flush=True)
        f()
    if only:                                   # partial render (development): no register or checks
        return META
    _num().write()
    # checks: CVD / greyscale, fonts, contact sheet
    from shieldmap.figstyle import checks
    names = list(funcs)
    imgs = checks.cvd_images(OUT, REP / "cvd", names)
    checks.cvd_report(names, REP / "cvd_check.md", imgs)
    META["_font_check"] = checks.font_report([OUT / f"{n}.pdf" for n in names], REP / "font_check.md")
    checks.contact_sheet(OUT, names, REP / "contact_sheet.pdf", {n: META[n]["size_mm"] for n in names})
    (REP / "figures_meta.json").write_text(json.dumps(META, indent=1, default=str), encoding="utf-8")
    return META


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
