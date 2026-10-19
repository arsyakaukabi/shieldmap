"""Publication figures: main Fig. 1-4, graphical abstract, Supplementary S1-S3.

Entry point: ``python -m shieldmap.analyze4b --only figures`` -> ``run()``. Outputs go to ``paper/figures``
(PDF with embedded TrueType fonts, TIFF 600 dpi LZW, PNG 300 dpi), the printed-number register to
``paper/figures/figure_numbers.json``; the checks (CVD, greyscale, fonts, contact sheet) to
``reports/phase4c/figures`` (``shieldmap.figstyle.checks``).

Changes the content, not the style: every figure still goes through ``shieldmap.figstyle.helpers`` (H) and
``shieldmap.mplstyle``. Rules: at most 3 marker types per panel, only curve labels inside
plot areas, captions carry the qualifications. Data come from ``data/phase4b`` (incl. ``revision/`` and ``mc_rev/``)
and, read only, from stage 3 / stage 4a outputs. No new analysis.

The stage 4b figures that left the paper (old S3-S6, the decomposition panel of Fig. 3, the 36-variant and estimator
panels of Fig. 4) are kept, with their code, in ``docs/figures/`` (``docs/figures/render_docs_figures.py``).

Supplementary numbering: S1 = density bias by class (old S1), S2 = B4-internal
element-exclusion map (old S2), S3 = Monte Carlo consistency check and build-up (old S7).
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
REP = ROOT / "reports" / "phase4c" / "figures"
D4B = ROOT / "data" / "phase4b"
D4A = ROOT / "data" / "phase4a"
D3 = ROOT / "data" / "phase3"

MAIN = ("Fig1", "Fig2", "Fig3", "Fig4", "GraphicalAbstract")
SUPP = ("FigS1", "FigS2", "FigS3")
RETIRED = ("FigS4", "FigS5", "FigS6", "FigS7")       # Stage 4b outputs removed from paper/figures

# ---------------------------------------------------------------------------------------------
# printed-number register (CONVENTIONS.md tags)
# ---------------------------------------------------------------------------------------------
DEFS = {
    # names shared with paper/defs.yaml where the quantity exists there
    "pb_k_edge": (H.E_PB_K, "keV", "Pb K-edge energy (reference marker; = defs.yaml pb_k_edge)"),
    "cs137_line": (H.E_CS137, "keV", "Cs-137 line, marked at 661.7 keV (defs.yaml cs137_line = 661.657)"),
    "co60_mean": (H.E_CO60, "keV", "mean of the Co-60 lines (defs.yaml co60_lines), the single Co-60 marker"),
    "zero": (0.0, "%", "zero price / zero advantage reference"),
    "identity_one": (1.0, "", "transmission 1 at the front face (Fig. 1a); x = 1, headline thickness (Fig. S3)"),
    "gap_threshold": (3.1, "eV", "transparency threshold, graphical abstract label (= defs.yaml gap_threshold)"),
    "T_target": (0.01, "", "transmission target 0.01 (Fig. 1a T(x) curve; = defs.yaml T_target)"),
    "frontier": (0.0, "%", "B4 frontier (0 % above itself)"),
    "ehull_bins": (None, "eV/atom", "Phase 3 e_hull bins (not drawn in Phase 4c)"),
    "excl_display_threshold": (2.0, "%", "Fig. S2 shows elements whose exclusion price reaches >= 2 % somewhere"),
    "amb_icsd_min": (3.0, "ICSD entries", "fully oxidised filter: at least 3 ICSD entries (revision E5)"),
    "mc_ci_z": (1.96, "", "Monte Carlo statistical interval ± 1.96 SE (= defs.yaml mc_ci_z)"),
}


class Numbers:
    """Every number printed inside a figure, with its source tag, checked against headline_table_v3.csv."""

    def __init__(self):
        self.ht = pd.read_csv(D4B / "headline_table_v3.csv").set_index("id")
        self.rows: dict[str, list[dict]] = {}
        self.plotted_untagged: list[dict] = []

    def value(self, tag: str) -> float:
        if tag.startswith(("def:", "count:")):
            # def:<name> from DEFS (else paper/defs.yaml), count:<name> from paper/defs.yaml; a trailing ":pct"
            # prints the quantity as a per cent (graphical abstract: T = 0.01 printed as "T = 1 %")
            kind, _, name = tag.partition(":")
            name, _, conv = name.partition(":")
            v = DEFS[name][0] if kind == "def" and name in DEFS else _paper_def(name)
            return float(v) * (100.0 if conv == "pct" else 1.0)
        hid, _, kind = tag.partition(":")
        r = self.ht.loc[hid]
        pct = r.unit == "%"                     # as paper/check_numbers.py
        if kind == "":
            return float(r.value_pct if pct else r.value)
        if kind in ("lo", "hi"):
            return float(r[f"band_{kind}_pct"])
        if kind == "delta":
            return float(r.delta_abs)
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
        q = self.value(tag) if not (tag.startswith("def:") and DEFS.get(tag[4:], (0.0,))[0] is None) else None
        ok = True if q is None else self.matches(printed, q)
        self.rows.setdefault(fig, []).append({"printed": printed, "tag": tag, "where": where,
                                              "source_value": q, "match": ok})
        if not ok:
            raise ValueError(f"{fig}: printed {printed} does not match {tag} = {q}")

    def write(self):
        out = {"note": "Every number printed inside a figure (annotations, labels), with its source tag "
                       "(paper/CONVENTIONS.md). Tick labels and axis limits are not listed, except the T(x) "
                       "tick labels of Fig. 1a. Checked by shieldmap.figstyle.figures.Numbers (half-up rounding "
                       "or two significant figures). Phase 4c set: Fig1-4, GraphicalAbstract, FigS1-S3.",
               "defs": {k: {"value": v[0], "unit": v[1], "meaning": v[2]} for k, v in DEFS.items()},
               "figures": self.rows,
               "plotted_untagged": self.plotted_untagged}
        (OUT / "figure_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


NUM: Numbers | None = None
META: dict = {}
_PAPER_DEFS: dict | None = None


def _paper_def(name: str) -> float:
    """A scalar from paper/defs.yaml (the same file paper/check_numbers.py resolves def:/count: tags against)."""
    global _PAPER_DEFS
    if _PAPER_DEFS is None:
        import yaml
        _PAPER_DEFS = yaml.safe_load((paths.ROOT / "paper" / "defs.yaml").read_text(encoding="utf-8")) or {}
    v = _PAPER_DEFS[name]
    v = v.get("value") if isinstance(v, dict) else v
    return float(v)


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


def _overlay(fig):
    """A full-figure axes in mm coordinates (x right, y up), no frame: for illustrations (circles stay round)."""
    W, Hh = fig._mm
    ax = fig.add_axes((0, 0, 1, 1), zorder=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, Hh)
    ax.axis("off")
    return ax


# =============================================================================================
# illustration primitives (matplotlib patches only; no external images)
# =============================================================================================
ATOM_HEAVY = "#7F7F7F"       # high-Z atom (darker tint, larger radius)
ATOM_LIGHT = "#C9CED3"       # low-Z atom (lighter tint, smaller radius)
SLAB_FILL = H.BOX_FILL
BEAM = H.ARROW


def _arrow(ax, x0, y0, x1, y1, color=BEAM, lw=0.6, ms=5.0, **kw):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, shrinkA=0, shrinkB=0, **kw))


def _lattice(ax, x0, y0, w, h, pitch, r_heavy, r_light, heavy_every=2, heavy_frac=1.0):
    """A slab drawn as a lattice of circles whose radius scales with Z: heavy (large, dark) on a checkerboard,
    light (small, pale) in between. ``heavy_frac`` < 1 replaces part of the heavy sites by medium atoms."""
    from matplotlib.patches import Circle, FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=SLAB_FILL, ec="none",
                                zorder=1))
    nx = max(1, int(round(w / pitch)))
    ny = max(1, int(round(h / pitch)))
    px, py = w / nx, h / ny
    for i in range(nx):
        for j in range(ny):
            cx, cy = x0 + (i + 0.5) * px, y0 + (j + 0.5) * py
            heavy = (i + j) % heavy_every == 0
            if heavy and heavy_frac < 1.0 and ((i * 7 + j * 3) % 10) / 10.0 >= heavy_frac:
                r, c = 0.5 * (r_heavy + r_light), "#A3A9AF"           # medium-Z atom (third size)
            elif heavy:
                r, c = r_heavy, ATOM_HEAVY
            else:
                r, c = r_light, ATOM_LIGHT
            ax.add_patch(Circle((cx, cy), r, fc=c, ec="none", zorder=2))


def _ruler(ax, x, y, w, h, color):
    """Ruler icon: an outlined bar with tick marks along its top edge."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=color, lw=0.7, zorder=3))
    n = 6
    for k in range(1, n):
        xx = x + k * w / n
        ax.plot([xx, xx], [y + h, y + h - (0.55 if k % 2 else 0.35) * h], color=color, lw=0.5, zorder=4,
                solid_capstyle="butt")


def _weight(ax, x, y, w, h, color):
    """Weight icon: a filled trapezoid with a ring handle on top."""
    from matplotlib.patches import Circle, Polygon
    body = h * 0.72
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + 0.78 * w, y + body), (x + 0.22 * w, y + body)], closed=True,
                         fc=color, ec="none", zorder=3))
    ax.add_patch(Circle((x + w / 2, y + body + 0.18 * h), 0.2 * h, fc="none", ec=color, lw=0.8, zorder=3))


def _bracket(ax, x0, x1, y, depth, color, lw=0.7, up=True):
    """Square bracket spanning x0..x1 at height y, its ends turned towards the measured object."""
    s = 1 if up else -1
    ax.plot([x0, x0, x1, x1], [y + s * depth, y, y, y + s * depth], color=color, lw=lw, zorder=4,
            solid_capstyle="butt", solid_joinstyle="miter")


# =============================================================================================
# Fig. 1: (a) physics, (b) the price of a requirement, (c) composition or density
# =============================================================================================
def fig1():
    from shieldmap.analysis4 import figures4 as f4
    n = _num()
    W, HH = H.DOUBLE, 62.0
    fig = H.figure(W, HH)
    ov = _overlay(fig)
    lab = dict(fontsize=H.FS_ANNOT, color=H.INK)
    # ---------------- (a) physics: beam, crystal, T(x)
    sx0, sw, sy0, sh = 24.0, 18.0, 30.0, 22.0
    _lattice(ov, sx0, sy0, sw, sh, pitch=2.6, r_heavy=0.95, r_light=0.48, heavy_frac=0.6)
    for yy in np.linspace(sy0 + 3.0, sy0 + sh - 3.0, 5):
        _arrow(ov, 5.0, yy, sx0 - 1.2, yy)
    _arrow(ov, sx0 + sw + 1.2, sy0 + sh / 2, sx0 + sw + 9.0, sy0 + sh / 2, color=H.REF, lw=0.5)
    ov.text(13.0, sy0 + sh + 1.6, "photons", ha="center", va="bottom", **lab)
    ov.text(sx0 + sw / 2, sy0 + sh + 1.6, "crystal", ha="center", va="bottom", **lab)
    # T(x): x = 0 at the front face, x = 1 at the back face = required thickness (T = 0.01)
    axT = H.add_axes_mm(fig, sx0, 11.0, sw * 1.3, 14.0)
    xs = np.linspace(0, 1.3, 200)
    axT.plot(xs[xs <= 1.0], 0.01 ** xs[xs <= 1.0], color=H.INK, lw=H.LW_DATA, zorder=3)
    axT.plot(xs[xs >= 1.0], 0.01 ** xs[xs >= 1.0], color=H.INK, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    axT.plot([1.0], [0.01], ls="none", marker="o", ms=H.MS - 0.5, mfc=H.INK, mec="white", mew=0.4, zorder=4)
    axT.set_yscale("log")
    axT.set_ylim(0.004, 1.6)
    axT.set_xlim(0, 1.3)
    s1 = "1"
    s001 = "0.01"
    n.add("Fig1a", s1, "def:identity_one", "T(x) tick: transmission 1 at the front face")
    n.add("Fig1a", s001, "def:T_target", "T(x) tick: transmission 0.01 at the required thickness")
    H.plain_log_ticks(axT.yaxis, [0.01, 1])
    axT.set_xticks([])
    axT.set_ylabel("T(x)", labelpad=1)
    axT.axhline(0.01, color=H.REF, lw=H.LW_REF, ls=(0, (3, 2)), zorder=1)
    # guides: slab faces down to the T(x) axis; the required thickness as a dimension arrow
    for xx in (sx0, sx0 + sw):
        ov.plot([xx, xx], [sy0 - 0.6, 6.4], color=H.REF, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    ov.annotate("", xy=(sx0, 7.2), xytext=(sx0 + sw, 7.2),
                arrowprops=dict(arrowstyle="<|-|>", mutation_scale=4.5, color=H.INK, lw=0.6, shrinkA=0, shrinkB=0))
    ov.text(sx0 + sw / 2, 5.6, "required thickness", ha="center", va="top", **lab)
    fig.text(0.5 / W, (HH - 1.0) / HH, "a", fontsize=H.FS_LETTER, fontweight="bold", va="top", color=H.INK)
    # ---------------- (b) the price of a requirement (proportions = transparency at Cs-137, H09 / H10)
    p_thk, p_mass = n.value("H09") / 100, n.value("H10") / 100
    x0b = 66.0
    w0 = 7.5
    w1 = w0 * (1 + p_thk)
    bL, bR = x0b + 10.0, x0b + 28.0
    by0, bh = 30.0, 22.0
    _lattice(ov, bL, by0, w0, bh, pitch=2.5, r_heavy=0.95, r_light=0.48)
    _lattice(ov, bR, by0, w1, bh, pitch=2.5, r_heavy=0.62, r_light=0.42, heavy_frac=0.0)
    for (xs_, xe) in ((x0b + 3.0, bL - 1.0), (bL + w0 + 3.0, bR - 1.0)):
        for yy in (by0 + 5.0, by0 + bh / 2, by0 + bh - 5.0):
            _arrow(ov, xs_, yy, xe, yy)
    ov.text(bL + w0 / 2, by0 + bh + 1.6, "frontier crystal", ha="center", va="bottom", **lab)
    ov.text(bR + w1 / 2, by0 + bh + 1.6, "frontier crystal\nwith the requirement", ha="center", va="bottom",
            linespacing=1.1, **lab)
    # where the best crystal would end, inside the thicker slab
    ov.plot([bR + w0, bR + w0], [by0, by0 + bh], color=H.INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    # thickness row (blue dimension lines) and areal-mass row (orange bars)
    ty, my = 25.0, 13.0
    tcol = H.text_color(H.THICKNESS)
    for x, w in ((bL, w0), (bR, w1)):
        ov.plot([x, x + w], [ty, ty], color=H.THICKNESS, lw=1.2, solid_capstyle="butt", zorder=3)
        for xx in (x, x + w):
            ov.plot([xx, xx], [ty - 1.0, ty + 1.0], color=H.THICKNESS, lw=0.7, zorder=3)
    m0 = 9.0
    from matplotlib.patches import Rectangle
    for x, w in ((bL, m0), (bR, m0 * (1 + p_mass))):
        ov.add_patch(Rectangle((x, my - 1.2), w, 2.4, fc=H.AREAL, ec="none", zorder=3))
    _bracket(ov, bR + w0, bR + w1, ty - 3.2, 1.2, H.INK, up=True)
    ov.text(bR + w0 + (w1 - w0) / 2, ty - 4.2, "price", ha="center", va="top", fontsize=H.FS_LABEL, color=H.INK,
            fontweight="bold")
    _ruler(ov, x0b + 1.0, ty - 1.0, 6.4, 2.0, H.THICKNESS)
    _weight(ov, x0b + 2.4, my - 1.7, 3.6, 3.8, H.AREAL)
    ov.text(bR + w1 + 1.5, ty, "thickness", ha="left", va="center", fontsize=H.FS_ANNOT, color=tcol)
    ov.text(bR + m0 * (1 + p_mass) + 1.5, my, "mass thickness", ha="left", va="center", fontsize=H.FS_ANNOT,
            color=H.text_color(H.AREAL))
    fig.text((x0b - 1.0) / W, (HH - 1.0) / HH, "b", fontsize=H.FS_LETTER, fontweight="bold", va="top", color=H.INK)
    # ---------------- (c) composition or density: two curves, one crossover line
    i1 = f4.fig1_inputs()
    dec, cross = i1["dec"], i1["cross"]
    ax = H.add_axes_mm(fig, 143, 10, 43, 44)
    d = dec[dec.subset == "all"]
    prim = d[(d.rule == "R2") & d.coherent].sort_values("energy_keV")
    xc = cross[(cross.subset == "all") & (cross.rule == "R2") & cross.coherent & (cross.direction == "comp->dens")]
    e_cross = float(xc.E_cross_keV.iloc[-1])
    if abs(e_cross - n.value("H79")) > 1e-6:
        raise ValueError("Fig1: crossing differs from H79")
    ax.axvline(e_cross, color=H.INK2, lw=H.LW_REF, zorder=1)
    for col, c in (("share_composition", H.COMPOSITION), ("share_density", H.DENSITY)):
        ax.plot(prim.energy_keV, prim[col], color=c, lw=H.LW_DATA, zorder=3,
                ls=H.DENSITY_LS if col == "share_density" else "-")
    H.log_energy_axis(ax)
    ax.set_ylim(-0.04, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Direct share of var(log μ)")

    def yat(col, e):
        return float(np.interp(np.log(e), np.log(prim.energy_keV), prim[col]))
    H.direct_label(ax, 22, yat("share_composition", 22), "Composition", H.COMPOSITION, dx=0, dy=9, ha="left",
                   va="bottom")
    H.direct_label(ax, 1700, 0.70, "Density", H.DENSITY, dx=0, dy=0, ha="center", va="center")   # inside the arch
    s = n.fmt("Fig1c", "H79", 0, "crossover energy label")
    ax.annotate(f"{s} keV", xy=(e_cross, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -1),
                textcoords="offset points", ha="left", va="top", fontsize=H.FS_ANNOT, color=H.INK)
    fig.text((143 - 13.0) / W, (HH - 1.0) / HH, "c", fontsize=H.FS_LETTER, fontweight="bold", va="top", color=H.INK)
    META["Fig1"] = {"crossing_keV": e_cross, "b_proportions": {"thickness_ratio": 1 + p_thk,
                                                                "mass_ratio": 1 + p_mass, "source": "H09, H10"},
                    "data": "data/phase3/L (figures4.fig1_inputs); proportions of b from H09/H10",
                    "marker_types": {"a": 1, "b": 0, "c": 0},
                    "text_in_plot": {"a": ["photons", "crystal", "required thickness"],
                                     "b": ["frontier crystal", "frontier crystal with the requirement", "price",
                                           "thickness", "mass thickness"],
                                     "c": ["Composition", "Density", "235 keV"]}}
    _save(fig, "Fig1")


# =============================================================================================
# Fig. 2: prices vs energy - (a) the three free requirements, (b) transparency, (c) scarcity proxy
# =============================================================================================
def noh_from_kev() -> float:
    """Switch point of the areal curves: the lowest energy at which NH3.H2O is a reference."""
    from shieldmap.analysis4b import patches
    return float(patches.noh_from_kev())


def fig2_curve(d: pd.DataFrame, label: str, basis: str) -> pd.DataFrame:
    """Thickness: measured-density point. Areal: primary set below the switch point (800 keV), H-excluded
    variant at and above it. The switch is in the data and stated in the caption, not drawn."""
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


def _end_labels(ax, items, x_end, min_gap_pt=7.5):
    """Direct labels just right of the curves' right ends, nudged apart vertically when closer than
    ``min_gap_pt`` (positions in display space)."""
    tr = ax.transData
    pts = sorted(((tr.transform((x_end, y))[1], y, t, c) for y, t, c in items), key=lambda r: r[0])
    placed = []
    for k, (py, y, t, c) in enumerate(pts):
        if placed and py - placed[-1] < min_gap_pt * ax.figure.dpi / 72:
            py = placed[-1] + min_gap_pt * ax.figure.dpi / 72
        placed.append(py)
        dy = (py - tr.transform((x_end, y))[1]) * 72 / ax.figure.dpi
        H.direct_label(ax, x_end, y, t, c, dx=2.5, dy=dy, ha="left", va="center", annotation_clip=False)


def fig2():
    d = pd.read_csv(D4B / "fig2_prices.csv")
    W, HH = H.DOUBLE, 92.0
    fig = H.figure(W, HH)
    # ---------------- (a) compact strip: lead-free, stable, observed (both bases), 0-10 %
    axa = H.add_axes_mm(fig, 14, 70, 158, 16)
    for lab in ("PB|B4", "STAB|B4", "EXP|B4"):
        for basis, col in (("thickness", H.THICKNESS), ("areal", H.AREAL)):
            c = fig2_curve(d, lab, basis)
            axa.plot(c.energy_keV, 100 * c.price, color=col, lw=0.8, zorder=3 if basis == "thickness" else 2)
    axa.set_ylim(-0.6, 10)
    axa.set_yticks([0, 5, 10])
    H.log_energy_axis(axa, xlabel=False)
    H.hgrid(axa)
    axa.set_ylabel("Price (%)")
    # labels on the two features; the stable curves are 0 on both bases everywhere
    ce = fig2_curve(d, "EXP|B4", "thickness")
    i = ce.price.idxmax()
    H.direct_label(axa, ce.loc[i, "energy_keV"], 100 * ce.loc[i, "price"], "observed (thickness)", H.THICKNESS,
                   dx=-3, dy=0, ha="right", va="center")
    cp = fig2_curve(d, "PB|B4", "areal")
    i = cp.price.idxmax()
    H.direct_label(axa, cp.loc[i, "energy_keV"], 100 * cp.loc[i, "price"], "lead-free (mass thickness)", H.AREAL,
                   dx=4, dy=0, ha="left", va="center")
    H.direct_label(axa, 3000, 0.0, "stable (both bases)", H.INK, dx=0, dy=5.5, ha="center", va="bottom")
    H.panel_letter(axa, "a", dx_mm=11.5)
    # ---------------- (b) transparency, (c) scarcity proxy
    axs = [H.add_axes_mm(fig, 14, 11, 64, 46), H.add_axes_mm(fig, 108, 11, 64, 46)]
    for k, (ax, lab, title) in enumerate(zip(axs, ("GAP(3.1)|B4", "ABUND(1)|B4"),
                                              ("Transparency", "Scarcity proxy"))):
        ends = []
        top = 0.0
        for basis, col, name in (("thickness", H.THICKNESS, "Thickness"), ("areal", H.AREAL, "Mass thickness")):
            c = fig2_curve(d, lab, basis)
            H.band(ax, c.energy_keV, 100 * c.band_lo, 100 * c.band_hi, col)          # plain band, no hatching
            ax.plot(c.energy_keV, 100 * c.price, color=col, lw=H.LW_DATA, zorder=3)
            ends.append((100 * float(c.price.iloc[-1]), name, col))
            top = max(top, 100 * float(c[["band_hi", "price"]].to_numpy().max()))
        step = 50 if top > 100 else 10
        hi = np.ceil(top / step) * step
        ax.set_ylim(-0.03 * hi, hi)
        ax.set_yticks(np.arange(0, hi + 0.1, step))
        H.zero_line(ax)
        H.log_energy_axis(ax)
        H.hgrid(ax)
        H.ref_markers(ax, label=(k == 0))
        ax.set_ylabel("Price (%)")
        ax.set_title(title, loc="left", fontsize=H.FS_LABEL, color=H.INK, pad=3)
        fig.canvas.draw()
        _end_labels(ax, ends, H.E_HI)
        H.panel_letter(ax, "bc"[k], dx_mm=11.5, dy_mm=1.5)
    META["Fig2"] = {"data": "data/phase4b/fig2_prices.csv",
                    "areal_switch_keV": noh_from_kev(),
                    "marker_types": {"a": 0, "b": 0, "c": 0},
                    "text_in_plot": {"a": ["observed (thickness)", "lead-free (mass thickness)", "stable (both bases)"],
                                     "b": ["Pb K", "Cs-137", "Co-60", "Thickness", "Mass thickness"],
                                     "c": ["Thickness", "Mass thickness"]}}
    _save(fig, "Fig2")


# =============================================================================================
# Fig. 3: transparency is a space price (one panel)
# =============================================================================================
def fig3():
    g = pd.read_csv(D4B / "fig3_gap_b4internal.csv")
    mc = pd.read_csv(D4B / "fig3_mc_points.csv")
    m = g[g.scenario == "MONO"].sort_values("energy_keV")
    n = _num()
    fig = H.figure(H.ONE_HALF, 62)
    ax = H.add_axes_mm(fig, 13, 10, 104, 46)
    if (m.mass_variant == "noH").any() and abs(m[m.mass_variant == "noH"].energy_keV.min() - noh_from_kev()) > 1e-6:
        raise ValueError("Fig3: mass_variant switch differs from patches.noh_from_kev()")
    ends = []
    for col, c, name in (("P_thickness", H.THICKNESS, "Thickness"), ("P_mass", H.AREAL, "Mass thickness")):
        # the curves are Fig. 2b's: drawn thinner, with lighter bands, so the Monte Carlo points lead
        H.band(ax, m.energy_keV, 100 * m[f"{col}_lo"], 100 * m[f"{col}_hi"], c, alpha=0.10)
        ax.plot(m.energy_keV, 100 * m[col], color=c, lw=0.8, alpha=0.75, zorder=3)
        ends.append((100 * float(m[col].iloc[-1]), name, c))
    H.zero_line(ax)
    H.log_energy_axis(ax)
    ax.set_ylim(-6, 225)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_ylabel("Price of transparency (%)")
    H.hgrid(ax)
    H.ref_markers(ax)
    # Monte Carlo with build-up (broad-beam estimator) at Cs-137 and Co-60: H146, H147
    E = {"c1_cs137": (H.E_CS137, "H146"), "c2_co60": (H.E_CO60, "H147")}
    pts = []
    for case, (e, tag) in E.items():
        r = mc[(mc.case_id == case) & (mc.estimator == "broad")].iloc[0]
        if abs(100 * r.price - n.value(tag)) > 1e-9:
            raise ValueError(f"Fig3: MC broad {case} differs from {tag}")
        ax.plot(e, 100 * r.price, ls="none", marker="o", ms=H.MS + 2.2, mfc=H.MC, mec="white", mew=0.8, zorder=6)
        pts.append((e, 100 * r.price))
    ex = float(np.sqrt(pts[0][0] * pts[1][0]))
    ly = 172.0
    ax.text(ex, ly, "Monte Carlo, with build-up", ha="center", va="bottom", fontsize=H.FS_LABEL, color=H.INK,
            fontweight="bold",
            zorder=7, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    for e, yv in pts:                       # one label, a thin leader to each of the two points
        ax.annotate("", xy=(e, yv), xytext=(ex, ly - 1.5), textcoords="data",
                    arrowprops=dict(arrowstyle="-", color=H.INK, lw=0.6, shrinkA=0, shrinkB=4.0))
    fig.canvas.draw()
    _end_labels(ax, ends, H.E_HI)
    META["Fig3"] = {"data": ["data/phase4b/fig3_gap_b4internal.csv (scenario MONO)", "data/phase4b/fig3_mc_points.csv"],
                    "mc_shown": "broad (with build-up): H146, H147", "mc_label_anchor_keV": ex,
                    "marker_types": {"a": 1},
                    "text_in_plot": {"a": ["Pb K", "Cs-137", "Co-60", "Monte Carlo, with build-up", "Thickness",
                                           "Mass thickness"]}}
    _save(fig, "Fig3")


# =============================================================================================
# Fig. 4: from ideal crystals to usable materials - (a) families vs textbook, (b) the staircase
# =============================================================================================
TEXTBOOK10 = ("W", "Bi", "Sn", "Sb2O3", "BaSO4", "Gd2O3", "WO3", "Bi2O3", "PbO", "Pb")
DOT = dict(marker="o", ms=H.MS + 0.2, mec="white", mew=0.4, ls="none")


def _tb_label(f: str) -> str:
    return "β-Sn" if f == "Sn" else H.chem(f)


def staircase_data(n: Numbers) -> pd.DataFrame:
    """Fig. 4b: advantage over Bi = Bi/X - 1 on mass thickness, narrow beam (step 1) and Monte Carlo (step 2).

    Stage 4d: the x axis carries only the method,
    and every material is its own series.
    - reduced lanthanide compound per beam: step 1 MC narrow-beam estimator, step 2 fluorescence-inclusive, both
      like-for-like (RQR5 LaBi H193/H189; RQR6 Ce5AgSn3 mc_rev/metrics.csv / H260; RQR8 Pr3OsI3 H194/H191);
    - "RQR6 blend": the Sn/Ba/Gd blend, steps 1-2 (mc_rev/metrics.csv / H272);
    - "<beam> oxide": the fully oxidised lanthanide frontier crystal with >= 3 ICSD entries. Transported only at
      RQR6 (case c7): step 1 MC narrow (mc_rev/metrics.csv), step 2 fluorescence-inclusive (H268). At RQR5 and RQR8
      only its narrow-beam map value exists (H217, H219, given as X/Bi - 1; drawn as Bi/X - 1 = 1/(1 + v) - 1, the
      same quantity), step 1 only, flagged ``transported = False``."""
    lfl = pd.read_csv(D4B / "revision" / "e3_like_for_like.csv").set_index(["case_id", "estimator"])
    m7 = pd.read_csv(D4B / "mc_rev" / "metrics.csv")
    amb = pd.read_csv(D4B / "revision" / "e5_amb_lanthanide.csv")
    amb = amb[amb.tier == "AMB-ANION&ICSD3"].set_index("variant_id")

    def c7(mat, est):
        r = m7[(m7.metric == f"Bi/{mat} − 1") & (m7.target == "like-for-like") & (m7.estimator == est)].iloc[0]
        return float(r.value), float(r.se)

    def chk(v, tag):
        if abs(100 * v - n.value(tag)) > 1e-9:
            raise ValueError(f"Fig4b: value differs from {tag}")
        return v
    rows = []
    # RQR5 (variant 3)
    a = lfl.loc[("c3_rqr5", "narrow")]
    b = lfl.loc[("c3_rqr5", "fl")]
    ox = float(amb.loc[3, "vs_Bi"])
    chk(ox, "H217")
    rows += [("RQR5", 1, chk(a.adv_lfl, "H193"), a.adv_lfl_se, True, "H193", "LaBi"),
             ("RQR5", 2, chk(b.adv_lfl, "H189"), b.adv_lfl_se, True, "H189", "LaBi"),
             ("RQR5 oxide", 1, 1 / (1 + ox) - 1, np.nan, False, "H217", amb.loc[3, "leader"])]
    # RQR6 (variant 13, case c7)
    v1, s1 = c7("Ce5AgSn3", "narrow")
    v2, s2 = c7("Ce5AgSn3", "fl")
    o1, so1 = c7("Ba2SmSbO6", "narrow")
    o2, so2 = c7("Ba2SmSbO6", "fl")
    if amb.loc[13, "leader"] != "Ba2SmSbO6":
        raise ValueError("Fig4b: RQR6 fully oxidised leader is not the transported compound")
    if abs(100 * v1 - n.value("H169")) > 0.5:          # MC narrow vs the analytic map value (35.6 vs 35.3 %)
        raise ValueError("Fig4b: c7 narrow-beam MC value far from H169")
    ox13 = 1 / (1 + float(amb.loc[13, "vs_Bi"])) - 1   # analytic narrow-beam map value of the oxide (H215, X/Bi - 1)
    chk(float(amb.loc[13, "vs_Bi"]), "H215")
    if abs(100 * (o1 - ox13)) > 1.0:
        raise ValueError("Fig4b: c7 narrow-beam MC value of the oxide far from the map value H215")
    rows += [("RQR6", 1, v1, s1, True, "mc_rev/metrics.csv (untagged, not printed)", "Ce5AgSn3"),
             ("RQR6", 2, chk(v2, "H260"), s2, True, "H260", "Ce5AgSn3"),
             ("RQR6 oxide", 1, o1, so1, True, "mc_rev/metrics.csv (untagged, not printed)", "Ba2SmSbO6"),
             ("RQR6 oxide", 2, chk(o2, "H268"), so2, True, "H268", "Ba2SmSbO6")]
    # RQR8 (variant 22)
    a = lfl.loc[("c4_rqr8", "narrow")]
    b = lfl.loc[("c4_rqr8", "fl")]
    ox = float(amb.loc[22, "vs_Bi"])
    chk(ox, "H219")
    rows += [("RQR8", 1, chk(a.adv_lfl, "H194"), a.adv_lfl_se, True, "H194", "Pr3OsI3"),
             ("RQR8", 2, chk(b.adv_lfl, "H191"), b.adv_lfl_se, True, "H191", "Pr3OsI3"),
             ("RQR8 oxide", 1, 1 / (1 + ox) - 1, np.nan, False, "H219", amb.loc[22, "leader"])]
    # elemental blend at the reference beam: steps 1-2 only, drawn as a
    # dashed line without markers (no fourth marker type). Narrow: MC narrow like-for-like (mc_rev/metrics.csv,
    # untagged, not printed; analytic narrow H181); Monte Carlo: H272 (fl, like-for-like).
    b1, sb1 = c7("SnBaGd_blend", "narrow")
    b2, sb2 = c7("SnBaGd_blend", "fl")
    rows += [("RQR6 blend", 1, b1, sb1, True, "mc_rev/metrics.csv (untagged, not printed)", "Sn/Ba/Gd blend"),
             ("RQR6 blend", 2, chk(b2, "H272"), sb2, True, "H272", "Sn/Ba/Gd blend")]
    n.plotted_untagged = [
        {"figure": "Fig4b", "what": "RQR6 step 1: Bi/Ce5AgSn3 - 1, MC narrow-beam estimator, like-for-like Pb target",
         "value": v1, "source": "data/phase4b/mc_rev/metrics.csv: case_id c7_rqr6, target like-for-like, metric "
                                "'Bi/Ce5AgSn3 − 1', estimator narrow",
         "headline_row": "none for this estimator; analytic narrow-beam counterpart H169 (35.33 %, Bi above the B4 "
                         "frontier Ce5AgSn3 at v13); code checks |MC - H169| < 0.5 pp", "printed": False},
        {"figure": "Fig4b", "what": "RQR6 elemental blend step 1: Bi/blend - 1, MC narrow-beam estimator, like-for-like",
         "value": b1, "source": "data/phase4b/mc_rev/metrics.csv: case_id c7_rqr6, target like-for-like, metric "
                                "'Bi/SnBaGd_blend − 1', estimator narrow",
         "headline_row": "none for this estimator; analytic narrow-beam counterpart H181 (blend vs Bi as X/Bi - 1 = "
                         "-26.4 %, i.e. Bi/X - 1 = 35.9 %)", "printed": False},
        {"figure": "Fig4b", "what": "RQR6 fully oxidised compound, narrow beam: Bi/Ba2SmSbO6 - 1, MC narrow-beam "
                                    "estimator, like-for-like Pb target", "value": o1,
         "source": "data/phase4b/mc_rev/metrics.csv: case_id c7_rqr6, target like-for-like, metric "
                   "'Bi/Ba2SmSbO6 − 1', estimator narrow",
         "headline_row": "none for this estimator; analytic narrow-beam counterpart H215 (X/Bi - 1); code checks "
                         "|MC - map| < 1 pp", "printed": False},
        {"figure": "Fig4b", "what": "RQR5 and RQR8 fully oxidised compound, narrow beam only (open diamonds): "
                                    "Bi/X - 1 = 1/(1 + v) - 1 of the tagged X/Bi - 1 values",
         "value": [r[2] for r in rows if r[0] in ("RQR5 oxide", "RQR8 oxide")], "source": "H217, H219 "
                                    "(data/phase4b/revision/e5_amb_lanthanide.csv vs_Bi)",
         "headline_row": "H217, H219 (same quantity, inverted ratio)", "printed": False}]
    return pd.DataFrame(rows, columns=["beam", "step", "adv", "se", "transported", "source", "material"])


def fig4():
    fa = pd.read_csv(D4B / "fig4a_families_v13.csv")
    tb = pd.read_csv(D4B / "fig4a_textbook_v13.csv")
    n = _num()
    blend = pd.read_csv(D4B / "revision" / "e2_blend_by_variant.csv").set_index("variant_id")
    amb = pd.read_csv(D4B / "revision" / "e5_amb_lanthanide.csv")
    W, HH = H.DOUBLE, 96.0
    fig = H.figure(W, HH)
    # ---------------- (a) narrow beam only
    ax = H.add_axes_mm(fig, 27, 10, 60, 76)
    ln = fa[(fa.level == "ln") & ~fa.family.str.contains(r"\+")]
    nb = ln[ln.state == "before"].set_index("family").best_ratio
    fams = nb.sort_values().index.tolist()
    t = tb.pivot(index="formula", columns="state", values="ratio_to_B4_frontier").loc[list(TEXTBOOK10)]
    t = t.sort_values("before")
    a13 = amb[(amb.tier == "AMB-ANION&ICSD3") & (amb.variant_id == 13)].iloc[0]
    if abs(100 * a13.vs_B4 - n.value("H214")) > 1e-9:
        raise ValueError("Fig4a: fully oxidised lanthanide value differs from H214")
    rows = [("ln", f) for f in fams] + [("amb", "amb")] + [("tb", f) for f in t.index]
    ypos, y = [], 0.0
    for k, (kind, _) in enumerate(rows):
        if k > 0 and kind != rows[k - 1][0]:
            y += 1.0 if kind == "tb" else 0.3
        ypos.append(y)
        y += 1.0
    for (kind, f), yy in zip(rows, ypos):
        if kind == "amb":
            ax.plot([0, 100 * a13.vs_B4], [yy, yy], color=H.GRID, lw=0.8, zorder=1, solid_capstyle="butt")
            ax.plot(100 * a13.vs_B4, yy, marker="D", ms=H.MS, mfc=H.LANTHANIDE, mec="white", mew=0.4, ls="none",
                    zorder=4)
            continue
        if kind == "ln":
            x, col = 100 * (nb[f] - 1), H.LANTHANIDE
        else:
            x, col = 100 * (t.loc[f, "before"] - 1), (H.BI if f == "Bi" else H.TEXTBOOK)
        ax.plot([0, x], [yy, yy], color=H.GRID, lw=0.8, zorder=1, solid_capstyle="butt")
        ax.plot(x, yy, mfc=col, zorder=4, **DOT)
    # elemental blend reference (narrow beam, H180)
    xb_ = 100 * float(blend.loc[13, "blend_vs_B4"])
    if abs(xb_ - n.value("H180")) > 1e-9:
        raise ValueError("Fig4a: blend value differs from H180")
    ax.axvline(xb_, color=H.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=2)
    ax.annotate("optimised blend", xy=(xb_, 1.0), xycoords=("data", "axes fraction"), xytext=(-1, 2.5),
                textcoords="offset points", ha="left", va="bottom", fontsize=H.FS_ANNOT, color=H.INK)
    ax.set_yticks(ypos)
    labels = []
    for kind, f in rows:
        labels.append(f if kind == "ln" else ("fully oxidised,\n≥ 3 ICSD entries" if kind == "amb" else _tb_label(f)))
    n.add("Fig4a", "3", "def:amb_icsd_min", "row label of the diamond 'fully oxidised, >= 3 ICSD entries'")
    ax.set_yticklabels(labels, fontsize=H.FS_CHEM)
    for lab_, (kind, f) in zip(ax.get_yticklabels(), rows):
        lab_.set_color(H.text_color(H.LANTHANIDE) if kind in ("ln", "amb")
                       else (H.text_color(H.BI) if f == "Bi" else H.INK))
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ypos[-1] + 0.8, -0.8)
    ax.set_xlim(-3, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(0, color=H.INK2, lw=H.LW_REF, zorder=0)
    H.vgrid(ax)
    ax.set_xlabel("Mass thickness above the\npractical-set frontier (%)")
    n.add("Fig4a", "0", "def:frontier", "x = 0: B4 frontier")
    # group labels outside the plot area (rotated, left of the tick labels)
    fig.canvas.draw()
    i_ln1, i_lnN = 0, len(fams) - 1
    i_tb1, i_tbN = rows.index(("tb", t.index[0])), len(rows) - 1
    tr = ax.transData
    for (i0, i1), txt, col in (((i_ln1, i_lnN), "Lanthanide families", H.text_color(H.LANTHANIDE)),
                               ((i_tb1, i_tbN), "Textbook materials", H.INK2)):
        ym = 0.5 * (ypos[i0] + ypos[i1])
        yf = fig.transFigure.inverted().transform(tr.transform((0, ym)))[1]
        fig.text(4.0 / W, yf, txt, rotation=90, ha="center", va="center", fontsize=H.FS_ANNOT, color=col)
        y0f = fig.transFigure.inverted().transform(tr.transform((0, ypos[i0] - 0.3)))[1]
        y1f = fig.transFigure.inverted().transform(tr.transform((0, ypos[i1] + 0.3)))[1]
        fig.add_artist(__import__("matplotlib").lines.Line2D([6.6 / W, 6.6 / W], [y0f, y1f], color=col, lw=0.5,
                                                             transform=fig.transFigure))
    ax.set_title("Reference beam (RQR6), narrow beam", loc="left", fontsize=H.FS_LABEL, color=H.INK, pad=13)
    H.panel_letter(ax, "a", x_mm=1.0, dy_mm=4.5)
    # ---------------- (b) the staircase
    sd = staircase_data(n)
    ax = H.add_axes_mm(fig, 104, 17, 70, 69)
    # x: 0 = narrow beam, 1 = Monte Carlo (fluorescence-inclusive). One series per material (stage 4d).
    DX = {"RQR5": -0.04, "RQR6": 0.0, "RQR8": 0.04, "RQR6 blend": 0.0,
          "RQR5 oxide": -0.04, "RQR6 oxide": 0.0, "RQR8 oxide": 0.04}

    def x_(series, step):
        return step - 1 + DX[series]
    # light background band behind each method column, so each point reads with the label below it (stage 4d)
    for xc in (0, 1):
        ax.axvspan(xc - 0.32, xc + 0.32, color=H.BOX_FILL, lw=0, zorder=0)
    # optimised blend at RQR6: dashed, ink dot at the Monte Carlo step
    gb = sd[sd.beam == "RQR6 blend"].sort_values("step")
    ax.plot([x_("RQR6 blend", k) for k in gb.step], 100 * gb.adv, color=H.INK, lw=0.8, ls=(0, (3, 1.6)), zorder=2.5)
    ax.errorbar(x_("RQR6 blend", 2), 100 * gb.adv.iloc[-1], yerr=100 * DEFS["mc_ci_z"][0] * gb.se.iloc[-1],
                fmt="none", ecolor=H.MC, elinewidth=0.6, capsize=0, zorder=4)
    ax.plot(x_("RQR6 blend", 2), 100 * gb.adv.iloc[-1], ls="none", marker="o", ms=H.MS - 0.3, mfc=H.INK,
            mec="white", mew=0.4, zorder=5)
    H.direct_label(ax, x_("RQR6 blend", 2), 100 * gb.adv.iloc[-1], "optimised blend, RQR6", H.INK, dx=4, dy=0,
                   ha="left", va="center")
    LABEL = {"LaBi": "La compound", "Ce5AgSn3": "Ce compound", "Pr3OsI3": "Pr compound"}
    for series, g in sd[sd.beam != "RQR6 blend"].groupby("beam"):
        g = g.sort_values("step")
        oxide = series.endswith("oxide")
        beam = series.split()[0]
        mk = "D" if oxide else "o"
        ls = (0, (1, 1.6)) if oxide else "-"
        if len(g) > 1:
            ax.plot([x_(series, k) for k in g.step], 100 * g.adv, color=H.LANTHANIDE, lw=0.8 if not oxide else 0.9,
                    ls=ls, zorder=3)
        for _, r in g.iterrows():
            x = x_(series, r.step)
            if r.transported and np.isfinite(r.se) and r.step == 2:
                ax.errorbar(x, 100 * r.adv, yerr=100 * DEFS["mc_ci_z"][0] * r.se, fmt="none",
                            ecolor=H.MC, elinewidth=0.6, capsize=0, zorder=4)
            filled = r.transported
            ax.plot(x, 100 * r.adv, ls="none", marker=mk, ms=H.MS + (0 if oxide else 0.2),
                    mfc=H.LANTHANIDE if filled else "white", mec=H.LANTHANIDE if not filled else "white",
                    mew=0.8 if not filled else 0.4, zorder=5)
        y1 = 100 * float(g[g.step == 1].adv.iloc[0])
        if oxide:
            if series == "RQR6 oxide":
                y2 = 100 * float(g[g.step == 2].adv.iloc[0])
                H.direct_label(ax, x_(series, 2), y2, "fully oxidised, RQR6", H.INK, dx=4, dy=0, ha="left",
                               va="center")
            else:
                H.direct_label(ax, x_(series, 1), y1, f"fully oxidised, {beam}\n(not transported)", H.INK, dx=-4,
                               dy=0, ha="right", va="center")
        else:
            H.direct_label(ax, x_(series, 1), y1, f"{LABEL[g.material.iloc[0]]}, {beam}", H.INK, dx=-4, ha="right",
                           va="center")
    ax.axhline(0, color=H.INK, lw=0.9, zorder=2)
    H.direct_label(ax, 1.65, 0.0, "Bi", H.BI, dx=0, dy=2, ha="right", va="bottom")
    n.add("Fig4b", "0", "def:zero", "zero line (Bi)")
    ax.set_xlim(-1.15, 1.7)
    ax.set_ylim(-14, 46)
    ax.set_yticks([-10, 0, 10, 20, 30, 40])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Narrow beam", "Monte Carlo\n(fluorescence-inclusive)"], fontsize=H.FS_ANNOT, linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    H.hgrid(ax)
    ax.set_ylabel("Extra mass thickness Bi needs (%)")
    H.panel_letter(ax, "b", dx_mm=12.0, dy_mm=4.5)
    META["Fig4"] = {"a": "narrow beam only (v13): frontier crystal of each single-lanthanide family, ten textbook "
                         "materials, elemental blend line (H180), fully oxidised >= 3 ICSD lanthanide (H214)",
                    "b": sd.to_dict(orient="records"),
                    "b_estimator": "fluorescence-inclusive (fl), like-for-like Pb target, bars +/- 1.96 SE (P4C-D-104)",
                    "marker_types": {"a": 2, "b": 2},
                    "text_in_plot": {"a": ["title 'Reference beam (RQR6), narrow beam' (above the axes)",
                                           "optimised blend (above the axes)", "fully oxidised, ≥ 3 ICSD entries (row label)"],
                                     "b": ["La compound, RQR5", "Ce compound, RQR6", "Pr compound, RQR8", "optimised blend, RQR6",
                                           "fully oxidised, RQR6", "fully oxidised, RQR5 (not transported)",
                                           "fully oxidised, RQR8 (not transported)", "Bi", "Narrow beam",
                                           "Monte Carlo (fluorescence-inclusive)"]},
                    "data": ["data/phase4b/fig4a_families_v13.csv", "fig4a_textbook_v13.csv",
                             "data/phase4b/revision/e2_blend_by_variant.csv",
                             "data/phase4b/revision/e3_like_for_like.csv",
                             "data/phase4b/revision/e5_amb_lanthanide.csv", "data/phase4b/mc_rev/metrics.csv"]}
    _save(fig, "Fig4")


# =============================================================================================
# Graphical abstract: method picture + price of three requirements
# =============================================================================================
# Canvas 1104 x 441.6 pt = 15.333 x 6.133 in (2.5 : 1). PNG 4600 x 1840 px (300 dpi), TIFF 600 dpi. Designed for the
# 500 x 200 px ScienceDirect display: 500 / 1104 = 0.453 px per pt, so 28 pt labels show at 12.7 px (cap height
# ~9.1 px), the 26 pt footnote at 11.8 px (cap height ~8.4 px, ink colour). Drawn in a full-figure axes in pt units.
GA_W_PT, GA_H_PT = 1104.0, 441.6
GA_W_MM, GA_H_MM = GA_W_PT / 72 * 25.4, GA_H_PT / 72 * 25.4
GA_PX = (4600, 1840)
GA_FS_LABEL, GA_FS_VALUE, GA_FS_TITLE, GA_FS_BIG, GA_FS_FOOT = 28.0, 28.0, 31.0, 32.0, 26.0


def graphical_abstract():
    from matplotlib.patches import FancyArrowPatch, Rectangle
    n = _num()
    fig = H.figure(GA_W_MM, GA_H_MM)
    ax = fig.add_axes((0, 0, 1, 1), zorder=0)
    ax.set_xlim(0, GA_W_PT)
    ax.set_ylim(0, GA_H_PT)
    ax.axis("off")
    G = "GraphicalAbstract"
    texts = []

    def txt(x, y, s, **kw):
        texts.append(s)
        return ax.text(x, y, s, **kw)

    # ---- left (~40 %): the method in one picture -----------------------------------------
    sx0, sw, sy0, sh = 172.0, 52.0, 196.0, 210.0            # slab
    ax.add_patch(Rectangle((sx0, sy0), sw, sh, facecolor=ATOM_LIGHT, edgecolor=H.SPINE, lw=2.0, zorder=2))
    for yy in (sy0 + 40.0, sy0 + sh / 2, sy0 + sh - 40.0):    # incident beam
        ax.add_patch(FancyArrowPatch((22.0, yy), (sx0 - 6.0, yy), arrowstyle="-|>", mutation_scale=34, lw=5.0,
                                     color=BEAM, zorder=3, shrinkA=0, shrinkB=0))
    yc = sy0 + sh / 2                                        # transmitted 1 %
    ax.add_patch(FancyArrowPatch((sx0 + sw + 6.0, yc), (sx0 + sw + 120.0, yc), arrowstyle="-|>", mutation_scale=20,
                                 lw=1.6, color=BEAM, zorder=3, shrinkA=0, shrinkB=0))
    t1 = n.fmt(G, "def:T_target:pct", 0, "left picture: T = 1 % (transmission target 0.01 as a per cent)")
    txt(sx0 + sw + 63.0, yc + 22.0, f"T = {t1} %", ha="center", va="bottom", fontsize=GA_FS_BIG, color=H.INK,
        fontweight="bold")
    nc = n.fmt(G, "count:universe_entries", 0, "left caption: number of crystals (baseline-eligible MP entries)")
    xl = 220.8                                               # centre of the left 40 %
    txt(xl, 122.0, f"{nc} crystals", ha="center", va="baseline", fontsize=GA_FS_BIG, color=H.INK, fontweight="bold")
    txt(xl, 72.0, "Materials Project × NIST XCOM", ha="center", va="baseline", fontsize=GA_FS_LABEL, color=H.INK)

    # ---- right (~60 %): price of a requirement, Cs-137 -------------------------------------
    x0 = 468.0                                               # left edge of the right panel; bar baseline
    txt(x0, GA_H_PT - 16.0, "Price of a requirement (Cs-137)", ha="left", va="top", fontsize=GA_FS_TITLE,
        color=H.INK, fontweight="bold")
    yl = 370.0                                               # legend row
    xx = x0
    for lab, col in (("thickness", H.THICKNESS), ("mass thickness", H.AREAL)):
        ax.add_patch(Rectangle((xx, yl - 10.0), 26.0, 20.0, facecolor=col, edgecolor="none", zorder=2))
        t = txt(xx + 34.0, yl, lab, ha="left", va="center_baseline", fontsize=GA_FS_LABEL, color=H.INK)
        bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        xx = bb.x1 + 40.0
    gt = n.fmt(G, "def:gap_threshold", 1, "row label: band gap >= 3.1 eV")
    ab = n.fmt(G, "def:abund_threshold", 0, "row label: abundant elements >= 1 ppm")
    v = {}
    for tag, dec, where in (("H03", 0, "row 1 thickness: lead-free"), ("H05", 0, "row 1 thickness: stable"),
                            ("H07", 0, "row 1 thickness: observed"), ("H04", 0, "row 1 mass thickness: lead-free"),
                            ("H06", 0, "row 1 mass thickness: stable"), ("H08", 0, "row 1 mass thickness: observed"),
                            ("H11", 0, "row 2 thickness: abundant elements"),
                            ("H12", 0, "row 2 mass thickness: abundant elements"),
                            ("H09", 0, "row 3 thickness: transparent"), ("H10", 1, "row 3 mass thickness: transparent")):
        v[tag] = (n.value(tag), n.fmt(G, tag, dec, where))
    # row 1 combines three requirements that are each 0 % on both bases: one printed value per basis, all three tags
    assert len({v[t][1] for t in ("H03", "H05", "H07")}) == 1 and len({v[t][1] for t in ("H04", "H06", "H08")}) == 1
    rows = (("Lead-free · stable · observed", v["H03"], v["H04"]),
            (f"Abundant elements (≥ {ab} ppm)", v["H11"], v["H12"]),
            (f"Transparent (band gap ≥ {gt} eV)", v["H09"], v["H10"]))
    scale = 470.0 / max(r[1][0] for r in rows)               # pt per % (longest bar 470 pt; room for its value)
    bh, step, top, pitch, dy = 22.0, 28.0, 342.0, 104.0, 42.0  # bar height, bar pitch, top of the row-1 label, row pitch
    for i, (lab, (vt, st), (vm, sm)) in enumerate(rows):
        yt = top - i * pitch
        txt(x0, yt, lab, ha="left", va="top", fontsize=GA_FS_LABEL, color=H.INK)
        ax.plot([x0, x0], [yt - dy + bh / 2 + 3.0, yt - dy - step - bh / 2 - 3.0], color=H.INK2, lw=2.0,
                solid_capstyle="butt", zorder=3)                # zero line of the row
        for k, (val, s, col) in enumerate(((vt, st, H.THICKNESS), (vm, sm, H.AREAL))):
            yb = yt - dy - k * step
            if val > 0:
                ax.add_patch(Rectangle((x0, yb - bh / 2), val * scale, bh, facecolor=col, edgecolor="none", zorder=2))
            lab_v = f"+{s} %" if val > 0 else f"{s} %"
            txt(x0 + val * scale + 9.0, yb, lab_v, ha="left", va="center_baseline", fontsize=GA_FS_VALUE,
                color=H.text_color(col), fontweight="bold")
    txt(x0, 14.0, "narrow beam, ideal crystals", ha="left", va="baseline", fontsize=GA_FS_FOOT, color=H.INK)

    files = H.export(fig, G, OUT, formats=("pdf", "tif", "png"), exact_px=GA_PX)
    from PIL import Image
    with Image.open(OUT / f"{G}.png") as im:
        META[G] = {"png_px": list(im.size), "size_mm": [GA_W_MM, GA_H_MM],
                   "files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in files],
                   "min_font_pt_artists": H.min_font_size(fig), "text": texts}
    import matplotlib.pyplot as plt
    plt.close(fig)


# =============================================================================================
# Supplementary figures (three)
# =============================================================================================
def figS1():
    """S1: COD vs MP density bias by compound class (old Fig. S1 panel a; the e_hull panel moved to docs)."""
    from shieldmap.analysis3 import cod
    from shieldmap.analysis3 import common as c3
    n = _num()
    m = pd.read_parquet(D3 / "D" / "matches.parquet")
    h = cod.headline(m)
    if abs(float(h.delta.median()) * 100 - n.value("H87")) > 1e-9:
        raise ValueError("FigS1: median delta differs from H87")
    el = np.asarray(cod._phase1_elemental(), float)
    rows = [("Elements (measured)", el, "ref"), ("All compounds", h.delta.to_numpy(), "all")]
    rows += [(cod.CLASS_LABEL[c].capitalize() if c != "3d_magnetic" else "3d-magnetic",
              h.loc[h[f"is_{c}"], "delta"].to_numpy(), "cls") for c in c3.REPORT_CLASSES]
    allx = [100 * np.asarray(r[1], float) for r in rows]
    XL = (float(np.floor(min(np.percentile(x, 5) for x in allx) - 1)),
          float(np.ceil(max(np.percentile(x, 95) for x in allx) + 1)))
    fig = H.figure(H.SINGLE, 58)
    ax = H.add_axes_mm(fig, 33, 10, 53, 45)
    col = {"ref": H.TEXTBOOK, "all": H.DENSITY, "cls": H.DENSITY}
    from matplotlib.patches import Rectangle
    ys = np.arange(len(rows))[::-1]
    for y, (lab, x, k) in zip(ys, rows):
        x = 100 * np.asarray(x, float)
        p5, p25, p50, p75, p95 = np.percentile(x, [5, 25, 50, 75, 95])
        c = col[k]
        ax.plot([p5, p95], [y, y], color=c, lw=0.9, solid_capstyle="butt", zorder=2)
        ax.add_patch(Rectangle((p25, y - 0.28), p75 - p25, 0.56, fc=c, alpha=0.35 if k != "all" else 0.55, lw=0,
                               zorder=2))
        ax.plot([p50, p50], [y - 0.32, y + 0.32], color=c, lw=1.6, solid_capstyle="butt", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=H.FS_ANNOT)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=H.INK2, lw=H.LW_REF, zorder=1)
    ax.axhline(ys[0] - 0.5, color=H.SPINE, lw=0.5)
    ax.set_xlim(*XL)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("δ = ρ$_{MP}$ / ρ$_{meas}$ − 1 (%)", fontsize=H.FS_LABEL)
    H.vgrid(ax)
    META["FigS1"] = {"data": "data/phase3/D/matches.parquet (cod.headline) + Phase 1 elemental distribution",
                     "xlim": XL, "marker_types": {"main": 0}, "text_in_plot": {"main": []},
                     "phase4b_panel_b": "moved to docs/figures/phase4b_FigS1.png"}
    _save(fig, "FigS1")


def _heat_cmap(color):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("w2c", ["#FFFFFF", color])
    cm.set_bad("white")
    return cm


def figS2(thresh: float = 0.02):
    """S2: element-exclusion filter prices inside B4 (B4-internal). No notes inside the maps; the
    NH3.H2O range keeps its grey hatching (explained in the caption)."""
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
    hgt = 2.7 * len(els)
    fig = H.figure(H.DOUBLE, hgt + 22)
    Wp = (H.DOUBLE - 12 - 6 - 2 * 18) / 2
    axs = [H.add_axes_mm(fig, 12 + k * (Wp + 18), 12, Wp, hgt) for k in range(2)]
    nh3 = h5no_range("B4")
    for k, (ax, basis, col, lab) in enumerate(zip(axs, ("thickness", "areal"), (H.THICKNESS, H.AREAL),
                                                  ("Price of excluding the element, thickness (%)",
                                                   "Price of excluding the element, mass thickness (%)"))):
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
        H.ref_markers(ax, label=(k == 0), zorder=3)
        if basis == "areal" and nh3:
            H.region_hatch(ax, nh3[0] / 1.015, nh3[1] * 1.015, None)
            META.setdefault("FigS2", {})["nh3_range_keV"] = list(nh3)
        bb = ax.get_position()
        cax = fig.add_axes((bb.x1 + 2 / H.DOUBLE, bb.y0, 2.2 / H.DOUBLE, bb.height))
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=H.FS_TICK, length=1.5)
        cb.set_label(lab, fontsize=H.FS_ANNOT)
        H.panel_letter(ax, "ab"[k], dx_mm=8.0)
    META.setdefault("FigS2", {}).update({"elements": els, "threshold": thresh,
                                         "data": "data/phase4a/S/bands.parquet (B4-internal, EXCL(el)|B4, phase4a "
                                                 "artefact flag)", "scale": "square-root colour scale, white = 0",
                                         "marker_types": {"a": 0, "b": 0},
                                         "text_in_plot": {"a": ["Pb K", "Cs-137", "Co-60"], "b": []}})
    _save(fig, "FigS2")


def figS3():
    """S3 (old S7, simplified): (a) MC narrow beam vs analytic ENDF transmission at every Cs-137 / Co-60 point,
    (b) build-up factor B = broad / narrow. One marker shape; fill = energy (filled Cs-137, open Co-60)."""
    n = _num()
    p = pd.read_csv(D4A / "mc" / "points.csv")
    fig = H.figure(H.ONE_HALF, 60)
    axs = [H.add_axes_mm(fig, 14, 11, 46, 44), H.add_axes_mm(fig, 76, 11, 44, 44)]
    # (a) consistency check
    ax = axs[0]
    XOFF = {("c6_cs137_pb", "Pb"): -0.045, ("c1_cs137", "Ir"): -0.015, ("c1_cs137", "HfO2"): 0.0,
            ("c1_cs137", "BiOF"): 0.015, ("c2_co60", "Ir"): 0.03, ("c2_co60", "HfO2"): 0.045}
    for (case, mat), dx in XOFF.items():
        s = p[(p.case_id == case) & (p.material == mat)]
        r = 100 * (s.narrow / s.T_endf_narrow - 1)
        e = 100 * 1.96 * s.narrow_se / s.T_endf_narrow
        co = case == "c2_co60"
        ax.errorbar(s.factor + dx, r, yerr=e, fmt="none", ecolor=H.TEXTBOOK_LIGHT, elinewidth=0.6, capsize=0,
                    zorder=2)
        ax.plot(s.factor + dx, r, ls="none", marker="o", ms=2.8, mfc="white" if co else H.MC, mec=H.MC, mew=0.6,
                zorder=3)
    H.zero_line(ax)
    ax.set_ylim(-1.3, 1.0)
    ax.set_xlim(0.5, 1.7)
    ax.set_xlabel("Slab thickness / thickness for T = 0.01")
    ax.set_ylabel("Monte Carlo narrow beam / analytic − 1 (%)")
    H.hgrid(ax)
    H.panel_letter(ax, "a", dx_mm=12.0)
    # (b) build-up
    ax = axs[1]
    curves = (("c6_cs137_pb", "Pb", "-"), ("c1_cs137", "Ir", "-"), ("c1_cs137", "BiOF", "-"),
              ("c1_cs137", "HfO2", "-"), ("c2_co60", "Ir", (0, (3, 1.5))), ("c2_co60", "HfO2", (0, (3, 1.5))))
    for case, mat, ls in curves:
        s = p[(p.case_id == case) & (p.material == mat)].sort_values("factor")
        B = s.broad / s.narrow
        is_pb = mat == "Pb"
        c = H.MC if is_pb else H.TEXTBOOK
        co = case == "c2_co60"
        ax.plot(s.factor, B, color=c, ls=ls, lw=1.0, marker="o", ms=2.6, mfc="white" if co else c, mec=c, mew=0.6,
                zorder=4 if is_pb else 3)
        lab = H.chem(mat) + (", Co-60" if co else "")
        dy = {("c6_cs137_pb", "Pb"): -4, ("c1_cs137", "BiOF"): 0.5, ("c1_cs137", "Ir"): 4.5,
              ("c1_cs137", "HfO2"): -2, ("c2_co60", "Ir"): 2}.get((case, mat), 0)
        ax.annotate(lab, xy=(s.factor.iloc[-1], B.iloc[-1]), xytext=(3, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=H.FS_CHEM, color=H.INK if is_pb else H.INK2)
    s = p[(p.case_id == "c6_cs137_pb") & np.isclose(p.factor, 1.0)].iloc[0]
    if abs(float(s.broad / s.narrow) - n.value("H124")) > 1e-6:
        raise ValueError("FigS3: Pb build-up at the headline thickness differs from H124")
    ax.axvline(1.0, color=H.REF, lw=H.LW_REF, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(0.5, 2.05)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_xlabel("Slab thickness / thickness for T = 0.01")
    n.add("FigS3", "0.01", "def:T_target", "x labels (a, b) 'Slab thickness / thickness for T = 0.01'")
    ax.set_ylabel("Build-up factor B")
    H.hgrid(ax)
    H.panel_letter(ax, "b", dx_mm=11.0)
    META["FigS3"] = {"data": ["data/phase4a/mc/points.csv"], "phase4b_name": "FigS7",
                     "marker_types": {"a": 2, "b": 2},
                     "text_in_plot": {"a": [], "b": ["Pb", "Ir", "BiOF", "HfO2", "Ir, Co-60", "HfO2, Co-60"]}}
    _save(fig, "FigS3")


# =============================================================================================
def run(only: list[str] | None = None) -> dict:
    H.use()
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    funcs = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "Fig4": fig4, "GraphicalAbstract": graphical_abstract,
             "FigS1": figS1, "FigS2": figS2, "FigS3": figS3}
    for name, f in funcs.items():
        if only and name not in only:
            continue
        print(f"[fig] {name}", flush=True)
        f()
    if only:                                   # partial render (development): no register or checks
        return META
    # the stage 4b supplementary figures that left the paper (kept in docs/figures/)
    for nm in RETIRED:
        for ext in ("pdf", "tif", "png"):
            q = OUT / f"{nm}.{ext}"
            if q.exists():
                q.unlink()
    _num().write()
    from shieldmap.figstyle import checks
    names = list(funcs)
    imgs = checks.cvd_images(OUT, REP / "cvd", names)
    checks.cvd_report(names, REP / "cvd_check.md", imgs)
    META["_font_check"] = checks.font_report([OUT / f"{n}.pdf" for n in names], REP / "font_check.md")
    checks.contact_sheet(OUT, names, REP / "contact_sheet.pdf", {n: META[n]["size_mm"] for n in names},
                         preview_dir=REP)
    (REP / "figures_meta.json").write_text(json.dumps(META, indent=1, default=str), encoding="utf-8")
    return META


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
