"""Figure style system: palette, sizes, typography helpers, export.

Every figure in ``shieldmap.figstyle.figures`` is styled only through this module and ``shieldmap.mplstyle``.

Conventions (see reports/phase4b/step2_figstyle.md):
  * sizes are final print sizes in mm (90 / 140 / 190 mm wide); layouts are placed in mm, never rescaled;
  * one semantic palette (``THICKNESS``, ``AREAL``, ``MC``, ``MODEL``, ``LANTHANIDE``, ``TEXTBOOK``, ``BI`` ...);
  * one log-energy tick convention (``log_energy_axis``): labelled 20, 100, 1 000, 10 000 keV (thin-space
    thousands separator), unlabelled ticks at 50, 200, 500, 2 000, 5 000 keV;
  * reference markers (``ref_markers``): Pb K edge 88.0 keV, Cs-137 661.7 keV, Co-60 at the mean of its two lines
    (1 252.9 keV), thin light-grey dashed lines with small top labels, identical in every figure;
  * export (``export``): PDF (TrueType / Type 42 fonts embedded), TIFF 600 dpi LZW, PNG 300 dpi.
"""
from __future__ import annotations

import colorsys
import io
from pathlib import Path

import numpy as np

STYLE_FILE = Path(__file__).with_name("shieldmap.mplstyle")

# ---------------------------------------------------------------------------------------------
# palette: each concept keeps its colour in every figure
# ---------------------------------------------------------------------------------------------
THICKNESS = "#1F4E79"      # thickness basis (space)
AREAL = "#D9822B"          # areal-mass basis (mass)
MC = "#222222"             # Monte Carlo (near-black markers)
MODEL = "#9AA0A6"          # first-order / model-only
LANTHANIDE = "#2A9D8F"     # lanthanide family
TEXTBOOK = "#7F7F7F"       # textbook materials (neutral grey)
TEXTBOOK_LIGHT = "#BFBFBF"
BI = "#D9822B"             # Bi highlighted among the textbook materials (orange)
# Okabe-Ito categorical set, fixed order; #F0E442 for fills only
OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442")
# fixed concept -> Okabe-Ito mapping for concepts outside the core palette (recorded)
DENSITY = "#CC79A7"        # density share / density factor
COMPOSITION = "#009E73"    # composition share / composition factor
PHOTOELECTRIC = "#D55E00"
COMPTON = "#56B4E9"
COHERENT = "#0072B2"
PAIR = "#E69F00"
DENSITY_LS = (0, (5, 1.6))  # density curves are also dashed: density vs composition survives greyscale
# neutrals
INK = "#222222"            # primary text
INK2 = "#555555"           # secondary text
SPINE = "#8C8C8C"
GRID = "#E6E6E6"
REF = "#BDBDBD"            # reference-marker lines
REF_TEXT = "#6E6E6E"       # reference-marker labels
ZERO = "#8C8C8C"
BOX_FILL = "#EEF0F2"       # schematic boxes (flat)
ARROW = "#6E6E6E"          # schematic arrows

BAND_ALPHA = 0.18          # credible bands: same hue as the line, 15-20 % opacity, no edge
LW_DATA = 1.1              # data lines 1.0-1.25 pt
LW_REF = 0.6               # reference lines
MS = 3.5                   # markers 3-4 pt

# ---------------------------------------------------------------------------------------------
# sizes (Elsevier), mm
# ---------------------------------------------------------------------------------------------
SINGLE, ONE_HALF, DOUBLE = 90.0, 140.0, 190.0
MM = 1 / 25.4
FS_LABEL, FS_TICK, FS_ANNOT, FS_LETTER, FS_MIN = 7.0, 6.5, 6.5, 8.0, 6.0

# reference energies (keV) - definitional values
E_PB_K = 88.0
E_CS137 = 661.7
E_CO60_LINES = (1173.2, 1332.5)
E_CO60 = 0.5 * (E_CO60_LINES[0] + E_CO60_LINES[1])     # 1252.85: marked once at the mean
REF_LINES = {"Pb K": E_PB_K, "Cs-137": E_CS137, "Co-60": E_CO60}

E_LO, E_HI = 20.0, 10000.0
E_MAJOR = (20, 100, 1000, 10000)
E_MINOR = (50, 200, 500, 2000, 5000)
THIN = " "            # thin space (thousands separator)
MINUS = "−"


SUB_SHRINK = 0.86          # mathtext sub/superscript scale: 7 pt text -> 6.0 pt subscripts
FS_CHEM = 7.0              # minimum size of any text that carries a subscript (formulas)


def use():
    """Apply the shieldmap style; returns pyplot.

    Mathtext sub/superscripts are scaled by 0.86 instead of 0.7, so a formula set at 7 pt keeps its
    subscripts at 6.0 pt (nothing below 6 pt anywhere)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import _mathtext
    _mathtext.SHRINK_FACTOR = SUB_SHRINK
    plt.style.use(str(STYLE_FILE))
    return plt


# ---------------------------------------------------------------------------------------------
# layout in mm
# ---------------------------------------------------------------------------------------------
def figure(width_mm: float, height_mm: float):
    plt = use()
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._mm = (width_mm, height_mm)
    return fig


def add_axes_mm(fig, left: float, bottom: float, width: float, height: float, **kw):
    """Axes placed in mm from the figure's bottom-left corner (design at final size)."""
    W, H = fig._mm
    return fig.add_axes((left / W, bottom / H, width / W, height / H), **kw)


def panel_letter(ax, letter: str, dx_mm: float = 7.5, dy_mm: float = 1.5, x_mm: float | None = None):
    """8 pt bold lowercase panel letter at the top-left, outside the axes.

    Placed ``dx_mm`` left of the axes' left edge (or at absolute ``x_mm`` from the figure's left edge) and
    ``dy_mm`` above its top edge."""
    fig = ax.figure
    W, H = fig._mm
    bb = ax.get_position()
    x = (x_mm / W) if x_mm is not None else bb.x0 - dx_mm / W
    y = bb.y1 + dy_mm / H
    fig.text(x, y, letter.lower(), fontsize=FS_LETTER, fontweight="bold", ha="left", va="bottom", color=INK)


# ---------------------------------------------------------------------------------------------
# colour utilities
# ---------------------------------------------------------------------------------------------
def _rgb(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _lum(rgb) -> float:
    def f(u):
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = (f(u) for u in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(c1: str, c2: str = "#FFFFFF") -> float:
    a, b = _lum(_rgb(c1)), _lum(_rgb(c2))
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def text_color(c: str, target: float = 4.5) -> str:
    """The series colour darkened (same hue) until it reaches ``target`` contrast on white: direct labels
    keep the identity of their series and stay legible (orange #D9822B is 2.9:1 on white)."""
    r, g, b = _rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    out = c
    while contrast(out) < target and l > 0:
        l = max(0.0, l - 0.01)
        rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
        out = "#{:02X}{:02X}{:02X}".format(round(rr * 255), round(gg * 255), round(bb * 255))
    return out


# ---------------------------------------------------------------------------------------------
# axes helpers
# ---------------------------------------------------------------------------------------------
def fmt_num(v: float, decimals: int = 0) -> str:
    """Number with a thin-space thousands separator and a true minus sign."""
    s = f"{abs(v):,.{decimals}f}".replace(",", THIN)
    return (MINUS if v < 0 else "") + s


def log_energy_axis(ax, lo: float = E_LO, hi: float = E_HI, labels: bool = True, xlabel: bool = True):
    """The one log-energy tick convention: labelled 20, 100, 1 000, 10 000 keV; unlabelled 50 ... 5 000."""
    from matplotlib.ticker import FixedLocator, NullFormatter
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([e for e in E_MAJOR if lo <= e <= hi]))
    ax.xaxis.set_minor_locator(FixedLocator([e for e in E_MINOR if lo <= e <= hi]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if labels:
        ax.set_xticklabels([fmt_num(e) for e in E_MAJOR if lo <= e <= hi])
    else:
        ax.tick_params(axis="x", which="major", labelbottom=False)
    ax.tick_params(axis="x", which="minor", length=1.5, bottom=True)
    if xlabel and labels:
        ax.set_xlabel("Photon energy (keV)")


def plain_log_ticks(axis, values):
    """Log axis with plain decimal tick labels (0.001, 0.01 ...) instead of mathtext powers, whose
    superscripts would print below 6 pt."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    axis.set_major_locator(FixedLocator(values))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())
    axis.set_ticklabels([f"{v:g}" if v >= 1e-4 else f"{v:.0e}" for v in values])


def hgrid(ax):
    """Horizontal gridlines only, 0.4 pt, very light, behind the data."""
    ax.grid(True, axis="y", color=GRID, lw=0.4, zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(True, axis="x", color=GRID, lw=0.4, zorder=0)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)


def zero_line(ax, y: float = 0.0, color: str = ZERO, lw: float = LW_REF, **kw):
    ax.axhline(y, color=color, lw=lw, zorder=1, **kw)


def ref_markers(ax, which=("Pb K", "Cs-137", "Co-60"), label: bool = True, y_label: float = 1.0,
                fontsize: float = FS_MIN, zorder: float = 0.5, sides: dict | None = None):
    """Characteristic energies as thin light-grey dashed lines with small top labels (identical everywhere).

    Labels sit rotated at the top of the axes: Pb K and Co-60 just right of their line, Cs-137 just left
    (so the close Cs-137 / Co-60 pair never collides). Pass ``label=False`` on all but one panel of a row of
    small multiples."""
    side = {"Pb K": 1, "Cs-137": -1, "Co-60": 1, **(sides or {})}
    for name in which:
        E = REF_LINES[name]
        ax.axvline(E, color=REF, lw=LW_REF, ls=(0, (3, 2)), zorder=zorder)
        if label:
            s = side[name]
            ax.annotate(name, xy=(E, y_label), xycoords=("data", "axes fraction"), xytext=(1.0 * s, -1),
                        textcoords="offset points", ha="left" if s > 0 else "right", va="top", rotation=90,
                        fontsize=fontsize, color=REF_TEXT)


def band(ax, x, lo, hi, color: str, alpha: float = BAND_ALPHA, hatched: bool = False, **kw):
    """Credible band: same hue as its line, 15-20 % opacity, no edge.

    ``hatched=True`` (the areal-mass band wherever it can overlap the thickness band): a lighter fill
    plus a fine diagonal hatch in the same hue, so the two bases stay apart in greyscale."""
    if not hatched:
        return ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, edgecolor="none", **kw)
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, lw=0, edgecolor="none", **kw)
    import matplotlib as mpl
    with mpl.rc_context({"hatch.color": color, "hatch.linewidth": 0.4}):
        return ax.fill_between(x, lo, hi, facecolor="none", edgecolor=color, hatch="//////", lw=0, alpha=0.55,
                               **kw)


def region_hatch(ax, x0, x1, label: str | None = None, y_label: float = 0.5, fontsize: float = FS_ANNOT):
    """Light-grey hatched x-range marking data that is set by an artefact of the reference set (e.g. the
    NH3.H2O mass frontier of the primary set), with an optional note."""
    import matplotlib as mpl
    with mpl.rc_context({"hatch.color": "#B0B0B0", "hatch.linewidth": 0.4}):
        ax.axvspan(x0, x1, facecolor="none", edgecolor="#B0B0B0", hatch="\\\\\\\\", lw=0, zorder=4)
    if label:
        ax.annotate(label, xy=(np.sqrt(x0 * x1), y_label), xycoords=("data", "axes fraction"), ha="center",
                    va="center", fontsize=fontsize, color=INK2, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))


def direct_label(ax, x, y, text: str, color: str, dx: float = 2.0, dy: float = 0.0, ha: str = "left",
                 va: str = "center", fontsize: float = FS_ANNOT, coords="data", **kw):
    """A label placed next to its series, in the series colour darkened to text contrast."""
    return ax.annotate(text, xy=(x, y), xycoords=coords, xytext=(dx, dy), textcoords="offset points", ha=ha,
                       va=va, fontsize=fontsize, color=text_color(color), **kw)


def note(ax, x, y, text: str, ha="left", va="center", color: str = INK2, fontsize: float = FS_ANNOT,
         coords="axes fraction", **kw):
    """A neutral annotation (secondary ink)."""
    return ax.annotate(text, xy=(x, y), xycoords=coords, ha=ha, va=va, fontsize=fontsize, color=color, **kw)


_SUBDIGITS = set("0123456789")


def chem(formula: str) -> str:
    """Chemical formula with mathtext subscripts in the sans font: 'Bi2O3' -> 'Bi$_{2}$O$_{3}$'."""
    out, i = [], 0
    while i < len(formula):
        ch = formula[i]
        if ch in _SUBDIGITS and i > 0 and (formula[i - 1].isalpha() or formula[i - 1] == ")"):
            j = i
            while j < len(formula) and (formula[j] in _SUBDIGITS or formula[j] == "."):
                j += 1
            out.append("$_{" + formula[i:j] + "}$")
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------
def export(fig, name: str, outdir: Path, png_dpi: int = 300, tif_dpi: int = 600, formats=("pdf", "tif", "png"),
           exact_px: tuple[int, int] | None = None) -> list[Path]:
    """Write <outdir>/<name>.pdf (fonts embedded as TrueType/Type 42), .tif (600 dpi, LZW) and .png (300 dpi).

    The figure is written at its design size (no tight bbox), so the files have the exact print width."""
    from PIL import Image
    outdir.mkdir(parents=True, exist_ok=True)
    out = []

    def _put(p: Path, data: bytes):
        # write via a temporary file and replace; retry briefly if another process holds the target (Windows)
        import os
        import time
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        for k in range(20):
            try:
                os.replace(tmp, p)
                return
            except OSError:
                time.sleep(0.5)
        raise OSError(f"could not replace {p}")

    if "pdf" in formats:
        p = outdir / f"{name}.pdf"
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", metadata={"Creator": "shieldmap.figstyle", "Producer": None,
                                                  "CreationDate": None})
        _put(p, buf.getvalue())
        out.append(p)
    if "tif" in formats:
        p = outdir / f"{name}.tif"
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=tif_dpi)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        b2 = io.BytesIO()
        im.save(b2, format="TIFF", compression="tiff_lzw", dpi=(tif_dpi, tif_dpi))
        _put(p, b2.getvalue())
        out.append(p)
    if "png" in formats:
        p = outdir / f"{name}.png"
        buf = io.BytesIO()
        if exact_px is not None:
            w_in = fig.get_size_inches()[0]
            fig.savefig(buf, format="png", dpi=exact_px[0] / w_in)
        else:
            fig.savefig(buf, format="png", dpi=png_dpi, metadata={"Software": None})
        _put(p, buf.getvalue())
        out.append(p)
    return out


def min_font_size(fig) -> float:
    """Smallest font size of any visible text artist in the figure (pt)."""
    from matplotlib.text import Text
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return float(min(sizes)) if sizes else float("nan")
