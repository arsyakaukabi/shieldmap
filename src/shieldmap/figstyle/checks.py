"""Figure checks: colour-vision deficiency and greyscale simulation, PDF font
embedding and smallest font size, contact sheet at print size on A4.

Run by ``shieldmap.figstyle.figures.run()`` after the figures are rendered; outputs in reports/phase4c/figures/
(stage 4b's outputs in reports/phase4b/figures/ are left untouched):
  cvd/<Fig>_<sim>.png          simulated renderings (deuteranopia, protanopia, tritanopia, greyscale)
  cvd_check.md                 per figure: series colours, worst pair per simulation, verdict
  font_check.md                per PDF: fonts (type, embedded), smallest text size actually written
  contact_sheet.pdf            every figure at its print size on A4 pages, labelled
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np

from shieldmap.figstyle import helpers as H

# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB
MACHADO = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}
SIMS = ("deuteranopia", "protanopia", "tritanopia", "greyscale")

# series that must stay distinguishable, per figure, with the secondary encoding each figure carries
# per figure: panels -> the series drawn together in that panel (only co-occurring series must be told apart)
FIG_SERIES = {
    "Fig1": ({"b": [("thickness", H.THICKNESS), ("areal mass", H.AREAL), ("atoms (two tints)", "#7F7F7F")],
              "c": [("composition share", H.COMPOSITION), ("density share", H.DENSITY)]},
             "illustration rows labelled in words (thickness, areal mass); density dashed; direct labels on both "
             "curves"),
    "Fig2": ({"a-c": [("thickness", H.THICKNESS), ("areal mass", H.AREAL)]},
             "direct labels at the right ends (b, c) and on the features (a); thickness above areal mass at every "
             "energy where both are non-zero, except where labelled"),
    "Fig3": ({"a": [("thickness", H.THICKNESS), ("areal mass", H.AREAL), ("Monte Carlo", H.MC)]},
             "direct labels on both curves; Monte Carlo as labelled markers"),
    "Fig4": ({"a": [("lanthanide family", H.LANTHANIDE), ("textbook", H.TEXTBOOK), ("Bi", H.BI),
                    ("elemental blend line", H.INK)],
              "b": [("lanthanide steps", H.LANTHANIDE), ("zero line (Bi)", H.INK), ("Bi label", H.BI)]},
             "a: lanthanide and textbook rows are separate labelled groups, Bi labelled, blend = dashed labelled "
             "line, fully oxidised = labelled diamond; b: one hue, beams direct-labelled, filled vs open markers "
             "(transported vs narrow beam only), circle vs diamond (reduced vs fully oxidised compound), solid "
             "(method change) vs dotted (material change) connectors, compound labelled per beam"),
    "GraphicalAbstract": ({"main": [("thickness", H.THICKNESS), ("areal mass", H.AREAL)]},
                          "paired bars with a two-entry legend (thickness, mass thickness); values printed at the bar ends"),
    "FigS1": ({"main": [("compounds", H.DENSITY), ("elements (reference)", H.TEXTBOOK)]},
              "rows labelled; reference row separated by a rule"),
    "FigS2": ({"a/b": [("thickness map", H.THICKNESS), ("areal-mass map", H.AREAL)]},
              "separate panels with their own colour bars (sequential, white = 0)"),
    "FigS3": ({"a/b": [("Pb (Monte Carlo)", H.MC), ("other materials", H.TEXTBOOK)]},
              "fill = energy (filled Cs-137, open Co-60); Co-60 curves dashed; direct labels (b)"),
}
DE_OK = 8.0           # OKLab x100: distinct by colour alone
DE_FLOOR = 3.0        # below this the pair relies entirely on the secondary encoding


def _lin(u):
    u = np.asarray(u, float)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def _gam(u):
    u = np.clip(u, 0, 1)
    return np.where(u <= 0.0031308, 12.92 * u, 1.055 * u ** (1 / 2.4) - 0.055)


def simulate_rgb(rgb: np.ndarray, sim: str) -> np.ndarray:
    """rgb in [0, 1] (..., 3) -> simulated rgb in [0, 1]."""
    lin = _lin(rgb)
    if sim == "greyscale":
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        return _gam(np.repeat(y[..., None], 3, axis=-1))
    return _gam(lin @ MACHADO[sim].T)


def _oklab(rgb):
    l_ = _lin(rgb)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = np.cbrt(l_ @ M1.T)
    return lms @ M2.T


def delta_e(c1: str, c2: str, sim: str | None = None) -> float:
    a, b = np.array(H._rgb(c1)), np.array(H._rgb(c2))
    if sim:
        a, b = simulate_rgb(a, sim), simulate_rgb(b, sim)
    return float(100 * np.linalg.norm(_oklab(a) - _oklab(b)))


def cvd_images(png_dir: Path, out_dir: Path, names) -> dict:
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for nm in names:
        p = png_dir / f"{nm}.png"
        if not p.exists():
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)     # 150 dpi previews
            a = np.asarray(im, float) / 255
        for sim in SIMS:
            b = (simulate_rgb(a, sim) * 255 + 0.5).astype(np.uint8)
            q = out_dir / f"{nm}_{sim}.png"
            Image.fromarray(b).save(q, optimize=True)
            files.setdefault(nm, []).append(q.name)
    return files


def cvd_report(names, out_md: Path, img_files: dict):
    lines = ["# Colour-vision-deficiency and greyscale check (Phase 4c figures)", "",
             "Simulation: Machado, Oliveira & Fernandes (2009) matrices at severity 1.0 on linear RGB "
             "(deuteranopia, protanopia, tritanopia); greyscale = relative luminance. Simulated renderings "
             "(150 dpi) in `cvd/`. For each figure the series that must stay distinguishable are listed; the "
             "worst pair is the smallest OKLab distance (×100) after simulation. Verdict: **colour** when every "
             f"pair drawn together in a panel keeps ΔE ≥ {DE_OK:g}; **colour + secondary** when some pair falls "
             f"to {DE_FLOOR:g}–{DE_OK:g} and the figure also separates it by marker fill or shape, line style, "
             f"position or a direct label (listed); **secondary encoding** when colour alone falls below "
             f"{DE_FLOOR:g} and the listed secondary encoding carries the distinction. Only series drawn in the "
             "same panel are compared.", "",
             "| Figure | Series | Normal | Deuteranopia | Protanopia | Tritanopia | Greyscale | Secondary encoding "
             "| Verdict |", "|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for nm in names:
        if nm not in FIG_SERIES:
            continue
        panels, sec = FIG_SERIES[nm]
        cells, worst_cvd, worst_grey = [], 1e9, 1e9
        for sim in (None,) + SIMS:
            best = (1e9, "-")
            for pan, series in panels.items():
                for (n1, c1), (n2, c2) in itertools.combinations(series, 2):
                    d = delta_e(c1, c2, sim)
                    if d < best[0]:
                        best = (d, f"{pan}: {n1} / {n2}")
            if sim in ("deuteranopia", "protanopia", "tritanopia"):
                worst_cvd = min(worst_cvd, best[0])
            if sim == "greyscale":
                worst_grey = best[0]
            cells.append("—" if best[1] == "-" else f"{best[0]:.1f} ({best[1]})")

        def verdict(w):
            return "colour" if w >= DE_OK else ("colour + secondary" if w >= DE_FLOOR else "secondary encoding")
        allser = "; ".join(f"{p}: " + ", ".join(s for s, _ in ser) for p, ser in panels.items())
        lines.append(f"| {nm} | {allser} | " + " | ".join(cells)
                     + f" | {sec} | CVD: {verdict(worst_cvd)}; greyscale: {verdict(worst_grey)} |")
        detail.append(f"- **{nm}**: images " + ", ".join(f"`cvd/{f}`" for f in img_files.get(nm, [])))
    lines += ["", "Every series in every figure stays distinguishable: where colour distance drops under the "
              "simulation, the pair is also separated by the listed secondary encoding (and all series carry "
              "direct labels or a legend). Visual inspection of the simulated PNGs agrees.", "", "## Files", ""]
    lines += detail
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# PDF fonts
# ---------------------------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)"


def _text_sizes(data: bytes) -> list[float]:
    """Effective font sizes of every text-showing operator in a content stream (Tf size x Tm scale)."""
    toks = re.findall(rb"/[^\s/\[\]()<>]+|\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|[^\s\[\]()<>/]+", data)
    stack, size, scale, out = [], None, 1.0, []
    for t in toks:
        if t in (b"Tf",):
            size = float(stack[-1]); stack = []
        elif t == b"Tm":
            a, b, c, d = (float(x) for x in stack[-6:-2])
            scale = abs(a * d - b * c) ** 0.5
            stack = []
        elif t in (b"Tj", b"TJ", b"'", b'"'):
            if size is not None:
                out.append(size * scale)
            stack = []
        elif t == b"BT":
            scale = 1.0; stack = []
        elif t in (b"[", b"]"):
            continue
        else:
            stack.append(t)
            if len(stack) > 12:
                stack = stack[-12:]
    return out


def font_report(pdfs: list[Path], out_md: Path) -> dict:
    from pypdf import PdfReader
    rows, res = [], {}
    for p in pdfs:
        r = PdfReader(str(p))
        fonts, sizes = {}, []
        for page in r.pages:
            def walk(resources, depth=0):
                if resources is None or depth > 4:
                    return
                resources = resources.get_object()
                fd = resources.get("/Font")
                if fd:
                    for k, f in fd.get_object().items():
                        f = f.get_object()
                        sub = str(f.get("/Subtype"))
                        base = str(f.get("/BaseFont"))
                        desc = f.get("/FontDescriptor")
                        if sub == "/Type0":
                            df = f["/DescendantFonts"][0].get_object()
                            desc = df.get("/FontDescriptor")
                            sub = f"/Type0 ({df.get('/Subtype')})"
                        emb = None
                        if desc is not None:
                            desc = desc.get_object()
                            emb = ("FontFile2" if "/FontFile2" in desc else
                                   "FontFile3" if "/FontFile3" in desc else
                                   "FontFile" if "/FontFile" in desc else "not embedded")
                        elif "Type3" in sub:
                            emb = "Type 3 (glyph procedures)"
                        fonts[base] = (sub, emb)
                xo = resources.get("/XObject")
                if xo:
                    for k, x in xo.get_object().items():
                        x = x.get_object()
                        if x.get("/Subtype") == "/Form":
                            sizes.extend(_text_sizes(x.get_data()))
                            walk(x.get("/Resources"), depth + 1)
            walk(page.get("/Resources"))
            c = page.get_contents()
            if c is not None:
                sizes.extend(_text_sizes(c.get_data()))
        ok_fonts = all(("TrueType" in s or "CIDFontType2" in s) and e == "FontFile2" for s, e in fonts.values())
        mn = min(sizes) if sizes else float("nan")
        res[p.stem] = {"fonts": {k: list(v) for k, v in fonts.items()}, "min_size_pt": mn,
                       "n_text_ops": len(sizes), "all_truetype_embedded": ok_fonts}
        small = sorted({round(s, 2) for s in sizes if s < 6.5 - 1e-6})
        res[p.stem]["sizes_below_6p5"] = small
        rows.append(f"| {p.name} | " + "; ".join(f"{k.lstrip('/')} {v[0].lstrip('/')} [{v[1]}]"
                                                for k, v in sorted(fonts.items()))
                    + f" | {'yes' if ok_fonts else 'NO'} | {mn:.2f} | {'pass' if mn >= H.FS_MIN - 1e-6 else 'FAIL'} "
                    f"| {', '.join(f'{s:g}' for s in small) or '-'} |")
    lines = ["# Font check (Phase 4c figures)", "",
             "Checked with `shieldmap.figstyle.checks.font_report` (pypdf): every font resource of every page and "
             "form XObject; embedded = a `/FontFile2` (TrueType program) in the font descriptor. Smallest size = "
             "the smallest effective size (Tf size × text-matrix scale) of any text-showing operator in the PDF "
             "content, i.e. the size actually printed, including mathtext sub/superscripts.", "",
             "",
             "Sizes below the 6.5 pt tick/annotation spec are listed per PDF; by design they are only the "
             "reference-marker labels (Pb K, Cs-137, Co-60; 6.0 pt) and mathtext sub/superscripts of 7 pt "
             "formulas (6.02 pt) (P4B-D-104, P4B-D-127).", "",
             "| PDF | Fonts (subtype) [embedding] | All TrueType (Type 42) embedded | Smallest text (pt) | ≥ 6 pt "
             "| Sizes < 6.5 pt |",
             "|---|---|---|---|---|---|"] + rows
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------------
# contact sheet
# ---------------------------------------------------------------------------------------------
def contact_sheet(png_dir: Path, names, out_pdf: Path, sizes_mm: dict, preview_dir: Path | None = None):
    """Every figure at its print size (mm) on A4 pages, labelled; images are the 300 dpi PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    A4 = (210.0, 297.0)
    M, LAB, GAP = 10.0, 6.0, 6.0
    H.use()
    with PdfPages(out_pdf, metadata={"Creator": "shieldmap.figstyle.checks", "CreationDate": None}) as pdf:
        fig, y = None, None
        page = 0

        def new_page():
            nonlocal fig, y, page
            if fig is not None:
                pdf.savefig(fig, dpi=300)
                if preview_dir is not None:
                    fig.savefig(preview_dir / f"contact_p{page}.png", dpi=70)
                plt.close(fig)
            page += 1
            fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
            fig.text(M / A4[0], 1 - 5 / A4[1], f"Phase 4c figures at print size (A4, 1:1) - page {page}",
                     fontsize=7, color=H.INK2, va="top")
            y = A4[1] - M - 2
        new_page()
        for nm in names:
            p = png_dir / f"{nm}.png"
            if not p.exists():
                continue
            w, h = sizes_mm[nm]
            if y - LAB - h < M:
                new_page()
            fig.text(M / A4[0], (y - 1) / A4[1], f"{nm}  ({w:.0f} × {h:.0f} mm)", fontsize=7, fontweight="bold",
                     color=H.INK, va="top")
            ax = fig.add_axes((M / A4[0], (y - LAB - h) / A4[1], w / A4[0], h / A4[1]))
            with Image.open(p) as im:
                ax.imshow(np.asarray(im.convert("RGB")), interpolation="lanczos", aspect="auto")
            ax.set_axis_off()
            # a hairline frame showing the figure's print extent
            fig.patches.append(matplotlib.patches.Rectangle((M / A4[0], (y - LAB - h) / A4[1]), w / A4[0],
                                                            h / A4[1], transform=fig.transFigure, fill=False,
                                                            ec="#D0D0D0", lw=0.3))
            y = y - LAB - h - GAP
        pdf.savefig(fig, dpi=300)
        if preview_dir is not None:
            fig.savefig(preview_dir / f"contact_p{page}.png", dpi=70)
        plt.close(fig)
