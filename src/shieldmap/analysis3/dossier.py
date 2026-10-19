"""Dossier support. Collects every step's summary.json into one file and checks figures.

Outputs
  data/phase3/dossier/summaries.json   the summary.json of every step (S4, F, D, P, L, C), verbatim
  data/phase3/dossier/figures.json     the eight dossier figures (PNG + SVG present, pixel size)
  data/phase3/dossier/inputs.json      SHA-256 of every stage 3 data file the dossier cites

The narrative dossier (reports/phase3/results_dossier.md, survivors.md) cites these files and the
step reports; it is written by hand and not regenerated.
"""
from __future__ import annotations

import json

from shieldmap import paths
from shieldmap.analysis3 import common

OUT = common.P3 / "dossier"
STEP_DIRS = {"step4": "S4", "step2": "F", "step3": "D", "step5": "P", "step6": "L", "step7": "C",
             "step10_density_model": "T", "step10_edge_windows": "E", "step10_reweight": "RW"}
FIGURES = {
    1: "fig1_variance_physics", 2: "fig2_price_curves", 3: "fig3_gap_inversion", 4: "fig4_load_bearing",
    5: "fig5_diag_plateau", 6: "fig6_tiers", 7: "fig7_density_bias_by_class", 8: "fig8_polymorph_spread",
}


def summaries() -> dict:
    out = {}
    for step, d in STEP_DIRS.items():
        p = common.P3 / d / "summary.json"
        if not p.exists():
            raise FileNotFoundError(f"{step}: {p} missing (run that step first)")
        out[step] = json.loads(p.read_text(encoding="utf-8"))
    return out


def figures() -> dict:
    from PIL import Image
    out = {}
    for n, name in FIGURES.items():
        png = common.FIGURES3 / f"{name}.png"
        svg = common.FIGURES3 / f"{name}.svg"
        if not (png.exists() and svg.exists()):
            raise FileNotFoundError(f"Figure {n} ({name}) missing")
        with Image.open(png) as im:
            w, h = im.size
            dpi = im.info.get("dpi", (None, None))[0]
        out[n] = {"name": name, "png_px": [w, h], "dpi": dpi, "width_in": round(w / dpi, 2) if dpi else None}
    return out


def inputs() -> dict:
    out = {}
    for d in STEP_DIRS.values():
        for p in sorted((common.P3 / d).glob("*")):
            if p.is_file() and not p.name.startswith("_"):
                out[str(p.relative_to(paths.ROOT)).replace("\\", "/")] = common.sha256(p)
    return out


def run() -> dict:
    paths.ensure(OUT)
    s = summaries()
    common.write_json(OUT / "summaries.json", s)
    f = figures()
    common.write_json(OUT / "figures.json", f)
    i = inputs()
    common.write_json(OUT / "inputs.json", i)
    return {"steps": list(s), "figures": {k: v["name"] for k, v in f.items()}, "n_inputs": len(i)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
