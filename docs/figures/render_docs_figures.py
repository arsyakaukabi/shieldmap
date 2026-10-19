"""Re-render the stage 4b figures kept in the repository documentation.

These figures left the paper: the headroom prices (old Fig. S3),
the Z-exponents (old S4), the polymorph density spread (old S5), the cross-section library sensitivity (old S6), the
decomposition panel of Fig. 3, the 36-variant and all-estimator Monte Carlo panels of Fig. 4 (with the first-order
model markers), the e_hull panel of Fig. S1 and the full three-panel Monte Carlo check of Fig. S7. Nothing is lost:
``figures_phase4b.py`` is the stage 4b figure module verbatim (git: shieldmap/src/shieldmap/figstyle/figures.py at
commit 5895ffa); this script runs it with the output redirected here, as ``phase4b_<name>.pdf/.png``.

    .venv\\Scripts\\python.exe docs\\figures\\render_docs_figures.py            (from shieldmap/)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ("Fig1", "Fig2", "Fig3", "Fig4", "GraphicalAbstract", "FigS1", "FigS2", "FigS3", "FigS4", "FigS5", "FigS6",
         "FigS7")


def main() -> dict:
    spec = importlib.util.spec_from_file_location("figures_phase4b", HERE / "figures_phase4b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    H = mod.H
    H.use()
    mod.OUT = HERE
    mod.REP = HERE

    def _save(fig, name, **kw):
        files = H.export(fig, f"phase4b_{name}", HERE, formats=("pdf", "png"))
        import matplotlib.pyplot as plt
        plt.close(fig)
        return files
    mod._save = _save
    # the graphical abstract exports itself (H.export with a fixed name): wrap export for that one call
    orig_export = H.export

    def export(fig, name, outdir, **kw):
        kw["formats"] = ("pdf", "png")
        return orig_export(fig, name if name.startswith("phase4b_") else f"phase4b_{name}", HERE, **kw)
    H.export = export
    funcs = {"Fig1": mod.fig1, "Fig2": mod.fig2, "Fig3": mod.fig3, "Fig4": mod.fig4,
             "GraphicalAbstract": mod.graphical_abstract, "FigS1": mod.figS1, "FigS2": mod.figS2,
             "FigS3": mod.figS3, "FigS4": mod.figS4, "FigS5": mod.figS5, "FigS6": mod.figS6, "FigS7": mod.figS7}
    done = {}
    try:
        for nm in NAMES:
            print(f"[docs fig] {nm}", flush=True)
            try:
                funcs[nm]()
            except FileNotFoundError:
                pass          # graphical abstract re-opens its PNG under the paper name; the file itself is written
            done[nm] = f"phase4b_{nm}"
    finally:
        H.export = orig_export
    (HERE / "figure_numbers_phase4b.json").write_text(
        json.dumps({"figures": mod._num().rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    return done


if __name__ == "__main__":
    print(main())
