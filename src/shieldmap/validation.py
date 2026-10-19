"""Validation computations behind the validation reports.

python -m shieldmap.validation step2   → reports/phase1/data/step2_*.csv
python -m shieldmap.validation step4   → reports/phase1/data/step4_*.csv
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from shieldmap import config, grid, paths
from shieldmap.physics import mixture, response, spectra, transmission

DATA = paths.REPORTS / "data"
XDIR = paths.RAW / "xcom" / "extracted" / "XCOM"


def _els():
    from shieldmap.ingest import xsec
    return xsec.load_elements(XDIR)


def _srd():
    return pd.read_csv(paths.FIXTURES / "srd126_mu_rho.csv")


def srd_value(df, material, e_mev, col="mu_rho_cm2_g"):
    d = df[(df.material == material) & df.edge_label.isna()].sort_values("energy_MeV")
    hit = d[np.isclose(d.energy_MeV, e_mev, rtol=1e-9)]
    if len(hit):
        return float(hit[col].iloc[0]), "tabulated"
    return float(np.exp(np.interp(np.log(e_mev), np.log(d.energy_MeV), np.log(d[col])))), "log-log interp"


def compound_mu_rho(els, w: np.ndarray, e_ev: np.ndarray, key="total_wcoh") -> np.ndarray:
    out = np.zeros_like(e_ev, dtype=float)
    for z in np.nonzero(w)[0]:
        out += w[z] * els[z + 1].mu_rho(e_ev)[key]
    return out


def step2_elements(els, srd) -> pd.DataFrame:
    rows = []
    for Z, mat in ((82, "Pb"), (74, "W"), (26, "Fe"), (13, "Al")):
        for e in (60.0, 100.0, 600.0, 661.657, 800.0, 1000.0):
            ref, how = srd_value(srd, mat, e / 1000)
            ours = float(els[Z].mu_rho(np.array([e * 1000]))["total_wcoh"][0])
            rows.append({"element": mat, "energy_keV": e, "srd126": ref, "srd126_basis": how,
                         "ours": ours, "rel_diff": ours / ref - 1, "pass_2pct": abs(ours / ref - 1) <= 0.02})
    return pd.DataFrame(rows)


def step2_water(els, srd) -> pd.DataFrame:
    d = srd[(srd.material == "H2O") & srd.edge_label.isna() & (srd.energy_MeV >= 0.01) & (srd.energy_MeV <= 15)]
    w = mixture.fractions_from_formula("H2O")
    ours = compound_mu_rho(els, w, d.energy_MeV.to_numpy() * 1e6)
    return pd.DataFrame({"energy_keV": d.energy_MeV.to_numpy() * 1000, "srd126": d.mu_rho_cm2_g.to_numpy(),
                         "ours": ours, "rel_diff": ours / d.mu_rho_cm2_g.to_numpy() - 1})


def step2_coherent(els) -> pd.DataFrame:
    rows = []
    for name, formula in (("Pb", "Pb"), ("WO3", "WO3")):
        w = mixture.fractions_from_formula(formula)
        for e in (20.0, 60.0, 100.0, 661.657):
            a = compound_mu_rho(els, w, np.array([e * 1000]), "total_wcoh")[0]
            b = compound_mu_rho(els, w, np.array([e * 1000]), "total_wocoh")[0]
            rows.append({"material": name, "energy_keV": e, "mu_rho_wcoh": a, "mu_rho_wocoh": b,
                         "coherent_share": (a - b) / a})
    return pd.DataFrame(rows)


def _hvl_al(mu_al: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x1 = transmission.solve_thickness(mu_al[None, :], w, 0.5)[0]
    x2 = transmission.solve_thickness(mu_al[None, :], w, 0.25)[0]
    return x1 * 10, (x2 - x1) * 10                         # mm


def step2_tubes(els) -> pd.DataFrame:
    """Per tube spectrum: our air-kerma HVL in Al (XCOM, ρ = 2.70 g/cm³) vs SpekPy; quadrature checks.

    * edge-split quadrature (production) vs SpekPy's own HVL
    * the same with log-log-linear air μ_en/ρ (interpolation sensitivity)
    * required thickness of test absorbers for T = T_Pb(0.35 mm): production quadrature vs
      (a) the plain 0.5 keV grid mapping and (b) a 5× finer subdivision (convergence)
    """
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet")
    rho_al = float(el.loc[el.symbol == "Al", "density_solid_exp"].iloc[0])
    rho_pb = float(el.loc[el.symbol == "Pb", "density_solid_exp"].iloc[0])
    e_grid = grid.energy_grid_keV()
    absorbers = ("BaSO4", "Gd2O3", "Bi", "Sn", "WO3", "CeO2", "Pb")
    rows = []

    def setup(energy, fluence, air="spline"):
        if air == "spline":
            w = fluence * response.response("air_kerma", energy)
        else:
            w = fluence * energy * response.air_muen_rho(energy, "loglog_linear")
        mu_pb = rho_pb * els[82].mu_rho(energy * 1000)["total_wcoh"]
        t_star = math.exp(transmission.log_transmission(mu_pb[None], w, np.array([0.035]))[0])
        return w, t_star

    def areal(energy, w, t_star, formula):
        wf = mixture.fractions_from_formula(formula)
        return transmission.solve_thickness(compound_mu_rho(els, wf, energy * 1000)[None], w, t_star)[0]

    for q in config.get("xray_tubes.qualities"):
        for ang in config.get("xray_tubes.anode_angle_deg"):
            sp = spectra.tube_spectrum(q, float(ang))
            fine = spectra.tube_spectrum(q, float(ang), subdivide=5)
            mu_al = rho_al * els[13].mu_rho(sp.energy_keV * 1000)["total_wcoh"]
            hvl1, hvl2 = _hvl_al(mu_al, sp.weights("air_kerma"))
            w_ll = sp.fluence * sp.energy_keV * response.air_muen_rho(sp.energy_keV, "loglog_linear")
            hvl1_ll, _ = _hvl_al(mu_al, w_ll)
            nodes, flu_g = spectra.map_to_grid(sp._k, sp._phi, e_grid)
            eg = e_grid[nodes]
            row = {"spectrum": sp.name, **{k_: sp.meta[k_] for k_ in (
                "kV", "anode_angle_deg", "total_al_filtration_mm", "hvl1_target_mm_al", "spekpy_hvl1_mm_al",
                "spekpy_hvl2_mm_al", "spekpy_homogeneity", "standard_homogeneity", "spekpy_mean_energy_keV",
                "dropped_fluence_fraction", "dropped_air_kerma_fraction", "n_cells", "n_pieces")},
                "our_hvl1_mm_al": hvl1, "our_hvl2_mm_al": hvl2, "our_homogeneity": hvl1 / hvl2,
                "our_vs_spekpy_hvl1_rel": hvl1 / sp.meta["spekpy_hvl1_mm_al"] - 1,
                "our_hvl1_loglog_air_mm_al": hvl1_ll,
                "loglog_air_vs_spekpy_hvl1_rel": hvl1_ll / sp.meta["spekpy_hvl1_mm_al"] - 1}
            w_q, t_q = setup(sp.energy_keV, sp.fluence)
            w_f, t_f = setup(fine.energy_keV, fine.fluence)
            w_g, t_g = setup(eg, flu_g)
            row["T_pb035_quadrature"] = t_q
            row["T_pb035_fine_rel"] = t_f / t_q - 1
            row["T_pb035_gridmap_rel"] = t_g / t_q - 1
            for f in absorbers:
                xq = areal(sp.energy_keV, w_q, t_q, f)
                row[f"fine_vs_quad_{f}"] = areal(fine.energy_keV, w_f, t_f, f) / xq - 1
                row[f"gridmap_vs_quad_{f}"] = areal(eg, w_g, t_g, f) / xq - 1
            rows.append(row)
    return pd.DataFrame(rows)


def step2_air_interp() -> dict:
    e = grid.energy_grid_keV()
    a = response.air_muen_rho(e, "spline"); b = response.air_muen_rho(e, "loglog_linear")
    r = np.abs(a / b - 1)
    return {"max_rel_diff_spline_vs_loglog": float(r.max()), "at_keV": float(e[r.argmax()])}


def step2_xraylib(els, n=50, seed=20261011) -> pd.DataFrame:
    """Independent implementation check: xraylib CS_Total_CP (different compilation) for 50 random MP compounds."""
    import xraylib
    comp = pd.read_parquet(paths.INTERIM / "mp" / "compounds.parquet",
                           columns=["material_id", "formula_reduced"])
    W = __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(paths.INTERIM / "mp" / "mass_fractions.npz").tocsr()
    from shieldmap.ingest.xsec import atomic_masses
    masses = atomic_masses()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(comp), n, replace=False)
    rows = []
    for i in pick:
        w = W[i].toarray().ravel()
        f = comp.formula_reduced.iloc[i]
        for e in (20.0, 60.0, 100.0, 661.657, 1000.0, 5000.0):
            ours = compound_mu_rho(els, w, np.array([e * 1000]))[0]
            try:
                # per-atom comparison: rescale xraylib's per-gram values from its own atomic weight
                # to the mass used for our mass fractions
                xr = sum(w[z] * xraylib.CS_Total(z + 1, e) * xraylib.AtomicWeight(z + 1) / masses[z]
                         for z in np.nonzero(w)[0])
            except Exception as ex:          # pragma: no cover
                xr = np.nan
            rows.append({"material_id": comp.material_id.iloc[i], "formula": f, "energy_keV": e,
                         "ours_xcom": ours, "xraylib_CS_Total": xr, "rel_diff": ours / xr - 1})
    return pd.DataFrame(rows)


def step4_density_bias() -> pd.DataFrame:
    """MP density of each element's representative (lowest e_hull) entry vs experimental solid density."""
    comp = pd.read_parquet(paths.INTERIM / "mp" / "compounds.parquet",
                           columns=["material_id", "formula_reduced", "nelements", "density", "energy_above_hull",
                                    "is_representative", "theoretical", "spacegroup_number", "crystal_system",
                                    "is_metal"])
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet")
    rep = comp[(comp.nelements == 1) & comp.is_representative].copy()
    from pymatgen.core import Composition
    rep["symbol"] = [Composition(f).elements[0].symbol for f in rep.formula_reduced]
    m = rep.merge(el[["Z", "symbol", "density_solid_exp"]], on="symbol", how="left")
    m = m[m.density_solid_exp.notna()].sort_values("Z")
    m["rel_bias"] = m.density / m.density_solid_exp - 1
    return m


def step4_compounds(els, srd) -> pd.DataFrame:
    """μ/ρ (implementation check vs SRD 126, off-edge) and MP density of matching phases vs SRD 126 Table 2."""
    mats = pd.read_csv(paths.FIXTURES / "srd126_materials.csv")
    comp = pd.read_parquet(paths.INTERIM / "mp" / "compounds.parquet",
                           columns=["material_id", "formula_reduced", "density", "energy_above_hull",
                                    "is_representative", "theoretical", "spacegroup_number"])
    from pymatgen.core import Composition
    rows = []
    for mat in sorted(set(srd[srd.kind == "compound"].material) - {"H2O"}):
        d = srd[(srd.material == mat) & srd.edge_label.isna() & (srd.energy_MeV >= 0.01) & (srd.energy_MeV <= 15)]
        # off-edge: exclude tabulated energies within 1 % of any constituent edge
        w = mixture.fractions_from_formula(mat)
        edges = np.array([en for z in np.nonzero(w)[0] for _, en in els[z + 1].edges_keV()])
        ek = d.energy_MeV.to_numpy() * 1000
        off = np.array([edges.size == 0 or np.min(np.abs(x / edges - 1)) > 0.01 for x in ek])
        ours = compound_mu_rho(els, w, ek * 1000)
        rel = ours / d.mu_rho_cm2_g.to_numpy() - 1
        red = Composition(mat).reduced_formula
        cand = comp[comp.formula_reduced == red].sort_values("energy_above_hull")
        mrow = mats[mats.material == mat]
        rho_ref = float(mrow.density_g_cm3.iloc[0]) if len(mrow) else np.nan
        rep = cand[cand.is_representative]
        icsd = cand[~cand.theoretical]
        bad = np.abs(rel) > 0.02
        if len(icsd) and np.isfinite(rho_ref):
            j = (icsd.density - rho_ref).abs().idxmin()
            close_id, close_rho, close_sg = icsd.loc[j, "material_id"], float(icsd.loc[j, "density"]), int(icsd.loc[j, "spacegroup_number"])
        else:
            close_id, close_rho, close_sg = None, np.nan, None
        rows.append({
            "material": mat, "n_energies_off_edge": int(off.sum()),
            "n_off_edge_fail_2pct": int((bad & off).sum()),
            "off_edge_fail_energies_keV": ";".join(f"{x:.4f}" for x in ek[bad & off]),
            "closest_icsd_matched_id": close_id, "closest_icsd_matched_density": close_rho,
            "closest_icsd_matched_spacegroup": close_sg,
            "closest_icsd_rel_diff": close_rho / rho_ref - 1 if close_id else np.nan,
            "polymorph_ambiguous": bool(len(cand) > 1),
            "max_abs_rel_mu_rho_off_edge": float(np.max(np.abs(rel[off]))),
            "median_abs_rel_mu_rho_off_edge": float(np.median(np.abs(rel[off]))),
            "max_abs_rel_mu_rho_all": float(np.max(np.abs(rel))),
            "srd126_density": rho_ref, "n_mp_polymorphs": len(cand),
            "mp_representative": rep.material_id.iloc[0] if len(rep) else None,
            "mp_rep_density": float(rep.density.iloc[0]) if len(rep) else np.nan,
            "mp_rep_ehull": float(rep.energy_above_hull.iloc[0]) if len(rep) else np.nan,
            "mp_rep_theoretical": bool(rep.theoretical.iloc[0]) if len(rep) else None,
            "mp_density_min": float(cand.density.min()) if len(cand) else np.nan,
            "mp_density_max": float(cand.density.max()) if len(cand) else np.nan,
            "n_icsd_matched": int(len(icsd)),
            "rep_density_rel_diff": (float(rep.density.iloc[0]) / rho_ref - 1) if len(rep) else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    which = (argv or sys.argv[1:] or ["step2"])[0]
    paths.ensure(DATA)
    els, srd = _els(), _srd()
    if which == "step2":
        step2_elements(els, srd).to_csv(DATA / "step2_elements.csv", index=False)
        step2_water(els, srd).to_csv(DATA / "step2_water.csv", index=False)
        step2_coherent(els).to_csv(DATA / "step2_coherent.csv", index=False)
        step2_tubes(els).to_csv(DATA / "step2_tubes.csv", index=False)
        pd.Series(step2_air_interp()).to_csv(DATA / "step2_air_interp.csv")
        step2_xraylib(els).to_csv(DATA / "step2_xraylib.csv", index=False)
    elif which == "step4":
        step4_density_bias().to_csv(DATA / "step4_density_bias.csv", index=False)
        step4_compounds(els, srd).to_csv(DATA / "step4_compounds.csv", index=False)
    else:
        raise SystemExit(f"unknown target {which}")


if __name__ == "__main__":
    main()


def step3_tableb(figdir=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = paths.ensure(figdir or (paths.REPORTS / "figures"))
    t = pd.read_parquet(paths.INTERIM / "mp" / "compounds.parquet")
    el = pd.read_parquet(paths.INTERIM / "elements" / "elements.parquet")
    present = sorted({s for lst in t["elements"] for s in lst})
    no_ab = el[el.crustal_abundance_ppm.isna()].symbol.tolist()
    counts = {
        "total": len(t), "representative": int(t.is_representative.sum()),
        "theoretical_false": int((~t.theoretical).sum()), "is_stable": int(t.is_stable.sum()),
        "ehull_le_0.05": int((t.energy_above_hull <= 0.05).sum()),
        "representative_and_ehull_le_0.05": int((t.is_representative & (t.energy_above_hull <= 0.05)).sum()),
        "ehull_null": int(t.energy_above_hull.isna().sum()), "band_gap_null": int(t.band_gap.isna().sum()),
        "band_gap_ge_3.1": int((t.band_gap >= 3.1).sum()),
        "contains_radioactive": int(t.contains_radioactive.sum()), "contains_noble_gas": int(t.contains_noble_gas.sum()),
        "contains_Pb": int(t.contains_Pb.sum()),
        "abundance_proxy_null": int(t.min_crustal_abundance_ppm.isna().sum()),
        "density_lt_0.5": int((t.density < 0.5).sum()), "density_max": float(t.density.max()),
        "density_median": float(t.density.median()),
        "n_elements_present": len(present),
        "elements_without_abundance": no_ab,
        "elements_without_abundance_present_in_MP": [s for s in no_ab if s in present],
        "compounds_per_missing_abundance_element": {s: int(t[f"contains_{s}"].sum()) for s in no_ab if f"contains_{s}" in t},
    }
    for col, xlab, fn, rng in (("density", "DFT density (g/cm³)", "step3_density_hist.png", (0, 25)),
                               ("band_gap", "PBE band gap (eV)", "step3_bandgap_hist.png", (0, 12))):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        bins = np.linspace(*rng, 101)
        ax.hist(t[col].dropna().clip(*rng), bins=bins, color="#4C72B0", alpha=0.85, label="all entries")
        ax.hist(t.loc[t.is_representative, col].dropna().clip(*rng), bins=bins, histtype="step", color="#C44E52",
                lw=1.2, label="representative")
        if col == "band_gap":
            ax.set_yscale("log")
            for g in (2.5, 3.1, 3.5):
                ax.axvline(g, color="0.3", lw=0.8, ls="--")
        ax.set_xlabel(xlab); ax.set_ylabel("entries"); ax.legend(frameon=False)
        ax.set_title("Table B: Materials Project 2026.04.13 (non-deprecated)", fontsize=10)
        fig.tight_layout(); fig.savefig(figdir / fn, dpi=130); plt.close(fig)
    hist = {}
    for col, edges in (("density", [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 25]),
                       ("band_gap", [0, 1e-9, 0.5, 1, 2, 2.5, 3.1, 3.5, 5, 8, 20])):
        h, _ = np.histogram(t[col].dropna(), bins=edges)
        hist[col] = {f"[{a:g},{b:g})": int(c) for a, b, c in zip(edges[:-1], edges[1:], h)}
    counts["histograms"] = hist
    return counts
