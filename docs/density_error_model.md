# Density-error model

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

Materials Project volumes are known to exceed measured ones (Pöhls et al. 2020; Hussein et al. 2022), so its densities are biased low.
Against 21 489 structure-matched COD compounds the median bias δ = ρ_MP/ρ_COD − 1 is −2.5 %, with class medians from −3.9 (halides) to −0.7 % (intermetallics) (Supplementary Fig. S1). [H302, H302:hi, H87]
The volume error is known to vary with bonding type and functional (Pöhls et al. 2020; Hinuma et al. 2017; Lejaeghere et al. 2013; Tran et al. 2016; Kingsbury et al. 2022); the breakdown by chemical class and distance from the hull in Supplementary Fig. S1 resolves it on this population.
Part of the bias is thermal expansion between the static lattice and room-temperature measurements (Hussein et al. 2022); the attribution is partial, because the calibration pools GGA-, GGA+U- and r²SCAN-relaxed structures, whose volume errors differ (Kingsbury et al. 2022).
The COD densities are crystallographic (X-ray) densities, and the temperature filter admits some low-temperature structures, which lowers the measured bias; neither the functional nor the temperature is a covariate of the model.

The model treats δ = ρ_MP/ρ_true − 1 per entry, so that the true thickness is the Materials Project thickness times (1 + δ).
Entries with a measured density take their own δ: unary entries that are plausibly the measured phase (same space group as the element's representative entry) use the experimental element density, and compounds with a COD structure match that passes a robust outlier screen use the COD density (Ong et al. 2013).
A replicate residual, the log deviation of structurally equivalent COD entries from their set median, is added to measured entries.
Every other entry draws δ from the screened COD distribution of its calibration cell: density bin × primary chemical class × stability for ICSD-matched entries, density bin × stability for theoretical ones, with coarser fallbacks for sparse cells.
Errors are correlated within a chemical system by a Gaussian copula (Nelsen 2006) with one shared shock per Materials Project chemical system; the intraclass correlation of the COD normal scores by chemical system is 0.381.
No draw makes an unmeasured compound denser than the densest measured element, 22.65 g/cm³.
The model is sampled with 5 000 draws per mode; sensitivity modes drop the correlation, change the outlier screen or use the unscreened residual.

The measured-density point recomputes every frontier after each density is replaced by its measured value where one exists, or by the median of its cell otherwise.
All four leaders of the main-text Table 2 have measured densities: Ir 22.65 and W 19.25 g/cm³ from the experimental element densities, BiOF 9.21 and HfO₂ 10.10 g/cm³ from screened COD matches. [H158, H159, H160, H161]
Areal-mass results do not depend on density by construction, since μ/ρ does not (Hubbell 1999).
On the thickness basis the B4-internal prices at Cs-137 and Co-60 move by up to 9.4 percentage points between Materials Project densities and the measured-density point, for the non-radioactive requirement (actinide headroom); among the main-text prices, the largest shift is 8.0 points (transparency at Cs-137). [H108, H108:lo]
Building on known DFT volume errors (Pöhls et al. 2020), we propagate a measured-density-conditioned density model into thickness prices; we found no prior study doing this for shielding.
P(best) is the share of draws in which a compound is the thinnest member of a set; Table 2 of the main text reports it for the leaders.

The density-bias figure is Supplementary Fig. S1 of the manuscript.

## Parameters (values from the code)

*Source: `src/shieldmap/analysis3/cod.py` and `src/shieldmap/analysis3/density_model.py`.*

**COD screening and matching (`analysis3/cod.py`).**
- CIF quality filters:
  - no COD error or retraction flag;
  - every site occupancy exactly 1 and no disorder assembly or group;
  - T ≤ 350 K or unspecified, and P ≤ 1 MPa or unspecified;
  - R ≤ 0.1 where given;
  - parsed density within 1 % of `_exptl_crystal_density_diffrn` where given.
- Structure matching uses pymatgen `StructureMatcher()` with its default tolerances (ltol 0.2, stol 0.3, angle 5°).
- One COD value is kept per Materials Project entry.
- Robust outlier screen: a pair is excluded when |ln(1+δ) − median_c| > 5 × 1.4826·MAD_c within its primary class c.
- Elements use the experimental element density where the unary entry is plausibly the measured phase (same space
  group as the element's representative entry).

**Calibration cells (`analysis3/density_model.py`).**
- ICSD-matched entries: ρ_MP bin × primary class × stability (e_hull ≤ 0.05 eV/atom or not). The ρ_MP bin edges are
  0, 4, 6, 8, 10, 12 and 15 g/cm³ (open above 15).
- Theoretical entries: ρ_MP bin × stability. The bin edges are 0, 4 and 8 g/cm³ (open above 8), and these entries
  have their own COD-matched theoretical distribution.
- A cell needs at least 100 COD pairs (`MIN_CELL`). Otherwise it falls back, in order, to bin × stability, then bin ×
  class, then bin.
- Primary class: an exclusive class by precedence f-electron > lanthanide > 3d-magnetic > halide > oxide >
  chalcogenide > intermetallic > other (`analysis3/common.primary_class`).
- Seed: 20261012 (density model).

## References

- Hinuma et al. 2017. Comparison of approximations in density functional theory calculations: Energetics and structure of binary oxides. Physical Review B 96. https://doi.org/10.1103/physrevb.96.094102
- Hubbell 1999. Review of photon interaction cross section data in the medical and biological context. Physics in Medicine and Biology 44, R1-R22. https://doi.org/10.1088/0031-9155/44/1/001
- Hussein et al. 2022. Machine-learning correction to density-functional crystal structure optimization. MRS Bulletin 47, 765-771. https://doi.org/10.1557/s43577-022-00310-9
- Kingsbury et al. 2022. Performance comparison of r$^2$SCAN and SCAN metaGGA density functionals for solid materials via an automated, high-throughput computational workflow. Physical Review Materials 6. https://doi.org/10.1103/physrevmaterials.6.013801
- Lejaeghere et al. 2013. Error Estimates for Solid-State Density-Functional Theory Predictions: An Overview by Means of the Ground-State Elemental Crystals. Critical Reviews in Solid State and Materials Sciences 39, 1-24. https://doi.org/10.1080/10408436.2013.772503
- Nelsen 2006. An Introduction to Copulas. https://doi.org/10.1007/0-387-28678-0
- Ong et al. 2013. Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. Computational Materials Science 68, 314-319. https://doi.org/10.1016/j.commatsci.2012.10.028
- Pöhls et al. 2020. Comparison of computational and experimental inorganic crystal structures. Journal of Solid State Chemistry 290, 121557. https://doi.org/10.1016/j.jssc.2020.121557
- Tran et al. 2016. Rungs 1 to 4 of DFT Jacob’s ladder: Extensive test on the lattice constant, bulk modulus, and cohesive energy of solids. The Journal of Chemical Physics 144. https://doi.org/10.1063/1.4948636
