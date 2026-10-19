# Element-exclusion filters inside B4

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

Element-exclusion filters remove one element from B4 at a time; 69 elements are priced: every element in a B4 top 10 at any energy, a hazard list (Tl, As, Cd, Hg, Be, Se) and a load-bearing list (Supplementary Fig. S2).
At the radionuclide lines the largest single-element prices are small: excluding Ir adds 2.1 % to the thickness at Cs-137 (Ir to Os, at the measured densities 22.65 and 22.61 g/cm³) and 1.3 % at Co-60, and excluding Bi adds 3.8 % to the areal mass at Cs-137 (Bi to Tl). [H37, H39, H38, H317]
The largest values sit just above heavy-element K edges in narrow beam.
On the thickness basis, excluding Ta adds up to 29 %, Hf up to 27 % and W up to 19 %; on the areal-mass basis the same filters add up to 3.4, 2.7 and 3.1 %. [H60, H62, H58, H59, H61, H57]
These maxima sit at the grid node just above the K edge, where single-node values carry several points of cross-section-library uncertainty ([Cross-section library sensitivity](cross_section_sensitivity.md)).
Under the first-order fluorescence correction, at Materials Project densities, the W, Ta and Hf windows on the thickness basis remain at 18, 12 and 6.9 % (first-order model, caveat in [Fluorescence model](fluorescence_model.md)). [H130, H131, H132]
K-edge windows of single heavy elements and the value of complementary edges are known (Yaffe et al. 1991; McCaffrey et al. 2009; Zeng et al. 2026); the map prices excluding each element inside the practical set.
We found no prior analysis of how many elements a shielding frontier depends on; the map reports where the element-exclusion windows lie rather than a single redundancy figure.

The element-exclusion map is Supplementary Fig. S2 of the manuscript.

## References

- McCaffrey et al. 2009. Optimizing non‐Pb radiation shielding materials using bilayers. Medical Physics 36, 5586-5594. https://doi.org/10.1118/1.3260839
- Yaffe et al. 1991. Composite Materials for X-ray Protection. Health Physics 60, 661-664. https://doi.org/10.1097/00004032-199105000-00004
- Zeng et al. 2026. Design and optimization of lead-free composite shielding for 40-200 keV medical X-rays. Applied Radiation and Isotopes 234, 112647. https://doi.org/10.1016/j.apradiso.2026.112647
