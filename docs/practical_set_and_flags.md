# Practical set, artefact flag and hydrogen-containing compounds

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

Supplementary Table S1 defines every requirement, the practical sets, the price and the data sources.
Under rule R2 the practical set B4 holds 29 846 entries, B5 (B4 with transparency) 5 670 and B6 (B4 with the scarcity proxy) 16 267.
The price of a requirement X is P_X = q*(S_X)/q*(S₀) − 1, where q* is the frontier requirement (thickness or areal mass) of the set.
For the requirements that define B4 (non-radioactive, lead-free, stability, observation) S₀ is B4 without X and S_X is B4; for transparency, the scarcity proxy and the element-exclusion filters S₀ is B4 and S_X is B4 with X.
Prices against the unconstrained artefact-free frontier F0 are headroom, not prices of practical choices ([Headroom](headroom.md)).

Hypothetical database entries that are unlikely to be synthesised are a recognised problem of DFT databases (Sun et al. 2016; Aykol et al. 2018), and fixed energy-above-hull thresholds are imperfect proxies for synthesisability (Pöhls et al. 2020).
An entry is an artefact if any of three conditions holds: it is theoretical (no ICSD match) with e_hull > 0.1 eV/atom; its Materials Project density exceeds 1.05 times the densest measured element (22.65 g/cm³, Ir); or it is ICSD-tagged with e_hull > 0.3 eV/atom.
Artefacts are removed from every frontier kind used for the bands (best_clean, best_icsd, median50_clean).
The third condition also removes genuine compounds with known GGA energy errors, such as hydrates and peroxides; it affects only reference sets outside B4, so the headroom values, not the prices, depend on it.
The bands of the headline prices (in the dataset) come from these alternatives: the 9.3 % upper end of the observed price at Cs-137 (thickness) is set by the median50_clean frontier definition under the all-polymorph rule R3, and the 119 % lower end of the Co-60 transparency thickness price by rule R3 without coherent scattering. [H07:hi, H21:lo]
Because B4 requires an ICSD match and e_hull ≤ 0.05 eV/atom, it contains no artefact; the flag changes only the reference sets outside B4.
Without the flag, artefact entries set 100 % of the unconstrained best-compound thickness frontiers for the radionuclide profiles, 83 % of the areal frontiers for the diagnostic profiles, 77 % of the single-line thickness frontiers and 4.7 % of the single-line areal frontiers. [H75, H76, H77, H78]

The hydrogen-containing compounds need a separate treatment on the areal-mass basis.
Where Compton scattering dominates, μ/ρ scales with the electrons per gram, Z/A, so hydrogen-rich compounds set the mass frontier (Hubbell and Seltzer 1995; Hubbell 1999); this is textbook physics.
In B4 the mass frontier above about 0.8 MeV is NH₃·H₂O, which the database records as a crystal but which is a liquid at ambient conditions; elemental Hg, also a liquid, is in B4 as well.
The ICSD tag does not distinguish ambient from high-pressure or high-temperature structures: B4 contains the high-pressure Pbca form of HfO₂ (the transparent leader at Cs-137 under rule R1) and a cubic form of Bi, while the ambient phase β-Sn is excluded at 0.12 eV/atom above the hull. [H301]
Excluding hydrogen-containing compounds changes the B4 areal-mass frontier by up to 12 % on single lines and by 9.3 % at Co-60. [H106, H107]
The main text therefore quotes areal-mass prices at Co-60 on B4 without hydrogen-containing compounds (rows H133–H137), and the areal-mass curves of the main-text figures switch to that variant at 0.8 MeV.
On the full set the Co-60 areal-mass prices of lead-free, stability, observation, transparency and the scarcity proxy are all 0 % (rows H16–H24 in the headline table), because NH₃·H₂O stays the frontier whichever requirement is imposed. [H16, H18, H20, H22, H24]
Per unit mass Pb needs a median of 3.4 % more than the best compound of the full B4 set over 100 keV–10 MeV, with an upper end of 13 % where NH₃·H₂O is the frontier; without hydrogen-containing compounds the range is 0.9–3.5 %. [H72, stmt:H72, H72:hi, H139:lo, H139:hi]

## References

- Aykol et al. 2018. Thermodynamic limit for synthesis of metastable inorganic materials. Science Advances 4. https://doi.org/10.1126/sciadv.aaq0148
- Hubbell 1999. Review of photon interaction cross section data in the medical and biological context. Physics in Medicine and Biology 44, R1-R22. https://doi.org/10.1088/0031-9155/44/1/001
- Hubbell and Seltzer 1995. Tables of x-ray mass attenuation coefficients and mass energy-absorption coefficients 1 keV to 20 MeV for elements Z = 1 to 92 and 48 additional substances of dosimetry interest. https://doi.org/10.6028/nist.ir.5632
- Pöhls et al. 2020. Comparison of computational and experimental inorganic crystal structures. Journal of Solid State Chemistry 290, 121557. https://doi.org/10.1016/j.jssc.2020.121557
- Sun et al. 2016. The thermodynamic scale of inorganic crystalline metastability. Science Advances 2. https://doi.org/10.1126/sciadv.1600225
