# Monte Carlo spot check: methods, validation and results

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

The spot check uses OpenMC 0.16.0 (Romano et al. 2015; Lund and Romano 2018) with ENDF/B-VIII.0 photoatomic and atomic-relaxation data (Brown et al. 2018) (Table S2).
Photons are transported with coherent scattering, Doppler-broadened incoherent scattering, photoelectric absorption with full atomic relaxation and pair production; electrons use the thick-target bremsstrahlung approximation, and the photon cutoff is 1 keV.
The slab is laterally infinite (reflective planes at x, y = ± 5 cm) under a broad parallel beam at normal incidence, and each photon crossing the back face adds its air-kerma contribution.
Densities are those of the measured-density point: measured for Ir, BiOF, HfO₂, Bi and Pb, and density-model medians for PrSbPd, Pr₃OsI₃ and HfAu₂ (areal-mass results do not depend on them).
Slab thicknesses are multiples of the map's headline requirement and bracket the broad-beam target; requirements at the target are interpolated in log T by a quadratic through the three nearest points, every requirement falls inside the simulated bracket, requirement standard errors come from a 2 000-draw parametric bootstrap, and the like-for-like advantages from a 4 000-draw joint bootstrap that includes the Pb target.
The fluorescence-inclusive estimator counts every back-face photon whose energy equals an ENDF radiative transition energy of a constituent and is at least 3.2 keV, of any generation and including Rayleigh-scattered lines; Compton-scattered line photons count only in the broad-beam estimator.
Every estimator weights a crossing by 1/|u_z|; a grazing-angle cut changes the broad-beam values by at most about one per cent.
The native track-length tally agrees with the crossing-record estimator, but in a void the two are mathematically identical, so this checks the record reduction, not the estimator.
Each point has 10–40 independent sub-runs with distinct seeds, and uncertainties are standard errors over sub-runs. [H315:lo, H315:hi]
Cases c1–c6 of Supplementary Table S2 use 78 slab points, 1 725 sub-runs and 7 591 million photon histories; the reference-beam case c7, added in revision, uses 22 slab points, 236 sub-runs and 752 million histories, with the same model, physics, estimators and statistics.
OpenMC 0.16 reads an incident-neutron file for every nuclide of a material even in a photon-only run; for the nuclides of Ce, Ag, Sn, Ba, Sm and Gd, which no earlier case used, c7 uses placeholder files with the structure of an existing neutron file and each nuclide's own Z, A and atomic-weight ratio. The neutron data are never sampled, the atomic-weight ratio enters only the conversion of mass fractions to atom densities, and the narrow-beam chi-square of c7 confirms those densities; the script that writes the placeholders (`src/shieldmap/analysis4b/mc_rev.py`, command `wsl-stubs`) is in the repository.


### Validation

The narrow-beam estimator agrees with the analytic narrow-beam transmission computed from the same ENDF cross sections at every slab, with a chi-square of 72.5 for 78 points (reduced 0.93), and of 21.0 for the 22 points of the reference-beam case; this is a consistency check of the sampling, not an external validation. [H174, H316, H279]
The remaining differences from the map are the XCOM–EPDL cross-section difference (the ENDF/B-VIII.0 release paper describes its photoatomic and atomic-relaxation sublibraries as ENDF/B-VI.8 with the EPICS2014 corrections (Brown et al. 2018; Cullen 2015); the NNDC release page (National Nuclear Data Center 2018) lists them as updated in the EPICS2017 project (Cullen 2019; Cullen 2018)), largest just above the Pb K edge (Supplementary Fig. S3).
The build-up factor is B = broad/narrow, with the strictly uncollided kerma in the denominator and coherent scattering counted as a collision, at the narrow-beam slab for T = 0.01 (4.6 mean free paths).
Coherently scattered primaries alone carry 0.23–0.29 of the uncollided kerma at Cs-137 and 0.10–0.13 at Co-60, so these factors are not comparable with point-isotropic tabulations: the moments-method calculations behind the standard reference data did not account for coherent scattering, whose effect the working group then examined (Trubey 1988), and coherent scattering changes build-up factors appreciably (Hirayama and Trubey 1988; Harima 1993). [H303, H303:hi, H304, H304:hi]
In the Pb validation case at Cs-137 the build-up factor at the headline thickness is 2.29, rising from 1.81 to 3.05 over the simulated thicknesses (Supplementary Fig. S3). [H124, H124:lo, H124:hi]
Build-up depends on the absorber's atomic number (Sakamoto et al. 1988; HARIMA et al. 1991; Trubey 1988; Trubey 1991), so narrow-beam comparisons between low-Z and high-Z absorbers omit a material-dependent factor; the spot check isolates it for the headline pairs.
At the headline thickness the build-up factors are 2.44 (Ir), 2.73 (HfO₂) and 2.36 (BiOF) at Cs-137, and 2.84 (Ir) and 3.12 (HfO₂) at Co-60. [H125, H126, H127, H128, H129]
With coherently scattered primaries counted in the denominator, B' =  broad/(uncollided + coherent-scattered primary) is 1.77 for Pb, 1.90 (Ir), 2.21 (HfO₂) and 1.87 (BiOF) at Cs-137, and 2.50 (Ir) and 2.83 (HfO₂) at Co-60. [H246, H247, H248, H249, H250, H251]


### Transparency price

At Cs-137 the transparency thickness price (Ir to BiOF) is 135 % in narrow beam and 133 % with build-up; fluorescence adds less than one per cent of the uncollided kerma at these energies. [H148, H146]
At Co-60 (Ir to HfO₂) it is 130 % and 135 %. [H149, H147]
HfO₂ needs 148 % more thickness than Ir at Cs-137 with transported fluorescence (148–154 % over estimators), and 5.8 % more than BiOF (9.3 % with build-up). [H110, H110:lo, H110:hi, H111, H111:hi]


### Diagnostic advantage

The lanthanide compounds simulated are LaBi (4 ICSD entries) at RQR5, Pr₃OsI₃ (1 entry) at RQR8, and Ce₅AgSn₃ and PrSbPd (1 entry each) at the reference beam; the ten narrow-beam variant leaders are PrSbPd, Er₆IrI₁₀, Ce₅AgSn₃, Tb₆IrI₁₀ and further reduced iodide clusters and rare-earth-rich intermetallics, most with a single ICSD entry (CeSnRh and Nd₅Sn₄ have 3, Nd₁₁Sn₁₀ has 2).
The R₃I₃Z chain and R₆I₁₀Z cluster iodides are made in sealed Ta containers and handled under inert gas (Payne et al. 1992; Hughbanks and Corbett 1989; Park et al. 1997), and the rare-earth-rich intermetallics are moisture-sensitive; none is proposed as a shielding filler.
For the textbook and radionuclide leaders the ICSD counts are larger: Ir 7, W 14, Bi 8, BiOF 2 and HfO₂ 11.
With the like-for-like target, Pb's own transmission under the same estimator at its nominal areal mass, Bi needs 6.8 % (fluorescence-inclusive) and 10 % (broad-beam) more areal mass than LaBi at RQR5, and 17 % and 23 % more than Pr₃OsI₃ at RQR8; with the reference beam below they give the headline range of 6.8–22.9 %. [H318, H318:hi, H189, H190, H191, H192, H195, H195:hi]
Like-for-like, the narrow-beam values are 15 % and 41 %, and Pb needs 3.8 % and 3.3 % (RQR5) and 2.1 % and 2.6 % (RQR8) more than Bi under the fluorescence-inclusive and broad-beam estimators. [H193, H194, H196, H197, H198, H199]
The fixed-target values below use the narrow-beam transmission of the Pb thickness as the target for every estimator; they are secondary.
At RQR5 behind 0.25 mm Pb (12°), Bi needs 15 % more areal mass than LaBi in narrow beam, 8.5 % with transported fluorescence and 13 % with the full broad-beam kerma. [H141, H112, H142]
At RQR8 behind 0.35 mm Pb (12°), Bi needs 42 %, 20 % and 30 % more than Pr₃OsI₃. [H143, H118, H144]
Pb needs 13 % more than LaBi and 23 % more than Pr₃OsI₃ with transported fluorescence. [H113, H119]
Pb needs 3.8 % (RQR5) and 1.8 % (RQR8) more than Bi with transported fluorescence; over all estimators the ranges are 3.5–4.3 % and 1.6–2.6 %. [H116, H120, H116:lo, H116:hi, H120:lo, H120:hi]
The Sb/Pd-partnered narrow-beam leader at RQR5, PrSbPd, keeps 1.5 % over Bi with transported fluorescence (up to 21 % over estimators), and LaBi beats it by 6.9 %. [H114, H114:hi, H115]
With the fixed target the fluorescence-inclusive and broad-beam advantages of the best lanthanide compound over Bi therefore span 8.5–30 % at RQR5 and RQR8. [H145, H145:hi]
A 1° acceptance cone, the small-acceptance limit, gives 15 % and 42 %, the same as narrow beam; adding fluorescence lines below 3.2 keV raises the fluorescence-inclusive values to 10 % and 23 %. [H253, H256, H252, H255]
For Bi and Pb 75–100 % of the escaping fluorescence kerma is L radiation below 16 keV, which air kerma weights far more than H_p(10) (International Commission on Radiological Protection 1996); an H_p(10)-weighted comparison would therefore favour the lanthanide compounds further. [H305, H305:hi]
Rare-earth-based lead-free X-ray shields have been reported to save mass relative to Pb (Wang et al. 2020), non-Pb bilayers with complementary edges save weight (McCaffrey et al. 2009), and measured garment data that include scatter and fluorescence report shrinking mass savings at higher tube voltages (Schöpf and Pichler 2016).


### Reference beam

The reference-beam case (RQR6, 12°, 0.35 mm Pb) simulates Pb, Bi, the narrow-beam frontier crystal Ce₅AgSn₃, the leader under the fluorescence model PrSbPd, the best fully oxidised lanthanide compound with at least three ICSD entries, Ba₂SmSbO₆ (3 entries), and the Sn/Ba/Gd blend of [Blends, fully oxidised filter and transparency threshold](blends_oxidised_threshold.md) as an ideal homogeneous mixture.
Areal-mass results do not depend on the densities used: in a laterally infinite slab at normal incidence the transport scales with the areal mass.
Like-for-like, Bi needs 9.1 % (fluorescence-inclusive) and 14 % (broad-beam) more areal mass than Ce₅AgSn₃, 12 % and 17 % more than PrSbPd and 12 % and 17 % more than the blend (range 9.1–16.9 %, inside the 6.8–22.9 % of the other two beams). [H260, H261, H264, H265, H272, H273, H278, H278:hi, H195, H195:hi]
With the fixed narrow-beam target the same advantages are 12 % and 21 % (Ce₅AgSn₃), 14 % and 22 % (PrSbPd) and 15 % and 24 % (blend). [H262, H263, H266, H267, H274, H275]
Ba₂SmSbO₆ needs 11 % less areal mass than Bi in narrow beam, but like-for-like Bi needs 7.3 % (fluorescence-inclusive) and 3.5 % (broad-beam) less than Ba₂SmSbO₆; with the fixed target Bi needs 4.5 % less with transported fluorescence, and the oxide keeps 2.7 % only under the broad-beam estimator. [H215, H268, H269, H270, H271]
The blend needs 2.3 % less areal mass than Ce₅AgSn₃ with transported fluorescence (0.45 % less in narrow beam), and Ba₂SmSbO₆ needs 18 % more (21 % in narrow beam), so the penalty of the fully oxidised lanthanide compound against the frontier survives transport. [H276, H180, H277, H214]
The blend omits a polymer matrix and particle-size effects, and the case covers one beam and one geometry.


### Pb K-edge window

At 89.5 keV, with transported fluorescence, Tl needs 2.2 % and HfAu₂ 22 % less areal mass than Pb, and HfAu₂ needs 20 % less than Tl. [H121, H122, H123]
In narrow beam the window exists under both libraries, but its size is 2.8 % with XCOM and 2.2 % with ENDF/B-VIII.0. [H63, H121:hi]
Pb's advantage just above its K edge is expected from its absorption edge (Wang et al. 2020).

## Sub-run sums and inputs

- Case definitions: `data/phase4a/mc/inputs/cases.json`, `data/phase4a/mc/inputs/diag_cases.json`, `data/phase4b/mc_rev/inputs/cases.json` (case c7).
- Per-point results with sub-run counts and histories: `data/phase4a/mc/points.csv`, `data/phase4a/mc/diag_points.csv`, `data/phase4b/mc_rev/points.csv`; totals in `data/phase4a/mc/summary.json`, `data/phase4a/mc/diag_summary.json`, `data/phase4b/mc_rev/summary.json`. Supplementary Table S2(a) prints the per-case sums.
- Derived metrics: `data/phase4a/mc/metrics.csv`; like-for-like advantages: `data/phase4b/revision/e3_like_for_like.csv`; alternative build-up definition: `data/phase4b/revision/o5_buildup_alt.csv`.
- Placeholder incident-neutron files for case c7 (Ce, Ag, Sn, Ba, Sm, Gd): written by `src/shieldmap/analysis4b/mc_rev.py`, command `wsl-stubs`.
- The Monte Carlo back-face records, from which every Monte Carlo table is regenerated, are archived with the release (Zenodo deposit; see the manuscript's Data availability statement).

The full Monte Carlo figure (transmission, consistency check and build-up per case) is [`figures/phase4b_FigS7.png`](figures/phase4b_FigS7.png); its simplified version is Supplementary Fig. S3.

## References

- Brown et al. 2018. ENDF/B-VIII.0: The 8th Major Release of the Nuclear Reaction Data Library with CIELO-project Cross Sections, New Standards and Thermal Scattering Data. Nuclear Data Sheets 148, 1-142. https://doi.org/10.1016/j.nds.2018.02.001
- Cullen 2015. EPICS2014: Electron Photon Interaction Cross Sections (Version 2014). https://doi.org/10.61092/iaea.t20f-v6fw
- Cullen 2018. A Survey of Photon Cross Section Data for use in EPICS2017. https://doi.org/10.61092/iaea.1371-x20k
- Cullen 2019. EPICS2017: April 2019 Status Report. https://doi.org/10.61092/iaea.nw3x-we53
- HARIMA et al. 1991. Development of New Gamma-Ray Buildup Factor and Application to Shielding Calculations. Journal of Nuclear Science and Technology 28, 74-84. https://doi.org/10.1080/18811248.1991.9731324
- Harima 1993. An historical review and current status of buildup factor calculations and applications. Radiation Physics and Chemistry 41, 631-672. https://doi.org/10.1016/0969-806x(93)90317-n
- Hirayama and Trubey 1988. Effects of Incoherent and Coherent Scattering on the Exposure Buildup Factors of Low-Energy Gamma Rays. Nuclear Science and Engineering 99, 145--156. https://doi.org/10.13182/nse88-a23555
- Hughbanks and Corbett 1989. Encapsulation of heavy transition metals in iodide clusters. Synthesis, structure, and bonding of the unusual cluster phase yttrium iodide-ruthenium (Y6I10Ru). Inorganic Chemistry 28, 631--635. https://doi.org/10.1021/ic00303a004
- International Commission on Radiological Protection 1996. Conversion coefficients for use in radiological protection against external radiation (ICRP Publication 74), chapter 4: Conversion coefficients. Annals of the ICRP 26, 35--81. https://doi.org/10.1016/s0146-6453(96)90005-6
- Lund and Romano 2018. Implementation and Validation of Photon Transport in OpenMC. https://doi.org/10.2172/1490825
- McCaffrey et al. 2009. Optimizing non‐Pb radiation shielding materials using bilayers. Medical Physics 36, 5586-5594. https://doi.org/10.1118/1.3260839
- National Nuclear Data Center 2018. ENDF/B-VIII.0 Evaluated Nuclear Data Library (release page). https://www.nndc.bnl.gov/endf-b8.0/
- Park et al. 1997. Three Types of Condensed Cluster Phases of Rare-Earth Metal Iodides with Transition Metal Interstitials: Pr4I5Ni, Pr3I3Os, Pr2INi2, and La2IZ2. Journal of Solid State Chemistry 129, 277--286. https://doi.org/10.1006/jssc.1996.7246
- Payne et al. 1992. Chains of centered metal clusters with a novel range of distortions: praseodymium ruthenium iodide (Pr3I3Ru), ruthenium yttrium iodide (Y3I3Ru), and iridium yttrium iodide (Y3I3Ir). Inorganic Chemistry 31, 1389--1394. https://doi.org/10.1021/ic00034a018
- Romano et al. 2015. OpenMC: A state-of-the-art Monte Carlo code for research and development. Annals of Nuclear Energy 82, 90-97. https://doi.org/10.1016/j.anucene.2014.07.048
- Sakamoto et al. 1988. Interpolation of Gamma-Ray Buildup Factors for Point Isotropic Source with Respect to Atomic Number. Nuclear Science and Engineering 100, 33-42. https://doi.org/10.13182/nse88-a29012
- Schöpf and Pichler 2016. Radiation Protection Clothing in X-Ray Diagnostics – Influence of the Different Methods of Measurement on the Lead Equivalent and the Required Mass. RöFo - Fortschritte auf dem Gebiet der Röntgenstrahlen und der bildgebenden Verfahren 188, 768-775. https://doi.org/10.1055/s-0042-106651
- Trubey 1988. New gamma-ray buildup factor data for point kernel calculations: ANS-6.4.3 standard reference data. https://doi.org/10.2172/7032799
- Trubey 1991. New Gamma-Ray Buildup Factor Data for Point Kernel Calculations: ANS-6.4.3 Standard Reference Data. https://doi.org/10.2172/5441669
- Wang et al. 2020. Enhanced photon shielding efficiency of a flexible and lightweight rare earth/polymer composite: A Monte Carlo simulation study. Nuclear Engineering and Technology 52, 1565-1570. https://doi.org/10.1016/j.net.2019.12.028
