# Data and provenance

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

Every input belongs to one of three data classes.
P1 are physical tables maintained by a standards body: the NIST XCOM 3.1 photon cross sections (Seltzer 1987; Berger and Hubbell 1987; Scofield 1973; Hubbell et al. 1975; Hubbell et al. 1980), the NIST SRD 126 mass energy-absorption coefficients of air (Seltzer 1995; Hubbell and Seltzer 1995), and ENSDF gamma lines and half-lives (National Nuclear Data Center 2022; Browne and Tuli 2007; Browne and Tuli 2013); atomic masses and experimental element densities come from the element table distributed with pymatgen (Ong et al. 2013), which compiles reference values.
P2 is the database of record, Materials Project release 2026.04.13 (Jain et al. 2013), from which 154 377 non-deprecated entries were retrieved in one query.
P3 are definitional or geochemical constants: the list of elements without a stable isotope, the noble gases, and crustal abundances from the CRC Handbook, 95th edition (Haynes 2014), through mendeleev 1.3.0 (Mentel 2026).
Two models and one comparison database are used outside these classes.
Tube spectra are SpekPy 2.5.4 model spectra (Poludniowski et al. 2021; Bujila et al. 2020) matched to the IEC 61267 RQR qualities (International Electrotechnical Commission 2006), whose half-value layers are taken from IAEA TRS-457 (International Atomic Energy Agency 2007). The total Al filtration of each spectrum is solved so that SpekPy's own first half-value layer (air kerma, in Al) equals the TRS-457 value; a match outside ±2 % is rejected (`config/default.yaml`, `xray_tubes.hvl_match_tolerance_rel`; `src/shieldmap/physics/spectra.py`, `match_filtration`).
Measured compound densities come from the Crystallography Open Database, revision 297631 (Gražulis et al. 2009); they calibrate the density-error model ([Density-error model](density_error_model.md)) and never replace a Materials Project density in the map.
The Monte Carlo spot check uses OpenMC 0.16.0 (Romano et al. 2015; Lund and Romano 2018) with ENDF/B-VIII.0 data (Brown et al. 2018); these data never enter the map.
Fluorescence yields, radiative rates and line energies, and the cross-section library sensitivity, use xraylib 4.3.0 (Schoonjans et al. 2011).
Natural isotopic abundances follow IUPAC (Meija et al. 2016), as distributed with mendeleev (Mentel 2026).

The energy grid has 545 nodes from 10 keV to 15 MeV: 0.5 keV steps up to 200 keV, logarithmic spacing above, the tabulated standard energies, and the exact ENSDF lines.
Absorption edges are stored separately and no node lies on an edge, so no value is interpolated across an edge.
The cross sections are a line-by-line port of the XCOM program; compound values follow the mixture rule with the same atomic masses for the cross-section conversion and the mass fractions.
The port reproduces the NIST SRD 126 values of Pb, W, Fe and Al to within 0.04 % at the tabulated energies, and the same mixture rule underlies programs such as WinXCom (XCOM data), Phy-X/PSD and EpiXS (EPICS2017 and EPDL97 data) (Gerward et al. 2004; Şakar et al. 2020; Hila et al. 2021).
Materials Project structures, and hence densities, come from GGA, GGA+U or r²SCAN relaxations depending on the entry, and the energy above the hull refers to the mixed GGA/GGA+U/r²SCAN hull with the MP2020 corrections (Kingsbury et al. 2022; Wang et al. 2021).
The statement that most named leaders carry r²SCAN volumes rests on the Materials Project calculation records (origins) of these entries queried at the time of review, after the archived snapshot, not on the snapshot itself.
Band gaps are GGA(+U) values. An entry is experimentally observed when its Materials Project flag theoretical is False, i.e. when it has at least one ICSD id; the flag can change between releases.
Cs-137 is modelled as its 661.657 keV line alone; the Ba K X-rays are omitted, which is immaterial at a transmission of 0.01.
Between the SRD 126 nodes the air mass energy-absorption coefficient is interpolated with a log-log spline, a 0.25 % effect on the half-value layer HVL₁; for single lines the air-kerma weighting is immaterial.
The SpekPy spectra have no air path; recomputed with the XCOM data of the map (Al at 2.70 g/cm³), their HVL₁ values are 0.83–1.11 % below SpekPy's own, and their homogeneity coefficients are within 0.012 of IEC 61267.
The X-ray spectra are sampled at discrete quadrature energies (376 pieces for RQR5 and 508 for RQR8 at 12°), which the Monte Carlo also uses.
Polyenergetic transmission is the exact sum over spectral bins, split at every element edge, and is inverted for the required thickness with a vectorised Brent solver.
Materials Project data are a single paginated query, archived with checksums; every raw input of the map is archived with its checksum and the whole map rebuilds offline from one command.
The baseline-eligible map has 151 655 entries and 102 806 formulas (Supplementary Table S1).

## References

- Berger and Hubbell 1987. XCOM : photon cross sections on a personal computer. https://doi.org/10.6028/nbs.ir.87-3597
- Brown et al. 2018. ENDF/B-VIII.0: The 8th Major Release of the Nuclear Reaction Data Library with CIELO-project Cross Sections, New Standards and Thermal Scattering Data. Nuclear Data Sheets 148, 1-142. https://doi.org/10.1016/j.nds.2018.02.001
- Browne and Tuli 2007. Nuclear Data Sheets for A = 137. Nuclear Data Sheets 108, 2173-2318. https://doi.org/10.1016/j.nds.2007.09.002
- Browne and Tuli 2013. Nuclear Data Sheets for A = 60. Nuclear Data Sheets 114, 1849-2022. https://doi.org/10.1016/j.nds.2013.11.002
- Bujila et al. 2020. A validation of SpekPy: A software toolkit for modelling X-ray tube spectra. Physica Medica 75, 44-54. https://doi.org/10.1016/j.ejmp.2020.04.026
- Gerward et al. 2004. WinXCom—a program for calculating X-ray attenuation coefficients. Radiation Physics and Chemistry 71, 653--654. https://doi.org/10.1016/j.radphyschem.2004.04.040
- Gražulis et al. 2009. Crystallography Open Database – an open-access collection of crystal structures. Journal of Applied Crystallography 42, 726-729. https://doi.org/10.1107/s0021889809016690
- Haynes 2014. CRC Handbook of Chemistry and Physics. https://doi.org/10.1201/b17118
- Hila et al. 2021. EpiXS: A Windows-based program for photon attenuation, dosimetry and shielding based on EPICS2017 (ENDF/B-VIII) and EPDL97 (ENDF/B-VI.8). Radiation Physics and Chemistry 182, 109331. https://doi.org/10.1016/j.radphyschem.2020.109331
- Hubbell and Seltzer 1995. Tables of x-ray mass attenuation coefficients and mass energy-absorption coefficients 1 keV to 20 MeV for elements Z = 1 to 92 and 48 additional substances of dosimetry interest. https://doi.org/10.6028/nist.ir.5632
- Hubbell et al. 1975. Atomic form factors, incoherent scattering functions, and photon scattering cross sections. Journal of Physical and Chemical Reference Data 4, 471-538. https://doi.org/10.1063/1.555523
- Hubbell et al. 1980. Pair, Triplet, and Total Atomic Cross Sections (and Mass Attenuation Coefficients) for 1 MeV-100 GeV Photons in Elements Z =1 to 100. Journal of Physical and Chemical Reference Data 9, 1023-1148. https://doi.org/10.1063/1.555629
- International Atomic Energy Agency 2007. Dosimetry in Diagnostic Radiology: An International Code of Practice. https://www-pub.iaea.org/MTCD/Publications/PDF/TRS457_web.pdf
- International Electrotechnical Commission 2006. Medical diagnostic X-ray equipment -- Radiation conditions for use in the determination of characteristics. https://doi.org/10.3403/30099649
- Jain et al. 2013. Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL Materials 1. https://doi.org/10.1063/1.4812323
- Kingsbury et al. 2022. A flexible and scalable scheme for mixing computed formation energies from different levels of theory. npj Computational Materials 8. https://doi.org/10.1038/s41524-022-00881-w
- Lund and Romano 2018. Implementation and Validation of Photon Transport in OpenMC. https://doi.org/10.2172/1490825
- Meija et al. 2016. Isotopic compositions of the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry 88, 293-306. https://doi.org/10.1515/pac-2015-0503
- Mentel 2026. mendeleev -- A Python package with properties of chemical elements, ions, isotopes and methods to manipulate and visualize periodic table. https://doi.org/10.5281/zenodo.5233823
- National Nuclear Data Center 2022. Evaluated Nuclear Structure Data File (ENSDF). https://doi.org/10.18139/nndc.ensdf/1845010
- Ong et al. 2013. Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. Computational Materials Science 68, 314-319. https://doi.org/10.1016/j.commatsci.2012.10.028
- Poludniowski et al. 2021. Technical Note: SpekPy v2.0—a software toolkit for modeling x‐ray tube spectra. Medical Physics 48, 3630-3637. https://doi.org/10.1002/mp.14945
- Romano et al. 2015. OpenMC: A state-of-the-art Monte Carlo code for research and development. Annals of Nuclear Energy 82, 90-97. https://doi.org/10.1016/j.anucene.2014.07.048
- Schoonjans et al. 2011. The xraylib library for X-ray–matter interactions. Recent developments. Spectrochimica Acta Part B: Atomic Spectroscopy 66, 776-784. https://doi.org/10.1016/j.sab.2011.09.011
- Scofield 1973. Theoretical photoionization cross sections from 1 to 1500 keV. https://doi.org/10.2172/4545040
- Seltzer 1987. XCOM-Photon Cross Sections Database, NIST Standard Reference Database 8. https://doi.org/10.18434/t48g6x
- Seltzer 1995. Tables of X-Ray Mass Attenuation Coefficients and Mass Energy-Absorption Coefficients, NIST Standard Reference Database 126. https://doi.org/10.18434/t4d01f
- Wang et al. 2021. A framework for quantifying uncertainty in DFT energy corrections. Scientific Reports 11. https://doi.org/10.1038/s41598-021-94550-5
- Şakar et al. 2020. Phy-X / PSD: Development of a user friendly online software for calculation of parameters relevant to radiation shielding and dosimetry. Radiation Physics and Chemistry 166, 108496. https://doi.org/10.1016/j.radphyschem.2019.108496
