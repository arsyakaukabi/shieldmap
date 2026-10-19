# First-order fluorescence model and its Monte Carlo comparison

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

The map is narrow-beam: every interacting photon is removed.
For diagnostic spectra this overstates materials whose K and L fluorescence escapes toward the protected point, a penalty reported for lead-free garments (Eder et al. 2005; Schlattl et al. 2007; Schmid et al. 2012).
A first-order model adds that escape to the narrow-beam primary.

For constituent i with mass fraction w_i, the K-shell photoabsorption is τ_K,i(E) = w_i τ_i(E)(1 − 1/J_i) above the K edge, with τ_i the XCOM photoelectric cross section and J_i the K jump ratio read from the XCOM table.
A K vacancy emits line l with probability ω_K p_l; L-line production uses the xraylib cascade cross sections, and every line of at least 3.2 keV is kept.
A line photon born at mass depth s in a slab of areal mass t is emitted isotropically and escapes through the remaining t − s with the compound's own μ/ρ at the line energy, b.
The fluence just behind a laterally infinite slab is half the exponential integral E₁(b(t−s)), which is the quantity air kerma is computed from.
The effective transmitted kerma adds the escaping line kerma to the narrow-beam primary, and the required areal mass is solved for the same target transmission as the narrow-beam map.
The model omits secondary fluorescence, fluorescence from scattered photons and lines below 3.2 keV, and it counts Rayleigh-scattered lines as removed.

The Monte Carlo spot check ([Monte Carlo spot check](monte_carlo.md)) finds that this first-order model under-counts escaping fluorescence, by an amount that depends on the case family: least for single emitters below their K edge, more in deep slabs just above the Pb K edge, and most for partner elements excited by lanthanide K lines.
Per family, the escaping line kerma is under-counted by 19–27 % for single emitters below their K edge (Bi, Pb at RQR5 and RQR8), 25–43 % for Pb and Tl above the Pb K edge, 28–61 % for partner compounds and 8.4–31 % in the 50 keV checks. [H242, H242:hi, H243, H243:hi, H244, H244:hi, H245, H245:hi, stmt:H245]
The partner range starts at 28 % because it includes LaBi; without LaBi (Pr₃OsI₃, PrSbPd, HfAu₂) the low end is higher. [H244]
Part of the gap is not model structure but atomic data: the Monte Carlo samples the EADL radiative yields and the ENDF subshell cross sections, the model uses xraylib yields and the XCOM K-shell share, and the two differ most for the L1 subshell.
Every first-order value below therefore carries that caveat; the manuscript quotes the diagnostic advantage from the Monte Carlo.
Under the first-order model the median advantage of the best B4 lanthanide compound over the best textbook material across the 36 diagnostic variants is 22 % (range 10–38 %; first-order model, caveat above). [H83, H83:lo, H83:hi]
The Monte Carlo lowers this advantage, like-for-like to 6.8 % at RQR5 behind 0.25 mm Pb and to 17 % at RQR8 behind 0.35 mm Pb (8.5 % and 20 % with the fixed target), so the median is biased high. [H189, H191, H112, H118]
Under the first-order model Bi is the closest textbook material in 33 of 36 variants, needing 14–38 % more areal mass than the corrected B4 frontier (first-order model, caveat above). [stmt:H85, H85:pct_over_1, H85:hi_over_1]
At the headline variant (RQR6, 12°, 0.35 mm Pb) the first-order model puts Bi 22 % and β-Sn 33 % above the corrected frontier, against 35 % and 23 % under narrow beam (first-order model, caveat above). [H170, H171, H169, H168]
This reversal is the reason the manuscript names Bi the closest textbook material once fluorescence escape is included.
The lanthanide-level ranking of the families is stable across the 36 variants after correction (median pairwise Spearman correlation 0.91, range 0.70–1; first-order model, caveat above). [H86, H86:lo, H86:hi]
Energy-dependent rankings of lanthanides against Bi have been reported at single energies (Şişman et al. 2025), and the lanthanide elements have been ranked against each other (Yavuzkanat 2024); what the map adds is that the lanthanide-level ranking holds across tube spectra while the partner-level ranking does not.
In the Pb K-edge window the lead-free price on the areal-mass basis falls from 2.8 % in narrow beam to 0.41 % against Pb-containing compounds under the first-order model (band 0–0.52 %; caveat above). [H63, H64, H64:lo, H64:hi]
Among non-radioactive multi-element compounds, 52 of the 203 that exceed the non-radioactive elemental envelope per unit length in the 47–67 keV window survive the first-order correction (32–57 over the sensitivity settings; caveat above), all lanthanide compounds with a dense 5d partner. [H82, H82:delta, stmt:H82, H82:lo, H82:hi]
Combining complementary K edges is an established design principle for X-ray protection (Yaffe et al. 1991; McCaffrey et al. 2009), and lanthanide oxides fill the weak window of Pb (McCaffrey et al. 2012; Wang et al. 2020); the envelope count quantifies how far catalogued crystals go in this direction.

The first-order model markers (open grey markers, model only) are in [Diagnostic plateau and first-order markers](diagnostic_plateau.md).

## References

- Eder et al. 2005. Ist der Bleigleichwert zur Beurteilung der Schutzwirkung bleifreier Röntgenschutzkleidung geeignet?. RöFo - Fortschritte auf dem Gebiet der Röntgenstrahlen und der bildgebenden Verfahren 177, 399-404. https://doi.org/10.1055/s-2005-857885
- McCaffrey et al. 2009. Optimizing non‐Pb radiation shielding materials using bilayers. Medical Physics 36, 5586-5594. https://doi.org/10.1118/1.3260839
- McCaffrey et al. 2012. Radiation shielding materials and radiation scatter effects for interventional radiology (IR) physicians. Medical Physics 39, 4537-4546. https://doi.org/10.1118/1.4730504
- Schlattl et al. 2007. Shielding properties of lead‐free protective clothing and their impact on radiation doses. Medical Physics 34, 4270-4280. https://doi.org/10.1118/1.2786861
- Schmid et al. 2012. Emission of fluorescent x-radiation from non-lead based shielding materials of protective clothing: a radiobiological problem?. Journal of Radiological Protection 32, N129-N139. https://doi.org/10.1088/0952-4746/32/3/n129
- Wang et al. 2020. Enhanced photon shielding efficiency of a flexible and lightweight rare earth/polymer composite: A Monte Carlo simulation study. Nuclear Engineering and Technology 52, 1565-1570. https://doi.org/10.1016/j.net.2019.12.028
- Yaffe et al. 1991. Composite Materials for X-ray Protection. Health Physics 60, 661-664. https://doi.org/10.1097/00004032-199105000-00004
- Yavuzkanat 2024. Exploring Radiation Shielding Properties of Lanthanide Elements. Bitlis Eren Üniversitesi Fen Bilimleri Dergisi 13, 1314-1324. https://doi.org/10.17798/bitlisfen.1558208
- Şişman et al. 2025. Development of New Radiation Shielding Materials: The Role of Rare Earth Elements. Dokuz Eylül Üniversitesi Mühendislik Fakültesi Fen ve Mühendislik Dergisi 27, 485-490. https://doi.org/10.21205/deufmd.2025278116
