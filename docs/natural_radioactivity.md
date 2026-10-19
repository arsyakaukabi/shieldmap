# Natural-radioactivity flag

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

The primary non-radioactive requirement excludes elements without a stable isotope (Tc, Pm and Z ≥ 84).
A natural-radioactivity flag extends it as a sensitivity (table below).
For every element with a naturally occurring radioactive isotope, the natural specific activity A is computed from IUPAC isotopic abundances (Meija et al. 2016) and ENSDF half-lives (National Nuclear Data Center 2022) and compared with that of natural potassium, 31.7 Bq g−¹. [H105:delta]
The moderate flag adds the elements whose activity exceeds that of potassium: Re (32.4 times), Rb (27.3), Sm (3.88) and Lu (1.65). [H102, H99, H100, H101]
The named flag adds La, Lu and Re; La reaches 0.026 of the potassium activity. [H103]
The natural radioactivity of elements such as lutetium is well established (Alva-Sánchez et al. 2018; Brinkman et al. 1965); we turn it into a reproducible screening flag by comparing each element's natural specific activity, from ENSDF half-lives and IUPAC abundances, with that of natural potassium (Meija et al. 2016; National Nuclear Data Center 2022).
Under the moderate flag the Lu oxides leave the Co-60 transparent top ten, which leaves HfO₂ and the Ta oxides.

## Natural-radioactivity flag

Natural specific activity *A* of each element with a naturally occurring radioactive isotope (IUPAC isotopic abundances,
ENSDF half-lives), relative to natural potassium (*A*_K = 31.7 Bq/g [H105:delta]). Source: `data/phase4a/S/natrad_elements.csv`.

| Element | Isotope(s) | *A* (Bq/g) | *A*/*A*_K | Flag moderate | Flag named | Row |
|---|---|---:|---:|:---:|:---:|---|
| Re | Re-187 | 1027 | 32.4 | yes | yes | H102 |
| Rb | Rb-87 | 867 | 27.3 | yes | – | H99 |
| Sm | Sm-147, Sm-148 | 123 | 3.88 | yes | – | H100 |
| Lu | Lu-176 | 52.3 | 1.65 | yes | yes | H101 |
| K | K-40 | 31.7 | 1.00 | – | – | H105 |
| La | La-138 | 0.821 | 0.0259 | – | yes | H103 |
| In | In-115 | 0.250 | 0.00788 | – | – | H104 |
| 22 further elements with *A*/*A*_K < 0.001: Ca, V, Cr, Zn, Ge, Se, Zr, Mo, Cd, Te, Xe, Ba, Ce, Nd, Eu, Gd, Hf, W, Os, Pt, Pb, Bi | | | ≤ 0.00052 | – | – | |

- Flags: strict (primary, used in B4) excludes elements without a stable isotope (Tc, Pm, Z ≥ 84); moderate adds every
  element with *A* > *A*_K; named adds Lu, Re and La.
- Th, Pa, U carry the strict flag; their *A*/*A*_K is at least 793 (U).
- ENSDF gives only a lower limit of the half-life for Cr, Zn, Te, Xe, Ba, Ce, Eu, W, Os, Pb; their *A* is an upper bound.
- All 84 elements of the isotopic-abundance table are in the data set (`natrad_elements.csv`).

## References

- Alva-Sánchez et al. 2018. Understanding the intrinsic radioactivity energy spectrum from 176Lu in LYSO/LSO scintillation crystals. Scientific Reports 8. https://doi.org/10.1038/s41598-018-35684-x
- Brinkman et al. 1965. Natural radioactivity of K-40, Rb-87 and Lu-176. Physica 31, 1305-1319. https://doi.org/10.1016/0031-8914(65)90056-x
- Meija et al. 2016. Isotopic compositions of the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry 88, 293-306. https://doi.org/10.1515/pac-2015-0503
- National Nuclear Data Center 2022. Evaluated Nuclear Structure Data File (ENSDF). https://doi.org/10.18139/nndc.ensdf/1845010
