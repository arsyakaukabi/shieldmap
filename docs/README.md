# Documentation

This folder holds the methods, sensitivities and results that the manuscript cites but does not print. The main text cites a document by its name ("repository documentation, *Density-error model*"), the Supplementary Material as "repository document *Density-error model*". Every number keeps its source row of the headline table in square brackets ([Full headline table](headline_table.md)).

## Documents

| Document | File | Contents |
|---|---|---|
| Data and provenance | [data_and_provenance.md](data_and_provenance.md) | data classes P1–P3, energy grid, cross-section port and its validation, air-kerma interpolation, tube spectra and quadrature |
| Practical set and flags | [practical_set_and_flags.md](practical_set_and_flags.md) | price definition, artefact flag and its effect, hydrogen-containing compounds and the H-excluded variant, ICSD-tag bookkeeping |
| Density-error model | [density_error_model.md](density_error_model.md) | DFT density bias, calibration cells, copula, measured-density point, P(best) |
| Fluorescence model | [fluorescence_model.md](fluorescence_model.md) | derivation of the first-order model, its per-family under-count against the Monte Carlo, first-order results |
| Monte Carlo spot check | [monte_carlo.md](monte_carlo.md) | physics, estimators, bootstrap, validation, build-up, transparency price, diagnostic advantage, reference beam, Pb K-edge window; pointers to sub-run sums and inputs |
| Headroom | [headroom.md](headroom.md) | headroom prices against F0, actinide headroom |
| Element-exclusion filters | [element_exclusion.md](element_exclusion.md) | single-element prices inside B4 (figure: Supplementary Fig. S2) |
| Z-exponents | [z_exponents.md](z_exponents.md) | physics of the composition–density crossover |
| Polymorphs | [polymorphs.md](polymorphs.md) | polymorph density spread, DFT against measured |
| Cross-section library sensitivity | [cross_section_sensitivity.md](cross_section_sensitivity.md) | xraylib and ENDF/B-VIII.0 against XCOM near edges |
| Blends, fully oxidised filter, transparency threshold | [blends_oxidised_threshold.md](blends_oxidised_threshold.md) | elemental blends, AMB-ANION/AMB-ICSD3 prices and leaders, band-gap threshold sensitivity |
| Natural radioactivity | [natural_radioactivity.md](natural_radioactivity.md) | natural-radioactivity flag and its table |
| Scarcity-proxy threshold | [scarcity_proxy_threshold.md](scarcity_proxy_threshold.md) | threshold curve and staircase; abundant-set leader at 0.5, 1 and 2 ppm |
| Other single-line results | [single_line_results.md](single_line_results.md) | further single-line prices |
| Diagnostic plateau | [diagnostic_plateau.md](diagnostic_plateau.md) | advantage over all 36 variants, first-order model markers |
| Full headline table | [headline_table.md](headline_table.md) | pointer to `data/phase4b/headline_table_v3.csv`, `data/phase4d/headline_table_v4.csv` and the dataset |

## Figures

Figure files are in [`figures/`](figures/). They were drawn for a longer version of the manuscript and are kept as files,
produced by `figures/render_docs_figures.py` (style system `src/shieldmap/figstyle/`):

| File | Shows | Used in |
|---|---|---|
| `phase4b_FigS3.png` | headroom prices | [headroom.md](headroom.md) |
| `phase4b_FigS4.png` | textbook Z-exponents | [z_exponents.md](z_exponents.md) |
| `phase4b_FigS5.png` | polymorph density spread | [polymorphs.md](polymorphs.md) |
| `phase4b_FigS6.png` | cross-section library sensitivity | [cross_section_sensitivity.md](cross_section_sensitivity.md) |
| `phase4b_Fig4.png` | diagnostic plateau and first-order markers | [diagnostic_plateau.md](diagnostic_plateau.md) |
| `phase4b_FigS7.png` | full Monte Carlo figure | [monte_carlo.md](monte_carlo.md) |

The other `phase4b_*` files complete that figure set and are kept for reference. The prefix names the pipeline
stage that drew them (see the stage table in the top-level `README.md`).

## Conventions

Unless a document says otherwise, values are narrow-beam (every interacting photon removed), polymorph rule R2, coherent
scattering included, best non-artefact compound (best_clean), and prices are B4-internal (B4 = the practical set) with
thickness at the measured-density point.
