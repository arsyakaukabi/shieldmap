# Data contract — shieldmap build (Tables A–C)

Every number in the map belongs to one of three classes. Nothing else may enter.

| Class | Meaning | Sources used |
|---|---|---|
| **P1** Physical table | Measured or evaluated reference table from a standards body | NIST XCOM 3.1 element files; NIST SRD 126 (air μ_en/ρ; validation fixtures); NNDC ENSDF decay data; pymatgen atomic masses (IUPAC) and experimental solid densities |
| **P2** Database of record | The database under study (DFT, accepted as substrate) | Materials Project, one versioned snapshot |
| **P3** Definitional / geochemical constant | Facts that do not change with markets or methods | No-stable-isotope list; noble gases; crustal abundance (CRC Handbook via mendeleev) |
| [STD] standard (config) | Cited normative values | IEC 61267 RQR qualities as tabulated in IAEA TRS-457 Table 6.2 |
| model | Physics model, not data | SpekPy 2.5.4 (W-anode spectra) |

**Forbidden:** market data (prices, HHI, production), compilations of compilations, per-paper values, web-form retrieval, and page scraping. Network access is limited to one retrieval of each reference set, as a bulk download, an official API, an official package, or an official document. Each retrieval is recorded in `config/sources/<id>.yaml` with its citation, URL, UTC date, and SHA-256. Rebuilds verify cached files against these checksums.

## Table A — elements (`data/interim/elements/`)

| Field | Class | Source / definition |
|---|---|---|
| `Z`, `symbol` | P1 | pymatgen `Element` |
| `atomic_mass` (u) | P1 | pymatgen `Element.atomic_mass` (IUPAC standard atomic weights) |
| `density_solid_exp` (g/cm³) | P1 | pymatgen `Element.density_of_solid` ÷ 1000 (experimental); null where pymatgen has no value |
| `crustal_abundance_ppm` (mg/kg) | P3 | CRC Handbook 95th ed. (2014) §14, frozen from `mendeleev` 1.3.0 → `sources/crustal_abundance.yaml`; null where not tabulated |
| `is_radioactive` | P3 | No stable isotope: Tc, Pm, Z ≥ 84; Bi stable |
| `is_noble_gas` | P3 | He, Ne, Ar, Kr, Xe, Rn |
| `xsec_grid.npz`: `coh`, `incoh`, `pe`, `pair_nuc`, `pair_el`, `total_wcoh`, `total_wocoh` (cm²/g), shape (100, n_E), + `energy_keV`, `Z` | P1 | XCOM 3.1 `MDATX3.nnn`, evaluated at each grid energy with XCOM's own interpolation (port of `XCOM.f`); b/atom → cm²/g with XCOM's `ATWTS.DAT` weights and N_A = 0.60221367 × 10²⁴ |
| `edges.parquet`: `Z`, `shell`, `energy_keV` | P1 | XCOM 3.1 edge records |
| `data/interim/references/air_muen.parquet` (`energy_keV`, `mu_rho`, `muen_rho`, `edge_label`, `edge_side`) | P1 | NIST SRD 126 Table 4, air (dry, near sea level); retrieved once and transcribed |

## Table B — compounds (`data/interim/mp/compounds.parquet`), one row per non-deprecated MP entry

| Field | Class | Source / definition |
|---|---|---|
| `material_id`, `formula_pretty`, `chemsys` | P2 | MP summary |
| `formula_reduced` | derived | `composition_reduced` from MP, rendered as a reduced formula |
| `mass_fractions` | derived | from MP `composition` using pymatgen atomic masses; asserted Σ = 1 (\|Σ−1\| < 1e-9). Stored as the long table `composition.parquet` (`material_id`, `Z`, `symbol`, `mass_fraction`, `atom_fraction`) and a sparse (N × 100) matrix |
| `density` (g/cm³), `volume` (Å³), `nsites` | P2 | MP summary (DFT-relaxed cell) |
| `energy_above_hull`, `formation_energy_per_atom` (eV/atom), `is_stable` | P2 | MP summary; the thermo mixing scheme is recorded in `sources/materials_project.yaml` |
| `theoretical`, `icsd_ids` (`database_IDs`) | P2 | `theoretical == False` ⇒ matched to an ICSD structure |
| `band_gap` (eV), `is_gap_direct`, `is_metal` | P2 | MP summary (PBE / GGA(+U)) |
| `spacegroup_number`, `crystal_system` | P2 | MP `symmetry` |
| `deprecated` | P2 | only `False` entries are kept (the query filters on it) |
| `nelements`, `elements`, `n_icsd_ids` | P2/derived | MP summary; count of ICSD ids |
| `material_id_api`, `id_prefix`, `id_number` | P2/derived | raw API id (new alphabetic form) and the parsed conventional id used for sorting |
| `is_representative` | derived | lowest `energy_above_hull` per `formula_reduced` (ties → lowest numeric material id); all entries kept |
| `contains_<Symbol>`, `contains_radioactive`, `contains_noble_gas` | derived | from composition × Table A flags |
| `min_crustal_abundance_ppm` | derived | min over constituents; null if any constituent lacks a value |
| `abundance_weighted_log_ppm` | derived | Σᵢ wᵢ log₁₀(abundanceᵢ) with mass fractions wᵢ; null if any constituent lacks a value |

## Table C — derived, computed and never fetched (`data/processed/`)

| Item | Definition |
|---|---|
| `mu_rho_wcoh.npy`, `mu_rho_wocoh.npy` (float32, N × n_E) + `index.parquet`, `energy_keV.npy` | μ/ρ = Σᵢ wᵢ (μ/ρ)ᵢ, one sparse × dense matrix product |
| μ, HVL, TVL, MFP | μ = ρ·μ/ρ; HVL = ln2/μ; TVL = ln10/μ; MFP = 1/μ. These are exact transforms; the accessor `shieldmap.physics.tablec` computes them. They are also tabulated at the key energies in `key_energies.parquet`, with partials. |
| `profiles.parquet` (long) + `profile_variants.parquet` | per compound × variant: `thickness_req_cm`, `areal_mass_req_g_cm2`, `pb_relative_thickness`, `pb_relative_areal_mass`. Tube spectra are integrated on the edge-split SpekPy quadrature; `spectra.parquet` / `spectrum_bins.parquet` hold the spectra |
| `envelope.parquet` | per energy: max μ/ρ over elements Z = 1–100, and max μ over elemental solids with `density_solid_exp`; argmax element; unfiltered |
| `MANIFEST.json` | versions, dates, config hash, row counts, SHA-256 of every output |

**Terminology:** "Pb-relative thickness (narrow-beam)", never "lead equivalence".
