# shieldmap

Software DOI: [10.5281/zenodo.23430637](https://doi.org/10.5281/zenodo.23430637) · Dataset DOI: [10.5281/zenodo.23457298](https://doi.org/10.5281/zenodo.23457298)

## Installation

Python 3.12 (the pinned versions need 3.12) and git. From the repository root:

```bash
python -m venv .venv                 # Windows: py -3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/prepare.py
```

`prepare.py` restores the per-compound thickness profiles and the crustal abundances, checked by SHA-256.

## One command per result (from the shipped tables, no credentials)

| Result | Command | Time |
|---|---|---|
| Headline table v3 | `python -m shieldmap.analyze4b --only table_v3` | 16 s |
| Headline table v4 | `python -m shieldmap.analysis4d.table_v4` | 1 s |
| Fig. 1 | `python -m shieldmap.figstyle.figures Fig1` | 2.4 s |
| Fig. 2 | `python -m shieldmap.figstyle.figures Fig2` | 2.7 s |
| Fig. 3 | `python -m shieldmap.figstyle.figures Fig3` | 1.9 s |
| Fig. 4 | `python -m shieldmap.figstyle.figures Fig4` | 2.1 s |
| Check against the published files | `python scripts/check_outputs.py` | < 1 s |

Times: AMD Ryzen 5 7500F (6 cores), 16 GB RAM, Windows 11, Python 3.12; installation ≈ 2 min. `REPRODUCE.md` has the exact sequence. In `pytest -q`, tests whose inputs are not shipped are skipped with the missing input named.

## Pipeline stages

Module and folder names carry the stage (`analysis3` and `data/phase3/` are stage 3); the names are kept so that every path in the code and the documentation stays valid.

| Stage | Entry point | Output | Needs | Time |
|---|---|---|---|---|
| 1 | `python -m shieldmap.build` | `data/interim/`, `data/processed/` | `MP_API_KEY`, network | ≈ 64 min |
| 2 | `python -m shieldmap.analyze` | `data/phase2/` | stage 1 | ≥ 6 min |
| 3 | `python -m shieldmap.analyze3` | `data/phase3/` | network (COD archive, 18.5 GB) | ≈ 3 h |
| 4a | `python -m shieldmap.analyze4a` | `data/phase4a/` | stage 3 | ≥ 5 min |
| 4a MC | `src/shieldmap/analysis4/mc/` | `data/phase4a/mc/` | Linux/WSL2, OpenMC 0.16.0, ENDF/B-VIII.0 | ≈ 3.3 h |
| 4b | `python -m shieldmap.analyze4b` | `data/phase4b/` | stage 4a | ≈ 3 min |
| 4b MC | `python -m shieldmap.analysis4b.mc_rev` | `data/phase4b/mc_rev/` | as 4a MC | 13 min |
| 4d | `python -m shieldmap.analysis4d.anchors`, `…mc_c8`, `…table_v4` | `data/phase4d/` | stage 4b (c8: as 4a MC) | minutes |

The Monte Carlo helpers ran under WSL2 and default to the author's paths (`--repo`).

## What is in `data/`

Only the derived tables that the commands above read (134 MB; listed in `data/README.md`). Not redistributed, with fetch or restore scripts verified by SHA-256: raw upstream files (`scripts/fetch_raw.py`), the NIST SRD 126 air table and test fixtures (`scripts/restore_nist_srd126.py`) and per-element crustal abundances (`scripts/restore_crustal_abundance.py`). ICSD collection codes are not distributed. The full map and the Monte Carlo records are in the dataset (10.5281/zenodo.23457298).

## Data sources

- Materials Project, database version 2026.04.13: Horton et al., *Nat. Mater.* (2025), doi:10.1038/s41563-025-02272-0; Jain et al., *APL Mater.* 1, 011002 (2013), doi:10.1063/1.4812323.
- NIST XCOM (Standard Reference Database 8): Berger et al. (2010), doi:10.18434/T48G6X.
- NIST Standard Reference Database 126: Hubbell and Seltzer (2004), doi:10.18434/T4D01F.
- Crystallography Open Database, revision 297631: Gražulis et al., *J. Appl. Cryst.* 42, 726 (2009), doi:10.1107/S0021889809016690.
- ENSDF (NNDC, Brookhaven National Laboratory): Browne and Tuli, *Nucl. Data Sheets* 114, 1849 (2013) (A = 60) and 108, 2173 (2007) (A = 137).
- IEC 61267 qualities as tabulated in IAEA Technical Reports Series No. 457 (2007), Table 6.2.
- SpekPy 2.5.4: Poludniowski et al., *Med. Phys.* 48, 3630 (2021), doi:10.1002/mp.14945; Bujila et al., *Phys. Med.* 75, 44 (2020).
- xraylib 4.3.0: Schoonjans et al., *Spectrochim. Acta B* 66, 776 (2011).
- OpenMC 0.16.0: Romano et al., *Ann. Nucl. Energy* 82, 90 (2015); ENDF/B-VIII.0: Brown et al., *Nucl. Data Sheets* 148, 1 (2018).
- pymatgen: Ong et al., *Comput. Mater. Sci.* 68, 314 (2013); mendeleev 1.3.0 (doi:10.5281/zenodo.5233823): crustal abundances from the CRC Handbook of Chemistry and Physics, 95th ed. (2014), and isotopic abundances, as distributed with mendeleev 1.3.0.

This work uses data from the Materials Project (database version 2026.04.13), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); entries were filtered and transformed as described in the documentation, and ICSD collection codes were removed. Photon cross sections are derived from NIST Standard Reference Database 8 (XCOM) and air energy-absorption coefficients from NIST Standard Reference Database 126, National Institute of Standards and Technology; NIST is the source of these data, which are not redistributed here. Measured structures come from the Crystallography Open Database; we acknowledge the original authors of the structural data. Decay data come from the Evaluated Nuclear Structure Data File (National Nuclear Data Center, Brookhaven National Laboratory). Diagnostic radiation qualities follow IEC 61267, with half-value layers from IAEA Technical Reports Series No. 457; the IAEA is acknowledged as the source.

## Licences

Code: MIT (`LICENSE`), except `src/shieldmap/ingest/xsec.py`, a port of NIST's XCOM.f with its non-commercial condition (`NOTICE`). Data, documentation and figures: CC BY 4.0 (`LICENSE-data`).

## How to cite

Cite the software (doi:10.5281/zenodo.23430637) and the dataset (doi:10.5281/zenodo.23457298); `CITATION.cff` has the details.
