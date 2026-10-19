# Reproducing the headline tables and the main figures

These steps start from a clean machine and end with the two headline tables
(`data/phase4b/headline_table_v3.csv`, `data/phase4d/headline_table_v4.csv`) and Figs. 1–4
(`paper/figures/Fig1`–`Fig4`, PDF, PNG and TIFF) regenerated from the derived tables in this repository and
compared with the published files. They need no credentials and no network after the installation, no Linux and
no Monte Carlo. The test of these steps in a fresh environment is recorded with the release.

## 1. Clean machine

Python 3.12 and git. In an empty directory:

```bash
git clone https://github.com/arsyakaukabi/shieldmap.git shieldmap
cd shieldmap
python -m venv .venv                   # Windows: py -3.12 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/prepare.py
```

`prepare.py` must end with `ok: ready`. It restores `data/phase2/profiles_thickness.npy` from its compressed parts
and the crustal-abundance block and column (from mendeleev 1.3.0), checking each against the SHA-256 of the file
used for the paper. Installation took ≈ 2 min; `prepare.py` 7 s.

## 2. Headline tables and Figs. 1–4

```bash
python -m shieldmap.analyze4b --only table_v3
python -m shieldmap.analysis4d.table_v4
python -m shieldmap.figstyle.figures Fig1
python -m shieldmap.figstyle.figures Fig2
python -m shieldmap.figstyle.figures Fig3
python -m shieldmap.figstyle.figures Fig4
python scripts/check_outputs.py
```

`table_v3` rebuilds headline table v3 from the stage-4a table, the corrections and revision analyses in
`data/phase4b/` and the per-compound map (`data/phase2/`, `data/processed/`), and writes a Markdown view to
`reports/phase4b/` (not tracked). `table_v4` appends the rows H401+ of `data/phase4d/rows_*.csv`. The figure
commands rewrite `paper/figures/FigN.pdf|png|tif`. Recorded times: 16 s and 1 s for the tables,
2–3 s per figure.

`check_outputs.py` must end with `all outputs identical`. On Windows the two CSV files are reported as
`identical (line endings)`: pandas writes CRLF there, and git stores the files with LF; the content is the same
byte for byte otherwise. `git diff --stat` then lists only the three files restored in section 1
(`config/sources/crustal_abundance.yaml`, `data/interim/elements/elements.parquet`,
`data/processed/table_a_elements.parquet`); on Windows `git status` also names the two CSV files, for their line
endings only.

## 3. Tests

```bash
pytest -q
```

Tests whose inputs are not in the repository (raw inputs, intermediates of the full pipeline, the NIST SRD 126
fixtures) are skipped with the missing input named; none fails. `python scripts/restore_nist_srd126.py` (network)
restores the SRD 126 air table and fixtures, after which their tests run too.

## 4. What needs credentials, network or long compute

| Step | Needs | Recorded time |
|---|---|---|
| Sections 1–3 | nothing beyond `pip install` | ≈ 3 min |
| `python scripts/restore_nist_srd126.py` | network (18 NIST pages) | 25 s |
| `python scripts/fetch_raw.py` | network (`--large`: COD 18.5 GB, ENDF/B-VIII.0 3.4 GB) | minutes to hours |
| Stage 1, `python -m shieldmap.build` | `MP_API_KEY` in the environment, network | ≈ 64 min |
| Stages 2–4d (`analyze`, `analyze3`, `analyze4a`, `analyze4b`, `analysis4d.*`) | stage 1 outputs; stage 3 the COD archive | ≈ 4 h |
| Monte Carlo (stages 4a, 4b, 4d) | Linux or WSL2, OpenMC 0.16.0, ENDF/B-VIII.0 | ≈ 3.5 h on 8 threads |

The README lists the stages. Without the Monte Carlo runs the analyses read the recorded sub-run sums; the Monte
Carlo raw records, statepoints and model files are in the dataset (https://doi.org/10.5281/zenodo.23457298).

Hardware of the recorded times: AMD Ryzen 5 7500F (6 cores, 12 threads), 16 GB RAM, Windows 11 (WSL2 Ubuntu 24.04
for the Monte Carlo), Python 3.12.
