"""Monte Carlo spot check (OpenMC, ENDF/B-VIII.0 photon data).

Windows side (project venv):
  export.py    case definitions, compositions, densities, sources, Table C / first-order values -> JSON
  compare.py   MC records + statepoints -> summary tables in data/phase4a/mc/ (one command, see compare.py)
WSL side (OpenMC 0.16.0 in a micromamba env, see wsl_setup.sh):
  openmc_cases.py  builds and runs every OpenMC model, reduces the surface-crossing records

ENDF/B-VIII.0 data are used only here and never enter the map.
"""
