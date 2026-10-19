#!/usr/bin/env bash
# User-space OpenMC install inside WSL (Ubuntu 24.04), no sudo.
# micromamba (static binary) -> env "openmc" from conda-forge.
# Run detached:  setsid nohup bash wsl_setup.sh > ~/mc4a/setup.log 2>&1 &
set -euo pipefail
ROOT="$HOME/mc4a"
mkdir -p "$ROOT/bin"
cd "$ROOT"
if [ ! -x "$ROOT/bin/micromamba" ]; then
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$ROOT/mamba"
"$ROOT/bin/micromamba" create -y -n openmc -c conda-forge python=3.12 openmc h5py numpy pandas scipy
"$ROOT/bin/micromamba" run -n openmc python -c "import openmc, sys; print('openmc', openmc.__version__, sys.version)"
"$ROOT/bin/micromamba" run -n openmc openmc --version
echo SETUP_DONE
