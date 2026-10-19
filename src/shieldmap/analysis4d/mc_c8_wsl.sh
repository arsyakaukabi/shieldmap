#!/bin/bash
# Stage 4d, MC case c8_cs137_irw (WSL side). From Windows:
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/USER/Desktop/shield/shieldmap/src/shieldmap/analysis4d/mc_c8_wsl.sh [endf|stubs|run] [deadline ISO]
# run is resumable (run_point skips finished sub-runs); the log ends with ALL_DONE or DEADLINE.
REPO=/mnt/c/Users/USER/Desktop/shield/shieldmap
CMD=${1:-run}
LOG=$REPO/data/raw/mc4d/run.log
mkdir -p "$REPO/data/raw/mc4d"
cd /tmp || exit 1
PY="$REPO/src/shieldmap/analysis4d/mc_c8.py"
if [ "$CMD" = "endf" ] || [ "$CMD" = "stubs" ]; then
  MAMBA_ROOT_PREFIX=~/mc4a/mamba ~/mc4a/bin/micromamba run -n openmc python "$PY" "wsl-$CMD"
else
  MAMBA_ROOT_PREFIX=~/mc4a/mamba ~/mc4a/bin/micromamba run -n openmc python "$PY" \
    wsl-run --deadline "$2" --threads 8 >> "$LOG" 2>&1
fi
