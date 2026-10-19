#!/bin/bash
# Stage 4b revision, MC case c7_rqr6: one resumable transport chunk (WSL side). From Windows:
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/USER/Desktop/shield/shieldmap/src/shieldmap/analysis4b/mc_rev_wsl.sh [run|endf|stubs] [max_seconds]
# Re-run until the log ends with ALL_DONE (run_point skips finished sub-runs).
REPO=/mnt/c/Users/USER/Desktop/shield/shieldmap
CMD=${1:-run}
MAXS=${2:-450}
LOG=$REPO/data/raw/mc4b/run.log
mkdir -p "$REPO/data/raw/mc4b"
cd /tmp || exit 1
if [ "$CMD" = "endf" ] || [ "$CMD" = "stubs" ]; then
  MAMBA_ROOT_PREFIX=~/mc4a/mamba ~/mc4a/bin/micromamba run -n openmc python "$REPO/src/shieldmap/analysis4b/mc_rev.py" "wsl-$CMD"
else
  MAMBA_ROOT_PREFIX=~/mc4a/mamba ~/mc4a/bin/micromamba run -n openmc python "$REPO/src/shieldmap/analysis4b/mc_rev.py" \
    wsl-run --max-seconds "$MAXS" --threads 8 2>&1 | tee -a "$LOG" | grep -v -E ' b(0[1-9]|1[0-9]) '
fi
