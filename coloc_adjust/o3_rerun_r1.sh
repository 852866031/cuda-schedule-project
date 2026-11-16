#!/usr/bin/env bash
# One-off for the night of Nov 16: R1's tenant was stopped by the cleanup of a GPU 0 run. When R3 ends, stop the first
# GPU 1 chain (PID in $1) before it starts R6, set the partial R1 files aside, and run R1 and then R6.
cd "$(dirname "$0")"
until grep -q "end r3" logs_o3/night_gpu1.log; do sleep 2; done
kill "$1"; sleep 3
mkdir -p results_o3/r1_killed; mv results_o3/r1_* results_o3/r1_killed/ 2>/dev/null
echo "$(date +%H:%M:%S) rerun r1 (its tenant was stopped by the cleanup of a GPU 0 run), then r6" >> logs_o3/night_gpu1.log
STEPS="r1 r6" exec ./run_o3_gpu1.sh
