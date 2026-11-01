#!/bin/bash
# E4b (added after E4 small): the small-cohort capacity sweep never saturated (N=8 still
# 96.5% delivered at 8 QPS/model). Extend upward until each N saturates. After queue 2.
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_night2?.sh" >/dev/null; do sleep 30; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
export OMP_NUM_THREADS=4
echo "### $(date '+%H:%M') E4b"
$PY $D --cohort small --n 8 --arm mps --cells dfits --qps 12 16 --name-suffix _omp4
$PY $D --cohort small --n 4 --arm mps --cells dfits --qps 12 16 24 --name-suffix _omp4
$PY $D --cohort small --n 1 --arm mps --cells dfits --qps 16 32 --name-suffix _omp4
echo "### $(date '+%H:%M') done"
