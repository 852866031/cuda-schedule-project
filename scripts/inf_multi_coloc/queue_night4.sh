#!/bin/bash
# E4c (added after E4 medium): medium capacity sweep never saturated by 4 QPS/model either.
# Extend upward. Medium engines run max_num_seqs=16, so the per-engine sequence cap
# (QPS x e2e > 16) is the predicted limit. After queue 3.
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_night[23]?.sh" >/dev/null; do sleep 30; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
export OMP_NUM_THREADS=4
echo "### $(date '+%H:%M') E4c"
$PY $D --cohort medium --n 4 --arm mps --cells dfits --qps 6 8
$PY $D --cohort medium --n 2 --arm mps --cells dfits --qps 6 8 12
$PY $D --cohort medium --n 1 --arm mps --cells dfits --qps 8 12 16
echo "### $(date '+%H:%M') done"
