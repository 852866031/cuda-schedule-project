#!/bin/bash
# E1b (added 02:25 after E1): thread-cap dose-response for medium N=4 decode-only offload.
# Capped at 4 the KV load is 401 ms (85 ms at N=1); uncapped it collapses (5.6 s). Does
# the load time depend on the cap size (2 / 8 threads per engine)? After queue_night.
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_night.sh" >/dev/null; do sleep 30; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
echo "### $(date '+%H:%M') E1b"
OMP_NUM_THREADS=2 $PY $D --cohort medium --n 4 --arm mps --cells doff --name-suffix _omp2
OMP_NUM_THREADS=8 $PY $D --cohort medium --n 4 --arm mps --cells doff --name-suffix _omp8
echo "### $(date '+%H:%M') done"
