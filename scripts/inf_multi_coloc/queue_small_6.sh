#!/bin/bash
# After the medium sweep. Strictly serial.
#  1. offload beyond N=4 with the CPU-thread cap (stock collapses at N=4): RAM ceiling.
#  2. no-MPS reference (time-slicing) at N=2,4 -- how much worse it gets with N.
#  3. MPS + idle-window gate (model 0 priority) at N=4,8, eager-no-gate control at N=8.
cd "$(dirname "$0")/../.."
sleep 120
while pgrep -f "queue_small_[1-5].sh|queue_medium_(probe2?|[12]).sh" >/dev/null; do sleep 20; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
OMP_NUM_THREADS=4 $PY $D --cohort small --n 5 6 --arm mps --cells doff foff --name-suffix _omp4
$PY $D --cohort small --n 2 4 --arm nomps --cells dfits
$PY $D --cohort small --n 4 8 --arm gate --cells dfits
$PY $D --cohort small --n 8 --arm eager --cells dfits
