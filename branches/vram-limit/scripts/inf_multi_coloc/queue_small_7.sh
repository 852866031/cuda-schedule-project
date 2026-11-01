#!/bin/bash
# Complete the thread-capped offload curves N=1..6 (small cohort) so the offload figure can
# show capped vs stock over the whole range. doff N=4 capped already exists (_omp4) -- the
# N=4 launch runs foff only, so that file is never overwritten. After queue 6.
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_small_[1-6].sh|queue_medium_(probe2?|[12]).sh" >/dev/null; do sleep 30; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
export OMP_NUM_THREADS=4
$PY $D --cohort small --n 1 2 3 --arm mps --cells doff foff --name-suffix _omp4
$PY $D --cohort small --n 4 --arm mps --cells foff --name-suffix _omp4
