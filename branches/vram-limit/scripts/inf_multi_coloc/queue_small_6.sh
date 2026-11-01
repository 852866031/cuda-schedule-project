#!/bin/bash
# After the medium MPS sweep. Strictly serial. All co-located models are EQUAL priority
# and every arm runs under MPS with the default equal share (no gate, no no-MPS arm).
#  1. small offload beyond N=4 with the CPU-thread cap: where host RAM binds.
#  2. small N=8 fits WITH the thread cap: the stock N=8 rerun had a 20 s spin-storm (8
#     EngineCores pinned 32 cores, GPU1 idle) that blew up the TTFT tail -- is the cap
#     a general colocation rule, not just an offload fix?
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_small_[1-5].sh|queue_medium_(probe2?|[12]).sh" >/dev/null; do sleep 30; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
OMP_NUM_THREADS=4 $PY $D --cohort small --n 5 6 --arm mps --cells doff foff --name-suffix _omp4
OMP_NUM_THREADS=4 $PY $D --cohort small --n 8 --arm mps --cells dfits ffits --name-suffix _omp4
