#!/bin/bash
# Test: is the N=4 decode-only-offload collapse CPU thread oversubscription? The trace
# showed the 4 EngineCores (not the stores) saturating all 32 cores. Cap each engine's
# torch/OpenMP pool at 4 threads (4 x 4 = 16 << 32 cores). No collapse => mechanism.
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_small_[1234].sh|queue_medium_probe2?.sh" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=4 .venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort small \
    --n 4 --arm mps --cells doff --name-suffix _omp4
