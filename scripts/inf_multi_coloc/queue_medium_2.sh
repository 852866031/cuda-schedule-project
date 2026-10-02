#!/bin/bash
# Medium cohort, enlarged KV grant (util 0.22, max_num_seqs 16, batched 1024 -> 0.94 GiB;
# fits=3 / offload=14 sessions): MPS sweep N=1-4 WITH the per-engine CPU thread cap
# (OMP_NUM_THREADS=4; the stock default oversubscribes the host at N=4 under offload).
# The uncapped N=1 + N=2-dfits cells are kept as *_nocap.
cd "$(dirname "$0")/../.."
OMP_NUM_THREADS=4 .venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort medium \
    --n 1 2 3 4 --arm mps
