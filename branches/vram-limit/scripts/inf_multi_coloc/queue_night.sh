#!/bin/bash
# Overnight queue (2026-10-02), plan + predictions in DECISIONS_MULTI_COLOC.md §B.
# Strictly serial; every launch through multi_launch.sh (mem_guard, heartbeat, RAM gate).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
step() { echo "### $(date '+%H:%M') $*"; }
# E1: medium N=4 decode-only offload, STOCK threads -- is the 5x KV-load slowdown the cap?
step E1; $PY $D --cohort medium --n 4 --arm mps --cells doff --name-suffix _nocap
# E2: stock small N=4 doff again, now with the per-thread sampler running (spin-wait test)
step E2; $PY $D --cohort small --n 4 --arm mps --cells doff --name-suffix _thr
# E3: N=8 resident spin-storm reproducibility: stock x2, capped x2
step E3; $PY $D --cohort small --n 8 --arm mps --cells dfits --name-suffix _r3
$PY $D --cohort small --n 8 --arm mps --cells dfits --name-suffix _r4
OMP_NUM_THREADS=4 $PY $D --cohort small --n 8 --arm mps --cells dfits --name-suffix _omp4_r2
OMP_NUM_THREADS=4 $PY $D --cohort small --n 8 --arm mps --cells dfits --name-suffix _omp4_r3
# E4: capacity -- per-model QPS sweep at fixed N, thread cap on, decode-only fits
step E4; export OMP_NUM_THREADS=4
$PY $D --cohort small --n 1 4 8 --arm mps --cells dfits --qps 3 4 6 8 --name-suffix _omp4
$PY $D --cohort medium --n 1 2 4 --arm mps --cells dfits --qps 2.5 3 4
step done
