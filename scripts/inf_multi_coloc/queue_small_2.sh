#!/bin/bash
# Queue 2: CPU at the last healthy offload point (N=3), to decide cause vs effect of the
# N=4 decode-only-offload collapse (host CPU 93% busy there).
cd "$(dirname "$0")/../.."
while pgrep -f queue_small_1.sh >/dev/null; do sleep 20; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
$PY $D --cohort small --n 3 --arm mps --cells doff foff --name-suffix _cpu
