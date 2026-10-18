#!/bin/bash
# S2b: swap-free attempts at the two config-A mps8b points that swapped twice. After queue_supp.sh.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_pf_prio/pfp_sweep.py
export OMP_NUM_THREADS=8
A=(--budget 23 --sm-quant "" --sm-util 0.205)
echo "### $(date '+%H:%M') S2b"
$PY $D --cells p2 --arm mps8b --sm-qps 8 "${A[@]}" --name-suffix _b23_r3
$PY $D --cells p2 --arm mps8b --sm-qps 2 "${A[@]}" --name-suffix _b23_r3
echo "### $(date '+%H:%M') done"
