#!/bin/bash
# S5: the 8B's own load raised to 2.5 QPS (config A, 3B at 4 req/s): fair vs mps8b, plus the
# 8B alone at 2.5 and 3 QPS. After the gate8bt rerun.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=8; PY=.venv/bin/python; D=scripts/inf_pf_prio/pfp_sweep.py
A=(--budget 23 --sm-quant "" --sm-util 0.205)
echo "### $(date '+%H:%M') S5"
$PY $D --cells base --qps 2.5 "${A[@]}" --name-suffix _b23_big2.5
$PY $D --cells base --qps 3 "${A[@]}" --name-suffix _b23_big3
$PY $D --cells p2 --arm fair  --qps 2.5 --sm-qps 4 "${A[@]}" --name-suffix _b23_big2.5
$PY $D --cells p2 --arm mps8b --qps 2.5 --sm-qps 4 "${A[@]}" --name-suffix _b23_big2.5
echo "### $(date '+%H:%M') done"
