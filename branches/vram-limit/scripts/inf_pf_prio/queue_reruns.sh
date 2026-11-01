#!/bin/bash
# Reruns of cells whose measurement window swapped > 0.5 GB in (decided before rerunning;
# both runs are kept: the rerun has an _r2 suffix). After queue_prio.sh.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_pf_prio/pfp_sweep.py
export OMP_NUM_THREADS=8
A=(--budget 23 --sm-quant "" --sm-util 0.205)
B=(--budget 25 --sm-quant fp8 --sm-util 0.13 --sm-first)
step() { echo "### $(date '+%H:%M') $*"; }
step "reruns A"
$PY $D --cells p2 --arm fair   --sm-qps 4 "${A[@]}" --name-suffix _b23_r2
$PY $D --cells p2 --arm mps8b  --sm-qps 2 "${A[@]}" --name-suffix _b23_r2
$PY $D --cells p2 --arm gate8b --sm-qps 4 "${A[@]}" --name-suffix _b23_r2
$PY $D --cells p2 --arm mps8b  --sm-qps 8 "${A[@]}" --name-suffix _b23_r2
$PY $D --cells p2 --arm gate3b --sm-qps 4 "${A[@]}" --name-suffix _b23_r2
step "reruns B"
$PY $D --cells base --arm eager8b "${B[@]}" --name-suffix _fp8b25_r2
$PY $D --cells p2 --arm gate8b --sm-qps 8 "${B[@]}" --name-suffix _fp8b25_r2
$PY $D --cells p2 --arm gate3b --sm-qps 2 "${B[@]}" --name-suffix _fp8b25_r2
$PY $D --cells p2 --arm eager8b --sm-qps 8 "${B[@]}" --name-suffix _fp8b25_r2
step done
