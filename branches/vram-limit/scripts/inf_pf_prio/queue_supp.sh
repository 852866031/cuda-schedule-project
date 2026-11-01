#!/bin/bash
# Supplements S1 (tight gate) and S2 (repeats), DECISIONS_PF_PRIO.md section C. After the reruns.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_pf_prio/pfp_sweep.py
export OMP_NUM_THREADS=8
A=(--budget 23 --sm-quant "" --sm-util 0.205)
B=(--budget 25 --sm-quant fp8 --sm-util 0.13 --sm-first)
step() { echo "### $(date '+%H:%M') $*"; }
step "S1 tight gate"
$PY $D --cells p2 --arm gate8bt --sm-qps 4 8 "${A[@]}" --name-suffix _b23
$PY $D --cells p2 --arm gate8bt --sm-qps 8 "${B[@]}" --name-suffix _fp8b25
step "S2 repeats"
$PY $D --cells p2 --arm fair  --sm-qps 8 "${A[@]}" --name-suffix _b23_r2
$PY $D --cells p2 --arm mps8b --sm-qps 4 "${A[@]}" --name-suffix _b23_r2
step done
