#!/bin/bash
# Ninth study, full run (planned 2026-10-17; predictions in PLAN_PF_PRIO.md, addendum).
# Two configurations, run one after the other, strictly serial, every launch through
# pfp_launch.sh (mem_guard, heartbeat, RAM gate):
#   A  bf16 3B (util 0.205), 8B decode b23, 8B starts first     suffix _b23
#   B  fp8  3B (util 0.13),  8B decode b25, 3B starts first     suffix _fp8b25
# The 8B-alone checkpoints (pfp_base_fair_b23, pfp_base_fair_fp8b25) run before this.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_pf_prio/pfp_sweep.py
export OMP_NUM_THREADS=8
step() { echo "### $(date '+%H:%M') $*"; }
A=(--budget 23 --sm-quant "" --sm-util 0.205 --name-suffix _b23)
B=(--budget 25 --sm-quant fp8 --sm-util 0.13 --sm-first --name-suffix _fp8b25)
for cfg in A B; do
    if [ "$cfg" = A ]; then C=("${A[@]}"); else C=("${B[@]}"); fi
    step "$cfg: 8B alone, eager decode (reference for gate3b)"
    $PY $D --cells base --arm eager8b "${C[@]}"
    step "$cfg: 3B alone"
    $PY $D --cells s2 --sm-qps 2 4 8 "${C[@]}"
    for arm in fair mps8b gate8b mps3b gate3b; do
        step "$cfg: $arm"
        $PY $D --cells p2 --arm $arm --sm-qps 2 4 8 "${C[@]}"
    done
    step "$cfg: eager-without-gate control"
    $PY $D --cells p2 --arm eager8b --sm-qps 8 "${C[@]}"
done
step done
