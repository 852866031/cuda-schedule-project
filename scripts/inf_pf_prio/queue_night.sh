#!/bin/bash
# Overnight 2026-10-17/18 (user: ~8 h, run, write the report, then fill gaps).
# 1. config B's 8B-alone checkpoint; 2. the full two-configuration queue.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=8
echo "### $(date '+%H:%M') checkpoint B: 8B alone at b25"
.venv/bin/python scripts/inf_pf_prio/pfp_sweep.py --cells base --budget 25 --sm-quant fp8 --sm-util 0.13 --sm-first --name-suffix _fp8b25
bash scripts/inf_pf_prio/queue_prio.sh
