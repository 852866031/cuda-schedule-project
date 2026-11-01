#!/bin/bash
# Last two cells: S3 8 req/s swap rerun; A mps8b 2 req/s third attempt (port failure).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=8; PY=.venv/bin/python; D=scripts/inf_pf_prio/pfp_sweep.py
echo "### $(date '+%H:%M') S4"
$PY $D --cells p2 --arm fair --sm-qps 8 --budget 23 --sm-quant fp8 --sm-util 0.13 --sm-first --name-suffix _fp8b23_r2
$PY $D --cells p2 --arm mps8b --sm-qps 2 --budget 23 --sm-quant "" --sm-util 0.205 --name-suffix _b23_r3
echo "### $(date '+%H:%M') done"
