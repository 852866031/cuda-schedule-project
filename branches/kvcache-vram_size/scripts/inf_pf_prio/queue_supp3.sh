#!/bin/bash
# S3: fp8 3B beside the 8B at b23 (cross of configs A and B). After queue_supp2.sh.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=8
echo "### $(date '+%H:%M') S3"
.venv/bin/python scripts/inf_pf_prio/pfp_sweep.py --cells p2 --arm fair --sm-qps 4 8 \
    --budget 23 --sm-quant fp8 --sm-util 0.13 --sm-first --name-suffix _fp8b23
echo "### $(date '+%H:%M') done"
