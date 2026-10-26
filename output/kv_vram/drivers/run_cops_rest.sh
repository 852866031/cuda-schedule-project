#!/bin/bash
# Remaining cops grid cells, detached from the Claude session (setsid nohup).
cd /home/jiaxuan/Documents/Projects/kvcache-vram_size
export OMP_NUM_THREADS=8
.venv/bin/python scripts/kv_vram/kv_sweep.py --method cops --budgets 26 --qps 2 2.5 3 3.5 4 4.5 5 --trace --stall-timeout 120
.venv/bin/python scripts/kv_vram/kv_sweep.py --method cops --budgets 28 --qps 1 1.5 2 2.5 3 3.5 4 4.5 5 --trace --stall-timeout 120
echo "cops grid complete $(date +%H:%M)"
