#!/bin/bash
# D10: enable_async_loading grid (vLLM 0.15.1, compiled kernels), detached from the session.
cd /home/jiaxuan/Documents/Projects/kvcache-vram_size
export OMP_NUM_THREADS=8
for b in 24 26 28; do
  .venv/bin/python scripts/kv_vram/kv_sweep.py --method cops_async --budgets $b --qps 1 1.5 2 2.5 3 3.5 4 4.5 5 --trace --stall-timeout 120 --name-suffix _g
  echo "budget $b exit $? $(date +%H:%M)"
  bash scripts/kv_vram/kv_stop.sh > /dev/null 2>&1
done
echo "async grid complete $(date +%H:%M)"
