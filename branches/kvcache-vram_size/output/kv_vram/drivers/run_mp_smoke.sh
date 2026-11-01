#!/bin/bash
# D9 smoke B: LMCacheMPConnector (asynchronous load), MP server in .venv-lmc with 32 GiB pinned L1.
cd /home/jiaxuan/Documents/Projects/kvcache-vram_size
export OMP_NUM_THREADS=8 MP_L1_GB=32
.venv/bin/python scripts/kv_vram/kv_sweep.py --method mp --budgets 24 --qps 2.5 --trace --stall-timeout 120 --name-suffix "${SUFFIX:-}"
echo "mp exit $? $(date +%H:%M)"
bash scripts/kv_vram/kv_stop.sh > /dev/null 2>&1
echo "mp smoke done $(date +%H:%M)"
