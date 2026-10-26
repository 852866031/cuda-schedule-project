#!/bin/bash
# D9 smoke A: enable_async_loading with compiled kernels; rerun with a 30 s lookup timeout if it fails.
cd /home/jiaxuan/Documents/Projects/kvcache-vram_size
export OMP_NUM_THREADS=8
.venv/bin/python scripts/kv_vram/kv_sweep.py --method cops_async --budgets 24 --qps 2.5 --trace --stall-timeout 120
st=$?
echo "cops_async exit $st $(date +%H:%M)"
if ! grep -q '"name": "kv_cops_async_b24_q2.5_tr"' output/raw/kv_cops_async_b24_q2.5_tr.json 2>/dev/null || \
   [ "$(.venv/bin/python -c "import json;print(json.load(open('output/raw/kv_cops_async_b24_q2.5_tr.json')).get('ok'))" 2>/dev/null)" != True ]; then
    .venv/bin/python scripts/kv_vram/kv_sweep.py --method cops_async_t30 --budgets 24 --qps 2.5 --trace --stall-timeout 120
    echo "cops_async_t30 exit $? $(date +%H:%M)"
fi
echo "async smoke done $(date +%H:%M)"
