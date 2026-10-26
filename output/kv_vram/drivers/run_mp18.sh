#!/bin/bash
# D11: MP connector on vLLM 0.18.1 (built-in LMCacheMPConnector, MP server in .venv-vllm018,
# 32 GiB pinned L1). Waits for the async work; smoke first, grid only if the smoke is ok.
cd /home/jiaxuan/Documents/Projects/kvcache-vram_size
export OMP_NUM_THREADS=8 MP_L1_GB=32
.venv/bin/python scripts/kv_vram/kv_sweep.py --method mp18 --budgets 24 --qps 2.5 --trace --stall-timeout 120 --name-suffix _r2
bash scripts/kv_vram/kv_stop.sh > /dev/null 2>&1
ok=$(.venv/bin/python -c "
import csv
print(any(r['name']=='kv_mp18_b24_q2.5_tr_r2' and r['ok']=='True' for r in csv.DictReader(open('output/summary_kv.csv'))))")
echo "mp18 smoke ok=$ok $(date +%H:%M)"
[ "$ok" = True ] || { echo "mp18 smoke failed -- grid not started"; exit 1; }
for b in 24 26 28; do
  .venv/bin/python scripts/kv_vram/kv_sweep.py --method mp18 --budgets $b --qps 1 1.5 2 2.5 3 3.5 4 4.5 5 --trace --stall-timeout 120 --name-suffix _g
  echo "mp18 budget $b exit $? $(date +%H:%M)"
  bash scripts/kv_vram/kv_stop.sh > /dev/null 2>&1
done
echo "mp18 grid complete $(date +%H:%M)"
