#!/bin/bash
# D10 deviation: after the async grid, rerun every cops_async cell that failed (LMCache's
# 3-s lookup-timeout KeyError at deep queues) with lookup_timeout_ms = 30000.
cd /home/jiaxuan/Documents/Projects/kvcache-vram_size
until grep -q "async grid complete" output/kv_vram/drivers/async_grid.log; do sleep 30; done
export OMP_NUM_THREADS=8
for b in 24 26 28; do
  qs=$(.venv/bin/python - "$b" <<'PY'
import csv, sys
b = sys.argv[1]
ok = {}
for r in csv.DictReader(open("output/summary_kv.csv")):
    n = r["name"]
    if n.startswith(f"kv_cops_async_b{b}_q") and n.endswith("_tr_g"):
        ok[r["qps"]] = ok.get(r["qps"], False) or r["ok"] == "True"
print(" ".join(q for q, v in sorted(ok.items(), key=lambda x: float(x[0])) if not v))
PY
)
  [ -z "$qs" ] && continue
  echo "budget $b: rerun with t30 at qps $qs"
  .venv/bin/python scripts/kv_vram/kv_sweep.py --method cops_async_t30 --budgets $b --qps $qs --trace --stall-timeout 120 --name-suffix _g
  bash scripts/kv_vram/kv_stop.sh > /dev/null 2>&1
done
echo "async t30 reruns complete $(date +%H:%M)"
