#!/bin/bash
# Medium cohort (Qwen2.5-3B, util 0.22, fits=2 / offload=9 sessions) MPS sweep N=1-4,
# stock settings (no thread cap) for comparability with the small cohort. After queue 5.
cd "$(dirname "$0")/../.."
sleep 120
while pgrep -f "queue_small_[12345].sh|queue_medium_probe2?.sh" >/dev/null; do sleep 20; done
.venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort medium --n 1 2 3 4 --arm mps
