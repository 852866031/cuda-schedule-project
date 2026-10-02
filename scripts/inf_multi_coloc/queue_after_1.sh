#!/bin/bash
# After the medium MPS sweep and the small-cohort queue 6. Strictly serial.
#  1. medium: no-MPS (N=2), MPS+gate (N=2,4), eager-no-gate control (N=4); thread cap on.
#  2. prefill-only calibration (small, N=1, no scheduler): per-model prefill capacity at
#     6144-token prompts, to size the N=2/4 external-scheduler runs.
cd "$(dirname "$0")/../.."
sleep 180
while pgrep -f "queue_small_[1-6].sh|queue_medium_(probe2?|[12]).sh" >/dev/null; do sleep 30; done
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
export OMP_NUM_THREADS=4
$PY $D --cohort medium --n 2 --arm nomps --cells dfits
$PY $D --cohort medium --n 2 4 --arm gate --cells dfits
$PY $D --cohort medium --n 4 --arm eager --cells dfits
for q in 4 8 16 24; do
    $PY $D --cohort small --n 1 --arm mps --cells pfill --sched none --pf-qps $q --name-suffix _pq$q
done
