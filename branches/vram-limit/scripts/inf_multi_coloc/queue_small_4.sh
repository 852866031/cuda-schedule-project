#!/bin/bash
# Instrumented rerun of the N=4 decode-only-offload collapse: per-second host CPU by
# role (host_monitor.py) + archived engine logs -> what moves first. After queue 3.
cd "$(dirname "$0")/../.."
sleep 60
while pgrep -f "queue_small_[123].sh|queue_medium_probe.sh" >/dev/null; do sleep 20; done
.venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort small --n 4 --arm mps \
    --cells doff --name-suffix _trace
