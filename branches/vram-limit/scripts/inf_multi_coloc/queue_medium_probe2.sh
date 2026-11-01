#!/bin/bash
# Re-probe the medium cohort at util 0.22 (KV grant + GPU1 footprint) before sweeping.
cd "$(dirname "$0")/../.."
sleep 90
while pgrep -f "queue_small_[1234].sh|queue_medium_probe.sh" >/dev/null; do sleep 20; done
.venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort medium --n 1 --arm mps \
    --cells dfits doff --smoke --name-suffix _u22
