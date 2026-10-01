#!/bin/bash
# Rerun N=8 decode-only fits (first run: 1 of 2400 requests failed with a client-side
# Broken pipe). Strictly after the medium probe -- one stack set at a time.
cd "$(dirname "$0")/../.."
sleep 30
while pgrep -f "queue_small_[12].sh|queue_medium_probe.sh" >/dev/null; do sleep 20; done
.venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort small --n 8 --arm mps \
    --cells dfits --name-suffix _r2
