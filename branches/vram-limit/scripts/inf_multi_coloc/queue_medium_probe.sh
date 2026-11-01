#!/bin/bash
# Size the medium cohort (Qwen2.5-3B) before sweeping: one model, decode-only fits,
# smoke-length -> KV grant, GPU1 footprint, host RSS. Runs after the small queues.
cd "$(dirname "$0")/../.."
while pgrep -f "queue_small_[12].sh" >/dev/null; do sleep 20; done
.venv/bin/python scripts/inf_multi_coloc/multi_sweep.py --cohort medium --n 1 --arm mps \
    --cells dfits doff --smoke
