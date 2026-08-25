#!/bin/bash
# Interleaved-placement sweeps at prefetch depth 1 and 2.
cd "$(dirname "$0")"
../.venv/bin/python finetune_sweep.py --batch 2 --steps 8 --warmup 3 --prefetch \
    --prefetch-depth 1 --pattern interleave --max-offload 16 --tag ft_int_d1
../.venv/bin/python finetune_sweep.py --batch 2 --steps 8 --warmup 3 --prefetch \
    --prefetch-depth 2 --pattern interleave --max-offload 16 --tag ft_int_d2
echo "INTERLEAVE SWEEPS COMPLETE"
