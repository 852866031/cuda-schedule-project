#!/usr/bin/env bash
# Night of Nov 21/22, GPU 1, tenant alone: bytes per segment of the best tenant (blocks compiled, output layer over vocabulary
# blocks of 4,096 with fused kernels) in the shapes other than the usual one: 1, 2 and 8 micro-batches (TIERS index 0, 1, 3).
C=$(cd "$(dirname "$0")" && pwd); cd "$C"
for t in 1 3 0; do
  SCRIPT=segment_bytes.py GPU=1 ./run_knee_corun.sh --variants cb_vocab4096f --tier $t --sms 32 --compile-wait 70 > results_prof/segment_bytes_run_tier$t.log 2>&1
done
echo "$(date +%H:%M:%S) shapes done" >> results_prof/gpu1_night.log
