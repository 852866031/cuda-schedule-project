#!/usr/bin/env bash
# Night of Nov 16: when the GPU 1 chain is done, test the compiled tenant under shape changes (before c2a starts on GPU 0).
cd "$(dirname "$0")"
until grep -q "GPU 1 block done" logs_o3/night_gpu1.log; do sleep 5; done
echo "$(date +%H:%M:%S) start compile test" >> logs_o3/night_gpu1.log
SCRIPT=compile_test.py GPU=1 ./run_knee_corun.sh > logs_o3/compile_test.log 2>&1
echo "$(date +%H:%M:%S) end compile test rc=$?" >> logs_o3/night_gpu1.log
