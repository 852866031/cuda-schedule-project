#!/usr/bin/env bash
# Night of Nov 21/22: when queue F is done, run the reserve block (third repeats and the Nov 16 SLO-aware run at 15 %).
C=$(cd "$(dirname "$0")" && pwd); LOG=$C/logs_n21/batch_gpu0.log
until grep -q "batch done: notenant_b" "$LOG"; do sleep 5; done
sleep 5
Q=c1_slow15,cvfm_slow5_step_r2,cvfm_slow15_w12_r2,cvfm_step_r4,notenant_e
echo "$Q" > "$C/logs_n21/queue_G.txt"
exec "$C/run_n21_gpu0g.sh" "$Q" > "$C/logs_n21_batchG.out" 2>&1
