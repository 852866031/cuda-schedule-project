#!/usr/bin/env bash
# One-off for the night of Nov 21/22: when the run v_w12 of the first batch has ended, stop that batch between two runs
# and start the batch that uses the fused-kernel tenant (run_n21_gpu0c.sh).
C=$(cd "$(dirname "$0")" && pwd); LOG=$C/logs_n21/batch_gpu0.log
until grep -q "end v_w12 rc=" "$LOG"; do sleep 2; done
pkill -f "run_n21_gpu0.sh v_step" && echo "$(date +%H:%M:%S) first batch stopped after v_w12" >> "$LOG"
sleep 20
"$C/run_n21_gpu0c.sh" cvf_step,cvf_w12,notenant_end,cvf_slow5_step,cvf_slow5_w12,cvf_slow15_step,cvf_slow15_w12,v_step_r2,v_w12_r2,cvf_step_r2,cvf_w12_r2,cv_step,cv_w12,notenant_end2 > "$C/logs_n21_batchC.out" 2>&1
