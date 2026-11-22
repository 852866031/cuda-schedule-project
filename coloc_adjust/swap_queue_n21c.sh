#!/usr/bin/env bash
# One-off, night of Nov 21/22: after the run cvf_step, replace queue D by the queue in which the best tenant uses its own table of
# GPU-memory needs per shape (cvfm_*), plus tonight's reruns of the Nov 16 methods, repeats and baselines.
C=$(cd "$(dirname "$0")" && pwd); LOG=$C/logs_n21/batch_gpu0.log
until grep -q "end cvf_step rc=" "$LOG"; do sleep 2; done
pkill -f "run_n21_gpu0d.sh cvf_step" && echo "$(date +%H:%M:%S) queue D stopped after cvf_step" >> "$LOG"
sleep 20
Q=notenant_b,cvfm_step,cvfm_w12,c2a,cvfm_slow5_w12,cvfm_slow5_step,cvfm_w12_r2,cvfm_step_r2,cvfm_slow15_w12,cvfm_slow15_step,c2a_r2,notenant_c,cvfm_w12_half,cvfm_w12_x2,cvfm_w12_sm16,cvfm_w12_sm64,c1_fix_r2,v_step_r2,v_w12_r2,c1_slow5,cvfm_w12_r3,cvfm_step_r3,cvfm_slow5_w12_r2,notenant_d
echo "$Q" > "$C/logs_n21/queue_E.txt"
exec "$C/run_n21_gpu0e.sh" "$Q" > "$C/logs_n21_batchE.out" 2>&1
