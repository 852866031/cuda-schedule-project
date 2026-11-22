#!/usr/bin/env bash
# One-off, night of Nov 21/22: let the run in progress (cvf_step) finish, then stop the queue before it starts the next one.
C=$(cd "$(dirname "$0")" && pwd); LOG=$C/logs_n21/batch_gpu0.log
until grep -q "end cvf_step rc=" "$LOG"; do sleep 2; done
pkill -f "run_n21_gpu0d.sh cvf_step" && echo "$(date +%H:%M:%S) queue paused after cvf_step (waiting for the plan to be agreed)" >> "$LOG"
