#!/usr/bin/env bash
# Overnight plan 3, synthetic block on GPU 1 after R0: R2 (bytes per token), R1 (C1 pacing), R3 (C2 next to decode), R6 (floor).
#   run_o3_gpu1.sh [first step: r2|r1|r3|r6]      or   STEPS="r1 r6" run_o3_gpu1.sh
set -uo pipefail
C=$(cd "$(dirname "$0")" && pwd); R=$C/results_o3; L=$C/logs_o3; LOG=$L/night_gpu1.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
free_check() { local u; u=$(nvidia-smi --query-gpu=memory.used -i 1 --format=csv,noheader,nounits); if (( u > 500 )); then say "GPU 1 has $u MiB in use before a run: stopping"; exit 7; fi; }
START=${1:-r2}; go=0
for step in ${STEPS:-r2 r1 r3 r6}; do
  [[ $step == "$START" || -n ${STEPS:-} ]] && go=1; [[ $go == 1 ]] || continue
  free_check; say "start $step"
  case $step in
    r2) SCRIPT=tenant_bytes.py GPU=1 "$C/run_knee_corun.sh" > "$L/r2.log" 2>&1;;
    r1) SCRIPT=corun2.py GPU=1 "$C/run_knee_corun.sh" --plan r1 > "$L/r1.log" 2>&1
        (cd "$C" && source ../kvcached_validation/.venv-main/bin/activate && python o3_synth_tables.py r1 --decide > "$L/r1_table.log" 2>&1);;
    r3) SCRIPT=corun2.py GPU=1 "$C/run_knee_corun.sh" --plan r3 > "$L/r3.log" 2>&1
        (cd "$C" && source ../kvcached_validation/.venv-main/bin/activate && python o3_synth_tables.py r3 > "$L/r3_table.log" 2>&1);;
    r6) SCRIPT=aggr_corun.py GPU=1 "$C/run_knee_corun.sh" --sms 32,8 --batches 1,8 --window 6 --fixed-split "" \
          --modes "tiny:-1;mm128:-1;compute:-1;mm2304:-1" --out "$R/r6_floor.csv" > "$L/r6.log" 2>&1;;
  esac
  say "end $step rc=$?"; sleep 10
done
say "GPU 1 block done"
