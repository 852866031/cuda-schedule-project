#!/usr/bin/env bash
# Overnight plan 3, real-engine block R4 on GPU 0: ten runs of the 11-minute reference schedule (run_eval7.sh).
# The controls run first because they need nothing from the synthetic runs on GPU 1; c1_* wait for R2's
# bytes-per-token table, c1_fine for R1's slice length, c2a/c2b for R2's choice of variants.
#   run_o3_gpu0.sh [first scheme to run, to resume]
set -uo pipefail
C=$(cd "$(dirname "$0")" && pwd); R=$C/results_o3; L=$C/logs_o3; mkdir -p "$R" "$L"
LOG=$L/night_gpu0.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
free_check() { local u; u=$(nvidia-smi --query-gpu=memory.used -i 0 --format=csv,noheader,nounits); if (( u > 700 )); then say "GPU 0 has $u MiB in use before a run: stopping"; exit 7; fi; }
wait_for() { local f=$1 max=$2 t=0; while [[ ! -s $f && $t -lt $max ]]; do sleep 20; t=$((t+20)); done; [[ -s $f ]] && say "found $(basename "$f")" || say "no $(basename "$f") after ${max}s: going on with the default"; }
run() { local scheme=$1 tag=${2:-$1}; free_check; say "start $tag"
  TAG=$tag "$C/run_eval7.sh" "$scheme" > "$L/${tag}_run.log" 2>&1; local rc=$?
  say "end $tag rc=$rc $(grep -c . "$R/${tag}_load.jsonl" 2>/dev/null) request rows"; sleep 15; }
START=${1:-adjust_notenant}; go=0
for step in adjust_notenant tpot_slo15 green_dyn2 c1_fix c1_slow15 c1_slow5 c1_fine c2b c2a adjust_notenant_end; do
  [[ $step == "$START" ]] && go=1; [[ $go == 1 ]] || continue
  case $step in
    c1_fix) wait_for "$R/bytes_per_token.json" 1200; run c1_fix;;
    c1_fine) wait_for "$R/r1_slice.txt" 1200; run c1_fine;;
    c2a|c2b) wait_for "$R/r2_variants.json" 600; run "$step";;
    adjust_notenant_end) run adjust_notenant adjust_notenant_end;;
    *) run "$step";;
  esac
done
say "R4 done"
