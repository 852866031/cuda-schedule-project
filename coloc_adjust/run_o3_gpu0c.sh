#!/usr/bin/env bash
# Night of Nov 16, R7 again: the prefill tenant engine without CUDA graphs (--enforce-eager), shared SMs and scheduler-driven green contexts.
C=$(cd "$(dirname "$0")" && pwd); L=$C/logs_o3; LOG=$L/night_gpu0.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
for arm in pf_mps_eager pf_green_dyn_eager; do
  u=$(nvidia-smi --query-gpu=memory.used -i 0 --format=csv,noheader,nounits); if (( u > 700 )); then say "GPU 0 has $u MiB in use: stopping"; exit 7; fi
  say "start $arm (R7, eager)"; "$C/run_eval_pf2.sh" "$arm" > "$L/${arm}_run.log" 2>&1; say "end $arm rc=$?"; sleep 15
done
say "R7 eager done"
