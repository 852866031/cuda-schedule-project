#!/usr/bin/env bash
# Overnight plan 3, GPU 0 after R4: R5 (every shape under the same allowance) and R7 (prefill tenant: pf_mps again, then
# the tenant engine on prebuilt green contexts that follow the VRAM scheduler). Waits for the R4 chain to finish.
set -uo pipefail
C=$(cd "$(dirname "$0")" && pwd); R=$C/results_o3; L=$C/logs_o3; LOG=$L/night_gpu0.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q "R4 done" "$LOG"; do sleep 10; done
free_check() { local u; u=$(nvidia-smi --query-gpu=memory.used -i 0 --format=csv,noheader,nounits); if (( u > 700 )); then say "GPU 0 has $u MiB in use before a run: stopping"; exit 7; fi; }
free_check; say "start probe2 (R5)"; "$C/run_probe2.sh" probe2 > "$L/probe2_run.log" 2>&1; say "end probe2 rc=$?"; sleep 15
for arm in pf_mps pf_green_dyn; do
  free_check; say "start $arm (R7)"; "$C/run_eval_pf2.sh" "$arm" > "$L/${arm}_run.log" 2>&1; say "end $arm rc=$?"; sleep 15
done
say "GPU 0 block done"
