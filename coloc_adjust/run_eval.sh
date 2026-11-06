#!/usr/bin/env bash
# One evaluation scheme end to end (design §3.7).
#   run_eval.sh <scheme> [schedule]      scheme = decode_alone | adjust_notenant | static_split | adjust | adjust_nogate | adjust_mps20
# Starts: MPS daemon, decode-only engine (kvcached main, 28 GiB DRAM tier, prefix caching off,
# busy-window patch + transfer tracer), router + VRAM scheduler on :8001, sampler, FT worker (per scheme),
# then the reference prefix workload with the RPS schedule through the router; tears everything
# down and runs analyze_coloc.py. Logs in coloc_adjust/logs, results in coloc_adjust/results.
set -uo pipefail
ARM=$1; SCHED=${2:-"0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"}
DUR=${DUR:-660}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
export VENV_NAME=.venv-main
source "$KV/scripts/env.sh"
export COLOC_XFER_DIR=/dev/shm
S=$C/results/$ARM; L=$C/logs; mkdir -p "$C/results" "$L"
rm -f /dev/shm/coloc_xfer_inf.csv /dev/shm/coloc_xfer_ft.csv /dev/shm/coloc_ft_budget /dev/shm/coloc_ft_state /dev/shm/coloc_ft_snapshot.pt /dev/shm/coloc_hp_busy
"$KV/scripts/stop.sh" > /dev/null 2>&1
pkill -f "[c]oloc_adjust/controller.py" 2>/dev/null; pkill -f "[c]oloc_adjust/ft_worker.py" 2>/dev/null; sleep 1   # bracket: the pattern must not match a shell whose command line quotes it

# --- MPS (user-space daemon on GPU 0); every GPU process below inherits the pipe dir
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps/log   # short path: Unix-socket length limit
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [[ ${MPS:-1} == 1 ]]; then CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; sleep 1; fi

# --- engine
export PYTHONPATH=$C/hp_patch COLOC_HP_SIGNAL=1 COLOC_XFER_TRACE=1
"$KV/scripts/serve.sh" "coloc_$ARM" kvcached --offload 28 > /dev/null
if ! "$KV/scripts/wait_ready.sh" "coloc_$ARM" 600 > "$L/${ARM}_engine_start.txt" 2>&1; then echo "engine failed"; cat "$L/${ARM}_engine_start.txt" | tail -5; exit 3; fi
unset PYTHONPATH COLOC_HP_SIGNAL COLOC_XFER_TRACE

# --- router / VRAM scheduler
CTL_OPTS=(--listen 8001 --upstream http://localhost:$VLLM_PORT --ipc "$KV_IPC" --log "${S}_controller.csv")
case $ARM in
  decode_alone) CTL_OPTS+=(--no-adjust);;
  adjust_notenant) ;;   # VRAM scheduler adjusts the KV limit by load, nobody uses the freed VRAM: the baseline for the prototype
  static_split) CTL_OPTS+=(--no-adjust --fixed-budget-gib "${STATIC_FT_GIB:-6}"); kvctl limit "$KV_IPC" "${STATIC_KV_GIB:-6.7}G" > /dev/null;;
  adjust|adjust_nogate|adjust_mps20) ;;
esac
python "$C/controller.py" "${CTL_OPTS[@]}" > "$L/${ARM}_controller.log" 2>&1 & CP=$!
sleep 3
# --- sampler
python "$KV/scripts/sampler.py" --out "${S}_sampler.csv" --duration $((DUR + 200)) --ipc "$KV_IPC" --metrics-url "http://localhost:$VLLM_PORT/metrics" & SP=$!
# --- tenant
FP=""
case $ARM in
  static_split) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --gate --fresh --fixed-budget-gib "${STATIC_FT_GIB:-6}" > "$L/${ARM}_ft.log" 2>&1 & FP=$!;;
  adjust)       CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --gate --fresh > "$L/${ARM}_ft.log" 2>&1 & FP=$!;;
  adjust_nogate) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh > "$L/${ARM}_ft.log" 2>&1 & FP=$!;;
  adjust_mps20)  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${MPS_PCT:-20} CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh > "$L/${ARM}_ft.log" 2>&1 & FP=$!;;   # spatial cap instead of the gate
esac
sleep 5
# --- load through the router
python "$KV/scripts/loadgen.py" --mode prefix --rate 1 --rps-schedule "$SCHED" --duration "$DUR" --url http://localhost:8001 \
    --n-sessions 32 --prefix-tokens 6144 --unique-tokens 128 --max-tokens 128 --skew zipf --out "${S}_load.jsonl" 2>&1 | tee "$L/${ARM}_load.log" | grep -v "^{" | tail -3
# --- teardown
[[ -n $FP ]] && kill $FP 2>/dev/null; sleep 2
kill $SP $CP 2>/dev/null; sleep 1
cp /dev/shm/coloc_xfer_inf.csv "${S}_xfer_inf.csv" 2>/dev/null; cp /dev/shm/coloc_xfer_ft.csv "${S}_xfer_ft.csv" 2>/dev/null
cp "${S}_load.jsonl.t0" "${S}_t0.txt" 2>/dev/null
"$KV/scripts/stop.sh" > /dev/null 2>&1
if [[ ${MPS:-1} == 1 ]]; then echo quit | nvidia-cuda-mps-control; fi
python "$C/analyze_coloc.py" --arm "$ARM" --dir "$C/results" --schedule "$SCHED" 2>&1 | tail -30
echo "DONE $ARM"
