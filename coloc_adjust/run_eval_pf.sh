#!/usr/bin/env bash
# P3: a prefill-only tenant (Qwen2.5-3B fp8 on vLLM 0.30 + kvcached, KV streamed to host RAM) beside the
# 8B decode engine, both on GPU 0 under the VRAM scheduler.
#   run_eval_pf.sh <scheme> [schedule]    scheme = pf_mps | pf_mps20 | pf_split | pf_solo
#     pf_mps    both engines share all SMs through MPS
#     pf_mps20  the tenant engine capped at 20 % of the SM threads (MPS)
#     pf_split  the 8B on the first 128 SMs (green context), the tenant on the remaining 42
#     pf_solo   the tenant alone (its own throughput/latency reference; main load still runs, no... see below)
# Env: PF_RATE (tenant req/s, default 1), PF_PROMPT (prompt tokens, default 3072), PF_SLEEP=1 to enable the sleep tier.
set -uo pipefail
ARM=$1; SCHED=${2:-"0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"}
DUR=${DUR:-660}; PF_RATE=${PF_RATE:-1}; PF_PROMPT=${PF_PROMPT:-3072}; PF_MODEL=${PF_MODEL:-Qwen/Qwen2.5-3B}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
PL=$(nvidia-smi --query-gpu=power.limit -i 0 --format=csv,noheader,nounits | cut -d. -f1)
if [[ -z $PL || $PL -gt 500 ]]; then echo "GPU 0 power limit is ${PL:-unknown} W; requires <= 500 W"; exit 9; fi
export VENV_NAME=.venv-main
source "$KV/scripts/env.sh"
export COLOC_XFER_DIR=/dev/shm
S=$C/results/$ARM; L=$C/logs; mkdir -p "$C/results" "$L"
rm -f /dev/shm/coloc_xfer_inf.csv /dev/shm/coloc_xfer_ft.csv /dev/shm/coloc_ft_budget /dev/shm/coloc_ft_state /dev/shm/coloc_hp_busy
stop_own() { for tag in "coloc_${ARM}" "coloc_${ARM}_pf"; do local pf="$KV/logs/vllm_${tag}.pid"; local pid; pid=$(cat "$pf" 2>/dev/null)
    if [[ -n $pid ]]; then pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; fi; done; sleep 4
  for tag in "coloc_${ARM}" "coloc_${ARM}_pf"; do local pf="$KV/logs/vllm_${tag}.pid"; local pid; pid=$(cat "$pf" 2>/dev/null)
    if [[ -n $pid ]]; then pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; rm -f "$pf"; fi; done
  pkill -f "[s]cripts/loadgen.py --mode prefix" 2>/dev/null; pkill -f "[s]cripts/loadgen.py --mode stream" 2>/dev/null; pkill -f "[s]ampler.py --out $C/results" 2>/dev/null; }
stop_own; pkill -f "[c]oloc_adjust/controller.py" 2>/dev/null; pkill -f "[c]oloc_adjust/pf_tenant.py" 2>/dev/null; sleep 1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; sleep 1
nv_used() { nvidia-smi --query-gpu=memory.used -i 0 --format=csv,noheader,nounits; }
# --- 8B decode engine (GPU 0)
export PYTHONPATH=$C/hp_patch COLOC_HP_SIGNAL=1 COLOC_XFER_TRACE=1
[[ $ARM == pf_split ]] && export COLOC_ENGINE_SMS=128
"$KV/scripts/serve.sh" "coloc_$ARM" kvcached --offload 28 > /dev/null
if ! "$KV/scripts/wait_ready.sh" "coloc_$ARM" 600 > "$L/${ARM}_engine_start.txt" 2>&1; then echo "engine failed"; tail -5 "$L/${ARM}_engine_start.txt"; exit 3; fi
unset PYTHONPATH COLOC_HP_SIGNAL COLOC_XFER_TRACE COLOC_ENGINE_SMS
sleep 3; U0=$(nv_used)
# --- tenant engine: 3B fp8, own port / kvcached instance, KV offloaded to 8 GiB of host RAM
PF_ENV=(MODEL="$PF_MODEL" VLLM_PORT=8200 KV_IPC=vllm_c)
PF_ARGS=(--quantization fp8 --max-model-len 4608 --max-num-seqs 8 --max-num-batched-tokens 8192)
[[ ${PF_SLEEP:-0} == 1 ]] && PF_ARGS+=(--enable-sleep-mode)
case $ARM in
  pf_mps20) env "${PF_ENV[@]}" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${MPS_PCT:-20} "$KV/scripts/serve.sh" "coloc_${ARM}_pf" kvcached --offload 8 -- "${PF_ARGS[@]}" > /dev/null;;
  pf_split) env "${PF_ENV[@]}" PYTHONPATH=$C/hp_patch COLOC_ENGINE_SMS=128 COLOC_ENGINE_SMS_REMAINING=1 "$KV/scripts/serve.sh" "coloc_${ARM}_pf" kvcached --offload 8 -- "${PF_ARGS[@]}" > /dev/null;;
  *)        env "${PF_ENV[@]}" "$KV/scripts/serve.sh" "coloc_${ARM}_pf" kvcached --offload 8 -- "${PF_ARGS[@]}" > /dev/null;;
esac
if ! VLLM_PORT=8200 "$KV/scripts/wait_ready.sh" "coloc_${ARM}_pf" 600 > "$L/${ARM}_pf_engine_start.txt" 2>&1; then echo "tenant engine failed"; tail -5 "$L/${ARM}_pf_engine_start.txt"; stop_own; exit 3; fi
sleep 3; U1=$(nv_used); TF=$(python -c "print(max(1.0, ($U1 - $U0) / 1024))"); echo "tenant engine fixed footprint ${TF} GiB (NVML ${U0} -> ${U1} MiB)" | tee "$L/${ARM}_pf_fixed.txt"
kvctl limit vllm_c 1G > /dev/null 2>&1
# --- VRAM scheduler (router for the 8B), tenant adapter (router for the 3B)
python "$C/controller.py" --listen 8001 --upstream http://localhost:$VLLM_PORT --ipc "$KV_IPC" --log "${S}_controller.csv" --tenant-fixed-gib "$TF" > "$L/${ARM}_controller.log" 2>&1 & CP=$!
sleep 3
python "$C/pf_tenant.py" --listen 8201 --upstream http://localhost:8200 --ipc vllm_c --fixed-gib "$TF" --log "${S}_pf.csv" $([[ ${PF_SLEEP:-0} == 1 ]] && echo --sleep) > "$L/${ARM}_pf_tenant.log" 2>&1 & TP=$!
sleep 2
python "$KV/scripts/sampler.py" --out "${S}_sampler.csv" --duration $((DUR + 200)) --ipc "$KV_IPC" --metrics-url "http://localhost:$VLLM_PORT/metrics" & SP=$!
# --- loads: the reference trace through the 8B's router; the tenant's prefill stream through its adapter
python "$KV/scripts/loadgen.py" --mode stream --prompt-tokens "$PF_PROMPT" --max-tokens 1 --rate "$PF_RATE" --duration $((DUR + 40)) --url http://localhost:8201 \
    --model "$PF_MODEL" --vocab 151936 --out "${S}_pfload.jsonl" > "$L/${ARM}_pfload.log" 2>&1 & PLP=$!
python "$KV/scripts/loadgen.py" --mode prefix --rate 1 --rps-schedule "$SCHED" --duration "$DUR" --url http://localhost:8001 \
    --n-sessions 32 --prefix-tokens 6144 --unique-tokens 128 --max-tokens 128 --skew zipf --out "${S}_load.jsonl" 2>&1 | tee "$L/${ARM}_load.log" | grep -v "^{" | tail -3
# --- teardown
kill $PLP 2>/dev/null; sleep 2; kill $TP $SP $CP 2>/dev/null; sleep 1
cp /dev/shm/coloc_xfer_inf.csv "${S}_xfer_inf.csv" 2>/dev/null; cp "${S}_load.jsonl.t0" "${S}_t0.txt" 2>/dev/null
stop_own; echo quit | nvidia-cuda-mps-control
python "$C/analyze_coloc.py" --arm "$ARM" --dir "$C/results" --schedule "$SCHED" 2>&1 | tail -8
python "$C/analyze_pf.py" --arm "$ARM" --dir "$C/results" --schedule "$SCHED" 2>&1 | tail -20
echo "DONE $ARM"
