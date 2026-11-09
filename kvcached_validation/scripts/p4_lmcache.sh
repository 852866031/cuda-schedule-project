#!/usr/bin/env bash
# P4 (overnight plan 2): kvcached + LMCache MP connector on GPU 1, fully isolated from the GPU-0
# experiments (own ports, own kvcached instance name, own venv, never scripts/stop.sh).
#   p4_lmcache.sh <mode: vmm|engine_driven> <tag> [--exportable] [--stress SECONDS] [--resize]
# Starts the LMCache server (:5557, http :8182) and vLLM 0.30 + patched kvcached (.venv-lmc) on
# GPU 1 (:8100, KVCACHED_IPC_NAME=vllm_b); if the engine comes up, runs the 8-prompt output check,
# then (optionally) the stress session load with/without a kvctl resize loop. Everything it
# started is killed by pid at the end. Logs: logs/p4_<tag>_*.log ; results: results/p4_<tag>_*.
set -uo pipefail
MODE=$1; TAG=$2; shift 2
EXPORTABLE=0; STRESS=0; RESIZE=0
while [[ $# -gt 0 ]]; do case $1 in
  --exportable) EXPORTABLE=1; shift;; --stress) STRESS=$2; shift 2;; --resize) RESIZE=1; shift;; *) echo "bad arg $1"; exit 2;; esac; done
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/.venv-lmc/bin/activate"
export MODEL=meta-llama/Meta-Llama-3-8B HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
PORT=8100; LMC_PORT=5557; LMC_HTTP=8182; IPC=vllm_b
L=$HERE/logs; R=$HERE/results; mkdir -p "$L" "$R"; N="p4_${TAG}"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n $p ]] && { pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; }; done; sleep 4
            for p in "${PIDS[@]:-}"; do [[ -n $p ]] && { pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; }; done; }
trap cleanup EXIT
# --- LMCache server
nohup lmcache server --host localhost --port $LMC_PORT --l1-size-gb 6 --l1-init-size-gb 2 --eviction-policy LRU --chunk-size 256 \
    --supported-transfer-mode auto --http-port $LMC_HTTP > "$L/${N}_lmcache.log" 2>&1 & PIDS+=($!)
sleep 6
# --- engine
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_LOGGING_LEVEL=INFO VLLM_USE_FLASHINFER_SAMPLER=0
export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_IPC_NAME=$IPC
[[ $EXPORTABLE == 1 ]] && export KVCACHED_EXPORTABLE=1
EXTRA="\"lmcache.mp.host\": \"localhost\", \"lmcache.mp.port\": $LMC_PORT"
case $MODE in
  vmm) EXTRA="$EXTRA, \"lmcache.mp.mp_transfer_mode\": \"lmcache_driven\", \"lmcache.mp.use_vmm_api\": true"; export KVCACHED_CONTIGUOUS_LAYOUT=false;;
  engine_driven) EXTRA="$EXTRA, \"lmcache.mp.mp_transfer_mode\": \"engine_driven\"";;
  *) echo "mode vmm|engine_driven"; exit 2;;
esac
ARGS=(serve "$MODEL" --port $PORT --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 8192 --seed 0 --no-enable-prefix-caching
      --kv-transfer-config "{\"kv_connector\": \"LMCacheMPConnector\", \"kv_connector_module_path\": \"lmcache.integration.vllm.lmcache_mp_connector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {$EXTRA}}")
{ echo "# $(date -Is) mode=$MODE exportable=$EXPORTABLE"; echo "# vllm ${ARGS[*]}"; env | grep -E "^(KVCACHED|ENABLE_KVCACHED|CUDA_VISIBLE)"; } > "$L/${N}_vllm.log"
nohup vllm "${ARGS[@]}" >> "$L/${N}_vllm.log" 2>&1 & EP=$!; PIDS+=($EP)
t0=$(date +%s)
while true; do
  curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1 && { echo "ready after $(( $(date +%s)-t0 )) s" | tee "$R/${N}_startup.txt"; break; }
  if ! kill -0 $EP 2>/dev/null; then echo "ENGINE DIED at startup" | tee "$R/${N}_startup.txt"
     grep -n -E "Error|error|Traceback|RuntimeError|ValueError|cuda|CUDA" "$L/${N}_vllm.log" | grep -v -i "FutureWarning\|pynvml" | tail -40 >> "$R/${N}_startup.txt"; exit 3; fi
  (( $(date +%s)-t0 > 600 )) && { echo "startup timeout" | tee "$R/${N}_startup.txt"; exit 3; }
  sleep 2
done
grep -n -i "lmcache\|connector\|transfer\|kvcached" "$L/${N}_vllm.log" | grep -v -i "deprecat" | cut -c1-240 | tail -30 > "$R/${N}_connector_lines.txt"
# --- output check: cold then warm
python "$HERE/scripts/loadgen.py" --mode check --n-check 8 --url "http://localhost:$PORT" --out "$R/${N}_check_cold.jsonl" > "$L/${N}_check_cold.out" 2>&1
sleep 2
python "$HERE/scripts/loadgen.py" --mode check --n-check 8 --url "http://localhost:$PORT" --out "$R/${N}_check_warm.jsonl" > "$L/${N}_check_warm.out" 2>&1
echo "checks done: $(wc -l < "$R/${N}_check_cold.jsonl") cold, $(wc -l < "$R/${N}_check_warm.jsonl") warm"
# --- stress: 6 sessions, 6000-token first turns, +512 per turn, think 2 s; optional 2G/12G resize loop every 15 s
if [[ $STRESS -gt 0 ]]; then
  if [[ $RESIZE == 1 ]]; then
    ( while true; do sleep 15; kvctl limit $IPC 2G > /dev/null 2>&1; sleep 15; kvctl limit $IPC 12G > /dev/null 2>&1; done ) & RP=$!; PIDS+=($RP)
  fi
  python "$HERE/scripts/loadgen.py" --mode session --sessions 6 --prompt-tokens 6000 --turn-tokens 512 --think 2 --duration "$STRESS" \
      --url "http://localhost:$PORT" --max-tokens 64 --out "$R/${N}_stress.jsonl" > "$L/${N}_stress.out" 2>&1
  tail -3 "$L/${N}_stress.out"
  [[ $RESIZE == 1 ]] && kill $RP 2>/dev/null
fi
kill -0 $EP 2>/dev/null && echo "engine alive at end" || echo "ENGINE DEAD at end"
grep -i -E "error|exception|traceback|illegal|fault|Xid|kvcached\]\[WARN" "$L/${N}_vllm.log" | grep -v -i "FutureWarning\|pynvml\|deprecat" | cut -c1-300 | tail -40 > "$R/${N}_engine_errors.txt"
grep -i -E "error|exception|traceback|illegal|fault" "$L/${N}_lmcache.log" | head -20 > "$R/${N}_lmcache_errors.txt"
echo "stored: $(grep -c -i Stored "$L/${N}_lmcache.log") retrieved: $(grep -c -i Retriev "$L/${N}_lmcache.log") engine error lines: $(wc -l < "$R/${N}_engine_errors.txt")"
echo "DONE $N"
