#!/usr/bin/env bash
# Launch one vLLM engine, plain or with kvcached, optionally with the LMCache MP connector.
#   serve.sh <tag> <plain|kvcached> [options] [-- extra vllm args]
# options:
#   --lmcache MODE      attach LMCacheMPConnector; MODE = lmcache_driven | engine_driven | vmm
#   --apc               keep vLLM prefix caching on (default: off, so every prefix must come
#                       from LMCache and kvcached's occupied pages are only running requests)
#   --kv-bytes N        plain only: static --kv-cache-memory-bytes (default 12G)
#   --gpu I             CUDA device (default 0)
# Log: $LOGS/vllm_<tag>.log ; pid: $LOGS/vllm_<tag>.pid
set -euo pipefail
source "$(dirname "$0")/env.sh"
TAG=$1; MODE=$2; shift 2
LMC=""; APC=0; KVB=$((12*1024*1024*1024)); GPU=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --lmcache) LMC=$2; shift 2;;
    --apc) APC=1; shift;;
    --kv-bytes) KVB=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --) shift; break;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_V2_MODEL_RUNNER=1        # kvcached requires Model Runner V2 on vLLM >= 0.29; use it in every run
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer JIT sampler refuses this GPU (see DECISIONS D3); unrelated to kvcached
ARGS=(serve "$MODEL" --port "$VLLM_PORT" --dtype bfloat16 --max-model-len 8192
      --max-num-seqs 32 --max-num-batched-tokens 8192 --seed 0)
[[ $APC -eq 0 ]] && ARGS+=(--no-enable-prefix-caching)
if [[ $MODE == plain ]]; then
  ARGS+=(--kv-cache-memory-bytes "$KVB")
elif [[ $MODE == kvcached ]]; then
  export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 KVCACHED_IPC_NAME=$KV_IPC
  # kvcached manages memory itself; README says no --gpu-memory-utilization needed.
else
  echo "mode must be plain|kvcached"; exit 2
fi
if [[ -n $LMC ]]; then
  EXTRA="\"lmcache.mp.host\": \"localhost\", \"lmcache.mp.port\": $LMC_PORT"
  case $LMC in
    lmcache_driven|engine_driven) EXTRA="$EXTRA, \"lmcache.mp.mp_transfer_mode\": \"$LMC\"";;
    vmm) EXTRA="$EXTRA, \"lmcache.mp.mp_transfer_mode\": \"lmcache_driven\", \"lmcache.mp.use_vmm_api\": true";;
    *) echo "bad --lmcache mode"; exit 2;;
  esac
  ARGS+=(--kv-transfer-config "{\"kv_connector\": \"LMCacheMPConnector\", \"kv_connector_module_path\": \"lmcache.integration.vllm.lmcache_mp_connector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {$EXTRA}}")
fi
ARGS+=("$@")
LOG="$LOGS/vllm_${TAG}.log"
echo "# $(date -Is) MODE=$MODE LMC=$LMC APC=$APC GPU=$GPU" > "$LOG"
echo "# vllm ${ARGS[*]}" >> "$LOG"
env | grep -E "^(ENABLE_KVCACHED|KVCACHED_|VLLM_USE_V2|PYTHONHASHSEED|OMP_NUM|CUDA_VISIBLE)" >> "$LOG"
nohup vllm "${ARGS[@]}" >> "$LOG" 2>&1 &
echo $! > "$LOGS/vllm_${TAG}.pid"
echo "started pid $(cat "$LOGS/vllm_${TAG}.pid") log $LOG"
