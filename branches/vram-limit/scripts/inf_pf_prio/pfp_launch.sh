#!/bin/bash
# Prefill-on-the-decode-GPU colocation: the 8B split-LMCache stack (prefill GPU0, decode
# GPU1, KV offloaded to a DRAM store) + a Qwen2.5-3B whose PREFILL runs on GPU1, beside
# the 8B decode. Driven step by step by pfp_sweep.py:
#
#   pfp_launch.sh guard                       mem_guard + heartbeat (once per launch)
#   pfp_launch.sh big <decode_util> <prefill_util>
#                                             8B store :8300, prefill GPU0 :8100, decode
#                                             GPU1 :8200, router :8000 (flags = the split
#                                             study's; only the two utils are arguments)
#   pfp_launch.sh small <P1|P2|P3>            the 3B, in one of three placements:
#       P1  prefill only on GPU1 (:8111), no KV connector -- the KV is dropped
#       P2  prefill on GPU1, KV written to the 3B's own DRAM store (:8301); no decode
#       P3  prefill on GPU1 -> store -> decode on GPU0 (:8211), beside the 8B prefill;
#           own router :8001 (forward-first-token)
#
# Ordering rule (driver): `small` only after `big` is healthy, so each 8B engine sizes
# its KV grant against an empty GPU. The two 3B engines sit on different GPUs and start
# together. MPS: the driver exports CUDA_MPS_PIPE_DIRECTORY; every engine inherits it.
# OMP_NUM_THREADS is inherited from the driver (up to 4 engines on this box).
set -u
cd "$(dirname "$0")"
HERE="$(pwd)"
REPO="$(cd ../.. && pwd)"
PY="$REPO/.venv-matched/bin/python"
LOGS="$REPO/output/logs"
HB_DIR=/tmp/pfp_hb
PID_DIR=/tmp/pfp_pids
mkdir -p "$LOGS" "$HB_DIR" "$PID_DIR"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

CHUNK_SIZE="${CHUNK_SIZE:-256}"
CMD="${1:?usage: pfp_launch.sh guard | big <decode_util> <prefill_util> | small <P1|P2|P3>}"

wait_health() {   # port log
    echo -n "waiting for :$1 "
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "ready"; return 0; }
        sleep 2
        [ "$i" = 240 ] && { echo "TIMEOUT -- see $2"; exit 1; }
    done
}
refuse_if_listening() {
    for p in "$@"; do
        if ss -ltn "sport = :$p" | grep -q ":$p"; then
            echo "REFUSING: something already listens on :$p" >&2; exit 1
        fi
    done
}
start_store() {   # port name cap_gb  -> sets STORE_PORT
    # The 3B store's port has twice failed to bind for minutes with nothing listening on it
    # (cause unknown). From the 3rd attempt on, try the next ports (base+1 ... base+6).
    for attempt in 1 2 3 4 5 6 7 8; do
        STORE_PORT="$1"; [ "$attempt" -ge 3 ] && STORE_PORT=$(( $1 + attempt - 2 ))
        LMC_STORE_CAP_GB="$3" "$PY" "$HERE/lmc_server_main.py" 127.0.0.1 "$STORE_PORT" cpu \
            > "$LOGS/pfp_store_$2.log" 2>&1 &
        echo $! > "$PID_DIR/store_$2.pid"
        sleep 2
        if kill -0 "$(cat "$PID_DIR/store_$2.pid")" 2>/dev/null && ss -ltn "sport = :$STORE_PORT" | grep -q ":$STORE_PORT"; then
            [ "$STORE_PORT" != "$1" ] && echo "  store bound :$STORE_PORT instead of :$1" >&2
            return 0
        fi
        echo "  store attempt $attempt died -- retrying" >&2
        sleep 4
        [ "$attempt" = 8 ] && { echo "store :$1 died 8x -- see $LOGS/pfp_store_$2.log" >&2; exit 1; }
    done
}
lmc_yaml() {   # path l1_gb port
    cat > "$1" <<YAML
chunk_size: $CHUNK_SIZE
local_cpu: true
max_local_cpu_size: $2
local_disk: null
max_local_disk_size: 0
remote_url: "lm://127.0.0.1:$3"
remote_serde: "naive"
save_unfull_chunk: false
YAML
}
register() {   # name port
    echo "$2 $(cat "$PID_DIR/$1.pid") $LOGS/pfp_$1.log" > "$HB_DIR/$1"
}
KV_CONN='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# Priority knobs for the two engines that share GPU1 (8B decode, 3B prefill).
#   <X>_ROLE=plain|hp|be   hp: publish execute_model busy window (site/hp_signal.py)
#                          be: CUPTI-gated -- kernel launches wait while hp is busy
#                              (orion_gate/cupti_gate.so; forces --enforce-eager: the gate's
#                              event records abort CUDA-graph capture)
#   <X>_MPS_LOW=1          MPS client priority "below normal" (CUDA_MPS_CLIENT_PRIORITY=1)
#   <X>_EAGER=1            --enforce-eager without the gate (control for be)
role_env() {   # role mps_low -> env assignments for `env`
    local e="PYTHONPATH=$HERE/site"
    case "$1" in
        plain) ;;
        hp) e="$e COLOC_HP_SIGNAL=1" ;;
        be) [ -f "$HERE/orion_gate/cupti_gate.so" ] || { echo "missing cupti_gate.so" >&2; exit 1; }
            e="$e LD_PRELOAD=$HERE/orion_gate/cupti_gate.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6 COLOC_ROLE=be COLOC_K=${GATE_K:-8} COLOC_MAXPEND=${GATE_MAXPEND:-3}" ;;
        *) echo "unknown role '$1'" >&2; exit 1 ;;
    esac
    [ "$2" = 1 ] && e="$e CUDA_MPS_CLIENT_PRIORITY=1"
    echo "$e"
}

# ------------------------------------------------------------------------- guard
if [ "$CMD" = guard ]; then
    "$REPO/scripts/common/mem_guard.sh" "${MEM_GUARD_FLOOR_MB:-4000}" 1 > "$LOGS/pfp_mem_guard.log" 2>&1 &
    echo $! > "$PID_DIR/memguard.pid"
    ./pfp_heartbeat.sh 3 > "$LOGS/pfp_heartbeat.log" 2>&1 &
    echo $! > "$PID_DIR/heartbeat.pid"
    echo "guard up"
    exit 0
fi

# --------------------------------------------------------------------- 8B stack
# Engine flags are byte-for-byte those of inf_inf_coloc/coloc2_launch.sh (= the split
# study). Pinned host memory: 2 x LMC_LOCAL_CPU_GB L1 pools.
if [ "$CMD" = big ]; then
    DECODE_UTIL="${2:?decode util}"; PREFILL_UTIL="${3:?prefill util}"
    MODEL="${BIG_MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
    L1="${LMC_LOCAL_CPU_GB:-6}"
    refuse_if_listening 8000 8100 8200 8300
    start_store 8300 big 0
    lmc_yaml /tmp/pfp_lmc_big_prefill.yaml "$L1" "$STORE_PORT"
    lmc_yaml /tmp/pfp_lmc_big_decode.yaml "$L1" "$STORE_PORT"

    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/tmp/pfp_lmc_big_prefill.yaml \
    "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port 8100 --gpu-memory-utilization "$PREFILL_UTIL" \
        --max-model-len 8192 --seed 0 --disable-log-requests --enable-prefix-caching \
        --no-enable-chunked-prefill --max-num-batched-tokens "${PREFILL_BATCHED_TOKENS:-32768}" \
        --kv-transfer-config "$KV_CONN" \
        > "$LOGS/pfp_big_prefill.log" 2>&1 &
    echo $! > "$PID_DIR/big_prefill.pid"

    DEC_ROLE="${BIG_DEC_ROLE:-plain}"
    DEC_ENV=$(role_env "$DEC_ROLE" "${BIG_DEC_MPS_LOW:-0}") || exit 1
    DEC_EAGER=""; { [ "$DEC_ROLE" = be ] || [ -n "${BIG_DEC_EAGER:-}" ]; } && DEC_EAGER="--enforce-eager"
    echo "8B decode: role=$DEC_ROLE mps_low=${BIG_DEC_MPS_LOW:-0} ${DEC_EAGER}"
    CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=/tmp/pfp_lmc_big_decode.yaml env $DEC_ENV \
    "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port 8200 --gpu-memory-utilization "$DECODE_UTIL" \
        --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching $DEC_EAGER \
        --kv-transfer-config "$KV_CONN" \
        > "$LOGS/pfp_big_decode.log" 2>&1 &
    echo $! > "$PID_DIR/big_decode.pid"

    wait_health 8100 "$LOGS/pfp_big_prefill.log"; register big_prefill 8100
    wait_health 8200 "$LOGS/pfp_big_decode.log";  register big_decode 8200

    "$REPO/.venv/bin/python" lmc_proxy.py --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200 \
        --max-inflight 999 --forward-first-token --port 8000 > "$LOGS/pfp_big_proxy.log" 2>&1 &
    echo $! > "$PID_DIR/big_proxy.pid"
    sleep 3
    echo "big up"
    exit 0
fi

# ------------------------------------------------------------------------ the 3B
[ "$CMD" = small ] || { echo "unknown command '$CMD'" >&2; exit 1; }
CASE="${2:?P1|P2|P3}"
case "$CASE" in P1|P2|P3) ;; *) echo "unknown case '$CASE'" >&2; exit 1;; esac
MODEL="${SM_MODEL:-Qwen/Qwen2.5-3B}"
UTIL="${SM_UTIL:-0.22}"
SEQS="${SM_MAX_NUM_SEQS:-16}"
PF_BATCHED="${SM_PF_BATCHED:-8192}"
DEC_BATCHED="${SM_DEC_BATCHED:-1024}"
L1="${SM_L1_GB:-1}"
CAP="${SM_STORE_CAP_GB:-4}"
refuse_if_listening 8001 8111 8211 8301
rm -rf /dev/shm/pfp_saved

PF_KV=(); PF_ENV=""
if [ "$CASE" != P1 ]; then
    start_store 8301 small "$CAP"
    lmc_yaml /tmp/pfp_lmc_small.yaml "$L1" "$STORE_PORT"
    PF_KV=(--kv-transfer-config "$KV_CONN")
    PF_ENV="LMCACHE_CONFIG_FILE=/tmp/pfp_lmc_small.yaml"
    # SM_ASYNC_SAVE=1: return token #1 before the GPU->host KV copy (site/async_save.py).
    [ -n "${SM_ASYNC_SAVE:-}" ] && PF_ENV="$PF_ENV LMC_ASYNC_SAVE=1"
fi
SM_R="${SM_ROLE:-plain}"
PF_ENV="$PF_ENV $(role_env "$SM_R" "${SM_MPS_LOW:-0}")" || exit 1
# the 3B prefill runs eager by default: saves the CUDA-graph memory it needs to fit beside
# the 8B at b24, and a 2048-token prefill is above every graph capture size anyway
SM_EAGER_ARG=""; { [ "$SM_R" = be ] || [ "${SM_EAGER:-1}" = 1 ]; } && SM_EAGER_ARG="--enforce-eager"
# SM_QUANT=fp8: online fp8 weight quantization of the 3B (fits beside the 8B at b24)
[ -n "${SM_QUANT:-}" ] && SM_EAGER_ARG="$SM_EAGER_ARG --quantization $SM_QUANT"
echo "3B prefill: role=$SM_R mps_low=${SM_MPS_LOW:-0} ${SM_EAGER_ARG}"

# 3B prefill on GPU1 (the 8B's decode GPU). Whole-prompt prefill in one step, like the
# 8B's prefill engine. Identical flags in P1/P2/P3 except the KV connector.
CUDA_VISIBLE_DEVICES=1 env $PF_ENV \
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8111 --gpu-memory-utilization "$UTIL" \
    --max-model-len "${SM_MAX_LEN:-8192}" --seed 0 --disable-log-requests --enable-prefix-caching $SM_EAGER_ARG \
    --max-num-seqs "$SEQS" --no-enable-chunked-prefill --max-num-batched-tokens "$PF_BATCHED" \
    "${PF_KV[@]}" \
    > "$LOGS/pfp_small_prefill.log" 2>&1 &
echo $! > "$PID_DIR/small_prefill.pid"

# P3: 3B decode on GPU0 (the 8B's prefill GPU). Flags = the multi-model study's 3B engine.
if [ "$CASE" = P3 ]; then
    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/tmp/pfp_lmc_small.yaml \
    "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port 8211 --gpu-memory-utilization "$UTIL" \
        --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching \
        --max-num-seqs "$SEQS" --max-num-batched-tokens "$DEC_BATCHED" \
        --kv-transfer-config "$KV_CONN" \
        > "$LOGS/pfp_small_decode.log" 2>&1 &
    echo $! > "$PID_DIR/small_decode.pid"
fi

wait_health 8111 "$LOGS/pfp_small_prefill.log"; register small_prefill 8111
if [ "$CASE" = P3 ]; then
    wait_health 8211 "$LOGS/pfp_small_decode.log"; register small_decode 8211
    "$REPO/.venv/bin/python" lmc_proxy.py --prefill 127.0.0.1:8111 --decode 127.0.0.1:8211 \
        --max-inflight 999 --forward-first-token --port 8001 \
        ${SM_ASYNC_SAVE:+--wait-saved-dir /dev/shm/pfp_saved} > "$LOGS/pfp_small_proxy.log" 2>&1 &
    echo $! > "$PID_DIR/small_proxy.pid"
    sleep 3
fi
echo "small up ($CASE)"
