#!/bin/bash
# KV-slot study: the 8B split-LMCache stack alone (prefill GPU0, decode GPU1, KV handed
# over through a DRAM store). Copied from inf_pf_prio/pfp_launch.sh `big` with the 3B
# neighbour, priority roles and MPS removed. Driven by kv_sweep.py:
#
#   kv_launch.sh guard                          mem_guard + heartbeat (once per launch)
#   kv_launch.sh big <decode_util> <prefill_util>
#                                               store :8300, prefill GPU0 :8100, decode
#                                               GPU1 :8200, router :8000
#
# Baseline engine flags are byte-for-byte those of pfp_launch.sh. Method knobs (E4/E5),
# all off by default:
#   KV_TRACE=1 KV_TRACE_PATH=<file>  decode engine only: record-only timeline (site/kv_trace.py)
#   DEC_APC=1                        decode engine with prefix caching ON
#   KV_DTYPE=fp8                     --kv-cache-dtype on BOTH engines (they must agree)
#   LMC_LAYERWISE=1                  LMCache use_layerwise: true (both engines)
#   LMC_ASYNC_LOAD=1                 LMCache enable_async_loading: true (decode engine)
#   LMC_LOOKUP_TIMEOUT_MS=<ms>       LMCache lookup_timeout_ms (decode engine; default 3000)
#   KV_COPS=1                        compiled lmcache.c_ops (from .venv-lmc) in both engines
#   VLLM_VENV=<venv dir>             engines' venv (default .venv-matched = vLLM 0.15.1)
#   KV_MP=1 [MP_L1_GB=32 MP_PORT=5655 MP_SERVER_VENV=.venv-lmc KV_MP_BUILTIN=0] LMCacheMPConnector (asynchronous load, D9): an LMCache
#                                    MP server (.venv-lmc, compiled ops, pinned L1 of MP_L1_GB,
#                                    LRU) replaces the TCP store and both engines' L1; :5555 is
#                                    DCGM's, hence 5655
#   ROUTER_BATCH=N [ROUTER_BATCH_TIMEOUT=s]  router holds decode legs until N are ready
#                                    (cohort-admission check; engines unchanged)
# OMP_NUM_THREADS is inherited from the driver.
set -u
cd "$(dirname "$0")"
HERE="$(pwd)"
REPO="$(cd ../.. && pwd)"
PY="$REPO/${VLLM_VENV:-.venv-matched}/bin/python"     # D11: VLLM_VENV=.venv-vllm018 for the MP grid
LOGS="$REPO/output/logs"
HB_DIR=/tmp/kvv_hb
PID_DIR=/tmp/kvv_pids
mkdir -p "$LOGS" "$HB_DIR" "$PID_DIR"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

CHUNK_SIZE="${CHUNK_SIZE:-256}"
CMD="${1:?usage: kv_launch.sh guard | big <decode_util> <prefill_util>}"

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
start_store() {   # port -> sets STORE_PORT. Uncapped (the 8B's 32 prefixes = 24 GiB).
    for attempt in 1 2 3 4 5 6 7 8; do
        STORE_PORT="$1"; [ "$attempt" -ge 3 ] && STORE_PORT=$(( $1 + attempt - 2 ))
        LMC_STORE_CAP_GB=0 "$PY" "$HERE/lmc_server_main.py" 127.0.0.1 "$STORE_PORT" cpu \
            > "$LOGS/kvv_store.log" 2>&1 &
        echo $! > "$PID_DIR/store.pid"
        sleep 2
        if kill -0 "$(cat "$PID_DIR/store.pid")" 2>/dev/null && ss -ltn "sport = :$STORE_PORT" | grep -q ":$STORE_PORT"; then
            [ "$STORE_PORT" != "$1" ] && echo "  store bound :$STORE_PORT instead of :$1" >&2
            return 0
        fi
        echo "  store attempt $attempt died -- retrying" >&2
        sleep 4
        [ "$attempt" = 8 ] && { echo "store :$1 died 8x -- see $LOGS/kvv_store.log" >&2; exit 1; }
    done
}
lmc_yaml() {   # path l1_gb port extra_lines
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
    [ -n "$4" ] && printf "%b" "$4" >> "$1"
    return 0
}
start_mp_server() {   # -> pid in store.pid; waits for the bind line
    CUDA_VISIBLE_DEVICES=0,1 "$REPO/${MP_SERVER_VENV:-.venv-lmc}/bin/python" -m lmcache.v1.multiprocess.server \
        --host 127.0.0.1 --port "$MP_PORT" --chunk-size "$CHUNK_SIZE" \
        --l1-size-gb "${MP_L1_GB:-32}" --eviction-policy LRU \
        > "$LOGS/kvv_mpserver.log" 2>&1 &
    echo $! > "$PID_DIR/store.pid"
    for i in $(seq 1 90); do
        grep -q "cache server is running" "$LOGS/kvv_mpserver.log" && return 0
        kill -0 "$(cat "$PID_DIR/store.pid")" 2>/dev/null \
            || { echo "MP server died -- see $LOGS/kvv_mpserver.log" >&2; exit 1; }
        sleep 2
    done
    echo "MP server not up after 180 s -- see $LOGS/kvv_mpserver.log" >&2; exit 1
}
register() {   # name port
    echo "$2 $(cat "$PID_DIR/$1.pid") $LOGS/kvv_$1.log" > "$HB_DIR/$1"
}
KV_CONN='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# ------------------------------------------------------------------------- guard
if [ "$CMD" = guard ]; then
    "$REPO/scripts/common/mem_guard.sh" "${MEM_GUARD_FLOOR_MB:-4000}" 1 > "$LOGS/kvv_mem_guard.log" 2>&1 &
    echo $! > "$PID_DIR/memguard.pid"
    ./kv_heartbeat.sh 3 > "$LOGS/kvv_heartbeat.log" 2>&1 &
    echo $! > "$PID_DIR/heartbeat.pid"
    echo "guard up"
    exit 0
fi

[ "$CMD" = big ] || { echo "unknown command '$CMD'" >&2; exit 1; }
DECODE_UTIL="${2:?decode util}"; PREFILL_UTIL="${3:?prefill util}"
MODEL="${BIG_MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
L1="${LMC_LOCAL_CPU_GB:-6}"
refuse_if_listening 8000 8100 8200 8300

# method knobs -> flags (empty for the baseline)
PF_X=(); DEC_X=(); LMC_PF=""; LMC_DEC=""
DEC_APC_FLAG="--no-enable-prefix-caching"
[ "${DEC_APC:-0}" = 1 ] && DEC_APC_FLAG="--enable-prefix-caching"
if [ -n "${KV_DTYPE:-}" ]; then PF_X+=(--kv-cache-dtype "$KV_DTYPE"); DEC_X+=(--kv-cache-dtype "$KV_DTYPE"); fi
if [ "${LMC_LAYERWISE:-0}" = 1 ]; then LMC_PF+="use_layerwise: true\n"; LMC_DEC+="use_layerwise: true\n"; fi
[ "${LMC_ASYNC_LOAD:-0}" = 1 ] && LMC_DEC+="enable_async_loading: true\n"
[ -n "${LMC_LOOKUP_TIMEOUT_MS:-}" ] && LMC_DEC+="lookup_timeout_ms: ${LMC_LOOKUP_TIMEOUT_MS}\n"
DEC_ENV=""; PF_ENV=""
[ "${KV_TRACE:-0}" = 1 ] && DEC_ENV="PYTHONPATH=$HERE/site KV_TRACE=1 KV_TRACE_PATH=${KV_TRACE_PATH:?KV_TRACE_PATH}"
if [ "${KV_COPS:-0}" = 1 ]; then       # compiled kernels in BOTH engines (store side saves with them too)
    DEC_ENV="${DEC_ENV:-PYTHONPATH=$HERE/site} KV_COPS=1"
    PF_ENV="PYTHONPATH=$HERE/site KV_COPS=1 KV_TRACE=0"   # the trace is the decode engine's only
fi
MP_PORT="${MP_PORT:-5655}"
if [ "${KV_MP:-0}" = 1 ]; then
    refuse_if_listening "$MP_PORT"
    # torch shares expandable-segment memory as fds that the receiver must take with
    # pidfd_getfd (ptrace rights; ptrace_scope=1 here) -> REGISTER_KV_CACHE fails. Without
    # expandable segments it uses cudaIpcGetMemHandle. Engines only; recorded in D9.
    unset PYTORCH_CUDA_ALLOC_CONF
    # LMCache's own connector (token-id keys), not vLLM 0.15.1's (block-hash keys): site/lmc_mp_conn.py;
    # with KV_MP_BUILTIN=1 (vLLM 0.18, whose built-in connector already keys by token ids) vLLM's own
    KV_CONN="{\"kv_connector\":\"LMCacheMP044Connector\",\"kv_connector_module_path\":\"lmc_mp_conn\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT}}"
    DEC_ENV="${DEC_ENV:-PYTHONPATH=$HERE/site}"; PF_ENV="PYTHONPATH=$HERE/site KV_TRACE=0"
    [ "${KV_MP_BUILTIN:-0}" = 1 ] && KV_CONN="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT}}"
fi
echo "methods: mp=${KV_MP:-0} cops=${KV_COPS:-0} decode $DEC_APC_FLAG ${DEC_X[*]:-} | prefill ${PF_X[*]:-} | lmc_pf [${LMC_PF}] lmc_dec [${LMC_DEC}] | trace ${KV_TRACE:-0}"

if [ "${KV_MP:-0}" = 1 ]; then
    start_mp_server; STORE_PORT=0
else
    start_store 8300
fi
lmc_yaml /tmp/kvv_lmc_prefill.yaml "$L1" "$STORE_PORT" "$LMC_PF"
lmc_yaml /tmp/kvv_lmc_decode.yaml "$L1" "$STORE_PORT" "$LMC_DEC"

CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/tmp/kvv_lmc_prefill.yaml env $PF_ENV \
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8100 --gpu-memory-utilization "$PREFILL_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --enable-prefix-caching \
    --no-enable-chunked-prefill --max-num-batched-tokens "${PREFILL_BATCHED_TOKENS:-32768}" \
    "${PF_X[@]}" --kv-transfer-config "$KV_CONN" \
    > "$LOGS/kvv_prefill.log" 2>&1 &
echo $! > "$PID_DIR/prefill.pid"

CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=/tmp/kvv_lmc_decode.yaml env $DEC_ENV \
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8200 --gpu-memory-utilization "$DECODE_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests $DEC_APC_FLAG \
    "${DEC_X[@]}" --kv-transfer-config "$KV_CONN" \
    > "$LOGS/kvv_decode.log" 2>&1 &
echo $! > "$PID_DIR/decode.pid"

wait_health 8100 "$LOGS/kvv_prefill.log"; register prefill 8100
wait_health 8200 "$LOGS/kvv_decode.log";  register decode 8200

"$REPO/.venv/bin/python" lmc_proxy.py --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200 \
    --max-inflight 999 --forward-first-token --port 8000 \
    --release-batch "${ROUTER_BATCH:-1}" --release-timeout "${ROUTER_BATCH_TIMEOUT:-2.0}" \
    > "$LOGS/kvv_proxy.log" 2>&1 &
echo $! > "$PID_DIR/proxy.pid"
sleep 3
echo "big up"
