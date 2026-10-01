#!/bin/bash
# N homogeneous models colocated on GPU1. Driven step-by-step by multi_sweep.py:
#
#   multi_launch.sh guard                 mem_guard + heartbeat (once per launch set)
#   multi_launch.sh model <i> <temp_pf>   model i: its own LMCache store (:8600+i), its
#                                         GPU1 engine (:8400+i) and, if temp_pf=1, a
#                                         TEMPORARY GPU0 prefill (:8500+i) that the driver
#                                         uses to populate the store and then kills
#                                         (multi_prefill_kill.sh) -- the decode-only config.
#
# Models are brought up ONE AT A TIME (the driver waits for "model <i> up" before i+1),
# so no engine's memory-profiling pass races another's on GPU1 and every engine's KV
# grant is sized the same way. The temp GPU0 prefill starts alongside its GPU1 engine
# (different GPU, no profiling race) to halve startup time.
#
# Env (set by the driver): MC_MODEL, MC_UTIL, MC_MAX_NUM_SEQS, MC_L1_GB, MC_MAX_LEN,
#   MC_ROLE=plain|hp|be   hp: publish execute_model busy window (orion_gate/hp_patch)
#                         be: gated by cupti_gate.so (forces --enforce-eager: the CUPTI
#                             gate's event record aborts CUDA-graph capture)
#   MC_EAGER=1            eager without the gate (control)
#   MC_NO_LMC=1           no store, no KV connector (prefill-only: unique prompts,
#                         nothing to reuse; a store would only grow)
#   MC_GATE_K / MC_GATE_MAXPEND
# MPS: the driver exports CUDA_MPS_PIPE_DIRECTORY when an MPS arm is running; every
# engine launched here inherits it and becomes an MPS client.
set -u
cd "$(dirname "$0")"
HERE="$(pwd)"
REPO="$(cd ../.. && pwd)"
PY="$REPO/.venv-matched/bin/python"
LOGS="$REPO/output/logs"
HB_DIR=/tmp/multi_hb
PID_DIR=/tmp/multi_pids
mkdir -p "$LOGS" "$HB_DIR" "$PID_DIR"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

CMD="${1:?usage: multi_launch.sh guard | model <i> <temp_pf 0|1>}"

if [ "$CMD" = guard ]; then
    "$REPO/scripts/common/mem_guard.sh" "${MEM_GUARD_FLOOR_MB:-6000}" 1 > "$LOGS/mc_mem_guard.log" 2>&1 &
    echo $! > "$PID_DIR/memguard.pid"
    ./multi_heartbeat.sh 3 > "$LOGS/mc_heartbeat.log" 2>&1 &
    echo $! > "$PID_DIR/heartbeat.pid"
    echo "guard up"
    exit 0
fi

[ "$CMD" = model ] || { echo "unknown command '$CMD'" >&2; exit 1; }
I="${2:?model index}"
TEMP_PF="${3:-0}"
MODEL="${MC_MODEL:?MC_MODEL}"
UTIL="${MC_UTIL:?MC_UTIL}"
MAX_NUM_SEQS="${MC_MAX_NUM_SEQS:-64}"
L1_GB="${MC_L1_GB:-1}"
MAX_LEN="${MC_MAX_LEN:-8192}"
ROLE="${MC_ROLE:-plain}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
EPORT=$((8400 + I)); PPORT=$((8500 + I)); SPORT=$((8600 + I))

NO_LMC="${MC_NO_LMC:-}"
for p in $EPORT $SPORT; do
    if ss -ltn "sport = :$p" | grep -q ":$p"; then
        echo "REFUSING: something already listens on :$p" >&2; exit 1
    fi
done

wait_health() {   # port log
    echo -n "waiting for :$1 "
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "ready"; return 0; }
        sleep 2
        [ "$i" = 240 ] && { echo "TIMEOUT -- see $2"; exit 1; }
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

# --- the model's own DRAM store ----------------------------------------------------
KV_ARGS=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
[ -n "$NO_LMC" ] && KV_ARGS=()
for attempt in 1 2 3; do
    [ -n "$NO_LMC" ] && break
    "$PY" "$HERE/lmc_server_main.py" 127.0.0.1 "$SPORT" cpu > "$LOGS/mc_store_$I.log" 2>&1 &
    echo $! > "$PID_DIR/store_$I.pid"
    sleep 2
    if kill -0 "$(cat "$PID_DIR/store_$I.pid")" 2>/dev/null && ss -ltn "sport = :$SPORT" | grep -q ":$SPORT"; then
        break
    fi
    echo "  store attempt $attempt died -- retrying" >&2
    sleep 2
    [ "$attempt" = 3 ] && { echo "store :$SPORT died 3x -- see $LOGS/mc_store_$I.log" >&2; exit 1; }
done
lmc_yaml "/tmp/mc_lmc_$I.yaml" "$L1_GB" "$SPORT"
[ -n "$NO_LMC" ] && rm -f "$PID_DIR/store_$I.pid"

# --- role-specific env for the GPU1 engine -----------------------------------------
GATE_DIR="$HERE/orion_gate"
ROLE_ENV=""; EAGER_ARG=""
case "$ROLE" in
    plain) ;;
    hp) ROLE_ENV="PYTHONPATH=$GATE_DIR/hp_patch COLOC_HP_SIGNAL=1" ;;
    be) [ -f "$GATE_DIR/cupti_gate.so" ] || { echo "missing cupti_gate.so" >&2; exit 1; }
        ROLE_ENV="LD_PRELOAD=$GATE_DIR/cupti_gate.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6 COLOC_ROLE=be COLOC_K=${MC_GATE_K:-8} COLOC_MAXPEND=${MC_GATE_MAXPEND:-3}"
        EAGER_ARG="--enforce-eager" ;;
    *) echo "unknown MC_ROLE '$ROLE'" >&2; exit 1 ;;
esac
[ -n "${MC_EAGER:-}" ] && EAGER_ARG="--enforce-eager"

# --- temporary GPU0 prefill (decode-only config). setsid => pid is its PGID, so
# multi_prefill_kill.sh takes the API server + EngineCore with one group kill.
if [ "$TEMP_PF" = 1 ]; then
    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE="/tmp/mc_lmc_$I.yaml" \
    setsid "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port "$PPORT" --gpu-memory-utilization "$UTIL" \
        --max-model-len "$MAX_LEN" --seed 0 --disable-log-requests --enable-prefix-caching \
        --max-num-seqs "$MAX_NUM_SEQS" --enforce-eager \
        --no-enable-chunked-prefill --max-num-batched-tokens "$MAX_LEN" \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        > "$LOGS/mc_prefill_$I.log" 2>&1 &
    echo $! > "$PID_DIR/prefill_$I.pid"
fi

# --- GPU1 engine: prefix caching ON (a retrieved/prefilled prefix stays VRAM-resident;
# only evicted prefixes -- the offload overflow -- reload from the store).
CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE="/tmp/mc_lmc_$I.yaml" env $ROLE_ENV \
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port "$EPORT" --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAX_LEN" --seed 0 --disable-log-requests --enable-prefix-caching $EAGER_ARG \
    --max-num-seqs "$MAX_NUM_SEQS" "${KV_ARGS[@]}" \
    > "$LOGS/mc_engine_$I.log" 2>&1 &
echo $! > "$PID_DIR/engine_$I.pid"

wait_health "$EPORT" "$LOGS/mc_engine_$I.log"
echo "$EPORT $(cat "$PID_DIR/engine_$I.pid") $LOGS/mc_engine_$I.log" > "$HB_DIR/engine_$I"
if [ "$TEMP_PF" = 1 ]; then
    wait_health "$PPORT" "$LOGS/mc_prefill_$I.log"
    echo "$PPORT $(cat "$PID_DIR/prefill_$I.pid") $LOGS/mc_prefill_$I.log" > "$HB_DIR/prefill_$I"
fi
echo "model $I up (role=$ROLE${EAGER_ARG:+ eager})"
