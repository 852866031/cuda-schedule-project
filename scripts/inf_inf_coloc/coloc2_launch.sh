#!/bin/bash
# Decode+decode colocation: the 8B split-LMCache stack + a Qwen2.5-0.5B tenant.
#
#   scripts/inf_inf_coloc/coloc2_launch.sh <scenario: A|B|C|solo> [decode_util]
#
# Scenarios (the 8B stack is identical in all non-solo scenarios; solo skips it):
#   A    whole Qwen (prefill+decode) on GPU1. One engine, prefix caching + an LMCache
#        DRAM offload tier (:8301) so W-offload spills to DRAM, never recomputes.
#   B    Qwen DECODE-ONLY on GPU1. Its prefix KV is pre-populated into the store by a
#        temporary GPU0 prefill (killed by the driver before measurement -- "prefill
#        from an invisible GPU"); during measurement the engine only retrieves KV from
#        DRAM and decodes, never prefills. prefix caching ON so a retrieved prefix stays
#        VRAM-resident and is reused, not re-fetched every request.
#   C    split Qwen: prefill GPU0 (:8101) + decode GPU1 (:8201) + own proxy (:8001).
#        Two full stacks -- memory-heavy; deferred on this box.
#   solo Qwen decode-only alone on GPU1 (no 8B) -- the tenant's own baseline for B.
#
# Ordering rule: the Qwen GPU1 engine starts only AFTER the 8B decode is healthy, so
# vLLM sizes the 8B KV grant against an empty GPU and the b26 baseline stays comparable.
set -u
cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"
PY="$REPO/.venv-matched/bin/python"
MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
LOGS="$REPO/output/logs"
mkdir -p "$LOGS"

SCENARIO="${1:?usage: coloc2_launch.sh <A|B|C|solo> [decode_util]}"
case "$SCENARIO" in A|B|C|solo) ;; *) echo "unknown scenario '$SCENARIO'" >&2; exit 1;; esac
DECODE_UTIL="${2:-0.8293}"
PREFILL_UTIL="${PREFILL_UTIL:-0.80}"
LMC_PORT="${LMC_PORT:-8300}"
MAX_INFLIGHT="${MAX_INFLIGHT:-999}"
LMC_LOCAL_CPU_GB="${LMC_LOCAL_CPU_GB:-6}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-0.5B}"
QWEN_UTIL="${QWEN_UTIL:-0.08}"
QWEN_MAX_INFLIGHT="${QWEN_MAX_INFLIGHT:-999}"
# Without this cap vLLM sizes its logits buffer + cudagraphs for 512+ seqs and leaves
# only 0.11 GiB for KV at util 0.08; 64 is ~8x the tenant's in-flight at 2 QPS and
# frees ~1 GiB for KV (measured 1.04 GiB grant).
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-64}"
QWEN_L1_GB="${QWEN_L1_GB:-1}"
QWEN_LMC_PORT="${QWEN_LMC_PORT:-8301}"
QWEN_STORE_MB="${QWEN_STORE_MB:-3500}"

# Which GPU0 Qwen prefill to run: none (A), temp=populate-then-killed (solo,B),
# persistent=serves prefill during measurement (C).
case "$SCENARIO" in
    A)         QWEN_GPU0=none ;;
    solo|B)    QWEN_GPU0=temp ;;
    C)         QWEN_GPU0=persistent ;;
esac

# --- RAM preflight -------------------------------------------------------------
# Overflow beyond physical RAM is SWAPPABLE store pages (cold prefixes), not pinned;
# the freeze hazard is pinned overcommit and the pinned total is fixed (8B 2x6 + Qwen
# 2x1). Swap traffic is recorded per run (pswpin/pswpout) so thrash is visible.
TOTAL_MB=$(free -m | awk '/^(Mem|内存)/ {print $2}')
AVAIL_MB=$(free -m | awk '/^(Mem|内存)/ {print $7}')
SWAP_ALLOW_MB="${SWAP_ALLOW_MB:-6000}"
QWEN_ENGINES=1; [ "$QWEN_GPU0" != none ] && QWEN_ENGINES=2
case "$SCENARIO" in
    solo) BASE_MB=10000 ;;
    *)    BASE_MB=$(( (2 * LMC_LOCAL_CPU_GB + 26) * 1024 + 14000 )) ;;
esac
QWEN_MB=$(( QWEN_L1_GB * 1024 + QWEN_STORE_MB + QWEN_ENGINES * 2500 ))
NEED_MB=$(( BASE_MB + QWEN_MB + 3000 ))
echo "RAM budget [$SCENARIO]: base ${BASE_MB} + qwen ${QWEN_MB} + 3000 = $(( NEED_MB / 1024 )) GiB of $(( TOTAL_MB / 1024 )) GiB (+${SWAP_ALLOW_MB} swap; ${AVAIL_MB} MB free now)"
if [ "$NEED_MB" -gt "$(( TOTAL_MB + SWAP_ALLOW_MB ))" ]; then
    echo "REFUSING TO LAUNCH: budget exceeds RAM + allowed swap." >&2
    exit 1
fi
[ "$NEED_MB" -gt "$TOTAL_MB" ] && echo "WARNING: budget exceeds physical RAM by $(( NEED_MB - TOTAL_MB )) MB -- swap-backed; watch pswpin/pswpout."

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

"$REPO/scripts/common/mem_guard.sh" "${MEM_GUARD_FLOOR_MB:-8000}" 1 > "$LOGS/mem_guard.log" 2>&1 &
echo $! > /tmp/disagg_memguard.pid

wait_port_clear() {   # port
    echo -n "waiting for :$1 to clear "
    for i in $(seq 1 20); do
        ss -ltn "sport = :$1" | grep -q ":$1" || { echo "clear"; return 0; }
        sleep 1
        [ "$i" = 20 ] && { echo "TIMEOUT: something still holds :$1" >&2; exit 1; }
    done
}
start_lmc_server() {   # port pidfile log
    for attempt in 1 2 3; do
        "$PY" "$(dirname "$0")/lmc_server_main.py" 127.0.0.1 "$1" cpu > "$3" 2>&1 &
        echo $! > "$2"
        sleep 2
        if kill -0 "$(cat "$2")" 2>/dev/null && ss -ltn "sport = :$1" | grep -q ":$1"; then
            return 0
        fi
        echo "  server attempt $attempt died -- retrying" >&2
        sleep 2
        [ "$attempt" = 3 ] && { echo "LMCache server on :$1 died 3x -- see $3" >&2; exit 1; }
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
wait_health() {   # port
    echo -n "waiting for :$1 "
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "ready"; return 0; }
        sleep 2
        [ "$i" = 240 ] && { echo "TIMEOUT -- see $LOGS/"; exit 1; }
    done
}

# --- 8B stack (byte-for-byte the flags of inf_ft_coloc/lmc_launch.sh) ----------
if [ "$SCENARIO" != solo ]; then
    wait_port_clear "$LMC_PORT"
    echo "starting 8B LMCache server (DRAM) on :$LMC_PORT..."
    start_lmc_server "$LMC_PORT" /tmp/disagg_lmcserver.pid "$LOGS/lmc_server.log"
    lmc_yaml /tmp/lmc_prefill.yaml "$LMC_LOCAL_CPU_GB" "$LMC_PORT"
    lmc_yaml /tmp/lmc_decode.yaml "$LMC_LOCAL_CPU_GB" "$LMC_PORT"

    echo "starting 8B prefill on GPU0 (util $PREFILL_UTIL)..."
    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/tmp/lmc_prefill.yaml \
    "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port 8100 --gpu-memory-utilization "$PREFILL_UTIL" \
        --max-model-len 8192 --seed 0 --disable-log-requests --enable-prefix-caching \
        --no-enable-chunked-prefill --max-num-batched-tokens "${PREFILL_BATCHED_TOKENS:-32768}" \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        > "$LOGS/disagg_prefill.log" 2>&1 &
    echo $! > /tmp/disagg_prefill.pid

    echo "starting 8B decode on GPU1 (util $DECODE_UTIL)..."
    CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=/tmp/lmc_decode.yaml \
    "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
        --port 8200 --gpu-memory-utilization "$DECODE_UTIL" \
        --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        > "$LOGS/disagg_decode.log" 2>&1 &
    echo $! > /tmp/disagg_decode.pid

    wait_health 8100
    wait_health 8200

    if ss -ltn "sport = :8000" | grep -q ":8000"; then
        echo "REFUSING TO START THE ROUTER: something already listens on :8000" >&2
        exit 1
    fi
    "$REPO/.venv/bin/python" lmc_proxy.py \
        --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200 \
        --max-inflight "$MAX_INFLIGHT" ${FORWARD_FIRST_TOKEN:+--forward-first-token} --port 8000 > "$LOGS/disagg_proxy.log" 2>&1 &
    echo $! > /tmp/disagg_proxy.pid
    sleep 3
    echo "proxy up on :8000"
fi

# --- Qwen tenant (strictly after the 8B decode has claimed GPU1) ---------------
for p in 8001 8101 8201 "$QWEN_LMC_PORT"; do
    if ss -ltn "sport = :$p" | grep -q ":$p"; then
        echo "REFUSING TO LAUNCH QWEN: something already listens on :$p" >&2
        exit 1
    fi
done

echo "starting Qwen LMCache server (DRAM, its own store) on :$QWEN_LMC_PORT..."
start_lmc_server "$QWEN_LMC_PORT" /tmp/qwen_lmcserver.pid "$LOGS/qwen_lmc_server.log"
lmc_yaml /tmp/lmc_qwen_prefill.yaml "$QWEN_L1_GB" "$QWEN_LMC_PORT"
lmc_yaml /tmp/lmc_qwen_decode.yaml "$QWEN_L1_GB" "$QWEN_LMC_PORT"

# GPU0 Qwen prefill (temp populate for solo/B, persistent split for C). setsid so
# qwen_prefill_kill.sh can take the whole engine tree with one PGID kill.
if [ "$QWEN_GPU0" != none ]; then
    echo "starting Qwen prefill on GPU0 (util $QWEN_UTIL, mode=$QWEN_GPU0)..."
    CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/tmp/lmc_qwen_prefill.yaml \
    setsid "$PY" -m vllm.entrypoints.cli.main serve "$QWEN_MODEL" \
        --port 8101 --gpu-memory-utilization "$QWEN_UTIL" \
        --max-model-len 8192 --seed 0 --disable-log-requests --enable-prefix-caching \
        --max-num-seqs "$QWEN_MAX_NUM_SEQS" \
        --no-enable-chunked-prefill --max-num-batched-tokens 8192 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        > "$LOGS/qwen_prefill.log" 2>&1 &
    echo $! > /tmp/qwen_prefill.pid
fi

# GPU1 Qwen engine. prefix caching ON: a prefix retrieved from the store stays
# VRAM-resident and is reused, NOT re-fetched every request (the bug that starved the
# 8B). Only VRAM-evicted prefixes (the offload overflow) reload from DRAM.
echo "starting Qwen decode on GPU1 (util $QWEN_UTIL)..."
CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=/tmp/lmc_qwen_decode.yaml \
"$PY" -m vllm.entrypoints.cli.main serve "$QWEN_MODEL" \
    --port 8201 --gpu-memory-utilization "$QWEN_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --enable-prefix-caching \
    --max-num-seqs "$QWEN_MAX_NUM_SEQS" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$LOGS/qwen_decode.log" 2>&1 &
echo $! > /tmp/qwen_decode.pid

[ "$QWEN_GPU0" != none ] && wait_health 8101
wait_health 8201

# C only: a forwarding proxy in front of the persistent split.
if [ "$SCENARIO" = C ]; then
    "$REPO/.venv/bin/python" lmc_proxy.py \
        --prefill 127.0.0.1:8101 --decode 127.0.0.1:8201 \
        --max-inflight "$QWEN_MAX_INFLIGHT" ${FORWARD_FIRST_TOKEN:+--forward-first-token} --port 8001 > "$LOGS/qwen_proxy.log" 2>&1 &
    echo $! > /tmp/qwen_proxy.pid
    sleep 3
fi

[ "$SCENARIO" = solo ] && echo "proxy up (none needed: solo)"
echo "qwen up ($SCENARIO)"

./coloc2_heartbeat.sh 3 > "$LOGS/disagg_heartbeat.log" 2>&1 &
echo $! > /tmp/disagg_heartbeat.pid
echo "heartbeat watching (cat /tmp/disagg_heartbeat_status)"
