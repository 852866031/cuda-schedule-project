#!/bin/bash
# LMCache-based prefill/decode split: everything KV lives in DRAM, shared.
#
#   scripts/inference/split_lmcache/lmc_launch.sh [decode_util]
#
# Topology:
#   GPU0 prefill  ──store──▶  LMCache server (DRAM, :8300)  ◀──retrieve── GPU1 decode
#
# One content-addressed cache replaces the split_simple arm's three private pools
# (decode transfer pool + prefill DRAM tier + producer pool): a session's prefix is
# stored once ever, both instances read it, and no GPU-to-GPU transfer path exists at
# all. NO DISK: local_disk is null and the server runs storage "cpu" -- everything DRAM.
#
# The cache is keyed by token-chunk hashes, so the two legs need no request-id
# coupling and none of split_simple's p2p_patch is loaded.
set -u
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"
PY="$REPO/.venv-matched/bin/python"
MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
LOGS="$REPO/output/logs"
mkdir -p "$LOGS"

DECODE_UTIL="${1:-0.9568}"
PREFILL_UTIL="${PREFILL_UTIL:-0.80}"
LMC_PORT="${LMC_PORT:-8300}"
MAX_INFLIGHT="${MAX_INFLIGHT:-14}"
# Per-instance L1 in local CPU RAM (pinned by lmcache for DMA speed) + the shared
# server, which holds the working set in ordinary swappable RAM. 32 sessions x 6144
# tokens x 0.125 MiB = ~24 GiB at the server when fully warm.
LMC_LOCAL_CPU_GB="${LMC_LOCAL_CPU_GB:-6}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"

TOTAL_MB=$(free -m | awk '/^(Mem|内存)/ {print $2}')
AVAIL_MB=$(free -m | awk '/^(Mem|内存)/ {print $7}')
NEED_MB=$(( (2 * LMC_LOCAL_CPU_GB + 26) * 1024 + 14000 + 6000 ))
echo "RAM budget: 2x${LMC_LOCAL_CPU_GB} GiB pinned L1 + ~26 GiB server (warm) + 14 engines/OS + 6 headroom = $(( NEED_MB / 1024 )) GiB of $(( TOTAL_MB / 1024 )) GiB (${AVAIL_MB} MB free now)"
if [ "$NEED_MB" -gt "$TOTAL_MB" ]; then
    echo "REFUSING TO LAUNCH: budget exceeds RAM." >&2
    exit 1
fi

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Chunk keys are python builtin hashes; without a fixed seed each process randomizes
# them and the two engines cannot agree on what is cached -- the decode node then
# silently recomputes and re-stores everything. Mandatory for cross-process sharing.
export PYTHONHASHSEED=0

# The L1 is a staging pool between the GPU and the TCP server, and the server drains at
# ~1 GB/s while a burst of prefills stores at ~1.5 GB/s -- an undersized pool fills and
# the engine dies on "Failed to allocate memory block". 8 GiB absorbs the bursts; the
# steady state (stores only on session misses) is under the drain rate. Warm the cache
# gently: the reference driver should be run with --warmup-qps 0.5 on this arm.

"$REPO/scripts/common/mem_guard.sh" "${MEM_GUARD_FLOOR_MB:-8000}" 1 > "$LOGS/mem_guard.log" 2>&1 &
echo $! > /tmp/disagg_memguard.pid

# Back-to-back sweep configs tear a server down and start the next within seconds; the
# old socket can linger and the fresh bind then dies with EADDRINUSE. Wait for the port
# to actually clear, and verify the new server is LISTENING, not merely alive.
echo -n "waiting for :$LMC_PORT to clear "
for i in $(seq 1 20); do
    ss -ltn "sport = :$LMC_PORT" | grep -q ":$LMC_PORT" || { echo "clear"; break; }
    sleep 1
    [ "$i" = 20 ] && { echo "TIMEOUT: something still holds :$LMC_PORT" >&2; exit 1; }
done

echo "starting LMCache server (DRAM) on :$LMC_PORT..."
for attempt in 1 2 3; do
    "$PY" "$(dirname "$0")/lmc_server_main.py" 127.0.0.1 "$LMC_PORT" cpu > "$LOGS/lmc_server.log" 2>&1 &
    echo $! > /tmp/disagg_lmcserver.pid
    sleep 2
    if kill -0 "$(cat /tmp/disagg_lmcserver.pid)" 2>/dev/null && \
       ss -ltn "sport = :$LMC_PORT" | grep -q ":$LMC_PORT"; then
        break
    fi
    echo "  server attempt $attempt died -- retrying" >&2
    sleep 2
    [ "$attempt" = 3 ] && { echo "LMCache server died 3x -- see $LOGS/lmc_server.log" >&2; exit 1; }
done

# Per-instance LMCache config. save_unfull_chunk false: only whole 256-token chunks are
# stored, so the 6144-token session prefix is exactly 24 chunks and per-request suffixes
# never pollute the cache. local_disk null: DRAM only, deliberately.
lmc_yaml() {   # path
    cat > "$1" <<YAML
chunk_size: $CHUNK_SIZE
local_cpu: true
max_local_cpu_size: $LMC_LOCAL_CPU_GB
local_disk: null
max_local_disk_size: 0
remote_url: "lm://127.0.0.1:$LMC_PORT"
remote_serde: "naive"
save_unfull_chunk: false
YAML
}
lmc_yaml /tmp/lmc_prefill.yaml
lmc_yaml /tmp/lmc_decode.yaml

echo "starting prefill on GPU0 (util $PREFILL_UTIL)..."
CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=/tmp/lmc_prefill.yaml \
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8100 --gpu-memory-utilization "$PREFILL_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --enable-prefix-caching \
    --no-enable-chunked-prefill --max-num-batched-tokens "${PREFILL_BATCHED_TOKENS:-32768}" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$LOGS/disagg_prefill.log" 2>&1 &
echo $! > /tmp/disagg_prefill.pid

echo "starting decode on GPU1 (util $DECODE_UTIL)..."
CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=/tmp/lmc_decode.yaml \
"$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8200 --gpu-memory-utilization "$DECODE_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$LOGS/disagg_decode.log" 2>&1 &
echo $! > /tmp/disagg_decode.pid

for p in 8100 8200; do
    echo -n "waiting for :$p "
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { echo "ready"; break; }
        sleep 2
        [ "$i" = 240 ] && { echo "TIMEOUT -- see $LOGS/"; exit 1; }
    done
done

if ss -ltn "sport = :8000" | grep -q ":8000"; then
    echo "REFUSING TO START THE ROUTER: something already listens on :8000" >&2
    ss -ltnp 2>/dev/null | grep 8000 >&2
    exit 1
fi
"$REPO/.venv/bin/python" lmc_proxy.py \
    --prefill 127.0.0.1:8100 --decode 127.0.0.1:8200 \
    --max-inflight "$MAX_INFLIGHT" ${FORWARD_FIRST_TOKEN:+--forward-first-token} --port 8000 > "$LOGS/disagg_proxy.log" 2>&1 &
echo $! > /tmp/disagg_proxy.pid
sleep 3
echo "proxy up on :8000"

"$REPO/scripts/inference/split_simple/disagg_heartbeat.sh" 3 > "$LOGS/disagg_heartbeat.log" 2>&1 &
echo $! > /tmp/disagg_heartbeat.pid
echo "heartbeat watching (cat /tmp/disagg_heartbeat_status)"
