#!/bin/bash
# Phase 0: bring up a prefill instance (GPU0), a decode instance (GPU1), and the router.
#
#   scripts/disagg_launch.sh [decode_util]
#
# decode_util defaults to 0.9568 (a 30 GiB budget). That is the swept knob in phase 1;
# the prefill side is deliberately fixed and generous so it never becomes the bottleneck.
set -u
cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"
PY="$REPO/.venv-matched/bin/python"
MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
LOGS="$REPO/output/logs"
mkdir -p "$LOGS"

DECODE_UTIL="${1:-0.9568}"
PREFILL_UTIL=0.8930          # 28 GiB, fixed
PREFILL_KV_PORT=21001
DECODE_KV_PORT=22001

# P2pNcclConnector pins mem_pool_size_gb of HOST memory per instance, default 32. Two instances
# therefore ask for 64 GiB of pinned, unswappable memory -- more than this box has (60 GiB).
# That is not an OOM kill but a hard freeze: the kernel cannot reclaim pinned pages, so it
# starves everything including the display server and the OOM killer. It rebooted the machine.
#
# The two sides do NOT need the same pool. The spill-to-host path lives in listen_for_requests
# under cmd == "PUT", i.e. on the RECEIVING side, so with send_type=PUT_ASYNC only the decode
# instance buffers anything; the prefill instance's pool is dead allocation. (With send_type=GET
# the burden flips to the producer, which holds tensors in send_store until they are pulled.)
#
# Decode-side sizing: it holds KV that has arrived but has not started decoding, so
#   pool >= (queued requests) x 0.8 GiB.
# Keeping client concurrency at (decode batch + 4) bounds that at ~3.2 GiB.
PREFILL_MEM_POOL_GB=1
DECODE_MEM_POOL_GB=24

# Preflight: refuse to launch if the two pools would take an unsafe share of RAM.
TOTAL_MB=$(free -m | awk '/^(Mem|内存)/ {print $2}')
NEED_MB=$(( (PREFILL_MEM_POOL_GB + DECODE_MEM_POOL_GB) * 1024 ))
if [ "$NEED_MB" -gt $(( TOTAL_MB / 2 )) ]; then
    echo "REFUSING TO LAUNCH: pinned pools total ${NEED_MB} MB," >&2
    echo "  which is more than half of this machine's ${TOTAL_MB} MB of RAM." >&2
    echo "  Pinned memory cannot be swapped or reclaimed; overcommitting it freezes the box." >&2
    exit 1
fi
echo "pinned pools: prefill ${PREFILL_MEM_POOL_GB} GiB + decode ${DECODE_MEM_POOL_GB} GiB = $(( NEED_MB / 1024 )) GiB of ${TOTAL_MB} MB RAM"

# vllm._C needs a newer libstdc++ than conda ships (see README).
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "starting prefill on GPU0 (util $PREFILL_UTIL, kv port $PREFILL_KV_PORT)..."
CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8100 --gpu-memory-utilization "$PREFILL_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":$PREFILL_KV_PORT,\"kv_connector_extra_config\":{\"send_type\":\"PUT_ASYNC\",\"mem_pool_size_gb\":$PREFILL_MEM_POOL_GB,\"http_port\":\"8100\",\"proxy_ip\":\"\",\"proxy_port\":\"\"}}" \
    > "$LOGS/disagg_prefill.log" 2>&1 &
echo $! > /tmp/disagg_prefill.pid

echo "starting decode on GPU1 (util $DECODE_UTIL, kv port $DECODE_KV_PORT)..."
CUDA_VISIBLE_DEVICES=1 "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8200 --gpu-memory-utilization "$DECODE_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching \
    --kv-transfer-config "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":$DECODE_KV_PORT,\"kv_connector_extra_config\":{\"send_type\":\"PUT_ASYNC\",\"mem_pool_size_gb\":$DECODE_MEM_POOL_GB,\"http_port\":\"8200\",\"proxy_ip\":\"\",\"proxy_port\":\"\"}}" \
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

"$REPO/.venv/bin/python" disagg_p2p_proxy.py \
    --prefill 127.0.0.1:8100 --prefill-kv-port $PREFILL_KV_PORT \
    --decode 127.0.0.1:8200 --decode-kv-port $DECODE_KV_PORT \
    --port 8000 > "$LOGS/disagg_proxy.log" 2>&1 &
echo $! > /tmp/disagg_proxy.pid
sleep 3
echo "proxy up on :8000"
