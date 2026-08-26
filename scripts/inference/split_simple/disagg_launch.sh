#!/bin/bash
# Bring up a prefill instance (GPU0), a decode instance (GPU1), the router, and the
# memory watchdog.
#
#   scripts/decode/disagg_launch.sh [decode_util]
#
# decode_util defaults to 0.9568 (a 30 GiB budget). That is the swept knob in phase 1;
# the prefill side is deliberately fixed and generous so it never becomes the bottleneck.
#
# Env overrides: PREFILL_MEM_POOL_GB, DECODE_MEM_POOL_GB, KV_BUFFER_SIZE, P2P_TRACE.
set -u
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"
PY="$REPO/.venv-matched/bin/python"
MODEL="NousResearch/Meta-Llama-3-8B-Instruct"
LOGS="$REPO/output/logs"
mkdir -p "$LOGS"

DECODE_UTIL="${1:-0.9568}"
# Prefix caching on the producer only. The decode node receives KV over the wire and has no
# reuse to exploit, and a hit there would confound the batch-size measurement; the prefill
# node re-sees the same session prefixes and is the one place reuse is real.
PREFILL_PREFIX_ARG="--no-enable-prefix-caching"
[ "${PREFILL_PREFIX_CACHING:-0}" = "1" ] && PREFILL_PREFIX_ARG="--enable-prefix-caching"
MAX_INFLIGHT="${MAX_INFLIGHT:-28}"
# The prefill node's budget is fixed and generous, but "generous" means generous in
# throughput, not in cache it never touches. It needs ~0.856 GiB of KV per in-flight
# prompt -- 4 prompts is ~3.4 GiB -- so most of a 28 GiB budget was idle cache while
# PyTorch had no allocator slack: at util 0.8930 with a 32k token budget it overshot its
# 27.99 GiB target by 1.6 GiB (28.15 allocated + 1.46 reserved-unallocated) and died on a
# 748 MiB MLP intermediate mid-run. Lower util hands that margin back.
PREFILL_UTIL="${PREFILL_UTIL:-0.80}"          # ~25 GiB, fixed; never swept
PREFILL_KV_PORT=21001
DECODE_KV_PORT=22001

# ---------------------------------------------------------------- pinned host memory
# P2pNcclConnector pins mem_pool_size_gb of HOST memory per instance, default 32. Two
# instances therefore ask for 64 GiB of pinned, unswappable memory -- more than this box
# has (60 GiB). That is not an OOM kill but a hard freeze: the kernel cannot reclaim
# pinned pages, so it starves everything including the display server and the OOM killer.
# It rebooted the machine.
#
# TensorMemoryPool is a buddy allocator and rounds its size UP to a power of two
# (`1 << (size - 1).bit_length()`), so asking for 24 GiB pins 32. Only powers of two are
# honest here; the preflight below reports what is actually pinned, not what was asked.
#
# The two sides do NOT need the same pool. The spill-to-host path lives in
# listen_for_requests under cmd == "PUT", i.e. on the RECEIVING side, so with
# send_type=PUT_ASYNC only the decode instance buffers anything; the prefill instance's
# pool is dead allocation. (With send_type=GET the burden flips to the producer, which
# holds tensors in send_store until they are pulled.)
#
# Sizing: an entry lives in the consumer's store from arrival until the request
# *finishes* -- recv_tensor reads it, only get_finished pops it -- so the pool must hold
# every in-flight request at the decode node, not just the queued ones. Measured against
# a real transfer at ISL 6144: each layer ships [2, 427, 16, 8, 128] bf16 = 26.75 MiB,
# which the buddy allocator rounds to 32 MiB, so 32 layers = **exactly 1.00 GiB of pool
# per in-flight request**. A 32 GiB pool therefore holds exactly 32 -- with zero slack,
# and running it full risks ValueError("Insufficient memory") inside the listener thread,
# which kills the thread and wedges the decode instance. Keep client concurrency a few
# below pool_gb.
PREFILL_MEM_POOL_GB="${PREFILL_MEM_POOL_GB:-1}"
DECODE_MEM_POOL_GB="${DECODE_MEM_POOL_GB:-16}"

# Optional DRAM tier behind the prefill node's VRAM prefix cache: on a session miss the
# KV is retrieved over PCIe (~70 ms for a 6144-token prefix) instead of recomputed
# (~530 ms). This is the colocated study's offload arm, transplanted. It CANNOT be done
# with --kv-offloading-size: that flag overwrites kv_transfer_config and would silently
# drop the P2P connector. It has to be MultiConnector, [OffloadingConnector, P2pNccl] --
# load from the first match (the DRAM tier; the P2P producer never claims matches), save
# to all (offload AND ship). The buffer is pinned (cpu_gpu.py pin_memory), so it counts
# against the same budget as the pools below. 0 disables.
PREFILL_OFFLOAD_GB="${PREFILL_OFFLOAD_GB:-0}"

# GPU-side buffer for arrived KV before it spills to the pinned pool. This is allocated
# on the decode card OUTSIDE vLLM's gpu-memory-utilization budget, so it is VRAM the
# sweep does not account for -- keep it small so the decode GPU holds only the swept
# budget and essentially every transfer lands in host memory. At 1e8 roughly 96% of
# transfers spill, measured.
KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-1e8}"

pow2_gb() {   # GiB requested -> GiB actually pinned, matching TensorMemoryPool
    python3 -c "import sys;g=int(sys.argv[1]);print((1<<((g*1024**3)-1).bit_length())//1024**3)" "$1"
}
PREFILL_PINNED_GB=$(pow2_gb "$PREFILL_MEM_POOL_GB")
DECODE_PINNED_GB=$(pow2_gb "$DECODE_MEM_POOL_GB")
PINNED_MB=$(( (PREFILL_PINNED_GB + DECODE_PINNED_GB + PREFILL_OFFLOAD_GB) * 1024 ))

# The freeze was two 32 GiB pools -- 64 GiB of unswappable memory on a 60 GiB box, which
# is unsurvivable by arithmetic. One large pool is a different proposition, so the check
# is a budget rather than a blanket fraction: what gets pinned, plus what the two engines
# and the desktop need in ordinary pageable memory, plus headroom, must fit.
# RESERVE is measured: with 17 GiB pinned, `free` reported 29.8 GiB used and 17.6 GiB
# shared, i.e. ~12 GiB of non-pinned working set (engines ~7, OS and display ~5).
RESERVE_MB="${RESERVE_MB:-14000}"
HEADROOM_MB="${HEADROOM_MB:-6000}"
TOTAL_MB=$(free -m | awk '/^(Mem|内存)/ {print $2}')
AVAIL_MB=$(free -m | awk '/^(Mem|内存)/ {print $7}')
echo "pinned pools: prefill ${PREFILL_MEM_POOL_GB}->${PREFILL_PINNED_GB} GiB + decode ${DECODE_MEM_POOL_GB}->${DECODE_PINNED_GB} GiB + offload ${PREFILL_OFFLOAD_GB} GiB = $(( PINNED_MB / 1024 )) GiB pinned"
echo "  budget: ${PINNED_MB} pinned + ${RESERVE_MB} engines/OS + ${HEADROOM_MB} headroom = $(( PINNED_MB + RESERVE_MB + HEADROOM_MB )) MB of ${TOTAL_MB} MB (${AVAIL_MB} MB free now)"
if [ $(( PINNED_MB + RESERVE_MB + HEADROOM_MB )) -gt "$TOTAL_MB" ]; then
    echo "REFUSING TO LAUNCH: does not fit in ${TOTAL_MB} MB of RAM." >&2
    echo "  Pinned memory cannot be swapped or reclaimed; overcommitting it freezes the box." >&2
    exit 1
fi
if [ $(( PINNED_MB + HEADROOM_MB )) -gt "$AVAIL_MB" ]; then
    echo "REFUSING TO LAUNCH: only ${AVAIL_MB} MB available right now, need" >&2
    echo "  ${PINNED_MB} MB to pin plus ${HEADROOM_MB} MB headroom. Something else is using RAM." >&2
    exit 1
fi

# ---------------------------------------------------------------- launch
# vllm._C needs a newer libstdc++ than conda ships (see README).
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# The prefill node allocates large, short-lived activation buffers whose size varies with
# how many whole prompts land in a step; that fragments the caching allocator, and the
# 1.46 GiB of reserved-but-unallocated memory in the crash above was exactly that.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Puts a deadline on the connector's otherwise unbounded, silent consumer wait; see
# p2p_patch/sitecustomize.py. Reaches the spawned EngineCore, which a monkeypatch in the
# parent would not.
export PYTHONPATH="$PWD/p2p_patch${PYTHONPATH:+:$PYTHONPATH}"

# P2pNcclEngine binds its ZMQ ROUTER to get_ip():kv_port -- that one interface only, never
# loopback -- and the request ids the router mints must name that same address. Getting it
# wrong wedges the prefill engine inside ncclCommInitRank with no timeout and no log line.
# Ask vLLM for the address rather than assuming it.
KV_HOST="$("$PY" -c 'from vllm.utils.network_utils import get_ip; print(get_ip())')"
echo "kv host (vLLM get_ip): $KV_HOST"

"$REPO/scripts/common/mem_guard.sh" "${MEM_GUARD_FLOOR_MB:-8000}" 1 > "$LOGS/mem_guard.log" 2>&1 &
echo $! > /tmp/disagg_memguard.pid
echo "mem_guard watching (floor ${MEM_GUARD_FLOOR_MB:-8000} MB, 1s poll)"

kv_cfg() {   # role, kv_port, mem_pool_gb, http_port
    cat <<JSON | tr -d '\n '
{"kv_connector":"P2pNcclConnector","kv_role":"$1","kv_port":$2,
 "kv_buffer_size":$KV_BUFFER_SIZE,
 "kv_connector_extra_config":{"send_type":"PUT_ASYNC","mem_pool_size_gb":$3,
                              "http_port":"$4","proxy_ip":"","proxy_port":""}}
JSON
}

# Same P2P producer wrapped in MultiConnector with a DRAM offload tier in front.
prefill_multi_cfg() {   # kv_port, mem_pool_gb, http_port, offload_gb
    local OFFLOAD_BYTES=$(( $4 * 1024 * 1024 * 1024 ))
    cat <<JSON | tr -d '\n '
{"kv_connector":"MultiConnector","kv_role":"kv_producer",
 "kv_connector_extra_config":{"connectors":[
   {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use":$OFFLOAD_BYTES}},
   {"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":$1,
    "kv_buffer_size":$KV_BUFFER_SIZE,
    "kv_connector_extra_config":{"send_type":"PUT_ASYNC","mem_pool_size_gb":$2,
                                 "http_port":"$3","proxy_ip":"","proxy_port":""}}]}}
JSON
}
PREFILL_KV_CFG="$(kv_cfg kv_producer $PREFILL_KV_PORT $PREFILL_MEM_POOL_GB 8100)"
if [ "$PREFILL_OFFLOAD_GB" -gt 0 ]; then
    PREFILL_KV_CFG="$(prefill_multi_cfg $PREFILL_KV_PORT $PREFILL_MEM_POOL_GB 8100 $PREFILL_OFFLOAD_GB)"
    echo "prefill DRAM offload tier: ${PREFILL_OFFLOAD_GB} GiB via MultiConnector"
fi

# Chunked prefill is OFF on the producer, and not for performance: P2pNcclConnector's
# build_connector_meta asserts `new_block_ids is not None` on the chunked path, which is
# None whenever a continuing chunk needs no new blocks. That is an unconditional
# AssertionError inside scheduler.schedule(), i.e. a dead EngineCore -- and it only shows
# up under concurrency, because a lone 6.8k-token prompt fits in one chunk.
# Without chunking, a prompt is only scheduled if it fits whole, so the token budget sets
# prompts-per-step outright: 16384/6847 = 2. Measured at that setting the producer ran 86%
# busy with 29 of 32 sessions queued on it and the decode node starved at batch 8.85 --
# GPU0 was the bottleneck, which is exactly the condition PLAN_DECODE §7 says invalidates
# the decode curve. 32768 buys 4 prompts per step. The producer can afford the activation
# memory: it was using 15% of its KV allocation.
echo "starting prefill on GPU0 (util $PREFILL_UTIL, kv port $PREFILL_KV_PORT)..."
CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8100 --gpu-memory-utilization "$PREFILL_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests $PREFILL_PREFIX_ARG \
    --no-enable-chunked-prefill --max-num-batched-tokens "${PREFILL_BATCHED_TOKENS:-32768}" \
    --kv-transfer-config "$PREFILL_KV_CFG" \
    > "$LOGS/disagg_prefill.log" 2>&1 &
echo $! > /tmp/disagg_prefill.pid

echo "starting decode on GPU1 (util $DECODE_UTIL, kv port $DECODE_KV_PORT)..."
CUDA_VISIBLE_DEVICES=1 "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --port 8200 --gpu-memory-utilization "$DECODE_UTIL" \
    --max-model-len 8192 --seed 0 --disable-log-requests --no-enable-prefix-caching \
    --kv-transfer-config "$(kv_cfg kv_consumer $DECODE_KV_PORT $DECODE_MEM_POOL_GB 8200)" \
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

# Preflight the KV path itself. /health only proves the HTTP servers are up; the ZMQ
# sockets are a separate listener, and a request that cannot reach them does not fail --
# it hangs both engines.
for kvp in $PREFILL_KV_PORT $DECODE_KV_PORT; do
    if ! ss -ltn "sport = :$kvp" | grep -q "$KV_HOST:$kvp"; then
        echo "REFUSING TO START THE ROUTER: nothing listening on $KV_HOST:$kvp." >&2
        echo "  The engines bind their KV sockets to vLLM's get_ip(); if that moved," >&2
        echo "  every request would hang in ncclCommInitRank instead of erroring." >&2
        ss -ltn | grep -E "2[12]001" >&2
        exit 1
    fi
done
echo "kv sockets listening on $KV_HOST:{$PREFILL_KV_PORT,$DECODE_KV_PORT}"

"$REPO/.venv/bin/python" disagg_p2p_proxy.py \
    --prefill 127.0.0.1:8100 --prefill-kv-port $PREFILL_KV_PORT \
    --decode  127.0.0.1:8200 --decode-kv-port $DECODE_KV_PORT \
    --kv-host "$KV_HOST" --max-inflight "$MAX_INFLIGHT" \
    --port 8000 > "$LOGS/disagg_proxy.log" 2>&1 &
echo $! > /tmp/disagg_proxy.pid
sleep 3
echo "proxy up on :8000"

"$REPO/scripts/decode/disagg_heartbeat.sh" 3 > "$LOGS/disagg_heartbeat.log" 2>&1 &
echo $! > /tmp/disagg_heartbeat.pid
echo "heartbeat watching both instances (cat /tmp/disagg_heartbeat_status)"
