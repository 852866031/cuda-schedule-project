#!/bin/bash
# Medium-cohort sizing probe: one engine at a time under MPS, no workload. For each
# (max_num_seqs, max_num_batched_tokens) variant at the SAME util, record the engine's
# KV grant and GPU1 footprint -> how much non-KV overhead the knobs actually free.
#   probe_kv.sh <util> "<seqs>:<batched>" ...      (batched "" = vLLM default)
cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"
UTIL="$1"; shift
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mc_mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mc_mps_log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d; sleep 2
echo "util,max_num_seqs,max_batched,kv_gib,gpu1_used_mib,startup_s"
for v in "$@"; do
    seqs="${v%%:*}"; bat="${v#*:}"
    bash multi_stop.sh >/dev/null 2>&1
    # wait until GPU1 is really empty: a previous engine still releasing memory during
    # this one's profiling pass corrupts its KV sizing (seen once: 0.48 vs ~0.9 GiB)
    for i in $(seq 1 30); do
        [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 300 ] && break
        sleep 1
    done
    sleep 3
    t0=$(date +%s)
    MC_MODEL=Qwen/Qwen2.5-3B MC_UTIL="$UTIL" MC_MAX_NUM_SEQS="$seqs" MC_MAX_BATCHED="$bat" \
        bash multi_launch.sh model 0 0 > /dev/null 2>&1
    kv=$(grep -oE "Available KV cache memory: [0-9.]+" "$REPO/output/logs/mc_engine_0.log" | tail -1 | awk '{print $NF}')
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "$UTIL,$seqs,${bat:-default},${kv:-FAIL},$used,$(( $(date +%s) - t0 ))"
done
bash multi_stop.sh >/dev/null 2>&1
echo quit | nvidia-cuda-mps-control
