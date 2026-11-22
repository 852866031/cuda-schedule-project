#!/usr/bin/env bash
# Night of Nov 21/22, GPU 1, tenant alone: the same 1,500 training steps (same seed, same data, usual shape) with the usual tenant
# and with the new tenant (blocks compiled, output layer over vocabulary blocks with fused kernels); the loss of every step is logged.
set -uo pipefail
GPU=1; C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
UUID=$(nvidia-smi --query-gpu=uuid -i "$GPU" --format=csv,noheader)
export VENV_NAME=.venv-main; source "$KV/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=$UUID CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps$GPU/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps$GPU/log COLOC_XFER_DIR=/dev/shm/traincmp_xfer
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$COLOC_XFER_DIR"
nvidia-cuda-mps-control -d; sleep 1; trap 'echo quit | nvidia-cuda-mps-control' EXIT
run() { local tag=$1; shift; rm -f "$C/results_prof/traincmp_$tag.csv"
  python "$C/ft_worker.py" --out "$C/results_prof/traincmp_$tag.csv" --fresh --seed 0 --fixed-budget-gib 20 --fixed-tier 2 --steps 1500 --green-sms 32 \
    --snapshot /dev/shm/tenant_bytes_traincmp.pt --snapshot-every 100000000 --state-file /dev/shm/tenant_bytes_traincmp_state "$@" > "$C/results_prof/traincmp_$tag.log" 2>&1; }
run usual
run new --compile-blocks --no-slow-shapes --head vocab --vocab-block 4096 --head-fused
run usual2          # the usual tenant once more: how far two runs of the same code drift apart
echo "$(date +%H:%M:%S) traincmp done" >> "$C/results_prof/gpu1_night.log"
