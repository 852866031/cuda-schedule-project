#!/usr/bin/env bash
# Does a tenant variant run in every shape (TIERS index 0..6: micro-batch counts, activations and layers in host RAM)?
#   GPU=1 ./shape_check.sh "<ft_worker flags>" [shapes]     e.g. ./shape_check.sh "--head vocab --vocab-block 4096"
# Each shape runs alone for 14 s under MPS on a 32-SM green context; prints steps, tokens/s and loss, or the error.
# (The snapshot path holds "tenant_bytes" so that a vLLM run starting on the other card does not kill this tenant.)
set -uo pipefail
GPU=${GPU:-1}; C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation; FL=${1:-}; SHAPES=${2:-"0 1 2 3 4 5 6"}
UUID=$(nvidia-smi --query-gpu=uuid -i "$GPU" --format=csv,noheader)
export VENV_NAME=.venv-main; source "$KV/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=$UUID CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps$GPU/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps$GPU/log COLOC_XFER_DIR=/dev/shm/shape_check_xfer
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$COLOC_XFER_DIR" "$C/results_prof"
nvidia-cuda-mps-control -d; sleep 1; trap 'echo quit | nvidia-cuda-mps-control' EXIT
for s in $SHAPES; do
  out=$C/results_prof/shape_check_$s; rm -f "$out.csv"
  python "$C/ft_worker.py" --out "$out.csv" --fresh --fixed-budget-gib 20 --fixed-tier "$s" --duration ${DUR:-14} --green-sms 32 --snapshot /dev/shm/tenant_bytes_shape_snapshot.pt --snapshot-every 100000000 --state-file /dev/shm/tenant_bytes_shape_state $FL > "$out.log" 2>&1
  rc=$?; n=$(($(wc -l < "$out.csv" 2>/dev/null || echo 1) - 1))
  if (( n > 2 )); then echo "shape $s: $n steps, last: $(tail -1 "$out.csv" | awk -F, '{print $8" ms, "$9" tok/s, loss "$10}')"; else echo "shape $s: FAILED rc=$rc: $(grep -v Warning "$out.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"; fi
done
