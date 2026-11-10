#!/usr/bin/env bash
# Co-run knee probe: MPS daemon on one GPU (default GPU 1, selected by UUID so the daemon and both
# clients agree), then knee_corun.py (decode step + ft_worker tenant), or another probe: SCRIPT=bw_aggr.py.   run_knee_corun.sh [args]
set -uo pipefail
GPU=${GPU:-1}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
PL=$(nvidia-smi --query-gpu=power.limit -i "$GPU" --format=csv,noheader,nounits | cut -d. -f1)
if [[ -z $PL || $PL -gt 500 ]]; then echo "GPU $GPU power limit is ${PL:-unknown} W; requires <= 500 W"; exit 9; fi
UUID=$(nvidia-smi --query-gpu=uuid -i "$GPU" --format=csv,noheader)
export VENV_NAME=.venv-main
source "$KV/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=$UUID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps$GPU/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps$GPU/log   # short path: Unix-socket length limit
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d; sleep 1
trap 'echo quit | nvidia-cuda-mps-control' EXIT
cd "$C" && python "${SCRIPT:-knee_corun.py}" "$@"
