# Shared environment for every script in this directory. Source it.
HERE_ENV=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE_ENV/.venv/bin/activate"
export MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B}
export HF_HUB_OFFLINE=1
export PYTHONHASHSEED=0            # reproducible LMCache token hashing
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_PORT=${VLLM_PORT:-8000}
export KV_IPC=${KV_IPC:-vllm_a}    # kvcached instance name (KVCACHED_IPC_NAME)
export LMC_PORT=${LMC_PORT:-5556}   # 5555 is taken by an unrelated listener on this machine
export LOGS="$HERE_ENV/logs"
export RESULTS="$HERE_ENV/results"
mkdir -p "$LOGS" "$RESULTS"
