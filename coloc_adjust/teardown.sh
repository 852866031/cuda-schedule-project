#!/usr/bin/env bash
# Stop everything the prototype starts: tenant, controller/router, load generator, sampler,
# engine (via the kvcached_validation stop script), MPS daemon + server. Invoke as `bash teardown.sh`
# and keep these pattern strings out of the invoking command line (pkill -f matches any process
# whose command line contains them, including the shell that typed them).
KV=$(cd "$(dirname "$0")/../kvcached_validation" && pwd)
pkill -f "[c]oloc_adjust/ft_worker.py" 2>/dev/null
pkill -f "[c]oloc_adjust/controller.py" 2>/dev/null
pkill -f "[s]cripts/loadgen.py" 2>/dev/null
pkill -f "[s]cripts/sampler.py" 2>/dev/null
"$KV/scripts/stop.sh" > /dev/null 2>&1
sleep 2
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps/log
if pgrep -f "[n]vidia-cuda-mps-control" > /dev/null; then
  timeout 5 bash -c 'echo quit | nvidia-cuda-mps-control' > /dev/null 2>&1
  sleep 1
  pkill -f "[n]vidia-cuda-mps-server" 2>/dev/null; pkill -f "[n]vidia-cuda-mps-control" 2>/dev/null
fi
rm -rf /tmp/coloc_mps
rm -f /dev/shm/coloc_ft_budget /dev/shm/coloc_ft_state /dev/shm/coloc_hp_busy
sleep 1
echo "remaining: $(pgrep -af 'nvidia-cuda-mps|vllm serve|EngineCore|ft_worker|controller.py|loadgen|sampler.py' | grep -v pgrep | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
