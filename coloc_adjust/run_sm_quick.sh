#!/bin/bash
# Quick evidence run: a plain vLLM engine (no kvcached, no host-RAM KV tier, no router, no tenant) whose decode steps are
# confined to N SMs (green context from the first step; "full" = all 170 SMs), with a small closed-loop decode load.
#   bash run_sm_quick.sh <gpu> <port> "full 96 64 32"      results: results_sm/q_gpu<gpu>_<N>.json
#   EAGER=1: the engine runs with --enforce-eager (no CUDA graphs); results in results_sm/eager_gpu<gpu>_<N>.json
G=$1; PORT=$2; LIST=$3
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation; mkdir -p "$C/results_sm" "$C/logs_sm"
export VENV_NAME=.venv-main VLLM_PORT=$PORT; source "$KV/scripts/env.sh"
PFX=q; [[ ${EAGER:-0} == 1 ]] && PFX=eager
[[ -n ${OUTPFX:-} ]] && PFX=$OUTPFX   # another name for the result files, e.g. for repeats
for n in $LIST; do
  TAG=smq_gpu${G}; export PYTHONPATH=$C/hp_patch
  if [[ $n == full ]]; then unset COLOC_ENGINE_SMS; else export COLOC_ENGINE_SMS=$n; fi
  if [[ ${EAGER:-0} == 1 ]]; then "$KV/scripts/serve.sh" "$TAG" plain --gpu "$G" -- --enforce-eager > /dev/null; else "$KV/scripts/serve.sh" "$TAG" plain --gpu "$G" > /dev/null; fi
  if "$KV/scripts/wait_ready.sh" "$TAG" 600 > "$C/logs_sm/${TAG}_${n}_start.txt" 2>&1; then
    unset PYTHONPATH
    python "$C/sm_quick_client.py" --url "http://localhost:$PORT" --out "$C/results_sm/${PFX}_gpu${G}_${n}.json" > "$C/logs_sm/${TAG}_${n}_client.txt" 2>&1
  fi
  g=$(grep -h "engine steps on a green context" "$KV/logs/vllm_${TAG}.log" | tail -1)
  pid=$(cat "$KV/logs/vllm_${TAG}.pid"); pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 5; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
  cp "$KV/logs/vllm_${TAG}.log" "$C/logs_sm/${TAG}_${PFX}_${n}_engine.log"
  echo "$(date +%T) gpu $G done $n: $(tail -1 "$C/logs_sm/${TAG}_${n}_client.txt" 2>/dev/null) | ${g:-no green context line}"; sleep 3
done
echo "QUICK DONE gpu $G"
