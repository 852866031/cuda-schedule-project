#!/usr/bin/env bash
# Block until the vLLM server answers /health or the engine process dies / timeout.
source "$(dirname "$0")/env.sh"
TAG=$1; TIMEOUT=${2:-900}
PID=$(cat "$LOGS/vllm_${TAG}.pid"); t0=$(date +%s)
while true; do
  if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then echo "ready after $(( $(date +%s)-t0 )) s"; exit 0; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "engine died"; tail -40 "$LOGS/vllm_${TAG}.log"; exit 1; fi
  if (( $(date +%s)-t0 > TIMEOUT )); then echo "timeout"; exit 1; fi
  sleep 2
done
