#!/usr/bin/env bash
# Stop engine/server processes started by serve.sh / lmcache_server.sh (all tags).
source "$(dirname "$0")/env.sh"
for f in "$LOGS"/*.pid; do
  [[ -f $f ]] || continue
  pid=$(cat "$f"); if kill -0 "$pid" 2>/dev/null; then echo "killing $(basename "$f") pid $pid"; kill "$pid"; fi
done
sleep 3
pkill -f "vllm serve" 2>/dev/null; pkill -f "lmcache server" 2>/dev/null; pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 2; rm -f "$LOGS"/*.pid
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
