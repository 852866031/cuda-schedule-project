#!/usr/bin/env bash
# Launch (or relaunch) the colocator replay server on this machine. Idempotent.
# Usage: bash serve.sh [port]     (default 8000; serves the repo's replays/ dir)
set -e
PORT="${1:-8000}"
cd "$(dirname "$0")"

# kill a previous instance of this server (matches our cmdline only)
pkill -f "python3 server\.py.*colocator" 2>/dev/null && sleep 0.3 || true
pkill -f "visualizer/server\.py" 2>/dev/null && sleep 0.3 || true

nohup python3 server.py "$PORT" > /tmp/colocator_replay_server.log 2>&1 &
sleep 0.5
if curl -sf "http://localhost:$PORT/api/runs" > /dev/null; then
  echo "colocator replay server up on localhost:$PORT (log: /tmp/colocator_replay_server.log)"
else
  echo "server failed to start — see /tmp/colocator_replay_server.log" >&2
  exit 1
fi
