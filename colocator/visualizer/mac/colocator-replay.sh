#!/usr/bin/env bash
# ── Runs on your MAC ─────────────────────────────────────────────────────────
# Starts the colocator replay server on the remote GPU box, tunnels it, and
# opens the browser. Same pattern as gpu-interfere's mac/gpu-replay.sh.
#
# One-time setup on the Mac:
#   1. copy this file over:
#        scp <remote>:~/Documents/Projects/colocation/colocator/visualizer/mac/colocator-replay.sh ~/bin/
#   2. edit REMOTE below to your ssh target
#   3. chmod +x ~/bin/colocator-replay.sh
# Then just run:  colocator-replay.sh
set -e

REMOTE="workstation-remote"     # <-- EDIT: as you'd type `ssh <this>`
REMOTE_DIR="~/Documents/Projects/colocation/colocator/visualizer"
PORT=8000

# 1. make sure the server is running on the remote
ssh "$REMOTE" "bash $REMOTE_DIR/serve.sh $PORT"

# 2. open the tunnel unless this port already answers locally
if ! curl -sf -o /dev/null --max-time 1 "http://localhost:$PORT/api/runs"; then
  ssh -f -N -L "$PORT:localhost:$PORT" "$REMOTE"
  echo "tunnel opened: localhost:$PORT -> $REMOTE:$PORT"
fi

# 3. wait for the page through the tunnel, then open the browser
for _ in $(seq 1 20); do
  curl -sf -o /dev/null --max-time 1 "http://localhost:$PORT/api/runs" && break
  sleep 0.3
done
open "http://localhost:$PORT"
