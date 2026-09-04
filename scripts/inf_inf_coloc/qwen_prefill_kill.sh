#!/bin/bash
# Scenario C: kill ONLY the temporary Qwen prefill engine (GPU0) after the store is
# populated. PGID-targeted -- the engine was launched under setsid, so its pidfile pid
# is also its process-group id and one group kill takes the API server + EngineCore.
# Deliberately contains NO pkill: everything else on the box must keep running.
#
# Order matters: remove the pidfile FIRST so the heartbeat (pidfile-gated) stops
# watching :8101 before the health check could ever fail.
set -u
PIDFILE=/tmp/qwen_prefill.pid
[ -f "$PIDFILE" ] || { echo "no $PIDFILE -- nothing to kill"; exit 0; }
PGID=$(cat "$PIDFILE")
rm -f "$PIDFILE"
kill -9 -- "-$PGID" 2>/dev/null
for i in $(seq 1 30); do
    pgrep -g "$PGID" >/dev/null 2>&1 || break
    sleep 1
    [ "$i" = 30 ] && { echo "WARNING: pgid $PGID still alive after 30s" >&2; exit 1; }
done
echo "qwen prefill (pgid $PGID) gone; GPU0 now:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
