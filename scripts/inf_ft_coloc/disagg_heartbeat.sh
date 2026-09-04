#!/bin/bash
# Heartbeat for the two-instance stack. An engine death here is silent: the EngineCore
# OOMs, the API server lingers (still answering /health for a while), the router keeps
# forwarding, and the load run reports garbage minutes later. This trips within seconds
# and takes the whole stack down so clients fail fast instead of measuring a corpse.
#
#   scripts/decode/disagg_heartbeat.sh [poll_seconds]
#
# Checks, per instance:
#   1. API server pid (from the pidfile) is alive
#   2. an EngineCore child exists on the right GPU's log
#   3. /health answers 200 within 5s
#   4. the log has not grown a fatal line (EngineDead / OOM / AssertionError)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS="$REPO/output/logs"
POLL="${1:-3}"
STATUS_FILE=/tmp/disagg_heartbeat_status

fatal() {
    echo "$(date '+%H:%M:%S') 💔 HEARTBEAT: $1" | tee -a "$STATUS_FILE" >&2
    echo "----- last log lines -----" >&2
    tail -5 "$2" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' >&2
    echo "killing the stack so clients fail fast" >&2
    bash "$REPO/scripts/inf_ft_coloc/disagg_stop.sh" >/dev/null 2>&1
    exit 1
}

echo "heartbeat: prefill(:8100) decode(:8200) every ${POLL}s" > "$STATUS_FILE"
declare -A last_fatal
while true; do
    for side in prefill decode; do
        [ "$side" = prefill ] && port=8100 || port=8200
        pidfile=/tmp/disagg_${side}.pid
        log="$LOGS/disagg_${side}.log"

        pid=$(cat "$pidfile" 2>/dev/null)
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
            || fatal "$side API server (pid ${pid:-?}) is gone" "$log"

        n=$(grep -cE "EngineDeadError|OutOfMemoryError|AssertionError" "$log" 2>/dev/null)
        if [ "${n:-0}" -gt "${last_fatal[$side]:-0}" ]; then
            last_fatal[$side]=$n
            fatal "$side log grew a fatal error (engine likely dead behind a live API)" "$log"
        fi

        curl -sf --max-time 5 "http://127.0.0.1:$port/health" >/dev/null \
            || fatal "$side /health not answering on :$port" "$log"
    done
    echo "$(date '+%H:%M:%S') ok" > "$STATUS_FILE"
    sleep "$POLL"
done
