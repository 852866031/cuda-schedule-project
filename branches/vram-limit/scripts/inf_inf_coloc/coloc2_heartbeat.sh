#!/bin/bash
# Heartbeat for the dual-stack experiment. An engine death is silent: the EngineCore
# OOMs, the API server lingers answering /health, and the load run reports garbage
# minutes later. This trips within seconds and takes the WHOLE experiment down.
#
#   scripts/inf_inf_coloc/coloc2_heartbeat.sh [poll_seconds]
#
# Checks are pidfile-gated: a side without a pidfile is simply not watched. That is
# what makes scenario C's planned prefill kill safe -- qwen_prefill_kill.sh removes
# /tmp/qwen_prefill.pid BEFORE killing, so the next poll skips that side.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOGS="$REPO/output/logs"
POLL="${1:-3}"
STATUS_FILE=/tmp/disagg_heartbeat_status

fatal() {
    echo "$(date '+%H:%M:%S') 💔 HEARTBEAT: $1" | tee -a "$STATUS_FILE" >&2
    echo "----- last log lines -----" >&2
    tail -5 "$2" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' >&2
    echo "killing the experiment so clients fail fast" >&2
    bash "$REPO/scripts/inf_inf_coloc/coloc2_stop.sh" >/dev/null 2>&1
    exit 1
}

echo "heartbeat: watching all pidfile-present engines every ${POLL}s" > "$STATUS_FILE"
declare -A last_fatal
while true; do
    for side in prefill:8100:disagg decode:8200:disagg prefill:8101:qwen decode:8201:qwen; do
        role="${side%%:*}"; rest="${side#*:}"; port="${rest%%:*}"; ns="${rest#*:}"
        pidfile="/tmp/${ns}_${role}.pid"
        log="$LOGS/${ns}_${role}.log"
        [ -f "$pidfile" ] || continue

        pid=$(cat "$pidfile" 2>/dev/null)
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
            || fatal "$ns $role API server (pid ${pid:-?}) is gone" "$log"

        n=$(grep -cE "EngineDeadError|OutOfMemoryError|AssertionError" "$log" 2>/dev/null)
        if [ "${n:-0}" -gt "${last_fatal[$ns$role]:-0}" ]; then
            last_fatal[$ns$role]=$n
            fatal "$ns $role log grew a fatal error (engine likely dead behind a live API)" "$log"
        fi

        curl -sf --max-time 5 "http://127.0.0.1:$port/health" >/dev/null \
            || fatal "$ns $role /health not answering on :$port" "$log"
    done
    echo "$(date '+%H:%M:%S') ok" > "$STATUS_FILE"
    sleep "$POLL"
done
