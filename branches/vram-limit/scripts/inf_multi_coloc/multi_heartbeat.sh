#!/bin/bash
# Heartbeat for the N-model experiment. An engine death is silent (EngineCore OOMs, the
# API server lingers answering /health), so this trips within seconds and tears the
# WHOLE experiment down. Registry-driven: every healthy engine has a file
# /tmp/multi_hb/<name> = "port pid log"; removing the file (multi_prefill_kill.sh does
# so BEFORE killing) stops the watch -- that is what makes the planned prefill kill safe.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HB_DIR=/tmp/multi_hb
POLL="${1:-3}"
STATUS_FILE=/tmp/multi_heartbeat_status

fatal() {
    echo "$(date '+%H:%M:%S') HEARTBEAT: $1" | tee -a "$STATUS_FILE" >&2
    tail -5 "$2" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' >&2
    echo "killing the experiment so clients fail fast" >&2
    bash "$REPO/scripts/inf_multi_coloc/multi_stop.sh" >/dev/null 2>&1
    exit 1
}

echo "heartbeat: watching $HB_DIR every ${POLL}s" > "$STATUS_FILE"
declare -A last_fatal
while true; do
    for f in "$HB_DIR"/*; do
        [ -f "$f" ] || continue
        read -r port pid log < "$f" 2>/dev/null || continue
        name=$(basename "$f")
        [ -f "$f" ] || continue          # removed between glob and read: planned kill
        kill -0 "$pid" 2>/dev/null || { [ -f "$f" ] && fatal "$name (pid $pid) is gone" "$log"; }
        n=$(grep -cE "EngineDeadError|OutOfMemoryError|AssertionError" "$log" 2>/dev/null)
        if [ "${n:-0}" -gt "${last_fatal[$name]:-0}" ]; then
            last_fatal[$name]=$n
            fatal "$name log grew a fatal error (engine likely dead behind a live API)" "$log"
        fi
        curl -sf --max-time 5 "http://127.0.0.1:$port/health" >/dev/null \
            || { [ -f "$f" ] && fatal "$name /health not answering on :$port" "$log"; }
    done
    echo "$(date '+%H:%M:%S') ok" > "$STATUS_FILE"
    sleep "$POLL"
done
