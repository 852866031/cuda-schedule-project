#!/bin/bash
# Decode-only config: kill ONLY model i's temporary GPU0 prefill once its store is
# populated. Heartbeat registry entry is removed FIRST so the watchdog never sees the
# planned death. PGID-targeted (launched under setsid); contains NO pkill.
set -u
I="${1:?model index}"
PIDFILE=/tmp/multi_pids/prefill_$I.pid
rm -f "/tmp/multi_hb/prefill_$I"
[ -f "$PIDFILE" ] || { echo "no $PIDFILE -- nothing to kill"; exit 0; }
PGID=$(cat "$PIDFILE")
rm -f "$PIDFILE"
kill -9 -- "-$PGID" 2>/dev/null
for i in $(seq 1 30); do
    pgrep -g "$PGID" >/dev/null 2>&1 || break
    sleep 1
    [ "$i" = 30 ] && { echo "WARNING: pgid $PGID still alive after 30s" >&2; exit 1; }
done
echo "prefill_$I (pgid $PGID) gone"
