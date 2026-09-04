#!/bin/bash
# Periodic human-readable status of a running coloc2_sweep matrix.
#
#   scripts/inf_inf_coloc/coloc2_progress.sh [interval_s=30]
#
# Rewrites output/logs/coloc2_progress.txt every poll:
#   - which run (N of total) and which phase it is in (from the driver log)
#   - which components are alive right now (pidfile + kill -0)
#   - GPU memory/utilization snapshot
#   - one line per completed run
# Exits when the driver exits. Check with:  cat output/logs/coloc2_progress.txt
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$REPO/output/logs/coloc2_matrix.log"
OUTFILE="$REPO/output/logs/coloc2_progress.txt"
POLL="${1:-30}"

while true; do
    {
        echo "inf+inf colocation matrix -- $(date '+%F %T')"
        echo
        if ! pgrep -f "coloc2_sweep.py --scenarios" >/dev/null 2>&1; then
            if grep -q "^done in" "$LOG" 2>/dev/null; then
                echo "STATUS: COMPLETE -- $(grep '^done in' "$LOG")"
            else
                echo "STATUS: DRIVER NOT RUNNING (did not complete -- see $LOG)"
            fi
        else
            started=$(grep -c "^=== " "$LOG" 2>/dev/null); started=${started:-0}
            finished=$(grep -c "qwen: TTFT" "$LOG" 2>/dev/null); finished=${finished:-0}
            cur=$(grep "^=== " "$LOG" 2>/dev/null | tail -1 | sed 's/=== //; s/ .*//')
            # Phase = the furthest milestone the current run has logged.
            phase="launching engines"
            block=$(awk '/^=== /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$LOG" 2>/dev/null)
            case "$block" in *"qwen: TTFT"*) phase="finishing / teardown";; esac
            [ "$phase" = "launching engines" ] && case "$block" in *"qwen warmup:"*) phase="MEASURING (both clients live, ~150 s)";; esac
            [ "$phase" = "launching engines" ] && case "$block" in *"8B warmup:"*) phase="qwen warmup";; esac
            [ "$phase" = "launching engines" ] && case "$block" in *"8B warmup"*|*"up in "*) phase="8B warmup (~75 s)";; esac
            [ "$phase" = "8B warmup (~75 s)" ] && case "$block" in *"populate:"*) : ;; esac
            case "$block" in *"STALLED"*) phase="STALLED -- watchdog killing the stack";; esac
            case "$block" in *"FAILED"*) phase="FAILED (see log)";; esac
            echo "STATUS: run $started of 8 in progress ($finished finished): $cur -- $phase"
        fi
        echo
        echo "components alive:"
        any=""
        for f in /tmp/disagg_prefill.pid /tmp/disagg_decode.pid /tmp/disagg_proxy.pid \
                 /tmp/disagg_lmcserver.pid /tmp/qwen_prefill.pid /tmp/qwen_decode.pid \
                 /tmp/qwen_proxy.pid /tmp/qwen_lmcserver.pid; do
            [ -f "$f" ] || continue
            pid=$(cat "$f" 2>/dev/null)
            name=$(basename "$f" .pid)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "  $name (pid $pid) up"; any=1
            else
                echo "  $name pidfile present but pid $pid DEAD"
            fi
        done
        [ -z "$any" ] && echo "  (none -- between runs, or stack starting/stopping)"
        echo
        echo "gpus:"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
        hb=$(cat /tmp/disagg_heartbeat_status 2>/dev/null)
        [ -n "$hb" ] && echo "heartbeat: $hb"
        echo
        echo "completed runs:"
        grep -E "^=== |8B: |qwen: " "$LOG" 2>/dev/null | \
            awk '/^=== /{name=$2} /8B: /{big=$0} /qwen: /{sub(/^ +/,""); print "  " name "  " $0 (big!=""? "  |  8B " substr(big, index(big,"TTFT")) : ""); big=""}'
    } > "$OUTFILE.tmp" && mv "$OUTFILE.tmp" "$OUTFILE"
    pgrep -f "coloc2_sweep.py --scenarios" >/dev/null 2>&1 || break
    sleep "$POLL"
done
