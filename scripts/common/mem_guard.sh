#!/bin/bash
# Kills the disagg instances if available RAM collapses.
#
# Pinned memory cannot be swapped or reclaimed, so overcommitting it does not produce a clean
# OOM kill -- it freezes the machine. This watchdog trips well before that point.
#
#   scripts/mem_guard.sh [floor_mb] [poll_seconds]
FLOOR_MB="${1:-8000}"
POLL="${2:-2}"
echo "mem_guard: killing vLLM if available RAM drops below ${FLOOR_MB} MB (poll ${POLL}s)"
while true; do
    AVAIL=$(free -m | awk '/^(Mem|内存)/ {print $7}')
    if [ -n "$AVAIL" ] && [ "$AVAIL" -lt "$FLOOR_MB" ]; then
        echo "$(date '+%H:%M:%S') MEM GUARD TRIPPED: ${AVAIL} MB available < ${FLOOR_MB} MB -- killing vLLM"
        pkill -9 -f "VLLM::EngineCore"
        pkill -9 -f "vllm.entrypoints.cli.main serve"
        exit 1
    fi
    sleep "$POLL"
done
