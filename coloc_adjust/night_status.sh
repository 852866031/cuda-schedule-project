#!/usr/bin/env bash
# Night of Nov 21/22: one screen of status for the vLLM queue on GPU 0.
C=$(cd "$(dirname "$0")" && pwd); cd "$C"
echo "$(date +%H:%M:%S) | queue alive: $(pgrep -f 'run_n21_gpu0[a-z]*.sh' | wc -l) | last: $(tail -1 logs_n21/batch_gpu0.log | cut -c1-70)"
grep " end " logs_n21/batch_gpu0.log | grep -v "rc=0 1217 request rows" | tail -5 | cut -c1-110
../kvcached_validation/.venv-main/bin/python analyze_night.py 2>/dev/null | sed -n '/## Repeats/,$p' | tail -n +4 | awk -F'|' '{printf "%-18s n=%s mean %s TPOTp90 %s step p50 %s p90 %s p99 %s | tok/s %s comp %s whole %s\n", $2, $3, $4, $6, $7, $8, $9, $13, $14, $15}' | sed 's/  */ /g'
