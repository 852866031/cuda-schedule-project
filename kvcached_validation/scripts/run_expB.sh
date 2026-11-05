#!/usr/bin/env bash
# Experiment B: kvcached + LMCache multi-process connector.
#   run_expB.sh <mode: lmcache_driven|vmm|engine_driven> [kvcached|plain]
# 1. restart LMCache server (auto transfer mode, small pinned L1) and the engine with the connector
# 2. if the engine fails to start, save the traceback and stop (that is a result)
# 3. output check: fixed prompts twice (cold store, then warm load from LMCache), then
# 4. multi-turn session load with a shrink/grow (40 s: 2G, 90 s: 12G), then LMCache log summary.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
source "$D/env.sh"
MODE=$1; ENGINE=${2:-kvcached}; NAME="expB_${MODE}_${ENGINE}${SUFFIX:-}"
"$D/stop.sh" >/dev/null 2>&1
"$D/lmcache_server.sh" "$NAME" --l1-gb 6 --transfer auto
sleep 6
"$D/serve.sh" "$NAME" "$ENGINE" --lmcache "$MODE"
if ! "$D/wait_ready.sh" "$NAME" 600 > "$RESULTS/${NAME}_startup.txt" 2>&1; then
  echo "ENGINE FAILED TO START — see $RESULTS/${NAME}_startup.txt"
  grep -n -E "Error|error|Traceback|RuntimeError|cuda|CUDA" "$LOGS/vllm_${NAME}.log" | grep -v FutureWarning | tail -40 >> "$RESULTS/${NAME}_startup.txt"
  tail -60 "$LOGS/lmcache_${NAME}.log" > "$RESULTS/${NAME}_lmcache_tail.txt"
  exit 3
fi
cat "$RESULTS/${NAME}_startup.txt"
grep -n -i "lmcache\|connector\|transfer" "$LOGS/vllm_${NAME}.log" | grep -v -i "warning: .*deprecat" | cut -c1-240 | tail -25 > "$RESULTS/${NAME}_connector_lines.txt"
# 3. output check (cold, then warm)
rm -f "$RESULTS/${NAME}_check_cold.jsonl" "$RESULTS/${NAME}_check_warm.jsonl"
python "$D/loadgen.py" --mode check --n-check 8 --out "$RESULTS/${NAME}_check_cold.jsonl" > "$LOGS/${NAME}_check_cold.out" 2>&1
sleep 2
python "$D/loadgen.py" --mode check --n-check 8 --out "$RESULTS/${NAME}_check_warm.jsonl" > "$LOGS/${NAME}_check_warm.out" 2>&1
# 4. session load + resize
"$D/run_resize_exp.sh" "$NAME" session "${SESSIONS:-6}" "${DUR:-150}" ${STEPS:-40:2G 90:12G}
# LMCache server-side evidence
L="$LOGS/lmcache_${NAME}.log"
S="$RESULTS/${NAME}_lmcache_summary.txt"
echo "stored lines: $(grep -c -i Stored "$L")" > "$S"
echo "retrieved lines: $(grep -c -i Retriev "$L")" >> "$S"
echo "error lines: $(grep -c -i -E "error|exception|traceback" "$L")" >> "$S"
grep -i -E "error|exception|traceback|illegal|fault" "$L" | head -20 >> "$S"
grep -i -E "error|exception|traceback|illegal|fault|kvcached\]\[WARN" "$LOGS/vllm_${NAME}.log" | grep -v -i "FutureWarning\|pynvml" | cut -c1-300 | tail -30 > "$RESULTS/${NAME}_engine_errors.txt"
curl -s localhost:8181/metrics 2>/dev/null | grep -i -E "^lmcache.*(store|retrieve|hit|miss).*(total|count)" | head -20 > "$RESULTS/${NAME}_lmcache_metrics.txt" || true
echo "DONE $NAME"
