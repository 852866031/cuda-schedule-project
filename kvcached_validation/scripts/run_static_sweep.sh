#!/usr/bin/env bash
# Static-budget sweep on the same vLLM version: one plain (no kvcached) engine restart per
# budget, --kv-cache-memory-bytes fixed, same prefix workload as run_shrink_sweep.sh.
#   run_static_sweep.sh <name> <rate> <duration_s> [--apc] [--offload GiB] -- <budgetGiB>...
#   e.g. run_static_sweep.sh static_offload_zipf 2 150 --apc --offload 24 -- 13 10 8 6 5 4 3 2
# Each budget gets its own sampler/load files (<name>_b<GiB>_*) and a one-window summary;
# the rows are gathered into results/<name>_windows.csv.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); source "$D/env.sh"
NAME=$1; RATE=$2; DUR=$3; shift 3
SERVE_OPTS=(); while [[ $# -gt 0 && $1 != "--" ]]; do SERVE_OPTS+=("$1"); shift; done; shift
OUT="$RESULTS/${NAME}_windows.csv"; rm -f "$OUT"
for B in "$@"; do
  TAG="${NAME}_b${B}"; R="$RESULTS/$TAG"; rm -f "${R}_sampler.csv" "${R}_load.jsonl"
  "$D/stop.sh" > /dev/null 2>&1
  "$D/serve.sh" "$TAG" plain --kv-bytes $((B * 1024 * 1024 * 1024)) "${SERVE_OPTS[@]}" > /dev/null
  if ! "$D/wait_ready.sh" "$TAG" 600 > "$RESULTS/${TAG}_startup.txt" 2>&1; then echo "budget $B: engine failed to start"; continue; fi
  python "$D/sampler.py" --out "${R}_sampler.csv" --duration $((DUR + 130)) --ipc "$KV_IPC" & SP=$!
  python "$D/loadgen.py" --mode prefix --rate "$RATE" --duration "$DUR" --n-sessions "${N_SESSIONS:-32}" \
      --prefix-tokens "${PREFIX_TOKENS:-6144}" --unique-tokens 128 --max-tokens "${MAX_TOKENS:-128}" \
      --skew "${SKEW:-zipf}" --out "${R}_load.jsonl" 2>&1 | while IFS= read -r line; do
        echo "$line"; [[ $line == "warm-up done" ]] && echo "$(date +%s.%N)" > "${R}_t0.txt"; done
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "[]" > /dev/null
  python "$D/analyze_sweep.py" --name "$TAG" --sampler "${R}_sampler.csv" --load "${R}_load.jsonl" \
      --resize /dev/null --t0 "$(cat "${R}_t0.txt")" --step "$DUR" --figdir "$RESULTS" | tail -3
  python - "$OUT" "$RESULTS/${TAG}_windows.csv" "$B" <<'PY'
import sys, pandas as pd, os
out, src, b = sys.argv[1], sys.argv[2], float(sys.argv[3])
df = pd.read_csv(src); df["limit"] = f"{b:g}G"; df["budget_gib"] = b
df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
PY
done
"$D/stop.sh" > /dev/null 2>&1
echo "DONE $NAME -> $OUT"
