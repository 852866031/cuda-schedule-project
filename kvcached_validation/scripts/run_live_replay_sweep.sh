#!/usr/bin/env bash
# Live sweep with the SAME request timeline at every budget: against a running kvcached engine,
# for each limit: apply it with kvctl, let the engine settle idle for a few seconds, then replay
# the identical seeded prefix workload (warm-up + <duration_s> measured) and score only that
# window. This makes every budget's window the same request sequence as the static control
# (run_static_sweep.sh); the engine's cache contents carry over between budgets (the live arm's
# real property), nothing is restarted.
#   run_live_replay_sweep.sh <name> <rate> <duration_s> <limit>...
#   e.g. run_live_replay_sweep.sh livereplay_offload_zipf 2 150 13G 10G 8G 6G 5G 4G 3G 2G
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); source "$D/env.sh"
NAME=$1; RATE=$2; DUR=$3; shift 3
OUT="$RESULTS/${NAME}_windows.csv"; rm -f "$OUT"; LOGF="$RESULTS/${NAME}_limits.jsonl"; rm -f "$LOGF"
for L in "$@"; do
  TAG="${NAME}_${L}"; R="$RESULTS/$TAG"; rm -f "${R}_sampler.csv" "${R}_load.jsonl"
  python "$D/resize.py" --ipc "$KV_IPC" --out "$LOGF" --step "0:$L" | cut -c1-160
  sleep "${SETTLE:-10}"
  python "$D/sampler.py" --out "${R}_sampler.csv" --duration $((DUR + 130)) --ipc "$KV_IPC" & SP=$!
  python "$D/loadgen.py" --mode prefix --rate "$RATE" --duration "$DUR" --n-sessions "${N_SESSIONS:-32}" \
      --prefix-tokens "${PREFIX_TOKENS:-6144}" --unique-tokens 128 --max-tokens "${MAX_TOKENS:-128}" \
      --skew "${SKEW:-zipf}" --out "${R}_load.jsonl" 2>&1 | while IFS= read -r line; do
        echo "$line"; [[ $line == "warm-up done" ]] && echo "$(date +%s.%N)" > "${R}_t0.txt"; done
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  python "$D/analyze_sweep.py" --name "$TAG" --sampler "${R}_sampler.csv" --load "${R}_load.jsonl" \
      --resize /dev/null --t0 "$(cat "${R}_t0.txt")" --step "$DUR" --figdir "$RESULTS" | tail -2
  python - "$OUT" "$RESULTS/${TAG}_windows.csv" "$L" <<'PY'
import sys, pandas as pd, os
out, src, lab = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src); df["limit"] = lab; df["budget_gib"] = float(lab.rstrip("G"))
df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
PY
done
echo "DONE $NAME -> $OUT"
