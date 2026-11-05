#!/usr/bin/env bash
# Live shrink sweep at a fixed request rate (study-1 workload: 32 sessions x 6144-token prefixes,
# +128 unique tokens, 128 output tokens, Poisson arrivals), against an already-running engine.
#   run_shrink_sweep.sh <name> <rate> <step_seconds> <limit>...
#   e.g. run_shrink_sweep.sh sweep_offload 2 60 10G 8G 6G 5G 4G 3G 2G
# The first <step_seconds> run at the engine's current limit (baseline), then each limit holds
# for <step_seconds>. Env: SKEW=zipf|uniform, N_SESSIONS, PREFIX_TOKENS, MAX_TOKENS.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); source "$D/env.sh"
NAME=$1; RATE=$2; STEP=$3; shift 3
R="$RESULTS/$NAME"; rm -f "${R}_sampler.csv" "${R}_load.jsonl" "${R}_resize.jsonl"
STEPS=(); i=1; for l in "$@"; do STEPS+=(--step "$((i*STEP)):$l"); i=$((i+1)); done
TOTAL=$((i*STEP))
WARM_ALLOW=90   # the load generator's warm-up (32 sessions) precedes its measured window
python "$D/sampler.py" --out "${R}_sampler.csv" --duration $((TOTAL + WARM_ALLOW + 40)) --ipc "$KV_IPC" & SP=$!
# resize clock starts when the load generator reports its warm-up done
python "$D/loadgen.py" --mode prefix --rate "$RATE" --duration "$TOTAL" --n-sessions "${N_SESSIONS:-32}" \
    --prefix-tokens "${PREFIX_TOKENS:-6144}" --unique-tokens 128 --max-tokens "${MAX_TOKENS:-128}" \
    --skew "${SKEW:-zipf}" --out "${R}_load.jsonl" 2>&1 | while IFS= read -r line; do
      echo "$line"
      if [[ $line == "warm-up done" ]]; then
        python "$D/resize.py" --ipc "$KV_IPC" --out "${R}_resize.jsonl" "${STEPS[@]}" > "${R}_resize.out" 2>&1 &
        echo "$(date +%s.%N)" > "${R}_t0.txt"
      fi
    done
wait $SP || true
python "$D/analyze_sweep.py" --name "$NAME" --sampler "${R}_sampler.csv" --load "${R}_load.jsonl" \
    --resize "${R}_resize.jsonl" --t0 "$(cat "${R}_t0.txt")" --step "$STEP" --figdir "$RESULTS"
