#!/usr/bin/env bash
# One shrink/grow experiment against an already-running engine:
# sampler (100 ms) + timed kvctl limits + streaming load, then analysis.
#   run_resize_exp.sh <name> <mode:stream|session> <rate_or_sessions> <duration_s> <step>...
#   e.g. run_resize_exp.sh expA1 stream 3.5 150 40:3G 90:12G   (no steps = plain load run)
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=$1; MODE=$2; LOAD=$3; DUR=$4; shift 4
R="$RESULTS/$NAME"; rm -f "${R}_sampler.csv" "${R}_load.jsonl" "${R}_resize.jsonl"   # only this run's own files
STEPS=(); for s in "$@"; do STEPS+=(--step "$s"); done
T0=$(date +%s.%N)
python "$(dirname "$0")/sampler.py" --out "${R}_sampler.csv" --duration $((DUR + 25)) --ipc "$KV_IPC" & SP=$!
RP=""; RARG=()
if [[ ${#STEPS[@]} -gt 0 ]]; then
  python "$(dirname "$0")/resize.py" --ipc "$KV_IPC" --out "${R}_resize.jsonl" --t0 "$T0" "${STEPS[@]}" & RP=$!
  RARG=(--resize "${R}_resize.jsonl")
fi
if [[ $MODE == stream ]]; then
  python "$(dirname "$0")/loadgen.py" --mode stream --rate "$LOAD" --duration "$DUR" --prompt-tokens "${PROMPT_TOKENS:-6000}" --max-tokens "${MAX_TOKENS:-128}" --out "${R}_load.jsonl"
else
  python "$(dirname "$0")/loadgen.py" --mode session --sessions "$LOAD" --duration "$DUR" --prompt-tokens "${PROMPT_TOKENS:-3000}" --turn-tokens "${TURN_TOKENS:-256}" --think "${THINK:-3}" --max-tokens "${MAX_TOKENS:-128}" --out "${R}_load.jsonl"
fi
[[ -n $RP ]] && wait $RP; wait $SP
# engine-side evidence: kvcached / error lines emitted during the run
python "$(dirname "$0")/analyze.py" --name "$NAME" --sampler "${R}_sampler.csv" --load "${R}_load.jsonl" "${RARG[@]}" --figdir "$RESULTS"
