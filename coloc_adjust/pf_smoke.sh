#!/usr/bin/env bash
# P3 bring-up on GPU 1 (while GPU 0 runs the main chain): Qwen2.5-3B fp8 on vLLM 0.30 + kvcached (own
# instance) with native host-RAM KV offload; 20 prefill-only requests; a kvcached limit change; sleep/wake.
#   pf_smoke.sh [--sleep]
set -uo pipefail
SLEEP=${1:-}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
export VENV_NAME=.venv-main MODEL=${PF_MODEL:-Qwen/Qwen2.5-3B} VLLM_PORT=8200 KV_IPC=vllm_c
source "$KV/scripts/env.sh"
L=$C/logs; mkdir -p "$L"; OUT="$L/pf_smoke.txt"; : > "$OUT"
ARGS=(--quantization fp8 --max-model-len 4608 --max-num-seqs 8 --max-num-batched-tokens 8192)
[[ $SLEEP == --sleep ]] && ARGS+=(--enable-sleep-mode)
"$KV/scripts/serve.sh" "pf_smoke" kvcached --gpu 1 --offload "${PF_OFFLOAD:-2}" -- "${ARGS[@]}" > /dev/null
PID=$(cat "$KV/logs/vllm_pf_smoke.pid")
cleanup() { pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null; sleep 4; pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null; rm -f "$KV/logs/vllm_pf_smoke.pid"; }
trap cleanup EXIT
if ! "$KV/scripts/wait_ready.sh" pf_smoke 600 >> "$OUT" 2>&1; then echo "ENGINE FAILED" | tee -a "$OUT"; grep -n -E "Error|Traceback" "$KV/logs/vllm_pf_smoke.log" | grep -v -i "FutureWarning\|KVCACHED_MEMORY_POOL" | tail -20 | tee -a "$OUT"; exit 3; fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT"
kvctl list 2>&1 | tail -3 | tee -a "$OUT"
# 20 prefill-only requests at 2 req/s
python "$KV/scripts/loadgen.py" --mode stream --prompt-tokens 3072 --max-tokens 1 --rate 2 --duration 10 --url http://localhost:8200 --model "$MODEL" --vocab 151936 --out "$L/pf_smoke_load.jsonl" 2>&1 | grep -v "^{" | tail -2 | tee -a "$OUT"
python - "$L/pf_smoke_load.jsonl" <<'EOF' | tee -a "$OUT"
import json, sys, statistics as st
R = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in R if r["status"] == 200 and r["tok_t"]]
lat = [r["tok_t"][0] - r["t_send"] for r in ok]
print(f"requests {len(R)} ok {len(ok)}; prefill latency median {st.median(lat)*1e3:.0f} ms, max {max(lat)*1e3:.0f} ms; "
      f"prompt tok/s over the burst {sum(r['prompt_tokens'] for r in ok)/(max(r['t_done'] for r in ok)-min(r['t_send'] for r in ok)):.0f}")
EOF
kvctl limit vllm_c 1G | tee -a "$OUT"; sleep 1; kvctl list 2>&1 | tail -2 | tee -a "$OUT"
python "$KV/scripts/loadgen.py" --mode stream --prompt-tokens 3072 --max-tokens 1 --rate 2 --duration 6 --url http://localhost:8200 --model "$MODEL" --vocab 151936 --out "$L/pf_smoke_load2.jsonl" 2>&1 | grep -v "^{" | tail -1 | tee -a "$OUT"
kvctl limit vllm_c 6G | tee -a "$OUT"
if [[ $SLEEP == --sleep ]]; then
  echo "sleep level 1:" | tee -a "$OUT"; U0=$(nvidia-smi --query-gpu=memory.used -i 1 --format=csv,noheader,nounits)
  curl -s -X POST "http://localhost:8200/sleep?level=1" | tee -a "$OUT"; sleep 3; U1=$(nvidia-smi --query-gpu=memory.used -i 1 --format=csv,noheader,nounits)
  echo " GPU1 used ${U0} -> ${U1} MiB; is_sleeping: $(curl -s http://localhost:8200/is_sleeping)" | tee -a "$OUT"
  t0=$(date +%s.%N); curl -s -X POST "http://localhost:8200/wake_up" | tee -a "$OUT"; echo " wake took $(python -c "import time;print(f'{time.time()-$t0:.2f}')") s" | tee -a "$OUT"
  python "$KV/scripts/loadgen.py" --mode stream --prompt-tokens 3072 --max-tokens 1 --rate 2 --duration 6 --url http://localhost:8200 --model "$MODEL" --vocab 151936 --out "$L/pf_smoke_load3.jsonl" 2>&1 | grep -v "^{" | tail -1 | tee -a "$OUT"
  kvctl list 2>&1 | tail -2 | tee -a "$OUT"
fi
grep -i -E "error|exception|Xid|illegal" "$KV/logs/vllm_pf_smoke.log" | grep -v -i "FutureWarning\|KVCACHED_MEMORY_POOL\|deprecat" | tail -5 | tee -a "$OUT"
echo "PF SMOKE DONE" | tee -a "$OUT"
