#!/usr/bin/env bash
# P1.2 shape probe: at a fixed 1 req/s, run each tenant shape alone for 60 s under a fixed
# 4k tok/s cap and record decode TPOT, to get "slowdown per tenant tok/s" per shape.
#   run_probe.sh [tag]      (GPU 0; engine KV limit fixed at 3 GiB, tenant budget fixed at 7.5 GiB)
# Segments: 60 s no tenant, then shapes 2..6 (k=4, k=8, +act offload, +12 layers, +20 layers),
# each a fresh tenant process for 60 s with 20 s gaps. Segment times -> results/<tag>_segments.json
set -uo pipefail
TAG=${1:-probe}; CAP=${CAP:-4000}; SEG=${SEG:-60}; GAP=${GAP:-20}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
PL=$(nvidia-smi --query-gpu=power.limit -i 0 --format=csv,noheader,nounits | cut -d. -f1)
if [[ -z $PL || $PL -gt 500 ]]; then echo "GPU 0 power limit is ${PL:-unknown} W; requires <= 500 W"; exit 9; fi
export VENV_NAME=.venv-main
source "$KV/scripts/env.sh"
export COLOC_XFER_DIR=/dev/shm
S=$C/results/$TAG; L=$C/logs; mkdir -p "$C/results" "$L"
rm -f /dev/shm/coloc_xfer_inf.csv /dev/shm/coloc_xfer_ft.csv /dev/shm/coloc_ft_budget /dev/shm/coloc_ft_state /dev/shm/coloc_ft_snapshot.pt /dev/shm/coloc_hp_busy
stop_own() { local pf="$KV/logs/vllm_coloc_${TAG}.pid"; local pid; pid=$(cat "$pf" 2>/dev/null)
  if [[ -n $pid ]]; then pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 4; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; rm -f "$pf"; fi
  pkill -f "[s]cripts/loadgen.py --mode prefix" 2>/dev/null; pkill -f "[s]ampler.py --out $C/results" 2>/dev/null; }
stop_own; pkill -f "[c]oloc_adjust/controller.py" 2>/dev/null; pkill -f "[c]oloc_adjust/ft_worker.py" 2>/dev/null; sleep 1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; sleep 1
export PYTHONPATH=$C/hp_patch COLOC_HP_SIGNAL=1 COLOC_XFER_TRACE=1
"$KV/scripts/serve.sh" "coloc_$TAG" kvcached --offload 28 > /dev/null
if ! "$KV/scripts/wait_ready.sh" "coloc_$TAG" 600 > "$L/${TAG}_engine_start.txt" 2>&1; then echo "engine failed"; tail -5 "$L/${TAG}_engine_start.txt"; exit 3; fi
unset PYTHONPATH COLOC_HP_SIGNAL COLOC_XFER_TRACE
kvctl limit "$KV_IPC" 3G > /dev/null
python "$C/controller.py" --listen 8001 --upstream http://localhost:$VLLM_PORT --ipc "$KV_IPC" --log "${S}_controller.csv" --no-adjust --fixed-budget-gib 7.5 > "$L/${TAG}_controller.log" 2>&1 & CP=$!
sleep 3
python "$KV/scripts/sampler.py" --out "${S}_sampler.csv" --duration 900 --ipc "$KV_IPC" --metrics-url "http://localhost:$VLLM_PORT/metrics" & SP=$!
DUR=$(( SEG + 5 * (SEG + GAP) + 20 ))
python "$KV/scripts/loadgen.py" --mode prefix --rate 1 --rps-schedule "0:1" --duration "$DUR" --url http://localhost:8001 \
    --n-sessions 32 --prefix-tokens 6144 --unique-tokens 128 --max-tokens 128 --skew zipf --out "${S}_load.jsonl" > "$L/${TAG}_load.log" 2>&1 & LP=$!
sleep 35   # warm-up (32 requests) + a few seconds into the measured window
T0=$(cat "${S}_load.jsonl.t0")
echo "[" > "${S}_segments.json"
python - "$T0" "$SEG" <<'EOF' >> "${S}_segments.json"
import sys, time; t0=float(sys.argv[1]); seg=float(sys.argv[2]); now=time.time()
print(f'{{"shape": -1, "label": "no tenant", "t_from": {now-t0:.1f}, "t_to": {now-t0+seg:.1f}}},')
EOF
sleep "$SEG"
i=0; for SHAPE in 2 3 4 5 6; do
  i=$((i+1))
  CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft${SHAPE}.csv" --fresh --fixed-budget-gib 7.5 --fixed-tier "$SHAPE" --tok-s-cap "$CAP" --duration "$SEG" > "$L/${TAG}_ft${SHAPE}.log" 2>&1 & FP=$!
  sleep 12   # model load; the segment starts when steps begin
  python - "$T0" "$SEG" "$SHAPE" "$i" <<'EOF' >> "${S}_segments.json"
import sys, time; t0=float(sys.argv[1]); seg=float(sys.argv[2]); shape=int(sys.argv[3]); last=int(sys.argv[4])==5; now=time.time()
print(f'{{"shape": {shape}, "t_from": {now-t0:.1f}, "t_to": {now-t0+seg-12:.1f}}}' + ("" if last else ","))
EOF
  wait $FP 2>/dev/null
  sleep "$GAP"
done
echo "]" >> "${S}_segments.json"
wait $LP 2>/dev/null
kill $SP $CP 2>/dev/null; sleep 1
cp /dev/shm/coloc_xfer_inf.csv "${S}_xfer_inf.csv" 2>/dev/null; cp /dev/shm/coloc_xfer_ft.csv "${S}_xfer_ft.csv" 2>/dev/null; cp "${S}_load.jsonl.t0" "${S}_t0.txt"
stop_own; echo quit | nvidia-cuda-mps-control
python "$C/analyze_probe.py" --tag "$TAG" --dir "$C/results" 2>&1 | tail -20
echo "DONE probe $TAG"
