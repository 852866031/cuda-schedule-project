#!/usr/bin/env bash
# Overnight plan 3, R5: every tenant shape under the same byte allowance, next to the real engine at 1 req/s.
# run_probe.sh redone inside the C1 frame: tenant on a 32-SM green context with a fixed GB/s allowance (GBPS, default 50),
# engine KV limit fixed at 3 GiB, tenant budget fixed at 7.5 GiB. Segments: no tenant, shapes 2..6, no tenant,
# shapes 6..2, no tenant; SEG seconds per shape (default 120), NT seconds per no-tenant window (default 60).
#   run_probe2.sh [tag]      (GPU 0)    -> results_o3/<tag>_segments.json, <tag>_ft<shape>_p<pass>.csv, <tag>_load.jsonl
set -uo pipefail
TAG=${1:-probe2}; GBPS=${GBPS:-50}; SEG=${SEG:-120}; GAP=${GAP:-20}; NT=${NT:-60}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
PL=$(nvidia-smi --query-gpu=power.limit -i 0 --format=csv,noheader,nounits | cut -d. -f1)
if [[ -z $PL || $PL -gt 500 ]]; then echo "GPU 0 power limit is ${PL:-unknown} W; requires <= 500 W"; exit 9; fi
export VENV_NAME=.venv-main
source "$KV/scripts/env.sh"
export COLOC_XFER_DIR=/dev/shm
S=$C/results_o3/$TAG; L=$C/logs_o3; mkdir -p "$C/results_o3" "$L"
rm -f /dev/shm/coloc_xfer_inf.csv /dev/shm/coloc_xfer_ft.csv /dev/shm/coloc_ft_budget /dev/shm/coloc_ft_state /dev/shm/coloc_ft_snapshot.pt /dev/shm/coloc_hp_busy
stop_own() { local pf="$KV/logs/vllm_coloc_${TAG}.pid"; local pid; pid=$(cat "$pf" 2>/dev/null)
  if [[ -n $pid ]]; then pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 4; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; rm -f "$pf"; fi
  pkill -f "[s]cripts/loadgen.py --mode prefix" 2>/dev/null; pkill -f "[s]ampler.py --out $C/results_o3" 2>/dev/null; }
stop_own; pkill -f "[c]oloc_adjust/controller.py" 2>/dev/null; pkill -f "[c]oloc_adjust/ft_worker.py --out $C/results_o3/$TAG" 2>/dev/null; sleep 1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; sleep 1
export PYTHONPATH=$C/hp_patch COLOC_HP_SIGNAL=1 COLOC_XFER_TRACE=1
"$KV/scripts/serve.sh" "coloc_$TAG" kvcached --offload "${OFFLOAD:-28}" > /dev/null
if ! "$KV/scripts/wait_ready.sh" "coloc_$TAG" 600 > "$L/${TAG}_engine_start.txt" 2>&1; then echo "engine failed"; tail -5 "$L/${TAG}_engine_start.txt"; exit 3; fi
unset PYTHONPATH COLOC_HP_SIGNAL COLOC_XFER_TRACE
kvctl limit "$KV_IPC" 3G > /dev/null
python "$C/controller.py" --listen 8001 --upstream http://localhost:$VLLM_PORT --ipc "$KV_IPC" --log "${S}_controller.csv" --no-adjust --fixed-budget-gib 7.5 > "$L/${TAG}_controller.log" 2>&1 & CP=$!
sleep 3
DUR=$(( 3 * NT + 10 * (SEG + GAP) + 60 ))
python "$KV/scripts/sampler.py" --out "${S}_sampler.csv" --duration $((DUR + 100)) --ipc "$KV_IPC" --metrics-url "http://localhost:$VLLM_PORT/metrics" & SP=$!
python "$KV/scripts/loadgen.py" --mode prefix --rate 1 --rps-schedule "0:1" --duration "$DUR" --url http://localhost:8001 \
    --n-sessions 32 --prefix-tokens 6144 --unique-tokens 128 --max-tokens 128 --skew zipf --out "${S}_load.jsonl" > "$L/${TAG}_load.log" 2>&1 & LP=$!
sleep 35   # warm-up (32 requests) + a few seconds into the measured window
T0=$(cat "${S}_load.jsonl.t0")
seg() { python - "$T0" "$@" <<'PYEOF'
import sys, time; t0 = float(sys.argv[1]); shape = int(sys.argv[2]); p = int(sys.argv[3]); lead = float(sys.argv[4]); length = float(sys.argv[5]); now = time.time()
print(f'{{"shape": {shape}, "pass": {p}, "t_from": {now - t0 + lead:.1f}, "t_to": {now - t0 + length:.1f}}},')
PYEOF
}
echo "[" > "${S}_segments.json"
notenant() { seg -1 "$1" 0 "$NT" >> "${S}_segments.json"; sleep "$NT"; }
shape() { local SHAPE=$1 PASS=$2
  seg "$SHAPE" "$PASS" 15 "$SEG" >> "${S}_segments.json"
  CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft${SHAPE}_p${PASS}.csv" --fresh --fixed-budget-gib 7.5 --fixed-tier "$SHAPE" --green-sms 32 --gbps-cap "$GBPS" --duration "$SEG" > "$L/${TAG}_ft${SHAPE}_p${PASS}.log" 2>&1
  sleep "$GAP"; }
notenant 0
for SHAPE in 2 3 4 5 6; do shape "$SHAPE" 1; done
notenant 1
for SHAPE in 6 5 4 3 2; do shape "$SHAPE" 2; done
notenant 2
echo '{"shape": -9, "pass": 0, "t_from": 0, "t_to": 0}]' >> "${S}_segments.json"
wait $LP 2>/dev/null
kill $SP $CP 2>/dev/null; sleep 1
cp /dev/shm/coloc_xfer_inf.csv "${S}_xfer_inf.csv" 2>/dev/null; cp /dev/shm/coloc_xfer_ft.csv "${S}_xfer_ft.csv" 2>/dev/null; cp "${S}_load.jsonl.t0" "${S}_t0.txt"
stop_own; echo quit | nvidia-cuda-mps-control
echo "DONE probe2 $TAG"
