#!/usr/bin/env bash
# One evaluation scheme end to end (design §3.7).
#   run_eval7.sh <scheme> [schedule]   (run_eval6 + the schemes of PLAN_overnight3.md, R4; results in results_o3/, logs in logs_o3/;
#   the TPOT reference of the feedback schemes is the baseline of the same night, results_o3/adjust_notenant_phases.csv)
#     c1_fix      tenant on a fixed 32-SM green context, fixed GB/s allowance (GBPS, default 50), pause after each step
#     c1_fine     the same allowance kept in slices at block boundaries (SLICE blocks, default: results_o3/r1_slice.txt, else 8)
#     c2a, c2b    c1_fix with tenant variant V1 / V2 of results_o3/r2_variants.json, paced to the usual tenant's tokens/s
#     c1_slow15, c1_slow5   32-SM tenant, GB/s allowance from the slow TPOT loop (target +15 % / +5 %)
#   earlier text:     500 W round: schemes of run_eval3.sh + tpot_slo5 / tpot_slo15 (P1.1) + green_dyn2 (P2)
#     green32        tenant on a fixed 32-SM green context (no gate, MPS on)
#     green_dyn      tenant SM share set by the VRAM scheduler from the arrival rate (SM_TABLE), no gate
#     engine_green   engine confined to ENGINE_SMS (default 128) SMs + tenant on the rest-ish (green_dyn table)
# Starts: MPS daemon, decode-only engine (kvcached main, 28 GiB host-RAM tier, prefix caching off,
# busy-window patch + transfer tracer), router + VRAM scheduler on :8001, sampler, FT worker (per scheme),
# then the reference prefix workload with the RPS schedule through the router; tears everything
# down and runs analyze_coloc.py. Logs in coloc_adjust/logs, results in coloc_adjust/results.
set -uo pipefail
PL=$(nvidia-smi --query-gpu=power.limit -i 0 --format=csv,noheader,nounits | cut -d. -f1)
if [[ -z $PL || $PL -gt 500 ]]; then echo "GPU 0 power limit is ${PL:-unknown} W; this round requires <= 500 W (sudo nvidia-smi -i 0,1 -pl 500)"; exit 9; fi
ARM=$1; SCHED=${2:-"0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"}
DUR=${DUR:-660}
C=$(cd "$(dirname "$0")" && pwd); KV=$C/../kvcached_validation
export VENV_NAME=.venv-main
source "$KV/scripts/env.sh"
export COLOC_XFER_DIR=/dev/shm
RES=$C/results_o3; TAG=${TAG:-$ARM}; S=$RES/$TAG; L=$C/logs_o3; mkdir -p "$RES" "$L"
GBPS=${GBPS:-50}; SLICE=${SLICE:-$(cat "$RES/r1_slice.txt" 2>/dev/null || echo 8)}
rm -f /dev/shm/coloc_xfer_inf.csv /dev/shm/coloc_xfer_ft.csv /dev/shm/coloc_ft_budget /dev/shm/coloc_ft_state /dev/shm/coloc_ft_snapshot.pt /dev/shm/coloc_hp_busy
# Only this runner's own engine is stopped (never scripts/stop.sh: it kills every vLLM on the box, incl. the GPU-1 LMCache tests)
stop_own() { local pf="$KV/logs/vllm_coloc_${ARM}.pid"; local pid; pid=$(cat "$pf" 2>/dev/null)
  if [[ -n $pid ]]; then pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 4; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; rm -f "$pf"; fi
  pkill -f "[s]cripts/loadgen.py --mode prefix" 2>/dev/null; pkill -f "[s]ampler.py --out $C/results_o3" 2>/dev/null; }
stop_own
pkill -f "[c]oloc_adjust/controller.py" 2>/dev/null
for p in $(pgrep -f "[c]oloc_adjust/ft_worker.py"); do grep -qa "corun2_\|tenant_bytes" "/proc/$p/cmdline" || kill "$p" 2>/dev/null; done; sleep 1   # not the tenants of the synthetic runs on the other card   # bracket: the pattern must not match a shell whose command line quotes it

# --- MPS (user-space daemon on GPU 0); every GPU process below inherits the pipe dir
export CUDA_MPS_PIPE_DIRECTORY=/tmp/coloc_mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/coloc_mps/log   # short path: Unix-socket length limit
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if [[ ${MPS:-1} == 1 ]]; then CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; sleep 1; fi

# --- engine
export PYTHONPATH=$C/hp_patch COLOC_HP_SIGNAL=1 COLOC_XFER_TRACE=1
[[ $ARM == engine_green || $ARM == green_split ]] && export COLOC_ENGINE_SMS=${ENGINE_SMS:-128}
"$KV/scripts/serve.sh" "coloc_$ARM" kvcached --offload "${OFFLOAD:-28}" > /dev/null
if ! "$KV/scripts/wait_ready.sh" "coloc_$ARM" 600 > "$L/${TAG}_engine_start.txt" 2>&1; then echo "engine failed"; cat "$L/${TAG}_engine_start.txt" | tail -5; exit 3; fi
unset PYTHONPATH COLOC_HP_SIGNAL COLOC_XFER_TRACE COLOC_ENGINE_SMS

# --- router / VRAM scheduler
CTL_OPTS=(--listen 8001 --upstream http://localhost:$VLLM_PORT --ipc "$KV_IPC" --log "${S}_controller.csv")
case $ARM in
  decode_alone) CTL_OPTS+=(--no-adjust);;
  adjust_notenant) ;;   # VRAM scheduler adjusts the KV limit by load, nobody uses the freed VRAM: the baseline for the prototype
  static_split) CTL_OPTS+=(--no-adjust --fixed-budget-gib "${STATIC_FT_GIB:-6}"); kvctl limit "$KV_IPC" "${STATIC_KV_GIB:-6.7}G" > /dev/null;;
  adjust|adjust_nogate|adjust_mps20|green32|green_split) ;;
  tpot_slo5)  CTL_OPTS+=(--tpot-slo 0.05 --tpot-ref "$RES/adjust_notenant_phases.csv");;   # P1.1: tenant tokens/s capped from the decode TPOT error
  tpot_slo15) CTL_OPTS+=(--tpot-slo 0.15 --tpot-ref "$RES/adjust_notenant_phases.csv");;
  c1_fix|c1_fine|c2a|c2b) CTL_OPTS+=(--gbps-fixed "$GBPS");;                                # overnight 3: a constant byte allowance
  c1_slow15) CTL_OPTS+=(--tpot-slo 0.15 --slo-mode slow --tpot-ref "$RES/adjust_notenant_phases.csv");;
  c1_slow5)  CTL_OPTS+=(--tpot-slo 0.05 --slo-mode slow --tpot-ref "$RES/adjust_notenant_phases.csv");;
  green_dyn2) CTL_OPTS+=(--sm-table "${SM_TABLE:-0:128,0.5:96,1:64,2:32,3:16,4:16}");;
  green_dyn|engine_green) CTL_OPTS+=(--sm-table "${SM_TABLE:-0:128,0.5:96,1:64,2:32,3:16,4:16}");;
  green_opt) CTL_OPTS+=(--sm-table "${SM_TABLE:-0:96,0.5:64,1:32,2:16,3:8,4:8}" --shrink-window "${SHRINK_WINDOW:-10}");;   # optimisation round: conservative SM share + smoothed shrink
esac
python "$C/controller.py" "${CTL_OPTS[@]}" > "$L/${TAG}_controller.log" 2>&1 & CP=$!
sleep 3
# --- sampler
python "$KV/scripts/sampler.py" --out "${S}_sampler.csv" --duration $((DUR + 200)) --ipc "$KV_IPC" --metrics-url "http://localhost:$VLLM_PORT/metrics" & SP=$!
# --- tenant
FP=""
case $ARM in
  static_split) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --fixed-budget-gib "${STATIC_FT_GIB:-6}" > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;   # round 2: no gate (the gate starves the tenant; the static split must be compared with the un-gated adaptive scheme)
  adjust)       CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --gate --fresh > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  adjust_nogate) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  tpot_slo5|tpot_slo15) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;   # MPS, no SM cap: the tokens/s cap is the only throttle
  green_dyn2) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-dynamic --green-sms 128 --green-prebuilt > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  green32)       CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-sms 32 > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  c1_fix|c1_slow15|c1_slow5) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-sms 32 > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  c1_fine)       CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-sms 32 --pace block --slice-blocks "$SLICE" > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  c2a|c2b)       VF=$(cd "$C" && python o3_vflags.py "$([[ $ARM == c2a ]] && echo V1 || echo V2)"); echo "$ARM tenant flags: $VF"
                 if [[ $VF == NONE ]]; then echo "no variant chosen for $ARM"; else [[ $VF == *--compile* ]] && VF="$VF --no-slow-shapes"   # a compiled model cannot run the shapes with offload (they work through hooks)
                      CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-sms 32 $VF > "$L/${TAG}_ft.log" 2>&1 & FP=$!; fi;;
  green_split)   CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-remaining-of "${ENGINE_SMS:-128}" > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;   # engine on the first 128 SMs, tenant on the other 42, no switching
  green_dyn|engine_green|green_opt) CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh --green-dynamic --green-sms 32 > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;
  adjust_mps20)  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${MPS_PCT:-20} CUDA_VISIBLE_DEVICES=0 python "$C/ft_worker.py" --out "${S}_ft.csv" --fresh > "$L/${TAG}_ft.log" 2>&1 & FP=$!;;   # spatial cap instead of the gate
esac
sleep 5
# --- load through the router
python "$KV/scripts/loadgen.py" --mode prefix --rate 1 --rps-schedule "$SCHED" --duration "$DUR" --url http://localhost:8001 \
    --n-sessions 32 --prefix-tokens 6144 --unique-tokens 128 --max-tokens 128 --skew zipf --out "${S}_load.jsonl" 2>&1 | tee "$L/${TAG}_load.log" | grep -v "^{" | tail -3
# --- teardown
[[ -n $FP ]] && kill $FP 2>/dev/null; sleep 2
kill $SP $CP 2>/dev/null; sleep 1
cp /dev/shm/coloc_xfer_inf.csv "${S}_xfer_inf.csv" 2>/dev/null; cp /dev/shm/coloc_xfer_ft.csv "${S}_xfer_ft.csv" 2>/dev/null
cp "${S}_load.jsonl.t0" "${S}_t0.txt" 2>/dev/null
stop_own
if [[ ${MPS:-1} == 1 ]]; then echo quit | nvidia-cuda-mps-control; fi
python "$C/analyze_coloc.py" --arm "$TAG" --dir "$RES" --schedule "$SCHED" 2>&1 | tail -30
echo "DONE $ARM"
