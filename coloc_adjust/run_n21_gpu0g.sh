#!/usr/bin/env bash
# (run_n21_gpu0.sh plus the fused-kernel tenant, cvf_*; a separate file because the first batch was still running.)
# Plan of Nov 21, S3 / S4 on GPU 0: runs of the 11-minute reference schedule through run_eval8.sh, results in results_n21/.
#   run_n21_gpu0.sh <batch>      batch = a comma list of the run names below, in the order given
# The vocabulary-block tenant (blocks of 4,096, 8.54 MB per token) gets the budget that gives the tenant work of Nov 16's c1_fix:
# 3,600 tok/s x 8.54 MB = 30.7 GB/s.  step = byte ledger with one pause after the step (BWS-Budget); w12 = chunks of budget x 12 ms.
set -uo pipefail
C=$(cd "$(dirname "$0")" && pwd); export RESDIR=${RESDIR:-results_n21}; R=$C/$RESDIR; L=$C/${RESDIR/results/logs}; mkdir -p "$R" "$L"
LOG=$L/batch_gpu0.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
free_check() { local u; u=$(nvidia-smi --query-gpu=memory.used -i 0 --format=csv,noheader,nounits); if (( u > 700 )); then say "GPU 0 has $u MiB in use before a run: stopping"; exit 7; fi; }
VOC="--head vocab --vocab-block ${VB:-4096} --bpt-json $C/results_prof/bytes_per_token_prof.json --bpt-variant vocab${VB:-4096}"
VG=${VGBPS:-30.7}
# the same tenant with every block compiled on its own (6.86 MB per token; the shapes with layers in host RAM are not supported): 3,600 tok/s = 24.7 GB/s
CVOC="--compile-blocks --no-slow-shapes --head vocab --vocab-block ${VB:-4096} --bpt-json $C/results_prof/bytes_per_token_prof.json --bpt-variant cb_vocab${VB:-4096}"
CVG=${CVGBPS:-24.7}
# and with the operators of a vocabulary block as fused kernels (5.34 MB per token): 3,600 tok/s = 19.2 GB/s
FVOC="--compile-blocks --no-slow-shapes --head vocab --vocab-block ${VB:-4096} --head-fused --bpt-json $C/results_prof/bytes_per_token_prof.json --bpt-variant cb_vocab${VB:-4096}f"
FG=${FGBPS:-19.2}
FMEM="--tiers-json $C/results_prof/tiers_cb_vocab4096f.json"    # this tenant needs 2.3 GiB in its usual shape, the usual tenant 5.0 GiB
run() { local tag=$1 scheme=$2; shift 2; free_check; say "start $tag ($scheme) ${TFLAGS:-}"
  TAG=$tag "$C/run_eval8.sh" "$scheme" > "$L/${tag}_run.log" 2>&1; local rc=$?
  say "end $tag rc=$rc $(grep -c . "$R/${tag}_load.jsonl" 2>/dev/null) request rows; tenant steps $(grep -c . "$R/${tag}_ft.csv" 2>/dev/null)"; sleep 15; }
for name in ${1//,/ }; do
  case $name in
    notenant*) TFLAGS="" run "adjust_${name}" adjust_notenant;;                                   # the engine alone (any name starting with notenant)
    c1_fix*)        GBPS=50 TFLAGS="" run "$name" c1_fix;;                                                   # control: Nov 16's BWS-Budget, usual tenant, old pacing
    v_step*)        GBPS=$VG TFLAGS="$VOC --pace bytes --pace-window-ms 0"  run "$name" gen_fix;;            # BWS-Budget by bytes
    v_w12*)         GBPS=$VG TFLAGS="$VOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;            # BWS-Budget-Slicing by bytes
    v_w6*)          GBPS=$VG TFLAGS="$VOC --pace bytes --pace-window-ms 6"  run "$name" gen_fix;;
    v_slow5_step*)  TFLAGS="$VOC --pace bytes --pace-window-ms 0"  run "$name" gen_slow5;;                   # SLO-aware BWS, target 5 %
    v_slow5_w12*)   TFLAGS="$VOC --pace bytes --pace-window-ms 12" run "$name" gen_slow5;;
    v_slow15_step*) TFLAGS="$VOC --pace bytes --pace-window-ms 0"  run "$name" gen_slow15;;
    v_slow15_w12*)  TFLAGS="$VOC --pace bytes --pace-window-ms 12" run "$name" gen_slow15;;
    cv_step*)        GBPS=$CVG TFLAGS="$CVOC --pace bytes --pace-window-ms 0"  run "$name" gen_fix;;
    cv_w12*)         GBPS=$CVG TFLAGS="$CVOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cv_slow5_step*)  TFLAGS="$CVOC --pace bytes --pace-window-ms 0"  run "$name" gen_slow5;;
    cv_slow5_w12*)   TFLAGS="$CVOC --pace bytes --pace-window-ms 12" run "$name" gen_slow5;;
    cv_slow15_step*) TFLAGS="$CVOC --pace bytes --pace-window-ms 0"  run "$name" gen_slow15;;
    cv_slow15_w12*)  TFLAGS="$CVOC --pace bytes --pace-window-ms 12" run "$name" gen_slow15;;
    c2a*)             GBPS=50 TFLAGS="--bpt-variant base --compile --no-slow-shapes" run "$name" gen_fix;;      # Nov 16's best: the usual tenant compiled as a whole, pause after the step, 3,712 tok/s cap
    c1_slow15*)       TFLAGS="" run "$name" c1_slow15;;                                                         # Nov 16's SLO-aware run, usual tenant, target 15 %
    c1_slow5*)        TFLAGS="" run "$name" c1_slow5;;                                                          # Nov 16's SLO-aware run, usual tenant, target 5 %
    cvf_w12_sm16*)    GREEN_SMS=16 GBPS=$FG TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;   # the same tenant work on fewer / more SMs
    cvf_w12_sm64*)    GREEN_SMS=64 GBPS=$FG TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvf_w12_half*)    GBPS=$(python3 -c "print($FG / 2)") TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;   # half the tenant work
    cvf_w12_x2*)      GBPS=$(python3 -c "print($FG * 2)") TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;   # twice the tenant work
    cvfm_w12_sm16*)   GREEN_SMS=16 GBPS=$FG TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvfm_w12_sm64*)   GREEN_SMS=64 GBPS=$FG TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvfm_w12_half*)   GBPS=$(python3 -c "print($FG / 2)") TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvfm_w12_x2*)     GBPS=$(python3 -c "print($FG * 2)") TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvfm_slow5_step*) TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 0"  run "$name" gen_slow5;;
    cvfm_slow15_w12*) TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_slow15;;
    cvfm_slow15_step*) TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 0" run "$name" gen_slow15;;
    cvfm_step*)       GBPS=$FG TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 0"  run "$name" gen_fix;;      # m: the tenant's own GPU-memory needs per shape
    cvfm_w12*)        GBPS=$FG TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvfm_slow5_w12*)  TFLAGS="$FVOC $FMEM --pace bytes --pace-window-ms 12" run "$name" gen_slow5;;
    cvf_step*)        GBPS=$FG TFLAGS="$FVOC --pace bytes --pace-window-ms 0"  run "$name" gen_fix;;
    cvf_w12*)         GBPS=$FG TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    cvf_slow5_step*)  TFLAGS="$FVOC --pace bytes --pace-window-ms 0"  run "$name" gen_slow5;;
    cvf_slow5_w12*)   TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_slow5;;
    cvf_slow15_step*) TFLAGS="$FVOC --pace bytes --pace-window-ms 0"  run "$name" gen_slow15;;
    cvf_slow15_w12*)  TFLAGS="$FVOC --pace bytes --pace-window-ms 12" run "$name" gen_slow15;;
    b_step*)        GBPS=50 TFLAGS="--bpt-json $C/results_prof/bytes_per_token_prof.json --pace bytes --pace-window-ms 0"  run "$name" gen_fix;;   # usual tenant, byte ledger
    b_w12*)         GBPS=50 TFLAGS="--bpt-json $C/results_prof/bytes_per_token_prof.json --pace bytes --pace-window-ms 12" run "$name" gen_fix;;
    *) say "unknown run $name";;
  esac
done
say "batch done: $1"
