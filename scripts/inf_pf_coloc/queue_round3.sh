#!/bin/bash
# Round 3 (2026-10-13): the async-save patch (--sm-async-save) against the sync cells of
# rounds 1-2. Predictions in DECISIONS_PF_COLOC.md section D. Strictly serial.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_pf_coloc/pfc_sweep.py
export OMP_NUM_THREADS=8
step() { echo "### $(date '+%H:%M') $*"; }
# sync s3 again, only to record the reference texts of the KV-content check (_v)
step R3a; $PY $D --cells s3 --sm-store-cap-gb 2 --name-suffix _v
step R3b; $PY $D --cells s2 s3 p2 --sm-store-cap-gb 2 --sm-async-save
# P3 beside the 8B needs the 4 GiB L1 pools (round 2); swap-backed, see the log
step R3c; LMC_LOCAL_CPU_GB=4 $PY $D --cells p3 --sm-store-cap-gb 2 --sm-async-save --name-suffix _l1x4
step done
