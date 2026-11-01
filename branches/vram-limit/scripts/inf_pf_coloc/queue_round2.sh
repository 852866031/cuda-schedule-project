#!/bin/bash
# Round 2 (2026-10-13), predictions in DECISIONS_PF_COLOC.md. Strictly serial; every
# launch through pfc_launch.sh (mem_guard, heartbeat, RAM gate).
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_pf_coloc/pfc_sweep.py
export OMP_NUM_THREADS=8
step() { echo "### $(date '+%H:%M') $*"; }
# R2a: is the baseline's 176 ms TTFT the smaller prefill grant? 8B alone at prefill 0.80.
step R2a; $PY $D --cells base --prefill-util 0.80 --name-suffix _pf80
# R2b: 3B rate sweep, P1 and P2, alone and beside the 8B. Request counts keep the 3B
# window at ~150 s, the length of the 8B's (300 requests at 2 QPS).
step R2b
$PY $D --cells s1 s2 p1 p2 --sm-qps 0.5 --sm-requests 75 --sm-store-cap-gb 2
$PY $D --cells s1 s2 p1 p2 --sm-qps 1 --sm-requests 150 --sm-store-cap-gb 2
# R2c: P3 with the 8B's pinned L1 pools at 4 GiB each (6 before) -> ~4 GB of host RAM
# back. A changed 8B config, so it gets its own baseline first.
step R2c; export LMC_LOCAL_CPU_GB=4
$PY $D --cells base p3 --sm-store-cap-gb 2 --name-suffix _l1x4
step done
