#!/bin/bash
# Queue: diagnose the N=4 decode-only-offload collapse, then scale fits to N=6/8.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; D=scripts/inf_multi_coloc/multi_sweep.py
$PY $D --cohort small --n 4 --arm mps --cells doff --name-suffix _r2
$PY $D --cohort small --n 4 --arm mps --cells doff --l1-gb 2 --name-suffix _l1x2
$PY $D --cohort small --n 6 8 --arm mps --cells dfits ffits
