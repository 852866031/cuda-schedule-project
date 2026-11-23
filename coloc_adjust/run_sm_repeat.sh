#!/bin/bash
# Is "128 SMs is 2-4 % faster than all SMs" real? Repeats of three settings on one GPU, eager engine, same quick load as
# run_sm_quick.sh: full = the engine's usual context (no green context), 170 = a green context over all SMs, 128 = a green
# context of 128 SMs. Two rounds per GPU, the order reversed in the second.
#   bash run_sm_repeat.sh <gpu> <port>        results: results_sm/rep<round>_gpu<gpu>_<setting>.json
C=$(cd "$(dirname "$0")" && pwd); cd "$C"
EAGER=1 OUTPFX=rep1 bash run_sm_quick.sh "$1" "$2" "full 170 128"
EAGER=1 OUTPFX=rep2 bash run_sm_quick.sh "$1" "$2" "128 170 full"
echo "REPEAT DONE gpu $1"
