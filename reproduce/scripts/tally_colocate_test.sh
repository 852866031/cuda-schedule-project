#!/bin/bash
# Tally co-location experiment: HP inference + BE training under the PRIORITY scheduler.
# Usage (inside container): bash /workspace/scripts/tally_colocate_test.sh
set -e

export TALLY_HOME=${TALLY_HOME:-/home/tally-bench/tally}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd $TALLY_HOME

# sm_120-capable nvcc from host mount
if [ -d /host-cuda/bin ]; then
    export PATH=/host-cuda/bin:$PATH
fi

cleanup() {
    pkill -f tally_server 2>/dev/null || true
    pkill -f iox-roudi 2>/dev/null || true
}
cleanup; sleep 1

echo "=== Phase 0a: HP alone, native (no Tally) ==="
python3 /workspace/scripts/hp_client.py 200

echo "=== Phase 0b: HP + BE co-located, native (hardware scheduler, no Tally) ==="
python3 /workspace/scripts/be_client.py 40 > /tmp/be_native.log 2>&1 &
BE_NATIVE_PID=$!
sleep 10
python3 /workspace/scripts/hp_client.py 200
wait $BE_NATIVE_PID || true
grep "BE RESULTS" /tmp/be_native.log || tail -3 /tmp/be_native.log

$TALLY_HOME/build/iox-roudi -c $TALLY_HOME/config/roudi_config.toml > /tmp/iox.log 2>&1 &
sleep 3

SCHEDULER_POLICY=PRIORITY ./scripts/start_server.sh > /tmp/tally_server.log 2>&1 &
sleep 5

echo "=== Phase 1: HP alone under Tally ==="
PRIORITY=2 ./scripts/start_client.sh python3 /workspace/scripts/hp_client.py 200

echo "=== Phase 2: HP + BE co-located under Tally (PRIORITY scheduler) ==="
PRIORITY=1 ./scripts/start_client.sh python3 /workspace/scripts/be_client.py 60 > /tmp/be_colo.log 2>&1 &
BE_PID=$!
sleep 15   # let BE warm up and saturate the GPU
PRIORITY=2 ./scripts/start_client.sh python3 /workspace/scripts/hp_client.py 200
wait $BE_PID || true
grep "BE RESULTS" /tmp/be_colo.log || tail -5 /tmp/be_colo.log

echo "=== server log tail ==="
tail -5 /tmp/tally_server.log
cleanup
