#!/bin/bash
# Smoke test for Tally inside the wzhao18/tally:latest container.
# Runs iox-roudi + tally_server, then a simple test client through LD_PRELOAD.
# Usage (inside container): bash /workspace/scripts/tally_smoke_test.sh [test_binary]
set -x

export TALLY_HOME=${TALLY_HOME:-/home/tally-bench/tally}
cd $TALLY_HOME

TEST_BIN=${1:-./build/tests/elementwise}

# Prefer host-mounted CUDA (sm_120-capable nvcc) if present
if [ -d /host-cuda/bin ]; then
    export PATH=/host-cuda/bin:$PATH
    echo "Using nvcc: $(which nvcc)"
    nvcc --version | tail -1
fi

# Clean up any previous runs
pkill -f tally_server 2>/dev/null
pkill -f iox-roudi 2>/dev/null
sleep 1

# 1. Start iceoryx shared-memory daemon (with Tally's mempool config: needs large chunks)
$TALLY_HOME/build/iox-roudi -c $TALLY_HOME/config/roudi_config.toml > /tmp/iox.log 2>&1 &
sleep 3

# 2. Start tally server with the priority scheduler
SCHEDULER_POLICY=${SCHEDULER_POLICY:-NAIVE} ./scripts/start_server.sh > /tmp/tally_server.log 2>&1 &
sleep 5

# 3. Run client workload through Tally
PRIORITY=1 ./scripts/start_client.sh $TEST_BIN
CLIENT_EXIT=$?

echo "=== client exit code: $CLIENT_EXIT ==="
echo "=== server log (last 30 lines) ==="
tail -30 /tmp/tally_server.log

./scripts/kill_server.sh 2>/dev/null
./scripts/kill_iox.sh 2>/dev/null
exit $CLIENT_EXIT
