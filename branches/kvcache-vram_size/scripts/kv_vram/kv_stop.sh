#!/bin/bash
# Tear down the WHOLE experiment: engines, router, store, watchdogs. Global pkill
# patterns are deliberate (mem_guard/heartbeat kill globally anyway).
rm -rf /tmp/kvv_hb                       # stop the heartbeat watching first
for f in /tmp/kvv_pids/*.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
done
sleep 3
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null
pkill -9 -f "mem_guard.sh" 2>/dev/null
pkill -9 -f "kv_heartbeat.sh" 2>/dev/null
pkill -9 -f "lmc_proxy.py" 2>/dev/null
pkill -9 -f "lmc_server_main" 2>/dev/null
pkill -9 -f "lmcache.v1.server" 2>/dev/null
pkill -9 -f "lmcache.v1.multiprocess.server" 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
true
