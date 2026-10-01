#!/bin/bash
# Tear down the WHOLE N-model experiment: engines, temp prefills, stores, watchdogs.
# Global pkill patterns are deliberate (mem_guard/heartbeat kill globally anyway).
# Does NOT stop the MPS daemon -- the driver owns that (it must outlive its clients).
rm -rf /tmp/multi_hb                     # stop the heartbeat watching first
for f in /tmp/multi_pids/*.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
done
sleep 3
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null
pkill -9 -f "mem_guard.sh" 2>/dev/null
pkill -9 -f "multi_heartbeat.sh" 2>/dev/null
pkill -9 -f "lmc_server_main" 2>/dev/null
pkill -9 -f "lmcache.v1.server" 2>/dev/null
pkill -9 -f "mc_scheduler.py" 2>/dev/null
rm -f /dev/shm/coloc_hp_busy
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
true
