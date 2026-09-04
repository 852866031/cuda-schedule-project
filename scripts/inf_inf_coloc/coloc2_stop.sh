#!/bin/bash
# Tear down the WHOLE experiment: both stacks, both proxies, both cache servers, the
# watchdogs. The global pkill patterns are deliberate -- there is no supported way to
# stop only one side, because mem_guard/heartbeat kill globally anyway.
for f in /tmp/disagg_proxy.pid /tmp/disagg_prefill.pid /tmp/disagg_decode.pid \
         /tmp/disagg_lmcserver.pid /tmp/disagg_memguard.pid /tmp/disagg_heartbeat.pid \
         /tmp/qwen_proxy.pid /tmp/qwen_prefill.pid /tmp/qwen_decode.pid \
         /tmp/qwen_lmcserver.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
done
sleep 3
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null
pkill -9 -f "mem_guard.sh" 2>/dev/null
# Proxy pidfiles get overwritten by every relaunch; an orphan silently serves stale
# code on :8000/:8001. Kill by name too.
pkill -9 -f "lmc_proxy.py" 2>/dev/null
pkill -9 -f "lmcache.v1.server" 2>/dev/null
pkill -9 -f "lmc_server_main" 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
true
