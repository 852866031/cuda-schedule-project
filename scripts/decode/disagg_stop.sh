#!/bin/bash
# Tear down both instances and the router. The engine child outlives its parent and keeps
# VRAM, so it must be killed explicitly (see README).
for f in /tmp/disagg_proxy.pid /tmp/disagg_prefill.pid /tmp/disagg_decode.pid; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
done
sleep 3
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm.entrypoints.cli.main serve" 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
