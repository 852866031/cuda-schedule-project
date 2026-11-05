#!/usr/bin/env bash
# Start the LMCache multi-process server with a small pinned-DRAM L1 (host RAM is 60 GiB;
# large pinned allocations have frozen this machine before, so stay at a few GiB).
#   lmcache_server.sh <tag> [--l1-gb N] [--transfer lmcache_driven|engine_driven|auto]
set -euo pipefail
source "$(dirname "$0")/env.sh"
TAG=$1; shift
L1=6; TM=auto
while [[ $# -gt 0 ]]; do
  case $1 in
    --l1-gb) L1=$2; shift 2;;
    --transfer) TM=$2; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG="$LOGS/lmcache_${TAG}.log"
echo "# $(date -Is) l1=${L1}GB transfer=$TM" > "$LOG"
nohup lmcache server --host localhost --port "$LMC_PORT" --l1-size-gb "$L1" --l1-init-size-gb 2 --eviction-policy LRU \
    --chunk-size 256 --supported-transfer-mode "$TM" --http-port 8181 >> "$LOG" 2>&1 &
echo $! > "$LOGS/lmcache_${TAG}.pid"
echo "started pid $(cat "$LOGS/lmcache_${TAG}.pid") log $LOG"
