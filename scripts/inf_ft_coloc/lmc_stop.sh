#!/bin/bash
# Tear down the lmcache arm: the shared stop handles engines/proxy/watchdogs, then the
# cache server (which the simple arm does not have) is killed explicitly.
[ -f /tmp/disagg_lmcserver.pid ] && kill "$(cat /tmp/disagg_lmcserver.pid)" 2>/dev/null
rm -f /tmp/disagg_lmcserver.pid
bash "$(dirname "$0")/disagg_stop.sh"
pkill -9 -f "lmcache.v1.server" 2>/dev/null
pkill -9 -f "lmc_server_main" 2>/dev/null
true
