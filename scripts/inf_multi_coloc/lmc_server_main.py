#!/usr/bin/env python3
"""LMCache's stock cache server, with SO_REUSEADDR.

Upstream's LMCacheServer binds without SO_REUSEADDR, so a server killed while holding
connections leaves the port in TIME_WAIT for ~60 s and the next bind dies with
EADDRINUSE -- fatal for a sweep that tears the stack down and relaunches within seconds.
This wrapper sets the option on every socket before bind and then defers entirely to the
stock __main__. Usage is identical:  python lmc_server_main.py <host> <port> cpu
"""

import runpy
import socket
import sys

_orig_bind = socket.socket.bind


def _bind(self, addr):
    try:
        self.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass
    return _orig_bind(self, addr)


socket.socket.bind = _bind

sys.argv = ["lmcache.v1.server", *sys.argv[1:]]
runpy.run_module("lmcache.v1.server", run_name="__main__")
