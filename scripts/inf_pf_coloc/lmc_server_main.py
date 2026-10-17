#!/usr/bin/env python3
"""LMCache's stock cache server, with SO_REUSEADDR and an optional size cap.

Upstream's LMCacheServer binds without SO_REUSEADDR, so a server killed while holding
connections leaves the port in TIME_WAIT for ~60 s and the next bind dies with
EADDRINUSE -- fatal for a sweep that tears the stack down and relaunches within seconds.
This wrapper sets the option on every socket before bind and then defers entirely to the
stock __main__. Usage is identical:  python lmc_server_main.py <host> <port> cpu

LMC_STORE_CAP_GB=<x> (this study only): the stock store has no evictor, and the small
model's prompts are fresh on every request (0.21 GiB of KV each for a 3B), so its store
would grow without bound. With the cap set, put() evicts the OLDEST chunks until the store
is back under the cap -- the store becomes a bounded transit buffer. A get() for a chunk
that was evicted is counted (miss_after_evict): in the split case that means the decode
leg came too late and silently re-prefilled, so the driver must see a zero there.
A "STORE ..." stats line is printed every 100 puts.
"""

import os
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

CAP = int(float(os.environ.get("LMC_STORE_CAP_GB", "0")) * (1 << 30))
if CAP > 0:
    from lmcache.v1.server.storage_backend.local_backend import LMSLocalBackend
    from lmcache.v1.server.utils import LMSMemoryObj

    st = {"bytes": 0, "puts": 0, "gets": 0, "evicted": 0, "miss_after_evict": 0,
          "peak_bytes": 0}
    gone = set()

    def _put(self, client_meta, kv_chunk_bytes):
        with self.lock:
            old = self.dict.pop(client_meta.key, None)
            if old is not None:
                st["bytes"] -= len(old.data)
            self.dict[client_meta.key] = LMSMemoryObj(
                kv_chunk_bytes, client_meta.length, client_meta.fmt,
                client_meta.dtype, client_meta.shape)
            st["bytes"] += len(kv_chunk_bytes)
            gone.discard(client_meta.key)
            while st["bytes"] > CAP and len(self.dict) > 1:
                k, v = self.dict.popitem(last=False)
                st["bytes"] -= len(v.data)
                st["evicted"] += 1
                gone.add(k)
            st["peak_bytes"] = max(st["peak_bytes"], st["bytes"])
            st["puts"] += 1
            if st["puts"] % 100 == 0:
                print("STORE " + " ".join(f"{k}={v}" for k, v in st.items()), flush=True)

    def _get(self, key):
        with self.lock:
            st["gets"] += 1
            obj = self.dict.get(key, None)
            if obj is None and key in gone:
                st["miss_after_evict"] += 1
                print("STORE " + " ".join(f"{k}={v}" for k, v in st.items()), flush=True)
            return obj

    LMSLocalBackend.put = _put
    LMSLocalBackend.get = _get
    print(f"store cap {CAP / (1 << 30):.2f} GiB (oldest-first eviction)", flush=True)

sys.argv = ["lmcache.v1.server", *sys.argv[1:]]
runpy.run_module("lmcache.v1.server", run_name="__main__")
