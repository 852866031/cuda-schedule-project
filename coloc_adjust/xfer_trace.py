"""Shared writer for host<->GPU transfer records (one CSV per source, appended, wall clock).

Columns: ts_start, ts_end, src (inf|ft), tag, dir (h2d|d2h), bytes, gpu_ms, extra
`ts_start`/`ts_end` are time.time() on the issuing side; `gpu_ms` is the CUDA-event elapsed
time of the copy itself when available (else empty). Records from different processes share
the clock, so they can be laid on one timeline against the NVML PCIe counters.
"""
import os, threading, time


class XferTrace:
    def __init__(self, path: str, src: str):
        self.src = src
        self.lock = threading.Lock()
        new = not os.path.exists(path)
        self.f = open(path, "a", buffering=1)
        if new:
            self.f.write("ts_start,ts_end,src,tag,dir,bytes,gpu_ms,extra\n")

    def record(self, tag: str, direction: str, nbytes: int, ts_start: float, ts_end: float,
               gpu_ms=None, extra: str = ""):
        with self.lock:
            self.f.write(f"{ts_start:.6f},{ts_end:.6f},{self.src},{tag},{direction},{int(nbytes)},"
                         f"{'' if gpu_ms is None else f'{gpu_ms:.3f}'},{extra}\n")


def default_path(src: str) -> str:
    return os.environ.get("COLOC_XFER_DIR", "/dev/shm") + f"/coloc_xfer_{src}.csv"
