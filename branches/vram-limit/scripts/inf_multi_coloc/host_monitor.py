#!/usr/bin/env python3
"""Per-second host-side sampler: CPU cores used by each process ROLE, plus RAM/swap.

DCGM covers the GPU; this covers the host, where the decode-only-offload collapse lives
(GPU1 ~idle while host CPU hits 93%). It answers "what moves first" in a time series:
the stores (GIL-bound LMCache servers), the EngineCores, the API servers, the clients.

    ts,role,cores          role in {store_<i>, enginecore, api, client, other_total,
                                    avail_mb, pswpin}   (avail_mb/pswpin carry values in
                                    the 'cores' column)

    host_monitor.py out.csv     # runs until killed
"""

import os
import re
import sys
import time
from pathlib import Path

TICK = os.sysconf("SC_CLK_TCK")
STORE_PORT = re.compile(r"lmc_server_main\.py\S*\s+127\.0\.0\.1\s+86(\d\d)")


def role_of(cmd):
    if "lmc_server_main" in cmd:
        m = STORE_PORT.search(cmd)
        return f"store_{int(m.group(1))}" if m else "store_?"
    if "VLLM::EngineCore" in cmd:
        return "enginecore"
    if "vllm.entrypoints.cli.main serve" in cmd:
        return "api"
    if "model_client_runner" in cmd:
        return "client"
    return None


def snapshot():
    """{pid: (role, jiffies)} for experiment processes; total busy jiffies of the box."""
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            role = role_of(cmd)
            if role is None:
                comm = (d / "comm").read_text().strip()
                role = "enginecore" if comm.startswith("VLLM::EngineCor") else None
            if role is None:
                continue
            f = (d / "stat").read_text().rsplit(")", 1)[1].split()
            out[int(d.name)] = (role, int(f[11]) + int(f[12]))   # utime + stime
        except (OSError, IndexError, ValueError):
            continue
    cpu = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return out, sum(cpu) - cpu[3] - cpu[4]


def meminfo():
    avail = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 1024
    pswpin = 0
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pswpin "):
            pswpin = int(line.split()[1])
    return avail, pswpin


def main():
    out = open(sys.argv[1], "w", buffering=1)
    out.write("ts,role,cores\n")
    prev, prev_busy, t_prev = *snapshot(), time.time()
    while True:
        time.sleep(1.0)
        cur, busy = snapshot()
        now = time.time()
        dt = max(1e-3, now - t_prev)
        agg = {}
        for pid, (role, j) in cur.items():
            if pid in prev:
                agg[role] = agg.get(role, 0.0) + (j - prev[pid][1]) / TICK / dt
        ours = sum(agg.values())
        agg["other_total"] = max(0.0, (busy - prev_busy) / TICK / dt - ours)
        avail, pswpin = meminfo()
        for role, cores in sorted(agg.items()):
            out.write(f"{now:.3f},{role},{cores:.3f}\n")
        out.write(f"{now:.3f},avail_mb,{avail}\n{now:.3f},pswpin,{pswpin}\n")
        prev, prev_busy, t_prev = cur, busy, now


if __name__ == "__main__":
    main()
