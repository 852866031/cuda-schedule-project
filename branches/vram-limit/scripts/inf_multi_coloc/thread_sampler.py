#!/usr/bin/env python3
"""Per-thread CPU sampler for the vLLM EngineCore processes (no ptrace, no perf).

The N=4 offload collapse pins all 32 cores inside the EngineCores; the report's reading is
OpenMP spin-wait oversubscription. Stack sampling is unavailable on this box
(ptrace_scope=1, perf_event_paranoid=4, no sudo), so this reads what /proc exposes to the
same user, every POLL seconds, per EngineCore process:

  ts, pid, n_threads, n_busy, user_cores, sys_cores, invol_cs_s, vol_cs_s

  n_busy       threads that used > 0.5 core over the interval
  user/sys     CPU cores spent in user vs kernel mode (all threads)
  invol_cs_s   involuntary context switches per second (preempted while runnable --
               the spinning signature) vs vol_cs_s (blocked/yielded voluntarily)

    thread_sampler.py out.csv [poll_seconds]      # runs until killed
"""

import sys
import time
from pathlib import Path

TICK = 100.0  # SC_CLK_TCK


def engine_pids():
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if (d / "comm").read_text().strip().startswith("VLLM::EngineCor"):
                out.append(int(d.name))
        except OSError:
            pass
    return out


def threads(pid):
    """{tid: (utime, stime, vol_cs, invol_cs)}"""
    out = {}
    for t in Path(f"/proc/{pid}/task").iterdir():
        try:
            f = (t / "stat").read_text().rsplit(")", 1)[1].split()
            vol = invol = 0
            for line in (t / "status").read_text().splitlines():
                if line.startswith("voluntary_ctxt_switches"):
                    vol = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches"):
                    invol = int(line.split()[1])
            out[int(t.name)] = (int(f[11]), int(f[12]), vol, invol)
        except (OSError, IndexError, ValueError):
            continue
    return out


def main():
    out = open(sys.argv[1], "w", buffering=1)
    poll = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    out.write("ts,pid,n_threads,n_busy,user_cores,sys_cores,invol_cs_s,vol_cs_s\n")
    prev, t_prev = {}, time.time()
    while True:
        time.sleep(poll)
        now = time.time()
        dt = max(1e-3, now - t_prev)
        cur = {}
        for pid in engine_pids():
            try:
                cur[pid] = threads(pid)
            except OSError:
                continue
        for pid, th in cur.items():
            old = prev.get(pid)
            if not old:
                continue
            busy = 0
            u = s = vol = invol = 0.0
            for tid, (ut, st, v, iv) in th.items():
                if tid not in old:
                    continue
                o = old[tid]
                du, ds = (ut - o[0]) / TICK, (st - o[1]) / TICK
                u += du; s += ds
                vol += v - o[2]; invol += iv - o[3]
                if (du + ds) / dt > 0.5:
                    busy += 1
            out.write(f"{now:.3f},{pid},{len(th)},{busy},{u / dt:.3f},{s / dt:.3f},"
                      f"{invol / dt:.1f},{vol / dt:.1f}\n")
        prev, t_prev = cur, now


if __name__ == "__main__":
    main()
