#!/usr/bin/env python
"""Demo driver: run the two workloads in one of three modes and record results.

    seq        clients run back-to-back, no colocator (solo baselines)
    streams    client threads on plain torch.cuda.Streams, no colocator
    colocated  through the colocator: LD_PRELOAD interception -> per-client
               queues -> FCFS scheduler -> per-client colocator streams
    all        seq, then streams, then colocated

Examples:

    python colocator/demo/run_demo.py --mode colocated --clients latency,throughput --out runs/x
    python colocator/demo/run_demo.py --mode all --out runs/full

Writes <out>/<mode>/results.json (per-client per-iteration latencies) and, in
colocated mode, <out>/<mode>/issue_log.csv from the scheduler.

In colocated mode each client runs inside its own torch.cuda.Stream context.
This is a correctness requirement, not decoration: it gives each client its
own caching-allocator pool, so a block freed by client A can't be handed to
client B while A's kernels (running on A's colocator stream, which torch
can't see) still use it.
"""

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from colocator import Colocator, ensure_managed_env  # noqa: E402

WARMUP = 3


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["seq", "streams", "colocated", "all"], required=True)
    ap.add_argument("--clients", default="latency,throughput",
                    help="comma-separated workload names (see workloads.py)")
    ap.add_argument("--iters", default=None,
                    help="timed iterations per client: single int, comma list matching "
                         "--clients, or omit to auto-calibrate each client to ~500 ms")
    ap.add_argument("--priorities", default=None,
                    help="comma-separated colocator stream priorities, e.g. '0,-1'")
    ap.add_argument("--policy", choices=["fcfs", "throttle"], default="fcfs",
                    help="scheduling policy (colocated mode; default: fcfs = no throttle)")
    ap.add_argument("--throttle", type=int, default=2,
                    help="throttle policy: hold a client once it has this many "
                         "unfinished ops in flight (in-flight never exceeds it)")
    ap.add_argument("--sm-limit", type=int, default=0,
                    help="colocated mode: route cuBLAS ops to a green-context "
                         "stream capped to this percent of the GPU's SMs "
                         "(0 = off; ops stay ordered via event chaining)")
    ap.add_argument("--check", action="store_true", default=True,
                    help="verify results vs CPU reference (default on)")
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--observer", choices=["on", "off"], default="off",
                    help="(Phase 4) attach the CUPTI observer in colocated mode")
    ap.add_argument("--out", default="runs/latest", help="output directory")
    return ap.parse_args()


def make_clients(args):
    from workloads import WORKLOADS
    names = args.clients.split(",")
    if args.iters is None:
        iters = [None] * len(names)          # calibrate to ~500 ms
    else:
        parts = str(args.iters).split(",")
        iters = [int(parts[i]) if len(parts) > 1 else int(parts[0])
                 for i in range(len(names))]
    return [WORKLOADS[n](iters=it) for n, it in zip(names, iters)]


def client_body(client, check, barrier=None, use_torch_stream=False):
    """setup -> optional check -> warmup -> timed loop. Runs on the calling thread."""
    import torch

    def body():
        client.setup()
        client.calibrate()  # no-op when iters were forced
        check_ok = client.check() if check else None
        for _ in range(WARMUP):
            client.run_iter()
        client.drain()      # cross the barrier with an empty pipeline
        if barrier is not None:
            barrier.wait()  # start timed sections together
        # Same clock domain as the interceptor/scheduler logs, so the replay
        # can place the barrier on the trace timeline.
        barrier_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        lat = []
        t_loop0 = time.perf_counter()
        for _ in range(client.iters):
            t0 = time.perf_counter()
            client.run_iter()   # pipelined: admission time at steady state
            lat.append(time.perf_counter() - t0)
        client.drain()
        loop_s = time.perf_counter() - t_loop0
        return {"name": client.name, "iters": client.iters, "check": check_ok,
                "desc": client.describe(), "marks": client.marks,
                "barrier_ns": barrier_ns, "lat_s": lat, "loop_s": loop_s}

    if use_torch_stream:
        with torch.cuda.stream(torch.cuda.Stream()):
            return body()
    return body()


def run_seq(args, out_dir):
    results = []
    for client in make_clients(args):
        results.append(client_body(client, args.check))
        import torch
        torch.cuda.synchronize()
    return {"mode": "seq", "clients": results,
            "tids": {"clients": [threading.get_native_id()] * len(results)}}


def run_streams(args, out_dir):
    clients = make_clients(args)
    barrier = threading.Barrier(len(clients))
    results = [None] * len(clients)
    errs = [None] * len(clients)
    tids = [None] * len(clients)

    def worker(i, c):
        try:
            tids[i] = threading.get_native_id()
            results[i] = client_body(c, args.check, barrier, use_torch_stream=True)
        except BaseException as e:  # noqa: BLE001
            errs[i] = e
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errs:
        if e:
            raise e
    return {"mode": "streams", "clients": results, "tids": {"clients": tids}}


def run_colocated(args, out_dir):
    # Policy and SM limit are read by libsched at col_setup time from the env.
    os.environ["COLOCATOR_POLICY"] = args.policy
    os.environ["COLOCATOR_THROTTLE"] = str(args.throttle)
    os.environ["COLOCATOR_SM_LIMIT"] = str(args.sm_limit)
    clients = make_clients(args)
    bad = [c.name for c in clients if not c.colocatable]
    if bad:
        sys.exit(f"[run_demo] refusing to colocate {bad}: these workloads launch "
                 f"kernels the interceptor cannot capture (cuBLAS static-runtime/"
                 f"driver-API), which would race across streams and corrupt results. "
                 f"Use --mode seq --observer on instead.")
    n = len(clients)
    priorities = [int(x) for x in args.priorities.split(",")] if args.priorities else [0] * n
    col = Colocator(n, priorities)
    barrier = threading.Barrier(n)

    fns = [
        (lambda c=c: client_body(c, args.check, barrier, use_torch_stream=True))
        for c in clients
    ]
    results = col.launch(fns)
    log_path = os.path.join(out_dir, "issue_log.csv")
    n_ops = col.dump_issue_log(log_path)
    print(f"[run_demo] issue log: {n_ops} ops -> {log_path}")
    print(f"[run_demo] accounting: {col.stats}")

    return {"mode": "colocated", "priorities": priorities, "clients": results,
            "policy": args.policy + (f"(cap {args.throttle})" if args.policy == "throttle" else ""),
            "sm_limit": args.sm_limit,
            "issue_log": log_path, "accounting": col.stats,
            "tids": {"sched": col.sched_tid, "clients": col.client_tids}}


def summarize(res):
    for c in res["clients"]:
        lat = sorted(c["lat_s"])
        n = len(lat)
        thr = c["iters"] / c["loop_s"]
        print(f"[{res['mode']:>9}] {c['name']:>10}: check={c['check']} "
              f"p50={lat[n // 2] * 1e3:8.2f} ms  p95={lat[min(n - 1, n * 95 // 100)] * 1e3:8.2f} ms  "
              f"{thr:7.1f} iters/s")


def main():
    args = parse_args()
    modes = ["seq", "streams", "colocated"] if args.mode == "all" else [args.mode]

    # colocated needs the managed env; must re-exec before torch is imported.
    if "colocated" in modes:
        ensure_managed_env()

    observer = None
    if args.observer == "on":
        # Init once, BEFORE any CUDA work, so every mode's kernels are
        # captured. Dumps are cumulative; each mode's records are isolated by
        # the [window] timestamps below (same CLOCK_MONOTONIC_RAW domain as
        # the observer's calibration samples).
        import ctypes
        from colocator import LIBCOL
        libobs = os.path.join(os.path.dirname(LIBCOL), "libobserver.so")
        # RTLD_GLOBAL so libsched can dlsym the obs_live_* feed at col_setup.
        observer = ctypes.CDLL(libobs, mode=ctypes.RTLD_GLOBAL)
        observer.obs_init.restype = ctypes.c_int
        observer.obs_dump.argtypes = [ctypes.c_char_p]
        observer.obs_dump.restype = ctypes.c_long
        assert observer.obs_init() == 0

    runners = {"seq": run_seq, "streams": run_streams, "colocated": run_colocated}
    for mode in modes:
        out_dir = os.path.join(args.out, mode)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        res = runners[mode](args, out_dir)
        res["window"] = [t0, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)]
        if observer is not None:
            nk = observer.obs_dump(out_dir.encode())
            res["observer"] = True
            print(f"[run_demo] observer: {nk} kernel records so far -> {out_dir}/obs_*.csv")
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(res, f, indent=1)
        summarize(res)
        print(f"[run_demo] wrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
