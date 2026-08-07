#!/usr/bin/env python
"""Workload registry: loads every workload from workload_defs/ (one per file).

Each non-underscore .py file in workload_defs/ (except base.py) must export a
module-level ``WORKLOAD`` class (see workload_defs/base.py for the contract);
this module collects them into ``WORKLOADS = {name: class}``.

Standalone usage (single-workload verification):

    python colocator/demo/workloads.py --list
    python colocator/demo/workloads.py --client dram --check
    python colocator/demo/workloads.py --client fma --iters 20
"""

import argparse
import importlib.util
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFS = os.path.join(_HERE, "workload_defs")
sys.path.insert(0, _DEFS)                              # `from base import ...`
sys.path.insert(0, os.path.join(_HERE, "..", "ext"))   # `import colocator_kernels`


def _load_all():
    registry = {}
    for fname in sorted(os.listdir(_DEFS)):
        if not fname.endswith(".py") or fname.startswith("_") or fname == "base.py":
            continue
        spec = importlib.util.spec_from_file_location(
            f"workload_defs.{fname[:-3]}", os.path.join(_DEFS, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, "WORKLOAD", None)
        if cls is None:
            print(f"[workloads] warning: {fname} defines no WORKLOAD, skipped")
            continue
        registry[cls.name] = cls
    return registry


WORKLOADS = _load_all()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list registered workloads")
    ap.add_argument("--client", choices=sorted(WORKLOADS))
    ap.add_argument("--iters", type=int, default=None, help="override (default: calibrate ~500ms)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.list or not args.client:
        for name, cls in sorted(WORKLOADS.items()):
            print(f"{name:>12}  {cls.__module__.split('.')[-1]}.py")
        return

    client = WORKLOADS[args.client](iters=args.iters)
    client.setup()
    client.calibrate()
    if args.check:
        ok = client.check()
        print(f"[{client.name}] check: {'OK' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)

    import torch
    torch.cuda.synchronize()
    lat = []
    for _ in range(client.iters):
        t0 = time.perf_counter()
        client.run_iter()   # pipelined (2 in flight)
        lat.append(time.perf_counter() - t0)
    client.drain()
    torch.cuda.synchronize()
    lat_ms = sorted(x * 1e3 for x in lat)
    n = len(lat_ms)
    print(f"[{client.name}] {n} iters: p50={lat_ms[n // 2]:.2f} ms  "
          f"p95={lat_ms[min(n - 1, n * 95 // 100)]:.2f} ms  total={sum(lat_ms) / 1e3:.2f} s")
    print("---- describe() ----")
    print(client.describe())


if __name__ == "__main__":
    main()
