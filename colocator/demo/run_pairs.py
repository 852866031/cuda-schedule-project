#!/usr/bin/env python
"""Run-all driver: colocate every pair of workloads, one run per pair.

    python colocator/demo/run_pairs.py                       # all pairs -> runs/
    python colocator/demo/run_pairs.py --workloads l2,dram,fma
    python colocator/demo/run_pairs.py --priorities=-5,0 --policy throttle

Steps:
 1. SOLO phase: one `run_demo.py --mode seq` process runs every workload
    back-to-back alone (auto-calibrated to ~500 ms each) -> <out>/solo/seq.
    This is the overhead-panel baseline AND fixes each workload's iteration
    count deterministically (calibrating inside a colocated run would measure
    contaminated iteration times).
 2. PAIR phase: for each unordered pair (a, b), a fresh process runs
    `run_demo.py --mode colocated --clients a,b --iters ia,ib --observer on`
    -> <out>/<config>/<a>__<b>/colocated, where <config> encodes the throttle
    setting: `throttle_none` (fcfs, the default) or `throttle_<N>`
    (`--policy throttle --throttle N`). Run the script once per throttle
    setting to build comparable matrices side by side. Fresh process per
    pair keeps the CUDA context, allocator and colocator state clean.

Afterwards: python colocator/visualizer/export_all.py  -> replays/ incl. the
pair-selector navigation.
"""

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DEMO = os.path.join(HERE, "run_demo.py")
TARGET_S = 0.5


def run(cmd):
    print(f"[run_pairs] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[run_pairs] FAILED (exit {r.returncode})")
    return r.returncode == 0


def main():
    sys.path.insert(0, HERE)
    from workloads import WORKLOADS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workloads", default=None,
                    help="comma list (default: every registered workload)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--priorities", default=None, help="forwarded to run_demo")
    ap.add_argument("--policy", default=None, help="forwarded to run_demo")
    ap.add_argument("--throttle", default=None, help="forwarded to run_demo")
    ap.add_argument("--sm-limit", default=None,
                    help="forwarded to run_demo (green-context SM cap, percent)")
    ap.add_argument("--skip-solo", action="store_true",
                    help="reuse an existing <out>/solo baseline")
    args = ap.parse_args()

    # Default set = every colocatable workload (llmdecode & co. are excluded:
    # their cuBLAS kernels bypass interception — colocating them is unsafe).
    names = sorted(args.workloads.split(",") if args.workloads
                   else (n for n, c in WORKLOADS.items() if c.colocatable))
    for n in names:
        assert n in WORKLOADS, f"unknown workload {n}"
        assert WORKLOADS[n].colocatable, \
            f"{n} is not colocatable (uninterceptable kernels) — solo seq runs only"
    py = sys.executable

    # 1. solo baseline (also determines per-workload iteration counts)
    solo_res = os.path.join(args.out, "solo", "seq", "results.json")
    if not (args.skip_solo and os.path.exists(solo_res)):
        ok = run([py, RUN_DEMO, "--mode", "seq", "--clients", ",".join(names),
                  "--out", os.path.join(args.out, "solo")])
        if not ok:
            sys.exit(1)

    with open(solo_res) as f:
        solo = {c["name"]: c for c in json.load(f)["clients"]}
    iters = {n: max(3, min(5000, round(TARGET_S / statistics.median(solo[n]["lat_s"]))))
             for n in names}
    print(f"[run_pairs] iteration counts (~{TARGET_S * 1e3:.0f} ms each): {iters}")

    # 2. every unordered pair, one fresh process each, grouped by throttle cfg
    extra = []
    if args.priorities:
        extra += [f"--priorities={args.priorities}"]
    if args.policy:
        extra += ["--policy", args.policy]
    if args.throttle:
        extra += ["--throttle", str(args.throttle)]
    if args.sm_limit:
        extra += ["--sm-limit", str(args.sm_limit)]
    cfg = (f"smlimit_{args.sm_limit}" if args.sm_limit
           else f"throttle_{args.throttle or 2}" if args.policy == "throttle"
           else "throttle_none")
    print(f"[run_pairs] config: {cfg}")

    # with_replacement: includes self-pairs (fma+fma, dram+dram, ...)
    pairs = list(itertools.combinations_with_replacement(names, 2))
    failed = []
    for i, (a, b) in enumerate(pairs):
        out = os.path.join(args.out, cfg, f"{a}__{b}")
        print(f"[run_pairs] ---- pair {i + 1}/{len(pairs)}: {a} + {b} -> {out}")
        ok = run([py, RUN_DEMO, "--mode", "colocated", "--clients", f"{a},{b}",
                  "--iters", f"{iters[a]},{iters[b]}", "--observer", "on",
                  "--out", out] + extra)
        if not ok:
            failed.append(f"{a}__{b}")

    print(f"[run_pairs] done: {len(pairs) - len(failed)}/{len(pairs)} pairs"
          + (f", FAILED: {', '.join(failed)}" if failed else ""))
    print("[run_pairs] next: python colocator/visualizer/export_all.py")


if __name__ == "__main__":
    main()
