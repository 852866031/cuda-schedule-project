#!/usr/bin/env python3
"""Load generator for the small (3B) model, run as a SEPARATE PROCESS by pfp_sweep.py.

Every request is a FRESH prompt of --prompt-len tokens (a 16-token stub prefix + a unique
suffix -- the multi-model study's prefill-only workload), so every request is a full
prefill. --max-tokens 1 = prefill only (P1/P2: TTFT is the whole request, no TPOT);
128 = prefill + decode through the 3B's own router (P3).

The 8B client runs in-process in the driver; this one is its own process so the two
streaming loops cannot contend for a GIL. The workload is rebuilt deterministically from
the args (same seed -> same prompts and arrival times).
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import client as client_mod            # noqa: E402
from workload import build_workload    # noqa: E402

STUB = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=6144)
    ap.add_argument("--kv-bytes", type=int, required=True)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--start-at", type=float, default=0.0,
                    help="epoch time to start the load (aligns it with the 8B client)")
    ap.add_argument("--phase", choices=["warmup", "measure"], required=True)
    ap.add_argument("--warmup-requests", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.phase == "warmup":     # different seed: warmup prompts never recur in measure
        n, qps, seed = args.warmup_requests, min(args.qps, 0.5), args.seed + 7
    else:
        n, qps, seed = args.requests, args.qps, args.seed
    wl = build_workload(1, STUB, args.prompt_len - STUB, n, "uniform", 1.1, seed,
                        kv_bytes_per_token=args.kv_bytes)

    if args.start_at:
        time.sleep(max(0.0, args.start_at - time.time()))
    t_start = time.time()
    results, duration = asyncio.run(client_mod.run_load(
        args.base_url, args.model, wl.requests, qps=qps, max_tokens=args.max_tokens,
        seed=seed + 1, timeout=args.timeout, label=f"small-{args.phase}"))

    out = {
        "phase": args.phase,
        "t_start_epoch": round(t_start, 3),
        "t_end_epoch": round(time.time(), 3),
        "duration_s": round(duration, 2),
        "summary": client_mod.summarize(results, duration, args.max_tokens),
        "records": [{k: r.get(k) for k in ("index", "t_submit", "ttft", "e2e")}
                    for r in results],
        "errors": sorted({r.get("error") for r in results if r.get("error")})[:5],
    }
    Path(args.out).write_text(json.dumps(out))
    s = out["summary"]
    print(f"small {args.phase}: n_ok={s['n_ok']} n_failed={s['n_failed']} "
          f"ttft_p50={s['ttft_ms']['p50']}ms tpot_p50={s['tpot_ms']['p50']}ms "
          f"{s['request_tok_per_s']} req/s", flush=True)


if __name__ == "__main__":
    main()
