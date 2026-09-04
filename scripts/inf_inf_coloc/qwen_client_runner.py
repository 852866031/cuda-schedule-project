#!/usr/bin/env python3
"""Load generator for the Qwen tenant, run as a SEPARATE PROCESS by coloc2_sweep.py.

The 8B client runs in-process in the driver, byte-identical to the path that produced
the b26 baseline; a second streaming client in the same interpreter would contend for
the GIL and pollute the 8B latencies. This runner rebuilds the Qwen workload
deterministically from its args (same code, same seed -> same requests) and writes one
JSON blob the driver embeds into the raw record.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import client as client_mod                                    # noqa: E402
from workload import (QWEN_KV_BYTES_PER_TOKEN, build_workload,  # noqa: E402
                      warmup_requests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--suffix-len", type=int, default=128)
    ap.add_argument("--skew", default="zipf", choices=["zipf", "uniform"])
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--phase", choices=["warmup", "measure"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wl = build_workload(args.sessions, args.prefix_len, args.suffix_len, args.requests,
                        args.skew, 1.1, args.seed,
                        kv_bytes_per_token=QWEN_KV_BYTES_PER_TOKEN)
    if args.phase == "warmup":
        reqs, qps, max_tokens, load_seed = warmup_requests(wl), min(args.qps, 0.5), 8, args.seed
    else:
        reqs, qps, max_tokens, load_seed = wl.requests, args.qps, args.max_tokens, args.seed + 1

    t_start = time.time()
    results, duration = asyncio.run(client_mod.run_load(
        args.base_url, args.model, reqs, qps=qps, max_tokens=max_tokens,
        seed=load_seed, timeout=args.timeout, label=f"qwen-{args.phase}"))

    out = {
        "phase": args.phase,
        "t_start_epoch": round(t_start, 3),
        "t_end_epoch": round(time.time(), 3),
        "duration_s": round(duration, 2),
        "summary": client_mod.summarize(results, duration, max_tokens),
        "records": [{k: r.get(k) for k in ("index", "session_id", "t_submit",
                                           "ttft", "e2e")} for r in results],
        "errors": sorted({r.get("error") for r in results if r.get("error")})[:5],
    }
    Path(args.out).write_text(json.dumps(out))
    s = out["summary"]
    print(f"qwen {args.phase}: n_ok={s['n_ok']} n_failed={s['n_failed']} "
          f"ttft_p50={s['ttft_ms']['p50']}ms {s['output_tok_per_s']} tok/s", flush=True)


if __name__ == "__main__":
    main()
