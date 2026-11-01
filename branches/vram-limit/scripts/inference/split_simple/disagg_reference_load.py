#!/usr/bin/env python3
"""The inference study's workload, replayed against the disaggregated stack.

Reuses scripts/inference/workload.py and client.py verbatim -- same 32 sessions x
6144-token block-aligned prefixes (24 GiB KV working set), same prefix+128-suffix
requests emitting exactly 128 tokens, same open-loop Poisson arrivals -- so the split
system is measured under the load RESULTS.md used, not a new one. The only difference is
the URL: requests go to the router, which runs each one as a prefill leg on GPU0 and a
decode leg on GPU1.

Run the stack with prefix caching on the producer first:

    PREFILL_PREFIX_CACHING=1 bash scripts/decode/disagg_launch.sh
    scripts/decode/disagg_reference_load.py --requests 120 --tag smoke

Open loop is deliberate here exactly as it was there: demand is fixed at --qps, so as
GPU1's VRAM shrinks the curve shows queueing and preemption, comparable point-for-point
with the colocated study's.
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "simple"))
sys.path.insert(0, str(_HERE))

import client as client_mod              # noqa: E402  scripts/inference/client.py
from disagg_closed_loop import Sampler, scrape   # noqa: E402
from workload import build_workload, warmup_requests  # noqa: E402

MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    ap.add_argument("--decode-url", default="http://127.0.0.1:8200")
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--suffix-len", type=int, default=128)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--skew", choices=["zipf", "uniform"], default="zipf")
    ap.add_argument("--zipf-a", type=float, default=1.1)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--warmup-qps", type=float, default=4.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()

    wl = build_workload(args.sessions, args.prefix_len, args.suffix_len, args.requests,
                        args.skew, args.zipf_a, args.seed)
    print(f"workload: {wl.summary()}", flush=True)

    # Warmup touches every session once, as in the inference study. Through the router it
    # also warms the producer's prefix cache AND pre-populates nothing on the decode side
    # (its cache is per-request), which is the intended asymmetry.
    wu = warmup_requests(wl)
    print(f"warmup: {len(wu)} requests at {args.warmup_qps} qps ...", flush=True)
    asyncio.run(client_mod.run_load(args.url, MODEL, wu, args.warmup_qps,
                                    max_tokens=8, seed=args.seed,
                                    timeout=args.timeout, label="warmup"))

    counters = ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                "vllm:external_prefix_cache_hits_total",
                "vllm:external_prefix_cache_queries_total",
                "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                "vllm:num_preemptions_total"]
    pre0 = scrape(args.prefill_url, counters)
    dec0 = scrape(args.decode_url, counters)
    dec_sampler = Sampler(args.decode_url)
    pre_sampler = Sampler(args.prefill_url, prefix="prefill_")
    dec_sampler.start()
    pre_sampler.start()

    print(f"measured run: {len(wl.requests)} requests at {args.qps} qps ...", flush=True)
    t0 = time.time()
    results, duration = asyncio.run(client_mod.run_load(
        args.url, MODEL, wl.requests, args.qps, max_tokens=args.max_tokens,
        seed=args.seed + 1, timeout=args.timeout, label=args.tag))

    dec_sampler.stop = pre_sampler.stop = True
    dec_sampler.join(timeout=2)
    pre_sampler.join(timeout=2)
    pre1 = scrape(args.prefill_url, counters)
    dec1 = scrape(args.decode_url, counters)

    summary = client_mod.summarize(results, duration, args.max_tokens)

    def delta(a, b, k):
        return (b.get(k, 0) or 0) - (a.get(k, 0) or 0)

    hits = delta(pre0, pre1, "vllm:prefix_cache_hits_total")
    queries = delta(pre0, pre1, "vllm:prefix_cache_queries_total")
    out = {
        "tag": args.tag, "workload": wl.summary(),
        "qps": args.qps, "max_tokens": args.max_tokens,
        **summary,
        "prefill_prefix_hit_rate": round(hits / queries, 4) if queries else None,
        "prefill_dram_hits": delta(pre0, pre1, "vllm:external_prefix_cache_hits_total"),
        "prefill_dram_queries": delta(pre0, pre1, "vllm:external_prefix_cache_queries_total"),
        "prefill_prompt_tok": delta(pre0, pre1, "vllm:prompt_tokens_total"),
        "decode_gen_tok": delta(dec0, dec1, "vllm:generation_tokens_total"),
        "decode_preemptions": delta(dec0, dec1, "vllm:num_preemptions_total"),
        **dec_sampler.summary(),
        **pre_sampler.summary(),
    }
    print(json.dumps(out, indent=2))
    out_path = _HERE.parents[2] / "output" / f"disagg_ref_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": out, "records": results}, f)
    print(f"wrote {out_path}", flush=True)
    _ = t0


if __name__ == "__main__":
    main()
