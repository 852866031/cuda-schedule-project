#!/usr/bin/env python3
"""Why does the offload arm stall?

The smoke run showed the offload arm's warmup taking 77 s against 4.5 s without offload, with
a ~40 s window of *zero* engine throughput while 3 requests sat "Running" and GPU KV was only
39% full. Measured PCIe time in that window was 0.39 s, so the stall is not the copies.

This sends cold requests one at a time and reports, per request, the wall time and the bytes
and transfer-time the connector accumulated. Two hypotheses it separates:

  - one-time cost (lazy pinned-pool allocation): only the first request or two are slow
  - per-store cost (stores blocking the engine loop): every cold request is slow, and the
    slowdown tracks bytes stored

Usage: python scripts/probe_offload_stall.py [--pool-gib 8] [--n 10]
"""

import argparse
import asyncio
import json
import time

import client as client_mod
import server as server_mod
from workload import build_workload, warmup_requests


def probe(pool_gib, n, prefix_len, model, extra, burst_qps=4.0):
    wl = build_workload(num_sessions=n, prefix_len=prefix_len, num_requests=n,
                        skew="uniform", seed=7)
    # Exactly one cold request per distinct session. warmup_requests() guarantees that;
    # sampling from wl.requests does not (a uniform draw repeats sessions, and a repeat is a
    # cache hit, not a prefill).
    reqs = warmup_requests(wl)

    out = {}
    for arm, pool in (("offload", pool_gib), ("nooffload", None)):
        print(f"\n=== {arm} (pool={pool}) ===", flush=True)
        srv = server_mod.VLLMServer(gpu_mem_util=0.7655, kv_offload_gib=pool,
                                    run_name=f"probe_{arm}", extra_args=extra, model=model)
        rows, burst_row = [], None
        try:
            srv.start()
            prev = srv.metrics()
            for i, req in enumerate(reqs):
                t0 = time.perf_counter()
                res, _ = asyncio.run(client_mod.run_load(
                    srv.base_url, model, [req], qps=0, max_tokens=8, seed=i, timeout=600))
                dt = time.perf_counter() - t0
                cur = srv.metrics()
                d = {k: cur.get(k, 0) - prev.get(k, 0) for k in cur if k != "error"}
                prev = cur
                store_gib = sum(v for k, v in d.items()
                                if "kv_offload_total_bytes" in k.lower()
                                and "gpu_to_cpu" in k.lower()) / (1 << 30)
                store_s = sum(v for k, v in d.items()
                              if "kv_offload_total_time" in k.lower()
                              and "gpu_to_cpu" in k.lower())
                rows.append({"i": i, "wall_s": round(dt, 2),
                             "ttft_s": round(res[0]["ttft"], 3) if res[0]["ttft"] else None,
                             "store_gib": round(store_gib, 3), "store_s": round(store_s, 3)})
                print(f"  req {i:2d}: wall {dt:6.2f}s  ttft {rows[-1]['ttft_s']}  "
                      f"stored {store_gib:5.2f} GiB in {store_s:.3f}s", flush=True)
            # Phase 2: the smoke-run condition -- all sessions fired at once. Sequential
            # stores overlap for free; the question is whether concurrent ones still do.
            for r in reqs:  # shift token ids so nothing is already cached
                r.token_ids = [t + 1 for t in r.token_ids]
            m0 = srv.metrics()
            t0 = time.perf_counter()
            burst, _ = asyncio.run(client_mod.run_load(
                srv.base_url, model, reqs, qps=burst_qps, max_tokens=8, seed=99, timeout=900))
            burst_wall = time.perf_counter() - t0
            m1 = srv.metrics()
            d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1 if k != "error"}
            store_gib = sum(v for k, v in d.items()
                            if "kv_offload_total_bytes" in k.lower()
                            and "gpu_to_cpu" in k.lower()) / (1 << 30)
            store_s = sum(v for k, v in d.items()
                          if "kv_offload_total_time" in k.lower()
                          and "gpu_to_cpu" in k.lower())
            ttfts = sorted(r["ttft"] for r in burst if r["ttft"])
            burst_row = {
                "n": len(reqs), "wall_s": round(burst_wall, 2),
                "ttft_p50_s": round(ttfts[len(ttfts) // 2], 3) if ttfts else None,
                "ttft_max_s": round(ttfts[-1], 3) if ttfts else None,
                "store_gib": round(store_gib, 3), "store_s": round(store_s, 3),
            }
            print(f"  BURST x{len(reqs)} @ {burst_qps} qps: wall {burst_wall:.2f}s  "
                  f"ttft p50 {burst_row['ttft_p50_s']}s max {burst_row['ttft_max_s']}s  "
                  f"stored {store_gib:.2f} GiB in {store_s:.3f}s", flush=True)
        finally:
            srv.stop()
        out[arm] = {"sequential": rows, "burst": burst_row}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-gib", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--model", default=server_mod.MODEL)
    ap.add_argument("--extra", nargs="*", default=["--enforce-eager"])
    ap.add_argument("--burst-qps", type=float, default=4.0)
    a = ap.parse_args()

    res = probe(a.pool_gib, a.n, a.prefix_len, a.model, a.extra, a.burst_qps)
    path = server_mod.REPO / "output" / "probe_offload_stall.json"
    path.write_text(json.dumps(res, indent=2))
    for arm, d in res.items():
        walls = [r["wall_s"] for r in d["sequential"]]
        print(f"{arm:10s} sequential: first={walls[0]:.2f}s median={sorted(walls)[len(walls)//2]:.2f}s "
              f"total={sum(walls):.1f}s | burst: {d['burst']}")
    print(f"wrote {path}")
