#!/usr/bin/env python3
"""Case A sweep driver: shrink the VRAM budget, with and without DRAM KV offloading.

For each point: start a fresh server, warm every session into the cache, run the measured
Poisson load, snapshot metrics, tear down. Results land in output/raw/<run>.json and are
rolled up into output/summary.csv.

  python scripts/run_sweep.py --smoke        # 2 configs, tiny workload -- validates plumbing
  python scripts/run_sweep.py                # the full Case A matrix
  python scripts/run_sweep.py --budgets 24 20 --skews zipf
"""

import argparse
import asyncio
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import client as client_mod
import server as server_mod
from workload import build_workload, warmup_requests

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "output"
RAW = OUT / "raw"

# vLLM sizes gpu_memory_utilization against torch's *total* device memory, not nvidia-smi's.
GPU_TOTAL_GIB = 31.3536

DEFAULT_BUDGETS = [30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 19.0, 18.0]
CPU_POOL_GIB = 24.0   # fixed: only the GPU tier shrinks across the sweep


def make_configs(budgets, skews, arms, cpu_pool):
    configs = []
    for skew in skews:
        for b in budgets:
            for arm in arms:
                offload = cpu_pool if arm == "offload" else None
                configs.append({
                    "name": f"{skew}_b{b:g}_{arm}",
                    "budget_gib": b,
                    "util": round(b / GPU_TOTAL_GIB, 4),
                    "arm": arm,
                    "kv_offload_gib": offload,
                    "skew": skew,
                })
    return configs


def load_with_watchdog(srv, model, requests, qps, max_tokens, seed, timeout, stall_timeout):
    """Run a load phase, aborting if the engine stops making progress.

    At small GPU KV tiers the offloading connector can wedge: requests sit in the waiting
    queue, nothing runs, and the engine stops logging entirely. Left alone, every in-flight
    request then burns the full client timeout. vLLM logs engine stats every ~10 s whenever
    requests are in flight, so a log file that has not been written for `stall_timeout`
    seconds while the client is still waiting means the engine is wedged. Killing the server
    fails the outstanding requests immediately and the run is recorded as hung -- which is a
    result about the offload path, not an error to retry.
    """
    box = {}

    def work():
        try:
            box["res"] = asyncio.run(client_mod.run_load(
                srv.base_url, model, requests, qps=qps, max_tokens=max_tokens,
                seed=seed, timeout=timeout))
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    hung, stall_metrics = False, None
    while True:
        t.join(timeout=15)
        if not t.is_alive():
            break
        try:
            idle = time.time() - srv.log_path.stat().st_mtime
        except OSError:
            idle = 0
        if idle > stall_timeout:
            hung = True
            print(f"  STALLED: no engine progress for {idle:.0f}s -- killing server",
                  flush=True)
            # Scrape before the kill: afterwards /metrics is gone and every counter delta
            # comes out negative, which silently corrupts the row for a hung config.
            stall_metrics = srv.metrics()
            srv.stop()
            t.join(timeout=180)
            break
    if "err" in box:
        raise RuntimeError(box["err"])
    return box.get("res", ([], 0.0)), hung, stall_metrics


def run_one(cfg, wl, args, repeat=0):
    name = cfg["name"] + (f"_r{repeat}" if repeat else "")
    print(f"\n=== {name}  (util={cfg['util']}, offload={cfg['kv_offload_gib']}) ===", flush=True)

    srv = server_mod.VLLMServer(
        gpu_mem_util=cfg["util"],
        kv_offload_gib=cfg["kv_offload_gib"],
        backend="native",
        run_name=name,
        gpu=args.gpu,
        extra_args=args.extra,
    )
    record = {"config": cfg, "repeat": repeat, "name": name,
              "timestamp": datetime.now().isoformat(timespec="seconds")}
    try:
        startup = srv.start()
        record["startup"] = startup
        print(f"  GPU KV: {startup.get('gpu_kv_gib')} GiB "
              f"({startup.get('gpu_kv_tokens')} tokens), up in {startup.get('startup_s')}s",
              flush=True)

        # Warmup: touch every session once so all KV exists somewhere before measuring.
        wu = warmup_requests(wl)
        (_, wu_dur), hung, _ = load_with_watchdog(
            srv, args.model, wu, args.warmup_qps, 8, args.seed,
            args.timeout, args.stall_timeout)
        record["warmup"] = {"n": len(wu), "duration_s": round(wu_dur, 2), "hung": hung}
        print(f"  warmup: {len(wu)} sessions in {wu_dur:.1f}s", flush=True)
        if hung:
            record["ok"], record["hung"], record["hung_phase"] = False, True, "warmup"
            record["error"] = "engine stalled during warmup"
            RAW.mkdir(parents=True, exist_ok=True)
            (RAW / f"{name}.json").write_text(json.dumps(record, indent=2))
            return record

        m_before = srv.metrics()
        (results, duration), hung, stall_metrics = load_with_watchdog(
            srv, args.model, wl.requests, args.qps, args.max_tokens, args.seed + 1,
            args.timeout, args.stall_timeout)
        record["hung"] = hung
        if hung:
            record["hung_phase"] = "measure"
        # A hung config's metrics come from the pre-kill scrape; the server is gone now.
        m_after = stall_metrics if hung and stall_metrics else srv.metrics()
        record["gpu_mem"] = server_mod.gpu_free_gib(args.gpu)

        summary = client_mod.summarize(results, duration, args.max_tokens)
        record["summary"] = summary
        # Metrics are cumulative counters; the measured phase is the difference.
        record["metrics_delta"] = {
            k: round(m_after.get(k, 0.0) - m_before.get(k, 0.0), 4)
            for k in set(m_after) | set(m_before) if k != "error"
        }
        record["metrics_raw_after"] = m_after
        record["requests"] = results
        record["ok"] = not hung
        print(f"  TTFT p50={summary['ttft_ms']['p50']}ms p95={summary['ttft_ms']['p95']}ms | "
              f"{summary['output_tok_per_s']} tok/s | "
              f"failed={summary['n_failed']} | {duration:.0f}s", flush=True)
    except Exception as e:
        record["ok"] = False
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED: {record['error']}", flush=True)
    finally:
        srv.stop()

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def derive_row(rec, wl):
    """Flatten one run into the summary CSV, deriving the quantities the plots need."""
    cfg, s = rec["config"], rec.get("summary", {})
    st, md = rec.get("startup", {}), rec.get("metrics_delta", {})

    gpu_q = md.get("vllm:prefix_cache_queries", 0.0)
    gpu_h = md.get("vllm:prefix_cache_hits", 0.0)
    ext_q = md.get("vllm:external_prefix_cache_queries", 0.0)
    ext_h = md.get("vllm:external_prefix_cache_hits", 0.0)
    # vLLM labels these "GPU_to_CPU"/"CPU_to_GPU"; match case-insensitively so a label
    # rename upstream shows up as a missing column rather than a silent zero.
    def dir_sum(prefix, direction):
        return sum(v for k, v in md.items()
                   if k.lower() == f"{prefix}:{direction}".lower())

    load_b = dir_sum("vllm:kv_offload_total_bytes", "cpu_to_gpu")
    store_b = dir_sum("vllm:kv_offload_total_bytes", "gpu_to_cpu")
    load_t = dir_sum("vllm:kv_offload_total_time", "cpu_to_gpu")
    store_t = dir_sum("vllm:kv_offload_total_time", "gpu_to_cpu")

    # vLLM 0.15.1 (release) does not implement KVConnectorStats for OffloadingConnector, so
    # vllm:kv_offload_total_bytes/_time are absent -- the Feb-2026 dev build had them. The
    # external_* counters are in TOKENS (queries == 300 req x 6272 tok exactly), so KV volume
    # is recoverable analytically. What is lost is transfer *time*, i.e. achieved bandwidth;
    # that was measured separately on the dev build (13.5-14.3 GB/s, matching calibration).
    ext_hit_gib = round(ext_h * 131072 / (1 << 30), 3)

    ttft = s.get("ttft_ms", {})
    return {
        "name": rec["name"],
        "ok": rec.get("ok"),
        "hung": rec.get("hung", False),
        "hung_phase": rec.get("hung_phase", ""),
        "skew": cfg["skew"],
        "arm": cfg["arm"],
        "budget_gib": cfg["budget_gib"],
        "util": cfg["util"],
        "gpu_kv_gib": st.get("gpu_kv_gib"),
        "gpu_kv_tokens": st.get("gpu_kv_tokens"),
        # How much of the 24 GiB working set the GPU tier can actually hold.
        "gpu_kv_frac_of_ws": round(st.get("gpu_kv_gib", 0) / wl.working_set_gib, 4)
        if st.get("gpu_kv_gib") else None,
        "sessions_resident": round(st.get("gpu_kv_gib", 0) / wl.prefix_gib, 2)
        if st.get("gpu_kv_gib") else None,
        "ttft_p50_ms": ttft.get("p50"), "ttft_p90_ms": ttft.get("p90"),
        "ttft_p95_ms": ttft.get("p95"), "ttft_p99_ms": ttft.get("p99"),
        "ttft_mean_ms": ttft.get("mean"),
        "itl_p50_ms": s.get("itl_ms", {}).get("p50"),
        "e2e_p50_ms": s.get("e2e_ms", {}).get("p50"),
        "output_tok_per_s": s.get("output_tok_per_s"),
        "req_per_s": s.get("request_tok_per_s"),
        "duration_s": s.get("duration_s"),
        "n_ok": s.get("n_ok"), "n_failed": s.get("n_failed"),
        "gpu_hit_rate": round(gpu_h / gpu_q, 4) if gpu_q else None,
        "dram_hit_rate": round(ext_h / ext_q, 4) if ext_q else None,
        "prefix_queries": gpu_q, "prefix_hits": gpu_h,
        "external_queries": ext_q, "external_hits": ext_h,
        "preemptions": md.get("vllm:num_preemptions", 0.0),
        "kv_load_gib": round(load_b / (1 << 30), 3),
        "kv_load_gib_derived": ext_hit_gib,
        "kv_load_gbps_derived": round(ext_hit_gib * 1.0737 / s["duration_s"], 3)
        if s.get("duration_s") else None,
        "kv_store_gib": round(store_b / (1 << 30), 3),
        "kv_load_s": round(load_t, 3), "kv_store_s": round(store_t, 3),
        # Effective PCIe bandwidth the connector actually achieved, to compare against the
        # 14.47 GB/s ceiling measured in Phase 0.
        "kv_load_gbps": round(load_b / load_t / 1e9, 2) if load_t else None,
        "kv_store_gbps": round(store_b / store_t / 1e9, 2) if store_t else None,
        "gpu_mem_used_gib": rec.get("gpu_mem", {}).get("used_gib"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS)
    ap.add_argument("--skews", nargs="*", default=["zipf", "uniform"])
    ap.add_argument("--arms", nargs="*", default=["offload", "nooffload"])
    ap.add_argument("--cpu-pool-gib", type=float, default=CPU_POOL_GIB)
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--suffix-len", type=int, default=128)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--warmup-qps", type=float, default=4.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--zipf-a", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--stall-timeout", type=float, default=180.0,
                    help="abort a config after this many seconds with no engine progress")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=server_mod.MODEL)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra vllm serve args, e.g. --extra=--enforce-eager")
    ap.add_argument("--tag", default="", help="suffix for the summary csv")
    ap.add_argument("--smoke", action="store_true", help="tiny 2-config validation run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.budgets = [24.0]
        args.skews = ["zipf"]
        args.sessions, args.prefix_len, args.requests = 8, 6144, 24
        args.cpu_pool_gib, args.tag = 8.0, "smoke"

    configs = make_configs(args.budgets, args.skews, args.arms, args.cpu_pool_gib)
    print(f"{len(configs)} configs x {args.repeats} repeat(s)")
    for c in configs:
        print(f"  {c['name']:32s} util={c['util']:.4f} offload={c['kv_offload_gib']}")
    if args.dry_run:
        return

    OUT.mkdir(exist_ok=True)
    workloads, rows = {}, []
    for skew in args.skews:
        wl = build_workload(args.sessions, args.prefix_len, args.suffix_len, args.requests,
                            skew, args.zipf_a, args.seed)
        workloads[skew] = wl
        s = wl.summary()
        print(f"\nworkload[{skew}]: {s['working_set_gib']} GiB working set, "
              f"{s['num_requests']} requests, hot90={s['hot_sessions_90pct']} sessions "
              f"({s['hot_set_90pct_gib']} GiB)")

    (OUT / f"workload{'_' + args.tag if args.tag else ''}.json").write_text(
        json.dumps({k: v.summary() for k, v in workloads.items()}, indent=2))

    t_start = time.time()
    for cfg in configs:
        for rep in range(args.repeats):
            rec = run_one(cfg, workloads[cfg["skew"]], args, repeat=rep)
            rows.append(derive_row(rec, workloads[cfg["skew"]]))

            csv_path = OUT / f"summary{'_' + args.tag if args.tag else ''}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> "
          f"output/summary{'_' + args.tag if args.tag else ''}.csv")


if __name__ == "__main__":
    main()
