#!/usr/bin/env python3
"""Case A's workload, run against a disaggregated prefill/decode pair.

Same request stream as the inference study -- same `workload.py`, same `client.py`, same
Poisson open loop at 2 QPS -- so the only thing that changes is where the work happens:

    same-card   one instance does prefill and decode; the swept budget is that card's VRAM
    split-card  GPU0 prefills, ships KV to GPU1, GPU1 decodes; the swept budget is GPU1's

The open loop is what makes the comparison meaningful. A closed loop would throttle itself
and hide the queueing that shows up when a tier is too small -- which is the effect being
measured on both sides.

    scripts/decode/disagg_sweep.py --tag disagg              # the full sweep
    scripts/decode/disagg_sweep.py --budgets 30 --tag pilot  # one point first
"""

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "simple"))

import client as client_mod                                    # noqa: E402
from workload import build_workload, warmup_requests           # noqa: E402

OUT = REPO / "output"
RAW = OUT / "raw"
LOGS = OUT / "logs"
GPUMON = OUT / "gpumon"

# vLLM sizes gpu_memory_utilization against torch's *total* device memory, not nvidia-smi's.
# Same constant as run_sweep.py, so a budget means the same thing in both studies.
GPU_TOTAL_GIB = 31.3536
DEFAULT_BUDGETS = [30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 19.0, 18.0]

PROXY = "http://127.0.0.1:8000"
_ACTIVE_STACK = ["p2p"]   # which stack the stall watchdog should tear down
PREFILL_URL = "http://127.0.0.1:8100"
DECODE_URL = "http://127.0.0.1:8200"

# Markers that mean a run's KV never arrived, so its numbers are not measurements.
INVALID_MARKERS = ("RECV TIMEOUT", "kv_cache is None", "Insufficient memory",
                   "Peer Out Of Memory", "EngineDeadError")
# Counted but not run-invalidating. The lmcache L1 logs thousands of transient allocation
# failures under churn (24 GiB flowing through a 6 GiB staging pool) yet stores and
# retrievals all complete -- measured: 32/32 sessions stored, full-prefix retrieval on
# every request alongside 6.7k of these lines. A crash from real exhaustion still shows
# up as EngineDeadError.
WARN_MARKERS = ("Failed to allocate memory block",)


def scrape(url):
    """Prometheus text -> {name: value}, summing across label sets."""
    out = {}
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#") or "{" not in line:
                    continue
                name, val = line.split("{", 1)[0], line.rsplit(" ", 1)[-1]
                try:
                    out[name] = out.get(name, 0.0) + float(val)
                except ValueError:
                    pass
    except Exception as e:
        out["error"] = str(e)
    return out


# The two split stacks share ports, pidfiles, log names and the reference-load client;
# only their launch/stop scripts differ. "p2p" is the patched P2pNcclConnector stack in
# this directory; "lmcache" is the shared-DRAM-cache stack in ../split_lmcache.
STACKS = {
    "p2p": (HERE / "disagg_launch.sh", HERE / "disagg_stop.sh"),
    "lmcache": (HERE.parent / "split_lmcache" / "lmc_launch.sh",
                HERE.parent / "split_lmcache" / "lmc_stop.sh"),
}


def launch(stack, decode_util, env_extra):
    env = {**os.environ, **env_extra}
    r = subprocess.run(["bash", str(STACKS[stack][0]), f"{decode_util:.4f}"],
                       env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or "proxy up" not in r.stdout:
        raise RuntimeError(f"launch failed: {r.stdout[-500:]}{r.stderr[-500:]}")
    return r.stdout


def stop(stack):
    subprocess.run(["bash", str(STACKS[stack][1])], capture_output=True,
                   text=True, timeout=180)


def parse_startup():
    """KV sizes as the engines themselves reported them, not as we asked for them."""
    out = {}
    for role, log in (("prefill", "disagg_prefill.log"), ("decode", "disagg_decode.log")):
        try:
            text = (LOGS / log).read_text(errors="replace")
        except OSError:
            continue
        if m := re.findall(r"Available KV cache memory: ([\d.]+) GiB", text):
            out[f"{role}_kv_gib"] = float(m[-1])
        if m := re.findall(r"GPU KV cache size: ([\d,]+) tokens", text):
            out[f"{role}_kv_tokens"] = int(m[-1].replace(",", ""))
    return out


def log_markers():
    """Count the connector's failure markers across both logs."""
    counts = {}
    for log in ("disagg_prefill.log", "disagg_decode.log"):
        try:
            text = (LOGS / log).read_text(errors="replace")
        except OSError:
            continue
        for marker in INVALID_MARKERS + WARN_MARKERS:
            n = text.count(marker)
            if n:
                counts[marker] = counts.get(marker, 0) + n
    return counts


def load_with_watchdog(requests, qps, max_tokens, seed, timeout, stall_timeout, model):
    """Run a load phase, aborting if the decode engine stops making progress.

    Carried over from run_sweep.py for the same reason, and it matters more here: one
    mis-keyed transfer wedges the decode engine permanently, and vLLM logs engine stats
    every ~10 s whenever requests are in flight, so a log that has not been written for
    `stall_timeout` while the client is still waiting means the engine is stuck.
    """
    box = {}

    def work():
        try:
            box["res"] = asyncio.run(client_mod.run_load(
                PROXY, model, requests, qps=qps, max_tokens=max_tokens,
                seed=seed, timeout=timeout))
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    hung, stall_metrics = False, None
    log_path = LOGS / "disagg_decode.log"
    while True:
        t.join(timeout=15)
        if not t.is_alive():
            break
        try:
            idle = time.time() - log_path.stat().st_mtime
        except OSError:
            idle = 0
        if idle > stall_timeout:
            hung = True
            print(f"  STALLED: no decode-engine progress for {idle:.0f}s -- killing",
                  flush=True)
            # Scrape before the kill: afterwards /metrics is gone and every counter delta
            # comes out negative, silently corrupting the row.
            stall_metrics = {"prefill": scrape(PREFILL_URL), "decode": scrape(DECODE_URL)}
            stop(_ACTIVE_STACK[0])
            t.join(timeout=240)
            break
    if "err" in box:
        raise RuntimeError(box["err"])
    return box.get("res", ([], 0.0)), hung, stall_metrics


def gpu_stats(csv_path, t0, t1):
    """Mean/max per GPU over [t0, t1] from a gpu_monitor.py capture."""
    per = {}
    try:
        for line in open(csv_path):
            f = line.strip().split(",")
            if len(f) != 6 or f[0] == "ts":
                continue
            ts = float(f[0])
            if not (t0 <= ts <= t1):
                continue
            g = per.setdefault(int(f[1]), {"n": 0, "smact": 0.0, "smocc": 0.0,
                                           "drama": 0.0, "fb_max": 0.0})
            g["n"] += 1
            g["smact"] += float(f[2]); g["smocc"] += float(f[3]); g["drama"] += float(f[4])
            g["fb_max"] = max(g["fb_max"], float(f[5]))
    except OSError:
        return {}
    out = {}
    for gpu, role in ((0, "prefill"), (1, "decode")):
        g = per.get(gpu)
        if g and g["n"]:
            out[f"{role}_sm_active_mean"] = round(g["smact"] / g["n"], 4)
            out[f"{role}_sm_occupancy_mean"] = round(g["smocc"] / g["n"], 4)
            out[f"{role}_dram_active_mean"] = round(g["drama"] / g["n"], 4)
            out[f"{role}_fb_used_max_gib"] = round(g["fb_max"] / 1024, 2)
    return out


def run_one(budget, wl, args):
    util = round(budget / GPU_TOTAL_GIB, 4)
    name = f"split_{args.stack}_{wl.skew}_b{budget:g}{args.name_suffix}"
    print(f"\n=== {name}  (decode util={util}) ===", flush=True)
    record = {"name": name, "budget_gib": budget, "util": util, "skew": wl.skew,
              "setup": f"split_{args.stack}",
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "max_inflight": args.max_inflight}
    GPUMON.mkdir(parents=True, exist_ok=True)
    mon_path = GPUMON / f"{name}.csv"
    mon = subprocess.Popen([sys.executable, str(REPO / "scripts/common/gpu_monitor.py"),
                            str(mon_path)])
    try:
        t0 = time.time()
        _ACTIVE_STACK[0] = args.stack
        env_extra = {"FORWARD_FIRST_TOKEN": "1"} if args.forward_first_token else {}
        launch(args.stack, util, {**env_extra,
                      "DECODE_MEM_POOL_GB": str(args.pool_gib),
                      "KV_BUFFER_SIZE": args.kv_buffer_size,
                      "MAX_INFLIGHT": str(args.max_inflight),
                      "PREFILL_PREFIX_CACHING": "1" if args.prefill_prefix_cache else "0",
                      "P2P_RECV_TIMEOUT_S": str(int(args.stall_timeout))})
        record["startup"] = {**parse_startup(), "startup_s": round(time.time() - t0, 1)}
        print(f"  decode KV: {record['startup'].get('decode_kv_gib')} GiB, "
              f"prefill KV: {record['startup'].get('prefill_kv_gib')} GiB, "
              f"up in {record['startup']['startup_s']}s", flush=True)

        # Warm every session once, so its prefix exists in the prefill node's cache before
        # measuring; otherwise the early part of a run measures cache-filling.
        wu = warmup_requests(wl)
        (_, wu_dur), hung, _ = load_with_watchdog(
            wu, args.warmup_qps, 8, args.seed, args.timeout, args.stall_timeout, args.model)
        record["warmup"] = {"n": len(wu), "duration_s": round(wu_dur, 2), "hung": hung}
        print(f"  warmup: {len(wu)} sessions in {wu_dur:.1f}s", flush=True)
        if hung:
            record.update(ok=False, hung=True, hung_phase="warmup",
                          error="decode engine stalled during warmup")
            return record

        before = {"prefill": scrape(PREFILL_URL), "decode": scrape(DECODE_URL)}
        markers_before = log_markers()
        t_measure0 = time.time()
        (results, duration), hung, stall_metrics = load_with_watchdog(
            wl.requests, args.qps, args.max_tokens, args.seed + 1, args.timeout,
            args.stall_timeout, args.model)
        after = stall_metrics or {"prefill": scrape(PREFILL_URL), "decode": scrape(DECODE_URL)}

        record["hung"] = hung
        record["gpu"] = gpu_stats(mon_path, t_measure0, time.time())
        if hung:
            record["hung_phase"] = "measure"
        record["summary"] = client_mod.summarize(results, duration, args.max_tokens)
        # Per-request records, compact (no per-token ITL lists): enough to reconstruct
        # the request/session timeline of a run offline.
        record["records"] = [
            {k: r.get(k) for k in ("index", "session_id", "t_submit", "ttft", "e2e")}
            for r in results]
        record["metrics_delta"] = {
            role: {k: round(after[role].get(k, 0.0) - before[role].get(k, 0.0), 4)
                   for k in set(after[role]) | set(before[role]) if k != "error"}
            for role in ("prefill", "decode")
        }
        markers = {k: v - markers_before.get(k, 0) for k, v in log_markers().items()}
        record["kv_transfer_markers"] = {k: v for k, v in markers.items()
                                         if v > 0 and k in INVALID_MARKERS}
        record["kv_transfer_warnings"] = {k: v for k, v in markers.items()
                                          if v > 0 and k in WARN_MARKERS}
        record["kv_transfer_ok"] = not record["kv_transfer_markers"]
        record["ok"] = not hung and record["kv_transfer_ok"]
        s = record["summary"]
        g = record.get("gpu", {})
        if g:
            print(f"  gpu[prefill] smact={g.get('prefill_sm_active_mean')} "
                  f"smocc={g.get('prefill_sm_occupancy_mean')} "
                  f"dram={g.get('prefill_dram_active_mean')} "
                  f"fb={g.get('prefill_fb_used_max_gib')}GiB | "
                  f"gpu[decode] smact={g.get('decode_sm_active_mean')} "
                  f"smocc={g.get('decode_sm_occupancy_mean')} "
                  f"dram={g.get('decode_dram_active_mean')} "
                  f"fb={g.get('decode_fb_used_max_gib')}GiB", flush=True)
        print(f"  TTFT p50={s['ttft_ms']['p50']}ms p95={s['ttft_ms']['p95']}ms | "
              f"{s['output_tok_per_s']} tok/s | failed={s['n_failed']} | {duration:.0f}s"
              f"{'  KV MARKERS: ' + str(record['kv_transfer_markers']) if not record['kv_transfer_ok'] else ''}",
              flush=True)
    except Exception as e:
        record.update(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"  FAILED: {record['error']}", flush=True)
    finally:
        mon.terminate()
        stop(args.stack)

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def derive_row(rec, wl):
    """Flatten one run, matching run_sweep.py's columns wherever the quantity exists here."""
    s = rec.get("summary", {})
    st = rec.get("startup", {})
    md = rec.get("metrics_delta", {})
    dec, pre = md.get("decode", {}), md.get("prefill", {})
    ttft = s.get("ttft_ms", {})
    dur = s.get("duration_s")

    kv_gib = st.get("decode_kv_gib")
    return {
        "name": rec["name"], "setup": rec.get("setup", "split"), "ok": rec.get("ok"),
        "hung": rec.get("hung", False), "hung_phase": rec.get("hung_phase", ""),
        "kv_transfer_ok": rec.get("kv_transfer_ok"),
        "skew": rec.get("skew"), "arm": "disagg",
        "budget_gib": rec["budget_gib"], "util": rec["util"],
        # The decode node's own KV tier -- the analogue of the same-card study's gpu_kv_gib.
        "gpu_kv_gib": kv_gib, "gpu_kv_tokens": st.get("decode_kv_tokens"),
        "prefill_kv_gib": st.get("prefill_kv_gib"),
        "gpu_kv_frac_of_ws": round(kv_gib / wl.working_set_gib, 4) if kv_gib else None,
        "sessions_resident": round(kv_gib / wl.prefix_gib, 2) if kv_gib else None,
        "ttft_p50_ms": ttft.get("p50"), "ttft_p90_ms": ttft.get("p90"),
        "ttft_p95_ms": ttft.get("p95"), "ttft_p99_ms": ttft.get("p99"),
        "ttft_mean_ms": ttft.get("mean"),
        "itl_p50_ms": s.get("itl_ms", {}).get("p50"),
        "e2e_p50_ms": s.get("e2e_ms", {}).get("p50"),
        "output_tok_per_s": s.get("output_tok_per_s"),
        "req_per_s": s.get("request_tok_per_s"),
        "duration_s": dur, "n_ok": s.get("n_ok"), "n_failed": s.get("n_failed"),
        "preemptions": dec.get("vllm:num_preemptions_total", 0.0),
        # Validity check: if the prefill node saturates, the decode curve measures GPU0.
        "prefill_prompt_tok_s": round(pre.get("vllm:prompt_tokens_total", 0.0) / dur, 1)
        if dur else None,
        "prefill_hit_rate": round(
            pre.get("vllm:prefix_cache_hits_total", 0.0)
            / pre["vllm:prefix_cache_queries_total"], 4)
        if pre.get("vllm:prefix_cache_queries_total") else None,
        "decode_gen_tok_s": round(dec.get("vllm:generation_tokens_total", 0.0) / dur, 1)
        if dur else None,
        "max_inflight": rec.get("max_inflight"),
        **rec.get("gpu", {}),
        "kv_markers": json.dumps(rec.get("kv_transfer_markers", {})),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS)
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--suffix-len", type=int, default=128)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--warmup-qps", type=float, default=4.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--stall-timeout", type=float, default=180.0)
    ap.add_argument("--pool-gib", type=int, default=32)
    ap.add_argument("--kv-buffer-size", default="1e8")
    ap.add_argument("--max-inflight", type=int, default=28)
    ap.add_argument("--prefill-prefix-cache", action="store_true", default=True)
    ap.add_argument("--no-prefill-prefix-cache", dest="prefill_prefix_cache",
                    action="store_false")
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--stack", choices=list(STACKS), default="p2p")
    ap.add_argument("--forward-first-token", action="store_true",
                    help="lmcache stack only: stream prefill's token #1 to the client "
                         "immediately; the KV retrieval moves into the first ITL")
    ap.add_argument("--name-suffix", default="",
                    help="appended to every config name so a variant never clobbers "
                         "earlier raw/log/gpumon artifacts")
    ap.add_argument("--skew", choices=["zipf", "uniform"], default="zipf")
    ap.add_argument("--zipf-a", type=float, default=1.1)
    ap.add_argument("--tag", default="disagg")
    args = ap.parse_args()

    wl = build_workload(args.sessions, args.prefix_len, args.suffix_len, args.requests,
                        args.skew, args.zipf_a, args.seed)
    s = wl.summary()
    print(f"workload[{args.skew}] on stack '{args.stack}': "
          f"{s['working_set_gib']} GiB working set, "
          f"{s['num_requests']} requests at {args.qps} QPS, "
          f"prefix {s['prefix_kv_gib']} GiB/session")
    print(f"budgets: {args.budgets}  pool={args.pool_gib} GiB  "
          f"max_inflight={args.max_inflight}  "
          f"prefill_prefix_cache={args.prefill_prefix_cache}")

    OUT.mkdir(exist_ok=True)
    rows, t_start = [], time.time()
    for b in args.budgets:
        rows.append(derive_row(run_one(b, wl, args), wl))
        csv_path = OUT / f"summary_{args.tag}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> "
          f"output/summary_{args.tag}.csv", flush=True)


if __name__ == "__main__":
    main()
