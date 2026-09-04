#!/usr/bin/env python3
"""Decode+decode colocation driver: the 8B split-LMCache stack + a Qwen2.5-0.5B tenant.

Adapted from scripts/inf_ft_coloc/coloc_sweep.py. Everything reused is COPIED into this
directory (workload.py, client.py, lmc_proxy.py, lmc_server_main.py) -- the study is
self-contained. The 8B side is measured with the exact in-process client path that
produced the b26 baseline (split_lmcache_zipf_b26_fwd_ng); the Qwen client runs in a
separate process (qwen_client_runner.py) so its streaming loop cannot contend with the
8B client for the GIL.

Scenarios (x workloads fits=8 / offload=48 sessions):
  solo  Qwen alone on GPU1 -- tenant baseline.
  B     one Qwen engine (prefill+decode) beside the 8B decode on GPU1.
  A     Qwen split like the 8B: prefill GPU0, decode GPU1, own proxy/server.
  C     Qwen decode-only on GPU1; prefixes pre-populated into its LMCache store by a
        temporary GPU0 prefill that is killed before measurement ("simulated prefill").

    scripts/inf_inf_coloc/coloc2_sweep.py --smoke --scenarios solo C A --wls fits
    scripts/inf_inf_coloc/coloc2_sweep.py --scenarios solo B A C --wls fits offload
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
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import client as client_mod                                    # noqa: E402
from workload import (QWEN_KV_BYTES_PER_TOKEN, build_workload,  # noqa: E402
                      warmup_requests)

OUT = REPO / "output"
RAW = OUT / "raw"
LOGS = OUT / "logs"
GPUMON = OUT / "gpumon"

GPU_TOTAL_GIB = 31.3536

PROXY = "http://127.0.0.1:8000"
PREFILL_URL = "http://127.0.0.1:8100"
DECODE_URL = "http://127.0.0.1:8200"
QWEN_PROXY = "http://127.0.0.1:8001"
QWEN_PREFILL_URL = "http://127.0.0.1:8101"
QWEN_ENGINE_URL = "http://127.0.0.1:8201"

# The b26 baseline's engine-reported KV grants. Any non-solo run must reproduce them,
# or the 8B comparison is not against the same machine state (launcher drift, or the
# Qwen engine racing the 8B decode's profiling pass on GPU1).
BASELINE_DECODE_KV_GIB = 9.77
BASELINE_PREFILL_KV_GIB = 6.66
KV_TOLERANCE_GIB = 0.05

INVALID_MARKERS = ("RECV TIMEOUT", "kv_cache is None", "Insufficient memory",
                   "Peer Out Of Memory", "EngineDeadError")
WARN_MARKERS = ("Failed to allocate memory block",)

LAUNCH = HERE / "coloc2_launch.sh"
STOP = HERE / "coloc2_stop.sh"


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


def swap_counters():
    """pswpin/pswpout from /proc/vmstat -- distinguishes host-DRAM thrash from GPU
    contention when the offload workloads squeeze RAM."""
    out = {}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            k, _, v = line.partition(" ")
            if k in ("pswpin", "pswpout"):
                out[k] = int(v)
    except OSError:
        pass
    return out


def engines_alive(scenario):
    """Which engine metric endpoints exist per scenario, keyed for the record."""
    eps = {"qwen": QWEN_ENGINE_URL}
    if scenario != "solo":
        eps["big_prefill"] = PREFILL_URL
        eps["big_decode"] = DECODE_URL
    if scenario == "C":               # only the split keeps a qwen prefill live at measure
        eps["qwen_prefill"] = QWEN_PREFILL_URL
    return eps


def launch(scenario, decode_util, env_extra):
    env = {**os.environ, **env_extra}
    r = subprocess.run(["bash", str(LAUNCH), scenario, f"{decode_util:.4f}"],
                       env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or "qwen up" not in r.stdout:
        raise RuntimeError(f"launch failed: {r.stdout[-500:]}{r.stderr[-500:]}")
    return r.stdout


def stop():
    subprocess.run(["bash", str(STOP)], capture_output=True, text=True, timeout=180)


def parse_startup(scenario):
    """KV sizes as the engines themselves reported them, not as we asked for them."""
    logs = [("qwen", "qwen_decode.log")]
    if scenario != "solo":
        logs += [("big_prefill", "disagg_prefill.log"), ("big_decode", "disagg_decode.log")]
    if scenario != "A":               # solo/B temp prefill, C persistent prefill
        logs += [("qwen_prefill", "qwen_prefill.log")]
    out = {}
    for role, log in logs:
        try:
            text = (LOGS / log).read_text(errors="replace")
        except OSError:
            continue
        if m := re.findall(r"Available KV cache memory: ([\d.]+) GiB", text):
            out[f"{role}_kv_gib"] = float(m[-1])
        if m := re.findall(r"GPU KV cache size: ([\d,]+) tokens", text):
            out[f"{role}_kv_tokens"] = int(m[-1].replace(",", ""))
    return out


def log_markers(scenario):
    """Failure-marker counts, kept separate per model."""
    groups = {"big": [], "qwen": ["qwen_decode.log"]}
    if scenario != "solo":
        groups["big"] = ["disagg_prefill.log", "disagg_decode.log"]
    if scenario != "A":
        groups["qwen"].append("qwen_prefill.log")
    counts = {"big": {}, "qwen": {}}
    for side, files in groups.items():
        for log in files:
            try:
                text = (LOGS / log).read_text(errors="replace")
            except OSError:
                continue
            for marker in INVALID_MARKERS + WARN_MARKERS:
                n = text.count(marker)
                if n:
                    counts[side][marker] = counts[side].get(marker, 0) + n
    return counts


def gpu_stats(csv_path, t0, t1):
    """Mean/max per GPU over [t0, t1]. Named gpu0/gpu1, NOT prefill/decode: on GPU1 two
    decodes coexist and DCGM cannot attribute counters to a process."""
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
    for gpu in (0, 1):
        g = per.get(gpu)
        if g and g["n"]:
            out[f"gpu{gpu}_sm_active_mean"] = round(g["smact"] / g["n"], 4)
            out[f"gpu{gpu}_sm_occupancy_mean"] = round(g["smocc"] / g["n"], 4)
            out[f"gpu{gpu}_dram_active_mean"] = round(g["drama"] / g["n"], 4)
            out[f"gpu{gpu}_fb_used_max_gib"] = round(g["fb_max"] / 1024, 2)
    return out


def qwen_runner_cmd(args, qwl, phase, out_path, base_url):
    # suffix_len/skew come from the workload: decode-only scenarios (B, solo) build it
    # with suffix_len=0 + uniform skew so each request reuses a whole preloaded prefix
    # (100% hit -> no prefill compute); A/C use the session workload (real prefill).
    return [str(REPO / ".venv" / "bin" / "python"), str(HERE / "qwen_client_runner.py"),
            "--base-url", base_url, "--model", args.qwen_model,
            "--sessions", str(qwl.num_sessions), "--prefix-len", str(qwl.prefix_len),
            "--suffix-len", str(qwl.suffix_len), "--skew", qwl.skew,
            "--requests", str(args.qwen_requests),
            "--qps", str(args.qwen_qps), "--seed", str(args.qwen_seed),
            "--max-tokens", str(args.max_tokens), "--timeout", str(args.timeout),
            "--phase", phase, "--out", str(out_path)]


def qwen_base_url(scenario):
    return QWEN_PROXY if scenario == "C" else QWEN_ENGINE_URL


async def _populate(prefixes, args):
    """POST each session prefix once to the temporary Qwen prefill (:8101)."""
    import aiohttp
    sem = asyncio.Semaphore(2)
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)) as sess:
        async def one(ids):
            async with sem:
                body = {"model": args.qwen_model, "prompt": ids, "max_tokens": 1,
                        "temperature": 0.0, "stream": False, "ignore_eos": True}
                async with sess.post(f"{QWEN_PREFILL_URL}/v1/completions",
                                     json=body) as r:
                    return r.status
        return await asyncio.gather(*(one(p) for p in prefixes))


def populate_store(qwl, args, record):
    """Scenario C: store every session prefix, verify, kill the temp prefill.

    Prefix-only prompts: 6144 ids = exactly 24 full 256-token chunks, and
    save_unfull_chunk=false means a suffix would store nothing extra anyway.
    """
    t0 = time.time()
    statuses = asyncio.run(_populate([s for s in qwl.sessions], args))
    ok = all(s == 200 for s in statuses)
    metrics = scrape(QWEN_PREFILL_URL)
    tokens = metrics.get("vllm:prompt_tokens_total", 0.0)
    need = qwl.num_sessions * args.prefix_len
    if not ok or tokens < need:
        record["populate"] = {"ok": False, "statuses": statuses, "prompt_tokens": tokens}
        raise RuntimeError(f"populate failed: statuses ok={ok}, "
                           f"prompt_tokens {tokens} < {need}")

    # Retrieval probe on the decode engine: session-0 prefix + a throwaway suffix must
    # come back fast (suffix-only compute). A slow probe means the store keys don't
    # match (e.g. PYTHONHASHSEED drift) and C would silently measure full recompute.
    probe = {"model": args.qwen_model,
             "prompt": qwl.sessions[0] + qwl.requests[0].token_ids[-16:],
             "max_tokens": 2, "temperature": 0.0, "stream": False, "ignore_eos": True}
    tp = time.time()
    req = urllib.request.Request(f"{QWEN_ENGINE_URL}/v1/completions",
                                 json.dumps(probe).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    probe_s = time.time() - tp

    kill = subprocess.run(["bash", str(HERE / "qwen_prefill_kill.sh")],
                          capture_output=True, text=True, timeout=60)
    record["populate"] = {"ok": True, "prompt_tokens": tokens,
                          "probe_s": round(probe_s, 3),
                          "populate_s": round(time.time() - t0, 1),
                          "kill": kill.stdout.strip()[-200:]}
    print(f"  populate: {qwl.num_sessions} prefixes ({tokens:.0f} tok) in "
          f"{record['populate']['populate_s']}s, probe {probe_s * 1000:.0f} ms, "
          f"temp prefill killed", flush=True)
    if probe_s > 2.0:
        print("  WARNING: retrieval probe slow -- store retrieval may not be working",
              flush=True)
    record["populate"]["probe_slow"] = probe_s > 2.0


def measure_both(scenario, wl8, qwen_cmd, args, record):
    """Run the 8B client (in-process thread, the baseline path) and the Qwen client
    (subprocess) concurrently, with one combined stall watchdog."""
    box = {}
    thread = None
    if scenario != "solo":
        def work():
            try:
                box["res"] = asyncio.run(client_mod.run_load(
                    PROXY, args.model, wl8.requests, qps=args.qps,
                    max_tokens=args.max_tokens, seed=args.seed + 1,
                    timeout=args.timeout))
            except Exception as e:
                box["err"] = f"{type(e).__name__}: {e}"
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    qproc = subprocess.Popen(qwen_cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)

    hung, hung_side, stall_metrics = False, "", None
    big_log = LOGS / "disagg_decode.log"
    qwen_log = LOGS / "qwen_decode.log"
    while True:
        time.sleep(15)
        big_running = thread is not None and thread.is_alive()
        qwen_running = qproc.poll() is None
        if not big_running and not qwen_running:
            break
        for running, log, side in ((big_running, big_log, "big"),
                                   (qwen_running, qwen_log, "qwen")):
            if not running:
                continue
            try:
                idle = time.time() - log.stat().st_mtime
            except OSError:
                idle = 0
            if idle > args.stall_timeout:
                hung, hung_side = True, side
                print(f"  STALLED: no {side} engine progress for {idle:.0f}s -- killing",
                      flush=True)
                # Scrape before the kill: post-kill deltas go negative.
                stall_metrics = {k: scrape(u)
                                 for k, u in engines_alive(scenario).items()}
                stop()
                qproc.kill()
                if thread is not None:
                    thread.join(timeout=240)
                break
        if hung:
            break
    qout, _ = qproc.communicate(timeout=60)
    if "err" in box:
        raise RuntimeError(box["err"])
    return box.get("res", ([], 0.0)), hung, hung_side, stall_metrics, qout


def run_one(scenario, wl_name, wl8, qwl, args):
    util = round(args.budget / GPU_TOTAL_GIB, 4)
    name = f"infc_{scenario}_{wl_name}{args.name_suffix}"
    print(f"\n=== {name}  (8B decode util={util}, qwen util={args.qwen_util}, "
          f"qwen sessions={qwl.num_sessions}) ===", flush=True)
    record = {"name": name, "scenario": scenario, "wl": wl_name,
              "budget_gib": args.budget, "util": util,
              "qwen_sessions": qwl.num_sessions,
              "qwen_ws_gib": round(qwl.working_set_gib, 3),
              "setup": "inf_inf_coloc",
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "max_inflight": args.max_inflight,
              "qwen_max_inflight": args.qwen_max_inflight}
    GPUMON.mkdir(parents=True, exist_ok=True)
    mon_path = GPUMON / f"{name}.csv"
    mon = subprocess.Popen([sys.executable, str(REPO / "scripts/common/gpu_monitor.py"),
                            str(mon_path)])
    qwen_json = OUT / "inf_inf_coloc"
    qwen_json.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        env_extra = {
            "MAX_INFLIGHT": str(args.max_inflight),
            "QWEN_MAX_INFLIGHT": str(args.qwen_max_inflight),
            "QWEN_MODEL": args.qwen_model,
            "QWEN_UTIL": str(args.qwen_util),
            "QWEN_STORE_MB": str(int(qwl.working_set_gib * 1024) + 200),
            "MEM_GUARD_FLOOR_MB": "4000" if scenario != "A" else "8000",
        }
        if args.forward_first_token:
            env_extra["FORWARD_FIRST_TOKEN"] = "1"
        launch(scenario, util, env_extra)
        record["startup"] = {**parse_startup(scenario),
                            "startup_s": round(time.time() - t0, 1)}
        st = record["startup"]
        print(f"  8B decode KV: {st.get('big_decode_kv_gib')} GiB, "
              f"8B prefill KV: {st.get('big_prefill_kv_gib')} GiB, "
              f"qwen KV: {st.get('qwen_kv_gib')} GiB, up in {st['startup_s']}s",
              flush=True)

        # Comparability gate: the 8B engines must report the exact baseline KV grants.
        record["comparable"] = True
        if scenario != "solo":
            for key, want in (("big_decode_kv_gib", BASELINE_DECODE_KV_GIB),
                              ("big_prefill_kv_gib", BASELINE_PREFILL_KV_GIB)):
                got = st.get(key)
                if got is None or abs(got - want) > KV_TOLERANCE_GIB:
                    record["comparable"] = False
                    print(f"  *** NOT COMPARABLE: {key}={got}, baseline {want} ***",
                          flush=True)

        # solo and B are decode-only: a temporary GPU0 prefill writes every prefix into
        # the store, is killed, and the GPU1 engine only ever retrieves + decodes.
        if scenario in ("B", "solo"):
            populate_store(qwl, args, record)

        # Warmups, sequential (not measured): 8B first at the gentle L1-safe rate,
        # then the tenant -- every session's KV exists in some tier before measuring.
        if scenario != "solo":
            wu = warmup_requests(wl8)
            def warm_big():
                return asyncio.run(client_mod.run_load(
                    PROXY, args.model, wu, qps=args.warmup_qps, max_tokens=8,
                    seed=args.seed, timeout=args.timeout))
            _, wu_dur = warm_big()
            record["warmup_big"] = {"n": len(wu), "duration_s": round(wu_dur, 2)}
            print(f"  8B warmup: {len(wu)} sessions in {wu_dur:.1f}s", flush=True)
        # Qwen warmup: touch every prefix once so it is VRAM-resident (decode-only) or
        # its prefill is cached (A/C) before measuring. For decode-only this is what
        # makes the fits set fully resident -> zero retrieval during measure.
        wu_out = qwen_json / f"qwen_{name}_warmup.json"
        r = subprocess.run(qwen_runner_cmd(args, qwl, "warmup",
                                           wu_out, qwen_base_url(scenario)),
                          capture_output=True, text=True,
                          timeout=args.timeout + 300)
        if r.returncode != 0:
            raise RuntimeError(f"qwen warmup failed: {r.stdout[-300:]}{r.stderr[-300:]}")
        print(f"  {r.stdout.strip()}", flush=True)

        eps = engines_alive(scenario)
        before = {k: scrape(u) for k, u in eps.items()}
        markers_before = log_markers(scenario)
        swap_before = swap_counters()

        t_measure0 = time.time()
        record["t_measure0"] = round(t_measure0, 3)
        q_out = qwen_json / f"qwen_{name}.json"
        (results, duration), hung, hung_side, stall_metrics, qlog = measure_both(
            scenario, wl8,
            qwen_runner_cmd(args, qwl, "measure", q_out,
                            qwen_base_url(scenario)),
            args, record)
        record["t_measure1"] = round(time.time(), 3)
        after = stall_metrics or {k: scrape(u) for k, u in eps.items()}

        record["hung"], record["hung_side"] = hung, hung_side
        if hung:
            record["hung_phase"] = "measure"
        record["gpu"] = gpu_stats(mon_path, t_measure0, record["t_measure1"])
        record["swap_delta"] = {k: swap_counters().get(k, 0) - v
                                for k, v in swap_before.items()}

        if scenario != "solo":
            record["summary"] = client_mod.summarize(results, duration, args.max_tokens)
            record["records"] = [
                {k: r.get(k) for k in ("index", "session_id", "t_submit", "ttft", "e2e")}
                for r in results]
        try:
            record["qwen"] = json.loads(q_out.read_text())
        except (OSError, json.JSONDecodeError):
            record["qwen"] = {"error": f"no runner output; last stdout: {qlog[-300:]}"}

        record["metrics_delta"] = {
            role: {k: round(after.get(role, {}).get(k, 0.0)
                            - before.get(role, {}).get(k, 0.0), 4)
                   for k in set(after.get(role, {})) | set(before.get(role, {}))
                   if k != "error"}
            for role in eps}
        markers_now = log_markers(scenario)
        record["kv_markers"] = {}
        record["kv_warnings"] = {}
        for side in ("big", "qwen"):
            delta = {k: v - markers_before[side].get(k, 0)
                     for k, v in markers_now[side].items()}
            record["kv_markers"][side] = {k: v for k, v in delta.items()
                                          if v > 0 and k in INVALID_MARKERS}
            record["kv_warnings"][side] = {k: v for k, v in delta.items()
                                           if v > 0 and k in WARN_MARKERS}
        record["kv_transfer_ok"] = not (record["kv_markers"]["big"]
                                        or record["kv_markers"]["qwen"])
        record["ok"] = (not hung and record["kv_transfer_ok"]
                        and "error" not in record.get("qwen", {}))

        if scenario != "solo" and "summary" in record:
            s = record["summary"]
            print(f"  8B:   TTFT p50={s['ttft_ms']['p50']}ms p95={s['ttft_ms']['p95']}ms"
                  f" | {s['output_tok_per_s']} tok/s | failed={s['n_failed']}", flush=True)
        qs = record.get("qwen", {}).get("summary")
        if qs:
            print(f"  qwen: TTFT p50={qs['ttft_ms']['p50']}ms p95={qs['ttft_ms']['p95']}ms"
                  f" | {qs['output_tok_per_s']} tok/s | failed={qs['n_failed']}",
                  flush=True)
        g = record.get("gpu", {})
        if g:
            print(f"  gpu0 smact={g.get('gpu0_sm_active_mean')} "
                  f"dram={g.get('gpu0_dram_active_mean')} fb={g.get('gpu0_fb_used_max_gib')}"
                  f" | gpu1 smact={g.get('gpu1_sm_active_mean')} "
                  f"dram={g.get('gpu1_dram_active_mean')} fb={g.get('gpu1_fb_used_max_gib')}",
                  flush=True)
    except Exception as e:
        record.update(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"  FAILED: {record['error']}", flush=True)
    finally:
        mon.terminate()
        stop()

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def derive_row(rec):
    s = rec.get("summary", {})
    q = rec.get("qwen", {}).get("summary", {})
    st = rec.get("startup", {})
    md = rec.get("metrics_delta", {})
    g = rec.get("gpu", {})
    dur, qdur = s.get("duration_s"), q.get("duration_s")
    dec, pre, qe = md.get("big_decode", {}), md.get("big_prefill", {}), md.get("qwen", {})

    # Decode-only scenarios (B, solo) have no Qwen prefill in the measured window, so a
    # Qwen "TTFT" would be queue + KV-copy, not a first-token latency. Report it N/A.
    decode_only = rec["scenario"] in ("B", "solo")

    def pct(d, grp, k):
        return d.get(grp, {}).get(k)

    def qpct(grp, k):                 # qwen percentile, nulled when decode-only + ttft
        if decode_only and grp == "ttft_ms":
            return None
        return q.get(grp, {}).get(k)

    return {
        "name": rec["name"], "scenario": rec["scenario"], "wl": rec["wl"],
        "ok": rec.get("ok"), "comparable": rec.get("comparable"),
        "hung": rec.get("hung", False), "hung_side": rec.get("hung_side", ""),
        "hung_phase": rec.get("hung_phase", ""),
        "qwen_sessions": rec.get("qwen_sessions"),
        "qwen_ws_gib": rec.get("qwen_ws_gib"),
        "qwen_kv_grant_gib": st.get("qwen_kv_gib"),
        "big_decode_kv_gib": st.get("big_decode_kv_gib"),
        "big_prefill_kv_gib": st.get("big_prefill_kv_gib"),
        "big_ttft_p50_ms": pct(s, "ttft_ms", "p50"),
        "big_ttft_p90_ms": pct(s, "ttft_ms", "p90"),
        "big_ttft_p95_ms": pct(s, "ttft_ms", "p95"),
        "big_ttft_p99_ms": pct(s, "ttft_ms", "p99"),
        "big_ttft_mean_ms": pct(s, "ttft_ms", "mean"),
        "big_itl_p50_ms": pct(s, "itl_ms", "p50"),
        "big_tpot_p50_ms": pct(s, "tpot_ms", "p50"),
        "big_tpot_p95_ms": pct(s, "tpot_ms", "p95"),
        "big_e2e_p50_ms": pct(s, "e2e_ms", "p50"),
        "big_tok_per_s": s.get("output_tok_per_s"),
        "big_n_ok": s.get("n_ok"), "big_n_failed": s.get("n_failed"),
        "big_duration_s": dur,
        "qwen_ttft_p50_ms": qpct("ttft_ms", "p50"),
        "qwen_ttft_p90_ms": qpct("ttft_ms", "p90"),
        "qwen_ttft_p95_ms": qpct("ttft_ms", "p95"),
        "qwen_ttft_p99_ms": qpct("ttft_ms", "p99"),
        "qwen_ttft_mean_ms": qpct("ttft_ms", "mean"),
        "qwen_itl_p50_ms": pct(q, "itl_ms", "p50"),
        "qwen_tpot_p50_ms": pct(q, "tpot_ms", "p50"),
        "qwen_tpot_p95_ms": pct(q, "tpot_ms", "p95"),
        "qwen_e2e_p50_ms": pct(q, "e2e_ms", "p50"),
        "qwen_tok_per_s": q.get("output_tok_per_s"),
        "qwen_n_ok": q.get("n_ok"), "qwen_n_failed": q.get("n_failed"),
        "qwen_duration_s": qdur,
        "big_preemptions": dec.get("vllm:num_preemptions_total", 0.0),
        "qwen_preemptions": qe.get("vllm:num_preemptions_total", 0.0),
        "big_prefill_hit_rate": round(
            pre.get("vllm:prefix_cache_hits_total", 0.0)
            / pre["vllm:prefix_cache_queries_total"], 4)
        if pre.get("vllm:prefix_cache_queries_total") else None,
        "qwen_prefix_hit_rate": round(
            qe.get("vllm:prefix_cache_hits_total", 0.0)
            / qe["vllm:prefix_cache_queries_total"], 4)
        if qe.get("vllm:prefix_cache_queries_total") else None,
        "big_decode_gen_tok_s": round(dec.get("vllm:generation_tokens_total", 0.0) / dur, 1)
        if dur else None,
        "qwen_gen_tok_s": round(qe.get("vllm:generation_tokens_total", 0.0) / qdur, 1)
        if qdur else None,
        "qwen_populate_ok": rec.get("populate", {}).get("ok"),
        "qwen_populate_probe_s": rec.get("populate", {}).get("probe_s"),
        "big_kv_markers": json.dumps(rec.get("kv_markers", {}).get("big", {})),
        "qwen_kv_markers": json.dumps(rec.get("kv_markers", {}).get("qwen", {})),
        # Explicit keys (not **g): a failed run must not change the CSV column set.
        **{k: g.get(k) for gpu in (0, 1) for k in (
            f"gpu{gpu}_sm_active_mean", f"gpu{gpu}_sm_occupancy_mean",
            f"gpu{gpu}_dram_active_mean", f"gpu{gpu}_fb_used_max_gib")},
        "swap_in_pages": rec.get("swap_delta", {}).get("pswpin"),
        "swap_out_pages": rec.get("swap_delta", {}).get("pswpout"),
        "max_inflight": rec.get("max_inflight"),
        "qwen_max_inflight": rec.get("qwen_max_inflight"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", choices=["solo", "A", "B", "C"],
                    default=["solo", "B", "A", "C"])
    ap.add_argument("--wls", nargs="+", choices=["fits", "offload"],
                    default=["fits", "offload"])
    ap.add_argument("--budget", type=float, default=26.0)
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--suffix-len", type=int, default=128)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--warmup-qps", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--stall-timeout", type=float, default=180.0)
    ap.add_argument("--max-inflight", type=int, default=999)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--qwen-util", type=float, default=0.08)
    ap.add_argument("--qwen-fits-sessions", type=int, default=8)
    ap.add_argument("--qwen-offload-sessions", type=int, default=48)
    ap.add_argument("--qwen-requests", type=int, default=300)
    ap.add_argument("--qwen-qps", type=float, default=2.0)
    ap.add_argument("--qwen-seed", type=int, default=1000)
    ap.add_argument("--qwen-max-inflight", type=int, default=999)
    ap.add_argument("--no-forward-first-token", dest="forward_first_token",
                    action="store_false", default=True,
                    help="forwarding is default-ON: the baseline row is _fwd_ng")
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--zipf-a", type=float, default=1.1)
    ap.add_argument("--tag", default="inf_coloc")
    ap.add_argument("--smoke", action="store_true",
                    help="30 requests, _smoke names and tag -- cannot clobber real data")
    args = ap.parse_args()

    if args.smoke:
        args.requests = args.qwen_requests = 30
        args.name_suffix = args.name_suffix + "_smoke"
        args.tag = "inf_coloc_smoke"

    wl8 = build_workload(args.sessions, args.prefix_len, args.suffix_len, args.requests,
                         "zipf", args.zipf_a, args.seed)

    def qwen_workload(scenario, wl_name):
        n = args.qwen_fits_sessions if wl_name == "fits" else args.qwen_offload_sessions
        if scenario in ("B", "solo"):
            # decode-only: N distinct prefixes, no unique suffix, uniform reuse -> every
            # request is a whole-prefix hit (no prefill compute) that only decodes.
            return build_workload(n, args.prefix_len, 0, args.qwen_requests,
                                  "uniform", args.zipf_a, args.qwen_seed,
                                  kv_bytes_per_token=QWEN_KV_BYTES_PER_TOKEN)
        return build_workload(n, args.prefix_len, args.suffix_len, args.qwen_requests,
                              "zipf", args.zipf_a, args.qwen_seed,
                              kv_bytes_per_token=QWEN_KV_BYTES_PER_TOKEN)

    print(f"8B workload: {wl8.summary()['working_set_gib']} GiB working set, "
          f"{len(wl8.requests)} requests at {args.qps} QPS")

    OUT.mkdir(exist_ok=True)
    rows, t_start = [], time.time()
    for scenario in args.scenarios:
        for wl_name in args.wls:
            rows.append(derive_row(run_one(scenario, wl_name, wl8,
                                           qwen_workload(scenario, wl_name), args)))
            csv_path = OUT / f"summary_{args.tag}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min -> "
          f"output/summary_{args.tag}.csv", flush=True)


if __name__ == "__main__":
    main()
