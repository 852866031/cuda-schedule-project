#!/usr/bin/env python3
"""KV-slot study driver: the 8B split alone, decode on GPU1 at a VRAM budget.

Copied from inf_pf_prio/pfp_sweep.py with the 3B neighbour, MPS and priority arms
removed. ONE CELL = ONE LAUNCH (method, budget, QPS): fresh engines, store and L1 pools,
the standard warm-up (one request per session at 0.5 QPS), then the reference workload
(32 sessions x 6144 + 128 suffix, 128 out, zipf-1.1, Poisson, seed 0, 300 requests) at
--qps through the router (--forward-first-token, --max-inflight 999).

    OMP_NUM_THREADS=8 scripts/kv_vram/kv_sweep.py --probe --budgets 22 23 24 25 26 28
    OMP_NUM_THREADS=8 scripts/kv_vram/kv_sweep.py --budgets 24 --qps 2 --smoke
    OMP_NUM_THREADS=8 scripts/kv_vram/kv_sweep.py --budgets 24 26 28 --qps 1 1.5 2 --trace

Raw records: output/raw/kv_<method>_b<budget>_q<qps><suffix>.json (never overwritten).
Engine trace (--trace): output/kv_vram/trace/<name>.jsonl.gz. Summary rebuilt from raw:
output/summary_kv.csv.
"""

import argparse
import asyncio
import csv
import gzip
import json
import os
import re
import shutil
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

import client as client_mod                              # noqa: E402
from workload import build_workload, warmup_requests     # noqa: E402

OUT = REPO / "output"
RAW = OUT / "raw"
LOGS = OUT / "logs"
GPUMON = OUT / "gpumon"
KVOUT = OUT / "kv_vram"
TRACE_DIR = KVOUT / "trace"
LAUNCH = HERE / "kv_launch.sh"
STOP = HERE / "kv_stop.sh"
PID_DIR = Path("/tmp/kvv_pids")

GPU_TOTAL_GIB = 31.3536
BLOCK = 16
REQ_TOKENS = 6144 + 128 + 128                 # prompt + output = 400 blocks
# method -> env for kv_launch.sh (empty = baseline flags of the earlier studies)
METHODS = {
    "base": {},
    "apc": {"DEC_APC": "1"},
    "fp8": {"KV_DTYPE": "fp8"},
    "lw": {"LMC_LAYERWISE": "1"},
    "async": {"LMC_ASYNC_LOAD": "1"},
    # cohort-admission check (report §4.3): engines unchanged, the router releases decode
    # legs in groups of 5 (or after 2 s) so they are admitted and loaded together
    "cohort5": {"ROUTER_BATCH": "5", "ROUTER_BATCH_TIMEOUT": "2.0"},
    # compiled LMCache kernels (built in .venv-lmc, loaded via site/cops_shim.py)
    "cops": {"KV_COPS": "1"},
    "cops_lw": {"KV_COPS": "1", "LMC_LAYERWISE": "1"},
}
PORTS = {"prefill": 8100, "decode": 8200}
INVALID_MARKERS = ("RECV TIMEOUT", "kv_cache is None", "Insufficient memory",
                   "EngineDeadError", "Traceback")
WARN_MARKERS = ("Failed to allocate memory block", "Pin timeout detected", "preempt")


def scrape(port):
    out = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
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
    out = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        k, _, v = line.partition(" ")
        if k in ("pswpin", "pswpout"):
            out[k] = int(v)
    return out


def meminfo_mb(key):
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024
    return 0


def avail_mb():
    return meminfo_mb("MemAvailable")


def cpu_times():
    f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = f[3] + f[4]
    return sum(f) - idle, sum(f)


def gpu_used_mib():
    fb = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [int(x) for x in fb.stdout.split()]


def gpu_stats(csv_path, t0, t1):
    per = {}
    try:
        for line in open(csv_path):
            f = line.strip().split(",")
            if len(f) != 7 or f[0] == "ts":
                continue
            try:
                ts = float(f[0])
                if not (t0 <= ts <= t1):
                    continue
                g = per.setdefault(int(f[1]), {"n": 0, "smact": 0.0, "smocc": 0.0,
                                               "tensor": 0.0, "drama": 0.0, "fb_max": 0.0})
                g["n"] += 1
                g["smact"] += float(f[2]); g["smocc"] += float(f[3])
                g["tensor"] += float(f[4]); g["drama"] += float(f[5])
                g["fb_max"] = max(g["fb_max"], float(f[6]))
            except ValueError:
                continue
    except OSError:
        return {}
    out = {}
    for gpu in (0, 1):
        g = per.get(gpu)
        if g and g["n"]:
            out[f"gpu{gpu}_sm_active_mean"] = round(g["smact"] / g["n"], 4)
            out[f"gpu{gpu}_sm_occupancy_mean"] = round(g["smocc"] / g["n"], 4)
            out[f"gpu{gpu}_tensor_active_mean"] = round(g["tensor"] / g["n"], 4)
            out[f"gpu{gpu}_dram_active_mean"] = round(g["drama"] / g["n"], 4)
            out[f"gpu{gpu}_fb_used_max_gib"] = round(g["fb_max"] / 1024, 2)
    return out


def engine_startup(name):
    try:
        text = (LOGS / f"kvv_{name}.log").read_text(errors="replace")
    except OSError:
        return {}
    out = {}
    if m := re.findall(r"Available KV cache memory: ([\d.]+) GiB", text):
        out["kv_gib"] = float(m[-1])
    if m := re.findall(r"GPU KV cache size: ([\d,]+) tokens", text):
        out["kv_tokens"] = int(m[-1].replace(",", ""))
        out["kv_blocks"] = out["kv_tokens"] // BLOCK
        out["slots"] = out["kv_blocks"] // (REQ_TOKENS // BLOCK)
    if m := re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", text):
        out["max_concurrency_8192"] = float(m[-1])
    if m := re.findall(r"Using backend: (lmcache\.[a-z_]+)", text):
        out["lmcache_backend"] = sorted(set(m))
    out["trace_patched"] = ("[kv_trace] scheduler wrapped" in text
                            and "[kv_trace] connector start_load_kv wrapped" in text)
    if m := re.findall(r"Using (\S+) attention backend", text):
        out["attn_backend"] = m[-1]
    if m := re.findall(r"Using (\S+) backend", text):
        out["backends"] = sorted(set(m))
    return out


def marker_counts(names):
    out = {}
    for n in names:
        try:
            text = (LOGS / f"kvv_{n}.log").read_text(errors="replace")
        except OSError:
            continue
        out[n] = {m: text.count(m) for m in INVALID_MARKERS + WARN_MARKERS if text.count(m)}
    return out


def rss_mb():
    out = {}
    for pf in sorted(PID_DIR.glob("*.pid")):
        if pf.stem in ("memguard", "heartbeat"):
            continue
        try:
            pid = pf.read_text().strip()
        except OSError:
            continue
        pids = [pid] + subprocess.run(["pgrep", "-P", pid], capture_output=True,
                                      text=True).stdout.split()
        tot = 0
        for p in pids:
            try:
                for line in Path(f"/proc/{p}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1])
            except OSError:
                pass
        out[pf.stem] = round(tot / 1024)
    return out


def sh(args, env, timeout, want):
    r = subprocess.run(["bash", str(LAUNCH), *args], env=env, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0 or want not in r.stdout:
        raise RuntimeError(f"{' '.join(args)} failed: {r.stdout[-500:]}{r.stderr[-500:]}")
    return r.stdout


def stop_stack():
    r = subprocess.run(["bash", str(STOP)], capture_output=True, text=True, timeout=180)
    return r.stdout.strip()


def wait_gpus_empty(limit_mib=200, timeout=60):
    t0 = time.time()
    while True:
        used = gpu_used_mib()
        if all(u < limit_mib for u in used) or time.time() - t0 > timeout:
            return used
        time.sleep(3)


def ram_gate(what, need_mb, args, info):
    av = avail_mb()
    swap_free = min(meminfo_mb("SwapFree"), args.swap_allow_mb)
    info.setdefault("ram_gate", {})[what] = {"avail_mb": av, "need_mb": need_mb,
                                             "swap_allow_mb": swap_free}
    print(f"  RAM gate [{what}]: {av} MB available, needs ~{need_mb} MB, "
          f"swap allowance {swap_free} MB, floor {args.mem_floor_mb} MB", flush=True)
    if av + swap_free - need_mb < args.mem_floor_mb:
        raise RuntimeError(f"RAM gate: {what} needs ~{need_mb} MB, {av} MB available "
                           f"(+{swap_free} swap), floor {args.mem_floor_mb}")


def launch_env(args, trace_path=None):
    env = {**os.environ, "MEM_GUARD_FLOOR_MB": str(args.mem_floor_mb),
           **METHODS[args.method]}
    for k in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        env.pop(k, None)                     # no MPS in this study
    if trace_path:
        env.update(KV_TRACE="1", KV_TRACE_PATH=str(trace_path))
    return env


def measure(args, wl):
    """Client in a thread; stall watchdog on the engines' token counters (every KV-path
    failure is a silent hang). Prints decode Running/Waiting every 30 s."""
    box = {}

    def work():
        try:
            box["res"] = asyncio.run(client_mod.run_load(
                "http://127.0.0.1:8000", args.model, wl.requests, qps=args.qps,
                max_tokens=args.max_tokens, seed=args.seed + 1, timeout=args.timeout))
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"
    th = threading.Thread(target=work, daemon=True)
    th.start()
    last, last_change, hung, t_print = None, time.time(), False, time.time()
    while th.is_alive():
        th.join(timeout=5)
        m = {n: scrape(p) for n, p in PORTS.items()}
        tok = sum(x.get("vllm:generation_tokens_total", 0.0) + x.get("vllm:prompt_tokens_total", 0.0)
                  for x in m.values())
        if tok != last:
            last, last_change = tok, time.time()
        if time.time() - t_print >= 30:
            d = m["decode"]
            print(f"    [measure] {datetime.now():%H:%M:%S} tokens={tok:.0f} decode "
                  f"run={d.get('vllm:num_requests_running', 0):.0f} "
                  f"wait={d.get('vllm:num_requests_waiting', 0):.0f} "
                  f"kv={100 * d.get('vllm:kv_cache_usage_perc', 0):.0f}% "
                  f"avail RAM {avail_mb()} MB", flush=True)
            t_print = time.time()
        if time.time() - last_change > args.stall_timeout:
            print(f"  STALLED: no token progress for {args.stall_timeout}s -- killing",
                  flush=True)
            hung = True
            stop_stack()
            th.join(timeout=args.timeout + 60)
            break
    if "err" in box:
        raise RuntimeError(box["err"])
    return box.get("res"), hung


def wait_drained(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = scrape(PORTS["decode"])
        p = scrape(PORTS["prefill"])
        if (d.get("vllm:num_requests_running", 1) == 0 and d.get("vllm:num_requests_waiting", 1) == 0
                and p.get("vllm:num_requests_running", 1) == 0):
            return round(time.time() - t0, 1)
        time.sleep(1)
    raise RuntimeError("engines did not drain after warm-up")


def cell_name(args, budget, qps):
    return f"kv_{args.method}_b{budget:g}_q{qps:g}{'_tr' if args.trace else ''}{args.name_suffix}"


def clear_logs():
    """A launch that dies early must not leave the previous launch's logs to be parsed
    or archived under this cell's tag."""
    for lg in LOGS.glob("kvv_*.log"):
        lg.unlink()


def new_tag(name):
    tag, k = name, 1
    while (GPUMON / f"{tag}.csv").exists():
        k += 1
        tag = f"{name}_L{k}"
    return tag


def run_cell(args, budget, qps, wl):
    name = cell_name(args, budget, qps)
    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / f"{name}.json").exists() and not args.smoke:
        print(f"\n=== {name}: raw record exists -- refusing to overwrite "
              f"(use --name-suffix) ===", flush=True)
        return None
    print(f"\n=== {name}  (method {args.method}, decode {budget:g} GiB, {qps:g} QPS, "
          f"trace {'on' if args.trace else 'off'}) ===", flush=True)
    dec_util = round(budget / GPU_TOTAL_GIB, 4)
    rec = {"name": name, "method": args.method, "method_env": METHODS[args.method],
           "budget_gib": budget, "decode_util": dec_util, "prefill_util": args.prefill_util,
           "qps": qps, "requests": args.requests, "max_tokens": args.max_tokens,
           "trace": args.trace, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "mps": False, "l1_gb": float(os.environ.get("LMC_LOCAL_CPU_GB", "6")),
           "setup": "kv_vram", "launch": {},
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    GPUMON.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    tag = new_tag(name)
    rec["telemetry_tag"] = tag
    mon_path = GPUMON / f"{tag}.csv"
    mon = subprocess.Popen([sys.executable, str(HERE / "gpu_monitor.py"), str(mon_path)])
    hmon = subprocess.Popen([sys.executable, str(HERE / "host_monitor.py"),
                             str(GPUMON / f"{tag}_host.csv")])
    trace_tmp = Path(f"/dev/shm/kvv_trace_{tag}.jsonl") if args.trace else None
    if trace_tmp:
        trace_tmp.unlink(missing_ok=True)
    info = rec["launch"]
    try:
        stop_stack()
        info["gpu_used_mib_before"] = wait_gpus_empty()
        if any(u >= 200 for u in info["gpu_used_mib_before"]):
            raise RuntimeError(f"GPUs not empty before launch: {info['gpu_used_mib_before']}")
        env = launch_env(args, trace_tmp)
        clear_logs()
        sh(["guard"], env, 60, "guard up")
        t0 = time.time()
        # 24 GiB store working set + 2 pinned L1 pools + two engines' own RSS
        ram_gate("big", int((24 + 2 * rec["l1_gb"]) * 1024 + 9000), args, info)
        info["launch_stdout"] = sh(["big", f"{dec_util:.4f}", f"{args.prefill_util:.4f}"],
                                   env, 900, "big up").strip().splitlines()[0]
        info["up_s"] = round(time.time() - t0, 1)
        for n in ("prefill", "decode"):
            info[n] = engine_startup(n)
        print(f"  up in {info['up_s']:.0f}s: decode KV {info['decode'].get('kv_gib')} GiB = "
              f"{info['decode'].get('kv_tokens')} tokens = {info['decode'].get('slots')} slots; "
              f"prefill KV {info['prefill'].get('kv_gib')} GiB", flush=True)
        # the compiled kernels (.venv-lmc, site/cops_shim.py) must be in exactly the arms
        # that ask for them, in both engines; every other arm must run the fallback that
        # all baseline data used
        want = "lmcache.c_ops" if METHODS[args.method].get("KV_COPS") == "1" else "lmcache.non_cuda_equivalents"
        for n in ("prefill", "decode"):
            got = info[n].get("lmcache_backend")
            if got != [want]:
                raise RuntimeError(f"{n} engine LMCache backend {got}, expected [{want}]")
        if args.trace and not info["decode"].get("trace_patched"):
            raise RuntimeError("KV_TRACE requested but the decode log has no [kv_trace] line")
        wu = warmup_requests(wl)
        _, wu_dur = asyncio.run(client_mod.run_load(
            "http://127.0.0.1:8000", args.model, wu, qps=args.warmup_qps, max_tokens=8,
            seed=args.seed, timeout=args.timeout))
        info["warmup_s"] = round(wu_dur, 1)
        info["drain_s"] = wait_drained()
        info["avail_mb_after_warmup"] = avail_mb()
        info["rss_mb"] = rss_mb()
        info["gpu_used_mib_idle"] = gpu_used_mib()
        print(f"  warm-up {wu_dur:.0f}s, drained; avail RAM {info['avail_mb_after_warmup']} MB, "
              f"RSS {info['rss_mb']}, GPU used {info['gpu_used_mib_idle']} MiB", flush=True)

        names = list(PORTS)
        before = {n: scrape(p) for n, p in PORTS.items()}
        mk_before = marker_counts(names)
        sw_before, cpu0 = swap_counters(), cpu_times()
        t_m0, t_mono0 = time.time(), time.monotonic()
        res, hung = measure(args, wl)
        t_m1, t_mono1 = time.time(), time.monotonic()
        after = {n: scrape(p) for n, p in PORTS.items()} if not hung else before
        cpu1 = cpu_times()
        rec.update(t_measure0=round(t_m0, 3), t_measure1=round(t_m1, 3),
                   t_mono_measure0=round(t_mono0, 4), t_mono_measure1=round(t_mono1, 4),
                   hung=hung)
        rec["gpu"] = gpu_stats(mon_path, t_m0 + 3, t_m1)
        rec["swap_delta"] = {k: swap_counters()[k] - v for k, v in sw_before.items()}
        rec["host_cpu_busy"] = round((cpu1[0] - cpu0[0]) / max(1, cpu1[1] - cpu0[1]), 4)
        rec["avail_mb_end"] = avail_mb()
        rec["rss_mb_end"] = rss_mb()
        rec["metrics_delta"] = {
            n: {k: round(after[n].get(k, 0.0) - before[n].get(k, 0.0), 4)
                for k in set(after[n]) | set(before[n])
                if k != "error" and "_bucket" not in k} for n in names}
        mk_after = marker_counts(names)
        delta = {n: {k: v - mk_before.get(n, {}).get(k, 0) for k, v in mk_after.get(n, {}).items()
                     if v - mk_before.get(n, {}).get(k, 0) > 0} for n in names}
        rec["markers"] = {n: {k: v for k, v in d.items() if k in INVALID_MARKERS}
                          for n, d in delta.items()}
        rec["warnings"] = {n: {k: v for k, v in d.items() if k in WARN_MARKERS}
                           for n, d in delta.items()}
        n_failed = 0
        if res is not None:
            results, duration = res
            rec["summary"] = client_mod.summarize(results, duration, args.max_tokens)
            rec["records"] = [{"index": r["index"], "session_id": r["session_id"],
                               "t_submit": round(r["t_submit"], 5),
                               "ttft": r["ttft"] and round(r["ttft"], 5),
                               "e2e": r["e2e"] and round(r["e2e"], 5), "ok": r["ok"],
                               "error": r["error"], "n_chunks": r["n_chunks"],
                               "itls": [round(x, 5) for x in r["itls"]]} for r in results]
            n_failed = rec["summary"]["n_failed"]
            rec["n_short"] = rec["summary"].get("n_short", 0)
            s = rec["summary"]
            print(f"  TTFT p50={s['ttft_ms']['p50']} p95={s['ttft_ms']['p95']} ms | "
                  f"TPOT p50={s['tpot_ms']['p50']} (decode-only {s['tpot2_ms']['p50']}) ms | "
                  f"ITL p50/mean={s['itl_ms']['p50']}/{s['itl_ms']['mean']} ms | "
                  f"{s['output_tok_per_s']} tok/s | failed={n_failed}", flush=True)
        md = rec["metrics_delta"]["decode"]
        q = md.get("vllm:request_queue_time_seconds_count")
        if q:
            print(f"  decode: queue {1000 * md['vllm:request_queue_time_seconds_sum'] / q:.0f} ms, "
                  f"KV load {1000 * md.get('vllm:request_prefill_time_seconds_sum', 0) / q:.0f} ms, "
                  f"decode {1000 * md.get('vllm:request_decode_time_seconds_sum', 0) / q:.0f} ms "
                  f"(means over {q:.0f}); preemptions {md.get('vllm:num_preemptions_total', 0):.0f}",
                  flush=True)
        # every decode leg must have found the session prefix in LMCache (or, with decode
        # APC, in its own cache): a silent re-prefill would read as an ok cell
        qry = md.get("vllm:prompt_tokens_total", 0.0)
        hit = (md.get("vllm:external_prefix_cache_hits_total", 0.0)
               + md.get("vllm:prefix_cache_hits_total", 0.0))
        rec["decode_hit_frac"] = round(hit / qry, 4) if qry else None
        rec["n_failed"] = n_failed
        # every request's tokens, counted by the decode engine itself (chunks can merge)
        rec["decode_gen_tokens"] = md.get("vllm:generation_tokens_total")
        rec["tokens_complete"] = rec["decode_gen_tokens"] == args.requests * args.max_tokens
        rec["ok"] = (not hung and n_failed == 0 and not any(rec["markers"].values())
                     and (rec["decode_hit_frac"] or 0) >= 0.95 and rec["tokens_complete"])
        print(f"  decode prefix hit fraction {rec['decode_hit_frac']} (6144/6272 = 0.980 expected); "
              f"decode tokens {rec['decode_gen_tokens']} (complete: {rec['tokens_complete']})",
              flush=True)
        g = rec["gpu"]
        print(f"  gpu1 smact={g.get('gpu1_sm_active_mean')} dram={g.get('gpu1_dram_active_mean')} "
              f"fb={g.get('gpu1_fb_used_max_gib')} | swap {rec['swap_delta']} | host cpu "
              f"{rec['host_cpu_busy']:.2f} | avail {rec['avail_mb_end']} MB | "
              f"markers {rec['markers']} warnings {rec['warnings']} | ok={rec['ok']}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"  FAILED: {rec['error']}", flush=True)
    finally:
        arch = LOGS / "kvv_archive" / tag
        arch.mkdir(parents=True, exist_ok=True)
        for lg in LOGS.glob("kvv_*.log"):
            (arch / lg.name).write_bytes(lg.read_bytes())
        rec["gpu_after_stop"] = stop_stack().splitlines()[-2:]
        mon.terminate()
        hmon.terminate()
        if trace_tmp and trace_tmp.exists():
            dst = TRACE_DIR / f"{tag}.jsonl.gz"
            with open(trace_tmp, "rb") as fi, gzip.open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            trace_tmp.unlink()
            rec["trace_file"] = str(dst.relative_to(REPO))
    (RAW / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec


def run_probe(args):
    """E0: KV grant per budget. Launch, read the decode log, stop, verify the GPU is
    empty before the next budget."""
    KVOUT.mkdir(parents=True, exist_ok=True)
    path = KVOUT / f"probe_{args.method}{args.name_suffix}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for b in args.budgets:
        print(f"\n=== probe {args.method} b{b:g} ===", flush=True)
        r = {"budget_gib": b, "decode_util": round(b / GPU_TOTAL_GIB, 4),
             "method_env": METHODS[args.method]}
        try:
            stop_stack()
            r["gpu_used_mib_before"] = wait_gpus_empty()
            if any(u >= 200 for u in r["gpu_used_mib_before"]):
                raise RuntimeError(f"GPUs not empty: {r['gpu_used_mib_before']}")
            env = launch_env(args)
            clear_logs()
            sh(["guard"], env, 60, "guard up")
            l1 = float(os.environ.get("LMC_LOCAL_CPU_GB", "6"))
            ram_gate("big", int((24 + 2 * l1) * 1024 + 9000), args, r)
            t0 = time.time()
            sh(["big", f"{r['decode_util']:.4f}", f"{args.prefill_util:.4f}"], env, 900, "big up")
            r["up_s"] = round(time.time() - t0, 1)
            r["decode"] = engine_startup("decode")
            r["prefill"] = engine_startup("prefill")
            r["gpu_used_mib_idle"] = gpu_used_mib()
            d = r["decode"]
            print(f"  decode KV {d.get('kv_gib')} GiB, {d.get('kv_tokens')} tokens, "
                  f"{d.get('kv_blocks')} blocks -> {d.get('slots')} whole requests "
                  f"({d.get('kv_blocks', 0) / 400:.2f}); GPU used {r['gpu_used_mib_idle']} MiB; "
                  f"up {r['up_s']:.0f}s", flush=True)
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {r['error']}", flush=True)
        finally:
            arch = LOGS / "kvv_archive" / f"probe_{args.method}_b{b:g}{args.name_suffix}"
            arch.mkdir(parents=True, exist_ok=True)
            for lg in LOGS.glob("kvv_*.log"):
                (arch / lg.name).write_bytes(lg.read_bytes())
            stop_stack()
            r["gpu_used_mib_after"] = wait_gpus_empty()
            print(f"  after stop: GPU used {r['gpu_used_mib_after']} MiB", flush=True)
        out[f"b{b:g}"] = r
        path.write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------- summary
def _mean(md, base):
    c = md.get(f"{base}_count")
    return round(1000 * md.get(f"{base}_sum", 0.0) / c, 1) if c else None


def rebuild_summary():
    rows = []
    for path in sorted(RAW.glob("kv_*.json")):
        if "_smoke" in path.name:
            continue
        rec = json.loads(path.read_text())
        s = rec.get("summary") or {}
        md = (rec.get("metrics_delta") or {}).get("decode", {})
        g, L = rec.get("gpu") or {}, rec.get("launch") or {}

        def p(grp, k):
            return (s.get(grp) or {}).get(k)
        rows.append({
            "name": rec["name"], "method": rec.get("method"), "budget_gib": rec.get("budget_gib"),
            "qps": rec.get("qps"), "trace": rec.get("trace"), "ok": rec.get("ok"),
            "error": rec.get("error", ""), "n_failed": rec.get("n_failed"),
            "decode_kv_gib": (L.get("decode") or {}).get("kv_gib"),
            "decode_kv_tokens": (L.get("decode") or {}).get("kv_tokens"),
            "slots": (L.get("decode") or {}).get("slots"),
            "lmcache_backend": ",".join((L.get("decode") or {}).get("lmcache_backend") or []),
            "attn_backend": (L.get("decode") or {}).get("attn_backend"),
            "decode_hit_frac": rec.get("decode_hit_frac"),
            "tok_per_s": s.get("output_tok_per_s"), "req_per_s": s.get("request_tok_per_s"),
            "ttft_p50_ms": p("ttft_ms", "p50"), "ttft_p95_ms": p("ttft_ms", "p95"),
            "tpot_p50_ms": p("tpot_ms", "p50"), "tpot_p95_ms": p("tpot_ms", "p95"),
            "tpot2_p50_ms": p("tpot2_ms", "p50"), "tpot2_p95_ms": p("tpot2_ms", "p95"),
            "itl_p50_ms": p("itl_ms", "p50"), "itl_mean_ms": p("itl_ms", "mean"),
            "itl_p95_ms": p("itl_ms", "p95"), "itl2_p99_ms": p("itl2_ms", "p99"),
            "gap1_p50_ms": p("gap1_ms", "p50"), "gap1_p95_ms": p("gap1_ms", "p95"),
            "e2e_p50_ms": p("e2e_ms", "p50"),
            "decode_queue_ms": _mean(md, "vllm:request_queue_time_seconds"),
            "decode_load_ms": _mean(md, "vllm:request_prefill_time_seconds"),
            "decode_time_ms": _mean(md, "vllm:request_decode_time_seconds"),
            "prefill_queue_ms": _mean((rec.get("metrics_delta") or {}).get("prefill", {}),
                                      "vllm:request_queue_time_seconds"),
            "preemptions": md.get("vllm:num_preemptions_total"),
            **{k: g.get(k) for k in ("gpu1_sm_active_mean", "gpu1_dram_active_mean",
                                     "gpu1_fb_used_max_gib", "gpu0_sm_active_mean")},
            "host_cpu_busy": rec.get("host_cpu_busy"),
            "swap_in_pages": (rec.get("swap_delta") or {}).get("pswpin"),
            "swap_out_pages": (rec.get("swap_delta") or {}).get("pswpout"),
            "avail_mb_end": rec.get("avail_mb_end"),
            "warnings": sum(sum(d.values()) for d in (rec.get("warnings") or {}).values()),
            "omp_num_threads": rec.get("omp_num_threads"),
            "telemetry_tag": rec.get("telemetry_tag"),
        })
    if rows:
        with open(OUT / "summary_kv.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHODS), default="base")
    ap.add_argument("--budgets", type=float, nargs="+", required=True, help="decode GiB")
    ap.add_argument("--qps", type=float, nargs="+", default=[2.0])
    ap.add_argument("--probe", action="store_true", help="E0: KV grant per budget only")
    ap.add_argument("--trace", action="store_true", help="E1 record-only engine trace")
    ap.add_argument("--prefill-util", type=float, default=0.80)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--warmup-qps", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--stall-timeout", type=float, default=180.0)
    ap.add_argument("--mem-floor-mb", type=int, default=4000)
    ap.add_argument("--swap-allow-mb", type=int, default=3000)
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="30 requests, _smoke names -- cannot clobber real data")
    args = ap.parse_args()
    if not os.environ.get("OMP_NUM_THREADS"):
        ap.error("export OMP_NUM_THREADS (8 for the two-engine stack)")
    if args.probe:
        run_probe(args)
        print("done", flush=True)
        return
    if args.smoke:
        args.requests = 30
        args.name_suffix += "_smoke"
    wl = build_workload(32, 6144, 128, args.requests, "zipf", 1.1, args.seed)
    for b in args.budgets:
        for q in args.qps:
            a = argparse.Namespace(**vars(args))
            a.qps = q
            run_cell(a, b, q, wl)
            rebuild_summary()
    print("done", flush=True)


if __name__ == "__main__":
    main()
