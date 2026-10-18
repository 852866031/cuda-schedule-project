#!/usr/bin/env python3
"""Ninth study: who gets priority on the shared GPU -- the 8B decode or a 3B prefill (P2).

Eighth study. Self-contained: workload.py/client.py/lmc_proxy.py are copies; the store
wrapper adds a size cap (lmc_server_main.py). One CELL = one launch:

  base  8B split alone (prefill GPU0, decode GPU1 at --budget, KV in the DRAM store)
  s1    3B alone, P1: prefill only on GPU1, KV dropped
  s2    3B alone, P2: prefill on GPU1, KV written to its DRAM store
  s3    3B alone, P3: prefill on GPU1 -> store -> decode on GPU0
  p1 p2 p3   the same three placements beside the 8B stack

The 8B runs the reference workload (32 sessions x 6144 + 128 suffix, 128 out, zipf-1.1,
2 QPS) through its router, in-process -- the client path of every earlier 8B number. The
3B gets a fresh 6144-token prompt per request (sm_client_runner.py, own process) at
--sm-qps; max_tokens 1 in P1/P2 (no decode -> no TPOT), 128 in P3. Both clients start at
the same instant. Everything runs under MPS (the driver owns the daemon).

    OMP_NUM_THREADS=8 scripts/inf_pf_prio/pfp_sweep.py --cells s1 --smoke
    OMP_NUM_THREADS=8 scripts/inf_pf_prio/pfp_sweep.py --cells base s1 s2 s3 p1 p2 p3
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

import client as client_mod                              # noqa: E402
from workload import build_workload, warmup_requests     # noqa: E402

OUT = REPO / "output"
RAW = OUT / "raw"
LOGS = OUT / "logs"
GPUMON = OUT / "gpumon"
CLIENT_OUT = OUT / "inf_pf_prio"
LAUNCH = HERE / "pfp_launch.sh"
STOP = HERE / "pfp_stop.sh"
PID_DIR = Path("/tmp/pfp_pids")
MPS_PIPE = "/tmp/pfp_mps_pipe"
MPS_LOG = "/tmp/pfp_mps_log"

GIB = 1 << 30
GPU_TOTAL_GIB = 31.3536
SM_KV_BYTES = 36 * 2 * 128 * 2 * 2      # Qwen2.5-3B: 36 KiB/token
CELLS = {   # name -> (8B present, 3B placement)
    "base": (True, None),
    "s1": (False, "P1"), "s2": (False, "P2"), "s3": (False, "P3"),
    "p1": (True, "P1"), "p2": (True, "P2"), "p3": (True, "P3"),
}
# Who has priority on GPU1 (8B decode vs 3B prefill), and by which mechanism.
ARMS = {
    "fair":    {},                                          # MPS, equal default share
    "mps8b":   {"SM_MPS_LOW": "1"},                         # MPS client priority: 8B first
    "gate8b":  {"BIG_DEC_ROLE": "hp", "SM_ROLE": "be"},     # 3B launches wait while 8B busy
    "mps3b":   {"BIG_DEC_MPS_LOW": "1"},                    # MPS client priority: 3B first
    "gate3b":  {"SM_ROLE": "hp", "BIG_DEC_ROLE": "be"},     # 8B launches wait while 3B busy
    "eager8b": {"BIG_DEC_EAGER": "1"},                      # control: 8B decode eager, no gate
    # supplement: gate8b with the tightest launch backlog (credit event every launch, at
    # most one unretired) -- bounds how much 3B work can sit in front of a decode step
    "gate8bt": {"BIG_DEC_ROLE": "hp", "SM_ROLE": "be", "GATE_K": "1", "GATE_MAXPEND": "1"},
}
PORTS = {"big_prefill": 8100, "big_decode": 8200,
         "small_prefill": 8111, "small_decode": 8211}
INVALID_MARKERS = ("RECV TIMEOUT", "kv_cache is None", "Insufficient memory",
                   "Peer Out Of Memory", "EngineDeadError")
WARN_MARKERS = ("Failed to allocate memory block", "Pin timeout detected")


def engines_of(cell):
    big, pl = CELLS[cell]
    out = ["big_prefill", "big_decode"] if big else []
    if pl:
        out.append("small_prefill")
    if pl == "P3":
        out.append("small_decode")
    return out


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


def gpu_stats(csv_path, t0, t1):
    per = {}
    try:
        for line in open(csv_path):
            if "\x00" in line:
                continue
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
        text = (LOGS / f"pfp_{name}.log").read_text(errors="replace")
    except OSError:
        return {}
    out = {}
    if m := re.findall(r"Available KV cache memory: ([\d.]+) GiB", text):
        out["kv_gib"] = float(m[-1])
    if m := re.findall(r"GPU KV cache size: ([\d,]+) tokens", text):
        out["kv_tokens"] = int(m[-1].replace(",", ""))
    return out


def marker_counts(names):
    out = {}
    for n in names:
        try:
            text = (LOGS / f"pfp_{n}.log").read_text(errors="replace")
        except OSError:
            continue
        out[n] = {m: text.count(m) for m in INVALID_MARKERS + WARN_MARKERS if text.count(m)}
    return out


def rss_mb():
    """Host RSS per process tree (API server + its EngineCore children; stores)."""
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


def store_stats():
    """Last STORE line of the 3B store (cap, evictions, late gets)."""
    try:
        lines = [x for x in (LOGS / "pfp_store_small.log").read_text(errors="replace")
                 .splitlines() if x.startswith("STORE ")]
    except OSError:
        return {}
    if not lines:
        return {}
    return {k: int(v) for k, v in (kv.split("=") for kv in lines[-1].split()[1:])}


def save_stats(n_last):
    """3B prefill engine: LMCache's GPU->host copy per request (sync: 'offload_time';
    async patch: ASYNC_SAVE lines) over the last n_last requests."""
    try:
        text = (LOGS / "pfp_small_prefill.log").read_text(errors="replace")
    except OSError:
        return {}
    out = {}
    off = [float(x) for x in re.findall(r"offload_time: ([\d.]+) ms", text)][-n_last:]
    if off:
        out["offload_ms_p50"] = round(sorted(off)[len(off) // 2], 1)
    a = re.findall(r"ASYNC_SAVE reqs=\d+ wait_ms=([\d.]+) copy_ms=([\d.]+) put_ms=[\d.]+ backlog=(\d+)", text)[-n_last:]
    if a:
        out["async_jobs"] = len(a)
        out["async_wait_ms_p50"] = round(sorted(float(x[0]) for x in a)[len(a) // 2], 1)
        out["async_copy_ms_p50"] = round(sorted(float(x[1]) for x in a)[len(a) // 2], 1)
        out["async_backlog_max"] = max(int(x[2]) for x in a)
    out["async_errors"] = text.count("ASYNC_SAVE ERROR")
    return out


def decode_hits(n_last, prompt_len):
    """3B decode engine (P3): how many of the last n_last requests found their whole
    prompt in LMCache. A miss means the decode leg arrived before the KV and silently
    re-prefilled on GPU0."""
    try:
        text = (LOGS / "pfp_small_decode.log").read_text(errors="replace")
    except OSError:
        return {}
    first = {}                      # the line repeats every step: first one per request
    for rid, h in re.findall(r"Reqid: (\S+), Total tokens \d+, Inference Engine computed "
                             r"tokens: \d+, LMCache hit tokens: (\d+)", text):
        first.setdefault(rid, int(h))
    hits = list(first.values())[-n_last:]
    full = sum(h >= prompt_len - 256 for h in hits)
    return {"n": len(hits), "full_hits": full, "partial_or_miss": len(hits) - full,
            "zero_hits": sum(h == 0 for h in hits)}


async def _verify_texts(args, url, n=8, max_tokens=32):
    import aiohttp
    wl = build_workload(1, 16, args.sm_prompt_len - 16, n, "uniform", 1.1, args.sm_seed + 13,
                        kv_bytes_per_token=SM_KV_BYTES)
    out = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as sess:
        for req in wl.requests:              # one at a time, through the 3B's router
            body = {"model": args.sm_model, "prompt": req.token_ids, "max_tokens": max_tokens,
                    "temperature": 0.0, "stream": True, "ignore_eos": True, "logprobs": 1}
            text, lps = "", []
            async with sess.post(f"{url}/v1/completions", json=body) as r:
                async for raw in r.content:
                    line = raw.decode().strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        try:
                            ch = json.loads(line[5:])["choices"][0]
                            text += ch["text"]
                            lps += [round(x, 4) for x in
                                    ((ch.get("logprobs") or {}).get("token_logprobs") or [])]
                        except (ValueError, KeyError, IndexError):
                            pass
            out.append({"text": text, "logprobs": lps})
    return out


def verify_texts(args, url="http://127.0.0.1:8001"):
    """P3 only: 8 fixed prompts through the split, greedy, 32 tokens. Tokens 2+ are
    decoded from the KV the prefill engine stored, so the texts must equal those of a
    sync-save run of the same prompts -- the content check for the async-save patch."""
    return asyncio.run(_verify_texts(args, url))


# ------------------------------------------------------------------------- MPS
def mps_env():
    return {"CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE, "CUDA_MPS_LOG_DIRECTORY": MPS_LOG}


def mps_start():
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    # A daemon from the previous cell may still be shutting down: retry, and clear its
    # stale pipe files only when no daemon is running.
    for attempt in range(6):
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **mps_env()})
        if r.returncode == 0:
            break
        time.sleep(5)
        if subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control -d"],
                          capture_output=True).returncode != 0:
            for fn in os.listdir(MPS_PIPE):
                try:
                    os.unlink(os.path.join(MPS_PIPE, fn))
                except OSError:
                    pass
    else:
        raise RuntimeError("MPS control daemon would not start")
    time.sleep(2)
    if subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"],
                      capture_output=True).returncode != 0:
        raise RuntimeError("MPS control daemon did not start")
    os.environ.update(mps_env())
    print("  MPS daemon up", flush=True)


def mps_stop():
    subprocess.run(["bash", "-c", "echo quit | nvidia-cuda-mps-control"],
                   env={**os.environ, **mps_env()}, capture_output=True, timeout=60)
    for k in mps_env():
        os.environ.pop(k, None)
    time.sleep(2)


# ---------------------------------------------------------------------- launch
def sh(args, env, timeout, want):
    r = subprocess.run(["bash", str(LAUNCH), *args], env=env, capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0 or want not in r.stdout:
        raise RuntimeError(f"{' '.join(args)} failed: {r.stdout[-500:]}{r.stderr[-500:]}")


def ram_gate(what, need_mb, args, info):
    """Refuse a launch step that would push available RAM under the guard floor. The
    swap allowance covers cold 8B store pages only (swappable; pinned L1s are not)."""
    av = avail_mb()
    swap_free = min(meminfo_mb("SwapFree"), args.swap_allow_mb)
    info.setdefault("ram_gate", {})[what] = {"avail_mb": av, "need_mb": need_mb,
                                             "swap_allow_mb": swap_free}
    print(f"  RAM gate [{what}]: {av} MB available, needs ~{need_mb} MB, "
          f"swap allowance {swap_free} MB, floor {args.mem_floor_mb} MB", flush=True)
    if av + swap_free - need_mb < args.mem_floor_mb:
        raise RuntimeError(f"RAM gate: {what} needs ~{need_mb} MB, {av} MB available "
                           f"(+{swap_free} swap), floor {args.mem_floor_mb}")


def sm_runner_cmd(args, cell, phase, out, start_at=0.0):
    pl = CELLS[cell][1]
    url = "http://127.0.0.1:8001" if pl == "P3" else "http://127.0.0.1:8111"
    return [str(REPO / ".venv" / "bin" / "python"), str(HERE / "sm_client_runner.py"),
            "--base-url", url, "--model", args.sm_model,
            "--prompt-len", str(args.sm_prompt_len), "--kv-bytes", str(SM_KV_BYTES),
            "--requests", str(args.sm_requests), "--qps", str(args.sm_qps),
            "--seed", str(args.sm_seed),
            "--max-tokens", str(args.max_tokens if pl == "P3" else 1),
            "--timeout", str(args.timeout), "--start-at", f"{start_at:.3f}",
            "--phase", phase, "--out", str(out)]


def measure(args, cell, wl8, name):
    """8B client (in-process thread) and 3B client (subprocess) over one aligned window,
    with a stall watchdog on the engines' token counters (every KV-path failure is a
    silent hang)."""
    big, pl = CELLS[cell]
    start_at = time.time() + 3.0
    box, thread, proc, q_out = {}, None, None, None
    if big:
        def work():
            try:
                time.sleep(max(0.0, start_at - time.time()))
                box["res"] = asyncio.run(client_mod.run_load(
                    "http://127.0.0.1:8000", args.model, wl8.requests, qps=args.qps,
                    max_tokens=args.max_tokens, seed=args.seed + 1, timeout=args.timeout))
            except Exception as e:
                box["err"] = f"{type(e).__name__}: {e}"
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
    if pl:
        q_out = CLIENT_OUT / f"{name}_small.json"
        q_out.unlink(missing_ok=True)
        proc = subprocess.Popen(sm_runner_cmd(args, cell, "measure", q_out, start_at),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    names = engines_of(cell)
    last, last_change, hung, t_print = None, time.time(), False, time.time()
    while (thread is not None and thread.is_alive()) or (proc is not None and proc.poll() is None):
        time.sleep(5)
        tok = 0.0
        for n in names:
            m = scrape(PORTS[n])
            tok += m.get("vllm:generation_tokens_total", 0.0) + m.get("vllm:prompt_tokens_total", 0.0)
        if tok != last:
            last, last_change = tok, time.time()
        if time.time() - t_print >= 60:
            print(f"    [measure] {datetime.now():%H:%M:%S} tokens={tok:.0f} "
                  f"avail RAM {avail_mb()} MB", flush=True)
            t_print = time.time()
        if time.time() - last_change > args.stall_timeout:
            print(f"  STALLED: no token progress for {args.stall_timeout}s -- killing",
                  flush=True)
            hung = True
            subprocess.run(["bash", str(STOP)], capture_output=True, timeout=180)
            if proc is not None:
                proc.kill()
            if thread is not None:
                thread.join(timeout=args.timeout + 60)
            break
    small = None
    if proc is not None:
        stdout, _ = proc.communicate(timeout=60)
        try:
            small = json.loads(q_out.read_text())
        except (OSError, json.JSONDecodeError):
            small = {"error": f"no runner output; {stdout[-300:] if stdout else ''}"}
    if "err" in box:
        raise RuntimeError(box["err"])
    return box.get("res"), small, hung


def run_cell(args, cell):
    big, pl = CELLS[cell]
    qtag = f"_q{args.sm_qps:g}" if pl else ""
    asave = pl in ("P2", "P3")              # this study: P2 always with the async save
    name = f"pfp_{cell}_{args.arm}{qtag}{args.name_suffix}"
    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / f"{name}.json").exists() and not args.smoke:
        print(f"\n=== {name}: raw record exists -- refusing to overwrite "
              f"(use --name-suffix) ===", flush=True)
        return None
    print(f"\n=== {name}  (8B {'on' if big else 'off'}, 3B {pl or 'off'}"
          f"{f' at {args.sm_qps:g} QPS' if pl else ''}) ===", flush=True)
    dec_util = round(args.budget / GPU_TOTAL_GIB, 4)
    rec = {"name": name, "cell": cell, "big": big, "placement": pl,
           "budget_gib": args.budget, "decode_util": dec_util,
           "prefill_util": args.prefill_util, "big_qps": args.qps if big else None,
           "sm_model": args.sm_model if pl else None, "sm_qps": args.sm_qps if pl else None,
           "sm_prompt_len": args.sm_prompt_len, "sm_util": args.sm_util,
           "sm_store_cap_gb": args.sm_store_cap_gb if pl in ("P2", "P3") else None,
           "sm_max_tokens": (args.max_tokens if pl == "P3" else 1) if pl else None,
           "requests": args.requests, "sm_requests": args.sm_requests,
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mps": True,
           "big_l1_gb": float(os.environ.get("LMC_LOCAL_CPU_GB", "6")),
           "sm_quant": args.sm_quant, "sm_async_save": asave, "sm_first": args.sm_first, "sm_quant": args.sm_quant, "arm": args.arm, "arm_env": ARMS[args.arm],
           "setup": "inf_pf_prio", "launch": {},
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    GPUMON.mkdir(parents=True, exist_ok=True)
    CLIENT_OUT.mkdir(parents=True, exist_ok=True)
    tag, k = name, 1
    while (GPUMON / f"{tag}.csv").exists() and not args.smoke:
        k += 1
        tag = f"{name}_L{k}"
    rec["telemetry_tag"] = tag
    mon_path = GPUMON / f"{tag}.csv"
    mon = subprocess.Popen([sys.executable, str(HERE / "gpu_monitor.py"),
                            str(mon_path)])
    hmon = subprocess.Popen([sys.executable, str(HERE / "host_monitor.py"),
                             str(GPUMON / f"{tag}_host.csv")])
    info = rec["launch"]
    try:
        subprocess.run(["bash", str(STOP)], capture_output=True, timeout=180)
        mps_start()
        env = {**os.environ, "MEM_GUARD_FLOOR_MB": str(args.mem_floor_mb),
               "SM_MODEL": args.sm_model, "SM_UTIL": str(args.sm_util),
               "SM_STORE_CAP_GB": str(args.sm_store_cap_gb),
               "SM_QUANT": args.sm_quant, "SM_PF_BATCHED": str(args.sm_prompt_len + 16), "SM_MAX_LEN": str(args.sm_prompt_len + 16), "SM_MAX_NUM_SEQS": str(args.sm_max_seqs),
               **ARMS[args.arm]}
        if asave:
            env["SM_ASYNC_SAVE"] = "1"
        sh(["guard"], env, 60, "guard up")
        t0 = time.time()
        # fp8 3B: online quantization loads the bf16 weights first (startup peak ~9 GiB,
        # steady 4.8 GiB), so with --sm-first it starts on the empty GPU, before the 8B.
        for step in (("small", "big") if args.sm_first else ("big", "small")):
            if step == "big" and big:
                # 24 GiB store working set + 2 pinned L1 pools + two engines' own RSS
                ram_gate("big", int((24 + 2 * rec["big_l1_gb"]) * 1024 + 9000), args, info)
                sh(["big", f"{dec_util:.4f}", f"{args.prefill_util:.4f}"], env, 900, "big up")
                for n in ("big_prefill", "big_decode"):
                    info[n] = engine_startup(n)
                print(f"  8B up in {time.time() - t0:.0f}s: prefill KV "
                      f"{info['big_prefill'].get('kv_gib')} GiB, decode KV "
                      f"{info['big_decode'].get('kv_gib')} GiB", flush=True)
                # 8B warmup BEFORE the 3B launches: the store is then full (24 GiB), so the
                # 3B's RAM gate sees the real headroom.
                wu = warmup_requests(wl8 := args.wl8)
                _, wu_dur = asyncio.run(client_mod.run_load(
                    "http://127.0.0.1:8000", args.model, wu, qps=args.warmup_qps, max_tokens=8,
                    seed=args.seed, timeout=args.timeout))
                info["big_warmup_s"] = round(wu_dur, 1)
                info["avail_mb_big_warm"] = avail_mb()
                info["rss_mb_big_warm"] = rss_mb()
                print(f"  8B warmup: {len(wu)} sessions in {wu_dur:.0f}s; avail RAM "
                      f"{info['avail_mb_big_warm']} MB; RSS {info['rss_mb_big_warm']}", flush=True)
            if step == "small" and pl:
                need = {"P1": 4500, "P2": 5500, "P3": 11000}[pl]
                if pl != "P1":
                    need += int(args.sm_store_cap_gb * 1024)
                ram_gate("small", need, args, info)
                t1 = time.time()
                sh(["small", pl], env, 900, "small up")
                info["small_prefill"] = engine_startup("small_prefill")
                if pl == "P3":
                    info["small_decode"] = engine_startup("small_decode")
                print(f"  3B up in {time.time() - t1:.0f}s: prefill KV "
                      f"{info['small_prefill'].get('kv_gib')} GiB"
                      + (f", decode KV {info['small_decode'].get('kv_gib')} GiB"
                         if pl == "P3" else ""), flush=True)
        # 3B warmup last, right before measuring: with --sm-first the 3B sat idle while the
        # 8B launched and filled its store, and its pages may have been swapped out.
        if pl:
            r = subprocess.run(sm_runner_cmd(args, cell, "warmup",
                                             CLIENT_OUT / f"{name}_small_warmup.json"),
                               capture_output=True, text=True, timeout=args.timeout + 300)
            if r.returncode != 0 or "n_failed=0" not in r.stdout:
                raise RuntimeError(f"3B warmup failed: {r.stdout[-300:]}{r.stderr[-300:]}")
            print(f"  {r.stdout.strip()}", flush=True)
        info["avail_mb_after_launch"] = avail_mb()
        info["rss_mb"] = rss_mb()
        fb = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
        info["gpu_used_mib_idle"] = [int(x) for x in fb.stdout.split()]
        print(f"  all up: avail RAM {info['avail_mb_after_launch']} MB, RSS {info['rss_mb']}, "
              f"GPU used {info['gpu_used_mib_idle']} MiB", flush=True)

        names = engines_of(cell)
        before = {n: scrape(PORTS[n]) for n in names}
        mk_before = marker_counts(names)
        sw_before, cpu0 = swap_counters(), cpu_times()
        st_before = store_stats()
        t_m0 = time.time()
        res8, small, hung = measure(args, cell, args.wl8, name)
        t_m1 = time.time()
        after = {n: scrape(PORTS[n]) for n in names} if not hung else before
        cpu1 = cpu_times()
        rec.update(t_measure0=round(t_m0, 3), t_measure1=round(t_m1, 3), hung=hung)
        rec["gpu"] = gpu_stats(mon_path, t_m0 + 3, t_m1)
        rec["swap_delta"] = {k: swap_counters()[k] - v for k, v in sw_before.items()}
        rec["host_cpu_busy"] = round((cpu1[0] - cpu0[0]) / max(1, cpu1[1] - cpu0[1]), 4)
        rec["avail_mb_end"] = avail_mb()
        rec["rss_mb_end"] = rss_mb()
        rec["metrics_delta"] = {
            n: {k: round(after[n].get(k, 0.0) - before[n].get(k, 0.0), 4)
                for k in set(after[n]) | set(before[n]) if k != "error"} for n in names}
        mk_after = marker_counts(names)
        delta = {n: {k: v - mk_before.get(n, {}).get(k, 0) for k, v in mk_after.get(n, {}).items()
                     if v - mk_before.get(n, {}).get(k, 0) > 0} for n in names}
        rec["markers"] = {n: {k: v for k, v in d.items() if k in INVALID_MARKERS}
                          for n, d in delta.items()}
        rec["warnings"] = {n: {k: v for k, v in d.items() if k in WARN_MARKERS}
                           for n, d in delta.items()}
        if pl in ("P2", "P3"):
            st = store_stats()
            rec["store"] = {**st, "miss_after_evict_delta":
                            st.get("miss_after_evict", 0) - st_before.get("miss_after_evict", 0)}
            rec["save"] = save_stats(args.sm_requests)
        if pl == "P3":
            rec["decode_hits"] = decode_hits(args.sm_requests, args.sm_prompt_len)
            if not hung:
                try:
                    rec["verify_texts"] = verify_texts(args)
                except Exception as e:
                    rec["verify_texts"] = [f"ERROR {type(e).__name__}: {e}"]
        if pl == "P1" and not big and not hung:
            # connector-free reference for the KV-content check: the same 8 prompts,
            # prefilled AND decoded by this one engine (no LMCache anywhere)
            try:
                rec["verify_texts"] = verify_texts(args, "http://127.0.0.1:8111")
            except Exception as e:
                rec["verify_texts"] = [f"ERROR {type(e).__name__}: {e}"]
        n_failed = 0
        if res8 is not None:
            results, duration = res8
            rec["big_summary"] = client_mod.summarize(results, duration, args.max_tokens)
            rec["big_records"] = [{k: r.get(k) for k in ("index", "session_id", "t_submit",
                                                         "ttft", "e2e")} for r in results]
            n_failed += rec["big_summary"]["n_failed"]
            s = rec["big_summary"]
            print(f"  8B: TTFT p50={s['ttft_ms']['p50']} p95={s['ttft_ms']['p95']} ms | "
                  f"TPOT p50={s['tpot_ms']['p50']} ms | {s['output_tok_per_s']} tok/s | "
                  f"failed={s['n_failed']}", flush=True)
        if small is not None:
            rec["small"] = small
            s = small.get("summary") or {}
            n_failed += s.get("n_failed") or 0
            print(f"  3B: TTFT p50={(s.get('ttft_ms') or {}).get('p50')} "
                  f"p95={(s.get('ttft_ms') or {}).get('p95')} ms | "
                  f"TPOT p50={(s.get('tpot_ms') or {}).get('p50')} ms | "
                  f"{s.get('request_tok_per_s')} req/s | {s.get('output_tok_per_s')} tok/s | "
                  f"failed={s.get('n_failed')}", flush=True)
        rec["n_failed"] = n_failed
        rec["ok"] = (not hung and n_failed == 0 and "error" not in (small or {})
                     and not any(rec["markers"].values()))
        g = rec["gpu"]
        print(f"  gpu0 smact={g.get('gpu0_sm_active_mean')} smocc={g.get('gpu0_sm_occupancy_mean')} "
              f"fb={g.get('gpu0_fb_used_max_gib')} | gpu1 smact={g.get('gpu1_sm_active_mean')} "
              f"smocc={g.get('gpu1_sm_occupancy_mean')} dram={g.get('gpu1_dram_active_mean')} "
              f"fb={g.get('gpu1_fb_used_max_gib')} | swap {rec['swap_delta']} | host cpu "
              f"{rec['host_cpu_busy']:.2f} | avail {rec['avail_mb_end']} MB | "
              f"store {rec.get('store')} | save {rec.get('save')} | decode hits "
              f"{rec.get('decode_hits')} | ok={rec['ok']}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"  FAILED: {rec['error']}", flush=True)
    finally:
        mon.terminate()
        hmon.terminate()
        arch = LOGS / "pfp_archive" / tag
        arch.mkdir(parents=True, exist_ok=True)
        for lg in LOGS.glob("pfp_*.log"):
            (arch / lg.name).write_bytes(lg.read_bytes())
        subprocess.run(["bash", str(STOP)], capture_output=True, timeout=180)
        mps_stop()
    (RAW / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec


# --------------------------------------------------------------------- summary
def _mean(md, base):
    c = md.get(f"{base}_count")
    return round(1000 * md.get(f"{base}_sum", 0.0) / c, 1) if c else None


def rebuild_summary():
    """Rebuild the summary CSV from every raw record -- nothing is appended by hand."""
    rows = []
    for path in sorted(RAW.glob("pfp_*.json")):
        if "_smoke" in path.name:
            continue
        rec = json.loads(path.read_text())
        b = rec.get("big_summary") or {}
        s = (rec.get("small") or {}).get("summary") or {}
        md, g, L = rec.get("metrics_delta") or {}, rec.get("gpu") or {}, rec.get("launch") or {}
        decode = rec.get("placement") == "P3"

        def p(d, grp, k):
            return (d.get(grp) or {}).get(k)
        dur = s.get("duration_s")
        rows.append({
            "name": rec["name"], "cell": rec["cell"], "placement": rec.get("placement") or "",
            "big": rec.get("big"), "sm_qps": rec.get("sm_qps"), "ok": rec.get("ok"),
            "error": rec.get("error", ""),
            "big_ttft_p50_ms": p(b, "ttft_ms", "p50"), "big_ttft_p95_ms": p(b, "ttft_ms", "p95"),
            "big_tpot_p50_ms": p(b, "tpot_ms", "p50"), "big_tpot_p95_ms": p(b, "tpot_ms", "p95"),
            "big_itl_p50_ms": p(b, "itl_ms", "p50"), "big_e2e_p50_ms": p(b, "e2e_ms", "p50"),
            "big_tok_per_s": b.get("output_tok_per_s"), "big_n_failed": b.get("n_failed"),
            "big_decode_queue_ms": _mean(md.get("big_decode", {}), "vllm:request_queue_time_seconds"),
            "big_prefill_queue_ms": _mean(md.get("big_prefill", {}), "vllm:request_queue_time_seconds"),
            "sm_ttft_p50_ms": p(s, "ttft_ms", "p50"), "sm_ttft_p95_ms": p(s, "ttft_ms", "p95"),
            # no decode in P1/P2 -> TPOT and output tok/s are not applicable
            "sm_tpot_p50_ms": p(s, "tpot_ms", "p50") if decode else None,
            "sm_tpot_p95_ms": p(s, "tpot_ms", "p95") if decode else None,
            "sm_tok_per_s": s.get("output_tok_per_s") if decode else None,
            "sm_req_per_s": s.get("request_tok_per_s"),
            "sm_prompt_tok_per_s": round(s["prompt_tokens"] / dur) if dur and s.get("prompt_tokens") else None,
            "sm_n_failed": s.get("n_failed"),
            "sm_prefill_queue_ms": _mean(md.get("small_prefill", {}), "vllm:request_queue_time_seconds"),
            "sm_prefill_step_ms": _mean(md.get("small_prefill", {}), "vllm:request_prefill_time_seconds"),
            "sm_decode_queue_ms": _mean(md.get("small_decode", {}), "vllm:request_queue_time_seconds"),
            **{k: g.get(k) for gpu in (0, 1) for k in (
                f"gpu{gpu}_sm_active_mean", f"gpu{gpu}_sm_occupancy_mean", f"gpu{gpu}_tensor_active_mean",
                f"gpu{gpu}_dram_active_mean", f"gpu{gpu}_fb_used_max_gib")},
            "big_prefill_kv_gib": (L.get("big_prefill") or {}).get("kv_gib"),
            "big_decode_kv_gib": (L.get("big_decode") or {}).get("kv_gib"),
            "sm_prefill_kv_gib": (L.get("small_prefill") or {}).get("kv_gib"),
            "sm_decode_kv_gib": (L.get("small_decode") or {}).get("kv_gib"),
            "host_cpu_busy": rec.get("host_cpu_busy"),
            "swap_in_pages": (rec.get("swap_delta") or {}).get("pswpin"),
            "swap_out_pages": (rec.get("swap_delta") or {}).get("pswpout"),
            "avail_mb_after_launch": L.get("avail_mb_after_launch"),
            "avail_mb_end": rec.get("avail_mb_end"),
            "store_evicted": (rec.get("store") or {}).get("evicted"),
            "store_late_gets": (rec.get("store") or {}).get("miss_after_evict_delta"),
            "l1_warnings": sum(sum(d.values()) for d in (rec.get("warnings") or {}).values()),
            "omp_num_threads": rec.get("omp_num_threads"),
            "arm": rec.get("arm"),
            "sm_async_save": rec.get("sm_async_save", False),
            "sm_copy_ms_p50": (rec.get("save") or {}).get("async_copy_ms_p50",
                                                         (rec.get("save") or {}).get("offload_ms_p50")),
            "sm_decode_full_hits": (rec.get("decode_hits") or {}).get("full_hits"),
            "sm_decode_misses": (rec.get("decode_hits") or {}).get("partial_or_miss"),
            "sm_preemptions": (md.get("small_prefill") or {}).get("vllm:num_preemptions_total"),
            "prefill_util": rec.get("prefill_util"), "big_l1_gb": rec.get("big_l1_gb", 6.0),
            "sm_requests": rec.get("sm_requests"),
        })
    if rows:
        with open(OUT / "summary_pfp.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", choices=list(CELLS), required=True)
    ap.add_argument("--arm", choices=list(ARMS), default="fair")
    ap.add_argument("--budget", type=float, default=23.5, help="8B decode budget, GiB")
    ap.add_argument("--prefill-util", type=float, default=0.80,
                    help="8B prefill engine on GPU0 (0.80 in earlier studies; lowered so "
                         "the 3B decode of P3 fits beside it)")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, default=2.0, help="8B QPS")
    ap.add_argument("--warmup-qps", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--sm-model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--sm-util", type=float, default=0.20)
    ap.add_argument("--sm-max-seqs", type=int, default=4)
    ap.add_argument("--sm-first", action="store_true",
                    help="start the 3B before the 8B (fp8 load peak); the 8B's KV grant is recorded")
    ap.add_argument("--sm-quant", default="fp8", help="3B weight quantization ('' = bf16)")
    ap.add_argument("--sm-window", type=float, default=150.0,
                    help="3B requests per cell = window x rate")
    ap.add_argument("--sm-qps", type=float, nargs="+", default=[2.0])
    ap.add_argument("--sm-requests", type=int, default=300)
    ap.add_argument("--sm-prompt-len", type=int, default=2048)
    ap.add_argument("--sm-seed", type=int, default=1000)
    ap.add_argument("--sm-store-cap-gb", type=float, default=2.0)
    ap.add_argument("--sm-async-save", action="store_true",
                    help="3B prefill engine returns token #1 before the KV copy "
                         "(async_save_patch/); P2/P3 cells get an _async name")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--stall-timeout", type=float, default=180.0)
    ap.add_argument("--mem-floor-mb", type=int, default=4000)
    ap.add_argument("--swap-allow-mb", type=int, default=3000)
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="30 requests, _smoke names -- cannot clobber real data")
    args = ap.parse_args()
    if not os.environ.get("OMP_NUM_THREADS"):
        ap.error("export OMP_NUM_THREADS (<= cores / engines; up to 4 engines here -> 8)")
    if args.smoke:
        args.requests = args.sm_requests = 30
        args.name_suffix += "_smoke"
    args.wl8 = build_workload(32, 6144, 128, args.requests, "zipf", 1.1, args.seed)
    qps_list = list(args.sm_qps)
    for cell in args.cells:
        for q in (qps_list if CELLS[cell][1] else qps_list[:1]):
            args.sm_qps = q
            if not args.smoke:          # keep the 3B's window ~ the 8B's (300 req at 2 QPS)
                args.sm_requests = max(30, round(args.sm_window * q))
            rec = run_cell(args, cell)
            rebuild_summary()
            if rec and not rec.get("ok") and "RAM gate" in str(rec.get("error", "")):
                print("  RAM gate refused this cell -- continuing with the next", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
