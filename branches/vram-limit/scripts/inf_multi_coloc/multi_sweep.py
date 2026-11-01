#!/usr/bin/env python3
"""Scaling N homogeneous colocated inference models on one GPU (GPU1).

Seventh study. Self-contained: workload.py/client.py/lmc_server_main.py/orion_gate are
copies from scripts/inf_inf_coloc/. One LAUNCH brings up N models (one at a time, each
with its own DRAM store) under one sharing ARM, then runs every requested CELL against
the same engines -- the engine flags are identical across cells, only the clients'
workload changes:

  cells   dfits  decode-only, working set fits the VRAM KV grant (resident)
          doff   decode-only, working set ~3x over the grant (streams from DRAM)
          ffits  full (prefill+decode on GPU1), resident
          foff   full, oversubscribed
  arms    nomps  plain time-slicing (reference only)
          mps    MPS, fair sharing
          gate   MPS + idle-window gate: model 0 publishes its busy window (hp_patch),
                 models 1..N-1 are CUPTI-gated best-effort (eager, see multi_launch.sh)
          eager  MPS, models 1..N-1 eager with NO gate (separates eager from the gate)

Decode-only = scenario B of the inf_inf_coloc study: each model's prefixes are written to
its store by a TEMPORARY GPU0 prefill that is then killed; during measurement each GPU1
engine only retrieves KV and decodes (100% prefix hits). Full = scenario A: the GPU1
engine prefills its 128-token suffix (prefixes cached / spilled to its own store).

All N clients start at the same epoch instant (aligned windows), one process each.

    scripts/inf_multi_coloc/multi_sweep.py --cohort small --n 2 --arm mps --cells dfits --smoke
    scripts/inf_multi_coloc/multi_sweep.py --cohort small --n 1 2 4 --arm mps
"""

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from workload import build_workload  # noqa: E402

OUT = REPO / "output"
RAW = OUT / "raw"
LOGS = OUT / "logs"
GPUMON = OUT / "gpumon"
CLIENT_OUT = OUT / "inf_multi_coloc"
LAUNCH = HERE / "multi_launch.sh"
STOP = HERE / "multi_stop.sh"
MPS_PIPE = "/tmp/mc_mps_pipe"
MPS_LOG = "/tmp/mc_mps_log"

GIB = 1 << 30
INVALID_MARKERS = ("RECV TIMEOUT", "kv_cache is None", "Insufficient memory",
                   "Peer Out Of Memory", "EngineDeadError")
# LMCache's pinned L1 staging pool running dry. Not fatal by itself, but it is the
# signature of the N=4 decode-only-offload collapse (in-flight KV pinned in L1 exceeds
# the pool, retrievals stall, requests re-queue), so it is counted per model per cell.
WARN_MARKERS = ("Failed to allocate memory block", "Pin timeout detected")

# Per-model sizing. kv_bytes = layers * kv_heads * head_dim * 2 (K,V) * 2 B (bf16).
# ram_mb = host RSS of one model (API server + EngineCore incl. pinned L1 + store
# process overhead) EXCLUDING the store's working set, which is added per cell.
COHORTS = {
    "small": dict(model="Qwen/Qwen2.5-0.5B", util=0.08, max_num_seqs=64, l1_gb=1,
                  kv_bytes=24 * 2 * 64 * 2 * 2, fits=8, offload=48, ram_mb=5000),
    # Qwen2.5-3B: 36 layers x 2 kv heads x 128 dim -> 36 KiB/token, 0.21 GiB per 6144
    # prefix; 5.8 GiB bf16 weights. util 0.24 fits only 3 on GPU1 (8.24 GiB each);
    # util 0.22 keeps N=4 (7.8 GiB each, ~31 GiB for 4). At util 0.22 the KV grant is set
    # by non-KV overhead: max_num_seqs 64 -> 0.59 GiB, 16 + batched 1024 -> 0.94 GiB
    # (probe_kv.sh; same footprint). 16 seqs ~3x the in-flight per model at 2 QPS.
    # fits = 3 sessions (0.63 GiB, 67% of grant); offload = 14 (2.95 GiB, 3.1x over),
    # matching the small cohort's ratios. ram_mb measured: 1.1 API + 3.6 EngineCore +
    # 0.7 store overhead. The earlier 0.59 GiB runs are kept as *_thin.
    "medium": dict(model="Qwen/Qwen2.5-3B", util=0.22, max_num_seqs=16, max_batched=1024,
                   l1_gb=1, kv_bytes=36 * 2 * 128 * 2 * 2, fits=3, offload=14, ram_mb=5500),
}

CELLS = {   # name -> (decode_only, wl)
    "dfits": (True, "fits"), "doff": (True, "offload"),
    "ffits": (False, "fits"), "foff": (False, "offload"),
    # prefill-only: every request a fresh pf_len-token prompt, 1 output token, routed
    # through the external scheduler (mc_scheduler.py). Engines run WITHOUT LMCache
    # (nothing to reuse; a store would just grow with every prompt). Own launch.
    "pfill": (False, "pfill"),
}


def eport(i):
    return 8400 + i


def pport(i):
    return 8500 + i


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


def avail_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def gpu_stats(csv_path, t0, t1):
    per = {}
    try:
        for line in open(csv_path):
            if "\x00" in line:
                continue
            f = line.strip().split(",")
            if len(f) != 6 or f[0] == "ts":
                continue
            try:
                ts = float(f[0])
                if not (t0 <= ts <= t1):
                    continue
                g = per.setdefault(int(f[1]), {"n": 0, "smact": 0.0, "smocc": 0.0,
                                               "drama": 0.0, "fb_max": 0.0})
                g["n"] += 1
                g["smact"] += float(f[2]); g["smocc"] += float(f[3]); g["drama"] += float(f[4])
                g["fb_max"] = max(g["fb_max"], float(f[5]))
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
            out[f"gpu{gpu}_dram_active_mean"] = round(g["drama"] / g["n"], 4)
            out[f"gpu{gpu}_fb_used_max_gib"] = round(g["fb_max"] / 1024, 2)
    return out


def engine_startup(i):
    try:
        text = (LOGS / f"mc_engine_{i}.log").read_text(errors="replace")
    except OSError:
        return {}
    out = {}
    if m := re.findall(r"Available KV cache memory: ([\d.]+) GiB", text):
        out["kv_gib"] = float(m[-1])
    if m := re.findall(r"GPU KV cache size: ([\d,]+) tokens", text):
        out["kv_tokens"] = int(m[-1].replace(",", ""))
    return out


def marker_counts(n):
    out = {}
    for i in range(n):
        try:
            text = (LOGS / f"mc_engine_{i}.log").read_text(errors="replace")
        except OSError:
            continue
        out[i] = {m: text.count(m) for m in INVALID_MARKERS + WARN_MARKERS
                  if text.count(m)}
    return out


def cpu_times():
    """(busy, total) jiffies over all cores from /proc/stat."""
    f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = f[3] + f[4]
    return sum(f) - idle, sum(f)


def proc_rss_mb():
    """Host RSS of the experiment's processes, by role -- sizes the RAM ceiling."""
    roles = {"engine_api": "vllm.entrypoints.cli.main serve", "enginecore": "VLLM::EngineCore",
             "store": "lmc_server_main"}
    out = {}
    for role, pat in roles.items():
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        tot = 0
        for pid in r.stdout.split():
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1])
            except OSError:
                pass
        out[role] = round(tot / 1024)
    return out


# ------------------------------------------------------------------------- MPS
def mps_env():
    return {"CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE, "CUDA_MPS_LOG_DIRECTORY": MPS_LOG}


def mps_start():
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    env = {**os.environ, **mps_env()}
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
    time.sleep(2)
    r = subprocess.run(["pgrep", "-f", "nvidia-cuda-mps-control"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("MPS control daemon did not start")
    os.environ.update(mps_env())          # every launched engine inherits it
    print("  MPS daemon up", flush=True)


def mps_stop():
    env = {**os.environ, **mps_env()}
    subprocess.run(["bash", "-c", "echo quit | nvidia-cuda-mps-control"], env=env,
                   capture_output=True, timeout=60)
    for k in mps_env():
        os.environ.pop(k, None)
    time.sleep(2)


# ------------------------------------------------------------------- workloads
def model_wl(args, coh, i, cell):
    decode_only, wl = CELLS[cell]
    seed = args.seed + 100 * i
    base = dict(prefix=args.prefix_len, max_tokens=args.max_tokens, qps=args.qps,
                url=f"http://127.0.0.1:{eport(i)}", seed=seed)
    if wl == "pfill":   # one 16-token stub prefix + a fresh pf_len suffix: all prefill
        url = (f"http://127.0.0.1:{args.sched_port}/m/{i}" if args.sched_active
               else base["url"])
        return {**base, "sessions": 1, "prefix": 16, "suffix": args.pf_len - 16,
                "skew": "uniform", "max_tokens": 1, "qps": args.pf_qps, "url": url}
    n = coh[wl]
    if decode_only:   # whole-prefix hits, no unique suffix: the engine only decodes
        return {**base, "sessions": n, "suffix": 0, "skew": "uniform"}
    return {**base, "sessions": n, "suffix": args.suffix_len, "skew": "zipf"}


def runner_cmd(args, coh, i, w, phase, out, start_at=0.0):
    return [str(REPO / ".venv" / "bin" / "python"), str(HERE / "model_client_runner.py"),
            "--base-url", w["url"], "--model", coh["model"],
            "--sessions", str(w["sessions"]), "--prefix-len", str(w["prefix"]),
            "--suffix-len", str(w["suffix"]), "--skew", w["skew"],
            "--kv-bytes", str(coh["kv_bytes"]),
            "--requests", str(args.requests), "--qps", str(w["qps"]),
            "--seed", str(w["seed"]), "--max-tokens", str(w["max_tokens"]),
            "--timeout", str(args.timeout), "--start-at", f"{start_at:.3f}",
            "--phase", phase, "--out", str(out)]


# --------------------------------------------------------------------- populate
async def _populate(port, model, prefixes):
    import aiohttp
    sem = asyncio.Semaphore(2)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as sess:
        async def one(ids):
            async with sem:
                body = {"model": model, "prompt": ids, "max_tokens": 1,
                        "temperature": 0.0, "stream": False, "ignore_eos": True}
                async with sess.post(f"http://127.0.0.1:{port}/v1/completions",
                                     json=body) as r:
                    return r.status
        return await asyncio.gather(*(one(p) for p in prefixes))


def populate(args, coh, i, n_sessions):
    """Write model i's prefixes into its store from the temp GPU0 prefill, probe the
    GPU1 engine's retrieval, then kill the temp prefill."""
    t0 = time.time()
    wl = build_workload(n_sessions, args.prefix_len, 0, 1, "uniform", 1.1,
                        args.seed + 100 * i, kv_bytes_per_token=coh["kv_bytes"])
    statuses = asyncio.run(_populate(pport(i), coh["model"], wl.sessions))
    tokens = scrape(pport(i)).get("vllm:prompt_tokens_total", 0.0)
    need = n_sessions * args.prefix_len
    if not all(s == 200 for s in statuses) or tokens < need:
        raise RuntimeError(f"populate model {i}: statuses={set(statuses)} tokens {tokens} < {need}")
    # Retrieval probe: a stored prefix + 16 fresh tokens must come back fast. Slow =
    # keys don't match (PYTHONHASHSEED drift) and decode-only would silently recompute.
    probe = {"model": coh["model"], "prompt": wl.sessions[0] + list(range(500, 516)),
             "max_tokens": 2, "temperature": 0.0, "stream": False, "ignore_eos": True}
    tp = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{eport(i)}/v1/completions",
                                 json.dumps(probe).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    probe_s = time.time() - tp
    k = subprocess.run(["bash", str(HERE / "multi_prefill_kill.sh"), str(i)],
                       capture_output=True, text=True, timeout=60)
    print(f"  populate model {i}: {n_sessions} prefixes in {time.time() - t0:.1f}s, "
          f"probe {probe_s * 1000:.0f} ms; {k.stdout.strip()}", flush=True)
    return {"prompt_tokens": tokens, "probe_s": round(probe_s, 3),
            "populate_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------- launch
def launch_models(args, coh, n, arm, need_temp_pf, populate_sessions, ws_gib):
    subprocess.run(["bash", str(STOP)], capture_output=True, timeout=180)
    if arm in ("mps", "gate", "eager"):
        mps_start()
    env = {**os.environ, "MEM_GUARD_FLOOR_MB": str(args.mem_floor_mb)}
    r = subprocess.run(["bash", str(LAUNCH), "guard"], env=env, capture_output=True,
                       text=True, timeout=60)
    if "guard up" not in r.stdout:
        raise RuntimeError(f"guard failed: {r.stdout}{r.stderr}")
    info = {"models": {}, "populate": {}}
    per_model_mb = coh["ram_mb"] + int(ws_gib * 1024) + (2500 if need_temp_pf else 0)
    for i in range(n):
        # Incremental RAM gate: bring up model i only if it leaves the guard floor
        # plus a margin intact. Pinned L1s are part of ram_mb.
        av = avail_mb()
        if av - per_model_mb < args.mem_floor_mb + 2000:
            raise RuntimeError(f"RAM gate: model {i} needs ~{per_model_mb} MB, "
                               f"{av} MB available, floor {args.mem_floor_mb}")
        role = "plain"
        if arm == "gate":
            role = "hp" if i == 0 else "be"
        menv = {**env, "MC_MODEL": coh["model"], "MC_UTIL": str(coh["util"]),
                "MC_MAX_NUM_SEQS": str(coh["max_num_seqs"]), "MC_L1_GB": str(coh["l1_gb"]),
                "MC_ROLE": role}
        if args.no_lmc:
            menv["MC_NO_LMC"] = "1"
        if coh.get("max_batched"):
            menv["MC_MAX_BATCHED"] = str(coh["max_batched"])
        if arm == "eager" and i > 0:
            menv["MC_EAGER"] = "1"
        t0 = time.time()
        r = subprocess.run(["bash", str(LAUNCH), "model", str(i), "1" if need_temp_pf else "0"],
                           env=menv, capture_output=True, text=True, timeout=900)
        if r.returncode != 0 or f"model {i} up" not in r.stdout:
            raise RuntimeError(f"model {i} launch failed: {r.stdout[-600:]}{r.stderr[-600:]}")
        st = engine_startup(i)
        info["models"][i] = {**st, "role": role, "startup_s": round(time.time() - t0, 1)}
        print(f"  model {i} up ({role}) KV {st.get('kv_gib')} GiB in "
              f"{info['models'][i]['startup_s']}s; avail RAM {avail_mb()} MB", flush=True)
        if need_temp_pf:
            info["populate"][i] = populate(args, coh, i, populate_sessions)
    grants = [m.get("kv_gib") for m in info["models"].values()]
    info["kv_grants_gib"] = grants
    info["homogeneous"] = (None not in grants
                           and max(grants) - min(grants) <= args.kv_tolerance)
    if not info["homogeneous"]:
        print(f"  *** KV grants differ across models: {grants} ***", flush=True)
    info["rss_mb"] = proc_rss_mb()
    info["avail_mb_after_launch"] = avail_mb()
    return info


# ----------------------------------------------------------------------- cells
def run_clients(args, coh, n, cell, phase, tag):
    """Start N runners (aligned start instant), watch progress, return outputs."""
    CLIENT_OUT.mkdir(parents=True, exist_ok=True)
    start_at = time.time() + 3.0
    procs, outs = [], []
    for i in range(n):
        w = model_wl(args, coh, i, cell)
        out = CLIENT_OUT / f"{tag}_m{i}_{phase}.json"
        out.unlink(missing_ok=True)
        outs.append(out)
        procs.append(subprocess.Popen(runner_cmd(args, coh, i, w, phase, out, start_at),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True))
    # Progress/stall watchdog on the engines' generated-token counters: a KV-path
    # failure is a silent hang, so "no new tokens anywhere for stall_timeout" = dead.
    last_tok, last_change, hung = None, time.time(), False
    t_print = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(10)
        tok = sum(scrape(eport(i)).get("vllm:generation_tokens_total", 0.0) for i in range(n))
        if tok != last_tok:
            last_tok, last_change = tok, time.time()
        if time.time() - t_print >= 60:
            done = sum(p.poll() is not None for p in procs)
            print(f"    [{phase}] {datetime.now():%H:%M:%S} gen_tokens={tok:.0f} "
                  f"runners done {done}/{n} avail RAM {avail_mb()} MB", flush=True)
            t_print = time.time()
        if time.time() - last_change > args.stall_timeout:
            print(f"  STALLED: no generated tokens for {args.stall_timeout}s -- killing",
                  flush=True)
            hung = True
            for p in procs:
                p.kill()
            break
    results = []
    for i, (p, out) in enumerate(zip(procs, outs)):
        stdout, _ = p.communicate(timeout=60)
        try:
            results.append(json.loads(out.read_text()))
        except (OSError, json.JSONDecodeError):
            results.append({"error": f"no output; {stdout[-300:] if stdout else ''}"})
    return results, hung


def run_sched_cell(args, coh, n, sc, launch_info, mon_path):
    """Prefill-only under one external-scheduler config ('none' = clients hit the
    engines directly; '<policy>:<k>' = through mc_scheduler.py with global cap k)."""
    proc = None
    args.sched_active = sc != "none"
    if args.sched_active:
        policy, k = sc.split(":")
        proc = subprocess.Popen(
            [str(REPO / ".venv" / "bin" / "python"), str(HERE / "mc_scheduler.py"),
             "--n", str(n), "--k", k, "--policy", policy, "--port", str(args.sched_port)],
            stdout=open(LOGS / "mc_scheduler.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.sched_port}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
    tag = "none" if sc == "none" else sc.replace(":", "k")
    try:
        rec = run_cell(args, coh, n, args.arm, "pfill", launch_info, mon_path,
                       name_extra=f"_{tag}")
        if proc is not None:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.sched_port}/stats", timeout=5) as r:
                    rec["sched_stats"] = json.loads(r.read())
            except Exception as e:
                rec["sched_stats"] = {"error": str(e)}
        rec["sched"] = sc
        (RAW / f"{rec['name']}.json").write_text(json.dumps(rec, indent=1))
        return rec
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)
        args.sched_active = False


def run_cell(args, coh, n, arm, cell, launch_info, mon_path, name_extra=""):
    name = f"mc_{args.cohort}_{cell}_n{n}_{arm}{name_extra}{args.name_suffix}"
    print(f"\n=== {name} ===", flush=True)
    rec = {"name": name, "cohort": args.cohort, "model": coh["model"], "n": n, "arm": arm,
           "cell": cell, "decode_only": CELLS[cell][0], "wl": CELLS[cell][1],
           "qps_per_model": args.qps, "requests": args.requests,
           "sessions": coh[CELLS[cell][1]],
           "ws_gib_per_model": round(coh[CELLS[cell][1]] * args.prefix_len
                                     * coh["kv_bytes"] / GIB, 3),
           "setup": "inf_multi_coloc", "launch": launch_info,
           "telemetry_tag": mon_path.stem,
           # per-engine CPU thread cap (unset = vLLM/torch default = all cores per engine;
           # stock N=4 decode-only offload collapses from thread oversubscription)
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    try:
        wu, wu_hung = run_clients(args, coh, n, cell, "warmup", name)
        rec["warmup"] = [r.get("summary", {}).get("n_ok") for r in wu]
        if wu_hung or any("error" in r for r in wu):
            raise RuntimeError(f"warmup failed: {[r.get('error') for r in wu]}")
        print(f"  warmup ok: {rec['warmup']}", flush=True)
        before = {i: scrape(eport(i)) for i in range(n)}
        mk_before = marker_counts(n)
        sw_before = swap_counters()
        cpu0 = cpu_times()
        t0 = time.time()
        res, hung = run_clients(args, coh, n, cell, "measure", name)
        t1 = time.time()
        after = {i: scrape(eport(i)) for i in range(n)}
        rec.update(t_measure0=round(t0, 3), t_measure1=round(t1, 3), hung=hung)
        rec["gpu"] = gpu_stats(mon_path, t0, t1)
        rec["swap_delta"] = {k: swap_counters()[k] - v for k, v in sw_before.items()}
        cpu1 = cpu_times()
        rec["host_cpu_busy"] = round((cpu1[0] - cpu0[0]) / max(1, cpu1[1] - cpu0[1]), 4)
        rec["host_cores"] = os.cpu_count()
        rec["avail_mb_end"] = avail_mb()
        mk_after = marker_counts(n)
        rec["models"] = []
        for i in range(n):
            md = {k: round(after[i].get(k, 0.0) - before[i].get(k, 0.0), 4)
                  for k in set(after[i]) | set(before[i]) if k != "error"}
            markers = {k: v - mk_before.get(i, {}).get(k, 0)
                       for k, v in mk_after.get(i, {}).items()
                       if v - mk_before.get(i, {}).get(k, 0) > 0}
            rec["models"].append({
                "i": i, "client": res[i], "metrics_delta": md,
                "markers": {k: v for k, v in markers.items() if k in INVALID_MARKERS},
                "warnings": {k: v for k, v in markers.items() if k in WARN_MARKERS}})
        rec["n_failed"] = sum((r.get("summary") or {}).get("n_failed") or 0 for r in res)
        rec["ok"] = (not hung and all("error" not in r for r in res)
                     and rec["n_failed"] == 0
                     and not any(m["markers"] for m in rec["models"]))
        for m in rec["models"]:
            s = m["client"].get("summary", {})
            print(f"  m{m['i']}: TTFT p50={s.get('ttft_ms', {}).get('p50')} "
                  f"TPOT p50={s.get('tpot_ms', {}).get('p50')} "
                  f"{s.get('output_tok_per_s')} tok/s failed={s.get('n_failed')}", flush=True)
        g = rec["gpu"]
        print(f"  gpu1 smact={g.get('gpu1_sm_active_mean')} smocc={g.get('gpu1_sm_occupancy_mean')} "
              f"dram={g.get('gpu1_dram_active_mean')} fb={g.get('gpu1_fb_used_max_gib')} GiB; "
              f"swap in/out {rec['swap_delta']}; host cpu {rec['host_cpu_busy']:.2f}; "
              f"L1 warnings {[sum(m['warnings'].values()) for m in rec['models']]}; "
              f"ok={rec['ok']}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"  FAILED: {rec['error']}", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec


# --------------------------------------------------------------------- summary
def _p(s, grp, k):
    return (s.get(grp) or {}).get(k)


def rebuild_summaries(prefix):
    """Rebuild BOTH summary CSVs from every raw JSON -- separate invocations (other N,
    other arms) append naturally and nothing is ever overwritten by a later run."""
    cell_rows, model_rows = [], []
    for path in sorted(RAW.glob(f"{prefix}*.json")):
        if "_smoke" in path.name:      # plumbing checks never enter the summaries
            continue
        rec = json.loads(path.read_text())
        if "models" not in rec:
            continue
        sums = [m["client"].get("summary", {}) for m in rec["models"]]
        ok = [s for s in sums if s]
        for m, s in zip(rec["models"], sums):
            md = m["metrics_delta"]
            q = md.get("vllm:prefix_cache_queries_total")
            model_rows.append({
                "name": rec["name"], "cohort": rec["cohort"], "cell": rec["cell"],
                "n": rec["n"], "arm": rec["arm"], "i": m["i"],
                "role": rec["launch"]["models"][str(m["i"])]["role"]
                if str(m["i"]) in rec["launch"]["models"] else "",
                "kv_gib": (rec["launch"]["models"].get(str(m["i"])) or {}).get("kv_gib"),
                "ttft_p50_ms": _p(s, "ttft_ms", "p50"), "ttft_p95_ms": _p(s, "ttft_ms", "p95"),
                "tpot_p50_ms": _p(s, "tpot_ms", "p50"), "tpot_p95_ms": _p(s, "tpot_ms", "p95"),
                "itl_p50_ms": _p(s, "itl_ms", "p50"),
                "tok_per_s": s.get("output_tok_per_s"), "n_ok": s.get("n_ok"),
                "n_failed": s.get("n_failed"),
                "prefix_hit_rate": round(md.get("vllm:prefix_cache_hits_total", 0) / q, 4)
                if q else None,
                "preemptions": md.get("vllm:num_preemptions_total", 0.0),
                "l1_warnings": sum((m.get("warnings") or {}).values()),
            })

        def col(grp, k):
            return [_p(s, grp, k) for s in ok if _p(s, grp, k) is not None]

        def mean(v):
            return round(sum(v) / len(v), 2) if v else None
        tps = [s.get("output_tok_per_s") or 0 for s in ok]
        g = rec.get("gpu", {})
        cell_rows.append({
            "name": rec["name"], "cohort": rec["cohort"], "cell": rec["cell"],
            "n": rec["n"], "arm": rec["arm"], "sched": rec.get("sched", ""),
            "ok": bool(rec.get("ok")) and not sum(s.get("n_failed") or 0 for s in ok),
            "homogeneous_kv": rec["launch"].get("homogeneous"),
            "kv_gib_min": min([x for x in rec["launch"].get("kv_grants_gib", []) if x] or [0]),
            "ws_gib_per_model": rec.get("ws_gib_per_model"),
            "ttft_p50_mean_ms": mean(col("ttft_ms", "p50")),
            "ttft_p50_max_ms": max(col("ttft_ms", "p50") or [0]) or None,
            "ttft_p95_max_ms": max(col("ttft_ms", "p95") or [0]) or None,
            "tpot_p50_mean_ms": mean(col("tpot_ms", "p50")),
            "tpot_p50_max_ms": max(col("tpot_ms", "p50") or [0]) or None,
            "m0_ttft_p50_ms": _p(sums[0], "ttft_ms", "p50") if sums else None,
            "m0_tpot_p50_ms": _p(sums[0], "tpot_ms", "p50") if sums else None,
            "m0_tok_per_s": sums[0].get("output_tok_per_s") if sums else None,
            "tok_per_s_per_model_min": min(tps) if tps else None,
            "tok_per_s_agg": round(sum(tps), 1),
            "n_failed_total": sum(s.get("n_failed") or 0 for s in ok),
            **{k: g.get(k) for k in ("gpu1_sm_active_mean", "gpu1_sm_occupancy_mean",
                                     "gpu1_dram_active_mean", "gpu1_fb_used_max_gib")},
            "swap_in_pages": rec.get("swap_delta", {}).get("pswpin"),
            "host_cpu_busy": rec.get("host_cpu_busy"),
            "l1_warnings": sum(sum((m.get("warnings") or {}).values())
                               for m in rec["models"]),
            "avail_mb_after_launch": rec["launch"].get("avail_mb_after_launch"),
            "avail_mb_end": rec.get("avail_mb_end"),
        })
    for rows, suffix in ((cell_rows, ""), (model_rows, "_models")):
        if rows:
            with open(OUT / f"summary_{prefix.rstrip('_')}{suffix}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=list(COHORTS), default="small")
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--arm", choices=["nomps", "mps", "gate", "eager"], default="mps")
    ap.add_argument("--cells", nargs="+", choices=list(CELLS),
                    default=["dfits", "doff", "ffits", "foff"])
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--suffix-len", type=int, default=128)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--qps", type=float, nargs="+", default=[2.0],
                    help="per-model QPS; several values = a capacity sweep in one launch "
                         "(cells at QPS != 2 get a _q<qps> name suffix)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--stall-timeout", type=float, default=180.0)
    ap.add_argument("--mem-floor-mb", type=int, default=6000)
    ap.add_argument("--kv-tolerance", type=float, default=0.05)
    ap.add_argument("--l1-gb", type=int, default=None,
                    help="override the cohort's LMCache L1 (pinned) size per engine")
    ap.add_argument("--pf-len", type=int, default=6144, help="prefill-only prompt length")
    ap.add_argument("--pf-qps", type=float, default=4.0, help="prefill-only QPS per model")
    ap.add_argument("--sched", nargs="+", default=["none"],
                    help="prefill-only scheduler configs: none | <policy>:<k>, "
                         "e.g. fcfs:0 fcfs:1 rr:1 prio:1")
    ap.add_argument("--sched-port", type=int, default=8390)
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="40 requests, _smoke names -- cannot clobber real data")
    args = ap.parse_args()
    if args.smoke:
        args.requests = 40
        args.name_suffix += "_smoke"
    qps_list = list(args.qps)
    args.qps = qps_list[0]
    coh = dict(COHORTS[args.cohort])
    if args.l1_gb:
        coh["l1_gb"] = args.l1_gb
    args.no_lmc = args.cells == ["pfill"]
    if "pfill" in args.cells and not args.no_lmc:
        ap.error("pfill needs its own launch (engines without LMCache): --cells pfill")
    args.sched_active = False
    need_temp_pf = any(CELLS[c][0] for c in args.cells)
    pop_sessions = max((coh[CELLS[c][1]] for c in args.cells if c != "pfill"), default=0)
    ws_gib = pop_sessions * args.prefix_len * coh["kv_bytes"] / GIB

    for n in args.n:
        tag = f"mc_{args.cohort}_n{n}_{args.arm}{args.name_suffix}"
        print(f"\n##### launch {tag}: {n} x {coh['model']} on GPU1, cells {args.cells}",
              flush=True)
        GPUMON.mkdir(parents=True, exist_ok=True)
        # Telemetry files are per LAUNCH. Never overwrite an earlier launch's trace with
        # the same N/arm/suffix (it happened: the capped N=4 offload trace was clobbered by
        # later launches) -- give a repeat launch its own tag (_L2, _L3, ...).
        base_tag, k = tag, 1
        while (GPUMON / f"{tag}.csv").exists():
            k += 1
            tag = f"{base_tag}_L{k}"
        mon_path = GPUMON / f"{tag}.csv"
        mon = subprocess.Popen([sys.executable, str(REPO / "scripts/common/gpu_monitor.py"),
                                str(mon_path)])
        # Host side, per second, by process role (stores / EngineCores / API / clients).
        hmon = subprocess.Popen([sys.executable, str(HERE / "host_monitor.py"),
                                 str(GPUMON / f"{tag}_host.csv")])
        # per-thread EngineCore CPU (spin-wait evidence; no ptrace/perf on this box)
        tmon = subprocess.Popen([sys.executable, str(HERE / "thread_sampler.py"),
                                 str(GPUMON / f"{tag}_threads.csv"), "2"])
        try:
            info = launch_models(args, coh, n, args.arm, need_temp_pf, pop_sessions, ws_gib)
            print(f"  all {n} up: KV grants {info['kv_grants_gib']}, RSS {info['rss_mb']}, "
                  f"avail {info['avail_mb_after_launch']} MB", flush=True)
            if args.no_lmc:
                for sc in args.sched:
                    rec = run_sched_cell(args, coh, n, sc, info, mon_path)
                    if rec.get("hung"):
                        break
            else:
                abort = False
                for cell in args.cells:
                    for q in qps_list:
                        args.qps = q
                        rec = run_cell(args, coh, n, args.arm, cell, info, mon_path,
                                       name_extra="" if q == 2.0 else f"_q{q:g}")
                        if rec.get("hung") or "heartbeat" in str(rec.get("error", "")):
                            print("  aborting remaining cells of this launch", flush=True)
                            abort = True
                            break
                        # past saturation: >50% failed -> higher QPS can only be worse
                        if (rec.get("n_failed") or 0) > 0.5 * args.requests * n:
                            print(f"  >50% failed at {q} QPS -- skipping higher QPS",
                                  flush=True)
                            break
                    if abort:
                        break
        except Exception as e:
            print(f"  LAUNCH FAILED: {type(e).__name__}: {e}", flush=True)
            RAW.mkdir(parents=True, exist_ok=True)
            (RAW / f"{tag}_launchfail.json").write_text(json.dumps(
                {"name": tag, "error": f"{type(e).__name__}: {e}"}))
        finally:
            mon.terminate()
            hmon.terminate()
            tmon.terminate()
            # Engine logs are reused by index across launches; archive this launch's
            # copies BEFORE teardown so per-cell warning counts stay attributable.
            arch = LOGS / "mc_archive" / tag
            arch.mkdir(parents=True, exist_ok=True)
            for lg in LOGS.glob("mc_engine_*.log"):
                (arch / lg.name).write_bytes(lg.read_bytes())
            subprocess.run(["bash", str(STOP)], capture_output=True, timeout=180)
            if args.arm in ("mps", "gate", "eager"):
                mps_stop()
        rebuild_summaries(f"mc_{args.cohort}_")
    print("done", flush=True)


if __name__ == "__main__":
    main()
