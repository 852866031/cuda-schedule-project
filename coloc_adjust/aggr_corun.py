#!/usr/bin/env python
"""E1.2 (TODO.md): the synthetic Llama-3-8B decode step of knee_corun.py, co-run with the synthetic bandwidth
aggressor bw_aggr.py instead of the fine-tune tenant. The aggressor's GB/s and SM count are set independently
and its bytes are known (calls x chunk size), so decode's slowdown can be put against GB/s at a fixed SM count
and against SM count at a fixed GB/s. The aggressor always runs on the first M SMs; decode never shares an SM.

Series (one CSV row per timed group of decode steps, or per aggressor-alone window):
  alone         decode alone on the full device and on each decode context, before the aggressor process exists
  aggr_alone    aggressor alone on M SMs: unpaced (target -1), then each target: achieved GB/s next to the counter
  corun         aggressor on the first M SMs at each target (0 = process idle: the baseline of that M, then the
                targets, then -1 = unpaced), decode on the SMs left after the M-SM split (170 - M)
  corun_fixed   the same with decode held on the SMs left after a --fixed-split split (default 64 -> 106 SMs)
                for every M below it, so decode's own SM count does not change with M
  alone_end     decode alone after the aggressor exited (drift check)
With --modes (E2.7) the aggr_alone / corun / corun_fixed series are replaced by
  mode_alone    each aggressor mode alone on M SMs: unpaced, then each target (rate and counter)
  mode_corun    decode on the SMs left after the M-SM split, next to: the idle aggressor (kernel "idle"), every
                mode at its targets, the idle aggressor again ("idle_end")
Modes: gemv32 (read), write32, copy32 take targets in GB/s; tiny (a launch with almost no bytes) and compute
(matrix product that stays in L2) take targets in thousand calls/s; -1 = unpaced, "50%" = half the unpaced rate alone.
Columns: aggr_gbps = achieved (bytes / time inside the decode window), target_gbps = requested; bw_util etc. are
the device-wide NVML GPM counters over the window. Start through run_knee_corun.sh:  SCRIPT=aggr_corun.py ./run_knee_corun.sh --out ...
"""
import argparse, csv, json, os, signal, subprocess, sys, time
import torch, torch.nn.functional as F
from green_ctx import GreenCtx
from bw_aggr import Gpm

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/aggr_corun.csv")
ap.add_argument("--sms", default="96,64,32,16,8", help="aggressor SM counts")
ap.add_argument("--gbps", default="50,100,200,400,800", help="aggressor targets in GB/s; targets above the unpaced rate alone are skipped")
ap.add_argument("--fixed-split", default="64", help="decode context for the corun_fixed series (SMs left after this split); empty = skip")
ap.add_argument("--batches", default="1,8"); ap.add_argument("--ctx", type=int, default=6272)
ap.add_argument("--window", type=float, default=4.0, help="seconds per measured point")
ap.add_argument("--kernel", default="gemv32"); ap.add_argument("--chunk-mb", type=int, default=64); ap.add_argument("--chunks", type=int, default=32)
ap.add_argument("--gpm-period", type=float, default=0.25)
ap.add_argument("--modes", default="", help='E2.7 plan, e.g. "gemv32:100,200,-1;write32:100,200,-1;copy32:100,200,-1;tiny:1,5,20,-1;compute:50%%,-1"')
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
tag = os.path.basename(args.out).replace(".csv", "")
CTL, AG_CSV = f"/dev/shm/{tag}_ctl", args.out.replace(".csv", "_aggr.csv")
SMS = [int(x) for x in args.sms.split(",")]; GBPS = [float(x) for x in args.gbps.split(",")]
FIX = int(args.fixed_split) if args.fixed_split else 0
BS = [int(b) for b in args.batches.split(",")]; BMAX = max(BS)
MODES = [(k, v.split(",")) for k, v in (x.split(":") for x in args.modes.split(";") if x)]
BYTE_MODES = ("gemv32", "write32", "copy32", "sum32", "sum64", "gemv16")

# ---- decode side: same step as knee_probe.py / knee_corun.py
H, I, L, NKV, GRP, HD, V = 4096, 14336, 32, 8, 4, 128, 128256
dev, dt = "cuda", torch.bfloat16
torch.cuda.init()
rem = {k: GreenCtx(k, remaining=True) for k in sorted(set(SMS) | ({FIX} if FIX else set()))}
for k, g in rem.items(): print(f"decode context after a {k}-SM split: {g.sm_count} of {g.device_sms} SMs", flush=True)
def w(*shape): return torch.empty(*shape, device=dev, dtype=dt).normal_(0, 0.02)
W = [dict(qkv=w(H + 2 * NKV * HD, H), o=w(H, H), gate_up=w(2 * I, H), down=w(H, I)) for _ in range(L)]
LM = w(V, H)
KT = [w(BMAX, NKV, HD, args.ctx) for _ in range(L)]; VV = [w(BMAX, NKV, args.ctx, HD) for _ in range(L)]
W_GB = (sum(sum(t.numel() for t in d.values()) for d in W) + LM.numel()) * 2 / 1e9; KV_GB_PER_REQ = 2 * L * NKV * HD * args.ctx * 2 / 1e9
@torch.inference_mode()
def step(x, B):
    for l in range(L):
        d = W[l]
        q = F.linear(F.rms_norm(x, (H,)), d["qkv"])[:, :H].reshape(B, 1, NKV, GRP, HD).permute(0, 2, 1, 3, 4).reshape(B, NKV, GRP, HD)
        a = torch.softmax((q @ KT[l][:B]) * (HD ** -0.5), dim=-1) @ VV[l][:B]
        x = x + F.linear(a.reshape(B, NKV, 1, GRP, HD).permute(0, 2, 1, 3, 4).reshape(B, H), d["o"])
        gu = F.linear(F.rms_norm(x, (H,)), d["gate_up"])
        x = x + F.linear(F.silu(gu[:, :I]) * gu[:, I:], d["down"])
    return F.linear(F.rms_norm(x, (H,)), LM).argmax(-1)
def measure(g, B, seconds):
    """timed groups of queued decode steps for `seconds`; returns [(t_start, t_end, step_ms)]"""
    stream = g.torch_stream() if g else torch.cuda.default_stream(); out = []
    with torch.cuda.stream(stream):
        x = w(B, H)
        t = time.perf_counter(); step(x, B); stream.synchronize(); step(x, B); stream.synchronize()
        steps = max(1, min(20, int(0.5 / ((time.perf_counter() - t) / 2))))
        t_begin = time.time()
        while time.time() - t_begin < seconds:
            ts = time.time(); t = time.perf_counter()
            for _ in range(steps): step(x, B)
            stream.synchronize(); torch.cuda.synchronize()
            out.append((ts, time.time(), (time.perf_counter() - t) / steps * 1e3))
    return out

# ---- aggressor side
seq = {"n": 0}; ag = None
def ag_rows():
    """aggressor rows: (t0, t1, seq, sms, target, bytes, calls)"""
    out = []
    try:
        with open(AG_CSV) as f:
            next(f)
            for line in f:
                p = line.split(",")
                if line.endswith("\n") and len(p) == 7: out.append((float(p[0]), float(p[1]), int(p[2]), int(p[3]), float(p[4]), int(p[5]), int(p[6])))
    except Exception: pass
    return out
def set_aggr(m, gbps, kernel=None, timeout=30):
    """new SM count, target and mode; returns once the aggressor has logged a row under the new setting"""
    seq["n"] += 1
    with open(CTL + ".tmp", "w") as f: json.dump(dict(seq=seq["n"], sms=m, gbps=gbps, kernel=kernel or args.kernel), f)
    os.replace(CTL + ".tmp", CTL); t0 = time.time()
    while time.time() - t0 < timeout:
        r = ag_rows()
        if r and r[-1][2] == seq["n"]: return
        if ag.poll() is not None: raise RuntimeError("aggressor exited")
        time.sleep(0.05)
    raise RuntimeError("aggressor did not take the new setting")
def aggr_rate(t0, t1):
    """achieved GB/s and calls/s over the aggressor rows inside [t0, t1] under the current setting"""
    time.sleep(max(0.0, t1 + 0.3 - time.time()))
    r = [x for x in ag_rows() if x[2] == seq["n"] and x[0] >= t0 and x[1] <= t1]
    if not r: return float("nan"), float("nan")
    d = sum(x[1] - x[0] for x in r); return sum(x[5] for x in r) / d / 1e9, sum(x[6] for x in r) / d

gpm = Gpm(args.gpm_period, args.out.replace(".csv", "_gpm.csv"))
out_f = open(args.out, "w", newline=""); out_w = None
def emit(new):
    global out_w
    if out_w is None: out_w = csv.DictWriter(out_f, fieldnames=list(new[0])); out_w.writeheader()
    out_w.writerows(new); out_f.flush()
def tg(t): return "" if t is None else ("max" if t < 0 else f"{t:g}")
def rec(series, m, target, g, B, groups, kernel=""):
    rate, calls = aggr_rate(groups[0][0], groups[-1][1]) if m is not None else (float("nan"), float("nan"))
    c = gpm.mean(groups[0][0], groups[-1][1]); gb = W_GB + B * KV_GB_PER_REQ
    emit([dict(series=series, kernel=kernel, aggr_sms="" if m is None else m, target_gbps=tg(target), decode_sms=g.sm_count if g else 170, B=B, grp=i,
               step_ms=f"{ms:.3f}", aggr_gbps=f"{rate:.1f}", aggr_calls_s=f"{calls:.0f}", gb_per_step=f"{gb:.3f}", **c) for i, (_, _, ms) in enumerate(groups)])
    med = sorted(x[2] for x in groups)[len(groups) // 2]
    print(f"{series:11s} {kernel:8s} aggr SMs {str(m):>4s} target {tg(target):>5s}  decode SMs {g.sm_count if g else 170:3d} B={B}  step {med:7.2f} ms  decode {gb / med * 1e3:5.0f} GB/s  "
          f"aggr {rate:6.0f} GB/s {calls:6.0f} calls/s  bw {c['bw_util']:>6s} %  sm_util {c['sm_util']:>6s}  {c['power_w']:>4s} W {c['clock_mhz']:>5s} MHz thr {c['throttled']}", flush=True)
def rec_aggr(m, target, t0, t1, series="aggr_alone", kernel=""):
    rate, calls = aggr_rate(t0, t1); c = gpm.mean(t0, t1)
    emit([dict(series=series, kernel=kernel, aggr_sms=m, target_gbps=tg(target), decode_sms="", B=0, grp=0, step_ms="", aggr_gbps=f"{rate:.1f}",
               aggr_calls_s=f"{calls:.0f}", gb_per_step="", **c)])
    print(f"{series:11s} {kernel:8s} aggr SMs {m:4d} target {tg(target):>5s}  aggr {rate:6.0f} GB/s  {calls:6.0f} calls/s  bw {c['bw_util']:>6s} %  sm_util {c['sm_util']:>6s}  {c['power_w']:>4s} W {c['clock_mhz']:>5s} MHz thr {c['throttled']}", flush=True)
    return rate, calls
def decode_alone(tag_, seconds):
    for g in [None] + list(rem.values()):
        for B in BS: rec(tag_, None, None, g, B, measure(g, B, seconds))

decode_alone("alone", 3.0)
for f in (CTL, AG_CSV):
    if os.path.exists(f): os.remove(f)
ag = subprocess.Popen([sys.executable, os.path.join(HERE, "bw_aggr.py"), "--ctl", CTL, "--out", AG_CSV, "--sms-set", ",".join(map(str, SMS)),
                       "--kernel", args.kernel, "--chunk-mb", str(args.chunk_mb), "--chunks", str(args.chunks)],
                      stdout=open(args.out.replace(".csv", "_aggr.log"), "w"), stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while not ag_rows():
        if ag.poll() is not None or time.time() - t0 > 120: raise RuntimeError("aggressor did not start")
        time.sleep(0.2)
    top = {}; plan = []
    if MODES:
        todo = {}
        for m in SMS:                               # every mode alone: unpaced rate (GB/s or thousand calls/s), then each target
            for k, spec in MODES:
                set_aggr(m, -1.0, k); a = time.time(); time.sleep(2.5); r, c = rec_aggr(m, -1.0, a, time.time(), "mode_alone", k)
                cap = r if k in BYTE_MODES else c / 1e3
                ts = sorted({round(float(x[:-1]) / 100 * cap, 2) if x.endswith("%") else float(x) for x in spec if x != "-1"})
                ts = [t for t in ts if t <= 0.97 * cap]
                for t in ts: set_aggr(m, t, k); a = time.time(); time.sleep(2.5); rec_aggr(m, t, a, time.time(), "mode_alone", k)
                todo[(m, k)] = ts + ([-1.0] if "-1" in spec else [])
        for m in SMS:
            set_aggr(m, 0.0)
            for B in BS: rec("mode_corun", m, 0.0, rem[m], B, measure(rem[m], B, args.window), "idle")
            for k, _ in MODES:
                for t in todo[(m, k)]:
                    set_aggr(m, t, k)
                    for B in BS: rec("mode_corun", m, t, rem[m], B, measure(rem[m], B, args.window), k)
            set_aggr(m, 0.0)
            for B in BS: rec("mode_corun", m, 0.0, rem[m], B, measure(rem[m], B, args.window), "idle_end")
    else:
        for m in SMS:                               # aggressor alone: unpaced rate, then how well each target is held
            for t in [-1.0] + GBPS:
                if t > 0 and t > 0.97 * top[m]: continue
                set_aggr(m, t); a = time.time(); time.sleep(2.5); r, _ = rec_aggr(m, t, a, time.time())
                if t < 0: top[m] = r
        plan = [("corun", m, rem[m], [0.0] + [t for t in GBPS if t <= 0.97 * top[m]] + [-1.0]) for m in SMS]
        if FIX: plan += [("corun_fixed", m, rem[FIX], [0.0] + [t for t in GBPS if t <= 0.97 * top[m]]) for m in SMS if m < FIX]
    for series, m, g, targets in plan:
        for t in targets:
            set_aggr(m, t)
            for B in BS: rec(series, m, t, g, B, measure(g, B, args.window), args.kernel)
    set_aggr(0, 0.0)
finally:
    with open(CTL + ".tmp", "w") as f: json.dump(dict(quit=True), f)
    os.replace(CTL + ".tmp", CTL)
    try: ag.wait(15)
    except Exception: ag.send_signal(signal.SIGTERM)
time.sleep(2)
decode_alone("alone_end", 3.0)
gpm.stop = True; out_f.close()
if os.path.exists(CTL): os.remove(CTL)
print("wrote", args.out)
