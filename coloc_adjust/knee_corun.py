#!/usr/bin/env python
"""Co-run version of knee_probe.py: the same synthetic Llama-3-8B decode step, measured while the
fine-tune tenant (ft_worker.py, fixed shape) runs on M SMs of the same GPU under MPS.

Series (one CSV row per timed group of decode steps, or per tenant-alone window):
  alone          decode alone, on the full device and on each "remaining" green context
  tenant_alone   tenant alone on M SMs (its own SM -> tokens/s curve)
  shared         decode on the full device, tenant on the first M SMs (the green_dyn2 placement)
  disjoint       decode on the SMs left after an M-SM split, tenant on the first M (no SM is shared)
  grid           (--grid-decode) decode on the last N SMs, tenant on the first M SMs, for every N x M;
                 the two overlap on max(0, N + M - 170) SMs, so N = 170 is "shared" and N <= 170 - M is disjoint
  alone_end      decode alone again after the tenant exited (drift check)
  idle, calib    (--calib-sms) nothing running, then decode alone on the first N SMs: calibration of the counter below
Every row also carries the device-wide NVML GPM counters averaged over its window: bw_util = GPU-memory
bandwidth utilisation (%), sm_util, sm_occ, tensor_util. The counter is per device, so it is one party's
traffic only when that party runs alone; in a co-run it is the sum. gb_per_step is the weights + KV a decode
step must read, so gb_per_step / step time is the known read rate the counter is checked against.
The tenant switches between green contexts it prebuilt at start (budget file's "sms" field), so one
tenant process serves every M. Start through run_knee_corun.sh (MPS daemon, GPU selection).
"""
import argparse, csv, json, os, signal, subprocess, sys, threading, time
import pynvml as N
import torch, torch.nn.functional as F
from green_ctx import GreenCtx

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/knee_corun.csv")
ap.add_argument("--tenant-sms", default="0,128,96,64,32,16", help="tenant SM counts for the shared series; 0 = full device")
ap.add_argument("--disjoint", default="16,32,64,96", help="tenant SM counts for the disjoint series (decode gets the rest)")
ap.add_argument("--grid-decode", default="", help="decode SM counts (170 - a multiple of 8, or 170) for the full N x M grid; replaces the shared and disjoint series")
ap.add_argument("--calib-sms", default="16,32,48,64,96,128", help="decode alone on the first N SMs (plus the full device from the alone series): known GB/s vs the bandwidth counter")
ap.add_argument("--gpm-period", type=float, default=0.25, help="seconds between GPM samples (the counter refreshes about every 100 ms)")
ap.add_argument("--batches", default="1,4,8", help="running requests in the decode step")
ap.add_argument("--ctx", type=int, default=6272)
ap.add_argument("--window", type=float, default=8.0, help="seconds per measured point")
ap.add_argument("--sm-tol", type=int, default=8, help="a tenant step counts as running on M SMs if its SM count is within this many of M; use 3 when the set holds both 8 and 16")
ap.add_argument("--tenant-tier", type=int, default=2, help="ft_worker shape index (2 = k=4, the usual shape)")
ap.add_argument("--budget-gib", type=float, default=9.5, help="tenant memory cap; each prebuilt context's stream keeps ~0.6 GiB of allocator cache on top of the 5 GiB shape")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
BUD, STATE, SNAP = "/dev/shm/knee_corun_budget", "/dev/shm/knee_corun_state", "/dev/shm/knee_corun_snapshot.pt"
FT_CSV = args.out.replace(".csv", "_ft.csv")
TEN = [int(x) for x in args.tenant_sms.split(",")]; DIS = [int(x) for x in args.disjoint.split(",") if x]
GRID = [int(x) for x in args.grid_decode.split(",") if x]; FULL = 170
if GRID: DIS = []
CALIB = [int(x) for x in args.calib_sms.split(",") if x]
BS = [int(b) for b in args.batches.split(",")]; BMAX = max(BS)

# ---- decode side: same step as knee_probe.py
H, I, L, NKV, GRP, HD, V = 4096, 14336, 32, 8, 4, 128, 128256
dev, dt = "cuda", torch.bfloat16
torch.cuda.init()
rem = {k: GreenCtx(k, remaining=True) for k in sorted(set(DIS) | {FULL - n for n in GRID if n < FULL})}   # decode on the SMs left after a k-SM split
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

# ---- tenant side
seq = {"n": 0}
def set_sms(m):
    seq["n"] += 1
    with open(BUD + ".tmp", "w") as f: json.dump(dict(bytes=args.budget_gib * 2**30, seq=seq["n"], sms=m), f)
    os.replace(BUD + ".tmp", BUD)
def ft_rows():
    """finished tenant steps: (t_start, t_end, tokens, sms)"""
    out = []
    try:
        for r in csv.DictReader(open(FT_CSV)):
            if r["tok_s"] and r["dt_ms"] and float(r["dt_ms"]) > 0:
                te = float(r["ts"]); d = float(r["dt_ms"]) / 1e3
                out.append((te - d, te, float(r["tok_s"]) * d, int(r["sms"] or 0)))
    except Exception: pass
    return out
def wait_switch(pred, timeout=90, extra=1):
    """wait until `extra`+1 finished tenant steps satisfy pred(sms) in a row"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = ft_rows()
        if len(rows) > extra and all(pred(r[3]) for r in rows[-(extra + 1):]) and rows[-1][0] > t0: return rows[-1][3]
        if ft.poll() is not None: raise RuntimeError("tenant exited")
        time.sleep(0.2)
    raise RuntimeError("tenant did not switch")
def tenant_rate(t0, t1):
    rows = [r for r in ft_rows() if r[0] >= t0 and r[1] <= t1]
    if not rows: return float("nan"), 0
    return sum(r[2] for r in rows) / (rows[-1][1] - rows[0][0]), len(rows)
def near(m): return (lambda s: s == 0) if m == 0 else (lambda s: s > 0 and abs(s - m) <= args.sm_tol)

# ---- device-wide GPM counters (NVML), sampled in a thread
class Gpm:
    IDS = dict(bw_util=N.NVML_GPM_METRIC_DRAM_BW_UTIL, sm_util=N.NVML_GPM_METRIC_SM_UTIL, sm_occ=N.NVML_GPM_METRIC_SM_OCCUPANCY,
               tensor_util=N.NVML_GPM_METRIC_ANY_TENSOR_UTIL)
    def __init__(self, period, path):
        N.nvmlInit(); v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.h = N.nvmlDeviceGetHandleByUUID(v.encode()) if v.startswith("GPU-") else N.nvmlDeviceGetHandleByIndex(int(v))
        print("GPM counters from", N.nvmlDeviceGetUUID(self.h), flush=True)
        self.period, self.rows, self.stop = period, [], False
        self.f = open(path, "w"); self.f.write("t0,t1," + ",".join(self.IDS) + "\n")
        self.th = threading.Thread(target=self.run, daemon=True); self.th.start()
    def run(self):
        prev = N.nvmlGpmSampleAlloc(); N.nvmlGpmSampleGet(self.h, prev); tp = time.time()
        while not self.stop:
            time.sleep(self.period)
            cur = N.nvmlGpmSampleAlloc(); N.nvmlGpmSampleGet(self.h, cur); tc = time.time()
            g = N.c_nvmlGpmMetricsGet_t(); g.version = N.NVML_GPM_METRICS_GET_VERSION; g.numMetrics = len(self.IDS); g.sample1 = prev; g.sample2 = cur
            for i, m in enumerate(self.IDS.values()): g.metrics[i].metricId = m
            try:
                N.nvmlGpmMetricsGet(g)
                vals = [g.metrics[i].value if g.metrics[i].nvmlReturn == 0 else float("nan") for i in range(len(self.IDS))]
            except Exception: vals = [float("nan")] * len(self.IDS)
            self.rows.append((tp, tc, vals)); self.f.write(f"{tp:.3f},{tc:.3f}," + ",".join(f"{x:.2f}" for x in vals) + "\n"); self.f.flush()
            N.nvmlGpmSampleFree(prev); prev, tp = cur, tc
    def mean(self, t0, t1, skip=0.5):
        """per-counter mean over the samples that lie inside [t0 + skip, t1]; nan samples dropped"""
        time.sleep(max(0.0, t1 + self.period - time.time()))            # let the last sample of the window land
        out = {}
        for i, k in enumerate(self.IDS):
            v = [r[2][i] for r in self.rows if r[0] >= t0 + skip and r[1] <= t1 and r[2][i] == r[2][i]]
            out[k] = f"{sum(v) / len(v):.2f}" if v else ""
        return out
gpm = Gpm(args.gpm_period, args.out.replace(".csv", "_gpm.csv"))

rows = []; out_f = open(args.out, "w", newline=""); out_w = None
def emit(new):
    """append rows to the CSV as they are measured, so a failed run keeps its data"""
    global out_w
    if out_w is None: out_w = csv.DictWriter(out_f, fieldnames=list(new[0])); out_w.writeheader()
    out_w.writerows(new); out_f.flush(); rows.extend(new)
def rec(series, m_req, m, g, B, groups, t0, t1):
    rate, n = tenant_rate(t0, t1) if m_req is not None else (float("nan"), 0)
    c = gpm.mean(groups[0][0], groups[-1][1]); gb = W_GB + B * KV_GB_PER_REQ
    emit([dict(series=series, tenant_sms_req="" if m_req is None else m_req, tenant_sms="" if m is None else m,
               decode_sms=g.sm_count if g else 0, B=B, grp=i, step_ms=f"{ms:.3f}", tenant_tok_s=f"{rate:.0f}", tenant_steps=n,
               gb_per_step=f"{gb:.3f}", **c)
          for i, (_, _, ms) in enumerate(groups)])
    med = sorted(x[2] for x in groups)[len(groups) // 2] if groups else float("nan")
    print(f"{series:12s} tenant SMs {str(m):>4s}  decode SMs {g.sm_count if g else 170:3d}  B={B}  step {med:7.2f} ms  {gb / med * 1e3:5.0f} GB/s  "
          f"tenant {rate:7.0f} tok/s ({n} steps)  bw {c['bw_util']:>6s} %  sm_util {c['sm_util']:>6s}  sm_occ {c['sm_occ']:>6s}", flush=True)
def rec_nodecode(series, m_req, m, t0, t1):
    rate, n = tenant_rate(t0, t1) if m_req is not None else (float("nan"), 0); c = gpm.mean(t0, t1)
    emit([dict(series=series, tenant_sms_req="" if m_req is None else m_req, tenant_sms="" if m is None else m, decode_sms="", B=0, grp=0,
               step_ms="", tenant_tok_s=f"{rate:.0f}", tenant_steps=n, gb_per_step="", **c)])
    print(f"{series:12s} tenant SMs {str(m):>4s}  {rate:7.0f} tok/s ({n} steps)  bw {c['bw_util']:>6s} %  sm_util {c['sm_util']:>6s}  sm_occ {c['sm_occ']:>6s}", flush=True)

def decode_alone(tag, seconds):
    for g in [None] + list(rem.values()):
        for B in BS: rec(tag, None, None, g, B, measure(g, B, seconds), 0, 0)

decode_alone("alone", 3.0)
if CALIB:
    time.sleep(1); t0 = time.time(); time.sleep(3); rec_nodecode("idle", None, None, t0, time.time())
    for k in CALIB:                                # decode alone on the first k SMs: known GB/s next to the counter
        g = GreenCtx(k)
        for B in BS: rec("calib", None, None, g, B, measure(g, B, 4.0), 0, 0)
        g.close()
for f in (BUD, STATE, SNAP, FT_CSV):
    if os.path.exists(f): os.remove(f)
set_sms(TEN[0])
pool = sorted({m for m in TEN + DIS if m > 0}, reverse=True)
ft = subprocess.Popen([sys.executable, os.path.join(HERE, "ft_worker.py"), "--out", FT_CSV, "--fresh", "--fixed-tier", str(args.tenant_tier),
                       "--budget-file", BUD, "--state-file", STATE, "--snapshot", SNAP, "--snapshot-every", "100000000",
                       "--green-prebuilt", "--green-dynamic", "--green-set", ",".join(map(str, pool))],
                      stdout=open(args.out.replace(".csv", "_ft.log"), "w"), stderr=subprocess.STDOUT)
try:
    wait_switch(near(TEN[0]), timeout=180, extra=3)
    for m in TEN:                                  # tenant alone
        set_sms(m); got = wait_switch(near(m)); t0 = time.time(); time.sleep(args.window); t1 = time.time()
        rec_nodecode("tenant_alone", m, got, t0, t1)
    for series, ms in (("grid", TEN),) if GRID else (("shared", TEN), ("disjoint", DIS)):
        for m in ms:
            set_sms(m); got = wait_switch(near(m))
            for g in [None if n >= FULL else rem[FULL - n] for n in GRID] if GRID else [rem[m] if series == "disjoint" else None]:
                for B in BS:
                    t0 = time.time(); groups = measure(g, B, args.window); t1 = time.time()
                    rec(series, m, got, g, B, groups, t0, t1)
finally:
    ft.send_signal(signal.SIGTERM)
    try: ft.wait(30)
    except Exception: ft.kill()
time.sleep(2)
decode_alone("alone_end", 3.0)
gpm.stop = True; out_f.close()
for f in (BUD, STATE, SNAP): 
    if os.path.exists(f): os.remove(f)
print("wrote", args.out)
