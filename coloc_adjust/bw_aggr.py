#!/usr/bin/env python
"""Synthetic bandwidth aggressor: a read-only kernel that streams a buffer much larger than the L2 cache,
on M SMs (prebuilt green contexts, the first M SMs like the tenant) at a set GB/s. The bytes per call are
known (one pass over a chunk), so its traffic is its call rate x chunk size, with no profile needed.

Worker (started by aggr_corun.py):  bw_aggr.py --ctl F --out rows.csv --sms-set 96,64,32,16
  control file F (JSON): {"seq": n, "sms": M (0 = full device), "gbps": target, "kernel": mode}   target 0 = idle, -1 = no pacing
  modes on one float32 buffer (E2.7), switched through "kernel":
    gemv32   read:  matrix-vector product over one chunk                         target in GB/s
    write32  write: fill one chunk                                               target in GB/s
    copy32   copy:  half a chunk read, half a chunk written (same bytes a call)  target in GB/s
    tiny     a 16 KB in-place add: a kernel launch with almost no bytes          target in thousand calls/s
    compute  512 x 512 float32 matrix product (3 MB working set, stays in L2)    target in thousand calls/s
  rows.csv: one row per 0.25 s: t0,t1,seq,sms,target_gbps,bytes,calls
Pacing: after each call the worker waits until bytes_done / elapsed <= target, so the traffic is spread at
sub-millisecond grain (one call is 0.3-2 ms), not in long on/off bursts.
Bench (alone, prints GB/s and the device counter per kernel and SM count):  bw_aggr.py --bench --sms-set 96,16
Also holds the Gpm sampler (device-wide NVML GPM counters) used by the drivers."""
import argparse, json, os, threading, time
import pynvml as N

class Gpm:
    IDS = dict(bw_util=N.NVML_GPM_METRIC_DRAM_BW_UTIL, sm_util=N.NVML_GPM_METRIC_SM_UTIL, sm_occ=N.NVML_GPM_METRIC_SM_OCCUPANCY,
               tensor_util=N.NVML_GPM_METRIC_ANY_TENSOR_UTIL)
    def __init__(self, period=0.25, path=None):
        N.nvmlInit(); v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.h = N.nvmlDeviceGetHandleByUUID(v.encode()) if v.startswith("GPU-") else N.nvmlDeviceGetHandleByIndex(int(v))
        print("GPM counters from", N.nvmlDeviceGetUUID(self.h), flush=True)
        self.period, self.rows, self.stop = period, [], False
        self.f = open(path, "w") if path else None
        if self.f: self.f.write("t0,t1," + ",".join(self.IDS) + ",power_w,clock_mhz,throttle\n")
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
            try:                                      # board power (instant), graphics clock, clock-throttle reasons (bit mask; 0x4 = power cap)
                vals += [N.nvmlDeviceGetFieldValues(self.h, [N.NVML_FI_DEV_POWER_INSTANT])[0].value.uiVal / 1e3,
                         float(N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_GRAPHICS)), float(N.nvmlDeviceGetCurrentClocksEventReasons(self.h))]
            except Exception: vals += [float("nan")] * 3
            self.rows.append((tp, tc, vals))
            if self.f: self.f.write(f"{tp:.3f},{tc:.3f}," + ",".join(f"{x:.2f}" for x in vals) + "\n"); self.f.flush()
            N.nvmlGpmSampleFree(prev); prev, tp = cur, tc
    def mean(self, t0, t1, skip=0.5):
        """per-counter mean over the samples inside [t0 + skip, t1]; nan samples dropped"""
        time.sleep(max(0.0, t1 + self.period - time.time()))
        out = {}
        for i, k in enumerate(self.IDS):
            v = [r[2][i] for r in self.rows if r[0] >= t0 + skip and r[1] <= t1 and r[2][i] == r[2][i]]
            out[k] = f"{sum(v) / len(v):.2f}" if v else ""
        n = len(self.IDS); ok = [r[2] for r in self.rows if r[0] >= t0 + skip and r[1] <= t1 and r[2][n] == r[2][n]]
        out["power_w"] = f"{sum(r[n] for r in ok) / len(ok):.0f}" if ok else ""
        out["clock_mhz"] = f"{sum(r[n + 1] for r in ok) / len(ok):.0f}" if ok else ""
        out["throttled"] = f"{sum(1 for r in ok if int(r[n + 2]) != 0) / len(ok):.2f}" if ok else ""     # share of samples with any throttle reason set
        return out

KERNELS = ("sum32", "sum64", "gemv16", "gemv32")
def make(kernel, chunk_mb, n_chunks):
    """returns (list of chunk tensors, fn(chunk), bytes per call)"""
    import torch, torch.nn.functional as F
    nbytes = chunk_mb * 2**20
    if kernel in ("sum32", "sum64"):
        dt = torch.float32 if kernel == "sum32" else torch.float64
        buf = torch.empty(n_chunks, nbytes // dt.itemsize, device="cuda", dtype=dt).uniform_()
        return [buf[i] for i in range(n_chunks)], (lambda c: c.sum()), nbytes
    dt = torch.bfloat16 if kernel == "gemv16" else torch.float32; cols = 4096
    buf = torch.empty(n_chunks, nbytes // dt.itemsize // cols, cols, device="cuda", dtype=dt).normal_(0, 0.02)
    x = torch.empty(1, cols, device="cuda", dtype=dt).normal_()
    return [buf[i] for i in range(n_chunks)], (lambda c: F.linear(x, c)), nbytes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctl"); ap.add_argument("--out")
    ap.add_argument("--sms-set", default="96,64,32,16")
    ap.add_argument("--kernel", default="gemv32", choices=KERNELS + ("write32", "copy32", "tiny", "compute"))
    ap.add_argument("--chunk-mb", type=int, default=256); ap.add_argument("--chunks", type=int, default=16, help="buffer = chunks x chunk-mb")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    import torch
    from green_ctx import GreenCtx
    torch.cuda.init(); torch.set_grad_enabled(False)
    pool = {m: GreenCtx(m) for m in sorted({int(x) for x in args.sms_set.split(",") if x}, reverse=True)}
    for m, g in pool.items(): print(f"bw_aggr: green context {m} -> {g.sm_count} of {g.device_sms} SMs", flush=True)
    def stream_of(m): return pool[m].torch_stream() if m else torch.cuda.default_stream()

    if args.bench:
        gpm = Gpm(0.25)
        for k in KERNELS:
            chunks, fn, nbytes = make(k, args.chunk_mb, args.chunks)
            for m in [0] + list(pool):
                st = stream_of(m)
                with torch.cuda.stream(st):
                    for c in chunks[:4]: fn(c)
                    st.synchronize(); t0 = time.time(); n = 0
                    while time.time() - t0 < 2.5:
                        fn(chunks[n % len(chunks)]); st.synchronize(); n += 1
                    t1 = time.time()
                c = gpm.mean(t0, t1)
                print(f"{k:7s} SMs {m or 170:3d}  {n * nbytes / (t1 - t0) / 1e9:6.0f} GB/s  {n * nbytes / (t1 - t0) / 1e9 / (m or 170):5.1f} per SM  "
                      f"call {(t1 - t0) / n * 1e3:5.2f} ms  counter {c['bw_util']:>6s} %  sm_util {c['sm_util']:>6s}", flush=True)
            del chunks, fn; torch.cuda.empty_cache()
        raise SystemExit

    import torch.nn.functional as F
    if args.kernel in ("sum32", "sum64", "gemv16"):                      # single-mode worker on its own buffer
        chunks, fn1, nbytes = make(args.kernel, args.chunk_mb, args.chunks)
        MODES = {args.kernel: ((lambda i: fn1(chunks[i])), nbytes, True)}
    else:
        n, rows = args.chunks, args.chunk_mb * 2**20 // 4 // 4096; h = rows // 2; nbytes = rows * 4096 * 4
        buf = torch.empty(n, rows, 4096, device="cuda", dtype=torch.float32).normal_(0, 0.02)
        x = torch.empty(1, 4096, device="cuda").normal_(); v = torch.zeros(4096, device="cuda")
        a = torch.empty(512, 512, device="cuda").normal_(0, 0.04); o = torch.empty(512, 512, device="cuda")
        MODES = dict(gemv32=((lambda i: F.linear(x, buf[i])), nbytes, True),
                     write32=((lambda i: buf[i].fill_(0.01)), nbytes, True),
                     copy32=((lambda i: buf[i, :h].copy_(buf[(i + n // 2) % n, h:])), 2 * h * 4096 * 4, True),
                     tiny=((lambda i: v.add_(1.0)), 0, False),
                     compute=((lambda i: torch.mm(a, a, out=o)), 0, False))
    for k, (f_, _, _) in MODES.items():
        for i in range(4): f_(i)
    torch.cuda.synchronize()
    out = open(args.out, "w"); out.write("t0,t1,seq,sms,target_gbps,bytes,calls\n"); out.flush()
    cfg = dict(seq=0, sms=0, gbps=0.0, kernel=args.kernel); mtime = None; fn, nb, by_bytes = MODES[args.kernel]; st = stream_of(0); torch.cuda.set_stream(st)
    i = 0; row_t0 = time.time(); row_bytes = row_calls = 0; t_ref = time.perf_counter(); done = 0
    def flush_row():
        global row_t0, row_bytes, row_calls
        now = time.time(); out.write(f"{row_t0:.4f},{now:.4f},{cfg['seq']},{cfg['sms']},{cfg['gbps']},{row_bytes},{row_calls}\n"); out.flush()
        row_t0, row_bytes, row_calls = now, 0, 0
    print("bw_aggr: ready", flush=True)
    while True:
        try: mt = os.stat(args.ctl).st_mtime_ns
        except FileNotFoundError: mt = mtime
        if mt != mtime:
            mtime = mt
            try: new = json.load(open(args.ctl))
            except Exception: new = cfg
            if new.get("quit"): break
            st.synchronize(); flush_row(); cfg = dict(seq=new["seq"], sms=int(new["sms"]), gbps=float(new["gbps"]), kernel=new.get("kernel") or args.kernel)
            fn, nb, by_bytes = MODES[cfg["kernel"]]
            st = stream_of(cfg["sms"]); torch.cuda.set_stream(st); t_ref = time.perf_counter(); done = 0
        if cfg["gbps"] == 0: time.sleep(0.002)
        else:
            fn(i); st.synchronize(); i = (i + 1) % args.chunks; done += nb if by_bytes else 1; row_bytes += nb; row_calls += 1
            if cfg["gbps"] > 0:
                due = t_ref + done / (cfg["gbps"] * (1e9 if by_bytes else 1e3)); now = time.perf_counter()
                if now - due > 0.02: t_ref += now - due                       # fell behind (cannot reach the target): drop the debt, no catch-up burst
                else:
                    if due - now > 0.0005: time.sleep(due - now - 0.0003)
                    while time.perf_counter() < due: pass
        if time.time() - row_t0 >= 0.25: flush_row()
    flush_row()
