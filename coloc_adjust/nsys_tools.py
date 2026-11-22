"""Shared by segment_bytes.py and pace_check.py: run a command under Nsight Systems with the card's memory read and
write throughput sampled (GPU metrics, no sudo needed here), read the capture, and calibrate the counter's % into GB/s."""
import bisect, json, os, sqlite3, statistics as st, subprocess, sys, time
NSYS = "/usr/local/cuda-12.8/bin/nsys"

def gpu_index():
    """nsys wants the device index; CUDA_VISIBLE_DEVICES holds a UUID in our launchers"""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not v.startswith("GPU-"): return int(v)
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines():
        i, u = [x.strip() for x in line.split(",")]
        if u == v: return int(i)
    raise SystemExit(f"GPU {v} not found")
def capture(out, tag, cmd, env=None, delay=0, nsys=NSYS, freq=10000):
    """run cmd under nsys with GPU metrics and NVTX (collection starts `delay` s after launch); returns the path of the exported sqlite file"""
    rep = f"{out}_{tag}.nsys-rep"; db = f"{out}_{tag}.sqlite"
    for f in (rep, db):
        if os.path.exists(f): os.remove(f)
    full = [nsys, "profile", f"--gpu-metrics-devices={gpu_index()}", f"--gpu-metrics-frequency={freq}", "--sample=none", "--cpuctxsw=none", "--trace=nvtx", "-o", rep[:-9]] + ([f"--delay={delay}"] if delay else []) + cmd
    with open(f"{out}_{tag}.log", "w") as log:
        rc = subprocess.call(full, stdout=log, stderr=subprocess.STDOUT, env=env)
        if not os.path.exists(rep): raise RuntimeError(f"{tag}: nsys wrote no report (exit {rc}); see {out}_{tag}.log")
        subprocess.call([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep], stdout=log, stderr=subprocess.STDOUT)
    return db
class Capture:
    def __init__(self, db):
        c = sqlite3.connect(db).cursor()
        strs = dict(c.execute("select id, value from StringIds").fetchall())
        self.marks = [(s, t or strs.get(i) or "") for s, t, i in c.execute("select start, text, textId from NVTX_EVENTS order by start")]
        self.ranges = [(s, e, t or strs.get(i) or "") for s, e, t, i in c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null order by start")]
        mid = {n: i for n, i in c.execute("select metricName, metricId from TARGET_INFO_GPU_METRICS")}
        self.pct = {}
        for key, name in (("read", "DRAM Read Bandwidth [Throughput %]"), ("write", "DRAM Write Bandwidth [Throughput %]")):
            if name not in mid: raise RuntimeError(f"metric '{name}' not in this capture: {sorted(mid)}")
            rows = c.execute("select timestamp, value from GPU_METRICS where metricId=? order by timestamp", (mid[name],)).fetchall()
            ts = [r[0] for r in rows]; cum = [0.0]
            for k in range(1, len(rows)): cum.append(cum[-1] + rows[k][1] * (ts[k] - ts[k - 1]) / 1e9)     # % x seconds
            self.pct[key] = (ts, cum); self.raw = getattr(self, "raw", {}); self.raw[key] = (ts, [r[1] for r in rows])
    def samples(self, key, a, b):
        ts, v = self.raw[key]; i, j = bisect.bisect_left(ts, a), bisect.bisect_right(ts, b)
        return ts[i:j], v[i:j]
    def pct_s(self, key, a, b):
        ts, cum = self.pct[key]
        return cum[min(len(cum) - 1, bisect.bisect_right(ts, b) - 1)] - cum[max(0, bisect.bisect_right(ts, a) - 1)]

# ---------------------------------------------------------------- calibration: GB/s per 1 % from copies of a 1 GiB tensor
CAL = r'''
import torch, time
nv = torch.cuda.nvtx; a = torch.ones((1 << 30) // 4, dtype=torch.float32, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(); time.sleep(0.3)
for cnt in (2, 4, 8, 8, 8):
    torch.cuda.synchronize(); nv.range_push(f"calib_x{cnt}")
    for _ in range(cnt): b.copy_(a)
    torch.cuda.synchronize(); nv.range_pop(); time.sleep(0.2)
'''
def calibrate(out, recalibrate=False, keep=False, nsys=NSYS, freq=10000):
    path = out + "_calib.json"
    if os.path.exists(path) and not recalibrate: return json.load(open(path))
    db = capture(out, "calib", [sys.executable, "-c", CAL], nsys=nsys, freq=freq); cap = Capture(db); pts = []
    for s, e, name in cap.ranges:
        if name.startswith("calib_x"):
            gb = int(name[7:]) * 1.073741824; pad = 300e3
            pts.append((gb / cap.pct_s("read", s - pad, e + pad), gb / cap.pct_s("write", s - pad, e + pad)))
    if not pts: raise RuntimeError("calibration: no ranges found")
    k = dict(read=st.mean(p[0] for p in pts), write=st.mean(p[1] for p in pts), spread_pct=max(abs(x / st.mean(q for p in pts for q in p) - 1) for p in pts for x in p) * 100, points=len(pts), date=time.strftime("%Y-%m-%d"))
    json.dump(k, open(path, "w")); print(f"calibration: 1 % = {k['read']:.2f} GB/s read, {k['write']:.2f} GB/s write (spread {k['spread_pct']:.1f} %, {len(pts)} points)", flush=True)
    if not keep: drop(db)
    return k

def drop(db):
    """remove a capture (the sqlite export and the report it came from)"""
    for f in (db, db.replace(".sqlite", ".nsys-rep")):
        if os.path.exists(f): os.remove(f)
def gbps_series(cap, K, a, b):
    """(timestamps in ns, read + write GB/s) of the samples between a and b"""
    tr, vr = cap.samples("read", a, b); tw, vw = cap.samples("write", a, b); n = min(len(vr), len(vw))
    return tr[:n], [vr[i] * K["read"] + vw[i] * K["write"] for i in range(n)]
