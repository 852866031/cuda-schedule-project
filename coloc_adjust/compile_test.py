#!/usr/bin/env python
"""Overnight 3: does the compiled tenant (V1) survive shape changes? Tenant alone on 32 SMs with --compile
--no-slow-shapes and a 50 GB/s allowance, while the budget file steps 8 -> 4 -> 3 -> 8 -> 9.5 GiB.
Prints the tenant's rows around each change and the longest gap between rows.
  SCRIPT=compile_test.py GPU=1 ./run_knee_corun.sh"""
import csv, json, os, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); BUD = "/dev/shm/corun2_test_budget"; OUT = os.path.join(HERE, "results_o3", "compile_test_ft.csv")
def setb(gib, seq):
    json.dump(dict(bytes=gib * 2**30, seq=seq, gbps=50.0), open(BUD + ".tmp", "w")); os.replace(BUD + ".tmp", BUD)
setb(8, 1)
p = subprocess.Popen([sys.executable, os.path.join(HERE, "ft_worker.py"), "--out", OUT, "--fresh", "--budget-file", BUD, "--state-file", "/dev/shm/corun2_test_state",
                      "--snapshot", "/dev/shm/corun2_test_snapshot.pt", "--green-sms", "32", "--compile", "--no-slow-shapes"],
                     stdout=open(OUT.replace(".csv", ".log"), "w"), stderr=subprocess.STDOUT, env=dict(os.environ, COLOC_XFER_DIR="/dev/shm/corun2x"))
def rows():
    try: return [r for r in csv.DictReader(open(OUT))]
    except Exception: return []
t0 = time.time()
while len([r for r in rows() if r.get("tok_s")]) < 5 and time.time() - t0 < 400: time.sleep(1)
print(f"first 5 steps after {time.time() - t0:.0f} s", flush=True)
marks = []
for i, gib in enumerate((4, 3, 8, 9.5), 2):
    time.sleep(20); setb(gib, i); marks.append((time.time(), gib)); print(f"{time.time() - t0:6.0f} s: budget -> {gib} GiB", flush=True)
time.sleep(60); p.send_signal(signal.SIGTERM)
try: p.wait(30)
except Exception: p.kill()
R = rows(); ts = [float(r["ts"]) for r in R]
for tm, gib in marks:
    after = [r for r in R if float(r["ts"]) >= tm][:4]
    print(f"budget {gib} GiB:", [(f"+{float(r['ts']) - tm:.1f}s", "cfg " + r["cfg"], r.get("tok_s") or "-", r.get("event", "")[:40]) for r in after])
gaps = sorted(((b - a, a - t0) for a, b in zip(ts, ts[1:])), reverse=True)[:3]
print("longest gaps between tenant rows:", [f"{g:.1f} s at {at:.0f} s" for g, at in gaps])
print(open(OUT.replace(".csv", ".log")).read()[-1500:])
