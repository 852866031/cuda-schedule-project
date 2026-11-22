#!/usr/bin/env python
"""S2 of the plan of Nov 21: does the paced tenant, alone, move its bytes the way the budget says?

Each case runs ft_worker.py alone on a green context under Nsight Systems (memory read + write throughput at 10 kHz,
marks at step begin and end only) and reports
  - the step period against bytes of a step / budget,
  - the mean GB/s over whole steps against the budget,
  - how even the traffic is: GB/s in consecutive windows of 12 ms and of 100 ms (median, p90, p99, max), and the share
    of time above 200 and 500 GB/s,
  - the pacer's own account: pauses per step, lateness.
  SCRIPT=pace_check.py GPU=1 ./run_knee_corun.sh [--cases a,b] [--gbps 50] [--sms 32]
Writes results_prof/pace_summary.csv / .md and results_prof/pace_<case>_trace.csv (1 ms bins of three steps)."""
import argparse, bisect, csv, os, re, statistics as st, sys, time
import nsys_tools
from nsys_tools import Capture, drop, gbps_series
from pacing import SegTable
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="base_free,base_step,base_w12,vocab4096_free,vocab4096_step,vocab4096_w12,vocab4096_w6,vocab2048_w12,base_block1")
ap.add_argument("--gbps", type=float, default=50.0); ap.add_argument("--sms", type=int, default=32); ap.add_argument("--seconds", type=float, default=16.0)
ap.add_argument("--skip-steps", type=int, default=4); ap.add_argument("--out", default=os.path.join(HERE, "results_prof", "pace")); ap.add_argument("--keep", action="store_true")
args = ap.parse_args(); os.makedirs(os.path.dirname(args.out), exist_ok=True)
def case_flags(c):
    """<variant>_<mode>: mode free (no budget), step (byte ledger, one pause after the step), wN (byte ledger, window N ms),
    blockN (the pacer of Nov 16: a pause every N blocks, equal work assumed per checkpoint)"""
    v, mode = c.rsplit("_", 1); f = []
    m = re.fullmatch(r"(compile_|cb_)?vocab(\d+)(f?)", v)
    if m: f += ({"compile_": ["--compile"], "cb_": ["--compile-blocks", "--no-slow-shapes"]}.get(m.group(1), [])) + ["--head", "vocab", "--vocab-block", m.group(2)] + (["--head-fused"] if m.group(3) else [])
    elif v == "split": f += ["--head", "split"]
    elif v == "compile": f += ["--compile"]
    elif v == "cb": f += ["--compile-blocks", "--no-slow-shapes"]
    elif v != "base": raise SystemExit(f"unknown variant in {c}")
    if mode == "free": return v, f
    f += ["--gbps-cap", str(args.gbps)]
    if mode == "step": f += ["--pace", "bytes", "--pace-window-ms", "0"]
    elif mode[0] == "w": f += ["--pace", "bytes", "--pace-window-ms", mode[1:]]
    elif mode.startswith("block"): f += ["--pace", "block", "--slice-blocks", mode[5:]]
    else: raise SystemExit(f"unknown mode in {c}")
    return v, f
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
K = nsys_tools.calibrate(os.path.join(os.path.dirname(args.out), "seg")); rows = []
if os.path.exists(args.out + "_summary.csv"):
    rows = [{k: v for k, v in r.items() if v not in ("", None)} for r in csv.DictReader(open(args.out + "_summary.csv")) if r["case"] not in args.cases.split(",")]
for case in ([] if args.cases == "none" else args.cases.split(",")):
    v, flags = case_flags(case); ftcsv = f"{args.out}_{case}_ft.csv"; comp = "compile" in v or v.startswith("cb") or v.endswith("f")
    if os.path.exists(ftcsv): os.remove(ftcsv)
    cmd = [sys.executable, os.path.join(HERE, "ft_worker.py"), "--nvtx-steps", "--out", ftcsv, "--fresh", "--fixed-budget-gib", "20", "--fixed-tier", "2", "--duration", str(args.seconds + (180 if comp else 0)),
           "--snapshot", "/dev/shm/tenant_bytes_pace_snapshot.pt", "--snapshot-every", "100000000", "--state-file", "/dev/shm/tenant_bytes_pace_state"] + (["--green-sms", str(args.sms)] if args.sms else []) + flags
    os.makedirs("/dev/shm/pace_check_xfer", exist_ok=True)
    try:
        db = nsys_tools.capture(args.out, case, cmd, env=dict(os.environ, COLOC_XFER_DIR="/dev/shm/pace_check_xfer"), delay=150 if comp else 0)
        if not os.path.exists(ftcsv) or os.path.getsize(ftcsv) < 300: raise RuntimeError(f"the tenant did not run; see {args.out}_{case}.log")
        cap = Capture(db); begins = sorted((int(m.group(1)), s) for s, t in cap.marks for m in [re.fullmatch(r"s(\d+)begin", t)] if m)
        ends = dict((int(m.group(1)), s) for s, t in cap.marks for m in [re.fullmatch(r"s(\d+)end", t)] if m)
        use = begins[args.skip_steps:]
        if len(use) < 4: raise RuntimeError(f"only {len(begins)} steps")
        a, b = use[0][1], use[-1][1]; nsteps = len(use) - 1
        ts, g = gbps_series(cap, K, a, b); dt = (ts[-1] - ts[0]) / (len(ts) - 1) / 1e9
        total_gb = sum(g) * dt; period = (b - a) / 1e9 / nsteps
        def bins(w):
            n = max(1, int(round(w / dt))); return [sum(g[i:i + n]) / n for i in range(0, len(g) - n + 1, n)]
        b12, b100 = bins(0.012), bins(0.100)
        compute = st.mean((ends[k] - s) / 1e9 for k, s in use[:-1] if k in ends)
        tab = None
        try: tab = SegTable(os.path.join(HERE, "results_prof", f"seg_{v}.csv"))
        except Exception: pass
        ft = [r for r in csv.DictReader(open(ftcsv)) if r.get("tok_s") and r.get("dt_ms")][args.skip_steps:]
        row = dict(case=case, variant=v, budget_gbps="" if case.endswith("_free") else f"{args.gbps:g}", steps=nsteps, period_ms=f"{period * 1e3:.1f}",
                   expected_ms="" if case.endswith("_free") or tab is None else f"{tab.total / args.gbps * 1e3:.1f}", begin_to_end_ms=f"{compute * 1e3:.1f}",
                   gb_per_step=f"{total_gb / nsteps:.2f}", table_gb="" if tab is None else f"{tab.total:.2f}", mean_gbps=f"{total_gb / ((b - a) / 1e9):.1f}", tok_s=f"{4096 / period:.0f}",
                   w12_p50=f"{q(b12, .5):.0f}", w12_p90=f"{q(b12, .9):.0f}", w12_p99=f"{q(b12, .99):.0f}", w12_max=f"{max(b12):.0f}",
                   w100_p50=f"{q(b100, .5):.0f}", w100_p90=f"{q(b100, .9):.0f}", w100_max=f"{max(b100):.0f}",
                   over200_pct=f"{sum(1 for x in g if x > 200) / len(g) * 100:.1f}", over500_pct=f"{sum(1 for x in g if x > 500) / len(g) * 100:.2f}",
                   pauses_per_step=f"{st.mean(float(r['slices'] or 0) for r in ft):.0f}", pause_in_step_ms=f"{st.mean(float(r['slice_sleep_ms'] or 0) for r in ft):.0f}", pause_after_ms=f"{st.mean(float(r['sleep_ms'] or 0) for r in ft):.0f}",
                   late_per_step=f"{st.mean(float(r.get('late') or 0) for r in ft):.1f}", late_max_ms=f"{max(float(r.get('late_ms') or 0) for r in ft):.1f}", forgiven_ms_per_step=f"{st.mean(float(r.get('forgiven_ms') or 0) for r in ft):.1f}")
        n1 = max(1, int(round(0.001 / dt))); a3, b3 = use[1][1], use[4][1] if len(use) > 4 else use[-1][1]; t3, g3 = gbps_series(cap, K, a3, b3)
        with open(f"{args.out}_{case}_trace.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["t_ms", "gbps"])
            for i in range(0, len(g3) - n1 + 1, n1): wr.writerow([f"{(t3[i] - a3) / 1e6:.1f}", f"{sum(g3[i:i + n1]) / n1:.0f}"])
        if not args.keep: drop(db)
    except Exception as e:
        row = dict(case=case, status=str(e))
    rows.append(row); print(row, flush=True); time.sleep(2)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(args.out + "_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
with open(args.out + "_summary.md", "w") as f:
    f.write(f"# The paced tenant alone: bytes against the budget ({time.strftime('%Y-%m-%d')})\n\nTenant alone on a green context of {args.sms} SMs; memory read + write throughput from Nsight Systems at 10 kHz (1 % = {K['read']:.2f} GB/s).\n"
            "Case = variant_mode: free = no budget; step = byte ledger, one pause after the step; wN = byte ledger, chunks of budget x N ms; blockN = the pacer of Nov 16.\n\n"
            "| case | budget GB/s | period ms | expected ms | GB per step | mean GB/s | tok/s | 12 ms windows: p50 / p90 / p99 / max GB/s | 100 ms windows: p50 / p90 / max | time over 200 / 500 GB/s | pauses per step | late per step (max ms) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        if r.get("period_ms"): f.write(f"| {r['case']} | {r.get('budget_gbps', '')} | {r['period_ms']} | {r.get('expected_ms', '')} | {r['gb_per_step']} | {r['mean_gbps']} | {r['tok_s']} | {r['w12_p50']} / {r['w12_p90']} / {r['w12_p99']} / {r['w12_max']} | {r['w100_p50']} / {r['w100_p90']} / {r['w100_max']} | {r['over200_pct']} % / {r['over500_pct']} % | {r['pauses_per_step']} | {r['late_per_step']} ({r['late_max_ms']}) |\n")
    for r in rows:
        if r.get("status"): f.write(f"\n{r['case']}: {r['status']}\n")
print(open(args.out + "_summary.md").read())
