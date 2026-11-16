#!/usr/bin/env python
"""Overnight plan 3, R2 (C2, tenant side): how many bytes each tenant shape and variant moves per token.
Each variant runs alone on M SMs for --seconds; the device-wide bandwidth counter over the steady part,
divided by the tokens/s of the same part, is its MB per token.

  SCRIPT=tenant_bytes.py GPU=1 ./run_knee_corun.sh            (MPS daemon and GPU selection as for the co-runs)
Writes results_o3/r2_bytes.csv and .md, results_o3/bytes_per_token.json (read by ft_worker.py to turn a GB/s
allowance into tokens/s) and results_o3/r2_variants.json (V1 = the "same training" variant with the fewest
bytes per token if it saves at least 5 %, V2 = the "less training" variant with the fewest)."""
import argparse, csv, json, os, signal, subprocess, sys, time
from bw_aggr import Gpm
from o3_common import VARIANTS, SAME_TRAINING, LESS_TRAINING, KIND, O3, gb_per_pct

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=45.0); ap.add_argument("--sms", type=int, default=32)
ap.add_argument("--only", default="", help="comma list of run names (for a smoke test)")
ap.add_argument("--out", default=os.path.join(O3, "r2_bytes"))
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); K = gb_per_pct()
SHAPE = {0: "1 micro-batch", 1: "2 micro-batches", 2: "4 micro-batches (usual)", 3: "8 micro-batches", 4: "8 mb + activations in host RAM",
         5: "+ 12 layers in host RAM", 6: "+ 20 layers in host RAM"}
RUNS = [(f"base_cfg{c}", "base", c, args.sms) for c in (2, 0, 1, 3, 4, 5, 6)]
RUNS += [(v, v, 2, args.sms) for v in ("top8", "top4", "losschunk", "compile", "compile_losschunk", "eager", "ckpt", "fwd")]
RUNS += [(f"{v}_full", v, 2, 0) for v in ("base", "top4", "losschunk")]
if args.only: RUNS = [r for r in RUNS if r[0] in args.only.split(",")]
gpm = Gpm(0.25, args.out + "_gpm.csv"); rows = []
def ft_rows(path):
    out = []
    try:
        for r in csv.DictReader(open(path)):
            if r.get("tok_s") and r.get("dt_ms") and float(r["dt_ms"]) > 0:
                te = float(r["ts"]); d = float(r["dt_ms"]) / 1e3; out.append((te - d, te, float(r["tok_s"]) * d, float(r["loss"]) if r.get("loss") else float("nan")))
    except Exception: pass
    return out
for name, variant, cfg, sms in RUNS:
    path = f"{args.out}_{name}_ft.csv"; secs = args.seconds * (2 if "compile" in variant else 1)
    cmd = [sys.executable, os.path.join(HERE, "ft_worker.py"), "--out", path, "--fresh", "--fixed-budget-gib", "20", "--fixed-tier", str(cfg), "--duration", str(secs + 6),
           "--snapshot", "/dev/shm/tenant_bytes_snapshot.pt", "--snapshot-every", "100000000", "--state-file", "/dev/shm/tenant_bytes_state"] + (["--green-sms", str(sms)] if sms else []) + VARIANTS[variant]
    t_launch = time.time()
    p = subprocess.Popen(cmd, stdout=open(f"{args.out}_{name}_ft.log", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, COLOC_XFER_DIR="/dev/shm/corun2x"))
    try: p.wait(secs + 300)
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGTERM); time.sleep(5); p.kill()
    r = ft_rows(path); row = dict(name=name, variant=variant, kind=KIND[variant], cfg=cfg, shape=SHAPE[cfg], sms=sms or 170, status="ok" if p.returncode == 0 else f"exit {p.returncode}")
    use = [x for x in r[5:] if x[0] >= r[0][0] + 8] if len(r) > 8 else []
    if len(use) >= 3:
        t0, t1 = use[0][0], use[-1][1]; tok = sum(x[2] for x in use) / (t1 - t0); c = gpm.mean(t0, t1, skip=0.0)
        gb = K * float(c["bw_util"]) if c.get("bw_util") else float("nan")
        row.update(steps=len(use), window_s=f"{t1 - t0:.1f}", tok_s=f"{tok:.0f}", bw_util=c["bw_util"], gbps=f"{gb:.1f}", mb_per_token=f"{gb / tok * 1e3:.2f}", sm_util=c["sm_util"],
                   power_w=c.get("power_w", ""), loss=f"{use[-1][3]:.3f}")
    else:
        row["status"] = (row["status"] + "; " if row["status"] != "ok" else "") + f"only {len(r)} steps"
    rows.append(row); print(row, flush=True)
    keys = []
    for x in rows:
        for k in x:
            if k not in keys: keys.append(k)
    with open(args.out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    time.sleep(3)
gpm.stop = True
def mb(n):
    for x in rows:
        if x["name"] == n and x.get("mb_per_token"): return float(x["mb_per_token"])
    return None
table = {"base": {str(c): mb(f"base_cfg{c}") for c in range(7) if mb(f"base_cfg{c}")}}
for v in VARIANTS:
    if v != "base" and mb(v): table[v] = {"2": mb(v)}
b = mb("base_cfg2")
if not args.only:
    json.dump(table, open(os.path.join(O3, "bytes_per_token.json"), "w"), indent=1)
    same = sorted((mb(v), v) for v in SAME_TRAINING if mb(v)); less = sorted((mb(v), v) for v in LESS_TRAINING if mb(v))
    ch = dict(V1=same[0][1] if same and b and same[0][0] <= 0.95 * b else None, V2=less[0][1] if less else None, base_mb=b,
              V1_mb=same[0][0] if same else None, V2_mb=less[0][0] if less else None, gb_per_pct=K)
    json.dump(ch, open(os.path.join(O3, "r2_variants.json"), "w"), indent=1); print("chosen:", ch)
md = [f"# R2: bytes per token of every tenant shape and variant (alone, counter calibration 1 % = {K:.2f} GB/s)\n",
      "| run | kind | shape | SMs | tokens/s | GB/s while running | MB per token | vs usual | SM util % | W | loss | status |", "|" + "---|" * 12]
for x in rows:
    m = float(x["mb_per_token"]) if x.get("mb_per_token") else None
    md.append(f"| {x['name']} | {x['kind']} | {x['shape']} | {x['sms']} | {x.get('tok_s', '–')} | {x.get('gbps', '–')} | {x.get('mb_per_token', '–')} | "
              f"{f'{(m / b - 1) * 100:+.0f} %' if m and b else '–'} | {x.get('sm_util', '')} | {x.get('power_w', '')} | {x.get('loss', '')} | {x['status']} |")
open(args.out + ".md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
