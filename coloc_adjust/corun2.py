#!/usr/bin/env python
"""Overnight plan 3, synthetic runs: the Llama-3-8B decode step of knee_probe.py next to the fine-tune
tenant, with the tenant limited in different ways (R1) or in different variants (R3), and the sanity run (R0).

  --plan r0   decode alone (grouped and per-step timing), counter calibration, tenant alone on 8/16/32 SMs
              without pacing and with block pacing at a cap it never reaches (the cost of the syncs)
  --plan r1   C1: same tenant work limited in nine ways (PLAN_overnight3.md, R1)
  --plan r3   C2: variants with fewer bytes per token next to decode, at equal tenant tokens/s (R3)
  --plan smoke  a two-minute pass over the machinery

Every decode step is timed on its own: a CUDA event is recorded on the decode stream after each step of
a queued group, so the steps run exactly as in knee_corun.py (--step-timing sync waits after every step instead).
A paced point takes its cap from the tenant rate measured at its reference point, same batch size, in pass 1.
Files: <out>_steps.csv (one row per decode step), <out>_points.csv (one row per point, batch size and pass),
<out>_summary.md, <out>_ft<n>.csv (tenant steps of the n-th tenant process), <out>_gpm.csv.
Start through run_knee_corun.sh (MPS daemon, GPU selection):  SCRIPT=corun2.py GPU=1 ./run_knee_corun.sh --plan r1
"""
import argparse, bisect, csv, json, os, signal, subprocess, sys, time, statistics as st
import torch, torch.nn.functional as F
from green_ctx import GreenCtx
from bw_aggr import Gpm
from pacing import Bpt
from o3_common import VARIANTS, O3, chosen_variants

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True, choices=["r0", "r1", "r3", "smoke"])
ap.add_argument("--out", default=None, help="prefix; default results_o3/<plan>")
ap.add_argument("--batches", default="1,4,8"); ap.add_argument("--ctx", type=int, default=6272)
ap.add_argument("--window", type=float, default=20.0); ap.add_argument("--alone-window", type=float, default=5.0)
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--step-timing", default="event", choices=["event", "sync"])
ap.add_argument("--tier", type=int, default=2, help="tenant shape index (2 = 4 micro-batches, the usual shape)")
ap.add_argument("--budget-gib", type=float, default=9.5)
ap.add_argument("--bpt-json", default=os.path.join(O3, "bytes_per_token.json"))
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = args.out or os.path.join(O3, args.plan); os.makedirs(os.path.dirname(OUT), exist_ok=True)
BUD, STATE, SNAP = "/dev/shm/corun2_budget", "/dev/shm/corun2_state", "/dev/shm/corun2_snapshot.pt"
os.makedirs("/dev/shm/corun2x", exist_ok=True)
BS = [int(b) for b in args.batches.split(",")]; BMAX = max(BS); FULL = 170
bpt = Bpt(args.bpt_json)

# ---------------------------------------------------------------- the plans
def P(label, variant="base", sms=32, disjoint=True, pace="none", slc=8, ref=None, decode=True, cap=None):
    return dict(label=label, variant=variant, sms=sms, disjoint=disjoint, pace=pace, slc=slc, ref=ref, decode=decode, cap=cap)
CALIB = []
if args.plan == "r1":
    pts = [P("a", sms=8), P("b", sms=16), P("e", sms=32),
           P("c", pace="step", ref=("a", "tok")), P("d", pace="step", ref=("b", "tok")),
           P("f", sms=0, disjoint=False, pace="step", ref=("a", "tok")),
           P("g", pace="block", slc=1, ref=("a", "tok")), P("h", pace="block", slc=8, ref=("a", "tok")), P("i", pace="block", slc=48, ref=("a", "tok"))]
    SEGS = [[("base", pts)], [("base", pts[::-1])]]
elif args.plan == "r3":
    cv = chosen_variants(); V1, V2 = cv.get("V1"), cv.get("V2") or "top4"
    seg_v1 = [P("q1", V1), P("r1", V1, pace="step", ref=("p", "tok"))] if V1 else []
    seg_v2 = [P("q2", V2), P("r2", V2, pace="step", ref=("p", "tok"))]
    p_, s_ = P("p"), P("s", pace="step", ref=("q2", "gb"))
    SEGS = [[("base", [p_])] + ([(V1, seg_v1)] if V1 else []) + [(V2, seg_v2), ("base", [s_])],
            [("base", [s_, p_]), (V2, seg_v2[::-1])] + ([(V1, seg_v1[::-1])] if V1 else [])]
    print(f"r3: V1 = {V1}, V2 = {V2}", flush=True)
elif args.plan == "r0":
    CALIB = [16, 32, 48, 64, 96, 128]
    ta = [P(f"alone{m}", sms=m, decode=False) for m in (8, 16, 32)]
    ta += [P(f"alone32_block{n}", sms=32, pace="block", slc=n, cap=1e9, decode=False) for n in (1, 8, 48)]
    SEGS = [[("base", ta)]]
    if args.window == 20.0: args.window = 8.0
else:
    pts = [P("a", sms=8), P("c", pace="step", ref=("a", "tok")), P("g", pace="block", slc=1, ref=("a", "tok")), P("f", sms=0, disjoint=False, pace="step", ref=("a", "tok"))]
    SEGS = [[("base", pts), ("top4", [P("q2", "top4"), P("r2", "top4", pace="step", ref=("a", "tok"))]), ("losschunk", [P("q1", "losschunk")])]]
    args.window, args.alone_window, BS, BMAX = 4.0, 2.0, [1], 1
SEGS = SEGS[:args.passes]
ALL = [p for ps in SEGS for _, pp in ps for p in pp]
SM_SET = sorted({p["sms"] for p in ALL if p["sms"] > 0}, reverse=True)
REM = sorted({p["sms"] for p in ALL if p["decode"] and p["disjoint"]} | ({8, 16, 32} if args.plan == "r0" else set()))

# ---------------------------------------------------------------- decode side: the step of knee_probe.py
H, I, L, NKV, GRP, HD, V = 4096, 14336, 32, 8, 4, 128, 128256
dev, dt = "cuda", torch.bfloat16
torch.cuda.init()
rem = {k: GreenCtx(k, remaining=True) for k in REM}        # decode on the SMs left after a k-SM split
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
    """decode steps for `seconds`. Returns (steps [(t_start, ms)], groups [(t_start, t_end, mean ms by wall clock)])."""
    stream = g.torch_stream() if g else torch.cuda.default_stream(); steps_out, groups = [], []
    with torch.cuda.stream(stream):
        x = w(B, H)
        t = time.perf_counter(); step(x, B); stream.synchronize(); step(x, B); stream.synchronize()
        n = max(1, min(20, int(0.25 / ((time.perf_counter() - t) / 2))))
        t_begin = time.time()
        while time.time() - t_begin < seconds:
            ts = time.time(); t = time.perf_counter()
            if args.step_timing == "event":
                evs = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
                evs[0].record(stream)
                for i in range(n): step(x, B); evs[i + 1].record(stream)
                stream.synchronize(); torch.cuda.synchronize(); wall = time.perf_counter() - t
                acc = ts
                for i in range(n):
                    ms = evs[i].elapsed_time(evs[i + 1]); steps_out.append((acc, ms)); acc += ms / 1e3
            else:
                acc = ts
                for i in range(n):
                    t1 = time.perf_counter(); step(x, B); stream.synchronize(); ms = (time.perf_counter() - t1) * 1e3
                    steps_out.append((acc, ms)); acc += ms / 1e3
                torch.cuda.synchronize(); wall = time.perf_counter() - t
            groups.append((ts, time.time(), wall / n * 1e3))
    return steps_out, groups

# ---------------------------------------------------------------- tenant side
seq = {"n": 0}; ft = {"p": None, "csv": None, "n": 0}
def set_budget(sms, cap=None, pace="step", slc=8):
    seq["n"] += 1
    with open(BUD + ".tmp", "w") as f:
        json.dump(dict(bytes=args.budget_gib * 2**30, seq=seq["n"], sms=sms, tok_s_cap=cap, pace="block" if pace == "block" else "step", slice=slc), f)
    os.replace(BUD + ".tmp", BUD)
def ft_rows():
    """finished tenant steps: dict(t0, t1, run_s, tokens, sms, cap, pace)"""
    out = []
    try:
        for r in csv.DictReader(open(ft["csv"])):
            if r.get("tok_s") and r.get("dt_ms") and float(r["dt_ms"]) > 0:
                te = float(r["ts"]); d = float(r["dt_ms"]) / 1e3; sl = float(r.get("sleep_ms") or 0) / 1e3; ss = float(r.get("slice_sleep_ms") or 0) / 1e3
                out.append(dict(t0=te - d, t1=te, run_end=te - sl, run_s=d - sl - ss, tokens=float(r["tok_s"]) * d, sms=int(r["sms"] or 0),
                                cap=float(r["cap_tok_s"]) if r.get("cap_tok_s") else None, pace=r.get("pace") or "step", slices=int(r.get("slices") or 0)))
    except Exception: pass
    return out
def start_tenant(variant):
    stop_tenant()
    for f in (BUD, STATE, SNAP):
        if os.path.exists(f): os.remove(f)
    ft["n"] += 1; ft["csv"] = f"{OUT}_ft{ft['n']}.csv"
    set_budget(SM_SET[0])
    cmd = [sys.executable, os.path.join(HERE, "ft_worker.py"), "--out", ft["csv"], "--fresh", "--fixed-tier", str(args.tier), "--budget-file", BUD,
           "--state-file", STATE, "--snapshot", SNAP, "--snapshot-every", "100000000", "--green-prebuilt", "--green-dynamic",
           "--green-set", ",".join(map(str, SM_SET)), "--bpt-json", args.bpt_json] + VARIANTS[variant]
    ft["p"] = subprocess.Popen(cmd, stdout=open(f"{OUT}_ft{ft['n']}.log", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, COLOC_XFER_DIR="/dev/shm/corun2x"))
    print(f"tenant {ft['n']} ({variant}) starting", flush=True)
    wait_ready(dict(sms=SM_SET[0], pace="none", cap=None), n=4, timeout=420)
def stop_tenant():
    if ft["p"] is None: return
    ft["p"].send_signal(signal.SIGTERM)
    try: ft["p"].wait(30)
    except Exception: ft["p"].kill()
    ft["p"] = None; time.sleep(2)
def matches(r, want):
    m = want["sms"]
    if (m == 0) != (r["sms"] == 0) or (m and abs(r["sms"] - m) > 3): return False
    if (r["pace"] == "block") != (want["pace"] == "block"): return False
    if want["cap"] is None: return r["cap"] is None
    return r["cap"] is not None and abs(r["cap"] - want["cap"]) <= 0.02 * want["cap"] + 1
def wait_ready(want, n=2, timeout=120):
    """wait until the last n finished tenant steps all started after now and match the wanted setting"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = [r for r in ft_rows() if r["t0"] >= t0]
        if len(rows) >= n and all(matches(r, want) for r in rows[-n:]): return rows[-1]["sms"]
        if ft["p"].poll() is not None: raise RuntimeError(f"tenant exited (see {OUT}_ft{ft['n']}.log)")
        time.sleep(0.2)
    raise RuntimeError(f"tenant did not reach {want}")
def tenant_stats(t0, t1):
    rows = [r for r in ft_rows() if r["t0"] >= t0 and r["t1"] <= t1]
    if not rows: return dict(tok_s=float("nan"), tok_s_running=float("nan"), run_share=float("nan"), steps=0, slices=0), []
    span = rows[-1]["t1"] - rows[0]["t0"]; tok = sum(r["tokens"] for r in rows); run = sum(r["run_s"] for r in rows)
    return dict(tok_s=tok / span, tok_s_running=tok / run, run_share=run / span, steps=len(rows), slices=sum(r["slices"] for r in rows) / len(rows)), rows

# ---------------------------------------------------------------- recording
gpm = Gpm(0.25, OUT + "_gpm.csv")
steps_f = open(OUT + "_steps.csv", "w", newline=""); steps_w = csv.writer(steps_f); steps_w.writerow(["pass", "label", "B", "t", "ms", "tenant_running"])
points = []; ref_rate = {}
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
def record(pass_i, label, B, g, steps, groups, pt=None, cap=None, sms_got=None, t0=None, t1=None):
    c = gpm.mean(groups[0][0], groups[-1][1]) if groups else gpm.mean(t0, t1)
    ts, rows = tenant_stats(t0, t1) if pt else (dict(tok_s=float("nan"), tok_s_running=float("nan"), run_share=float("nan"), steps=0, slices=0), [])
    row = dict(pass_=pass_i, label=label, B=B, decode_sms=(g.sm_count if g else FULL) if steps else 0, variant=pt["variant"] if pt else "", tenant_sms=sms_got if pt else "",
               pace=pt["pace"] if pt else "", slice=pt["slc"] if pt else "", cap=f"{cap:.0f}" if cap else "", n=len(steps), gb_per_step=f"{W_GB + B * KV_GB_PER_REQ:.3f}" if steps else "",
               tenant_tok_s=f"{ts['tok_s']:.0f}", tenant_tok_s_running=f"{ts['tok_s_running']:.0f}", tenant_run_share=f"{ts['run_share']:.3f}", tenant_steps=ts["steps"], slices_per_step=f"{ts['slices']:.1f}", **c)
    state = [""] * len(steps)
    if steps:
        ms = [m for _, m in steps]
        row.update(mean_ms=f"{st.fmean(ms):.3f}", p50_ms=f"{q(ms, .5):.3f}", p90_ms=f"{q(ms, .9):.3f}", p99_ms=f"{q(ms, .99):.3f}", group_mean_ms=f"{st.fmean(x[2] for x in groups):.3f}")
        if pt and rows and pt["pace"] != "block":          # split the decode steps by whether a tenant step was computing at their midpoint
            starts = [r["t0"] for r in rows]; run, idle = [], []
            for j, (t, m) in enumerate(steps):
                mid = t + m / 2e3; k = bisect.bisect_right(starts, mid) - 1
                on = k >= 0 and mid <= rows[k]["run_end"]; (run if on else idle).append(m); state[j] = int(on)
            row.update(run_ms=f"{st.fmean(run):.3f}" if run else "", idle_ms=f"{st.fmean(idle):.3f}" if idle else "", run_frac=f"{len(run) / len(steps):.3f}")
        for (t, m), s_ in zip(steps, state): steps_w.writerow([pass_i, label, B, f"{t:.4f}", f"{m:.3f}", s_])
        steps_f.flush()
    points.append(row)
    print(f"pass {pass_i} {label:16s} B={B} decode SMs {row['decode_sms']:>3}  step mean {row.get('mean_ms', ''):>7} p90 {row.get('p90_ms', ''):>7} ms"
          f"  tenant {row['tenant_tok_s']:>6} tok/s (running {row['tenant_tok_s_running']:>6}, {row['tenant_run_share']} of the time)  cap {row['cap'] or '-':>6}  bw {c['bw_util']:>6} %", flush=True)
    return ts
def decode_alone(pass_i, tag, ctxs=None):
    for g in ([None] + list(rem.values())) if ctxs is None else ctxs:
        for B in BS:
            steps, groups = measure(g, B, args.alone_window); record(pass_i, tag, B, g, steps, groups)
def write_out():
    keys = []
    for r in points:
        for k in r:
            if k not in keys: keys.append(k)
    with open(OUT + "_points.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(points)

# ---------------------------------------------------------------- run
try:
    decode_alone(0, "alone")
    if CALIB:
        time.sleep(1); t0 = time.time(); time.sleep(3); record(0, "idle", 0, None, [], [], t0=t0, t1=time.time())
        for k in CALIB:                                    # decode alone on the first k SMs: known GB/s next to the counter
            g = GreenCtx(k); decode_alone(0, "calib", [g]); g.close()
        xs, ys = [], []
        for r in points:
            if r["label"] in ("alone", "calib") and r.get("bw_util"):
                xs.append(float(r["bw_util"])); ys.append(float(r["gb_per_step"]) / float(r["p50_ms"]) * 1e3)
        K = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs); err = [abs(K * x - y) / y * 100 for x, y in zip(xs, ys)]
        json.dump(dict(gb_per_pct=K, points=len(xs), err_median_pct=st.median(err), err_max_pct=max(err)), open(os.path.join(O3, "calib.json"), "w"))
        print(f"calibration: 1 % = {K:.2f} GB/s over {len(xs)} points, error median {st.median(err):.1f} %, max {max(err):.1f} %", flush=True)
    for pass_i, segs in enumerate(SEGS, 1):
        for variant, pts in segs:
            start_tenant(variant)
            for pt in pts:
                for B in (BS if pt["decode"] else [0]):
                    cap = pt["cap"]
                    if pt["ref"]:
                        lab, mode = pt["ref"]; rr = ref_rate.get((lab, B))
                        if rr is None: print(f"  no reference rate for {pt['label']} B={B} ({lab}); skipped", flush=True); continue
                        cap = rr["tok_s"] if mode == "tok" else rr["tok_s"] * bpt.mb(args.tier, rr["variant"]) / bpt.mb(args.tier, pt["variant"])
                    want = dict(sms=pt["sms"], pace=pt["pace"], cap=cap)
                    set_budget(pt["sms"], cap, pt["pace"], pt["slc"])
                    try: got = wait_ready(want, n=2, timeout=90)
                    except RuntimeError as e:
                        print(f"  {pt['label']} B={B}: {e}; skipped", flush=True)
                        if ft["p"].poll() is not None: raise
                        continue
                    if pt["decode"]:
                        g = rem[pt["sms"]] if pt["disjoint"] else None
                        t0 = time.time(); steps, groups = measure(g, B, args.window); t1 = time.time()
                    else:
                        g, steps, groups = None, [], []; t0 = time.time(); time.sleep(args.window); t1 = time.time()
                    ts = record(pass_i, pt["label"], B, g, steps, groups, pt, cap, got, t0, t1)
                    if pass_i == 1 and (pt["label"], B) not in ref_rate: ref_rate[(pt["label"], B)] = dict(tok_s=ts["tok_s"], variant=pt["variant"])
                write_out()
            stop_tenant()
        decode_alone(pass_i, "alone")
        write_out()
finally:
    stop_tenant(); gpm.stop = True; write_out(); steps_f.close()
    for f in (BUD, STATE, SNAP):
        if os.path.exists(f): os.remove(f)

# ---------------------------------------------------------------- summary: every point against decode alone on the same SMs
def fl(x):
    try: return float(x)
    except Exception: return float("nan")
alone = {}
for r in points:
    if r["label"] == "alone": alone.setdefault((r["decode_sms"], r["B"]), []).append(r)
def base(dsms, B, key): v = [fl(r[key]) for r in alone.get((dsms, B), [])]; return sum(v) / len(v) if v else float("nan")
md = [f"# {os.path.basename(OUT)}: decode step next to the tenant (plan {args.plan}, {args.step_timing} timing, {args.window:.0f}-s windows)\n",
      "Slowdowns are against decode alone on the same SMs in this run (mean of the alone series at the start, between and after the passes), each statistic against the same statistic alone.\n",
      "| pass | point | variant | tenant SMs | pacing | cap tok/s | B | tenant tok/s | while running | running share | step mean | p90 | p99 | while tenant runs | while it pauses | counter % | W |", "|" + "---|" * 17]
for r in points:
    if r["label"] in ("alone", "calib", "idle"): continue
    if not r.get("mean_ms"):
        md.append(f"| {r['pass_']} | {r['label']} | {r['variant']} | {r['tenant_sms']} | {r['pace']}{' ' + str(r['slice']) if r['pace'] == 'block' else ''} | {r['cap']} | – | {r['tenant_tok_s']} | {r['tenant_tok_s_running']} | {r['tenant_run_share']} | – | – | – | – | – | {r['bw_util']} | {r.get('power_w', '')} |"); continue
    d, B = r["decode_sms"], r["B"]
    def pc(key, bkey=None):
        v = fl(r.get(key, "")); b = base(d, B, bkey or key); return f"{(v / b - 1) * 100:+.1f} %" if v == v and b == b else "–"
    md.append(f"| {r['pass_']} | {r['label']} | {r['variant']} | {r['tenant_sms']} | {r['pace']}{' ' + str(r['slice']) if r['pace'] == 'block' else ''} | {r['cap']} | {B} | {r['tenant_tok_s']} | {r['tenant_tok_s_running']} | {r['tenant_run_share']} | "
              f"{pc('mean_ms')} | {pc('p90_ms')} | {pc('p99_ms')} | {pc('run_ms', 'mean_ms')} | {pc('idle_ms', 'mean_ms')} | {r['bw_util']} | {r.get('power_w', '')} |")
md += ["\n## Decode alone\n", "| pass | decode SMs | B | mean ms | p50 | p90 | p99 | mean by wall clock (grouped) | counter % |", "|---|---|---|---|---|---|---|---|---|"]
for r in points:
    if r["label"] in ("alone", "calib"): md.append(f"| {r['pass_']} {r['label']} | {r['decode_sms']} | {r['B']} | {r['mean_ms']} | {r['p50_ms']} | {r['p90_ms']} | {r['p99_ms']} | {r['group_mean_ms']} | {r['bw_util']} |")
open(OUT + "_summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md)); print("wrote", OUT + "_summary.md")
