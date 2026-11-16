#!/usr/bin/env python3
"""Overnight plan 3, real-engine runs (R4): every scheme of results_o3/ against the baseline of the same night.
  python analyze_o3.py [--dir results_o3] [--slo-tpot 20] [--slo-ttft 500]
Writes <dir>/r4_summary.md:
  1  per-request TPOT, median and p90, per rate (as analyze_overlap.py)
  2  token gaps per rate: mean, p90, p99 against the baseline's same phases; tenant tokens/s and share of time computing;
     the mean split by tenant computing or not (only where the tenant pauses after whole steps)
  3  the load served within the SLO (share of requests meeting both targets, and requests/s that do) next to the tenant's tokens/s
  4  the published allowance of the feedback schemes per phase, after the first 20 s
  5  the two controls against their Nov 9 runs (results/)"""
import argparse, json, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_o3"))
ap.add_argument("--old", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
ap.add_argument("--schemes", default="tpot_slo15,green_dyn2,c1_fix,c1_fine,c2a,c2b,c1_slow15,c1_slow5,adjust_notenant_end")
ap.add_argument("--schedule", default="0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"); ap.add_argument("--end", type=float, default=660)
ap.add_argument("--slo-tpot", type=float, default=20.0); ap.add_argument("--slo-ttft", type=float, default=500.0)
args = ap.parse_args(); D = args.dir
sched = [(float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(","))]
phases = [(t, (sched[i + 1][0] if i + 1 < len(sched) else args.end), r) for i, (t, r) in enumerate(sched)]
RATES = (0.5, 1.0, 2.0); TOKENS = 4096
def have(tag, d=None): return os.path.exists(f"{d or D}/{tag}_load.jsonl") and os.path.exists(f"{d or D}/{tag}_t0.txt")
def load(tag, d=None):
    d = d or D; t0 = float(open(f"{d}/{tag}_t0.txt").read()); reqs, gaps = [], []
    for l in open(f"{d}/{tag}_load.jsonl"):
        r = json.loads(l)
        if r["rid"].startswith("w"): continue
        ok = r.get("status") == 200 and r.get("n_out", 0) >= 2 and len(r.get("tok_t", [])) >= 3
        if ok:
            t = np.array(r["tok_t"]) - t0
            reqs.append((r["t_send"] - t0, (r["t_done"] - r["tok_t"][0]) / max(1, r["n_out"] - 1) * 1e3, (r["tok_t"][0] - r["t_send"]) * 1e3, 1))
            gaps += [((a + b) / 2, (b - a) * 1e3, a, b) for a, b in zip(t[:-1], t[1:])]
        else: reqs.append((r["t_send"] - t0, np.nan, np.nan, 0))
    R = pd.DataFrame(reqs, columns=["t", "tpot", "ttft", "ok"]); G = pd.DataFrame(gaps, columns=["t", "ms", "a", "b"])
    for X in (R, G):
        X["rps"] = np.nan
        for a, b, r in phases: X.loc[(X.t >= a + 10) & (X.t < b), "rps"] = r
    return R, G, t0
def tenant(tag, t0, d=None):
    p = f"{d or D}/{tag}_ft.csv"
    if not os.path.exists(p): return None
    f = pd.read_csv(p); f = f[pd.to_numeric(f.dt_ms, errors="coerce").notna() & pd.to_numeric(f.tok_s, errors="coerce").notna()].copy()
    f["dt"] = pd.to_numeric(f.dt_ms) / 1e3; f["sl"] = pd.to_numeric(f.sleep_ms, errors="coerce").fillna(0) / 1e3
    f["ss"] = pd.to_numeric(f.get("slice_sleep_ms", 0), errors="coerce").fillna(0) / 1e3 if "slice_sleep_ms" in f else 0.0
    f["te"] = f.ts - t0; f["ts0"] = f.te - f.dt; f["tok"] = pd.to_numeric(f.tok_s) * f.dt; f["run"] = f.dt - f.sl - f.ss
    f["block"] = (f.get("pace", "step") == "block") if "pace" in f else False
    return f
def run_frac(g, f):
    """fraction of each gap inside a tenant step's computing span (steps that pause after the step only)"""
    iv = sorted(zip(f.ts0.values, (f.te - f.sl).values)); m = []
    for a, b in iv:
        if m and a <= m[-1][1] + 1e-3: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    if not m: return np.zeros(len(g))
    m = np.array(m); s, e = m[:, 0], m[:, 1]; cum = np.concatenate([[0], np.cumsum(e - s)])
    def cov(x):
        i = np.searchsorted(s, x, side="right"); c = cum[i].copy(); j = i - 1; ok = j >= 0; c[ok] -= np.maximum(0, e[j[ok]] - x[ok]); return c
    return (cov(g.b.values) - cov(g.a.values)) / np.maximum(1e-9, (g.b - g.a).values)
pct = lambda x: "–" if not np.isfinite(x) else f"{x * 100:+.1f} %"
def req_q(R, q, rate):
    v = [np.nanpercentile(R[(R.t >= a + 10) & (R.t < b) & (R.ok == 1)].tpot, q) for a, b, r in phases if r == rate and ((R.t >= a + 10) & (R.t < b) & (R.ok == 1)).sum() > 3]
    return np.mean(v) if v else np.nan
if not have("adjust_notenant"): raise SystemExit("no baseline in " + D)
BR, BG, _ = load("adjust_notenant")
md = [f"# R4: schemes on the reference schedule against the baseline of the same night ({os.path.basename(D)})", ""]
tags = [t for t in args.schemes.split(",") if have(t)]
# ---- 1
md += ["## 1. Per-request TPOT, median and p90 (ms; change against the no-tenant run)", "", "| scheme | p50 @0.5 | p50 @1 | p50 @2 | p90 @0.5 | p90 @1 | p90 @2 |", "|---|---|---|---|---|---|---|",
       "| adjust_notenant | " + " | ".join(f"{req_q(BR, q, r):.1f}" for q in (50, 90) for r in RATES) + " |"]
DATA = {}
for tag in tags:
    R, G, t0 = load(tag); DATA[tag] = (R, G, t0, tenant(tag, t0))
    md.append(f"| {tag} | " + " | ".join(f"{req_q(R, q, r):.1f} ({(req_q(R, q, r) / req_q(BR, q, r) - 1) * 100:+.0f} %)" for q in (50, 90) for r in RATES) + " |")
# ---- 2
md += ["", "## 2. Token gaps against the baseline's same phases, and the tenant", "",
       "Tenant tokens/s over the phases of that rate; share = part of the time it was computing. The split needs at least 300 gaps and 5 % of the row's gaps per state; it is not made where the tenant pauses inside a step.", "",
       "| scheme | req/s | tenant tok/s | share computing | gap mean | gap p90 | gap p99 | mean while tenant computes | while it does not | per 1 k tok/s |", "|" + "---|" * 10]
for tag in tags:
    R, G, t0, f = DATA[tag]
    for r in RATES:
        w = G[G.rps == r]; b = BG[BG.rps == r]
        if len(w) < 300 or len(b) < 300: continue
        dur = sum(bb - (a + 10) for a, bb, rr in phases if rr == r); tok = share = np.nan; run = idle = w.iloc[:0]
        if f is not None and len(f):
            ph = pd.concat([f[(f.te >= a + 10) & (f.te < bb)] for a, bb, rr in phases if rr == r])
            tok = ph.tok.sum() / dur; share = ph.run.sum() / dur
            if not ph.block.any():
                rf = run_frac(w, f); run, idle = w[rf >= 0.9], w[rf <= 0.1]
                run = run if len(run) >= max(300, 0.05 * len(w)) else run.iloc[:0]; idle = idle if len(idle) >= max(300, 0.05 * len(w)) else idle.iloc[:0]
        m = w.ms.mean() / b.ms.mean() - 1
        md.append(f"| {tag} | {r:g} | {'–' if not np.isfinite(tok) else f'{tok:,.0f}'} | {'–' if not np.isfinite(share) else f'{share * 100:.0f} %'} | {pct(m)} | {pct(np.percentile(w.ms, 90) / np.percentile(b.ms, 90) - 1)} | "
                  f"{pct(np.percentile(w.ms, 99) / np.percentile(b.ms, 99) - 1)} | {pct(run.ms.mean() / b.ms.mean() - 1 if len(run) else np.nan)} | {pct(idle.ms.mean() / b.ms.mean() - 1 if len(idle) else np.nan)} | "
                  f"{'–' if not (np.isfinite(tok) and tok > 200) else f'{m * 100 / tok * 1e3:.2f} %'} |")
# ---- 3
def slo_row(tag, R, f):
    cells = []
    for r in (0.5, 1.0, 2.0, 3.0, 4.0):
        w = R[R.rps == r]
        if not len(w): cells.append("–"); continue
        good = ((w.ok == 1) & (w.tpot <= args.slo_tpot) & (w.ttft <= args.slo_ttft)).mean(); cells.append(f"{good * 100:.0f} % ({good * r:.2f})")
    w = R[R.rps.notna()]; good = ((w.ok == 1) & (w.tpot <= args.slo_tpot) & (w.ttft <= args.slo_ttft))
    span = sum(b - (a + 10) for a, b, _ in phases); tok = f[(f.te >= 10) & (f.te < args.end)].tok.sum() / args.end if f is not None and len(f) else 0
    return f"| {tag} | " + " | ".join(cells) + f" | {good.mean() * 100:.1f} % | {good.sum() / span:.2f} | {int((w.ok == 0).sum())} | {tok:,.0f} |"
md += ["", f"## 3. Load served within the SLO (TPOT <= {args.slo_tpot:g} ms and TTFT <= {args.slo_ttft:g} ms per request)", "",
       "Per rate: share of requests within both targets (requests/s within them). Last columns: the whole schedule.", "",
       "| scheme | 0.5 req/s | 1 | 2 | 3 | 4 | all requests | req/s within the SLO | failed | tenant tok/s over the run |", "|" + "---|" * 10, slo_row("adjust_notenant", BR, None)]
for tag in tags: md.append(slo_row(tag, DATA[tag][0], DATA[tag][3]))
# ---- 4
rows4 = []
for tag in tags:
    p = f"{D}/{tag}_controller.csv"
    if not os.path.exists(p): continue
    c = pd.read_csv(p); t0 = DATA[tag][2]; c["rel"] = c.t - t0
    for col, unit in (("gbps_loop", "GB/s"), ("tok_s_cap", "tok/s")):
        if col not in c or not pd.to_numeric(c[col], errors="coerce").notna().any(): continue
        v = pd.to_numeric(c[col], errors="coerce")
        if v.nunique() <= 1: continue
        for a, b, r in phases:
            if r not in RATES: continue
            x = v[(c.rel >= a + 20) & (c.rel < b)].dropna()
            if len(x): rows4.append(f"| {tag} | {a:.0f}–{b:.0f} s, {r:g} req/s | {x.min():,.0f}–{x.max():,.0f} {unit} | {x.max() / max(1e-9, x.min()):.1f} | {x.median():,.0f} |")
if rows4: md += ["", "## 4. The allowance the feedback loops published, after the first 20 s of each phase", "", "| scheme | phase | range | max ÷ min | median |", "|---|---|---|---|---|"] + rows4
# ---- 5
old = [t for t in ("tpot_slo15", "green_dyn2") if t in tags and have(t, args.old) and have("adjust_notenant", args.old)]
if old:
    OB, _, _ = load("adjust_notenant", args.old)
    md += ["", "## 5. The controls against their Nov 9 runs (each against the baseline of its own night)", "", "| scheme | night | p50 @0.5 | p50 @1 | p50 @2 | p90 @0.5 | p90 @1 | p90 @2 |", "|---|---|---|---|---|---|---|---|",
           "| adjust_notenant (ms) | Nov 9 | " + " | ".join(f"{req_q(OB, q, r):.1f}" for q in (50, 90) for r in RATES) + " |",
           "| adjust_notenant (ms) | tonight | " + " | ".join(f"{req_q(BR, q, r):.1f}" for q in (50, 90) for r in RATES) + " |"]
    for t in old:
        OR, _, _ = load(t, args.old)
        md.append(f"| {t} | Nov 9 | " + " | ".join(f"{(req_q(OR, q, r) / req_q(OB, q, r) - 1) * 100:+.0f} %" for q in (50, 90) for r in RATES) + " |")
        md.append(f"| {t} | tonight | " + " | ".join(f"{(req_q(DATA[t][0], q, r) / req_q(BR, q, r) - 1) * 100:+.0f} %" for q in (50, 90) for r in RATES) + " |")
open(f"{D}/r4_summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
