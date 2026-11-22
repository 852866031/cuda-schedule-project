#!/usr/bin/env python3
"""Decode steps of the vLLM-engine runs, one by one: how much slower the ordinary steps are next to a tenant.
  python analyze_steps.py [--dir results_n21] [--base adjust_notenant,adjust_notenant_end] [--schemes a,b,...] [--rates 0.5,1]
A token gap of a request is one decode step. Left out: gaps during which another request was admitted (a prefill ran),
and the engine's own double and near-zero gaps (over 1.6x or under 0.4x the request's median gap; they are as frequent
with no tenant). The rest is grouped by how many requests were generating at that moment (1, 2, 3) and compared with
the same group of the baseline: p10, p50, p90, p99 of the step time. The summary column is the mean over the three groups.
Also the tenant's own account from <scheme>_ft.csv: tokens/s, share of time computing, pauses and lateness per step.
Writes <dir>/steps_summary.md."""
import argparse, bisect, csv, json, os, statistics as st
ap = argparse.ArgumentParser(); H = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--dir", default=os.path.join(H, "results_n21")); ap.add_argument("--base", default="adjust_notenant,adjust_notenant_end")
ap.add_argument("--schemes", default=""); ap.add_argument("--rates", default="0.5,1")
ap.add_argument("--schedule", default="0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"); ap.add_argument("--end", type=float, default=660)
args = ap.parse_args(); D = args.dir
sch = [(float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(","))]
PH = [(t, sch[i + 1][0] if i + 1 < len(sch) else args.end, r) for i, (t, r) in enumerate(sch)]; RATES = [float(x) for x in args.rates.split(",")]
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
def rate_at(t):
    for a, b, r in PH:
        if a <= t < b: return r, a
    return None, None
def load(tag):
    t0 = float(open(f"{D}/{tag}_t0.txt").read().split()[0]); reqs = []
    for line in open(f"{D}/{tag}_load.jsonl"):
        d = json.loads(line); tt = d.get("tok_t") or []
        if d.get("status") == 200 and len(tt) >= 20 and not str(d.get("rid", "")).startswith("w"): reqs.append((d["t_send"] - t0, [x - t0 for x in tt]))
    starts = sorted(r[1][0] for r in reqs); ends = sorted(r[1][-1] for r in reqs); sends = sorted(r[0] for r in reqs)
    by = {c: [] for c in (1, 2, 3)}; allg = []
    for t, tt in reqs:
        r, a = rate_at(t)
        if r not in RATES or t < a + 10: continue
        med = st.median(y - x for x, y in zip(tt, tt[1:]))
        for x, y in zip(tt, tt[1:]):
            allg.append((y - x) * 1e3)
            if bisect.bisect_right(sends, y) - bisect.bisect_right(starts, x) > 0 or y - x > 1.6 * med or y - x < 0.4 * med: continue
            c = bisect.bisect_right(starts, (x + y) / 2) - bisect.bisect_left(ends, (x + y) / 2)
            if c in by: by[c].append((y - x) * 1e3)
    return by, allg, len(reqs)
def tenant(tag):
    p = f"{D}/{tag}_ft.csv"
    if not os.path.exists(p): return {}
    t0 = float(open(f"{D}/{tag}_t0.txt").read().split()[0]); tok = comp = 0.0; n = 0; pauses = late = 0.0; lmax = 0.0
    for r in csv.DictReader(open(p)):
        if not r.get("dt_ms"): continue
        te = float(r["ts"]) - t0; ra, a = rate_at(te)
        if ra not in RATES: continue
        dt = float(r["dt_ms"]); tok += float(r["tok_s"]) * dt / 1e3; comp += dt - float(r.get("sleep_ms") or 0) - float(r.get("slice_sleep_ms") or 0); n += 1
        pauses += float(r.get("slices") or 0); late += float(r.get("late") or 0); lmax = max(lmax, float(r.get("late_ms") or 0))
    span = sum(b - a for a, b, r in PH if r in RATES)
    return dict(tok_s=tok / span, computing=comp / 1e3 / span, steps=n, pauses=pauses / max(n, 1), late=late / max(n, 1), late_max=lmax)
bases = [b for b in args.base.split(",") if os.path.exists(f"{D}/{b}_load.jsonl") and os.path.exists(f"{D}/{b}_t0.txt")]
B = {c: [] for c in (1, 2, 3)}; Ball = []
for b in bases:
    by, allg, _ = load(b)
    for c in B: B[c] += by[c]
    Ball += allg
tags = [t for t in args.schemes.split(",") if t] or sorted(f[:-11] for f in os.listdir(D) if f.endswith("_load.jsonl") and f[:-11] not in bases and not f.startswith("adjust_"))
P = (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99)); out = [f"# Decode steps next to each tenant, rates {args.rates} req/s\n", f"Baseline: {', '.join(bases)} (pooled). Step counts of the baseline by requests generating: " + ", ".join(f"{c}: {len(B[c])}" for c in B) + ".\n",
     "| run | requests | mean token gap | ordinary steps p10 | p50 | p90 | p99 | tenant tok/s | computing | pauses per step | late per step (max ms) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
detail = ["\n## By number of requests generating (1 / 2 / 3)\n", "| run | p10 | p50 | p90 | p99 |", "|---|---|---|---|---|"]
for t in tags + [b for b in bases if len(bases) > 1]:
    if not (os.path.exists(f"{D}/{t}_load.jsonl") and os.path.exists(f"{D}/{t}_t0.txt")): continue      # not there, or still running
    by, allg, nreq = load(t); T = tenant(t) if t not in bases else {}
    ch = {n: [(q(by[c], p) / q(B[c], p) - 1) * 100 for c in (1, 2, 3) if len(by[c]) > 50 and len(B[c]) > 50] for n, p in P}
    m = (st.mean(allg) / st.mean(Ball) - 1) * 100
    out.append(f"| {t} | {nreq} | {m:+.1f} % | " + " | ".join(f"{st.mean(ch[n]):+.1f} %" if ch[n] else "" for n, _ in P) +
               (f" | {T['tok_s']:,.0f} | {T['computing'] * 100:.0f} % | {T['pauses']:.0f} | {T['late']:.1f} ({T['late_max']:.0f}) |" if T else " | | | | |"))
    detail.append(f"| {t} | " + " | ".join(" / ".join(f"{x:+.1f}" for x in ch[n]) for n, _ in P) + " |")
open(f"{D}/steps_summary.md", "w").write("\n".join(out + detail) + "\n"); print("\n".join(out + detail))
