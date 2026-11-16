#!/usr/bin/env python3
"""Overnight plan 3, R5: every tenant shape under the same byte allowance (run_probe2.sh).
  python analyze_probe2.py [--tag probe2] [--dir results_o3]
Decode token gaps of each shape segment against the three no-tenant windows of the same run; the tenant's
tokens/s, its share of time computing, and its GB/s while computing (tokens/s while computing x MB per token
of that shape from R2). Writes <dir>/<tag>_summary.md."""
import argparse, json, os
import numpy as np, pandas as pd
from pacing import Bpt
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="probe2")
ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_o3"))
args = ap.parse_args(); D, T = args.dir, args.tag
bpt = Bpt(f"{D}/bytes_per_token.json")
NAME = {2: "4 micro-batches", 3: "8 micro-batches", 4: "8 mb + activations in host RAM", 5: "+ 12 layers in host RAM", 6: "+ 20 layers in host RAM"}
t0 = float(open(f"{D}/{T}_t0.txt").read()); segs = [s for s in json.load(open(f"{D}/{T}_segments.json")) if s["shape"] != -9]
gaps = []; st, en = [], []
for l in open(f"{D}/{T}_load.jsonl"):
    r = json.loads(l)
    if r.get("status") != 200 or len(r.get("tok_t", [])) < 3: continue
    t = np.array(r["tok_t"]) - t0; st.append(t[0]); en.append(t[-1])
    if r["rid"].startswith("w"): continue
    gaps += [((a + b) / 2, (b - a) * 1e3, a, b) for a, b in zip(t[:-1], t[1:])]
G = pd.DataFrame(gaps, columns=["t", "ms", "a", "b"]); st, en = np.sort(st), np.sort(en)
G["n"] = np.searchsorted(st, G.t.values, side="right") - np.searchsorted(en, G.t.values, side="right")
nt = pd.concat([G[(G.t >= s["t_from"] + 5) & (G.t < s["t_to"])] for s in segs if s["shape"] == -1])
PN = nt.groupby("n").ms.mean(); PN = PN[nt.groupby("n").ms.size() >= 100]; base = nt.ms.mean()
def same_n(w):
    w = w[w.n.isin(PN.index)]; return np.nan if len(w) <= 30 else (w.ms.values / PN.loc[w.n].values).mean() - 1
pct = lambda x: "–" if not np.isfinite(x) else f"{x * 100:+.1f} %"
md = [f"# R5: every tenant shape under the same byte allowance ({T}), real engine at 1 req/s\n",
      f"No-tenant windows: " + ", ".join(f"{G[(G.t >= s['t_from'] + 5) & (G.t < s['t_to'])].ms.mean():.2f} ms" for s in segs if s["shape"] == -1) + f" (mean token gap; all three together {base:.2f} ms, p90 {np.percentile(nt.ms, 90):.2f}).\n",
      "| shape | pass | tenant tok/s | while computing | share computing | GB/s while computing | GB/s on average | decode gap mean | p90 | while tenant computes | while it does not | mean, same number of running requests | per 1 k tok/s |", "|" + "---|" * 13]
rows = []
for s in segs:
    if s["shape"] < 0: continue
    p = f"{D}/{T}_ft{s['shape']}_p{s['pass']}.csv"
    if not os.path.exists(p): continue
    f = pd.read_csv(p); f = f[pd.to_numeric(f.dt_ms, errors="coerce").notna() & pd.to_numeric(f.tok_s, errors="coerce").notna()].copy()
    f["dt"] = pd.to_numeric(f.dt_ms) / 1e3; f["sl"] = pd.to_numeric(f.sleep_ms, errors="coerce").fillna(0) / 1e3; f["te"] = f.ts - t0; f["ts0"] = f.te - f.dt
    f = f.iloc[2:]                                           # the first steps include warm-up
    if len(f) < 3: continue
    a, b = f.ts0.iloc[0], f.te.iloc[-1]; w = G[(G.t >= a) & (G.t < b)].copy()
    tok = (pd.to_numeric(f.tok_s) * f.dt).sum(); run_s = (f.dt - f.sl).sum(); span = b - a
    iv = np.array(sorted(zip(f.ts0.values, (f.te - f.sl).values))); s_, e_ = iv[:, 0], iv[:, 1]; cum = np.concatenate([[0], np.cumsum(e_ - s_)])
    def cov(x):
        i = np.searchsorted(s_, x, side="right"); c = cum[i].copy(); j = i - 1; ok = j >= 0; c[ok] -= np.maximum(0, e_[j[ok]] - x[ok]); return c
    rf = (cov(w.b.values) - cov(w.a.values)) / np.maximum(1e-9, (w.b - w.a).values)
    run, idle = w[rf >= 0.9], w[rf <= 0.1]; idle = idle if len(idle) >= max(200, 0.05 * len(w)) else idle.iloc[:0]; run = run if len(run) >= max(200, 0.05 * len(w)) else run.iloc[:0]
    m = w.ms.mean() / base - 1; mb = bpt.mb(s["shape"]); r = dict(shape=s["shape"], p=s["pass"], tok=tok / span, tok_run=tok / run_s, share=run_s / span, gb_run=tok / run_s * mb / 1e3, gb=tok / span * mb / 1e3,
                                                                  mean=m, p90=np.percentile(w.ms, 90) / np.percentile(nt.ms, 90) - 1, run=run.ms.mean() / base - 1 if len(run) else np.nan,
                                                                  idle=idle.ms.mean() / base - 1 if len(idle) else np.nan, same=same_n(w))
    rows.append(r)
    md.append(f"| {NAME[r['shape']]} | {r['p']} | {r['tok']:,.0f} | {r['tok_run']:,.0f} | {r['share'] * 100:.0f} % | {r['gb_run']:.0f} | {r['gb']:.0f} | {pct(r['mean'])} | {pct(r['p90'])} | {pct(r['run'])} | {pct(r['idle'])} | {pct(r['same'])} | {r['mean'] * 100 / r['tok'] * 1e3:.2f} % |")
md += ["\n## Both passes averaged\n", "| shape | MB per token (R2) | tenant tok/s | share computing | GB/s while computing | decode gap mean | while tenant computes | per 1 k tok/s | per 100 GB/s on average |", "|" + "---|" * 9]
for sh in sorted({r["shape"] for r in rows}):
    x = [r for r in rows if r["shape"] == sh]; av = lambda k: float(np.nanmean([r[k] for r in x]))
    md.append(f"| {NAME[sh]} | {bpt.mb(sh):.1f} | {av('tok'):,.0f} | {av('share') * 100:.0f} % | {av('gb_run'):.0f} | {pct(av('mean'))} | {pct(av('run'))} | {av('mean') * 100 / av('tok') * 1e3:.2f} % | {av('mean') * 100 / av('gb') * 100:.1f} % |")
open(f"{D}/{T}_summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
