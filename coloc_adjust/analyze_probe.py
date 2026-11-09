#!/usr/bin/env python
"""P1.2 shape probe analysis: per segment, decode TPOT vs the no-tenant segment and the tenant's
actual tokens/s -> slowdown per 1k tok/s per shape. Writes <tag>_probe.md and <tag>_probe.png."""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_names import CFG_LABELS

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="probe"); ap.add_argument("--dir", default="results")
args = ap.parse_args(); P = lambda s: os.path.join(args.dir, f"{args.tag}_{s}")
t0 = float(open(P("t0.txt")).read())
segs = json.load(open(P("segments.json")))
reqs = [json.loads(l) for l in open(P("load.jsonl")) if not json.loads(l)["rid"].startswith("w")]
R = pd.DataFrame([dict(t=r["t_send"] - t0, tpot=((r["t_done"] - r["tok_t"][0]) / max(1, r["n_out"] - 1)) * 1e3)
                  for r in reqs if r["status"] == 200 and r["tok_t"]])
samp = pd.read_csv(P("sampler.csv")); samp["rel"] = samp.t - t0
rows = []
for s in segs:
    a, b = s["t_from"] + 10, s["t_to"]
    w = R[(R.t >= a) & (R.t < b)]
    row = dict(shape=s["shape"], label=s.get("label") or CFG_LABELS[s["shape"]], tpot50=w.tpot.median(), n=len(w))
    if s["shape"] >= 0:
        f = pd.read_csv(P(f"ft{s['shape']}.csv")); f["rel"] = f.ts - t0
        f = f[(f.rel >= a) & (f.rel < b)]; ok = f[pd.to_numeric(f.tok_s, errors="coerce").notna()]
        row["ft_tok_s"] = (pd.to_numeric(ok.tok_s) * pd.to_numeric(ok.dt_ms) / 1e3).sum() / max(1e-9, b - a)
        row["ft_steps"] = len(ok)
    sw = samp[(samp.rel >= a) & (samp.rel < b)]
    for c in ("util_mem", "util_gpu", "power_w"):
        if c in sw: row[c] = pd.to_numeric(sw[c], errors="coerce").mean()
    rows.append(row)
T = pd.DataFrame(rows); base = T[T["shape"] < 0].tpot50.iloc[0]
T["slow_pct"] = (T.tpot50 / base - 1) * 100
T["pct_per_k"] = T.slow_pct / (T.ft_tok_s / 1e3)
T.to_csv(P("probe.csv"), index=False)
md = ["| shape | tenant tok/s | decode TPOT (ms) | slowdown | % per 1k tok/s | GPU-mem busy % | power W |", "|---|---|---|---|---|---|---|"]
for _, r in T.iterrows():
    md.append(f"| {r.label} | {'' if pd.isna(r.get('ft_tok_s')) else f'{r.ft_tok_s:.0f}'} | {r.tpot50:.1f} | {r.slow_pct:+.1f} % | "
              f"{'' if pd.isna(r.get('pct_per_k')) else f'{r.pct_per_k:.2f}'} | {r.get('util_mem', float('nan')):.0f} | {r.get('power_w', float('nan')):.0f} |")
open(P("probe.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
Tn = T[T["shape"] >= 0]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(Tn.label, Tn.pct_per_k, color="#2a9d8f"); ax[0].set_ylabel("decode slowdown per 1k tenant tok/s (%)"); ax[0].set_title("(a) cost per unit of tenant throughput, by shape")
ax[1].bar(Tn.label, Tn.get("util_mem", pd.Series([0] * len(Tn))), color="#6a4c93"); ax[1].axhline(T[T["shape"] < 0].get("util_mem", pd.Series([np.nan])).iloc[0], color="k", ls=":", label="no tenant")
ax[1].set_ylabel("GPU memory-controller busy (%, NVML)"); ax[1].set_title("(b) memory-bandwidth use, by shape"); ax[1].legend(fontsize=8)
for a_ in ax: a_.tick_params(axis="x", labelrotation=20); a_.grid(True, axis="y", alpha=0.3)
fig.suptitle(f"Shape probe at 1 req/s, tenant capped at the same tokens/s", fontsize=12); fig.tight_layout(); fig.savefig(P("probe.png"), dpi=140); print("figure:", P("probe.png"))
