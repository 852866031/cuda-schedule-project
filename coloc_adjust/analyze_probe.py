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
# four panels with plain quantities: tenant speed, decode slowdown, their ratio, and the NVML counter
NICE = {"k=4": "4 micro-\nbatches", "k=8": "8 micro-\nbatches", "+act offload": "+ activation\noffload",
        "+12 layers": "+ 12 layers\nstreamed", "+20 layers": "+ 20 layers\nstreamed"}
lab = [NICE.get(l, l) for l in Tn.label]; x = np.arange(len(Tn))
ORANGE, BLUE, TEAL, GREY = "#eb6834", "#2a78d6", "#1b8a7a", "#8a93a0"
fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
def bars(a_, vals, color, fmt, title, unit, ymax, inside=False):
    a_.bar(x, vals, 0.62, color=color)
    for xi, v in zip(x, vals):
        if inside: a_.text(xi, v * 0.5, fmt.format(v), ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        else: a_.text(xi, v + ymax * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=10)
    a_.set_title(title, loc="left", fontsize=11.5, fontweight="bold"); a_.set_ylabel(unit); a_.set_ylim(0, ymax)
    a_.grid(True, axis="y", alpha=0.3); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
ktok = (Tn.ft_tok_s / 1e3).values
bars(ax[0, 0], ktok, ORANGE, "{:.1f}k", "(a) How fast the tenant trains", "thousand tokens per second", 5.2)
ax[0, 0].axhline(4.0, color="k", ls=":", lw=1.2); ax[0, 0].text(len(x) - 0.55, 4.45, "speed cap", ha="right", va="bottom", fontsize=9)
bars(ax[0, 1], Tn.slow_pct.values, BLUE, "+{:.1f} %", "(b) How much slower decode gets", "% slower than engine alone", 25)
bars(ax[1, 0], Tn.pct_per_k.values, TEAL, "{:.2f} %", "(c) Slowdown per 1,000 tenant tokens/s  =  (b) / (a)", "% slower", 11)
um = Tn.get("util_mem", pd.Series([0] * len(Tn))).values; um0 = T[T["shape"] < 0].get("util_mem", pd.Series([np.nan])).iloc[0]
bars(ax[1, 1], um, "#6f7885", "{:.0f} %", "(d) NVML counter: memory controller busy", "% of time", 100, inside=True)
ax[1, 1].axhline(um0, color="k", ls=":", lw=1.2); ax[1, 1].text(len(x) - 0.55, um0 + 8, f"engine alone, {um0:.0f} %", ha="right", va="bottom", fontsize=9)
for a_ in ax[1]: a_.set_xticks(x); a_.set_xticklabels(lab, fontsize=9.5)
fig.suptitle("Five tenant shapes under the same 4k tokens/s cap, engine at 1 req/s", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(P("probe.png"), dpi=140); print("figure:", P("probe.png"))
