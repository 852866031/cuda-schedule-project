#!/usr/bin/env python
"""Figure for knee_probe.py: decode step time, effective GPU-memory read rate and speed relative to the
full device against the SM count; the knee is the smallest SM count within 10 % of full-device speed.
    python plot_knee.py --csv results/knee_probe.csv --out results/knee_probe.png
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="results/knee_probe.csv"); ap.add_argument("--out", default="results/knee_probe.png")
ap.add_argument("--knee-frac", type=float, default=0.9)
args = ap.parse_args()
d = pd.read_csv(args.csv)
g = d.groupby(["kind", "B", "tokens", "sms"]).step_ms.agg(["median", "min", "max"]).reset_index()
gb = d.groupby(["kind", "B", "sms"]).gb_per_step.first().reset_index()
g = g.merge(gb, on=["kind", "B", "sms"])
FULL = g.sms.max()
colors = {1: "#1f77b4", 4: "#2ca02c", 8: "#ff7f0e", 14: "#d62728"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
knees = {}
for B, c in colors.items():
    s = g[(g.kind == "decode") & (g.B == B)].sort_values("sms")
    if s.empty: continue
    lab = f"decode, {B} running request{'s' if B > 1 else ''}"
    ax[0].plot(s.sms, s["median"], "o-", color=c, ms=4, label=lab); ax[0].fill_between(s.sms, s["min"], s["max"], color=c, alpha=0.2)
    ax[1].plot(s.sms, s.gb_per_step / s["median"] * 1e3, "o-", color=c, ms=4, label=lab)
    full = s[s.sms == FULL]["median"].iloc[0]; rel = full / s["median"] * 100
    ax[2].plot(s.sms, rel, "o-", color=c, ms=4, label=lab)
    k = s.sms[rel.values >= args.knee_frac * 100].min(); knees[B] = int(k)
    ax[2].plot([k], [rel.values[list(s.sms).index(k)]], "o", ms=11, mfc="none", mec=c, mew=1.6)
p = g[g.kind == "prefill"].sort_values("sms")
if not p.empty:
    full = p[p.sms == FULL]["median"].iloc[0]
    ax[2].plot(p.sms, full / p["median"] * 100, "s--", color="#7f7f7f", ms=4, label=f"prefill chunk, {int(p.tokens.iloc[0])} tokens")
ax[2].plot([0, FULL], [0, 100], ":", color="k", lw=1, label="proportional to SMs")
ax[2].plot([], [], "o", ms=11, mfc="none", mec="k", mew=1.6, label=f"knee: first point above {args.knee_frac*100:.0f} %")
ax[2].axhline(args.knee_frac * 100, color="k", lw=0.6, alpha=0.5)
ax[0].set_yscale("log"); ax[0].set_yticks([12, 15, 20, 30, 50, 80, 150]); ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax[0].minorticks_off()
ax[0].set_ylabel("time per decode step (ms)"); ax[0].set_title("(a) decode step time")
ax[1].set_ylabel("weights + KV read per second (GB/s)"); ax[1].set_title("(b) effective GPU-memory read rate"); ax[1].set_ylim(0, None)
ax[2].set_ylabel("speed relative to the full device (%)"); ax[2].set_title("(c) speed relative to the full device"); ax[2].set_ylim(0, 108)
for a in ax:
    a.set_xlabel(f"SMs given to the workload (of {FULL})"); a.grid(True, alpha=0.3); a.set_xlim(0, FULL + 4)
h, l = ax[2].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(args.out, dpi=150, bbox_inches="tight")
print("figure:", args.out); print("knee (first SM count >= %.0f %% of full-device speed):" % (args.knee_frac * 100), knees)
t = g[g.kind == "decode"].pivot(index="sms", columns="B", values="median").round(2); print(t.to_string())
