#!/usr/bin/env python
"""Cross-arm figures, one message each.

  arms.png          decode TPOT, scheduler queue wait and fine-tune throughput against the
                    arrival rate (phases with the same rate are pooled: marker = mean,
                    error bar = min..max across the repeats).
  arms_tradeoff.png decode slowdown vs fine-tune throughput, one point per scheme and phase:
                    the cost of colocation scales with how fast the tenant runs, whatever
                    the compute-sharing mechanism.

  plot_arms.py --dir results --out results/arms
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from arm_names import ARMS, PF_ARMS, BASE, PF_BASE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results"); ap.add_argument("--out", default="results/arms")
ap.add_argument("--pf", action="store_true", help="the prefill-tenant schemes instead of the training-tenant ones")
args = ap.parse_args()
if args.pf: ARMS, BASE = PF_ARMS, PF_BASE
P = {}
for a, *_ in ARMS:
    f = os.path.join(args.dir, f"{a}_phases.csv")
    if os.path.exists(f):
        P[a] = pd.read_csv(f)
arms = [t for t in ARMS if t[0] in P]
rates = sorted(P[BASE].rps.unique())


def pooled(d, key):
    g = d.groupby("rps")[key]
    m, lo, hi = g.mean(), g.min(), g.max()
    m, lo, hi = (s.reindex(rates) for s in (m, lo, hi))
    return m.values, np.vstack([m.values - lo.values, hi.values - m.values])


# ---- figure 1: metrics vs arrival rate ------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for a, label, color, marker in arms:
    d = P[a]
    for k, (key, scale) in enumerate((("tpot50", 1.0), ("queue_ms", 1.0), ("ft_tok_s", 1e-3))):
        if key not in d:
            continue
        m, err = pooled(d, key)
        ax[k].errorbar(rates, m * scale, yerr=err * scale, color=color, marker=marker, ms=7,
                       lw=1.8, capsize=3, label=label)
ax[0].set_ylabel("Decode TPOT p50 (ms)"); ax[0].set_title("(a) per-token latency")
ax[1].set_ylabel("Scheduler queue wait (ms, log)"); ax[1].set_yscale("log"); ax[1].set_title("(b) queueing")
ax[2].set_ylabel("Prefill throughput (k prompt tok/s)" if args.pf else "Fine-tune throughput (k tok/s)"); ax[2].set_title("(c) tenant progress")
for a_ in ax:
    a_.set_xlabel("Arrival rate (req/s)"); a_.set_xticks(rates); a_.grid(True, alpha=0.3)
ax[2].set_ylim(bottom=0)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight"); print("figure:", args.out + ".png")

# ---- figure 2: slowdown vs tenant throughput ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.6))
base = P[BASE].set_index("t_from").tpot50
xs, ys = [], []
for a, label, color, marker in arms:
    if a == BASE or "ft_tok_s" not in P[a]:
        continue
    d = P[a].set_index("t_from")
    common = d.index.intersection(base.index)
    x = d.loc[common, "ft_tok_s"].fillna(0).values / 1e3
    y = (d.loc[common, "tpot50"].values / base.loc[common].values - 1) * 100
    if a == "static_split":  # its slowdown comes from KV starvation, not from the tenant's speed
        ax.scatter(x, y, facecolors="none", edgecolors=color, marker=marker, s=55,
                   label=label + " (not fitted)", zorder=3)
        continue
    ax.scatter(x, y, color=color, marker=marker, s=55, label=label, zorder=3)
    xs += list(x); ys += list(y)
xs, ys = np.array(xs), np.array(ys)
k = (xs * ys).sum() / (xs * xs).sum()  # fit through the origin: no tenant, no cost
xx = np.linspace(0, xs.max() * 1.05, 10)
ax.plot(xx, k * xx, "k--", lw=1, alpha=0.6, label=f"adaptive schemes, fit through origin: +{k:.1f} % per 1k FT tok/s")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(("Prefill" if args.pf else "Fine-tune") + " throughput in the phase (k tok/s)")
ax.set_ylabel("Decode TPOT p50 vs baseline (%)")
ax.set_title("Decode slowdown vs tenant progress, every scheme and phase")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(args.out + "_tradeoff.png", dpi=150); print("figure:", args.out + "_tradeoff.png")
