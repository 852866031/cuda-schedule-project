#!/usr/bin/env python
"""Budget-sweep figure in the layout of branches/vram-limit/figures/split_lmcache_sweep.png:
x = GPU KV budget (GiB), tighter to the right; three stacked panels — TPOT (log), scheduler
queueing, output throughput with the offered-load line. One line per arm (live kvcached shrink
vs static restart), p50 solid, p95 dashed.
  plot_sweeps.py --out results/fig.png --title "..." --offered 256 \
      --arm "live (kvcached)" results/sweep_x_windows.csv --arm "static (restart)" results/static_x_windows.csv
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator

def plain(ax):
    """Plain numbers on a log axis instead of 10^k notation."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(fmt); ax.yaxis.set_minor_formatter(NullFormatter())

ap = argparse.ArgumentParser()
ap.add_argument("--arm", nargs=2, action="append", metavar=("LABEL", "CSV"), required=True)
ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
ap.add_argument("--offered", type=float, default=2 * 128, help="offered output tok/s")
ap.add_argument("--start-gib", type=float, default=13.0, help="budget to assign to the live run's 'start' window")
args = ap.parse_args()

def load(csv):
    df = pd.read_csv(csv)
    def gib(x):
        if "budget_gib" in df.columns and not pd.isna(x.get("budget_gib", np.nan)): return float(x["budget_gib"])
        lab = str(x["limit"]); return args.start_gib if lab == "start" else float(lab.rstrip("G"))
    df["gib"] = df.apply(gib, axis=1)
    return df.sort_values("gib", ascending=False)

arms = [(lab, load(csv)) for lab, csv in args.arm]
colors = ["#1f6fe5", "#555555", "#d9534f", "#2a9d8f"]
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for (lab, df), c in zip(arms, colors):
    ax[0].plot(df.gib, df.tpot50, "o-", color=c, label=f"{lab} TPOT p50")
    ax[0].plot(df.gib, df.tpot95, "^--", color=c, alpha=0.7, label=f"{lab} TPOT p95")
    q = df.queue_mean_ms if "queue_mean_ms" in df else pd.Series(np.nan, index=df.index)
    ax[1].plot(df.gib, q.clip(lower=0.1), "o-", color=c, label=f"{lab} scheduler queue wait (mean)")
    ax[1].plot(df.gib, df.waiting_mean.clip(lower=0.01) if "waiting_mean" in df else np.nan, "s:", color=c, alpha=0.7, label=f"{lab} mean # waiting")
    ax[2].plot(df.gib, df.tok_s_done if "tok_s_done" in df else df.tok_s, "o-", color=c, label=f"{lab} delivered output tok/s")
    bad = df[df.failed > 0]
    if len(bad): ax[2].plot(bad.gib, bad.tok_s_done if "tok_s_done" in bad else bad.tok_s, "o", mfc="none", mec="red", ms=14, mew=2, label=f"{lab}: failed requests")
    pre = df[df.preempt > 0]
    if len(pre): ax[0].plot(pre.gib, pre.tpot50, "x", color="red", ms=10, mew=2, label=f"{lab}: preemptions")
ax[0].set_ylabel("TPOT (ms)"); ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8, loc="upper left")
ax[1].set_yscale("log"); plain(ax[1]); ax[1].set_ylabel("queueing: wait (ms) / # waiting"); ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=8, loc="upper left")
ax[2].axhline(args.offered, color="k", ls="--", lw=1); ax[2].text(ax[2].get_xlim()[0] if False else max(d.gib.max() for _, d in arms), args.offered * 0.93, f"offered load: {args.offered:.0f} tok/s", fontsize=9)
ax[2].set_ylim(0, args.offered * 1.2); ax[2].set_ylabel("output tok/s"); ax[2].grid(True, alpha=0.3); ax[2].legend(fontsize=8, loc="lower left")
ax[2].set_xlabel("GPU KV budget (GiB)  —  tighter to the right"); ax[2].invert_xaxis()
fig.suptitle(args.title); fig.tight_layout(); fig.savefig(args.out, dpi=130); print("figure:", args.out)
