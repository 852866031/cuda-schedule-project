#!/usr/bin/env python
"""The direct test of the bandwidth claim: per phase and scheme, NVML's memory-controller busy
fraction against the tenant's throughput, with the decode slowdown as the point colour.
  plot_dram.py --dir results --out report/figures/dram_busy.png
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_names import ARMS, BASE  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results"); ap.add_argument("--out", default="report/figures/dram_busy.png")
args = ap.parse_args()
P = {}
for a, *_ in ARMS:
    f = os.path.join(args.dir, f"{a}_phases.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        if "dram_busy" in d and d.dram_busy.notna().any(): P[a] = d
base = P[BASE].set_index("t_from")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, label, color, marker in ARMS:
    if a not in P: continue
    d = P[a].set_index("t_from"); common = d.index.intersection(base.index)
    x = d.loc[common, "ft_tok_s"].fillna(0).values / 1e3 if "ft_tok_s" in d else np.zeros(len(common))
    ax[0].scatter(x, d.loc[common, "dram_busy"].values, color=color, marker=marker, s=50, label=label, zorder=3)
    slow = (d.loc[common, "tpot50"].values / base.loc[common, "tpot50"].values - 1) * 100
    ax[1].scatter(d.loc[common, "dram_busy"].values, slow, color=color, marker=marker, s=50, label=label, zorder=3)
ax[0].set_xlabel("tenant throughput in the phase (k tok/s)"); ax[0].set_ylabel("GPU memory-controller busy (%, NVML, phase mean)")
ax[0].set_title("(a) memory-bandwidth use vs tenant throughput"); ax[0].set_ylim(0, 100)
ax[1].set_xlabel("GPU memory-controller busy (%)"); ax[1].set_ylabel("decode TPOT p50 vs baseline (%)"); ax[1].set_title("(b) decode slowdown vs memory-bandwidth use")
ax[1].axhline(0, color="k", lw=0.6)
for a_ in ax: a_.grid(True, alpha=0.3)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(args.out, dpi=150, bbox_inches="tight"); print("figure:", args.out)
