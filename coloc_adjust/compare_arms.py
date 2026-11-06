#!/usr/bin/env python
"""Side-by-side phase tables and the arms figure.
  compare_arms.py --dir results --arms decode_alone static_split adjust adjust_nogate --out results/arms
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results"); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
P = {a: pd.read_csv(os.path.join(args.dir, f"{a}_phases.csv")) for a in args.arms if os.path.exists(os.path.join(args.dir, f"{a}_phases.csv"))}
arms = list(P)
metrics = [("tpot50", "TPOT p50 ms", "{:.1f}"), ("tpot95", "TPOT p95 ms", "{:.1f}"), ("queue_ms", "scheduler queue wait ms", "{:.0f}"),
           ("tok_s_done", "delivered tok/s", "{:.0f}"), ("failed", "failed", "{:.0f}"), ("preempt", "preemptions", "{:.0f}"),
           ("ft_tok_s", "FT tok/s", "{:.0f}"), ("ft_cfg_mode", "FT cfg (mode)", "{:.0f}"), ("ft_reserved_gib", "FT GiB", "{:.2f}"),
           ("limit_gib", "KV limit GiB", "{:.2f}")]
lines = ["# Arms side by side (per RPS phase, first 10 s of each phase excluded)", ""]
ref = P[arms[0]]
for key, label, fmt in metrics:
    if not any(key in P[a] for a in arms): continue
    lines += [f"## {label}", "", "| phase (RPS) | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for i, r in ref.iterrows():
        cells = []
        for a in arms:
            v = P[a][key].iloc[i] if key in P[a] and i < len(P[a]) else np.nan
            cells.append("" if pd.isna(v) else fmt.format(v))
        lines.append(f"| {r.t_from:.0f}–{r.t_to:.0f} s ({r.rps:g}) | " + " | ".join(cells) + " |")
    lines.append("")
open(args.out + "_table.md", "w").write("\n".join(lines)); print("\n".join(lines))
# dense every-arm figure (the report's figures come from plot_arms.py)
fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
x = np.arange(len(ref)); wdt = 0.8 / len(arms); labels = [f"{r.rps:g} rps\n{r.t_from:.0f}s" for _, r in ref.iterrows()]
for j, a in enumerate(arms):
    d = P[a]; off = (j - (len(arms) - 1) / 2) * wdt
    ax[0].bar(x + off, d.tpot50, wdt, label=a); ax[1].bar(x + off, d.queue_ms.clip(lower=1), wdt, label=a)
    ax[2].bar(x + off, d.tok_s_done, wdt, label=a)
    ax[3].bar(x + off, d.ft_tok_s.fillna(0) if "ft_tok_s" in d else np.zeros(len(d)), wdt, label=a)
ax[2].plot(x, ref.offered_tok_s, "k_", ms=25, mew=2, label="offered")
ax[0].set_ylabel("TPOT p50 ms"); ax[1].set_ylabel("queue wait ms (log)"); ax[1].set_yscale("log"); ax[2].set_ylabel("delivered tok/s"); ax[3].set_ylabel("FT tok/s")
for a_ in ax: a_.grid(True, alpha=0.3)
ax[0].legend(fontsize=8); ax[2].legend(fontsize=8); ax[3].set_xticks(x); ax[3].set_xticklabels(labels, fontsize=8)
fig.suptitle("Colocation arms per RPS phase"); fig.tight_layout(); fig.savefig(args.out + "_all.png", dpi=130); print("figure:", args.out + "_all.png")
