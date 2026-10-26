#!/usr/bin/env python3
"""KV-slot study: where a request's time on the decode engine goes, per cell (from the
per-request timelines + step/load traces):

  queue  (arrival at the decode scheduler -> admitted)
  load   (admitted -> first decode token: the synchronous KV copy + suffix prefill)
  decode steps that overlap no KV load   (the request's own token generation)
  decode steps stalled by someone's load (the batch-wide stall it sat through)

Stacked bars, one per (budget, rate), means over the cell's 300 requests.

    .venv/bin/python scripts/plots/plot_kv_split.py   -> figures/kv_time_split.png
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
TL = REPO / "output" / "kv_vram" / "timeline"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
BUDGETS = [24, 26, 28]
METHOD = os.environ.get("KV_METHOD", "base")
QPS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
C = {"queue": "#9a9a95", "load": "#eb6834", "steps": "#2a78d6", "stall": "#f4b48f"}


def f(x):
    return float(x) if x not in ("", None) else np.nan


def split(name):
    rq = [r for r in csv.DictReader(open(TL / f"{name}_req.csv")) if r.get("decode")]
    st = list(csv.DictReader(open(TL / f"{name}_steps.csv")))
    ld = list(csv.DictReader(open(TL / f"{name}_loads.csv")))
    t = np.array([f(s["t"]) for s in st])
    dt = np.r_[np.nan, np.diff(t)]
    L0 = np.array([f(l["t"]) for l in ld])
    L1 = L0 + np.array([f(l["dt"]) for l in ld])
    ov = np.zeros(len(t), bool)
    for i in range(1, len(t)):
        ov[i] = ((L0 < t[i]) & (L1 > t[i - 1])).any()
    g = dt < 0.6
    out = {"queue": [], "load": [], "steps": [], "stall": []}
    for r in rq:
        a, b = f(r["t_first"]), f(r["t_fin"])
        m = g & (t > a) & (t <= b)
        out["queue"].append(f(r["wait"]))
        out["load"].append(f(r["load"]))
        out["steps"].append(dt[m & ~ov].sum())
        out["stall"].append(dt[m & ov].sum())
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, b in zip(axes, BUDGETS):
        xs, vals = [], []
        for q in QPS:
            p = TL / f"kv_{METHOD}_b{b}_q{q:g}_tr_r2_req.csv"      # a rerun replaces a failed run
            if not p.exists():
                p = TL / f"kv_{METHOD}_b{b}_q{q:g}_tr_req.csv"
                if not p.exists():
                    continue
            xs.append(q)
            vals.append(split(p.name[:-8]))
        bottom = np.zeros(len(xs))
        for k, lab in (("steps", "decode steps (own tokens)"), ("stall", "stalled by others' KV loads"),
                       ("load", "own KV load (admitted → token 1)"), ("queue", "queue before a slot")):
            v = np.array([d[k] for d in vals])
            ax.bar(range(len(xs)), v, bottom=bottom, color=C[k], width=0.7, label=lab, edgecolor="white")
            bottom += v
        for i, d in enumerate(vals):
            tot = sum(d.values())
            ax.text(i, min(tot, 14.5) + 0.2, f"{tot:.1f} s", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{q:g}" for q in xs])
        ax.set_xlabel("request rate (req/s)")
        ax.set_title(f"{b} GiB")
        ax.set_ylim(0, 15)
        ax.grid(axis="y", color="#ececea", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("time per request (s, clipped at 15)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Where a request's time goes on the decode engine (means over 300 requests per cell)",
                 y=1.08)
    fig.savefig(REPO / "figures" / "kv_time_split.png", dpi=150, bbox_inches="tight")
    print("-> figures/kv_time_split.png")


if __name__ == "__main__":
    main()
