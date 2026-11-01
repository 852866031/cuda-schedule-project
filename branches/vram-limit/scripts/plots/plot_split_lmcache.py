#!/usr/bin/env python3
"""Figures for the disaggregated (LMCache) decode-VRAM sweep, house style.

Reads the forwarding-router sweep summaries (the honest client TTFT: prefill emits
token #1) plus the colocated lmcache sweep as reference.

    .venv/bin/python scripts/plots/plot_split_lmcache.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"
BLUE, RED, GREY = "#1f6feb", "#c1440e", "#57606a"


def load(files, keep):
    rows = {}
    for name in files:
        f = OUT / name
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if keep(r):
                rows[r["name"]] = r
    return sorted(rows.values(), key=lambda r: float(r["budget_gib"]))


split = load(["summary_split_lmc_ng.csv", "summary_split_lmc_ng2.csv",
              "summary_split_lmc_ng3.csv", "summary_split_lmc_ng4.csv",
              "summary_split_lmc_ng6.csv"],
             lambda r: r["name"].endswith("_fwd_ng") and float(r["budget_gib"]) != 24)
coloc = load(["summary_lmcache.csv"],
             lambda r: r["name"].startswith("zipf_") and r["arm"] == "offload")

def f(rows, k):
    return [float(r[k]) for r in rows]

# ---------------------------------------------------------------- fig 1: the sweep
fig, (ax, axt) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [1.9, 1], "hspace": 0.12})
x, xc = f(split, "budget_gib"), f(coloc, "budget_gib")

ax.plot(xc, f(coloc, "ttft_p50_ms"), color=GREY, marker="o", lw=1.6, ls="--",
        label="colocated lmcache p50 (reference)")
ax.plot(x, f(split, "ttft_p50_ms"), color=BLUE, marker="o", lw=2, label="split p50")
ax.plot(x, f(split, "ttft_p95_ms"), color=BLUE, marker="^", ls="--", lw=1.2,
        alpha=0.7, label="split p95")
ax.fill_between(x, f(split, "ttft_p50_ms"), f(split, "ttft_p95_ms"),
                color=BLUE, alpha=0.10)
ax.axhline(530, color=GREY, lw=1, ls=":")
ax.annotate("full-prefix recompute ≈ 530 ms", xy=(30, 530), xytext=(4, 4),
            textcoords="offset points", fontsize=8, color=GREY)
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_ylabel("client TTFT (ms, log)")
ax.set_title("Decode-node VRAM sweep, prefill/decode split with a shared LMCache "
             "(router forwards prefill's token #1)", fontsize=11)
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8, loc="upper left")

axt.plot(x, f(split, "output_tok_per_s"), color=BLUE, marker="o", lw=2)
axt.axhline(256.0, color=GREY, lw=1, ls="--")
axt.annotate("offered load: 2 req/s × 128 tok", xy=(30, 256), xytext=(4, -13),
             textcoords="offset points", fontsize=8, color=GREY)
axt.set_ylim(0, 300)
axt.set_xlabel("decode GPU VRAM budget (GiB)   —   tighter to the right")
axt.set_ylabel("output tok/s")
axt.grid(alpha=0.25)
fig.savefig(FIGS / "split_lmcache_sweep.png", dpi=140, bbox_inches="tight")
print("wrote", FIGS / "split_lmcache_sweep.png")

# ------------------------------------------------------- fig 2: what the GPUs did
fig2, ax2 = plt.subplots(figsize=(9.5, 4.2))
for key, color, ls, label in (
        ("decode_sm_active_mean", BLUE, "-", "decode SM active"),
        ("decode_dram_active_mean", RED, "-", "decode DRAM active"),
        ("prefill_sm_active_mean", GREY, "-", "prefill SM active"),
        ("prefill_dram_active_mean", GREY, "--", "prefill DRAM active")):
    try:
        ax2.plot(x, f(split, key), color=color, ls=ls, marker="o", lw=2, label=label)
    except (KeyError, ValueError):
        pass
ax2.invert_xaxis()
ax2.set_ylim(0, 1)
ax2.set_xlabel("decode GPU VRAM budget (GiB)   —   tighter to the right")
ax2.set_ylabel("mean fraction of time active")
ax2.set_title("DCGM telemetry across the sweep: one GPU idles, the other saturates "
              "its memory system", fontsize=11)
ax2.grid(alpha=0.25)
ax2.legend(fontsize=8)
fig2.savefig(FIGS / "split_lmcache_gpu.png", dpi=140, bbox_inches="tight")
print("wrote", FIGS / "split_lmcache_gpu.png")
