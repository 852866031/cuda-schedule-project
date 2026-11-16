#!/usr/bin/env python
"""Figures for reports/1 (live vs static) and reports/4 (prefill tenant per phase), also used by .lavish/project-recap.html (run from the repo root):
    python coloc_adjust/plot_recap_figs.py   -> coloc_adjust/report/figures/live_vs_static.png, coloc_adjust/report/figures/pf_tenant_perf.png
Live shrink (kvcached limit changed in place) against a restart per budget, same request timeline at every budget.
Data: kvcached_validation/results/{livereplay,static}_offload_zipf_windows.csv (the report's fig_live_vs_static_offload.png,
redrawn with the legend outside the panels and one quantity per panel)."""
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

INK, INK2, GRID, BLUE, GREY = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "legend.frameon": False})
R = "kvcached_validation/results/"
def load(f):
    rows = sorted(csv.DictReader(open(R + f)), key=lambda r: -float(r["budget_gib"]))
    return [float(r["budget_gib"]) for r in rows], rows
xl, live = load("livereplay_offload_zipf_windows.csv"); xs, stat = load("static_offload_zipf_windows.csv")
panels = [("ttft50", "Time to first token, p50 (ms)", True), ("queue_mean_ms", "Scheduler queue wait, mean (ms)", True), ("tpot50", "Time per output token, p50 (ms)", False)]
fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.2))
for a, (k, title, log) in zip(ax, panels):
    a.plot(xl, [float(r[k]) for r in live], "o-", color=BLUE, lw=2.2, ms=8, mec="white", mew=1.5, label="live: limit changed in place")
    a.plot(xs, [float(r[k]) for r in stat], "s--", color=GREY, lw=1.3, ms=10, mfc="none", mec=GREY, mew=1.5, label="static: engine restarted at each budget")
    if log:
        a.set_yscale("log"); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}")); a.yaxis.set_minor_formatter(NullFormatter())
    else: a.set_ylim(11, 14)
    a.set_title(title); a.set_xticks([13, 10, 8, 6, 5, 4, 3, 2]); a.invert_xaxis(); a.set_xlabel("GPU KV budget (GiB), tighter to the right")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("coloc_adjust/report/figures/live_vs_static.png", dpi=140); print("coloc_adjust/report/figures/live_vs_static.png")

# ---- the prefill tenant's own performance, phase by phase (coloc_adjust/results/pf_*_phases.csv)
import numpy as np
AQUA, ORANGE = "#1baf7a", "#eb6834"
C = "coloc_adjust/results/"
arms = [("pf_mps", "shared SMs (MPS)", BLUE), ("pf_mps20", "MPS, tenant capped at 20 %", ORANGE), ("pf_split", "disjoint SMs (128 / 42)", AQUA)]
data = {a: list(csv.DictReader(open(C + a + "_phases.csv"))) for a, _, _ in arms}
ph = data["pf_mps"]; x = np.arange(len(ph)); w = 0.27
fig, ax = plt.subplots(3, 1, figsize=(12.6, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 0.8, 1]))
for i, (a, lab, c) in enumerate(arms):
    r = data[a]; off = (i - 1) * w
    ax[0].bar(x + off, [float(v["ft_tok_s"]) / 1000 for v in r], w * 0.92, color=c, label=lab)
    ax[1].bar(x + off, [(1 - float(v["pf_asleep_frac"])) * 100 for v in r], w * 0.92, color=c)
    ax[2].plot(x + off, [max(float(v["pf_lat50"]) / 1000, 0.05) for v in r], "o", color=c, ms=8, mec="white", mew=1.2)
ax[0].axhline(3.1, color=GREY, lw=1.2, ls=":"); ax[0].annotate("offered to the tenant: 3.1k prompt tok/s", (6.9, 3.1), xytext=(0, 5), textcoords="offset points", ha="center", color=INK2, fontsize=10)
ax[0].set_title("Prompt tokens per second delivered by the prefill tenant (k tok/s)"); ax[0].set_ylim(0, 15)
ax[1].set_title("Share of the phase the tenant engine is awake (%)"); ax[1].set_ylim(0, 105); ax[1].set_yticks([0, 50, 100])
ax[2].set_title("Prompt latency, p50 (s, log scale)"); ax[2].set_yscale("log"); ax[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax[2].yaxis.set_minor_formatter(NullFormatter())
for a_ in ax:
    a_.grid(axis="x", visible=False)
    for k in range(len(ph)):
        if float(ph[k]["rps"]) >= 2: a_.axvspan(k - 0.5, k + 0.5, color="#f1f0ec", zorder=0)
ax[2].set_xticks(x); ax[2].set_xticklabels([f"{float(v['rps']):g}" for v in ph]); ax[2].set_xlim(-0.5, len(ph) - 0.5)
ax[2].set_xlabel("decode load in each phase (req/s), phases in run order; shaded = 2 req/s or more")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig("coloc_adjust/report/figures/pf_tenant_perf.png", dpi=140); print("coloc_adjust/report/figures/pf_tenant_perf.png")
