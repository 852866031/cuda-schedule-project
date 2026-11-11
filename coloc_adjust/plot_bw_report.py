#!/usr/bin/env python
"""Data figures of report/REPORT_bandwidth.md (one message per figure).
    python plot_bw_report.py            -> report/figures/bw_*.png
  bw_knee.png        decode alone: read rate against the SMs it gets (results/knee_probe.csv)
  bw_slowdown.png    decode slowdown against the reader's GB/s, one line per reader SM count (results/aggr_corun.csv)
  bw_tenant.png      the fine-tune tenant on the same axes, against the reader's lines (results/knee_bw2.csv)
  bw_modes.png       what kind of aggressor activity costs decode (results/aggr_modes.csv)
  bw_tenant_alone.png  the fine-tune tenant alone: traffic against the SMs it gets (results/knee_bw2_summary.md)
Colours: one blue ramp for the SM count (ordered), blue / orange / aqua for unordered kinds; both sets were run
through the dataviz validator.  Text is never in a series colour."""
import csv, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report/figures/"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
RAMP = {8: "#86b6ef", 16: "#5598e7", 32: "#2a78d6", 64: "#1c5cab", 96: "#0d366b"}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7, "lines.markeredgecolor": "white", "lines.markeredgewidth": 1.5})

def med(rows): return st.median(float(r["step_ms"]) for r in rows)
def corun(path, series="corun"):
    """{(aggr SMs, B): [(aggr GB/s, slowdown %), ...]} for the paced points, against the idle-aggressor baseline of the same block"""
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["series"] == series: g[(int(r["aggr_sms"]), int(r["B"]), r["target_gbps"])].append(r)
    out = defaultdict(list)
    for (m, B, t), rs in g.items():
        if t in ("0", "max"): continue
        out[(m, B)].append((float(rs[0]["aggr_gbps"]), (med(rs) / med(g[(m, B, "0")]) - 1) * 100))
    return {k: sorted(v) for k, v in out.items()}

def fig_knee():
    g = defaultdict(list)
    for r in csv.DictReader(open("results/knee_probe.csv")):
        if r["kind"] == "decode" and r["B"] in ("1", "8"): g[(int(r["B"]), int(r["sms"]))].append(float(r["gb_per_step"]) / float(r["step_ms"]) * 1000)
    fig, a = plt.subplots(figsize=(6.6, 4.3))
    for B, c in ((1, BLUE), (8, ORANGE)):
        x = sorted(m for b, m in g if b == B); y = [st.median(g[(B, m)]) for m in x]
        a.plot(x, y, "o-", color=c, label=f"{B} running request{'s' if B > 1 else ''}")
        if B == 1: a.plot([0, 48], [0, y[0] / x[0] * 48], "--", color=GREY, lw=1.2, zorder=1); x1, y1 = x, y
    a.annotate("if SMs were the limit", (40, y1[0] / x1[0] * 40), xytext=(-8, 4), textcoords="offset points", ha="right", color=INK2, fontsize=10)
    a.axhline(1779, color=GREY, lw=1.2, ls=":"); a.annotate("nominal bandwidth of the card, 1.78 TB/s", (170, 1779), xytext=(0, 5), textcoords="offset points", ha="right", color=INK2, fontsize=10)
    a.annotate("flat from about 64 SMs:\nthe path to GPU memory is the limit", (112, 1335), xytext=(0, -42), textcoords="offset points", ha="center", color=INK, fontsize=10)
    a.set_xlim(0, 176); a.set_ylim(0, 1950); a.set_xticks([8, 32, 64, 96, 128, 170]); a.set_xlabel("SMs given to decode (of 170)"); a.set_ylabel("GPU-memory traffic of decode (GB/s)")
    a.set_title("Decode alone: more SMs stop helping"); a.legend(loc="lower right", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + "bw_knee.png", dpi=150); print(OUT + "bw_knee.png")

def fig_slowdown():
    d = corun("results/aggr_corun.csv"); fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for a, B in zip(ax, (1, 8)):
        for m, c in RAMP.items(): x, y = zip(*d[(m, B)]); a.plot(x, y, "o-", color=c, label=f"{m} SMs")
        a.set_title(f"Decode with {B} running request{'s' if B > 1 else ''}"); a.set_xlabel("GPU-memory traffic of the second job (GB/s)")
        a.set_xlim(0, 860); a.set_ylim(0, None)
    ax[0].set_ylabel("decode slowdown (%)"); ax[0].legend(title="second job runs on", loc="upper left", fontsize=10, title_fontsize=10)
    x, y = zip(*d[(96, 1)]); k = min(range(len(x)), key=lambda i: abs(x[i] - 100))
    ax[0].annotate(f"100 GB/s → {y[k]:.0f} % slower", (x[k], y[k]), xytext=(300, 3), textcoords="data", color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.tight_layout(); fig.savefig(OUT + "bw_slowdown.png", dpi=150); print(OUT + "bw_slowdown.png")

def fig_tenant():
    """the headline: decode alone against decode next to the real fine-tune tenant, each on its own SMs"""
    g = defaultdict(list)
    for r in csv.DictReader(open("results/knee_bw2.csv")): g[(r["series"], r["tenant_sms"], r["decode_sms"], r["B"])].append(r)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True); Ms = (8, 16, 32, 64, 96); w = 0.38
    for a, B in zip(ax, ("1", "8")):
        for i, m in enumerate(Ms):
            n = str(170 - m); al = st.mean(med(g[(s_, "", n, B)]) for s_ in ("alone", "alone_end")); co = med(g[("disjoint", str(m), n, B)])
            a.bar(i - w / 2 - 0.02, al, w, color=BLUE, label="decode alone" if i == 0 else None); a.bar(i + w / 2 + 0.02, co, w, color=ORANGE, label="decode next to the tenant" if i == 0 else None)
            a.annotate(f"+{(co / al - 1) * 100:.0f} %", (i + w / 2 + 0.02, co), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=10)
        a.set_xticks(range(len(Ms))); a.set_xticklabels([f"{m}" for m in Ms]); a.grid(axis="x", visible=False)
        a.set_xlabel("SMs given to the tenant (decode keeps the other 170 − x)"); a.set_title(f"Decode with {B} running request{'s' if B != '1' else ''}")
    ax[0].set_ylabel("time per decode step (ms)"); ax[0].set_ylim(0, 27); ax[0].legend(loc="upper left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + "bw_tenant.png", dpi=150); print(OUT + "bw_tenant.png")

def fig_modes():
    """E2.7: the same decode step next to five kinds of second job, B=1; byte kinds at 100 GB/s, no-byte kinds at a fixed call rate"""
    g = defaultdict(list)
    for r in csv.DictReader(open("results/aggr_modes.csv")):
        if r["series"] == "mode_corun" and r["B"] == "1": g[(r["aggr_sms"], r["kernel"], r["target_gbps"])].append(r)
    rows = [("gemv32", "100", "reads 100 GB/s", BLUE), ("write32", "100", "writes 100 GB/s", BLUE), ("copy32", "100", "copies 100 GB/s", BLUE),
            ("tiny", "20", "20,000 tiny kernels/s", AQUA), ("compute", "50%", "arithmetic only, half speed", AQUA)]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=True)
    for a, m in zip(ax, ("8", "96")):
        base = st.mean(med(g[(m, k, "0")]) for k in ("idle", "idle_end"))
        for i, (k, t, lab, c) in enumerate(rows):
            key = (m, k, t) if t != "50%" else sorted((kk for kk in g if kk[0] == m and kk[1] == k and kk[2] not in ("-1", "max")), key=lambda kk: float(kk[2]))[-1]
            v = (med(g[key]) / base - 1) * 100; a.barh(i, v, 0.62, color=c)
            a.annotate(f"+{v:.1f} %", (v, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
        a.set_yticks(range(len(rows))); a.set_yticklabels([r[2] for r in rows] if m == "8" else []); a.invert_yaxis(); a.grid(axis="y", visible=False)
        a.set_title(f"Second job on {m} SMs"); a.set_xlabel("decode slowdown (%)"); a.set_xlim(0, 9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="moves bytes to or from GPU memory"), Patch(color=AQUA, label="moves almost no bytes")], loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + "bw_modes.png", dpi=150); print(OUT + "bw_modes.png")

def fig_tenant_alone():
    """the fine-tune tenant alone on M SMs: GPU-memory traffic against M (table "Tenant alone" of results/knee_bw2_summary.md)"""
    rows, on = [], False
    for ln in open("results/knee_bw2_summary.md"):
        if ln.startswith("## "): on = ln.strip() == "## Tenant alone"
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if on and c and c[0].isdigit(): rows.append((int(c[0]), float(c[3])))
    rows.sort(); x = [m for m, _ in rows]; y = [v for _, v in rows]
    fig, a = plt.subplots(figsize=(6.6, 4.3))
    a.plot([0, x[-1]], [0, y[-1]], "--", color=GREY, lw=1.2, zorder=1)
    a.plot(x, y, "o-", color=ORANGE)
    a.annotate("if traffic followed the SM count", (110, y[-1] / x[-1] * 110), xytext=(8, -14), textcoords="offset points", color=INK2, fontsize=10)
    a.annotate(f"{x[0]} SMs: {y[0]:.0f} GB/s, {y[0] / (y[-1] / x[-1] * x[0]):.1f}x the proportional share", (x[0], y[0]), xytext=(26, 22), textcoords="data",
               va="center", color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=5))
    a.annotate(f"all {x[-1]} SMs: {y[-1]:.0f} GB/s", (x[-1], y[-1]), xytext=(-6, 9), textcoords="offset points", ha="right", color=INK, fontsize=10)
    a.set_xlim(0, 176); a.set_ylim(0, 480); a.set_xticks([8, 32, 64, 96, 128, 170]); a.set_xlabel("SMs given to the tenant (of 170)")
    a.set_ylabel("GPU-memory traffic of the tenant (GB/s)"); a.set_title("Tenant alone: traffic falls more slowly than its SMs")
    fig.tight_layout(); fig.savefig(OUT + "bw_tenant_alone.png", dpi=150); print(OUT + "bw_tenant_alone.png")

if __name__ == "__main__":
    fig_tenant(); fig_knee(); fig_tenant_alone(); fig_slowdown(); fig_modes()
