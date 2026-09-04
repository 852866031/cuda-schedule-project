#!/usr/bin/env python3
"""The proposed colocation view: a resizable decode process and a preemptable
workload sharing GPU1, with the workload's snapshot living in DRAM.

Companion to the Act II memory-layout figure in plot_native_pipeline.py — same
three physical locations, redrawn for the elastic phase proposed in
reports/report_colocation_ft.md. No measured data; run from the repo root.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, BLUEF = "#1f6feb", "#dbe7fb"      # prefill process
GRN, GRNF = "#2e7d4f", "#ddeee4"        # decode process
LMC, LMCF = "#c1440e", "#f7e3d8"        # LMCache server
WKL, WKLF = "#9a6700", "#fbf0d9"        # preemptable workload
GREY = "#57606a"

fig, ax = plt.subplots(figsize=(13, 7.0))
ax.set_xlim(0, 130); ax.set_ylim(0, 70); ax.axis("off")
ax.text(65, 67.5, "Colocation view: resizable decode + a preemptable workload on GPU1",
        fontsize=12, ha="center", fontweight="bold", color=GREY)


def cbox(x, y, w, h, txt, n, fc, ec, fs=8.8):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
    ax.text(x + (w - 4.5) / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, zorder=4)
    ax.add_patch(Rectangle((x + w - 4.5, y + h - 2.6), 4.5, 2.6, facecolor=ec,
                           edgecolor=ec, zorder=5))
    ax.text(x + w - 2.25, y + h - 1.3, n, color="white", fontsize=9,
            fontweight="bold", ha="center", va="center", zorder=6)


def arrow(x0, y0, x1, y1, color, style="-|>", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=lw,
                                 linestyle=ls, zorder=5))


# legend
for lx, c, cf, t in ((6, BLUE, BLUEF, "prefill process"),
                     (30, GRN, GRNF, "decode process"),
                     (54, LMC, LMCF, "LMCache server"),
                     (78, WKL, WKLF, "preemptable workload (new)")):
    ax.add_patch(Rectangle((lx, 62.2), 3, 2.2, facecolor=cf, edgecolor=c, lw=1.3))
    ax.text(lx + 4, 63.3, t, fontsize=8.6, va="center", color=GREY)

# ---- GPU0: unchanged by this study, drawn small and quiet
ax.add_patch(Rectangle((4, 14), 26, 44, fill=False, edgecolor=GREY, lw=1.2))
ax.text(17, 55.3, "GPU0 VRAM", fontsize=10.5, color=GREY, fontweight="bold", ha="center")
cbox(6.5, 42, 21, 9, "vLLM prefix cache\n~9 sessions", "①", BLUEF, BLUE, fs=8.4)
ax.text(17, 36.5, "unchanged here —\nprefill is ~97% idle\n(its own colocation\nis a later question)",
        fontsize=8.2, color=GREY, ha="center", va="top")

# ---- host DRAM
ax.add_patch(Rectangle((36, 6), 52, 52, fill=False, edgecolor=GREY, lw=1.2))
ax.text(62, 55.3, "host DRAM (one physical 60 GiB)", fontsize=10.5, color=GREY,
        fontweight="bold", ha="center")
cbox(39, 42, 46, 9, "THE store — each session's prefix ONCE\n≤ 24 GiB, content-addressed",
     "②", LMCF, LMC)
cbox(39, 32, 21, 6, "prefill L1\n6 GiB pinned", "③", BLUEF, BLUE, fs=8.3)
cbox(64, 32, 21, 6, "decode L1\n6 GiB pinned", "④", GRNF, GRN, fs=8.3)
cbox(39, 10, 39, 11, "workload snapshot\nmodel + optimizer + step counter,\n"
     "refreshed every N steps —\na kill loses at most one interval",
     "⑦", WKLF, WKL, fs=8.2)

# ---- GPU1: the elastic memory map
ax.add_patch(Rectangle((94, 6), 32, 52, fill=False, edgecolor=LMC, lw=2.0))
ax.text(110, 55.3, "GPU1 VRAM (32 GB)", fontsize=10.5, color=GREY, fontweight="bold",
        ha="center")
cbox(97, 45, 26, 7, "decode weights\n~15 GiB, fixed", "⑤", GRNF, GRN, fs=8.4)
cbox(97, 29, 26, 13, "in-flight KV: ELASTIC region\n~6 GiB at 2 QPS, ~13 at 4\n"
     "no session KV lives here,\nso shrink moves no data", "⑥", GRNF, GRN, fs=8.2)
cbox(97, 10, 26, 10, "preemptable workload\n~5 GiB, exists only\nwhile decode is shrunk",
     "⑧", WKLF, WKL, fs=8.4)

# the handoff: same physical GiB traded between ⑥ and ⑧
ax.add_patch(FancyArrowPatch((104, 29), (104, 20), arrowstyle="<|-|>",
                             mutation_scale=13, color=LMC, lw=2.2, zorder=5))
ax.text(106.5, 24.5, "★ the same\nphysical GiB,\ntraded with RPS", fontsize=8.2,
        color=LMC, ha="left", va="center")

# decode statelessness: store streams to the L1, L1 feeds VRAM, every request
arrow(74, 42, 74, 38.3, LMC)
ax.add_patch(FancyArrowPatch((85, 35.5), (97, 34.5), arrowstyle="<|-|>",
                             mutation_scale=13, color=LMC, lw=1.6, zorder=5))
ax.text(89.5, 37.6, "retrieve per\nrequest", fontsize=8.0, color=LMC, ha="center",
        va="bottom", zorder=7)

# snapshot flows: periodic checkpoint out, resume back in
arrow(78, 12.5, 97, 12.5, WKL)
ax.text(83, 9.4, "resume from\nsnapshot", fontsize=8.0, color=WKL, ha="center",
        va="top", zorder=7)
arrow(97, 17.5, 78, 17.5, WKL, ls=(0, (4, 3)))
ax.text(83, 18.8, "checkpoint\nevery N steps", fontsize=8.0, color=WKL, ha="center",
        va="bottom", zorder=7)

# control loop captions
ax.text(65, 2.9, "RPS ↑ : kill the workload → decode reclaims ⑧ and grows ⑥ (the fast path — "
        "until it completes, decode queues gracefully rather than failing).",
        fontsize=9.2, color=GREY, ha="center")
ax.text(65, 0.4, "RPS ↓ : in-flight KV drains by itself in seconds → shrink ⑥, return the GiB → "
        "resume the workload from ⑦. TTFT is immune throughout: token #1 comes from GPU0.",
        fontsize=9.2, color=GREY, ha="center")

fig.savefig("figures/colocation_decode_view.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("wrote figures/colocation_decode_view.png")
