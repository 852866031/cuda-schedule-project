#!/usr/bin/env python3
"""System-components view of the decode+decode colocation (scenario B, the measured one):
the 8B split stack + a Qwen2.5-0.5B decode-only tenant across GPU0 / host DRAM / GPU1,
with KV flows and the shared host copy path that governs the bottleneck.

Companion to plot_colocation_view.py in report_colocation_ft.md -- same three physical
locations, same house style. No measured data; run from the repo root.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
BLUE, BLUEF = "#1f6feb", "#dbe7fb"      # 8B prefill
GRN, GRNF = "#2e7d4f", "#ddeee4"        # 8B decode
LMC, LMCF = "#c1440e", "#f7e3d8"        # LMCache stores (host DRAM)
QWN, QWNF = "#8250df", "#ece5fb"        # Qwen processes
GREY = "#57606a"


def main():
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 132); ax.set_ylim(0, 72); ax.axis("off")
    ax.text(66, 69.5, "Scenario B: 8B split stack + Qwen decode-only tenant "
            "(prefill from an invisible GPU)", fontsize=12.5, ha="center",
            fontweight="bold", color=GREY)

    # three physical location bands
    for x, w, name in ((2, 34, "GPU0"), (48, 36, "host DRAM (60 GiB)"), (96, 34, "GPU1")):
        ax.add_patch(Rectangle((x, 6), w, 56, fill=False, ec=GREY, lw=1.3, ls=(0, (6, 4))))
        ax.text(x + w / 2, 63.4, name, ha="center", fontsize=10, color=GREY)

    def box(x, y, w, h, txt, fc, ec, fs=8.6, ls="-"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.5,
                               zorder=3, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=4)

    # GPU0: 8B prefill (live) + Qwen temp prefill (populate, then killed)
    box(5, 34, 28, 20, "8B prefill\n(24.8 GiB, GPU0)\nprefix cache + LMCache", BLUEF, BLUE)
    box(5, 12, 28, 14, "Qwen prefill (2.9 GiB)\nTEMP: populates the store\nonce, then killed",
        "white", QWN, 8.0, ls="--")

    # host DRAM: two independent LMCache stores
    box(51, 34, 30, 18, "8B LMCache store\n(<=24 GiB) + L1 2x6", LMCF, LMC)
    box(51, 13, 30, 13, "Qwen LMCache store\n(0.6 / 3.4 GiB) + L1", QWNF, QWN)
    ax.text(66, 8.6, "separate servers (:8300 / :8301) --\nbut one host CPU/DRAM copy engine",
            ha="center", fontsize=7.6, color=LMC, style="italic")

    # GPU1: 8B decode (b26) + Qwen decode-only
    box(99, 34, 28, 20, "8B decode\n(27.0 GiB, b26)\nrunning KV only", GRNF, GRN)
    box(99, 12, 28, 14, "Qwen decode-only\n(2.9 GiB)\nretrieve KV + decode", QWNF, QWN)

    def arrow(xy0, xy1, color, txt="", rad=0.0, off=(0, 1.6), fs=7.4, lw=1.6):
        ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=13,
                     color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=5))
        if txt:
            mx, my = (xy0[0] + xy1[0]) / 2 + off[0], (xy0[1] + xy1[1]) / 2 + off[1]
            ax.text(mx, my, txt, ha="center", fontsize=fs, color=color)

    # 8B: prefill -> store (write), store -> decode (retrieve, into first-ITL)
    arrow((33, 45), (51, 44), BLUE, "store KV", off=(0, 1.8))
    arrow((81, 41), (99, 43), GRN, "retrieve", off=(0, 1.8))
    # Qwen: temp prefill -> store (populate), store -> decode (the "simulated prefill")
    arrow((33, 18), (51, 19), QWN, "populate", off=(0, -2.2))
    arrow((81, 19), (99, 18), QWN, "KV in (from DRAM)", off=(0, -2.2))

    # the shared-resource annotation
    ax.text(66, 55.5, "Both stores drive the SAME host copy path (CPU memcpy + DRAM BW).\n"
            "It -- not PCIe, not GPU compute -- is the shared resource the two decodes contend for.",
            ha="center", fontsize=8.0, color=GREY,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f8fa", ec=GREY, lw=0.8))

    # clients
    ax.text(113, 58.5, "8B client -> proxy :8000\nQwen client -> :8201 (direct)",
            ha="center", fontsize=7.6, color=GREY)

    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_view.png", dpi=140, bbox_inches="tight")
    print("wrote figures/inf_coloc_view.png")


if __name__ == "__main__":
    main()
