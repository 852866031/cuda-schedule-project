#!/usr/bin/env python3
"""Eighth study: where each process lives in the three placements of the 3B.

Columns are physical: GPU0 | host DRAM | GPU1. The 8B split (blue) is identical in every
panel; only the 3B (green) changes. Arrows are KV movement. Sizes are the measured
footprints (GPU memory from nvidia-smi, store sizes from the configs).

    .venv/bin/python scripts/plots/plot_pfc_layout.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent.parent
plt.rcParams.update({"font.size": 12})
BIG, BIG_L, SM, SM_L = "#1f6feb", "#dbe9ff", "#2e7d4f", "#d8f0e0"
COLX = {"gpu0": 0.17, "dram": 0.5, "gpu1": 0.83}
W = 0.24


def box(ax, col, y, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((COLX[col] - W / 2, y), W, h, boxstyle="round,pad=0.008",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(COLX[col], y + h / 2, text, ha="center", va="center", fontsize=10.5)


def arrow(ax, p, q, c):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=c, lw=2))


def panel(ax, case):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for col, lab in (("gpu0", "GPU0"), ("dram", "host DRAM"), ("gpu1", "GPU1")):
        ax.text(COLX[col], 0.97, lab, ha="center", va="top", fontsize=13, fontweight="bold")
    # 8B split -- same in every panel
    box(ax, "gpu0", 0.66, 0.2, "8B prefill\n23 GiB", BIG_L, BIG)
    box(ax, "dram", 0.66, 0.2, "8B store 24 GiB\n+ 12 GiB pinned", BIG_L, BIG)
    box(ax, "gpu1", 0.66, 0.2, "8B decode\n23 GiB", BIG_L, BIG)
    arrow(ax, (COLX["gpu0"] + W / 2, 0.79), (COLX["dram"] - W / 2, 0.79), BIG)
    arrow(ax, (COLX["dram"] + W / 2, 0.73), (COLX["gpu1"] - W / 2, 0.73), BIG)
    # the 3B
    box(ax, "gpu1", 0.3, 0.2, "3B prefill\n7.6 GiB", SM_L, SM)
    if case == "P1":
        ax.text(COLX["gpu1"], 0.22, "KV dropped", ha="center", va="top", color=SM, fontsize=11)
    if case in ("P2", "P3"):
        box(ax, "dram", 0.3, 0.2, "3B store\n≤ 2 GiB", SM_L, SM)
        arrow(ax, (COLX["gpu1"] - W / 2, 0.4), (COLX["dram"] + W / 2, 0.4), SM)
    if case == "P3":
        box(ax, "gpu0", 0.3, 0.2, "3B decode\n7.8 GiB", SM_L, SM)
        arrow(ax, (COLX["dram"] - W / 2, 0.4), (COLX["gpu0"] + W / 2, 0.4), SM)
        ax.text(0.5, 0.16, "8B pinned memory here: 8 GiB", ha="center", va="top",
                fontsize=10.5, color="#555")
    titles = {"P1": "P1  prefill only", "P2": "P2  prefill, KV to DRAM",
              "P3": "P3  prefill → DRAM → decode on GPU0"}
    ax.set_title(titles[case], fontsize=14, pad=4)


fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
for ax, case in zip(axes, ("P1", "P2", "P3")):
    panel(ax, case)
fig.text(0.5, -0.02, "Blue = 8B split (identical in all panels); green = Qwen2.5-3B, fresh 6144-token "
         "prompts. Arrows = KV movement. All engines are MPS clients.", ha="center", fontsize=12)
fig.tight_layout()
fig.savefig(REPO / "figures" / "pfc_layout.png", dpi=130, bbox_inches="tight")
print("wrote figures/pfc_layout.png")
