#!/usr/bin/env python3
"""Memory/placement layout of the tenant scenarios (decode+decode colocation study).

Four panels: solo, A (whole Qwen on GPU1), B (Qwen decode-only, KV from DRAM), C
(split Qwen). Physical columns GPU0 | host DRAM | GPU1; boxes scaled to GiB, colored by
process family (8B = blue, Qwen = red, dashed = transient/deferred). No measured data.
Run from the repo root.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
BLUE, RED, GREY = "#1f6feb", "#c1440e", "#57606a"
BLUE_L, RED_L = "#d6e4ff", "#f0d9cd"

GPU_GIB, DRAM_GIB = 31.8, 60.0
COL_H, COL_W = 3.0, 0.66


def column(ax, x, title, total_gib, boxes):
    h = COL_H * total_gib / DRAM_GIB
    ax.add_patch(Rectangle((x, 0), COL_W, h, fill=False, ec=GREY, lw=1.1))
    ax.text(x + COL_W / 2, -0.12, f"{title}\n{total_gib:g}", ha="center", va="top",
            fontsize=7.2, color=GREY)
    y = 0.0
    for label, gib, face, edge, ls in boxes:
        bh = h * gib / total_gib
        ax.add_patch(Rectangle((x + 0.02, y + 0.01), COL_W - 0.04, max(bh - 0.02, 0.03),
                               fc=face, ec=edge, lw=1.0, ls=ls, zorder=3))
        if bh > 0.28:
            ax.text(x + COL_W / 2, y + bh / 2, label, ha="center", va="center",
                    fontsize=6.4, zorder=4)
        else:
            ax.text(x + COL_W + 0.04, y + bh / 2, label, ha="left", va="center",
                    fontsize=6.0, zorder=4)
        y += bh
    return h


def free(gib):
    return (f"free {gib:g}", gib, "white", GREY, ":")


def panel(ax, scenario):
    x0, x1, x2 = 0.0, 1.02, 2.04
    has8b = scenario != "solo"

    # GPU0
    gpu0 = []
    if has8b:
        gpu0.append(("8B prefill 24.8", 24.8, BLUE_L, BLUE, "-"))
    if scenario in ("B", "solo"):
        gpu0.append(("Qwen prefill 2.9\n(temp: populate)", 2.9, "white", RED, "--"))
    elif scenario == "C":
        gpu0.append(("Qwen prefill 2.9", 2.9, RED_L, RED, "-"))
    gpu0.append(free(round(GPU_GIB - sum(b[1] for b in gpu0), 1)))

    # host DRAM
    dram = []
    if has8b:
        dram.append(("8B store <=24 + L1 2x6", 36.0, BLUE_L, BLUE, "-"))
    dram.append(("Qwen store + L1", 4.4, RED_L, RED, "-"))
    dram.append(("engines/OS ~14", 14.0, "white", GREY, "-"))
    dram.append(free(round(DRAM_GIB - sum(b[1] for b in dram), 1)))

    # GPU1
    gpu1 = []
    if has8b:
        gpu1.append(("8B decode 27.0 (b26)", 27.0, BLUE_L, BLUE, "-"))
    gpu1.append(("Qwen 2.9", 2.9, RED_L, RED, "-"))
    gpu1.append(free(round(GPU_GIB - sum(b[1] for b in gpu1), 1)))

    column(ax, x0, "GPU0", GPU_GIB, gpu0)
    hd = column(ax, x1, "host DRAM", DRAM_GIB, dram)
    column(ax, x2, "GPU1", GPU_GIB, gpu1)

    # KV-flow arrow: DRAM store -> GPU1 (the transfer that matters for B/C)
    if scenario in ("B", "C"):
        ax.add_patch(FancyArrowPatch((x1 + COL_W, hd * 0.55), (x2, COL_H * 0.55),
                     arrowstyle="-|>", mutation_scale=9, color=RED, lw=1.1))
        ax.text((x1 + COL_W + x2) / 2, COL_H * 0.62, "KV in", ha="center",
                fontsize=5.8, color=RED)

    titles = {"solo": "solo\nQwen decode-only, no 8B",
              "A": "A - whole Qwen on GPU1\n(prefill+decode here)",
              "B": "B - Qwen decode-only\n(KV from DRAM, prefill invisible)",
              "C": "C - split Qwen\n(prefill GPU0 + decode GPU1)"}
    ax.text(1.35, COL_H + 0.28, titles[scenario], ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(-0.5, COL_H + 0.75)
    ax.axis("off")


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, sc in zip(axes, ("solo", "A", "B", "C")):
        panel(ax, sc)
    fig.suptitle("Tenant placements (8B split stack identical in A/B/C; deferred: C by memory)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_layout.png", dpi=140, bbox_inches="tight")
    print("wrote figures/inf_coloc_layout.png")


if __name__ == "__main__":
    main()
