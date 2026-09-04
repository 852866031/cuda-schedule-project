#!/usr/bin/env python3
"""How the tenant hits the incumbent on two independent axes (decode+decode colocation).

One 8B request has two legs: the prefill leg on GPU0 (TTFT, axis 2, via the shared host
store path) and the decode leg on GPU1 (TPOT, axis 1, sharing GPU1 with the Qwen decode).
No measured data. Run from the repo root.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
BLU, BLUF = "#1f6feb", "#dbe7fb"     # 8B
GRN, GRNF = "#2e7d4f", "#ddeee4"     # 8B decode
PUR, PURF = "#8250df", "#ece5fb"     # Qwen
ORG, ORGF = "#c1440e", "#f7e3d8"     # host store / axis-2
GRY = "#57606a"


def main():
    fig, axT = plt.subplots(figsize=(13.5, 5.0))
    axT.set_xlim(0, 132); axT.set_ylim(0, 46); axT.axis("off")
    axT.text(66, 44.5, "One 8B request: two legs, two contention points",
             fontsize=12.5, ha="center", fontweight="bold", color=GRY)

    def box(x, y, w, h, txt, fc, ec, fs=8.4):
        axT.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.5, zorder=3))
        axT.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=4)

    def arr(xy0, xy1, color=GRY, lw=1.6):
        axT.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=13,
                      color=color, lw=lw, zorder=5))

    box(2, 16, 15, 9, "client", "white", GRY)
    box(21, 16, 17, 9, "proxy :8000\n(forwards\ntoken #1)", "white", GRY, 7.8)
    box(44, 15, 22, 11, "GPU0\n8B prefill leg\ncompute + host store", BLUF, BLU)
    box(74, 15, 24, 11, "GPU1\n8B decode leg\ntokens 2..128", GRNF, GRN)
    box(104, 15, 24, 11, "Qwen decode\n(shares GPU1)", PURF, PUR)

    arr((17, 20.5), (21, 20.5))
    arr((38, 20.5), (44, 20.5))
    arr((66, 20.5), (74, 20.5))
    # token #1 back to the client, routed as an arc through the clear band below the boxes
    axT.add_patch(FancyArrowPatch((51, 15), (9.5, 15.5), arrowstyle="-|>",
                  mutation_scale=12, color=BLU, lw=1.5, zorder=6,
                  connectionstyle="arc3,rad=-0.55"))
    axT.text(30, 3.2, "token #1 (TTFT) returns to the client the instant "
             "GPU0's forward pass finishes", fontsize=7.6, color=BLU, ha="center")

    # axis-2 callout on the prefill leg (host store)
    box(40, 31, 30, 6, "shared host store path\n(:8300 server, CPU memcpy)", ORGF, ORG, 7.4)
    arr((55, 31), (55, 26), ORG, 1.4)
    axT.text(55, 39.4, "AXIS 2 → TTFT", fontsize=8.4, color=ORG, ha="center",
             fontweight="bold")

    # axis-1 callout between the two decodes
    axT.add_patch(FancyArrowPatch((98, 20.5), (104, 20.5), arrowstyle="<|-|>",
                  mutation_scale=12, color=PUR, lw=1.8, zorder=5))
    axT.text(101, 33.5, "AXIS 1 → TPOT", fontsize=8.4, color=PUR, ha="center",
             fontweight="bold")
    axT.text(101, 30.8, "share GPU1\n(no MPS)", fontsize=7.2, color=PUR, ha="center")

    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_mechanism.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("wrote figures/inf_coloc_mechanism.png")


if __name__ == "__main__":
    main()
