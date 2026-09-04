#!/usr/bin/env python3
"""System-components view of the decode+decode colocation, one figure per scenario.

Physical columns GPU0 | host DRAM | GPU1. The 8B split stack (prefill GPU0 -> store :8300
-> decode GPU1) is identical in every scenario; only the Qwen tenant's placement and its
KV flows change:
  A  whole Qwen on GPU1 (prefill+decode), spilling to its own DRAM store on eviction.
  B  Qwen decode-only on GPU1; a temporary GPU0 prefill populates the store, then dies.
  C  Qwen split like the 8B: prefill GPU0 + decode GPU1 + own proxy.
The middle callout is the point of every panel: the two stores are separate processes but
drive the SAME host copy engine. No measured data. Run from the repo root.
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
LMC, LMCF = "#c1440e", "#f7e3d8"        # 8B store
QWN, QWNF = "#8250df", "#ece5fb"        # Qwen processes
GREY = "#57606a"

TITLES = {
    "A": "Scenario A — whole Qwen on GPU1 (prefill + decode here)",
    "B": "Scenario B — Qwen decode-only (prefill from an invisible GPU)",
    "C": "Scenario C — split Qwen (prefill GPU0 + decode GPU1)  [memory-deferred]",
}


def box(ax, x, y, w, h, txt, fc, ec, fs=8.4, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.5, zorder=3,
                           linestyle=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=4)


def arrow(ax, xy0, xy1, color, txt="", off=(0, 1.6), fs=7.4, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=12,
                 color=color, lw=lw, zorder=5))
    if txt:
        mx, my = (xy0[0] + xy1[0]) / 2 + off[0], (xy0[1] + xy1[1]) / 2 + off[1]
        ax.text(mx, my, txt, ha="center", fontsize=fs, color=color)


def draw(scenario):
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 132); ax.set_ylim(0, 72); ax.axis("off")
    ax.text(66, 69.5, TITLES[scenario], fontsize=12.5, ha="center", fontweight="bold",
            color=GREY)

    for x, w, name in ((2, 34, "GPU0"), (48, 36, "host DRAM (60 GiB)"), (96, 34, "GPU1")):
        ax.add_patch(Rectangle((x, 6), w, 56, fill=False, ec=GREY, lw=1.3, ls=(0, (6, 4))))
        ax.text(x + w / 2, 63.4, name, ha="center", fontsize=10, color=GREY)

    # ---- 8B split stack (identical every scenario) ----
    box(ax, 5, 34, 28, 20, "8B prefill\n(24.8 GiB, GPU0)\nprefix cache + LMCache", BLUEF, BLUE)
    box(ax, 51, 34, 30, 18, "8B LMCache store\n(<=24 GiB) + L1 2x6", LMCF, LMC)
    box(ax, 99, 34, 28, 20, "8B decode\n(27.0 GiB, b26)\nrunning KV only", GRNF, GRN)
    arrow(ax, (33, 45), (51, 44), BLUE, "store KV", off=(0, 1.8))
    arrow(ax, (81, 41), (99, 43), GRN, "retrieve", off=(0, 1.8))

    # ---- Qwen store (host DRAM) -- present in every scenario ----
    box(ax, 51, 13, 30, 13, "Qwen LMCache store\n(0.6 / 3.4 GiB) + L1", QWNF, QWN)

    # ---- Qwen engines + flows, per scenario ----
    if scenario == "A":
        box(ax, 99, 12, 28, 14, "Qwen engine (GPU1)\nprefill + decode\n(2.9 GiB)", QWNF, QWN, 8.0)
        # single engine spills to / reloads from its own store on eviction
        ax.add_patch(FancyArrowPatch((99, 19), (81, 19), arrowstyle="<|-|>",
                     mutation_scale=12, color=QWN, lw=1.8, zorder=5))
        ax.text(90, 21.4, "spill / reload\non eviction", ha="center", fontsize=7.2, color=QWN)
        client_note = "8B client → proxy :8000\nQwen client → :8201 (direct)"
    elif scenario == "B":
        box(ax, 5, 12, 28, 14, "Qwen prefill (2.9 GiB)\nTEMP: populates the store\nonce, then killed",
            "white", QWN, 8.0, ls="--")
        box(ax, 99, 12, 28, 14, "Qwen decode-only\n(2.9 GiB)\nretrieve KV + decode", QWNF, QWN)
        arrow(ax, (33, 18), (51, 19), QWN, "populate", off=(0, -2.2))
        arrow(ax, (81, 19), (99, 18), QWN, "KV in (from DRAM)", off=(0, -2.2))
        client_note = "8B client → proxy :8000\nQwen client → :8201 (direct)"
    else:  # C
        box(ax, 5, 12, 28, 14, "Qwen prefill\n(2.9 GiB, GPU0)\nprefix cache + LMCache", QWNF, QWN, 8.0)
        box(ax, 99, 12, 28, 14, "Qwen decode\n(2.9 GiB, GPU1)\nrunning KV only", QWNF, QWN)
        arrow(ax, (33, 18), (51, 19), QWN, "store KV", off=(0, -2.2))
        arrow(ax, (81, 19), (99, 18), QWN, "retrieve", off=(0, -2.2))
        # its own forwarding proxy
        box(ax, 40, 1.5, 30, 6, "Qwen proxy :8001 (forwards token #1)", "white", QWN, 7.6)
        client_note = "8B client → proxy :8000\nQwen client → proxy :8001"

    # ---- the shared-resource callout (the point of the figure) ----
    ax.text(66, 55.5, "Both stores drive the SAME host copy path (CPU memcpy + DRAM BW).\n"
            "It -- not PCIe, not GPU compute -- is the shared resource the two stacks contend for.",
            ha="center", fontsize=8.0, color=GREY,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f8fa", ec=GREY, lw=0.8))
    ax.text(66, 9.4, "separate servers (:8300 / :8301), separate L1 pools", ha="center",
            fontsize=7.4, color=LMC, style="italic")
    ax.text(113, 30.2, client_note, ha="center", fontsize=7.4, color=GREY)

    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"inf_coloc_view_{scenario}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def main():
    for s in ("A", "B", "C"):
        draw(s)


if __name__ == "__main__":
    main()
