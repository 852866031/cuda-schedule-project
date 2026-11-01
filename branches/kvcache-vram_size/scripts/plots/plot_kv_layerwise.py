#!/usr/bin/env python3
"""KV-slot study, report §6.1: the admitting step with and without `use_layerwise`, as two
lanes per arm — transfer (PCIe copy) and compute (the forward) — so that overlap, or its
absence, is visible. Lengths are to scale from the 24 GiB / 2.5 req/s traces.

    .venv/bin/python scripts/plots/plot_kv_layerwise.py -> figures/kv_layerwise.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
C_T, C_C = "#eb6834", "#2a78d6"
import os
METHOD = os.environ.get("KV_METHOD", "base")
# one request's copy and one forward, measured at 24 GiB / 2.5 req/s
COPY_MS, FWD_MS = {"base": (190.0, 16.0), "cops": (115.0, 14.4)}[METHOD]
sc = 88 / (COPY_MS + FWD_MS)

fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), gridspec_kw={"hspace": 0.6})


def arm(ax, title, transfer, compute, note):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12.5, pad=4)
    for y, lab in ((66, "transfer\n(PCIe)"), (36, "compute\n(GPU)")):
        ax.text(9.5, y + 9, lab, ha="right", va="center", fontsize=10.5)
        ax.add_patch(Rectangle((10, y), 88, 18, fc="#f4f4f1", ec="none"))
    for s, e in transfer:
        ax.add_patch(Rectangle((10 + s * sc, 66), (e - s) * sc, 18, fc=C_T, ec="white", lw=0.5))
    for s, e in compute:
        ax.add_patch(Rectangle((10 + s * sc, 36), (e - s) * sc, 18, fc=C_C, ec="white", lw=0.5))
    end = max(e for _, e in transfer + compute)
    ax.plot([10 + end * sc] * 2, [30, 90], color="#444", lw=1, ls="--")
    ax.text(10 + end * sc + 0.8, 60, f"{end:.0f} ms", fontsize=10.5, va="center")
    ax.text(10, 12, note, fontsize=10.5, color="#7a1a12")


# baseline: whole copy, then forward
arm(axes[0], "baseline: copy all 32 layers, then run the 32-layer forward",
    [(0, COPY_MS)], [(COPY_MS, COPY_MS + FWD_MS)],
    "the batch stops for the whole copy")

# layerwise: 32 alternating pieces, still serial
t, tr, cp = 0.0, [], []
for i in range(32):
    tr.append((t, t + COPY_MS / 32)); t += COPY_MS / 32
    cp.append((t, t + FWD_MS / 32)); t += FWD_MS / 32
arm(axes[1], "use_layerwise: copy layer i, run layer i, … — alternating, still never at the same time",
    tr, cp,
    ("same total: the copy is split, not hidden   (what it does fix: the L1 staging pool no longer thrashes → preemptions 35 → 12)"
     if METHOD == "base" else
     "same total: split, not hidden — each layer still waits for its L1/store fetch   (queue 143 → 136 ms, preemptions 8 → 4)"))
fig.suptitle("One admitting step on the decode engine, to scale (24 GiB, 2.5 req/s"
             + (", compiled LMCache kernels)" if METHOD == "cops" else ")"), fontsize=13, y=1.0)
fig.savefig(REPO / "figures" / "kv_layerwise.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_layerwise.png")
