#!/usr/bin/env python3
"""KV-slot study, three explainer figures:

  kv_blocks.png   a vLLM KV block, and how one request of this workload fills 392 → 400
  kv_grant.png    the decode engine's VRAM budget → KV grant → block pool (24 GiB)
  kv_slots.png    slots (this study's unit) at the three budgets

    .venv/bin/python scripts/plots/plot_kv_blocks_slots.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
C_PRE, C_SUF, C_GEN, C_RUN, C_FREE, C_W = "#2a78d6", "#1baf7a", "#eb6834", "#2a78d6", "#e3e3df", "#b9c4d4"


def t(ax, x, y, s, fs=11, c="black", ha="left", va="center", bold=False):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, linespacing=1.35,
            fontweight="bold" if bold else "normal")


# ================================================================ 1. blocks
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
t(ax, 2, 94, "vLLM keeps every request's KV in blocks of 16 tokens", 14, bold=True)
X0, X1, Y, H = 2, 98, 70, 12
sc = (X1 - X0) / 6400
ax.add_patch(Rectangle((X0, Y), 6144 * sc, H, fc=C_PRE, ec="white"))
ax.add_patch(Rectangle((X0 + 6144 * sc, Y), 128 * sc, H, fc=C_SUF, ec="white"))
ax.add_patch(Rectangle((X0 + 6272 * sc, Y), 128 * sc, H, fc=C_GEN, ec="white"))
for k in range(1, 400):
    xx = X0 + k * 16 * sc
    ax.plot([xx, xx], [Y, Y + H], color="white", lw=0.4, alpha=0.5)
ax.text(X0 + 3072 * sc, Y + H / 2, "prefix 6,144 tokens = 384 blocks", fontsize=12, color="white",
        ha="center", va="center", fontweight="bold", bbox=dict(fc=C_PRE, ec="none", pad=3))
t(ax, X0 + 6144 * sc, Y + H + 2, "128 new = 8 blocks", 10.5, "#0f6e4a", "right", "bottom")
t(ax, X0 + 6400 * sc, Y - 2, "128 generated = 8 blocks", 10.5, "#9a3f12", "right", "top")
t(ax, X0, Y - 2, "one request of this workload", 10.5, "#444", "left", "top")
# block detail
bx, by, bw, bh = 2, 24, 44, 28
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.5", fc="#eef3f9", ec="#8aa4c4", lw=1.4))
t(ax, bx + bw / 2, by + bh - 3.5, "one block", 13, "#0d366b", "center", bold=True)
t(ax, bx + bw / 2, by + bh / 2 + 0.5, "16 tokens × 32 layers × K and V\n× 8 KV heads × 128 × 2 bytes", 11, "#0d366b", "center")
t(ax, bx + bw / 2, by + 3, "= 2 MiB", 13, "#0d366b", "center", bold=True)
ax.add_patch(FancyArrowPatch((X0 + 200 * sc, Y), (bx + bw / 2, by + bh + 0.8), arrowstyle="-|>",
                             mutation_scale=14, color="#8aa4c4", lw=1.2))
# allocation rule
t(ax, 52, 47, "Allocation is by whole block", 13, bold=True)
t(ax, 52, 39, "admitted only when 392 free blocks exist (its prompt)", 11.5)
t(ax, 52, 32, "token 6,273 opens block 393 … token 6,385 opens block 400", 11.5)
t(ax, 52, 25, "no free block at such a step → a request is preempted", 11.5, "#7a1a12")
t(ax, 52, 15, "This is how vLLM manages KV for any request (block_size = 16).", 10.5, "#555")
fig.savefig(FIGS / "kv_blocks.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================================================================ 2. grant
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
t(ax, 2, 93, "The decode engine's KV grant at a 24 GiB budget", 14, bold=True)
GX0, GX1, GY, GH = 2, 98, 62, 16
gsc = (GX1 - GX0) / 32
ax.add_patch(Rectangle((GX0, GY), 16.23 * gsc, GH, fc=C_W, ec="white"))
ax.add_patch(Rectangle((GX0 + 16.23 * gsc, GY), 7.77 * gsc, GH, fc=C_RUN, ec="white"))
ax.add_patch(Rectangle((GX0 + 24 * gsc, GY), 8 * gsc, GH, fc="#f4f4f1", ec="#444", lw=1, hatch="..."))
ax.add_patch(Rectangle((GX0, GY), 24 * gsc, GH, fc="none", ec="#444", lw=1.6))
t(ax, GX0 + 8.1 * gsc, GY + GH / 2, "weights, activations, CUDA graphs\n16.23 GiB", 11.5, "#0d366b", "center")
t(ax, GX0 + 20.1 * gsc, GY + GH / 2, "KV grant\n7.77 GiB", 12, "white", "center", bold=True)
ax.text(GX0 + 28 * gsc, GY + GH / 2, "unused\n7.35 GiB", fontsize=11, color="#444", ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=2))
t(ax, GX0, GY + GH + 2, "GPU1: 31.35 GiB usable (32 GB card)", 11, "#222", "left", "bottom")
t(ax, GX0 + 24 * gsc, GY + GH + 2, "budget 24 GiB (--gpu-memory-utilization)", 11, "#222", "right", "bottom")
t(ax, 2, 44, "grant = budget − 16.23 GiB  (the same 16.23 GiB at every budget tested)", 12)
t(ax, 2, 30, "7.77 GiB ÷ 2 MiB = 3,980 blocks, one shared pool", 12)
t(ax, 2, 16, "= 9.95 requests of this workload", 12, "#7a1a12", bold=True)
fig.savefig(FIGS / "kv_grant.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ================================================================ 3. slots
fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
t(ax, 2, 95, "Slots: how many whole requests the pool holds (this study's unit)", 14, bold=True)
t(ax, 2, 87, "slot = the 400 blocks one request holds by its last token; admission checks only its 392 prompt blocks", 11.5, "#444")
rows = [(24, 3980, 9, 380, "9.95 slots: after 9 admissions 452 blocks free ≥ 392 → a 10th gets in, later one is evicted"),
        (26, 5004, 12, 204, "12.51 slots: after 12 admissions 300 blocks free < 392 → 12"),
        (28, 6027, 15, 27, "15.07 slots: after 15 admissions 147 free < 392 → 15")]
k = 94 / 6100
BY = 70
for b, blocks, c, spare, note in rows:
    t(ax, 2, BY + 10.5, f"{b} GiB — {blocks:,} blocks", 12.5, bold=True)
    t(ax, 30, BY + 10.5, note, 11, "#7a1a12" if b == 24 else "#444")
    ax.add_patch(Rectangle((2, BY), blocks * k, 8, fc="white", ec="#444", lw=1.2))
    x = 2
    for i in range(c):
        ax.add_patch(Rectangle((x, BY + 0.8), 400 * k - 0.3, 6.4, fc=C_RUN, ec="white", lw=0.6))
        if i == 0:
            t(ax, x + 200 * k, BY + 4, "400", 9.5, "white", "center")
        x += 400 * k
    if spare and b == 24:            # the 10th slot: admitted on 392 blocks, evicted once all grow
        ax.add_patch(Rectangle((x, BY + 0.8), spare * k - 0.3, 6.4, fc="white", ec=C_RUN, lw=1.2, ls="--",
                               hatch="///"))
        t(ax, x + spare * k / 2, BY + 4, "lent", 9.5, C_RUN, "center")
    elif spare:
        ax.add_patch(Rectangle((x, BY + 0.8), spare * k - 0.3, 6.4, fc=C_FREE, ec="#444", lw=0.6, ls="--"))
        if spare > 100:
            t(ax, x + spare * k / 2, BY + 4, str(spare), 9.5, "#444", "center")
    BY -= 24
t(ax, 2, 6, "blue = one slot (400 blocks, 0.78 GiB);  hatched = the lent 10th slot;  grey = left once every slot is full", 11, "#444")
fig.savefig(FIGS / "kv_slots.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("-> kv_blocks.png kv_grant.png kv_slots.png")
