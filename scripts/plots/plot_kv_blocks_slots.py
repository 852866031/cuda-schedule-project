#!/usr/bin/env python3
"""KV-slot study: what a KV block is, what a slot is, and where the numbers come from.
Three rows: (1) one request's tokens → 16-token blocks → bytes per block;
(2) GPU1's 24 GiB budget → the KV grant → the block pool, and how many whole requests fit;
(3) the three budgets side by side.

    .venv/bin/python scripts/plots/plot_kv_blocks_slots.py  -> figures/kv_blocks_slots.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent.parent
C_PRE, C_SUF, C_GEN, C_RUN, C_FREE, C_W = "#2a78d6", "#1baf7a", "#eb6834", "#2a78d6", "#e3e3df", "#b9c4d4"

fig, ax = plt.subplots(figsize=(15, 10.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def t(x, y, s, fs=10, c="black", ha="left", va="center", bold=False, style="normal"):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, linespacing=1.3,
            fontweight="bold" if bold else "normal", style=style)


def arrow(p, q, c="#333", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=c, lw=lw, ls=ls))


# ---------------------------------------------------------------- row 1: tokens -> blocks
t(2, 96, "1. vLLM stores every request's KV in fixed blocks of 16 tokens (PagedAttention; block_size = 16)", 12.5, bold=True)
X0, X1, Y, H = 4, 96, 86, 5
tot = 6400
sc = (X1 - X0) / tot
for a, b, c, lab in ((0, 6144, C_PRE, "prefix: 6,144 tokens (one of 32 sessions, in LMCache)"),
                     (6144, 6272, C_SUF, "128 new"), (6272, 6400, C_GEN, "128 generated")):
    ax.add_patch(Rectangle((X0 + a * sc, Y), (b - a) * sc, H, fc=c, ec="white", lw=0.8))
t(X0 + 3072 * sc, Y + H / 2, "prefix: 6,144 tokens (one of 32 sessions; its KV is in LMCache)", 10, "white", "center")
t(X0 + 6208 * sc, Y - 1.5, "128 new ↑", 8.5, "#0f6e4a", "right", "top")
t(X0 + 6336 * sc, Y + H + 1.2, "↓ 128 generated", 8.5, "#9a3f12", "right", "bottom")
# block ticks
for k in range(0, 401, 16):
    xx = X0 + k * 16 * sc
    ax.plot([xx, xx], [Y, Y + H], color="white", lw=0.4)
t(X0, Y + H + 1.2, "this workload's request: 6,272 prompt tokens ÷ 16 = 392 blocks at admission; +128 generated tokens → 400 blocks at the end",
  10, "#222", va="bottom")
# one block detail
bx, by = 4, 70
ax.add_patch(FancyBboxPatch((bx, by), 38, 9, boxstyle="round,pad=0.3", fc="#eef3f9", ec="#8aa4c4"))
t(bx + 19, by + 6.6, "one block = 16 tokens of K and V for all 32 layers", 9.5, "#0d366b", "center", bold=True)
t(bx + 19, by + 2.8, "16 × 32 layers × 2 (K,V) × 8 KV heads × 128 × 2 B = 2 MiB", 9.5, "#0d366b", "center")
arrow((X0 + 1500 * sc, Y), (bx + 19, by + 9.3), "#8aa4c4", lw=1)
t(46, 77.5, "vLLM allocates whole blocks, never tokens: a request is admitted only when all 392 are free;", 10, "#222")
t(46, 74.2, "generating token 6,273 opens block #393, token 6,289 opens #394, … token 6,385 opens #400.", 10, "#222")
t(46, 70.9, "Run out of free blocks at such a step → vLLM preempts a request (§3, case 2).", 10, "#7a1a12")

# ---------------------------------------------------------------- row 2: GPU memory -> pool -> slots
t(2, 62, "2. VRAM budget → KV grant → block pool (vLLM) → slots (this study's unit, 24 GiB shown)", 12.5, bold=True)
GX0, GX1, GY, GH = 4, 96, 52, 6
gsc = (GX1 - GX0) / 32
ax.add_patch(Rectangle((GX0, GY), 24 * gsc, GH, fc="white", ec="#444", lw=1.2))
ax.add_patch(Rectangle((GX0 + 24 * gsc, GY), 8 * gsc, GH, fc="#f4f4f1", ec="#444", lw=1.2, hatch="..."))
ax.add_patch(Rectangle((GX0, GY), 16.23 * gsc, GH, fc=C_W, ec="white", lw=0.8))
ax.add_patch(Rectangle((GX0 + 16.23 * gsc, GY), 7.77 * gsc, GH, fc=C_RUN, ec="white", lw=0.8))
t(GX0 + 8.1 * gsc, GY + GH / 2, "weights 15 GiB + activations, CUDA graphs, workspace ≈ 16.2 GiB", 9.5, "#0d366b", "center")
t(GX0 + 20.1 * gsc, GY + GH / 2, "KV grant 7.77 GiB", 9.5, "white", "center", bold=True)
ax.text(GX0 + 28 * gsc, GY + GH / 2, "not given to\nthe engine", fontsize=9, color="#444", ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=1.5))
t(GX0, GY + GH + 1.2, "GPU1 (32 GB): the engine may use 24 GiB (--gpu-memory-utilization); what the model does not need "
  "becomes the KV grant: budget − 16.23 GiB at every budget tested", 9.5, "#222", va="bottom")
# pool bar
PX0, PX1, PY, PH = 4, 96, 40, 6
psc = (PX1 - PX0) / 4000
ax.add_patch(Rectangle((PX0, PY), 3980 * psc, PH, fc="white", ec="#444", lw=1.2))
x = PX0
for i in range(9):
    ax.add_patch(Rectangle((x, PY + 0.6), 400 * psc - 0.3, PH - 1.2, fc=C_RUN, ec="white", lw=0.6))
    t(x + 200 * psc, PY + PH / 2, f"slot {i + 1}", 8.5, "white", "center")
    x += 400 * psc
ax.add_patch(Rectangle((x, PY + 0.6), 380 * psc - 0.3, PH - 1.2, fc=C_FREE, ec="#444", lw=0.6, ls="--"))
t(x + 190 * psc, PY + PH / 2, "380", 8, "#444", "center")
arrow((GX0 + 20.1 * gsc, GY), (PX0 + 1990 * psc, PY + PH + 0.3), "#8aa4c4", lw=1)
t(PX0, PY - 1.8, "vLLM: 7.77 GiB ÷ 2 MiB = 3,980 blocks, one shared pool.   This study: a SLOT = the 400 blocks one request of this "
  "workload holds from admission to its last token → 3,980 ÷ 400 = 9.95 → 9 whole slots + 380 spare blocks.", 10, "#222", va="top")
t(PX0, PY - 5.2, "Admission needs only 392 free blocks: 3,980 − 9 × 392 = 452 ≥ 392, so a 10th request gets in while the nine are young;",
  9.5, "#7a1a12", va="top")
t(PX0, PY - 8.2, "once all ten grow (4,000 > 3,980) one is evicted → 24 GiB behaves as 10 slots with occasional evictions.",
  9.5, "#7a1a12", va="top")

# ---------------------------------------------------------------- row 3: three budgets
t(2, 26.5, "3. The three budgets measured", 12.5, bold=True)
rows = [(24, 7.77, 3980, 9, 380, "10 slots with evictions"), (26, 9.77, 5004, 12, 204, "12 slots"), (28, 11.77, 6027, 15, 27, "15 slots")]
BY = 19
k3 = (PX1 - PX0) / 6100
for b, g, blocks, c, spare, eff in rows:
    t(4, BY + 5.2, f"{b} GiB: grant {g:.2f} GiB = {blocks:,} blocks = {blocks / 400:.2f} requests → {c} whole slots"
      f"{' + ' + str(spare) + ' spare blocks' if spare else ''}  →  behaves as {eff}", 9.5, "#222")
    ax.add_patch(Rectangle((4, BY), blocks * k3, 3.6, fc="white", ec="#444", lw=1))
    x = 4
    for i in range(c):
        ax.add_patch(Rectangle((x, BY + 0.4), 400 * k3 - 0.25, 2.8, fc=C_RUN, ec="white", lw=0.5))
        x += 400 * k3
    if spare:
        ax.add_patch(Rectangle((x, BY + 0.4), spare * k3 - 0.25, 2.8, fc=C_FREE, ec="#444", lw=0.5, ls="--"))
    BY -= 8
t(4, 0.5, "blue cell = 400 blocks = one request's KV for its whole decode (0.78 GiB); grey = spare blocks, fewer than one request needs",
  9.5, "#444")

fig.savefig(REPO / "figures" / "kv_blocks_slots.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_blocks_slots.png")
