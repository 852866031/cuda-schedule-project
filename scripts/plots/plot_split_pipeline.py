#!/usr/bin/env python3
"""Swimlane of one session's requests through the LMCache split pipeline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, BLUEF = "#1f6feb", "#dbe7fb"
COP, COPF = "#c1440e", "#f7e3d8"
GREY = "#57606a"

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 130)
ax.set_ylim(0, 62)
ax.axis("off")

LANES = [("client / router", 52), ("GPU0 · prefill", 38), ("LMCache (host DRAM)", 24),
         ("GPU1 · decode", 8)]
for name, y in LANES:
    ax.axhline(y + 7, color="#dddddd", lw=0.8, zorder=0)
    ly = y + 5.5 if name.startswith("LMCache") else y + 3
    ax.text(0.5, ly, name, fontsize=11, fontweight="bold", va="center", color=GREY)

def box(x, y, w, h, txt, fc, ec, fs=9):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=4)

def arrow(x0, y0, x1, y1, color, txt="", tx=0, ty=0, fs=8.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.6, linestyle=ls, zorder=5))
    if txt:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, txt, fontsize=fs, color=color,
                ha="center", zorder=6)

# The persistent store: written once, read forever.
ax.add_patch(Rectangle((17, 24), 109, 5, facecolor=COPF, edgecolor=COP, lw=1.4, zorder=2))
ax.text(19, 26.5, "session S prefix KV — stored ONCE (24 × 256-token chunks, "
        "content-addressed), read by both GPUs forever", fontsize=9, va="center",
        color=COP, zorder=4)

for x0, tag in ((14, "①"), (56, "②"), (94, "③")):
    ax.text(x0 + 1, 59.5, tag, fontsize=13, fontweight="bold", color=GREY)
ax.text(17.5, 59.5, "first request of session S", fontsize=10, color=GREY)
ax.text(59.5, 59.5, "returning request (GPU0 VRAM hit)", fontsize=10, color=GREY)
ax.text(97.5, 59.5, "returning after GPU0 eviction", fontsize=10, color=GREY)
for xd in (54, 92):
    ax.axvline(xd, color="#cccccc", lw=0.9, ls="--")

# ---- episode 1: cold session
box(17, 38, 16, 6, "prefill prefix + suffix\n~530 ms (cold)", BLUEF, BLUE)
arrow(21, 44, 21, 51, BLUE, "", fs=9)
ax.text(22, 52.3, "token #1 → client TTFT", fontsize=9, color=BLUE, ha="left")
arrow(32, 38, 35, 29.2, COP, "store prefix\n0.75 GB", tx=6.5, ty=1.2)
arrow(38, 24, 38, 15, COP, "retrieve\n0.75 GB", tx=-4.5)
box(34, 8, 17, 6, "compute suffix +\ndecode 128 steps", BLUEF, BLUE)
arrow(48, 14, 48, 51, GREY, "", ls=":")
ax.text(47, 46.3, "tokens #2 … #128\n(first gap absorbs\nthe retrieval)", fontsize=8.5,
        color=GREY, ha="right")

# ---- episode 2: warm on GPU0
box(57, 38, 13, 6, "suffix only\n~30 ms (VRAM hit)", BLUEF, BLUE)
arrow(61, 44, 61, 51, BLUE, "token #1 — TTFT ≈ 105 ms", tx=11, fs=9)
ax.text(70.5, 34.2, "nothing stored: prefix already present,\n128-token suffix < 1 chunk",
        fontsize=8, color=GREY, ha="center")
arrow(76, 24, 76, 15, COP, "retrieve\nagain", tx=-4)
box(72, 8, 17, 6, "compute suffix +\ndecode 128 steps", BLUEF, BLUE)
arrow(86, 14, 86, 51, GREY, "", ls=":")

# ---- episode 3: evicted from GPU0's VRAM cache
arrow(97, 29.2, 97, 38, COP, "GPU0 retrieves the\nprefix it computed in ①", tx=10.5, ty=0.6)
box(95, 38, 14, 6, "suffix only ~30 ms\n(no recompute)", BLUEF, BLUE)
arrow(99, 44, 99, 51, BLUE, "token #1", tx=5, fs=9)
arrow(114, 24, 114, 15, COP, "retrieve\nagain", tx=-4)
box(110, 8, 17, 6, "compute suffix +\ndecode 128 steps", BLUEF, BLUE)
arrow(124, 14, 124, 51, GREY, "", ls=":")

ax.text(65, 1.5, "GPU1 keeps no session KV between requests — it retrieves the prefix every time; "
        "with the forwarding router that retrieval sits in the token-1→2 gap, not in TTFT.",
        fontsize=9.5, color=GREY, ha="center")

fig.savefig("figures/split_lmcache_pipeline.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("wrote figures/split_lmcache_pipeline.png")
