#!/usr/bin/env python3
"""KV-slot study, report §5: could a request hold less KV by streaming layers in each step?

  (a) today: all 32 layers resident for the whole decode
  (b) the idea: a few layers resident, the rest streamed in every step
  (c) measured step time, resident vs streamed (bench_stream.json)

    .venv/bin/python scripts/plots/plot_kv_stream_idea.py -> figures/kv_stream_idea.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent.parent
C_RES, C_STR, C_HOST = "#2a78d6", "#eb6834", "#f3f1ea"
rows = json.load(open(REPO / "output" / "kv_vram" / "bench_stream.json"))["rows"]
b1 = next(r for r in rows if r["batch"] == 1)
b16 = next(r for r in rows if r["batch"] == 16)

fig = plt.figure(figsize=(16, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.45)


def pool(ax, title, resident, note, arrow):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    ax.add_patch(Rectangle((2, 14), 62, 76, fc="#eef3f9", ec="#8aa4c4"))
    ax.text(5, 85, "GPU", fontsize=11, color="#0d366b", fontweight="bold")
    rh = 60 / 32
    for k in range(5):
        x0 = 7 + k * 11.5
        for i in range(32):
            ax.add_patch(Rectangle((x0, 18 + i * rh), 8.5, rh * 0.8,
                                   fc=C_RES if i in resident else "white",
                                   ec="#444" if i in resident else "#ccc", lw=0.4))
    ax.add_patch(Rectangle((70, 14), 28, 76, fc=C_HOST, ec="#b9b3a1"))
    ax.text(73, 85, "host RAM", fontsize=11, color="#4a4639", fontweight="bold")
    ax.text(84, 50, "KV", ha="center", fontsize=11, color="#4a4639")
    if arrow:
        ax.add_patch(FancyArrowPatch((70, 42), (20, 42), arrowstyle="-|>", mutation_scale=18, color=C_STR, lw=3))
    ax.text(50, 5, note, ha="center", fontsize=11, color="#7a1a12" if arrow else "#222")


pool(fig.add_subplot(gs[0]), "(a) today", set(range(32)), "5 requests × 32 layers resident", False)
pool(fig.add_subplot(gs[1]), "(b) the idea", {12, 13, 14, 15}, "4 layers resident, 28 streamed in — every step", True)

ax = fig.add_subplot(gs[2])
ax.set_title("(c) one decode step, measured", fontsize=12, loc="left")
labels = ["batch 1\nresident", "batch 1\nstreamed", "batch 16\nresident", "batch 16\nstreamed"]
vals = [b1["resident_ms"], b1["stream_d1_ms"], b16["resident_ms"], b16["stream_d1_ms"]]
cols = [C_RES, C_STR, C_RES, C_STR]
bars = ax.barh(range(4), vals, color=cols, height=0.65)
for i, v in enumerate(vals):
    ax.text(v * 1.08, i, f"{v:.0f} ms", va="center", fontsize=11)
ax.set_yticks(range(4))
ax.set_yticklabels(labels, fontsize=10.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(8, 3000)
ax.set_xlabel("step time (ms, log)")
ax.spines[["top", "right"]].set_visible(False)
ax.text(8.5, 4.0, "streamed: 0.78 GiB per request per step at 14.5 GB/s", fontsize=10, color="#7a1a12", va="top")
fig.suptitle("Could a request keep less KV on the GPU by streaming layers in each step? "
             f"No: {b16['stream_d1_ms'] / b16['resident_ms']:.0f}× slower at batch 16", fontsize=13, y=1.0)
fig.savefig(REPO / "figures" / "kv_stream_idea.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_stream_idea.png")
