#!/usr/bin/env python3
"""KV-slot study: schematic of the re-simulation in report §4.2 — what goes in (measured
arrival times, measured residences, slot count), what the rule is (FCFS onto c slots),
what comes out (a simulated wait per request) and what it is compared with (the measured
wait). Toy example with c = 3 and 8 requests; numbers are illustrative.

    .venv/bin/python scripts/plots/plot_kv_resim.py  -> figures/kv_resim_schematic.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
C_RUN, C_WAIT, C_IN, C_OUT = "#2a78d6", "#9a9a95", "#e3f1ea", "#fdeee7"

# toy inputs: (arrival, residence) measured; c = 3
arr = [0.0, 0.5, 1.0, 1.6, 3.3, 3.9, 5.1, 6.0]
res = [3.0, 2.6, 3.2, 2.8, 3.0, 2.7, 2.9, 3.1]
c = 3
# FCFS replay
free = [0.0] * c
start, lane = [], []
for a, s in zip(arr, res):
    i = min(range(c), key=lambda k: free[k])
    t0 = max(a, free[i])
    start.append(t0)
    lane.append(i)
    free[i] = t0 + s
wait = [t0 - a for a, t0 in zip(arr, start)]

fig = plt.figure(figsize=(14, 7.2))
gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.6, 1.3], hspace=0.55)

# ---- inputs
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(-0.3, 9.5)
ax0.set_ylim(0, 1)
ax0.axis("off")
ax0.add_patch(FancyBboxPatch((-0.2, 0.05), 9.6, 0.9, boxstyle="round,pad=0.1", fc=C_IN, ec="#1a7a55", lw=1.4))
ax0.text(0.0, 0.72, "INPUTS — all measured in the very cell being re-simulated (same budget, same rate, same run as §4.1)", fontsize=12, fontweight="bold", color="#0f4d35", va="center")
for a in arr:
    ax0.plot([a * 0.45, a * 0.45], [0.12, 0.42], color="black", lw=1.6)
ax0.text(3.3, 0.27, "arrival time of each request (ticks)", fontsize=10.5, va="center")
ax0.text(9.3, 0.72, f"c = {c} slots (probe)", fontsize=11, ha="right", va="center", color="#0f4d35", fontweight="bold")
ax0.text(9.3, 0.27, "residence of each request (traced)", fontsize=10.5, ha="right", va="center")

# ---- the rule
ax1 = fig.add_subplot(gs[1])
ax1.set_xlim(-0.3, 9.5)
ax1.set_ylim(-0.8, c + 2.2)
ax1.axis("off")
ax1.text(0.0, c + 1.8, "RULE — first come, first served onto the c slots; each request keeps its slot for its measured residence",
         fontsize=12, fontweight="bold", va="center")
for k in range(c):
    ax1.plot([-0.1, 9.4], [k, k], color="#ececea", lw=12, solid_capstyle="butt", zorder=0)
    ax1.text(-0.2, k, f"slot {k + 1}", ha="right", va="center", fontsize=10)
ax1.text(-0.2, c, "queue", ha="right", va="center", fontsize=10, color="#555")
for i, (a, s, t0, k) in enumerate(zip(arr, res, start, lane)):
    ax1.plot([a, a], [c - 0.35, c + 0.35], color="black", lw=1.6)        # arrival tick, on the queue row
    if t0 > a + 1e-9:
        ax1.add_patch(Rectangle((a, c - 0.25), t0 - a, 0.5, fc=C_WAIT, ec="none"))
        ax1.text(a + (t0 - a) / 2, c, f"#{i + 1}", color="white", ha="center", va="center", fontsize=8.5)
        ax1.plot([t0, t0], [c - 0.25, k + 0.42], color="#777", lw=0.8, ls=":")
    ax1.add_patch(Rectangle((t0, k - 0.42), s, 0.84, fc=C_RUN, ec="white", lw=1))
    ax1.text(t0 + s / 2, k, f"#{i + 1}", color="white", ha="center", va="center", fontsize=10, fontweight="bold")
i = next(j for j, w in enumerate(wait) if w > 0.05)
ax1.annotate(f"#{i + 1} arrives, all {c} slots taken → waits in the queue until #{lane[i] and 1 or 1} frees slot {lane[i] + 1}",
             xy=(arr[i], c + 0.3), xytext=(arr[i] + 0.8, c + 1.1), fontsize=10,
             arrowprops=dict(arrowstyle="->", color="#444"), va="center")
ax1.text(9.4, -0.65, "tick = arrival;  grey = simulated wait;  blue = measured residence, placed by the rule",
         fontsize=9.5, ha="right", color="#444")

# ---- output vs measured
ax2 = fig.add_subplot(gs[2])
ax2.set_xlim(-0.3, 9.5)
ax2.set_ylim(0, 1)
ax2.axis("off")
ax2.add_patch(FancyBboxPatch((-0.2, 0.05), 4.6, 0.9, boxstyle="round,pad=0.1", fc=C_OUT, ec="#c2410c", lw=1.4))
ax2.text(0.0, 0.72, "OUTPUT — simulated wait per request", fontsize=12, fontweight="bold", color="#7c2d12", va="center")
ax2.text(0.0, 0.3, "mean over the 300 requests of the cell", fontsize=10.5, va="center")
ax2.add_patch(FancyBboxPatch((5.2, 0.05), 4.2, 0.9, boxstyle="round,pad=0.1", fc="#f4f4f1", ec="#444", lw=1.4))
ax2.text(5.4, 0.72, "COMPARE with the measured wait", fontsize=12, fontweight="bold", va="center")
ax2.text(5.4, 0.3, "vLLM's request_queue_time on the decode engine, same cell", fontsize=10.5, va="center")
ax2.add_patch(FancyArrowPatch((4.45, 0.5), (5.15, 0.5), arrowstyle="<->", mutation_scale=16, color="#444", lw=1.6))
fig.suptitle("Re-simulating our explanation of the waiting: a measured result replayed through \"c slots, FCFS\" "
             "— not a predictive model", fontsize=13, y=0.98)
fig.savefig(REPO / "figures" / "kv_resim_schematic.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_resim_schematic.png")
