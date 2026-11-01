#!/usr/bin/env python3
"""KV-slot study: what the decode scheduler decides in one step (vLLM 0.15.1 v1
scheduler + LMCache connector, as read from the source and confirmed by the E1/E2
traces). Two situations share one pool of free KV blocks:

  CASE 2  running requests grow (1 block per 16 tokens); out of blocks -> PREEMPT the
          most recently admitted running request (back to the HEAD of the queue);
  CASE 1  the head of the waiting queue needs its whole prompt's blocks; admitted only
          if they are free, else it and everyone behind it wait. Admission triggers
          the synchronous KV load that stalls the whole batch.

    .venv/bin/python scripts/plots/plot_kv_sched_logic.py   -> figures/kv_sched_logic.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent.parent
C_RUN, C_WAIT, C_LOAD, C_PRE, C_POOL = "#2a78d6", "#57606a", "#c2410c", "#b3261e", "#1a7a55"

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 140)
ax.set_ylim(-4, 100)
ax.axis("off")


def box(x, y, w, h, text, fc="#f4f4f1", ec="#444", lw=1.2, fs=10.5, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.35)


def diamond(x, y, w, h, text, fs=10):
    ax.add_patch(plt.Polygon([(x, y + h / 2), (x + w / 2, y + h), (x + w, y + h / 2),
                              (x + w / 2, y)], closed=True, fc="#fff8e6", ec="#444", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def line(pts, color="#333", arrow=True, ls="-", lw=1.4):
    for a, b in zip(pts[:-1], pts[1:]):
        last = b is pts[-1]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if (arrow and last) else "-",
                                     mutation_scale=13, color=color, lw=lw, ls=ls,
                                     shrinkA=0, shrinkB=0))


def label(x, y, text, color="#333", fs=9.5, ha="center", style="normal", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style,
            fontweight=weight, bbox=dict(fc="white", ec="none", pad=1.0), linespacing=1.3)


# ---------------------------------------------------------------- title + pool
ax.text(70, 97.5, "One decode-scheduler step: two ways to run out of KV blocks",
        ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(70, 93.5, "block = 16 tokens; a 6,400-token request holds 392 blocks when admitted and "
        "400 when finished", ha="center", va="center", fontsize=11, color="#444")
box(40, 82, 60, 8,
    "ONE POOL OF FREE KV BLOCKS  (24 GiB: 3,980 = 9.95 requests)\n"
    "every allocation below takes from it; finished and preempted requests give theirs back",
    fc="#e3f1ea", ec=C_POOL, lw=1.6, fs=10.5, bold=True, tc="#0f4d35")

# ---------------------------------------------------------------- left: case 2
LX = 3
box(LX, 70, 40, 8, "CASE 2 — requests already running\n(each gets 1 new token per step)",
    fc="#e4eefb", ec=C_RUN, lw=1.6, fs=11, bold=True, tc="#0d366b")
diamond(LX + 2, 52, 36, 13, "does it need one more block\nthis step?\n(every 16th generated token)")
diamond(LX + 2, 33, 36, 12, "is one block free?")
box(LX, 10, 40, 16,
    "PREEMPT the most recently admitted\nrunning request\n"
    "free all its blocks · computed tokens := 0\n"
    "move it to the HEAD of the waiting queue\n(then retry this check)",
    fc="#fbe9e7", ec=C_PRE, lw=1.6, fs=10, tc="#7a1a12")
line([(LX + 20, 70), (LX + 20, 65)])
line([(LX + 20, 52), (LX + 20, 45)])
label(LX + 23.5, 48.5, "yes")
line([(LX + 20, 33), (LX + 20, 26)], color=C_PRE)
label(LX + 23, 29.5, "no", color=C_PRE)
line([(LX, 18), (LX - 2, 18), (LX - 2, 58.5), (LX + 2, 58.5)], color=C_PRE, ls="--")
label(LX - 2, 40, "retry", color=C_PRE, fs=9)
ax.text(LX + 20, 5.5, "when it comes back: its 6,144 prefix tokens are LOADED AGAIN and the\n"
        "suffix + every token it had already generated are recomputed",
        ha="center", va="center", fontsize=9.5, color="#7a1a12", style="italic", linespacing=1.3)

# ---------------------------------------------------------------- right: case 1
RX = 97
box(RX, 70, 40, 8, "CASE 1 — the waiting queue (FCFS)\nonly the HEAD request is examined",
    fc="#eceeef", ec=C_WAIT, lw=1.6, fs=11, bold=True, tc="#2b3136")
diamond(RX + 2, 52, 36, 13, "are ⌈tokens / 16⌉ blocks free?\nfresh request: 392\n"
        "preempted one: 393 or more")
box(RX, 33, 40, 12,
    "ADMIT: take the blocks; the LMCache lookup\nfinds 6,144 of its 6,272 prompt tokens\n"
    "→ then examine the next head request",
    fc="#e4eefb", ec=C_RUN, lw=1.4, fs=9.5, tc="#0d366b")
box(RX, 10, 40, 16,
    "WAIT: the head request stays queued,\nand so does everyone behind it\n\n"
    "= the measured decode queue time\n(24 GiB: 0 ms at 1 req/s … 21 s at 4 req/s)",
    fc="#eceeef", ec=C_WAIT, lw=1.6, fs=10, tc="#2b3136")
line([(RX + 20, 70), (RX + 20, 65)])
line([(RX + 20, 52), (RX + 20, 45)])
label(RX + 23.5, 48.5, "yes")
line([(RX + 38, 58.5), (RX + 42, 58.5), (RX + 42, 18), (RX + 40, 18)], color=C_WAIT)
label(RX + 42, 40, "no", color=C_WAIT, fs=9)

# ---------------------------------------------------------------- centre: the step
CX, CW = 45, 50
box(CX, 42, CW, 14,
    "START_LOAD_KV — synchronous, engine thread\n"
    "copies each admitted request's 6,144-token KV\n"
    "(0.75 GiB) from host memory into its blocks\n"
    "~200 ms (p95 300 ms); every running request\n"
    "waits: 34% of decode wall time (24 GiB, 2 req/s)",
    fc="#fdeee7", ec=C_LOAD, lw=1.8, fs=9.5, bold=True, tc="#7c2d12")
box(CX, 24, CW, 12,
    "ONE FORWARD PASS over the whole batch\n"
    "1 token for every running request\n+ 128 suffix tokens for each newly admitted request",
    fc="#e4eefb", ec=C_RUN, lw=1.6, fs=10.5, tc="#0d366b")
ax.text(CX + CW / 2, 19.5, "11–19 ms per step (3 → 14 running)\nthen the next step begins again at CASE 2",
        ha="center", va="center", fontsize=9.5, color="#0d366b", style="italic", linespacing=1.3)
line([(CX + CW / 2, 42), (CX + CW / 2, 36)], color=C_LOAD)
label(CX + CW / 2 + 10, 39, "only in a step that\nadmitted someone", color=C_LOAD, fs=9)

# feeds into the forward pass
line([(LX + 38, 58.5), (CX - 2, 58.5), (CX - 2, 30), (CX, 30)], color=C_RUN)
label(LX + 42, 62, "no: keeps its blocks, 1 decode token", color=C_RUN, fs=9, ha="left")
line([(LX + 38, 39), (CX - 2, 39)], color=C_RUN)
label(LX + 42, 40.8, "yes: takes 1 block", color=C_RUN, fs=9, ha="left")
line([(RX, 39), (CX + CW + 2, 39), (CX + CW + 2, 50), (CX + CW, 50)], color=C_RUN)
label(RX - 4.5, 36.5, "admitted", color=C_RUN, fs=9)
line([(CX + CW + 2, 39), (CX + CW + 2, 30), (CX + CW, 30)], color=C_RUN)

# pool links
line([(CX + CW / 2 - 30, 82), (LX + 20, 78)], color=C_POOL, arrow=False, ls=":", lw=1)
line([(CX + CW / 2 + 30, 82), (RX + 20, 78)], color=C_POOL, arrow=False, ls=":", lw=1)

fig.savefig(REPO / "figures" / "kv_sched_logic.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_sched_logic.png")
