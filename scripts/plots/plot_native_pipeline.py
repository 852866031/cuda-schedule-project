#!/usr/bin/env python3
"""The two failure cases of the hand-built (P2pNcclConnector) split, as swimlanes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, BLUEF = "#1f6feb", "#dbe7fb"
COP, COPF = "#c1440e", "#f7e3d8"
GREY = "#57606a"


def lanes(ax, rows):
    for name, y in rows:
        ax.axhline(y + 7, color="#dddddd", lw=0.8, zorder=0)
        ax.text(0.5, y + 3, name, fontsize=11, fontweight="bold", va="center", color=GREY)


def box(ax, x, y, w, h, txt, fc, ec, fs=9):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=4)


def arrow(ax, x0, y0, x1, y1, color, txt="", tx=0, ty=0, fs=8.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.6, linestyle=ls, zorder=5))
    if txt:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, txt, fontsize=fs, color=color,
                ha="center", zorder=6)


# ============ case 1: DRAM for decode only -> returning sessions recompute ============
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 130); ax.set_ylim(0, 56); ax.axis("off")
lanes(ax, [("client / router", 46), ("GPU0 · prefill", 32),
           ("decode pinned pool\n(host DRAM)", 18), ("GPU1 · decode", 4)])

ax.text(15, 53.5, "① first request of session S", fontsize=10, color=GREY)
ax.text(72, 53.5, "② session S returns (evicted from GPU0's ~9-session VRAM cache)",
        fontsize=10, color=GREY)
ax.axvline(68, color="#cccccc", lw=0.9, ls="--")

box(ax, 17, 32, 17, 6, "prefill prefix + suffix\n~530 ms", BLUEF, BLUE, fs=8.6)
arrow(ax, 21, 38, 21, 45, BLUE, "token #1", tx=6, fs=9)
arrow(ax, 34, 34, 41, 34, COP, "push 0.86 GB over NCCL,\nlayer by layer ×32", tx=1.5, ty=4.2)
box(ax, 41, 30.5, 13, 6, "sent to GPU1", COPF, COP)
arrow(ax, 47, 30.5, 47, 25, COP)
box(ax, 37, 18, 22, 6.4, "spill: full private copy\n1.0 GiB × EVERY in-flight request",
    COPF, COP, fs=8.3)
arrow(ax, 48, 18, 48, 11, COP, "reload\nat admission", tx=8)
box(ax, 41, 4, 15, 6, "decode 128 steps", BLUEF, BLUE)

box(ax, 72, 32, 19, 6, "recompute the ENTIRE\nprefix again — 530 ms", COPF, COP, fs=8.6)
ax.text(104, 45.5, "nothing below GPU0 holds session KV:\nthe pool is per-request, decode-side only",
        fontsize=8.5, color=COP, ha="center")
arrow(ax, 76, 38, 76, 45, BLUE, "", fs=9)
ax.text(70, 47.8, "token #1 (slow)", fontsize=9, color=BLUE, ha="left")
arrow(ax, 91, 34, 97, 34, COP, "push 0.86 GB\nagain", tx=1.5, ty=4.0)
box(ax, 97, 30.5, 13, 6, "sent to GPU1", COPF, COP)
arrow(ax, 103, 30.5, 103, 25, COP)
box(ax, 94, 18, 20, 6.4, "another full private copy", COPF, COP, fs=8.5)
arrow(ax, 104, 18, 104, 11, COP)
box(ax, 97, 4, 15, 6, "decode 128 steps", BLUEF, BLUE)

ax.text(65, 0.2, "Measured: ~31% of requests recomputed; prefill-leg p95 reached 5.7 s. "
        "The pool cannot help — it is keyed per request, not per session, and only the decode side may read it.",
        fontsize=9.5, color=GREY, ha="center")
fig.savefig("figures/split_native_recompute.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("wrote figures/split_native_recompute.png")

# ============ case 2: bolt on a prefill tier -> the same bytes live in five places ============
GRN, GRNF = "#2e7d4f", "#ddeee4"      # decode-process ownership
SHR, SHRF = "#8250df", "#ece4f8"      # shared between the two processes

fig2, ax2 = plt.subplots(figsize=(13, 7.2))
ax2.set_xlim(0, 130); ax2.set_ylim(0, 72); ax2.axis("off")
ax2.text(65, 69.5, "Where one hot session's 768 MiB of prefix KV lives "
         "(Act I native split + bolted-on prefill DRAM tier — LMCache appears nowhere here)",
         fontsize=11, ha="center", fontweight="bold", color=GREY)

def copybox(x, y, w, h, title, sub, n, fc, ec, fs=8.8):
    ax2.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
    ax2.text(x + (w - 5) / 2, y + h / 2, f"{title}\n{sub}", ha="center", va="center",
             fontsize=fs, zorder=4)
    ax2.add_patch(Rectangle((x + w - 4.5, y + h - 2.6), 4.5, 2.6, facecolor=ec,
                            edgecolor=ec, zorder=5))
    ax2.text(x + w - 2.25, y + h - 1.3, n, color="white", fontsize=9,
             fontweight="bold", ha="center", va="center", zorder=6)

# legend: ownership = color
for lx, c, cf, t in ((6, BLUE, BLUEF, "owned by the prefill engine process"),
                     (48, GRN, GRNF, "owned by the decode engine process"),
                     (90, SHR, SHRF, "shared transit (not a store)")):
    ax2.add_patch(Rectangle((lx, 63.2), 3, 2.2, facecolor=cf, edgecolor=c, lw=1.3))
    ax2.text(lx + 4, 64.3, t, fontsize=8.8, va="center", color=GREY)

# ---- three PHYSICAL locations
ax2.add_patch(Rectangle((4, 22), 30, 38, fill=False, edgecolor=GREY, lw=1.2))
ax2.text(19, 57.3, "GPU0 VRAM (32 GB)", fontsize=10.5, color=GREY, fontweight="bold",
         ha="center")
copybox(7, 42, 24, 9, "vLLM prefix cache", "~9 hot sessions\n(6.7 GiB of KV)", "①",
        BLUEF, BLUE)

ax2.add_patch(Rectangle((40, 10), 50, 50, fill=False, edgecolor=GREY, lw=1.2))
ax2.text(65, 57.3, "host DRAM — ONE physical 60 GiB", fontsize=10.5, color=GREY,
         fontweight="bold", ha="center")
copybox(43, 42, 44, 9, "DRAM tier: durable session-prefix store",
        "12 GiB, pinned by the prefill process", "②", BLUEF, BLUE)

# ③ is a PIPE, not a store: an open conduit the bytes flow through, ~64 MB in flight.
# Drawn as a tube with open ends rather than a box, so it cannot be read as a peer of
# the two durable stores above and below it.
PIPE_Y0, PIPE_Y1, PIPE_X0, PIPE_X1 = 31.5, 35.5, 42, 88
for yy in (PIPE_Y0, PIPE_Y1):
    ax2.plot([PIPE_X0, PIPE_X1], [yy, yy], color=SHR, lw=2.2, zorder=3)
ax2.fill_between([PIPE_X0, PIPE_X1], PIPE_Y0, PIPE_Y1, color=SHRF, alpha=0.6, zorder=2)
for xx in (PIPE_X0, PIPE_X1):   # dashed open ends
    ax2.plot([xx, xx], [PIPE_Y0, PIPE_Y1], color=SHR, lw=1.2, ls=(0, (2, 2)), zorder=3)
# flow THROUGH the pipe
ax2.annotate("", xy=(84, 33.5), xytext=(46, 33.5),
             arrowprops=dict(arrowstyle="-|>", color=SHR, lw=1.8,
                             linestyle=(0, (4, 3))), zorder=4)
ax2.text(60, 30.6, "③ NCCL staging — a pipe, not a store\n(~64 MB of /dev/shm, mapped by "
         "both processes; bytes only pass through)", fontsize=8.3, color=SHR, ha="center",
         va="top", zorder=4)

copybox(43, 13, 44, 12, "transfer pool: durable per-request store",
        "16 GiB, pinned by the decode process\n1.0 GiB × each in-flight request, undeduplicated",
        "④", GRNF, GRN)

ax2.add_patch(Rectangle((96, 22), 30, 38, fill=False, edgecolor=GREY, lw=1.2))
ax2.text(111, 57.3, "GPU1 VRAM (32 GB)", fontsize=10.5, color=GREY, fontweight="bold",
         ha="center")
copybox(99, 42, 24, 9, "working KV of the", "running request\n(0.8 GiB)", "⑤",
        GRNF, GRN)

# flow arrows
arrow(ax2, 31, 46.5, 43, 46.5, BLUE)
ax2.text(32, 44.6, "saved to the tier", fontsize=8.5, color=BLUE, ha="left")
arrow(ax2, 26, 42, 42, 33.5, COP)
ax2.text(30.5, 33.8, "pushed,\nlayer by layer", fontsize=8.5, color=COP, ha="center")
arrow(ax2, 88, 33.5, 87, 25, COP)
ax2.text(90.5, 29.0, "spill", fontsize=8.5, color=COP, ha="left")
arrow(ax2, 87, 20, 99, 42, COP)
ax2.text(99, 17.5, "reloaded at admission", fontsize=8.5, color=COP, ha="center")

ax2.text(65, 6.8, "RAM as configured: 12 (②) + 16 (④) + 1 (producer's unused pool) = 29 GiB pinned "
         "+ ~13 GiB engine working sets ≈ 42 of 60 GiB — holding ≈ 3 GiB of distinct hot KV.",
         fontsize=9.5, color=COP, ha="center")
ax2.text(65, 3.2, "② and ④ are NOT copies of each other's structure — ② stores prefixes per session, "
         "④ stores full prompts per request — but a hot session's bytes sit in both (and in ④ once per "
         "concurrent request).", fontsize=9, color=GREY, ha="center")
ax2.text(65, 0.2, "They cannot be merged as-is: pinning, allocator metadata, and content keys are all "
         "per-process. Act II replaces ②+③+④ with one content-addressed cache in a separate LMCache "
         "server process.", fontsize=9, color=GREY, ha="center")
fig2.savefig("figures/split_native_copies.png", dpi=140, bbox_inches="tight",
             facecolor="white")
print("wrote figures/split_native_copies.png")

# ============ companion: the SAME memory-layout question, Act II (LMCache) ============
fig3, ax3 = plt.subplots(figsize=(13, 6.4))
ax3.set_xlim(0, 130); ax3.set_ylim(0, 64); ax3.axis("off")
LMC, LMCF = "#c1440e", "#f7e3d8"
ax3.text(65, 61.5, "Act II memory layout: one shared LMCache", fontsize=12,
         ha="center", fontweight="bold", color=GREY)

def cbox3(x, y, w, h, txt, n, fc, ec, fs=8.8):
    ax3.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
    ax3.text(x + (w - 4.5) / 2, y + h / 2, txt, ha="center", va="center",
             fontsize=fs, zorder=4)
    ax3.add_patch(Rectangle((x + w - 4.5, y + h - 2.6), 4.5, 2.6, facecolor=ec,
                            edgecolor=ec, zorder=5))
    ax3.text(x + w - 2.25, y + h - 1.3, n, color="white", fontsize=9,
             fontweight="bold", ha="center", va="center", zorder=6)

for lx, c, cf, t in ((14, BLUE, BLUEF, "prefill process"),
                     (40, GRN, GRNF, "decode process"),
                     (66, LMC, LMCF, "LMCache server (3rd process, reached over TCP)")):
    ax3.add_patch(Rectangle((lx, 56.2), 3, 2.2, facecolor=cf, edgecolor=c, lw=1.3))
    ax3.text(lx + 4, 57.3, t, fontsize=8.6, va="center", color=GREY)

ax3.add_patch(Rectangle((4, 10), 30, 42, fill=False, edgecolor=GREY, lw=1.2))
ax3.text(19, 49.3, "GPU0 VRAM", fontsize=10.5, color=GREY, fontweight="bold", ha="center")
cbox3(7, 36, 24, 9, "vLLM prefix cache\n~9 sessions, 6.7 GiB", "①", BLUEF, BLUE)

ax3.add_patch(Rectangle((40, 4), 50, 48, fill=False, edgecolor=GREY, lw=1.2))
ax3.text(65, 49.3, "host DRAM (one physical 60 GiB)", fontsize=10.5, color=GREY,
         fontweight="bold", ha="center")
cbox3(43, 32, 44, 12, "THE store — each session's prefix ONCE\n≤ 24 GiB, unpinned, "
      "content-addressed", "②", LMCF, LMC)
cbox3(43, 17, 20, 7, "prefill L1\n6 GiB pinned", "③", BLUEF, BLUE, fs=8.5)
cbox3(67, 17, 20, 7, "decode L1\n6 GiB pinned", "④", GRNF, GRN, fs=8.5)

ax3.add_patch(Rectangle((96, 10), 30, 42, fill=False, edgecolor=COP, lw=2.0))
ax3.text(111, 49.3, "GPU1 VRAM", fontsize=10.5, color=GREY, fontweight="bold", ha="center")
cbox3(99, 36, 24, 9, "running requests' KV\n0.8 GiB each", "⑤", GRNF, GRN)
ax3.text(111, 15.5, "★ swept: 30 → 18 GiB", fontsize=10, color=COP, ha="center",
         fontweight="bold")

# prefill side
arrow(ax3, 31, 42, 43, 41, COP)
arrow(ax3, 43, 37, 31, 38, BLUE)
ax3.text(19, 31.5, "→ store once (first request)", fontsize=8.4, color=COP, ha="center")
ax3.text(19, 24.5, "← retrieve on miss — BLOCKING:\nno compute/transfer overlap,\n"
         "the wait lands in TTFT", fontsize=8.4, color=BLUE, ha="center")

# decode side
arrow(ax3, 77, 32, 77, 24, COP)
ax3.text(75, 27.5, "write\nto L1", fontsize=8.4, color=COP, ha="right")
# double-headed: chunks stream INTO the L1 (from the store) while earlier chunks are
# copied OUT to VRAM -- the two directions run concurrently, pipelined
ax3.add_patch(FancyArrowPatch((87, 21.5), (99, 39), arrowstyle="<|-|>",
                              mutation_scale=13, color=COP, lw=1.8, zorder=5))
ax3.text(100.5, 31.5, "streams in & out\nconcurrently", fontsize=8.3, color=COP,
         ha="left")
ax3.text(65, 11.0, "store → L1 → VRAM: the transfers stream concurrently\nand overlap "
         "decode compute (ITL p95 ≈ 12 ms)", fontsize=8.5, color=GRN, ha="center")

ax3.text(65, 0.8, "One durable copy per session; the L1s hold bounded transient copies. "
         "RAM ≈ 24 + 2×6 + 13 ≈ 49 of 60 GiB.", fontsize=9, color=GREY, ha="center")
fig3.savefig("figures/split_lmcache_memory.png", dpi=140, bbox_inches="tight",
             facecolor="white")
print("wrote figures/split_lmcache_memory.png")
