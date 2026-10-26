#!/usr/bin/env python3
"""KV-slot study: the whole decode-side mechanism in one picture, drawn on the physical
layout (GPU1 VRAM above, host RAM below, PCIe between), two panels:

  left   today: LMCache 0.4.4 connector, synchronous load (what E1/E2 measured)
  right  the asynchronous-load alternative, dashed (vLLM has the hook -- a request can
         sit in WAITING_FOR_REMOTE_KVS while its KV is copied -- E4 tests whether the
         LMCache options reach it)

Shown in both: the KV block pool as one bar of 3,980 blocks (24 GiB) holding the running
requests; requests grow 1 block per 16 tokens until they hit the wall -> the most
recently admitted one is evicted back to the HEAD of the queue; a new request is admitted
only if 392 blocks are free, otherwise it queues; admission moves its 0.75 GiB of KV from
host RAM over PCIe into the blocks. Under each panel: the engine thread's timeline.

    .venv/bin/python scripts/plots/plot_kv_slot_mechanism.py  -> figures/kv_slot_mechanism.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
C_RUN, C_NEW, C_WAIT, C_PRE = "#2a78d6", "#eb6834", "#9a9a95", "#b3261e"
C_GPU, C_HOST, C_PCIE = "#eef3f9", "#f3f1ea", "#6b6b66"

fig, axes = plt.subplots(1, 2, figsize=(19, 10.5))
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.02)


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def txt(ax, x, y, s, fs=10, color="black", ha="center", va="center", bold=False, style="normal",
        bg=None):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, linespacing=1.3,
            fontweight="bold" if bold else "normal", style=style,
            bbox=dict(fc=bg, ec="none", pad=1.2) if bg else None)


def arrow(ax, p, q, color, lw=1.6, ls="-", style="-|>", ms=14, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def panel(ax, async_mode):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    dash = "--" if async_mode else "-"

    # ------------------------------------------------------------ GPU1
    rbox(ax, 2, 50, 96, 46, C_GPU, "#8aa4c4", lw=1.6)
    txt(ax, 4, 94, "GPU1 — 32 GB VRAM, decode engine at 24 GiB", 11.5, "#0d366b", ha="left", bold=True)
    rbox(ax, 4, 58, 14, 32, "#d9e3f0", "#8aa4c4")
    txt(ax, 11, 74, "model\nweights\n16 GiB\n(read once\nper step)", 9.5, "#0d366b")
    PX0, PX1, PY, PH = 21, 95, 74, 13
    scale = (PX1 - PX0) / 4000.0
    rbox(ax, PX0, PY, (PX1 - PX0) * 3980 / 4000, PH, "white", "#444", lw=1.4, r=0.6)
    txt(ax, (PX0 + PX1) / 2, 90.5, "KV block pool: 3,980 blocks of 16 tokens (7.77 GiB) = 9.95 whole requests",
        10, "#222", bold=True)
    x = PX0 + 0.4
    held = [400, 400, 398, 396, 395, 394, 393, 392, 392]
    for i, b in enumerate(held):
        w = b * scale - 0.5
        ax.add_patch(Rectangle((x, PY + 1), w, PH - 2, fc=C_RUN, ec="white", lw=0.8))
        txt(ax, x + w / 2, PY + PH / 2 + 1.5, f"#{i + 1}", 8.5, "white", bold=True)
        txt(ax, x + w / 2, PY + PH / 2 - 2.3, f"{b}", 7.5, "white")
        x += b * scale
    w10 = 392 * scale - 0.5
    if async_mode:
        ax.add_patch(Rectangle((x, PY + 1), w10, PH - 2, fc="white", ec=C_NEW, lw=1.4, ls="--",
                               hatch="///"))
        txt(ax, x + w10 / 2, PY + PH / 2 + 1.5, "#10", 8.5, C_NEW, bold=True)
        txt(ax, x + w10 / 2, PY + PH / 2 - 2.3, "filling", 7, C_NEW)
    else:
        ax.add_patch(Rectangle((x, PY + 1), w10, PH - 2, fc=C_NEW, ec="white", lw=0.8))
        txt(ax, x + w10 / 2, PY + PH / 2 + 1.5, "#10", 8.5, "white", bold=True)
        txt(ax, x + w10 / 2, PY + PH / 2 - 2.3, "392", 7.5, "white")
    xs10 = x
    wall = PX0 + 3980 * scale
    ax.plot([wall, wall], [PY - 2.5, PY + PH + 2.5], color=C_PRE, lw=2.5)
    txt(ax, wall + 0.8, PY + PH + 1.5, "wall", 9, C_PRE, ha="left", bold=True)
    txt(ax, wall - 0.5, PY - 4.3, f"free: {3980 - sum(held) - 392} blocks", 8.5, "#444", ha="right")
    txt(ax, PX0 + 0.5, PY - 4.3, "each running request takes 1 more block per 16 generated tokens "
        "(392 → 400) ——→", 8.5, C_RUN, ha="left")
    txt(ax, PX0, 60.5, "CASE 2  out of blocks while growing → EVICT the most recently admitted running "
        "request (#9):\nits blocks are freed, it goes to the HEAD of the queue and its KV must be "
        "RELOADED later.\n24 GiB: 9–35 evictions per 300 requests; none at 26 / 28 GiB (the spare "
        "0.51 / 0.07 request is too small to admit into)", 8.6, C_PRE, ha="left", bg=C_GPU)
    ex = PX0 + (sum(held[:8]) + 196) * scale

    # ------------------------------------------------------------ between: PCIe
    txt(ax, 44, 42.3, "PCIe Gen4 ×8 — 14.5 GB/s measured; the LMCache copy path reaches ~4 GB/s",
        9, C_PCIE, bold=True)
    arrow(ax, (61, 36.5), (xs10 + w10 / 2, PY - 0.3), C_NEW, lw=3.2, ls=dash, ms=22)
    txt(ax, 97.5, 46.8,
        ("SYNC: 0.75 GiB copied inside\nthe admitting step (~200 ms,\np95 300 ms); nothing else runs"
         if not async_mode else
         "ASYNC: 0.75 GiB copied in the\nbackground; #10 joins the batch\nonly once it is loaded"),
        8.5, C_NEW, ha="right", bold=True)
    arrow(ax, (10.3, 29.5), (xs10 + 1.5, PY - 0.3), "#333", lw=1.8, ls=dash, rad=-0.12)
    txt(ax, 14, 48, "ADMIT\n(≥ 392 blocks free)", 8.5, "#333", ha="left")
    arrow(ax, (ex, PY + 0.5), (34, 29.5), C_PRE, lw=2, ls="--", rad=0.22)
    txt(ax, 36, 40.5, "EVICT", 8.5, C_PRE, ha="left", bold=True)

    # ------------------------------------------------------------ host RAM + CPU
    rbox(ax, 2, 1, 96, 40, C_HOST, "#b9b3a1", lw=1.6)
    txt(ax, 4, 39, "host — 32 cores, 60 GiB RAM", 11.5, "#4a4639", ha="left", bold=True)
    rbox(ax, 4, 14, 44, 22, "white", "#b9b3a1")
    txt(ax, 26, 34, "EngineCore: scheduler's waiting queue (FCFS)", 9.5, "#222", bold=True)
    qx = 7
    for i, lab in enumerate(["head", "", "", "", ""]):
        c = C_WAIT if i else "#6b6b66"
        ax.add_patch(Rectangle((qx, 24), 6.5, 5.5, fc=c, ec="white", lw=0.8))
        txt(ax, qx + 3.25, 26.75, lab or "…", 8, "white", bold=bool(lab))
        qx += 7.2
    arrow(ax, (47, 26.75), (qx + 0.5, 26.75), "#333", lw=1.4)
    txt(ax, 47, 31.5, "arrivals ← prefill GPU0 (token #1 already sent)", 7.8, "#333", ha="right")
    txt(ax, 26, 18.6, "CASE 1  the head needs ⌈tokens/16⌉ blocks (392 fresh, 393+ if evicted)\n"
        "free → ADMIT and load its KV;  not free → it and all behind it wait\n"
        "(= measured decode queue time: 0 ms at 1 req/s … 21 s at 4 req/s)", 8, "#222")
    rbox(ax, 51, 14, 45, 22, "white", "#b9b3a1")
    txt(ax, 73.5, 34, "LMCache: the KV of all 32 session prefixes (24 GiB)", 9.5, "#222", bold=True)
    rbox(ax, 53, 16, 17, 15, "#fde9dd", C_NEW, lw=1.2)
    txt(ax, 61.5, 23.5, "L1 pool\n6 GiB pinned\n(per engine)", 9, "#7c2d12")
    rbox(ax, 74, 16, 20, 15, "#fff3eb", C_NEW, lw=1.2, ls="--")
    txt(ax, 84, 23.5, "store process\n24 GiB in\n256-token chunks\n(TCP, Python copy)", 8.6, "#7c2d12")
    arrow(ax, (74, 23.5), (70.2, 23.5), C_NEW, lw=1.6)
    txt(ax, 72.1, 26.2, "miss", 7.5, C_NEW)

    # engine-thread timeline (it runs on a host core)
    TY, TH = 5.5, 4.2
    txt(ax, 4, 12, "host CPU — the EngineCore thread (one step = one forward pass over the whole batch; "
        "≈ 15 ms each):", 9, "#222", ha="left", bold=True)
    if not async_mode:
        seq = [("step", 16, C_RUN)] * 3 + [("LOAD #10 — all 9 running wait", 200, C_NEW)] + \
              [("step +#10", 34, C_RUN)] + [("step", 16, C_RUN)] * 2
    else:
        seq = [("step", 16, C_RUN)] * 13 + [("step +#10", 34, C_RUN)] + [("step", 16, C_RUN)] * 2
    tot = sum(v[1] for v in seq)
    sc = 90 / tot
    x = 5
    for lab, ms, c in seq:
        w = ms * sc
        ax.add_patch(Rectangle((x, TY), w - 0.25, TH, fc=c, ec="white", lw=0.6,
                               alpha=1 if c == C_NEW else 0.85))
        if ms >= 30:
            txt(ax, x + w / 2, TY + TH / 2, lab, 8 if ms > 100 else 7, "white", bold=ms > 100)
        x += w
    if async_mode:
        ax.add_patch(Rectangle((5, 2.4), 200 * sc, 1.6, fc="white", ec=C_NEW, lw=1.2, ls="--",
                               hatch="////"))
        txt(ax, 5 + 200 * sc + 1, 3.2, "#10's KV copied on a side stream;\nthe steps continue",
            8, C_NEW, ha="left")
    else:
        txt(ax, 5, 3.2, "the load call blocks this thread for ~200 ms: 34% of all decode wall time at "
            "24 GiB, 2 req/s", 8, C_NEW, ha="left")


panel(axes[0], False)
panel(axes[1], True)
axes[0].set_title("TODAY — LMCache 0.4.4 connector: the KV load is synchronous (measured, E1/E2)",
                  fontsize=12.5, fontweight="bold", color="#222", pad=8)
axes[1].set_title("POSSIBLE — asynchronous load (vLLM has the hook: WAITING_FOR_REMOTE_KVS; "
                  "E4 tests it)", fontsize=12.5, fontweight="bold", color="#7c2d12", pad=8)
fig.suptitle("How the decode engine's KV slots fill, stall, evict and queue — Llama-3-8B decode "
             "on GPU1, 6,400-token requests (prefix 6,144 from LMCache + 128 new + 128 generated)",
             fontsize=13.5, y=0.975)
fig.savefig(REPO / "figures" / "kv_slot_mechanism.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_slot_mechanism.png")
