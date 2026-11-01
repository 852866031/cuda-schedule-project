#!/usr/bin/env python3
"""KV-slot study: the two ways the decode engine runs out of KV blocks, one figure each,
drawn on the physical layout (GPU1 VRAM above, host RAM + the EngineCore thread below,
PCIe between). Numbers are the 24 GiB cells of E1/E2.

  kv_case1_admission.png   a new request: admitted only if 392 blocks are free (else it
                           and everyone behind it in the FCFS queue wait); admission
                           copies its 0.75 GiB of KV from host RAM -- synchronously today
                           (left), asynchronously in the dashed alternative (right)
  kv_case2_eviction.png    running requests grow 1 block / 16 tokens; when the pool is
                           exhausted the most recently admitted one is EVICTED to the
                           head of the queue (left), and comes back only after a slot
                           frees, with its KV loaded again (right)

    .venv/bin/python scripts/plots/plot_kv_cases.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
import os as _os
_M = _os.environ.get("KV_METHOD", "base")
# measured numbers per LMCache copy path (24 GiB): fallback (base) vs compiled kernels (cops)
N = {"base": dict(bw="~4 GB/s", load=200, share="34%", load_rate="2 req/s", reload_from="0.3 s", reload_to="0.9–1.4 s"),
     "cops": dict(bw="~7 GB/s", load=115, share="20%", load_rate="2 req/s", reload_from="0.18 s", reload_to="0.45–0.8 s")}[_M]
C_RUN, C_NEW, C_WAIT, C_PRE = "#2a78d6", "#eb6834", "#9a9a95", "#b3261e"
C_GPU, C_HOST, C_PCIE = "#eef3f9", "#f3f1ea", "#6b6b66"
PX0, PX1, PY, PH = 21, 95, 74, 13
SCALE = (PX1 - PX0) / 4000.0
POOL = 3980


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, r=1.2, ls="-", hatch=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, hatch=hatch))


def txt(ax, x, y, s, fs=10, color="black", ha="center", va="center", bold=False, style="normal",
        bg=None):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, linespacing=1.3,
            fontweight="bold" if bold else "normal", style=style,
            bbox=dict(fc=bg, ec="none", pad=1.2) if bg else None)


def arrow(ax, p, q, color, lw=1.6, ls="-", style="-|>", ms=14, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def base(ax, held, queue_head="head", queue_head_color="#6b6b66"):
    """GPU box with the block pool holding `held` (list of (label, blocks, kind)), host box
    with the queue and LMCache. Returns dict of x positions per label."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    rbox(ax, 2, 50, 96, 46, C_GPU, "#8aa4c4", lw=1.6)
    txt(ax, 4, 94, "GPU1 — decode engine, 24 GiB", 11.5, "#0d366b", ha="left", bold=True)
    rbox(ax, 4, 58, 14, 32, "#d9e3f0", "#8aa4c4")
    txt(ax, 11, 74, "weights\n16 GiB", 10, "#0d366b")
    rbox(ax, PX0, PY, (PX1 - PX0) * POOL / 4000, PH, "white", "#444", lw=1.4, r=0.6)
    txt(ax, (PX0 + PX1) / 2, 90.5, f"KV pool: {POOL:,} blocks = {POOL / 400:.2f} requests of 400",
        10, "#222", bold=True)
    pos = {}
    x = PX0 + 0.4
    for lab, b, kind in held:
        w = b * SCALE - 0.5
        if kind == "run":
            ax.add_patch(Rectangle((x, PY + 1), w, PH - 2, fc=C_RUN, ec="white", lw=0.8))
            tc = "white"
        elif kind == "new":
            ax.add_patch(Rectangle((x, PY + 1), w, PH - 2, fc=C_NEW, ec="white", lw=0.8))
            tc = "white"
        elif kind == "fill":
            ax.add_patch(Rectangle((x, PY + 1), w, PH - 2, fc="white", ec=C_NEW, lw=1.4, ls="--",
                                   hatch="///"))
            tc = C_NEW
        elif kind == "gone":
            ax.add_patch(Rectangle((x, PY + 1), w, PH - 2, fc="#f6f6f4", ec=C_PRE, lw=1.2, ls=":"))
            tc = C_PRE
        pos[lab] = (x, w)
        txt(ax, x + w / 2, PY + PH / 2 + 1.5, lab, 8.5, tc, bold=True)
        txt(ax, x + w / 2, PY + PH / 2 - 2.3, str(b) if kind != "gone" else "freed", 7.5, tc)
        x += b * SCALE
    pos["_end"] = x
    wall = PX0 + POOL * SCALE
    ax.plot([wall, wall], [PY - 2.5, PY + PH + 2.5], color=C_PRE, lw=2.5)
    txt(ax, wall + 0.8, PY + PH + 1.5, "wall", 9, C_PRE, ha="left", bold=True)
    txt(ax, PX0 + 0.5, PY - 4.3, "+1 block per 16 generated tokens: 392 → 400", 8.5, C_RUN, ha="left")
    # PCIe
    txt(ax, 50, 44.5, f"PCIe Gen4 ×8: 14.5 GB/s link\n{N['bw']} LMCache copy", 9, C_PCIE, bold=True)
    # host
    rbox(ax, 2, 1, 96, 40, C_HOST, "#b9b3a1", lw=1.6)
    txt(ax, 4, 39, "host RAM", 11.5, "#4a4639", ha="left", bold=True)
    rbox(ax, 4, 14, 44, 22, "white", "#b9b3a1")
    txt(ax, 32, 34, "scheduler queue (FCFS)", 9.5, "#222", bold=True)
    qx = 7
    for i in range(5):
        c = C_WAIT if i else queue_head_color
        ax.add_patch(Rectangle((qx, 24), 6.5, 5.5, fc=c, ec="white", lw=0.8))
        txt(ax, qx + 3.25, 26.75, queue_head if i == 0 else "…", 8, "white", bold=(i == 0))
        qx += 7.2
    arrow(ax, (47, 26.75), (qx + 0.5, 26.75), "#333", lw=1.4)
    txt(ax, 47, 31.5, "arrivals from prefill (GPU0)", 7.8, "#333", ha="right")
    rbox(ax, 51, 14, 45, 22, "white", "#b9b3a1")
    txt(ax, 73.5, 34, "LMCache: KV of all 32 prefixes", 9.5, "#222", bold=True)
    rbox(ax, 53, 16, 17, 15, "#fde9dd", C_NEW, lw=1.2)
    txt(ax, 61.5, 23.5, "L1 pool\n6 GiB pinned", 9, "#7c2d12")
    rbox(ax, 74, 16, 20, 15, "#fff3eb", C_NEW, lw=1.2, ls="--")
    txt(ax, 84, 23.5, "store\n24 GiB\n(TCP)", 9, "#7c2d12")
    arrow(ax, (74, 23.5), (70.2, 23.5), C_NEW, lw=1.6)
    txt(ax, 72.1, 26.2, "miss", 7.5, C_NEW)
    txt(ax, 4, 12.8, "engine thread (host CPU): one step = one forward pass of the whole batch, ≈ 15 ms",
        9, "#222", ha="left", bold=True)
    return pos


def timeline(ax, seq, note, note_color=C_NEW, side=None):
    TY, TH = 5.5, 4.2
    tot = sum(v[1] for v in seq)
    sc = 90 / tot
    x = 5
    for lab, ms, c in seq:
        w = ms * sc
        ax.add_patch(Rectangle((x, TY), w - 0.25, TH, fc=c, ec="white", lw=0.6,
                               alpha=1 if c != C_RUN else 0.85))
        if ms >= 30:
            txt(ax, x + w / 2, TY + TH / 2, lab, 8 if ms > 100 else 7, "white", bold=ms > 100)
        elif lab and ms < 30 and c == C_PRE:
            txt(ax, x + w / 2, TY + TH + 1.3, lab, 7.5, C_PRE, bold=True)
        x += w
    if side:
        ms_side, label = side
        ax.add_patch(Rectangle((5, 2.4), ms_side * sc, 1.6, fc="white", ec=C_NEW, lw=1.2, ls="--",
                               hatch="////"))
        txt(ax, 5 + ms_side * sc + 1, 3.2, label, 8, C_NEW, ha="left")
    else:
        txt(ax, 5, 3.2, note, 8, note_color, ha="left")
    return sc


# ===================================================================== CASE 1
def case1(ax, async_mode):
    held = [(f"#{i + 1}", b, "run") for i, b in enumerate([400, 400, 398, 396, 395, 394, 393, 392, 392])]
    held.append(("#10", 392, "fill" if async_mode else "new"))
    pos = base(ax, held)
    dash = "--" if async_mode else "-"
    x10, w10 = pos["#10"]
    free_before = POOL - sum(b for _, b, k in held if k == "run")
    txt(ax, PX0, 60.5,
        "head of queue needs 392 free blocks\n"
        "free → admitted, KV copied in\n"
        "not free → it and all behind it wait\n"
        f"here: {free_before} free → #10 admitted ({free_before - 392} left)",
        8.5, "#222", ha="left", bg=C_GPU)
    # admit arrow + load arrow
    arrow(ax, (10.3, 29.5), (x10 + 1.5, PY - 0.3), "#333", lw=1.8, ls=dash, rad=-0.12)
    txt(ax, 14, 48, "ADMIT", 9, "#333", ha="left", bold=True)
    arrow(ax, (61, 36.5), (x10 + w10 / 2, PY - 0.3), C_NEW, lw=3.2, ls=dash, ms=22)
    txt(ax, 97.5, 45.5,
        (f"SYNC: 0.75 GiB copied\ninside the step, ~{N['load']} ms;\nthe 9 running get no token" if not async_mode else
         "ASYNC: copied in the\nbackground; the 9 keep\ndecoding, #10 joins\nwhen loaded"),
        8.5, C_NEW, ha="right", bold=True)
    if not async_mode:
        seq = [("step", 16, C_RUN)] * 3 + [("LOAD #10 — all 9 wait", N["load"], C_NEW)] + \
              [("step +#10", 34, C_RUN)] + [("step", 16, C_RUN)] * 2
        timeline(ax, seq, f"load stalls = {N['share']} of decode wall time (24 GiB, {N['load_rate']})")
    else:
        seq = [("step", 16, C_RUN)] * 13 + [("step +#10", 34, C_RUN)] + [("step", 16, C_RUN)] * 2
        timeline(ax, seq, "", side=(N["load"], "copy on a side stream"))


fig, axes = plt.subplots(1, 2, figsize=(19, 10.5))
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.02)
case1(axes[0], False)
case1(axes[1], True)
axes[0].set_title("today: synchronous load (LMCache 0.4.4, measured)",
                  fontsize=12.5, fontweight="bold", color="#222", pad=8)
axes[1].set_title("alternative: asynchronous load (the options tested in §6)",
                  fontsize=12.5, fontweight="bold", color="#7c2d12", pad=8)
fig.suptitle("CASE 1 — admitting a request: no slot → queue; slot → the batch stalls while its KV loads",
             fontsize=13.5, y=0.975)
fig.savefig(REPO / "figures" / "kv_case1_admission.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_case1_admission.png")


# ===================================================================== CASE 2
def case2(ax, after):
    if not after:
        held = [(f"#{i + 1}", b, "run") for i, b in
                enumerate([400, 400, 400, 399, 399, 398, 398, 397, 396, 393])]
        pos = base(ax, held)
        x10, w10 = pos["#10"]
        txt(ax, PX0, 61,
            "10 × 392 fit at admission,\n10 × 400 = 4,000 do not\n"
            "free: 0, and #5 needs a block\n"
            "→ last-admitted #10 is evicted,\nback to the HEAD of the queue",
            8.5, "#222", ha="left", bg=C_GPU)
        arrow(ax, (x10 + w10 / 2, PY + 0.5), (12.5, 29.6), C_PRE, lw=2.2, ls="--", rad=0.0)
        txt(ax, 58, 51.5, "EVICT #10 → head of queue", 9, C_PRE, ha="left", bold=True)
        seq = [("step", 16, C_RUN)] * 6 + [("evict #10", 4, C_PRE)] + [("step", 16, C_RUN)] * 6
        timeline(ax, seq, "no stall at eviction; the cost comes later (right)", C_PRE)
    else:
        held = [(f"#{i + 1}", b, "run") for i, b in enumerate([400, 400, 400, 399, 399, 398, 398, 397, 396])]
        held.append(("#10", 393, "gone"))
        pos = base(ax, held, queue_head="#10", queue_head_color=C_PRE)
        x10, w10 = pos["#10"]
        txt(ax, PX0, 61,
            "free: 393, #10 now needs 394+\n(its generated tokens count)\n"
            "waits until one of the nine\nfinishes (1–3 s), then the prefix\n"
            "KV is loaded a second time",
            8.5, "#222", ha="left", bg=C_GPU)
        arrow(ax, (10.3, 29.5), (x10 + 1.5, PY - 0.3), "#333", lw=1.8, ls="--", rad=-0.12)
        txt(ax, 14, 48, "RE-ADMIT later", 9, "#333", ha="left", bold=True)
        arrow(ax, (61, 36.5), (x10 + w10 / 2, PY - 0.3), C_NEW, lw=3.2, ls="--", ms=22)
        txt(ax, 97.5, 46.8, f"LOAD AGAIN: same 0.75 GiB,\nanother ~{N['load']} ms stall", 8.5, C_NEW,
            ha="right", bold=True)
        seq = [("step", 16, C_RUN)] * 5 + [("#3 done", 4, C_PRE)] + \
              [("LOAD #10 again — all wait", N["load"], C_NEW)] + [("+#10", 40, C_RUN)] + \
              [("step", 16, C_RUN)] * 2
        timeline(ax, seq, f"reloads raise the mean load per request from {N['reload_from']} to {N['reload_to']} at saturation")


fig, axes = plt.subplots(1, 2, figsize=(19, 10.5))
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.02)
case2(axes[0], False)
case2(axes[1], True)
axes[0].set_title("growing requests exhaust the pool → evict", fontsize=12.5,
                  fontweight="bold", color=C_PRE, pad=8)
axes[1].set_title("seconds later: the evicted request returns, its KV is loaded again",
                  fontsize=12.5, fontweight="bold", color="#7c2d12", pad=8)
fig.suptitle("CASE 2 — running requests hit the wall (24 GiB only: 9.95 slots; none at 26 / 28 GiB)",
             fontsize=13.5, y=0.975)
fig.savefig(REPO / "figures" / "kv_case2_eviction.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_case2_eviction.png")
