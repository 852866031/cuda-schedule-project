#!/usr/bin/env python3
"""KV-slot study, slide figures (figures/slides/). New diagrams for the background and
setup slides, a fallback-vs-compiled comparison, and slide copies of report figures with
the red circled markers (❶❷❸) the slide text refers to.

    .venv/bin/python scripts/plots/plot_slides.py            # everything
    .venv/bin/python scripts/plots/plot_slides.py system ... # only the named figures
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
FIG = REPO / "figures"
OUT = FIG / "slides"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})

C_DEC, C_LOAD, C_WAIT, C_RED = "#2a78d6", "#eb6834", "#9a9a95", "#b3261e"
C_PF, C_HOST, C_GPU, C_INK = "#1b8a6b", "#f3f1ea", "#eef3f9", "#1d2430"
MARK = "❶❷❸❹❺❻"


def rbox(ax, x, y, w, h, fc, ec, lw=1.4, r=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def txt(ax, x, y, s, fs=13, color=C_INK, ha="center", va="center", bold=False, **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, linespacing=1.3,
            fontweight="bold" if bold else "normal", **kw)


def arrow(ax, p, q, color=C_INK, lw=2.0, ls="-", rad=0.0, ms=18):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def badge(ax, x, y, n, fs=22):
    """red circled number, the slide's pointer"""
    ax.text(x, y, MARK[n - 1], fontsize=fs, color=C_RED, ha="center", va="center",
            fontweight="bold", zorder=20)


def canvas(w=16, h=9):
    fig, ax = plt.subplots(figsize=(w * 0.75, h * 0.75))   # 0.75: figure text >= 24 px on a slide
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"-> figures/slides/{name}.png")


# ------------------------------------------------------------------ background: the split
def fig_system():
    fig, ax = canvas()
    # client + router
    rbox(ax, 2, 37, 17, 16, "white", "#888")
    txt(ax, 10.5, 45, "client\n300 requests\nopen loop", 13)
    rbox(ax, 24, 35, 24, 20, "#f6f6f4", "#666")
    txt(ax, 36, 45, "router\nforwards token #1;\nno admission cap", 13)
    arrow(ax, (19, 45), (24, 45))
    # GPU0 prefill
    rbox(ax, 56, 56, 48, 30, "#e8f4ef", C_PF, lw=2)
    txt(ax, 58, 82, "GPU0 — prefill engine", 15, C_PF, ha="left", bold=True)
    txt(ax, 80, 68, "computes the 6,272-token prompt\n→ token #1; saves its prefix KV\nto LMCache (24 chunks of 256)", 13)
    # GPU1 decode
    rbox(ax, 56, 4, 48, 30, C_GPU, C_DEC, lw=2)
    txt(ax, 58, 30, "GPU1 — decode engine", 15, C_DEC, ha="left", bold=True)
    txt(ax, 80, 16, "budget 24 / 26 / 28 GiB: loads the\n6,144-token prefix KV, recomputes the\n128-token suffix, generates the rest", 13)
    # host RAM / LMCache
    rbox(ax, 112, 24, 46, 42, C_HOST, "#b9b3a1", lw=2)
    txt(ax, 114, 62, "host RAM — LMCache", 15, "#4a4639", ha="left", bold=True)
    rbox(ax, 114, 40, 20, 16, "#fde9dd", C_LOAD)
    txt(ax, 124, 48, "L1 pool\n6 GiB pinned\nper engine", 12, "#7c2d12")
    rbox(ax, 136, 40, 20, 16, "#fff3eb", C_LOAD, ls="--")
    txt(ax, 146, 48, "store\nall 32\nprefixes", 12, "#7c2d12")
    txt(ax, 135, 31, "no GPU-to-GPU P2P (GeForce):\nKV goes GPU0 → host → GPU1", 12, "#4a4639")
    # flows
    arrow(ax, (48, 50), (56, 66), C_PF, rad=-0.15)
    badge(ax, 49, 60, 1)
    arrow(ax, (104, 74), (128, 66), C_LOAD, lw=3, rad=-0.2)
    badge(ax, 120, 76, 2)
    arrow(ax, (128, 24), (104, 16), C_LOAD, lw=3, rad=-0.2)
    badge(ax, 120, 13, 3)
    arrow(ax, (48, 40), (56, 24), C_DEC, rad=0.15)
    badge(ax, 49, 30, 4)
    arrow(ax, (56, 9), (10.5, 36.5), C_DEC, ls="--", rad=-0.25)
    txt(ax, 24, 12, "tokens 2…128\nstream back", 12.5, C_DEC)
    txt(ax, 142, 84, "PCIe Gen4 ×8, 14.5 GB/s", 12, "#6b6b66", bold=True)
    txt(ax, 133, 8, "PCIe Gen4 ×8, 14.5 GB/s", 12, "#6b6b66", bold=True)
    save(fig, "s_system")


# ------------------------------------------------------------------ setup: testbed
def fig_testbed():
    fig, ax = canvas(16, 8)
    ax.set_ylim(0, 80)
    rbox(ax, 3, 4, 74, 72, "#fafaf8", "#888", lw=1.6)
    txt(ax, 6, 72, "hardware", 15, ha="left", bold=True)
    rbox(ax, 8, 50, 30, 16, "#e8f4ef", C_PF, lw=1.8)
    txt(ax, 23, 58, "GPU0  RTX 5090\n32 GB, sm_120\nprefill engine", 12.5)
    rbox(ax, 42, 50, 30, 16, C_GPU, C_DEC, lw=1.8)
    txt(ax, 57, 58, "GPU1  RTX 5090\n32 GB, sm_120\ndecode engine", 12.5)
    rbox(ax, 8, 10, 64, 20, C_HOST, "#b9b3a1", lw=1.8)
    txt(ax, 40, 20, "host: 32 CPU cores, 60 GiB RAM\nLMCache store + pinned L1 pools live here", 12.5)
    for x in (23, 57):
        ax.plot([x, x], [30, 50], color="#6b6b66", lw=4)
    txt(ax, 40, 40, "PCIe Gen4 ×8\n14.47 GB/s\n(measured)\nno P2P", 12, "#6b6b66", bold=True)
    rbox(ax, 83, 4, 74, 72, "#fafaf8", "#888", lw=1.6)
    txt(ax, 86, 72, "software", 15, ha="left", bold=True)
    rows = [("model", "Llama-3-8B-Instruct, bf16\nKV = 128 KiB per token"),
            ("serving", "vLLM 0.15.1: FLASH_ATTN, CUDA graphs,\nasync scheduling, block = 16 tokens"),
            ("KV transfer", "LMCache 0.4.4, chunk 256 tokens,\nits CUDA kernels built for\nthis box's torch 2.9.1"),
            ("isolation", "one stack at a time, fresh launch per cell,\nnothing else on either GPU")]
    for i, (k, v) in enumerate(rows):
        y = 60 - i * 15
        txt(ax, 87, y, k, 13, C_DEC, ha="left", bold=True)
        txt(ax, 104, y, v, 12, ha="left")
    save(fig, "s_testbed")


# ------------------------------------------------------------------ setup: workload
def fig_workload():
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.55, wspace=0.22,
                          left=0.05, right=0.98, top=0.93, bottom=0.1)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 6400)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for (a, b, c, lab) in ((0, 6144, "#c9dcf3", "shared session prefix: 6,144 tokens (one of 32 sessions)"),
                           (6144, 6272, C_LOAD, ""), (6272, 6400, C_DEC, "")):
        ax.add_patch(Rectangle((a, 0.3), b - a, 0.45, fc=c, ec="white", lw=1))
        if lab:
            ax.text((a + b) / 2, 0.525, lab, ha="center", va="center", fontsize=14)
    ax.annotate("", (6208, 0.3), (6208, 0.05), arrowprops=dict(arrowstyle="-", color="#555"),
                annotation_clip=False)
    ax.annotate("", (6336, 0.3), (6336, 0.05), arrowprops=dict(arrowstyle="-", color="#555"),
                annotation_clip=False)
    ax.text(6400, -0.12, "+ 128 unique suffix tokens (orange) + 128 output tokens (blue)", ha="right",
            va="center", fontsize=13)
    ax.set_title("one request = 6,400 tokens = 400 KV blocks of 16 tokens = 0.78 GiB of KV", loc="left",
                 fontsize=15)
    # zipf
    ax = fig.add_subplot(gs[1, 0])
    k = np.arange(1, 33)
    p = 1 / k ** 1.1
    p /= p.sum()
    ax.bar(k, p * 100, color=C_DEC, width=0.75)
    ax.set_xlabel("session rank")
    ax.set_ylabel("share of requests (%)")
    ax.set_title("which session: zipf 1.1 over 32 sessions", loc="left", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    # poisson
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    for i, q in enumerate((1, 3, 5)):
        t = np.cumsum(rng.exponential(1 / q, 200))
        t = t[t < 20]
        ax.vlines(t, i + 0.15, i + 0.85, color=C_INK, lw=1.2)
        ax.text(-0.4, i + 0.5, f"{q} req/s", ha="right", va="center", fontsize=12.5)
    ax.set_xlim(-0.2, 20)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title("when: Poisson arrivals, open loop, seed 0", loc="left", fontsize=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(OUT / "s_workload.png", dpi=130, facecolor="white")
    plt.close(fig)
    print("-> figures/slides/s_workload.png")


# ------------------------------------------------------------------ setup: one request's path
def fig_path():
    fig, ax = canvas(16, 7)
    ax.set_ylim(0, 70)
    lanes = [(56, "router / client", "#f6f6f4", "#666"), (40, "GPU0 prefill", "#e8f4ef", C_PF),
             (24, "GPU1 decode", C_GPU, C_DEC)]
    for y, lab, fc, ec in lanes:
        rbox(ax, 2, y - 5, 156, 10, fc, ec, lw=1.2, r=0.6)
        txt(ax, 4, y, lab, 13, ec, ha="left", bold=True)
    seg = [  # lane y, x0, x1, color, label
        (40, 24, 46, C_PF, "prefill"),
        (40, 46, 56, C_LOAD, "save"),
        (24, 58, 82, C_WAIT, "wait for a slot"),
        (24, 82, 96, C_LOAD, "load KV"),
        (24, 96, 150, C_DEC, "decode 127 more tokens"),
    ]
    for y, a, b, c, lab in seg:
        ax.add_patch(Rectangle((a, y - 3.5), b - a, 7, fc=c, ec="white"))
        txt(ax, (a + b) / 2, y, lab, 11.5, "white", bold=True)
    for x, lab in ((22, "submit"), (56, "token #1"), (150, "last token")):
        ax.plot([x, x], [16, 61], color="#444", lw=1, ls=":")
        txt(ax, x, 63.5, lab, 12, "#444")
    # metrics
    def span(a, b, y, lab, n):
        ax.annotate("", (a, y), (b, y), arrowprops=dict(arrowstyle="<->", color=C_INK, lw=1.4))
        txt(ax, (a + b) / 2, y - 3.2, lab, 12.5)
        badge(ax, (a + b) / 2 - 0.66 * len(lab) - 2.5, y - 3.2, n, 17)   # just left of its label
    span(22, 56, 12, "TTFT (end user)", 1)
    span(58, 82, 6.5, "decode queue wait (vLLM metric)", 2)
    span(56, 150, 12, "TPOT = (last − token #1) ÷ 127, end user", 3)
    span(82, 150, 6.5, "residence = holds 392–400 blocks", 4)
    txt(ax, 107, 2, "", 1)
    save(fig, "s_path")


# ------------------------------------------------------------------ setup: one cell
def fig_method():
    fig, ax = canvas(16, 6)
    ax.set_ylim(0, 60)
    steps = [("launch", "store + 2 engines +\nrouter; budget and\nmethod set by flags"),
             ("check", "KV grant, slots,\nLMCache backend =\ncompiled kernels"),
             ("warm-up", "every session's\nprefix saved to\nLMCache once"),
             ("measure", "300 requests at\nthe cell's rate;\ntrace on decode"),
             ("audit", "0 failed, prefix hit\n≥ 0.95, all tokens,\nno hang; else rerun")]
    for i, (h, b) in enumerate(steps):
        x = 4 + i * 31
        rbox(ax, x, 16, 26, 34, "#f6f8fb", C_DEC, lw=1.6)
        txt(ax, x + 13, 44, h, 15, C_DEC, bold=True)
        txt(ax, x + 13, 30, b, 12.5)
        if i:
            arrow(ax, (x - 5, 33), (x, 33))
    txt(ax, 80, 7, "one fresh launch per cell · 3 budgets × 9 rates = 27 cells per method · "
                   "teardown and empty GPUs between cells", 13, "#4a5260")
    save(fig, "s_method")


# ------------------------------------------------------------------ results: compiled vs fallback
def fig_kernels():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.4})
    b = np.arange(3)
    w = 0.36
    data = [("(a) KV copy per request (ms)", [189, 194, 202], [113, 114, 115], "{:.0f}"),
            ("(b) residence at 3 req/s (s)", [4.05, 4.36, 5.18], [3.15, 3.25, 3.35], "{:.2f}"),
            ("(c) saturation (req/s)", [2.38, 2.66, 2.75], [2.97, 3.35, 3.58], "{:.2f}")]
    for ax, (t, f, c, fmt) in zip(axes, data):
        ax.bar(b - w / 2, f, w, color="#c9c6bd", label="Python fallback")
        ax.bar(b + w / 2, c, w, color=C_DEC, label="compiled kernels")
        for x, v in zip(b - w / 2, f):
            ax.text(x, v * 1.015, fmt.format(v), ha="center", va="bottom", fontsize=11, color="#555")
        for x, v in zip(b + w / 2, c):
            ax.text(x, v * 1.015, fmt.format(v), ha="center", va="bottom", fontsize=11, color=C_DEC)
        ax.set_xticks(b)
        ax.set_xticklabels(["24 GiB", "26 GiB", "28 GiB"])
        ax.set_title(t, loc="left", fontsize=14)
        ax.set_ylim(0, max(f + c) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=12, bbox_to_anchor=(0, 1.0))
    axes[0].set_ylim(0, 260)
    fig.suptitle("compiled LMCache kernels vs the Python fallback (same flags, same workload; (a), (b) at 3 req/s)",
                 fontsize=14, y=1.04)
    fig.savefig(OUT / "s_kernels.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("-> figures/slides/s_kernels.png")


# ------------------------------------------------------------------ load path: four designs
def fig_loadpath():
    fig, ax = canvas(16, 8)
    ax.set_ylim(0, 80)
    X0, SW, H = 60, 9, 5.5                  # first step's x, one decode step's width, bar height

    def steps(x, y, n):
        for k in range(n):
            ax.add_patch(Rectangle((x + k * SW, y), SW - 0.4, H, fc=C_DEC, ec="none"))
        return x + n * SW

    def label(y, head, note):
        txt(ax, 2, y + H / 2 + 2.2, head, 14, C_INK, ha="left", bold=True)
        txt(ax, 2, y + H / 2 - 2.0, note, 11.5, "#4a5260", ha="left", va="top")

    def hatch(x, y, w):
        ax.add_patch(Rectangle((x, y - 3.6), w, 2.6, fc="white", ec=C_LOAD, hatch="///", lw=1.2))

    y = 66                                                     # synchronous
    label(y, "synchronous (baseline)", "copy inside the step:\nthe whole batch waits")
    x = steps(X0, y, 3)
    ax.add_patch(Rectangle((x, y), 32, H, fc=C_LOAD, ec="none"))
    txt(ax, x + 16, y + H / 2, "KV copy, no tokens", 11.5, "white", bold=True)
    steps(x + 32, y, 4)
    y = 48                                                     # layerwise
    label(y, "use_layerwise", "copy split per layer,\nstill inside the step")
    x = steps(X0, y, 3)
    for k in range(8):
        ax.add_patch(Rectangle((x + k * 4, y), 3.2, H, fc=C_LOAD, ec="none"))
        ax.add_patch(Rectangle((x + k * 4 + 3.2, y), 0.8, H, fc=C_DEC, ec="none"))
    steps(x + 32, y, 4)
    y = 30                                                     # async_loading
    label(y, "enable_async_loading", "store → L1 in the background;\nL1 → GPU still inside the step")
    x = steps(X0, y, 3)
    hatch(X0 + 2, y, x - X0 - 2)
    ax.add_patch(Rectangle((x, y), 19, H, fc=C_LOAD, ec="none"))
    txt(ax, x + 9.5, y + H / 2, "L1 → GPU", 11.5, "white", bold=True)
    steps(x + 19, y, 5)
    y = 12                                                     # MP
    label(y, "MP connector", "another process copies;\nthe batch keeps decoding")
    steps(X0, y, 10)
    hatch(X0 + 3 * SW, y, 32)
    txt(ax, 105, 2.5, "blue: decode step  ·  orange: copy inside a step  ·  hatched: off the critical path",
        11.5, "#4a5260")
    save(fig, "s_loadpath")


# ------------------------------------------------------------------ async_loading, smoke (24 GiB, 2.5 req/s)
def fig_async_smoke():
    """24 GiB, 2.5 req/s: kv_cops_b24_q2.5_tr_r2 vs kv_cops_async_b24_q2.5_tr_g (D10 grid cell)"""
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.55})
    data = [("KV copy inside the step\n(ms per request)", 116, 70, "{:.0f}"),
            ("stalls from others'\nloads (s per request)", 0.75, 0.47, "{:.2f}"),
            ("residence (s)", 2.83, 2.48, "{:.2f}")]
    for i, (ax, (t, a, b, fmt)) in enumerate(zip(axes, data)):
        ax.bar([0, 1], [a, b], color=["#c9c6bd", C_DEC], width=0.6)
        for x, v in ((0, a), (1, b)):
            ax.text(x, v + max(a, b) * 0.03, fmt.format(v), ha="center", va="bottom", fontsize=13)
        ax.set_ylim(0, max(a, b) * 1.25)
        ax.set_title(t, fontsize=13.5)
    ax = axes[3]                                   # queue time = wait for a slot + async prefetch
    ax.bar([0], [143], color="#c9c6bd", width=0.6)
    ax.bar([1], [50], color=C_DEC, width=0.6, label="wait for a slot")
    ax.bar([1], [130], bottom=[50], color="white", edgecolor=C_DEC, hatch="///", width=0.6,
           label="async prefetch (no slot held)")
    ax.text(0, 147, "143", ha="center", va="bottom", fontsize=13)
    ax.text(1, 50 / 2, "50", ha="center", va="center", fontsize=13, color="white")
    ax.text(1, 50 + 130 / 2, "130", ha="center", va="center", fontsize=13, color=C_DEC,
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_ylim(0, 230)
    ax.set_title("vLLM queue time, mean (ms)", fontsize=13.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", bbox_to_anchor=(-0.1, 1.0))
    for i, ax in enumerate(axes):
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["sync", "async"], fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.12, 1.2, MARK[i], transform=ax.transAxes, fontsize=18, color=C_RED, ha="center",
                va="center", fontweight="bold")
    fig.savefig(OUT / "s_async_smoke.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("-> figures/slides/s_async_smoke.png")


# ------------------------------------------------------------------ MP connector architecture
def fig_mp():
    fig, ax = canvas(16, 8)
    ax.set_ylim(0, 80)
    rbox(ax, 4, 46, 44, 28, "#e8f4ef", C_PF, lw=2)
    txt(ax, 6, 70, "GPU0 — prefill engine", 14, C_PF, ha="left", bold=True)
    txt(ax, 26, 56, "saves each prompt's KV\nthrough the server", 12.5)
    rbox(ax, 4, 6, 44, 30, C_GPU, C_DEC, lw=2)
    txt(ax, 6, 32, "GPU1 — decode engine", 14, C_DEC, ha="left", bold=True)
    txt(ax, 26, 18, "a new request gets its blocks,\nthen waits for its KV while\nthe batch keeps decoding", 12.5)
    rbox(ax, 84, 18, 72, 44, C_HOST, "#b9b3a1", lw=2)
    txt(ax, 86, 58, "LMCache MP server — its own process", 14, "#4a4639", ha="left", bold=True)
    rbox(ax, 88, 24, 30, 26, "#fde9dd", C_LOAD)
    txt(ax, 103, 37, "pinned L1\n32 GiB\nall 32 prefixes", 12.5, "#7c2d12")
    txt(ax, 137, 37, "compiled copy\nkernels; maps both\nengines' KV blocks\nthrough CUDA IPC", 12.5, "#4a4639")
    arrow(ax, (48, 60), (84, 50), C_LOAD, lw=3, rad=-0.1)
    badge(ax, 66, 61, 1)
    arrow(ax, (84, 28), (48, 20), C_LOAD, lw=3, rad=-0.1)
    badge(ax, 66, 18, 2)
    txt(ax, 66, 40, "lookup / store / retrieve\nrequests over ZMQ", 12, "#6b6b66")
    save(fig, "s_mp")


# ------------------------------------------------------------------ mechanism, slide-native versions
def pool(ax, x0, y0, held, w_slot=8.6, h=9, wall=True, label=None):
    """KV pool of 24 GiB as slots; held = list of (label, kind) with kind run|new|gone|evict"""
    x = x0
    for lab, kind in held:
        fc, ec, tc, ls = {"run": (C_DEC, "white", "white", "-"), "new": (C_LOAD, "white", "white", "-"),
                          "gone": ("#f6f6f4", C_RED, C_RED, ":"), "evict": (C_DEC, C_RED, "white", "-")}[kind]
        ax.add_patch(Rectangle((x, y0), w_slot - 0.6, h, fc=fc, ec=ec, lw=2.5 if kind == "evict" else 1, ls=ls))
        txt(ax, x + (w_slot - 0.6) / 2, y0 + h / 2, lab, 12.5, tc, bold=True)
        x += w_slot
    if wall:
        xw = x0 + 9.95 * w_slot
        ax.plot([xw, xw], [y0 - 2, y0 + h + 2], color=C_RED, lw=3)
        txt(ax, xw + 1, y0 + h + 1.5, "pool ends", 12, C_RED, ha="left", bold=True)
    if label:
        txt(ax, x0, y0 + h + 3.5, label, 13.5, C_INK, ha="left", bold=True)


def steps_row(ax, x0, y0, seq, unit=0.75, h=6):
    x = x0
    for lab, ms, c in seq:
        w = ms * unit
        ax.add_patch(Rectangle((x, y0), w - 0.3, h, fc=c, ec="white"))
        if lab:
            txt(ax, x + w / 2, y0 + h / 2, lab, 12.5, "white", bold=True)
        x += w
    return x


def fig_case1_s():
    fig, ax = canvas(16, 8)
    ax.set_ylim(0, 80)
    held = [(f"#{i}", "run") for i in range(1, 10)] + [("#10", "new")]
    pool(ax, 4, 58, held, label="decode GPU, 24 GiB: KV pool of 3,980 blocks = 9.95 slots")
    # queue
    rbox(ax, 4, 22, 46, 24, "#fafaf8", "#888")
    txt(ax, 6, 25, "waiting queue, first come first served", 13, ha="left", bold=True)
    for i in range(4):
        ax.add_patch(Rectangle((7 + i * 9, 30), 8, 8, fc="#6b6b66" if i == 0 else C_WAIT, ec="white"))
    txt(ax, 11, 34, "head", 12, "white", bold=True)
    txt(ax, 27, 24.5, "", 1)
    arrow(ax, (11, 38.5), (84, 57.5), C_INK, rad=-0.15)
    badge(ax, 54, 33, 1)
    txt(ax, 57, 33, "the head is admitted only\nif 392 blocks are free;\nuntil then all of them wait", 12.5, ha="left")
    # LMCache
    rbox(ax, 106, 22, 50, 24, C_HOST, "#b9b3a1")
    txt(ax, 108, 42, "host RAM: LMCache", 13, "#4a4639", ha="left", bold=True)
    txt(ax, 131, 31, "the request's 6,144-token\nprefix KV, 0.75 GiB", 12.5, "#7c2d12")
    arrow(ax, (115, 46), (87, 57.5), C_LOAD, lw=3.5, rad=0.1)
    badge(ax, 108, 53, 2)
    txt(ax, 111, 53, "copy ≈ 115 ms, inside the step", 13, C_LOAD, ha="left", bold=True)
    # engine steps
    txt(ax, 4, 15.5, "engine steps (one forward pass of the whole batch each):", 13, ha="left", bold=True)
    end = steps_row(ax, 4, 4, [("", 16, C_DEC)] * 3 + [("copy for #10: the 9 running get no token", 115, C_LOAD)]
                    + [("", 16, C_DEC)] * 3, unit=0.6)
    badge(ax, end + 3, 7, 3)
    txt(ax, end + 6, 7, "≈ 20% of decode\nwall time", 12.5, C_LOAD, ha="left")
    save(fig, "s_case1")


def fig_case2_s():
    fig, ax = canvas(16, 8)
    ax.set_ylim(0, 80)
    txt(ax, 4, 75, "Step 1: all ten grow toward 400 blocks, and 10 × 400 > 3,980", 13.5, ha="left", bold=True)
    held = [(f"#{i}", "run") for i in range(1, 10)] + [("#10", "evict")]
    pool(ax, 4, 58, held)
    ax.add_patch(Rectangle((4 + 9 * 8.6, 56), 8.0, 1.2, fc=C_RED, ec="none"))
    badge(ax, 100, 63, 1)
    txt(ax, 103, 63, "no free block → the last admitted (#10)\nis evicted: blocks freed, progress lost", 12.5,
        C_RED, ha="left")
    arrow(ax, (82, 57), (22, 44), C_RED, ls="--", rad=0.1)
    rbox(ax, 4, 30, 43, 13, "#fafaf8", "#888")
    ax.add_patch(Rectangle((7, 32.5), 9, 8, fc=C_RED, ec="white"))
    txt(ax, 11.5, 36.5, "#10", 12, "white", bold=True)
    for i in range(1, 4):
        ax.add_patch(Rectangle((7 + i * 10, 32.5), 9, 8, fc=C_WAIT, ec="white"))
    badge(ax, 51, 36.5, 2)
    txt(ax, 54, 36.5, "back to the HEAD of the queue; nothing stalls yet", 13, C_RED, ha="left")
    # later
    txt(ax, 4, 22, "Step 2: seconds later a slot frees; #10 is admitted again and its prefix copied again", 13.5,
        ha="left", bold=True)
    end = steps_row(ax, 4, 6, [("", 16, C_DEC)] * 3 + [("copy for #10, again", 115, C_LOAD)]
                    + [("", 16, C_DEC)] * 3, unit=0.6)
    badge(ax, end + 3, 9, 3)
    txt(ax, end + 6, 9, "another ≈ 115 ms\nbatch stall", 12.5, C_LOAD, ha="left")
    save(fig, "s_case2")


# ------------------------------------------------------------------ layerwise: where the residence goes
def fig_layerwise_s():
    tl = REPO / "output" / "kv_vram" / "timeline"
    arms = [("synchronous", "kv_cops_b24_q2.5_tr_r2"), ("use_layerwise", "kv_cops_lw_b24_q2.5_tr")]
    parts = []
    for _, n in arms:
        rq = [r for r in csv.DictReader(open(tl / f"{n}_req.csv")) if r.get("resident")]
        st = list(csv.DictReader(open(tl / f"{n}_steps.csv")))
        t = np.array([float(x["t"]) for x in st])
        dt = np.r_[np.nan, np.diff(t)]
        ok = dt < 2
        med = np.nanmedian(dt[ok])
        norm, exc = [], []
        for r in rq:
            m = ok & (t > float(r["t_first"])) & (t <= float(r["t_fin"]))
            norm.append(np.minimum(dt[m], 1.5 * med).sum())
            exc.append(np.maximum(dt[m] - 1.5 * med, 0).sum())
        parts.append([np.mean([float(r["load"]) for r in rq]), np.mean(norm), np.mean(exc)])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labs = ["own KV load (admitted → token 1)", "decode steps at normal speed",
            "extra time of steps that carried a load (anyone's)"]
    cols = [C_LOAD, C_DEC, "#f4b48f"]
    for i, (name, _) in enumerate(arms):
        left = 0
        for k in range(3):
            v = parts[i][k]
            ax.barh(i, v, left=left, color=cols[k], height=0.55, label=labs[k] if i == 0 else None,
                    edgecolor="white")
            if v > 0.3:
                ax.text(left + v / 2, i, f"{v:.2f} s", ha="center", va="center", fontsize=13,
                        color="white" if k < 2 else C_INK)
            left += v
        ax.text(left + 0.05, i, f"{left:.2f} s", va="center", fontsize=13)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([a for a, _ in arms], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("seconds per request, admitted → last token (24 GiB, 2.5 req/s)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=12)
    ax.text(parts[0][0] + parts[0][1] + parts[0][2] / 2, -0.45, MARK[0], fontsize=18, color=C_RED,
            ha="center", fontweight="bold")
    ax.text(parts[1][0] + parts[1][1] + parts[1][2] / 2, 0.55, MARK[1], fontsize=18, color=C_RED,
            ha="center", fontweight="bold")
    fig.savefig(OUT / "s_layerwise.png", dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("-> figures/slides/s_layerwise.png", [[round(x, 2) for x in p] for p in parts])


# ------------------------------------------------------------------ markers on report figures
def mark(src, name, marks, boxes=(), crop=None):
    """copy figures/<src> with red circled numbers at fractional (x, y) from the top-left,
    and red rectangles (x0, y0, x1, y1) in the same coordinates"""
    img = mpimg.imread(FIG / src)
    if crop:                       # marks/boxes are given in the uncropped figure's fractions
        h, w = img.shape[:2]
        cx0, cy0, cx1, cy1 = crop
        img = img[int(cy0 * h):int(cy1 * h), int(cx0 * w):int(cx1 * w)]
        fx = lambda x: (x - cx0) / (cx1 - cx0)
        fy = lambda y: (y - cy0) / (cy1 - cy0)
        marks = [(fx(x), fy(y), n) for x, y, n in marks]
        boxes = [(fx(a), fy(b), fx(c), fy(d)) for a, b, c, d in boxes]
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 130, h / 130), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    fs = max(18, min(34, w / 75))
    for (x, y, n) in marks:
        ax.text(x * w, y * h, MARK[n - 1], fontsize=fs, color=C_RED, ha="center", va="center",
                fontweight="bold", zorder=20,
                bbox=dict(boxstyle="circle,pad=0.05", fc="white", ec="none", alpha=0.85))
    for (x0, y0, x1, y1) in boxes:
        ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False,
                               ec=C_RED, lw=3))
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"-> figures/slides/{name}.png")


# name -> (src, marks [(x, y, n)], boxes [(x0, y0, x1, y1)], crop); fractions of the source
# figure from its top-left, read off the rendered PNGs
MARKED = {
    "m_blocks": ("kv_blocks.png", [(0.015, 0.47, 1), (0.497, 0.60, 2), (0.497, 0.735, 3)], [],
                 (0, 0.135, 1, 1)),
    "m_grant": ("kv_grant.png", [(0.25, 0.44, 1), (0.61, 0.44, 2), (0.015, 0.81, 3)], [],
                (0, 0.12, 1, 0.9)),
    "m_slots": ("kv_slots.png", [(0.285, 0.207, 1), (0.285, 0.44, 2), (0.285, 0.672, 3)], [],
                (0, 0.09, 1, 1)),
    "m_case1": ("kv_case1_admission.png", [(0.095, 0.425, 1), (0.37, 0.575, 2), (0.215, 0.96, 3),
                                           (0.565, 0.069, 4)], [], None),
    "m_case2": ("kv_case2_eviction.png", [(0.095, 0.43, 1), (0.275, 0.525, 2), (0.872, 0.605, 3),
                                          (0.78, 0.96, 4)], [], None),
    "m_heat": ("kv_grid_heat.png", [(0.083, 0.2, 1), (0.142, 0.2, 2), (0.587, 0.2, 3), (0.857, 0.2, 4)],
               [(0.0485, 0.153, 0.118, 0.884)], None),
    "m_3d": ("kv_grid_3d.png", [(0.07, 0.63, 1), (0.105, 0.47, 2)], [], (0, 0.15, 0.34, 0.92)),
    "m_mgc": ("kv_mgc.png", [(0.285, 0.37, 1), (0.17, 0.37, 2), (0.72, 0.65, 3)], [], None),
    "m_wait_free": ("kv_wait_vs_free.png", [(0.1, 0.18, 1), (0.25, 0.82, 2), (0.778, 0.69, 3)], [], None),
    "m_time_split": ("kv_time_split.png", [(0.854, 0.855, 1), (0.854, 0.78, 2), (0.185, 0.6, 3)], [], None),
    "m_timeline": ("kv_slots_timeline.png", [(0.08, 0.23, 1), (0.535, 0.23, 2), (0.1, 0.66, 3),
                                             (0.55, 0.85, 4)], [], None),
    "m_stream": ("kv_stream_idea.png", [(0.47, 0.56, 1), (0.92, 0.69, 2)], [], None),
    "m_layerwise": ("kv_layerwise_measured.png", [(0.28, 0.24, 1), (0.785, 0.2, 2)], [], None),
    "m_timeline_s": ("kv_slots_timeline.png", [(0.08, 0.23, 1), (0.535, 0.23, 2)], [], (0, 0.172, 1, 0.545)),
}


FIGS = {"system": fig_system, "testbed": fig_testbed, "workload": fig_workload, "path": fig_path,
        "method": fig_method, "kernels": fig_kernels,
        "loadpath": fig_loadpath, "async_smoke": fig_async_smoke, "mp": fig_mp, "case1": fig_case1_s, "case2": fig_case2_s, "layerwise": fig_layerwise_s}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS) + list(MARKED)
    for k in want:
        if k in FIGS:
            FIGS[k]()
        elif k in MARKED:
            src, m, bx, cr = MARKED[k]
            mark(src, k, m, bx, cr)
