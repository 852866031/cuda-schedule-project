#!/usr/bin/env python3
"""Where the N colocated models live: data flow (top) and memory to scale (bottom).

(a) decode-only: a TEMPORARY GPU0 prefill per model writes its prefixes into that
    model's DRAM store and is killed; during measurement the N GPU1 engines only
    retrieve KV (on a miss) and decode.
(b) full: each GPU1 engine prefills its own suffix and decodes; evicted prefixes spill
    to / reload from its own store.
(c) GPU1 memory to scale, the largest N run per cohort: weights / KV grant / other
    (activations, CUDA graphs, CUDA+MPS context), from engine logs and DCGM fb_used.
(d) host RAM to scale at the offload working set: per-model API server + EngineCore
    (incl. 1 GiB pinned L1) + store, from measured RSS, stacked N times.

Numbers are measured (see the constants); run from the repo root.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
FIGS = REPO / "figures"
GREY = "#57606a"
ENG, ENGF = "#2e7d4f", "#ddeee4"          # GPU1 engines
PRE, PREF = "#1f6feb", "#dbe7fb"          # temp GPU0 prefill
STO, STOF = "#c1440e", "#f7e3d8"          # DRAM stores
W_C, KV_C, OT_C = "#8c959f", "#2e7d4f", "#d0d7de"   # weights / KV / other
API_C, EC_C = "#b6c2cf", "#57606a"
plt.rcParams.update({"font.size": 13})

GPU_GIB = 31.35
# measured: weights (engine log), KV grant (engine log), per-model fb (DCGM, N=1 run)
VRAM = {"small": dict(n=8, w=0.93, kv=1.04, per=3.22, label="8 × Qwen2.5-0.5B"),
        "medium": dict(n=4, w=5.76, kv=0.94, per=7.65, label="4 × Qwen2.5-3B")}
# measured RSS (GB) per model at the largest offload run (launch RSS totals / N):
# small N=6 (mc_small_n6_mps_omp4: API 5.76, EngineCore 16.95, stores 21.91 GB),
# medium N=4 (mc_medium_n4_mps: 4.31 / 13.25 / 14.57 GB). EngineCore includes the 1 GiB
# pinned LMCache L1; the store holds the working set (+ ~0.65 GB process overhead).
HOST = {"small": dict(api=0.96, ec=2.83, store=3.65, label="0.5B, offload (48 sessions)",
                      note="swap-backed"),
        "medium": dict(api=1.08, ec=3.31, store=3.64, label="3B, offload (14 sessions)",
                       note="")}
HOST_GIB, OS_GB, FLOOR_GB = 60.5, 3.9, 6.0   # OS = 'used' at idle (measured 02:02)


def box(ax, x, y, w, h, txt, fc, ec, fs=12, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.6, zorder=3,
                           linestyle=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=4)


def arrow(ax, p0, p1, color, txt="", off=(0, 1.2), ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color=color,
                                 lw=1.8, zorder=5, linestyle=ls))
    if txt:
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], txt,
                ha="center", fontsize=11, color=color)


def flow(ax, decode_only):
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    for x, w, name in ((1, 24, "GPU0"), (36, 24, "host DRAM"), (71, 28, "GPU1 (shared)")):
        ax.add_patch(Rectangle((x, 3), w, 50, fill=False, ec=GREY, lw=1.2, ls=(0, (5, 4))))
        ax.text(x + w / 2, 54.5, name, ha="center", fontsize=13, color=GREY)
    ys = [40, 28, 10]
    names = ["1", "2", "N"]
    for y, nm in zip(ys, names):
        box(ax, 39, y, 18, 8, f"store {nm}", STOF, STO)
        box(ax, 74, y, 22, 8, f"engine {nm}", ENGF, ENG)
        if decode_only:
            box(ax, 3, y, 20, 8, f"prefill {nm}", PREF, PRE, ls="--")
            arrow(ax, (23, y + 4), (39, y + 4), PRE)
            arrow(ax, (57, y + 4), (74, y + 4), STO)
        else:
            arrow(ax, (74, y + 5.5), (57, y + 5.5), STO)
            arrow(ax, (57, y + 2.5), (74, y + 2.5), STO)
    ax.text(48, 21, "⋮", ha="center", fontsize=20, color=GREY)
    ax.text(85, 21, "⋮", ha="center", fontsize=20, color=GREY)
    if decode_only:
        ax.text(13, 21, "⋮", ha="center", fontsize=20, color=GREY)
        ax.text(13, 4.6, "temporary: killed\nbefore measuring", ha="center", fontsize=10.5,
                color=PRE)
        ax.text(65.5, 20, "KV\nreload\non miss", ha="center", fontsize=10, color=STO)
        ax.set_title("(a) decode-only: prefixes pre-stored,\nGPU1 only retrieves + decodes",
                     fontsize=14)
    else:
        ax.text(65.5, 22, "spill /\nreload", ha="center", fontsize=10.5, color=STO)
        ax.text(13, 28, "unused", ha="center", fontsize=12, color=GREY)
        ax.set_title("(b) full: each engine prefills its suffix\nand decodes on GPU1",
                     fontsize=14)
    ax.text(85, 5.0, "N engines, MPS", ha="center", fontsize=10.5, color=ENG)


def vram(ax):
    for j, (key, v) in enumerate(VRAM.items()):
        other = v["per"] - v["w"] - v["kv"]
        y = 0
        for i in range(v["n"]):
            for h, c in ((v["w"], W_C), (v["kv"], KV_C), (other, OT_C)):
                ax.add_patch(Rectangle((j - 0.32, y), 0.64, h, facecolor=c, edgecolor="white",
                                       lw=0.6))
                y += h
        ax.text(j + 0.36, y - 0.4, f"{y:.1f} GiB", ha="left", va="top", fontsize=12)
    ax.axhline(GPU_GIB, color="#cf222e", ls="--", lw=1.5)
    ax.text(-0.55, GPU_GIB + 0.5, f"GPU1 capacity {GPU_GIB} GiB", ha="left",
            color="#cf222e", fontsize=12)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 36)
    ax.set_xticks([0, 1], [v["label"] for v in VRAM.values()])
    ax.set_ylabel("GiB")
    for c, lab in ((W_C, "weights"), (KV_C, "KV grant"), (OT_C, "other (act., graphs, ctx)")):
        ax.bar([0], [0], color=c, label=lab)
    ax.legend(loc="upper center", frameon=False, fontsize=11, ncol=3,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_title("(c) GPU1 memory, largest N, to scale", fontsize=14)


def host(ax, ceilings):
    for j, (key, h) in enumerate(HOST.items()):
        n = ceilings[key]
        y = OS_GB
        ax.add_patch(Rectangle((j - 0.32, 0), 0.64, OS_GB, facecolor="#eaeef2",
                               edgecolor="white"))
        for i in range(n):
            for v, c in ((h["api"], API_C), (h["ec"], EC_C), (h["store"], STOF)):
                ax.add_patch(Rectangle((j - 0.32, y), 0.64, v, facecolor=c,
                                       edgecolor="white", lw=0.6))
                y += v
        ax.text(j, y + 0.6, f"N={n}: {y:.0f} GB" + (f" ({h['note']})" if h["note"] else ""),
                ha="center", fontsize=12)
    ax.axhline(HOST_GIB, color="#cf222e", ls="--", lw=1.5)
    ax.axhline(HOST_GIB - FLOOR_GB, color="#bc4c00", ls=":", lw=1.5)
    ax.text(1.55, HOST_GIB + 0.7, "host RAM 60.5 GiB", ha="right", color="#cf222e",
            fontsize=12)
    ax.text(1.55, HOST_GIB - FLOOR_GB - 3.2, "mem-guard floor (6 GB free)", ha="right",
            color="#bc4c00", fontsize=12)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 68)
    ax.set_xticks([0, 1], [h["label"] for h in HOST.values()])
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("GB")
    for c, lab in (("#eaeef2", "OS + rest"), (API_C, "API server"), (EC_C, "EngineCore"),
                   (STOF, "store")):
        ax.bar([0], [0], color=c, label=lab)
    ax.legend(loc="upper center", frameon=False, fontsize=11, ncol=4,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_title("(d) host RAM at the offload working set, to scale", fontsize=14)


def main(ceilings):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.22, wspace=0.18)
    flow(fig.add_subplot(gs[0, 0]), True)
    flow(fig.add_subplot(gs[0, 1]), False)
    vram(fig.add_subplot(gs[1, 0]))
    host(fig.add_subplot(gs[1, 1]), ceilings)
    out = FIGS / "mc_layout.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    import sys
    # largest offload N that fits the host, per cohort (set from the measured runs)
    main({"small": int(sys.argv[1]) if len(sys.argv) > 1 else 6,
          "medium": int(sys.argv[2]) if len(sys.argv) > 2 else 4})
