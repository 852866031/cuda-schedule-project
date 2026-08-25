#!/usr/bin/env python3
"""Figures for the LoRA finetuning sweep: five offload strategies compared.

    python scripts/plot_finetune.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"

PCIE = 14.468          # GB/s, measured in phase 0
GIB2GB = 1.0737
MAX_OFFLOAD = 16       # the range every arm covers

# (file, label, colour, marker, linestyle)
ARMS = [
    ("ft_sweep.json",           "no prefetch",              "#c1440e", "o", "-"),
    ("ft_prefetch_sweep.json",  "prefetch d1",              "#1f6feb", "s", "-"),
    ("ft_prefetch_d2_sweep.json", "prefetch d2",            "#54aeff", "^", "--"),
    ("ft_int_d1_sweep.json",    "interleaved, prefetch d1", "#8250df", "D", "-"),
    ("ft_int_d2_sweep.json",    "interleaved, prefetch d2", "#c297ff", "v", "--"),
]
GREY = "#57606a"


def load(fname):
    p = OUT / fname
    if not p.exists():
        return []
    rows = [r for r in json.loads(p.read_text())
            if "error" not in r and r["n_offload"] <= MAX_OFFLOAD]
    return sorted(rows, key=lambda r: r["n_offload"])


def main():
    FIGS.mkdir(exist_ok=True)
    arms = [(lbl, c, m, ls, load(f)) for f, lbl, c, m, ls in ARMS]
    arms = [a for a in arms if a[4]]
    base = arms[0][4][0]["tokens_per_s"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    # --- 1. throughput vs layers offloaded ------------------------------------------
    ax = axes[0]
    for lbl, c, m, ls, rows in arms:
        ax.plot([r["n_offload"] for r in rows], [r["tokens_per_s"] for r in rows],
                color=c, marker=m, ls=ls, lw=2, ms=6, label=lbl)
    ax.axhline(base, color=GREY, lw=1, ls=":")
    ax.annotate("no offloading", xy=(0, base), xytext=(6, -13),
                textcoords="offset points", fontsize=8, color=GREY)
    ax.set_xlabel("transformer layers offloaded to DRAM (of 32)")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Throughput vs how much is offloaded", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    # --- 2. cost relative to no offloading ------------------------------------------
    ax = axes[1]
    for lbl, c, m, ls, rows in arms:
        ax.plot([r["n_offload"] for r in rows],
                [(1 - r["tokens_per_s"] / base) * 100 for r in rows],
                color=c, marker=m, ls=ls, lw=2, ms=6, label=lbl)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("transformer layers offloaded to DRAM (of 32)")
    ax.set_ylabel("throughput lost vs no offloading (%)")
    ax.set_title("What each strategy costs", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    # --- 3. the actual tradeoff: throughput per GiB freed ---------------------------
    ax = axes[2]
    for lbl, c, m, ls, rows in arms:
        ax.plot([r["peak_vram_gib"] for r in rows], [r["tokens_per_s"] for r in rows],
                color=c, marker=m, ls=ls, lw=2, ms=6, label=lbl)
    ax.axhline(base, color=GREY, lw=1, ls=":")
    ax.set_xlabel("peak VRAM during training (GiB)  —  less to the right")
    ax.invert_xaxis()
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Throughput bought per GiB freed", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("LoRA finetuning Llama-3-8B with base weights streamed from DRAM — "
                 "rank 16 on attention, batch 2 × 2048 tokens, RTX 5090 (PCIe Gen4 ×8)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIGS / "finetune_offload.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p.relative_to(REPO)}\n")

    hdr = f"{'layers':>7s}" + "".join(f"{lbl[:22]:>24s}" for lbl, *_ in arms)
    print(hdr)
    for n in [r["n_offload"] for r in arms[0][4]]:
        line = f"{n:>7d}"
        for lbl, c, m, ls, rows in arms:
            hit = [r for r in rows if r["n_offload"] == n]
            line += f"{hit[0]['tokens_per_s']:>16.0f} tok/s" if hit else f"{'-':>24s}"
        print(line)


if __name__ == "__main__":
    main()
