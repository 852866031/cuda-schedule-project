#!/usr/bin/env python3
"""Figures for the LoRA finetuning sweep: throughput and VRAM vs offloaded layers.

    python scripts/plot_finetune.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"

BLUE, RED, PURPLE, GREY = "#1f6feb", "#c1440e", "#8250df", "#57606a"
PCIE = 14.468          # GB/s, measured in phase 0
GIB2GB = 1.0737


def main():
    FIGS.mkdir(exist_ok=True)
    rows = [r for r in json.loads((OUT / "ft_sweep.json").read_text()) if "error" not in r]
    rows.sort(key=lambda r: r["n_offload"])
    pre_path = OUT / "ft_prefetch_sweep.json"
    pre = sorted([r for r in json.loads(pre_path.read_text()) if "error" not in r],
                 key=lambda r: r["n_offload"]) if pre_path.exists() else []

    n = [r["n_offload"] for r in rows]
    off_gib = [r["offloaded_weight_gib"] for r in rows]
    step = [r["step_s_median"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows]
    peak = [r["peak_vram_gib"] for r in rows]
    mfu = [r["mfu"] * 100 for r in rows]
    base_step = step[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # --- 1. step time: measured vs the cost model -----------------------------------
    ax = axes[0]
    ax.plot(n, step, color=BLUE, marker="o", lw=2, label="measured step time")
    # Every offloaded layer crosses PCIe twice per step: once for forward, once for the
    # recomputed forward inside backward.
    pred = [base_step + g * 2 * GIB2GB / PCIE for g in off_gib]
    ax.plot(n, pred, color=RED, ls="--", lw=1.5,
            label=f"predicted: {base_step:.2f}s + 2×GiB/{PCIE:.1f} GB/s")
    if pre:
        ax.plot([r["n_offload"] for r in pre], [r["step_s_median"] for r in pre],
                color=PURPLE, marker="s", lw=2, label="with prefetch (overlapped)")
    ax.set_xlabel("transformer layers offloaded to DRAM (of 32)")
    ax.set_ylabel("step time (s)")
    ax.set_title("Fetch-on-demand is exactly additive; prefetch hides it", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # --- 2. the tradeoff: throughput bought per GiB of VRAM freed --------------------
    ax = axes[1]
    ax.plot(peak, tps, color=BLUE, marker="o", lw=2, label="fetch-on-demand")
    if pre:
        ax.plot([r["peak_vram_gib"] for r in pre], [r["tokens_per_s"] for r in pre],
                color=PURPLE, marker="s", lw=2, label="with prefetch")
    ax.axhline(tps[0], color=GREY, lw=1, ls="--")
    ax.annotate("no offloading at all", xy=(peak[-1], tps[0]), xytext=(4, 5),
                textcoords="offset points", fontsize=8, color=GREY, ha="left")
    ax.set_xlabel("peak VRAM during training (GiB)  —  less to the right")
    ax.invert_xaxis()
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("What each freed GiB actually costs", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    # --- 3. MFU: how much of the GPU is left doing useful work -----------------------
    ax = axes[2]
    ax.plot(n, mfu, color=BLUE, marker="o", lw=2, label="fetch-on-demand")
    ax.fill_between(n, 0, mfu, color=BLUE, alpha=0.08)
    if pre:
        ax.plot([r["n_offload"] for r in pre], [r["mfu"] * 100 for r in pre],
                color=PURPLE, marker="s", lw=2, label="with prefetch")
    ax.legend(fontsize=8)
    ax.set_xlabel("transformer layers offloaded to DRAM (of 32)")
    ax.set_ylabel("model FLOPs utilisation (%)")
    ax.set_title("How much of the GPU is left doing useful work", fontsize=11)
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.25)

    fig.suptitle("LoRA finetuning Llama-3-8B with base weights streamed from DRAM — "
                 "rank 16 on attention, batch 2 × 2048 tokens, RTX 5090 (PCIe Gen4 ×8)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIGS / "finetune_offload.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p.relative_to(REPO)}")

    # residual check, printed rather than plotted
    print("\nmeasured vs predicted step time:")
    for i, r in enumerate(rows):
        err = (step[i] - pred[i]) / step[i] * 100
        print(f"  {n[i]:>2d} layers ({off_gib[i]:5.2f} GiB): "
              f"{step[i]:.3f}s vs {pred[i]:.3f}s predicted  ({err:+.1f}%)")


if __name__ == "__main__":
    main()
