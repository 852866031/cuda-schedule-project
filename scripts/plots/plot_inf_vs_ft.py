#!/usr/bin/env python3
"""Colocating inference vs colocating fine-tuning on the decode GPU: why one is cheap.

Left: what each neighbor costs the incumbent's decode latency (TPOT × the decode-alone
baseline). The fine-tuning arms (compute-dense) cost 1.2-1.7×; a second inference/decode
tenant (bandwidth-dense) costs 3.3-13×. Right: why — decode leaves ~95% of SM occupancy
idle but is bound on HBM bandwidth, so a compute-dense neighbor fills the idle resource
while a decode neighbor fights for the binding one.

FT numbers from reports/report_colocation_ft.md (§6.2-6.3, GPT-2 trainer, b26); inf
numbers from this study's raws. Run from the repo root.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
GRY, GRN, RED, AMB, PUR = "#57606a", "#2e7d4f", "#c1440e", "#9a6700", "#8250df"

BASE_TPOT = 27.9   # decode alone, b26 (ft report §6.2 and this study's control ~28)

# fine-tuning neighbor (report_colocation_ft.md)
FT = [("FT MPS 10%", 34.2), ("FT idle-gate", 39.2), ("FT MPS 50%", 36.8),
      ("FT uncapped", 46.6)]
# inference/decode neighbor (this study)
def tpot(name):
    return json.load(open(RAW / f"{name}.json"))["summary"]["tpot_ms"]["p50"]
INF = [("inf B/fits", tpot("infc_B_fits")), ("inf A/fits", tpot("infc_A_fits")),
       ("inf A/offload", tpot("infc_A_offload")), ("inf B/offload", tpot("infc_B_offload"))]


def main():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- left: TPOT penalty, FT arms vs inf arms ----
    names = [n for n, _ in FT] + [n for n, _ in INF]
    vals = [v / BASE_TPOT for _, v in FT] + [v / BASE_TPOT for _, v in INF]
    cols = [AMB] * len(FT) + [PUR] * len(INF)
    xp = np.arange(len(names))
    ax.bar(xp, vals, 0.68, color=cols, zorder=3)
    ax.axhline(1.0, color=GRY, lw=0.9, ls=":")
    ax.set_yscale("log")
    for x, v in zip(xp, vals):
        ax.annotate(f"{v:.1f}×", (x, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=GRY)
    ax.set_xticks(xp, names, rotation=25, ha="right", fontsize=8.6)
    ax.set_ylabel("decode TPOT, × the decode-alone baseline (log)", fontsize=10)
    ax.set_title("What the neighbor costs the incumbent's decode", fontsize=11.5, color=GRY)
    ax.grid(alpha=0.25, axis="y", which="both")
    # group labels
    ax.plot([], [], "s", color=AMB, ms=9, label="fine-tuning neighbor (compute-dense)")
    ax.plot([], [], "s", color=PUR, ms=9, label="inference/decode neighbor (bandwidth-dense)")
    ax.legend(fontsize=8.6, loc="upper left")

    # ---- right: why -- decode's resource profile ----
    res = ["SM occupancy\n(compute)", "HBM bandwidth\n(memory)"]
    used = [5, 50]           # decode alone: ~5% occupancy, ~50% DRAM-active (split study)
    xp2 = np.arange(2)
    ax2.bar(xp2, [100, 100], 0.6, color="#eef1f4", edgecolor=GRY, lw=1, zorder=2)
    ax2.bar(xp2, used, 0.6, color=[GRN, RED], zorder=3)
    ax2.set_xticks(xp2, res, fontsize=9)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("% used by decode alone", fontsize=10)
    ax2.set_title("Why: decode's own resource profile", fontsize=11.5, color=GRY)
    ax2.text(0, 8, "~95% idle", ha="center", fontsize=8.5, color=GRN, fontweight="bold")
    ax2.text(1, 54, "the binding\nresource", ha="center", fontsize=8.5, color=RED,
             fontweight="bold")
    ax2.annotate("FT (GEMMs) fills this", (0, 60), fontsize=8, color=AMB, ha="center")
    ax2.annotate("a 2nd decode\nfights for this", (1, 74), fontsize=8, color=PUR,
                 ha="center")
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle("Colocating fine-tuning vs colocating a second model on the decode GPU "
                 "(b26, 2 QPS)", fontsize=12, y=1.02)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_vs_ft.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("wrote figures/inf_coloc_vs_ft.png")


if __name__ == "__main__":
    main()
