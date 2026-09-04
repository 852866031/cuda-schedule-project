#!/usr/bin/env python3
"""Decode+decode colocation results, in the style of report_colocation_ft.md's arms figure.

Left: what the 8B incumbent pays -- TTFT p50 and TPOT p50 as ratios to the 8B-alone
baseline (log y, since the offload cells collapse), with absolute 8B throughput on a
twin axis. Right: what the Qwen tenant keeps -- achieved decode throughput vs its solo
ceiling, per scenario x workload.

Baseline = the fresh 8B-alone control (output/raw/split_lmcache_zipf_b26_ctrl.json).
Run from the repo root.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
GRY, BLU, ORG, GRN, AMB = "#57606a", "#1f6feb", "#c1440e", "#2e7d4f", "#9a6700"


def load(name):
    return json.load(open(RAW / f"{name}.json"))


def s8(d):      # 8B summary
    return d.get("summary", {})


def qtp(d):     # qwen throughput
    return d.get("qwen", {}).get("summary", {}).get("output_tok_per_s")


def main():
    base = load("split_lmcache_zipf_b26_ctrl")
    cells = {k: load(f"infc_{k}") for k in
             ("solo_fits", "solo_offload", "A_fits", "A_offload", "B_fits", "B_offload")}

    b_ttft = s8(base)["ttft_ms"]["p50"]
    b_tpot = s8(base)["tpot_ms"]["p50"]
    b_tput = s8(base)["output_tok_per_s"]

    order = ["A_fits", "B_fits", "A_offload", "B_offload"]
    labels = ["A\nfits", "B\nfits", "A\noffload", "B\noffload"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- left: 8B incumbent cost, ratios to baseline (log y) + throughput twin ----
    xpos = np.arange(len(order) + 1)
    xticks = ["8B alone\n(baseline)"] + labels
    ttft = [b_ttft] + [s8(cells[k])["ttft_ms"]["p50"] for k in order]
    tpot = [b_tpot] + [s8(cells[k])["tpot_ms"]["p50"] for k in order]
    tput = [b_tput] + [s8(cells[k])["output_tok_per_s"] for k in order]

    ax.plot(xpos, [v / b_ttft for v in ttft], "-^", color=ORG, lw=1.8, ms=8, zorder=3,
            label="TTFT p50 / baseline  (GPU0 prefill leg)")
    ax.plot(xpos, [v / b_tpot for v in tpot], "-o", color=BLU, lw=1.8, ms=7, zorder=3,
            label="TPOT p50 / baseline  (GPU1 decode leg)")
    ax.axhline(1.0, color=GRY, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(xpos, xticks, fontsize=9)
    ax.set_ylabel("latency, × the 8B-alone baseline (log)", fontsize=10)
    ax.set_title("What the 8B incumbent pays", fontsize=11, color=GRY)
    ax.set_xlim(-0.4, len(order) + 0.4)
    ax.grid(alpha=0.25, which="both")
    # annotate the two extreme ratios so the log axis is readable
    ax.annotate(f"{ttft[-1] / b_ttft:.0f}×  ({ttft[-1] / 1000:.1f}s TTFT)",
                (xpos[-1], ttft[-1] / b_ttft), textcoords="offset points",
                xytext=(-6, 6), fontsize=8, color=ORG, ha="right")
    ax.annotate(f"{tpot[-1] / b_tpot:.0f}×", (xpos[-1], tpot[-1] / b_tpot),
                textcoords="offset points", xytext=(6, -2), fontsize=8, color=BLU)

    ax_t = ax.twinx()
    ax_t.plot(xpos, tput, "-s", color=GRN, lw=1.6, ms=5.5, alpha=0.85)
    ax_t.set_ylim(0, 280)
    ax_t.set_ylabel("8B throughput, tok/s", fontsize=10, color=GRN)
    ax_t.tick_params(axis="y", labelcolor=GRN)
    ax.plot([], [], "-s", color=GRN, ms=5.5, label="8B throughput (right axis)")
    ax.legend(fontsize=8.3, loc="upper left", handlelength=2.6, handletextpad=0.8,
              borderpad=0.6)

    # ---- right: Qwen tenant -- achieved vs its solo ceiling ----
    ceil = {"fits": qtp(cells["solo_fits"]), "offload": qtp(cells["solo_offload"])}
    xp = np.arange(len(order))
    cvals = [ceil["fits" if "fits" in k else "offload"] for k in order]
    gvals = [qtp(cells[k]) for k in order]
    ax2.bar(xp - 0.18, cvals, 0.36, color="#e8ddcf", edgecolor=AMB, lw=1.2,
            label="Qwen solo (ceiling)")
    ax2.bar(xp + 0.18, gvals, 0.36, color=AMB, label="beside the 8B (achieved)")
    for i, (c, gg) in enumerate(zip(cvals, gvals)):
        ax2.text(i + 0.18, gg + 4, f"{gg / c * 100:.0f}%", ha="center", fontsize=9,
                 color=AMB)
    ax2.set_xticks(xp, labels, fontsize=9)
    ax2.set_ylabel("Qwen decode throughput, tok/s", fontsize=10)
    ax2.set_ylim(0, 300)
    ax2.set_title("What the 0.5B tenant keeps", fontsize=11, color=GRY)
    ax2.legend(fontsize=8.5, loc="lower left")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("Qwen2.5-0.5B colocated with the 8B split-decode (b26) — reference "
                 "workload, 2 QPS; A=whole-Qwen, B=decode-only", fontsize=11.5, y=1.02)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_result.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("wrote figures/inf_coloc_result.png")


if __name__ == "__main__":
    main()
