#!/usr/bin/env python3
"""Colocating inference vs colocating fine-tuning on the decode GPU — the fair comparison.

Left: decode TPOT as a multiple of the decode-alone baseline (log), the bars grouped into
three regimes — (1) fine-tuning, always under MPS; (2) a second model resident in the
margin; (3) a second model whose KV oversubscribes the margin and streams. The apples-to-
apples pair (fine-tune 10%-cap vs 2nd-model fits, both MPS + resident) is highlighted: both
~1.2x. Right: why the decode GPU can host either neighbor — its SMs are ~95% idle.

FT numbers from reports/report_colocation_ft.md (§6.2, MPS arms); inf numbers from this
study's raws. Run from the repo root.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RAW, FIGS = REPO / "output" / "raw", REPO / "figures"
GRY, GRN, RED, AMB, PUR = "#57606a", "#2e7d4f", "#c1440e", "#9a6700", "#8250df"
BASE = 28.0


def tpot(name):
    return json.load(open(RAW / f"{name}.json"))["summary"]["tpot_ms"]["p50"]


FITS_NOMPS = st.mean([tpot(f) for f in
                      ("infc_B_fits", "infc_B_fits_iso", "infc_B_fits_legs", "infc_B_fits_cpu")])

# (x, label, TPOT ms, colour, group)
BARS = [
    (0.0, "10% SM cap\n(+MPS)", 34.2, AMB),
    (0.9, "uncapped\n(+MPS)", 46.6, AMB),
    (2.3, "fits\n+ MPS", tpot("infc_B_fits_mps"), GRN),
    (3.2, "fits\nno MPS", FITS_NOMPS, PUR),
    (4.6, "offload\n+ MPS", tpot("infc_B_offload_mps"), GRN),
    (5.5, "offload\nno MPS", tpot("infc_B_offload"), PUR),
]
GROUPS = [(0.45, "Fine-tuning neighbor"), (2.75, "2nd model — resident"),
          (5.05, "2nd model — KV offloaded")]


def main():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                  gridspec_kw={"width_ratios": [1.7, 1]})

    xs = [b[0] for b in BARS]
    vals = [b[2] / BASE for b in BARS]
    cols = [b[3] for b in BARS]

    # highlight band behind the two apples-to-apples bars (FT 10% cap, 2nd-model fits+MPS)
    ax.axvspan(-0.35, 0.35, color="#eaf4ec", zorder=0)
    ax.axvspan(1.95, 2.65, color="#eaf4ec", zorder=0)

    ax.bar(xs, vals, 0.72, color=cols, zorder=3)
    ax.axhline(1.0, color=GRY, lw=0.9, ls=":")
    ax.set_yscale("log")
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.1f}×", (x, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=11, color=GRY, fontweight="bold")

    # "≈ equal" callout tying the two highlighted ~1.2x bars
    ax.annotate("", (2.3, 1.95), (0.0, 1.95),
                arrowprops=dict(arrowstyle="<->", color=GRN, lw=1.6))
    ax.text(1.15, 2.15, "≈ equal — the affordable regime\n(both MPS + resident, ~1.2×)",
            ha="center", va="bottom", fontsize=9.5, color=GRN, fontweight="bold")

    # group headers
    for gx, gt in GROUPS:
        ax.text(gx, 26, gt, ha="center", fontsize=10.5, color=GRY, fontweight="bold")
    # dividers between groups
    for dx in (1.6, 3.95):
        ax.axvline(dx, color=GRY, lw=0.7, ls="--", alpha=0.4)

    ax.set_xticks(xs, [b[1] for b in BARS], fontsize=9.5)
    ax.set_ylabel("decode TPOT, × the decode-alone baseline (log)", fontsize=10.5)
    ax.set_ylim(0.85, 34)
    ax.set_xlim(-0.6, 6.1)
    ax.set_title("What the neighbor costs the incumbent's decode", fontsize=12.5, color=GRY)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.plot([], [], "s", color=AMB, ms=10, label="fine-tuning (MPS)")
    ax.plot([], [], "s", color=GRN, ms=10, label="2nd model, + MPS")
    ax.plot([], [], "s", color=PUR, ms=10, label="2nd model, no MPS (serializes)")
    ax.legend(fontsize=9.3, loc="upper left", bbox_to_anchor=(0.0, 0.80), ncol=1,
              framealpha=0.95)

    # ---- right: why the decode GPU has room ----
    res = ["SM occupancy\n(compute)", "HBM bandwidth\n(memory)"]
    xp2 = np.arange(2)
    ax2.bar(xp2, [100, 100], 0.62, color="#eef1f4", edgecolor=GRY, lw=1.2, zorder=2)
    ax2.bar(xp2, [5, 50], 0.62, color=[GRN, RED], zorder=3)
    ax2.set_xticks(xp2, res, fontsize=10.5)
    ax2.set_ylim(0, 116)
    ax2.set_ylabel("% used by decode alone", fontsize=10.5)
    ax2.set_title("Why: the decode GPU's own profile", fontsize=12.5, color=GRY)
    ax2.text(0, 10, "~95% idle", ha="center", fontsize=11, color=GRN, fontweight="bold")
    ax2.text(1, 56, "the binding\nresource", ha="center", fontsize=11, color=RED,
             fontweight="bold")
    ax2.annotate("MPS lets any resident\nneighbor use these SMs —\nFT or a 2nd model", (0, 72),
                 fontsize=9.2, color=GRY, ha="center")
    ax2.annotate("a 2nd model hits this\nonly when its KV\nstreams (offload)", (1, 80),
                 fontsize=9.2, color=PUR, ha="center")
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle("Fine-tuning vs a second model on the decode GPU — fair (MPS vs MPS) "
                 "comparison (b26)", fontsize=13, y=1.0)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_vs_ft.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("wrote figures/inf_coloc_vs_ft.png")


if __name__ == "__main__":
    main()
