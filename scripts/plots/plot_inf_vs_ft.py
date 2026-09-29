#!/usr/bin/env python3
"""Colocating inference vs colocating fine-tuning on the decode GPU — the fair comparison.

The fine-tune study ran under MPS, so comparing it to a *no-MPS* inference tenant is
unfair. With MPS on both sides, a resident (fits) second-model tenant costs the incumbent
about as much as the fine-tune neighbor (~1.2x vs 1.2-1.7x TPOT): decode leaves ~95% of
its SMs idle, and MPS lets either neighbor's kernels co-reside there. The tenant's penalty
without MPS (~3.5x) is context serialization, which MPS removes. The inference tenant's
distinctive, MPS-resistant cost is KV *streaming* when it oversubscribes VRAM (offload) —
a memory problem the fine-tune neighbor, whose small state stays resident, never has.

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


# (label, TPOT ms, colour). FT arms are MPS (the study's regime).
FITS_NOMPS = st.mean([tpot(f) for f in
                      ("infc_B_fits", "infc_B_fits_iso", "infc_B_fits_legs", "infc_B_fits_cpu")])
BARS = [
    ("FT\nMPS 10%", 34.2, AMB),
    ("FT\nuncapped", 46.6, AMB),
    ("inf fits\nno MPS", FITS_NOMPS, PUR),
    ("inf fits\n+ MPS", tpot("infc_B_fits_mps"), GRN),
    ("inf offload\nno MPS", tpot("infc_B_offload"), PUR),
    ("inf offload\n+ MPS", tpot("infc_B_offload_mps"), GRN),
]


def main():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.9),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    names = [b[0] for b in BARS]
    vals = [b[1] / BASE for b in BARS]
    cols = [b[2] for b in BARS]
    xp = np.arange(len(BARS))
    ax.bar(xp, vals, 0.7, color=cols, zorder=3)
    ax.axhline(1.0, color=GRY, lw=0.9, ls=":")
    ax.set_yscale("log")
    for x, v in zip(xp, vals):
        ax.annotate(f"{v:.1f}×", (x, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=GRY)
    # bracket: FT arms and inf-fits+MPS are comparable (both MPS, resident)
    ax.plot([-0.3, 3.3], [2.4, 2.4], color=GRY, lw=0.8)
    ax.text(1.5, 2.55, "≈ comparable (both MPS, resident)", ha="center", fontsize=8,
            color=GRY)
    ax.set_xticks(xp, names, fontsize=8.4)
    ax.set_ylabel("decode TPOT, × the decode-alone baseline (log)", fontsize=10)
    ax.set_ylim(0.85, 20)
    ax.set_title("What the neighbor costs the incumbent's decode", fontsize=11.5, color=GRY)
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.plot([], [], "s", color=AMB, ms=9, label="fine-tuning (MPS)")
    ax.plot([], [], "s", color=PUR, ms=9, label="2nd model, no MPS (serializes)")
    ax.plot([], [], "s", color=GRN, ms=9, label="2nd model, + MPS")
    ax.legend(fontsize=8.4, loc="upper left", ncol=1)

    # ---- right: why -- decode's resource profile ----
    res = ["SM occupancy\n(compute)", "HBM bandwidth\n(memory)"]
    used = [5, 50]
    xp2 = np.arange(2)
    ax2.bar(xp2, [100, 100], 0.6, color="#eef1f4", edgecolor=GRY, lw=1, zorder=2)
    ax2.bar(xp2, used, 0.6, color=[GRN, RED], zorder=3)
    ax2.set_xticks(xp2, res, fontsize=9)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("% used by decode alone", fontsize=10)
    ax2.set_title("Why: decode's own resource profile", fontsize=11.5, color=GRY)
    ax2.text(0, 8, "~95% idle", ha="center", fontsize=8.5, color=GRN, fontweight="bold")
    ax2.text(1, 54, "the binding\nresource", ha="center", fontsize=8.5, color=RED,
             fontweight="bold")
    ax2.annotate("MPS lets any resident\nneighbor use these SMs", (0, 66), fontsize=7.8,
                 color=GRY, ha="center")
    ax2.annotate("the inf tenant only\nhits this when it\nstreams KV (offload)", (1, 74),
                 fontsize=7.8, color=PUR, ha="center")
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle("Colocating fine-tuning vs a second model on the decode GPU — fair "
                 "(MPS vs MPS) comparison (b26)", fontsize=12, y=1.02)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_vs_ft.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    print("wrote figures/inf_coloc_vs_ft.png")


if __name__ == "__main__":
    main()
