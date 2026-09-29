#!/usr/bin/env python3
"""MPS settles the axis-1 mechanism: the decode-side cost is GPU1 context serialization.

Without MPS two decode processes cannot co-reside on GPU1, so they serialize and the 8B
decode's TPOT rises ~3.4x. Turning on MPS (both engines become clients, kernels share the
SMs) collapses TPOT back near the 8B-alone baseline and recovers throughput — proving the
waiting was context serialization, not a fundamental resource limit. Data from this study
(fits cells, n=4 without MPS). Run from the repo root.
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
GRY, GRN, PUR, AMB = "#57606a", "#2e7d4f", "#8250df", "#9a6700"

BASE_TPOT = 28
NOMPS = {"A/fits": ["infc_A_fits", "infc_A_fits_r2", "infc_A_fits_r3", "infc_A_fits_r4"],
         "B/fits": ["infc_B_fits", "infc_B_fits_iso", "infc_B_fits_legs", "infc_B_fits_cpu"]}
MPS = {"A/fits": "infc_A_fits_mps", "B/fits": "infc_B_fits_mps"}


def tpot(f):
    return json.load(open(RAW / f"{f}.json"))["summary"]["tpot_ms"]["p50"]


def tput(f):
    return json.load(open(RAW / f"{f}.json"))["summary"]["output_tok_per_s"]


def main():
    cells = ["A/fits", "B/fits"]
    x = np.arange(len(cells))
    w = 0.36
    no_mean = [st.mean([tpot(f) for f in NOMPS[c]]) for c in cells]
    no_err = [[st.mean([tpot(f) for f in NOMPS[c]]) - min(tpot(f) for f in NOMPS[c]) for c in cells],
              [max(tpot(f) for f in NOMPS[c]) - st.mean([tpot(f) for f in NOMPS[c]]) for c in cells]]
    mps = [tpot(MPS[c]) for c in cells]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    b1 = ax.bar(x - w / 2, no_mean, w, yerr=no_err, capsize=4, color=PUR,
                label="no MPS (decodes serialize)", zorder=3)
    b2 = ax.bar(x + w / 2, mps, w, color=GRN, label="+ MPS (decodes co-reside)", zorder=3)
    ax.axhline(BASE_TPOT, color=GRY, ls="--", lw=1.2)
    ax.annotate(f"8B-alone baseline: {BASE_TPOT} ms", (0.02, BASE_TPOT),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(0, 4), fontsize=9, color=GRY)
    # no-MPS value labels above the error-bar caps; +MPS just above the short bars
    for i, (bb, v) in enumerate(zip(b1, no_mean)):
        ax.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, v + no_err[1][i]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=PUR)
    for bb, v in zip(b2, mps):
        ax.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=9, color=GRN)
    # throughput recovery, on its own line at the top of each cell
    for i, c in enumerate(cells):
        tk_no = round(st.mean([tput(f) for f in NOMPS[c]]))
        ax.annotate(f"tput {tk_no}→{round(tput(MPS[c]))}", (i, 126), ha="center",
                    fontsize=8.5, color=GRY, fontweight="bold")
    ax.set_xticks(x, cells, fontsize=10)
    ax.set_ylabel("8B decode TPOT p50, ms", fontsize=10)
    ax.set_ylim(0, 134)
    ax.set_title("MPS collapses the decode-side cost → axis-1 is GPU1 context serialization",
                 fontsize=11, color=GRY)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "inf_coloc_mps.png", dpi=140, bbox_inches="tight", facecolor="white")
    print("wrote figures/inf_coloc_mps.png")


if __name__ == "__main__":
    main()
