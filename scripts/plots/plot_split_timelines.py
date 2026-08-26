#!/usr/bin/env python3
"""Per-second DCGM timelines at three sweep points, both GPUs. Generates two figures:
one pairing SM *active* with DRAM active + VRAM, one pairing SM *occupancy* with them."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"
BLUE, RED, GREY = "#1f6feb", "#c1440e", "#57606a"
BUDGETS = (30, 22, 18)

# (metric column in gpumon csv, line label, output file)
GREEN = "#2e7d4f"

for outname in ("split_lmcache_timelines_occ.png",):
    fig, axes = plt.subplots(len(BUDGETS), 2, figsize=(13, 10.8), sharex="row")
    _legend_handles, _legend_labels = [], []
    for row, b in enumerate(BUDGETS):
        name = f"split_lmcache_zipf_b{b}_fwd_ng"
        data = {0: [], 1: []}
        for line in open(OUT / "gpumon" / f"{name}.csv"):
            f = line.strip().split(",")
            if len(f) != 6 or f[0] == "ts":
                continue
            data[int(f[1])].append((float(f[0]), float(f[2]), float(f[3]),
                                    float(f[4]), float(f[5])))
        t0 = min(d[0][0] for d in data.values() if d)
        rec = json.load(open(OUT / "raw" / f"{name}.json"))
        m0 = rec["startup"]["startup_s"] + rec["warmup"]["duration_s"] + 4
        m1 = m0 + rec["summary"]["duration_s"]

        for col, (gpu, role) in enumerate(((0, "prefill GPU"), (1, "decode GPU"))):
            ax = axes[row][col]
            t = [x[0] - t0 for x in data[gpu]]
            ax.plot(t, [x[1] for x in data[gpu]], color=GREEN, lw=1.0, alpha=0.8,
                    label="SM active")
            ax.plot(t, [x[2] for x in data[gpu]], color=BLUE, lw=1.1,
                    label="SM occupancy")
            ax.plot(t, [x[3] for x in data[gpu]], color=RED, lw=1.1, label="DRAM active")
            ax.set_ylim(0, 1.02)
            ax.axvspan(m0, m1, color="#888888", alpha=0.08)
            ax2 = ax.twinx()
            ax2.plot(t, [x[4] / 1024 for x in data[gpu]], color=GREY, lw=1.3, ls="--",
                     label="VRAM used")
            ax2.set_ylim(0, 33)
            if col == 0:
                ax.set_ylabel(f"{b} GiB budget\nfraction")
            else:
                ax2.set_ylabel("VRAM used (GiB)", color=GREY)
            if row == 0:
                ax.set_title(role, fontsize=11)
            if row == 0 and col == 0:
                _legend_handles[:] = (ax.get_legend_handles_labels()[0]
                                      + ax2.get_legend_handles_labels()[0])
                _legend_labels[:] = (ax.get_legend_handles_labels()[1]
                                     + ax2.get_legend_handles_labels()[1])
            if row == len(BUDGETS) - 1:
                ax.set_xlabel("seconds since launch (shaded: measured load)")
            ax.grid(alpha=0.2)
    fig.suptitle("Per-second telemetry at three decode budgets; shaded region is the "
                 "measured 2 QPS load", fontsize=12, y=0.995)
    # one global legend, centered directly under the title, above all panels
    fig.legend(_legend_handles, _legend_labels, ncol=4, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 0.972), frameon=False,
               columnspacing=2.0, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(FIGS / outname, dpi=130, bbox_inches="tight")
    print("wrote", FIGS / outname)
