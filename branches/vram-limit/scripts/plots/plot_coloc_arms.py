#!/usr/bin/env python3
"""Colocation study, first measured arms: the decode-cost vs FT-progress trade-off.

Left: the frontier — each MPS cap as a point (x = fine-tune wall-clock progress,
y = decode TPOT). Right: how much of its capped solo ceiling the trainer keeps
while decode runs beside it. Data: output/raw/coloc_lmcache_zipf_b26_g2mps*.json,
the split study's b26 row as the decode-alone reference, and the solo trainer
calibrations in output/inf_ft_coloc/. Run from the repo root.
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRY = "#57606a"
BLU = "#1f6feb"
ORG = "#c1440e"
AMB = "#9a6700"

TOK_PER_STEP = 2 * 512


def tpot_ms(rec):
    s = rec["summary"]
    if s.get("tpot_ms", {}).get("mean") is not None:
        return s["tpot_ms"]["mean"]
    vals = [(r["e2e"] - r["ttft"]) / 127 * 1e3 for r in rec.get("records", [])
            if r.get("e2e") and r.get("ttft")]
    if vals:
        return round(float(np.mean(vals)), 2)
    return s["itl_ms"]["mean"]   # equal to mean TPOT for fixed 128-token outputs


def ft_wall_rate(rec):
    """Fine-tune progress in tok/s of wall-clock over the measured window."""
    steps = rec["ft"]["measure_window"]["steps"]
    dur = rec["t_measure1"] - rec["t_measure0"]
    return steps * TOK_PER_STEP / dur


def solo_rate(path):
    dts = [float(r["dt_ms"]) for r in csv.DictReader(open(path))]
    return TOK_PER_STEP / (sum(dts) / len(dts) / 1e3)


ref = json.load(open("output/raw/split_lmcache_zipf_b26_fwd_ng.json"))
arms = {p: json.load(open(f"output/raw/coloc_lmcache_zipf_b26_g2mps{p}.json"))
        for p in (10, 50, 100)}
solo = {p: solo_rate(f"output/inf_ft_coloc/ft_gpt2_mps{p}_solo.csv") for p in (10, 50, 100)}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                              gridspec_kw={"width_ratios": [1.15, 1]})

# ---- left: TTFT, TPOT (normalized to the decode-alone baseline) and throughput,
# all against the trainer's MPS share
GRN = "#2e7d4f"
xpos_l = np.arange(4)
xticks_l = ["decode alone\n(split study b26)", "10%", "50%", "no cap\n(100%)"]
tpot = [tpot_ms(ref)] + [tpot_ms(arms[p]) for p in (10, 50, 100)]
ttft = [ref["summary"]["ttft_ms"]["p50"]] + \
       [arms[p]["summary"]["ttft_ms"]["p50"] for p in (10, 50, 100)]
tput = [ref["summary"]["output_tok_per_s"]] + \
       [arms[p]["summary"]["output_tok_per_s"] for p in (10, 50, 100)]

ax.plot(xpos_l, [v / tpot[0] for v in tpot], "-o", color=BLU, lw=1.8, ms=7,
        zorder=3, label="TPOT / baseline")
ax.plot(xpos_l, [v / ttft[0] for v in ttft], "-^", color=ORG, lw=1.8, ms=7,
        zorder=3, label="TTFT p50 / baseline")
ax.axhline(1.0, color=GRY, lw=0.8, ls=":")
ax.set_xticks(xpos_l, xticks_l, fontsize=9)
ax.set_xlabel("trainer's MPS SM share", fontsize=10)
ax.set_ylabel("latency, × the decode-alone baseline", fontsize=10)
ax.set_ylim(0.9, 2.0)
ax.set_title("Decode pays in latency, never in throughput", fontsize=11, color=GRY)
ax.set_xlim(-0.4, 3.4)
ax.grid(alpha=0.25)

# throughput on a second axis: flat at the offered rate in every arm -- the open
# loop pins it there while capacity exceeds 2 QPS x 128 tok
ax_t = ax.twinx()
ax_t.plot(xpos_l, tput, "-s", color=GRN, lw=1.6, ms=5.5, alpha=0.85)
ax_t.set_ylim(0, 280)
ax_t.set_ylabel("decode throughput, tok/s", fontsize=10, color=GRN)
ax_t.tick_params(axis="y", labelcolor=GRN)
ax.plot([], [], "-s", color=GRN, ms=5.5, label="throughput (right axis)")
ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.62, 0.03))

# ---- right: FT achieved vs its capped solo ceiling
caps = (10, 50, 100)
xpos = np.arange(len(caps))
ceil = [solo[p] / 1e3 for p in caps]
got = [ft_wall_rate(arms[p]) / 1e3 for p in caps]
ax2.bar(xpos - 0.18, ceil, 0.36, color="#e8ddcf", edgecolor=AMB, lw=1.2,
        label="solo at same cap (ceiling)")
ax2.bar(xpos + 0.18, got, 0.36, color=AMB, label="beside decode (achieved)")
for i, (c, g) in enumerate(zip(ceil, got)):
    ax2.text(i + 0.18, g + 1.2, f"{g / c * 100:.0f}%", ha="center", fontsize=9,
             color=AMB)
ax2.set_xticks(xpos, ["10%", "50%", "no cap\n(100%)"], fontsize=9)
ax2.set_xlabel("trainer's MPS SM share", fontsize=10)
ax2.set_ylabel("fine-tune tok/s (thousands)", fontsize=10)
ax2.set_title("The trainer keeps 72–77% of its capped ceiling", fontsize=11,
              color=GRY)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.grid(alpha=0.25, axis="y")

fig.suptitle("GPT-2 LoRA colocated with split-decode (26 GiB budget), spatial sharing "
             "via MPS — reference workload, 2 QPS", fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig("figures/coloc_mps_arms.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("wrote figures/coloc_mps_arms.png")
