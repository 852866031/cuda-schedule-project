#!/usr/bin/env python3
"""Storyline figure: how fast decode is when it may use only N of the 170 SMs. Runs nothing: reads the existing measurement
coloc_adjust/results/knee_probe.csv (coloc_adjust/knee_probe.py: one synthetic decode step with Llama-3-8B's shapes and random
weights, 6272-token context per request, alone on the GPU, in a green context of N SMs; median of the repeats of each point).
This is the simulated decode step, not the vLLM engine.
  python3 storyline/make_sm_fig.py   ->  storyline/figures/decode_vs_sms.png"""
import os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(H, "..", "coloc_adjust", "results", "knee_probe.csv")); d = d[d.kind == "decode"]
med = d.groupby(["B", "sms"]).step_ms.median().unstack(0)   # rows: SMs, columns: requests in the batch
rel = med.loc[med.index.max()] / med * 100                   # speed against all 170 SMs
COL = {1: "#9ecae1", 4: "#4292c6", 8: "#2171b5", 14: "#08306b"}   # one hue, darker = more requests in the batch
fig, ax = plt.subplots(figsize=(9.6, 4.6)); fig.subplots_adjust(top=0.90, bottom=0.14, left=0.085, right=0.73)
for B in sorted(rel.columns):
    ax.plot(rel.index, rel[B], "-o", color=COL[B], lw=2, ms=5, markeredgecolor="white", markeredgewidth=0.8,
            label=f"{B} request{'s' if B > 1 else ''} in the batch")
ax.axhline(100, color="#5f6368", lw=0.8, ls="--")
ax.axvline(96, color="#5f6368", lw=1.0, ls="--")
ax.set_xlim(0, 176); ax.set_ylim(0, 120); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_xticks([8, 32, 56, 80, 96, 128, 170])
ax.set_xlabel("SMs decode may use (the card has 170)"); ax.set_ylabel("decode speed, % of the speed with all 170 SMs")
ax.set_title("Decode speed against the number of SMs it may use", loc="left", fontsize=11, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9.5, title="one simulated decode step,\ndecode alone on the GPU", title_fontsize=9, alignment="left")
out = os.path.join(H, "figures", "decode_vs_sms.png"); fig.savefig(out, dpi=140); print(out)
print(rel.round(0).astype(int).to_string())
