#!/usr/bin/env python3
"""KV-slot study, report §6.2: what `use_layerwise` measurably changes with LMCache's
compiled kernels, at 24 GiB / 2.5 req/s (means over 300 requests, from the engine traces).
(a) decode queue wait; (b) a request's time on the decode engine once admitted:
own KV load (admitted → token 1) + decode (token 1 → last, incl. stalls from others' loads).

    .venv/bin/python scripts/plots/plot_kv_layerwise_measured.py -> figures/kv_layerwise_measured.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
TL = REPO / "output" / "kv_vram" / "timeline"
C_Q, C_L, C_D = "#9a9a95", "#eb6834", "#2a78d6"
ARMS = [("baseline", ["kv_cops_b24_q2.5_tr_r2"]), ("use_layerwise", ["kv_cops_lw_b24_q2.5_tr"])]


def mean_split(names):
    """per-request queue wait, load and decode straight from the joined timeline"""
    rq = [r for n in names for r in csv.DictReader(open(TL / f"{n}_req.csv")) if r.get("decode")]
    return {k: float(np.mean([float(r[c]) for r in rq])) for k, c in
            (("queue", "wait"), ("load", "load"), ("decode", "decode"))}


fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"wspace": 0.35})
xs = [0, 1]
labs = [a for a, _ in ARMS]
vals = [mean_split(n) for _, n in ARMS]

ax = axes[0]
q = [v["queue"] for v in vals]
ax.bar(xs, q, color=C_Q, width=0.75)
for xi, v in zip(xs, q):
    ax.text(xi, v + 0.004, f"{v * 1e3:,.0f} ms", ha="center", fontsize=9.5)
ax.set_ylim(0, 0.2)
ax.set_ylabel("mean decode queue wait (s)")
ax.set_title("(a) waiting for a slot", loc="left")

ax = axes[1]
ld = np.array([v["load"] for v in vals]); dc = np.array([v["decode"] for v in vals])
ax.bar(xs, ld, color=C_L, width=0.75, label="own KV load (admitted → token 1)")
ax.bar(xs, dc, bottom=ld, color=C_D, width=0.75, label="decode (token 1 → last, incl. others' loads)")
for xi, a, b in zip(xs, ld, dc):
    ax.text(xi, a + b + 0.06, f"{a + b:.2f} s", ha="center", fontsize=9.5)
ax.set_ylim(0, 3.6)
ax.set_ylabel("time per request once admitted (s)")
ax.set_title("(b) holding a slot (residence)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)

for ax in axes:
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=10)
    ax.set_xlim(-0.6, 1.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ececea", lw=0.6)
fig.suptitle("use_layerwise, measured (24 GiB, 2.5 req/s, compiled LMCache kernels)", fontsize=12.5, y=1.0)
fig.savefig(REPO / "figures" / "kv_layerwise_measured.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_layerwise_measured.png")
