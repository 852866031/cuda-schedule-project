#!/usr/bin/env python3
"""KV-slot study, report §6.2: what `use_layerwise` measurably changes, per LMCache copy
path, at 24 GiB / 2.5 req/s (means over 300 requests, from the engine traces).
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
ARMS = [("Python fallback", ["kv_base_b24_q2.5_tr"], ["kv_lw_b24_q2.5_tr", "kv_lw_b24_q2.5_tr_r2"]),
        ("compiled kernels", ["kv_cops_b24_q2.5_tr_r2"], ["kv_cops_lw_b24_q2.5_tr"])]


def mean_split(names):
    """per-request queue wait, load and decode straight from the joined timeline"""
    rq = [r for n in names for r in csv.DictReader(open(TL / f"{n}_req.csv")) if r.get("decode")]
    return {k: float(np.mean([float(r[c]) for r in rq])) for k, c in
            (("queue", "wait"), ("load", "load"), ("decode", "decode"))}


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.3})
xs, labs, vals = [], [], []
x = 0
for path, base, lw in ARMS:
    for arm, names in (("baseline", base), ("use_layerwise", lw)):
        xs.append(x); labs.append(f"{arm}\n({len(names)} run{'s' if len(names) > 1 else ''})")
        vals.append(mean_split(names)); x += 1
    x += 0.6
centers = [(xs[0] + xs[1]) / 2, (xs[2] + xs[3]) / 2]

ax = axes[0]
q = [v["queue"] for v in vals]
ax.bar(xs, q, color=C_Q, width=0.75)
for xi, v in zip(xs, q):
    ax.text(xi, v * 1.08, f"{v * 1e3:,.0f} ms", ha="center", fontsize=9.5)
ax.set_yscale("log"); ax.set_ylim(0.05, 12)
ax.set_ylabel("mean decode queue wait (s, log)")
ax.set_title("(a) waiting for a slot", loc="left")

ax = axes[1]
ld = np.array([v["load"] for v in vals]); dc = np.array([v["decode"] for v in vals])
ax.bar(xs, ld, color=C_L, width=0.75, label="own KV load (admitted → token 1)")
ax.bar(xs, dc, bottom=ld, color=C_D, width=0.75, label="decode (token 1 → last, incl. others' loads)")
for xi, a, b in zip(xs, ld, dc):
    ax.text(xi, a + b + 0.06, f"{a + b:.2f} s", ha="center", fontsize=9.5)
ax.set_ylim(0, 4.6)
ax.set_ylabel("time per request once admitted (s)")
ax.set_title("(b) holding a slot (residence)", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=9)

for ax in axes:
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ececea", lw=0.6)
    for c, (path, _, _) in zip(centers, ARMS):
        ax.text(c, -0.25, path, transform=ax.get_xaxis_transform(), ha="center", fontsize=10.5,
                fontweight="bold")
fig.suptitle("use_layerwise, measured (24 GiB, 2.5 req/s): a large gain on the fallback, none with compiled kernels",
             fontsize=12.5, y=1.0)
fig.savefig(REPO / "figures" / "kv_layerwise_measured.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_layerwise_measured.png")
