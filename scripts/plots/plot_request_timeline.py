#!/usr/bin/env python3
"""Request/session timeline: every measured request as a lifetime bar on its session's
row, at a healthy budget and at the floor."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT, FIGS = REPO / "output", REPO / "figures"
BLUE, RED, GREY = "#1f6feb", "#c1440e", "#57606a"

fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharey=True)
for ax, b in zip(axes, (30, 18)):
    rec = json.load(open(OUT / "raw" / f"split_lmcache_zipf_b{b}_fwd_ng.json"))
    rows = rec["records"] if "records" in rec else rec["summary"].get("records", [])
    if not rows:
        rows = rec.get("results", [])
    t0 = min(r["t_submit"] for r in rows)
    counts = {}
    for r in rows:
        counts[r["session_id"]] = counts.get(r["session_id"], 0) + 1
    # hottest session at the bottom row, coldest on top
    order = {sid: i for i, (sid, _) in enumerate(
        sorted(counts.items(), key=lambda kv: -kv[1]))}
    for r in rows:
        y = order[r["session_id"]]
        t = r["t_submit"] - t0
        ax.plot([t, t + r["e2e"] / 1000], [y, y], color=BLUE, lw=1.6, alpha=0.55,
                solid_capstyle="butt")
        ax.plot(t, y, marker="|", color=RED, ms=6, mew=1.4)
    ax.set_ylabel(f"{b} GiB budget\nsession (hottest at bottom)")
    ax.grid(alpha=0.2, axis="x")
    ax.set_xlim(left=0)
axes[0].set_title("each bar: one request, from arrival (red tick) to its last token — "
                  "rows are sessions", fontsize=11)
axes[1].set_xlabel("seconds since first measured request")
fig.tight_layout()
fig.savefig(FIGS / "split_request_timeline.png", dpi=130, bbox_inches="tight")
print("wrote", FIGS / "split_request_timeline.png")
