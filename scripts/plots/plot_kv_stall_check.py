#!/usr/bin/env python3
"""KV-slot study: the cohort-admission check (DECISIONS, "Check of the §4.3 explanation").

  (a) per residence, from the traces: number of load stalls met × their mean length =
      stall time, for 24 GiB (lockstep turnover), 28 GiB as is, 28 GiB with the router
      releasing decode legs in groups of 5. Grouping halves the count and doubles the
      length; the product does not move.
  (b) schematic: where the loads land. Lockstep (24 GiB): a whole batch finishes, the
      next is loaded while nobody decodes. Spread (28 GiB): loads land in the middle of
      other requests' decode, grouped or not.

    .venv/bin/python scripts/plots/plot_kv_stall_check.py  -> figures/kv_stall_check.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent.parent
TL = REPO / "output" / "kv_vram" / "timeline"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
C_LOAD, C_DEC, C_G = "#eb6834", "#2a78d6", "#9a9a95"
CELLS = [("kv_base_b24_q4_tr", "24 GiB\nas is"), ("kv_base_b28_q4_tr", "28 GiB\nas is"),
         ("kv_base_b28_q4_tr_r2", "28 GiB\nas is (rerun)"), ("kv_cohort5_b28_q4_tr", "28 GiB\ngroups of 5")]


def f(x):
    return float(x) if x not in ("", None) else np.nan


def split(name):
    rq = [r for r in csv.DictReader(open(TL / f"{name}_req.csv")) if r.get("decode")]
    st = list(csv.DictReader(open(TL / f"{name}_steps.csv")))
    ld = list(csv.DictReader(open(TL / f"{name}_loads.csv")))
    t = np.array([f(s["t"]) for s in st])
    dt = np.r_[np.nan, np.diff(t)]
    L0 = np.array([f(l["t"]) for l in ld])
    L1 = L0 + np.array([f(l["dt"]) for l in ld])
    nl = np.array([int(l["n"]) for l in ld])
    ov = np.zeros(len(t), bool)
    for i in range(1, len(t)):
        ov[i] = ((L0 < t[i]) & (L1 > t[i - 1])).any()
    g = dt < 0.9
    ns, stl = [], []
    for r in rq:
        a, b = f(r["t_first"]), f(r["t_fin"])
        m = g & (t > a) & (t <= b)
        ns.append((m & ov).sum())
        stl.append(dt[m & ov].sum())
    ns, stl = np.array(ns), np.array(stl)
    return {"stalls": ns.mean(), "stall_s": stl.mean(), "len_ms": 1e3 * stl.sum() / max(ns.sum(), 1),
            "per_call": nl.mean(), "per_req_ms": 1e3 * (L1 - L0).sum() / nl.sum()}


fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.1, 1.4]})
rows = [(lab, split(n)) for n, lab in CELLS if (TL / f"{n}_req.csv").exists()]
x = np.arange(len(rows))
ax.bar(x, [r["stall_s"] for _, r in rows], color=C_LOAD, width=0.6)
for i, (lab, r) in enumerate(rows):
    ax.text(i, r["stall_s"] + 0.03, f"{r['stalls']:.1f} stalls\n× {r['len_ms']:.0f} ms\n= {r['stall_s']:.2f} s",
            ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{lab}\n{r['per_call']:.1f} req/call\n{r['per_req_ms']:.0f} ms/req"
                    for lab, r in rows], fontsize=9)
ax.set_ylim(0, max(r["stall_s"] for _, r in rows) * 1.45)
ax.set_ylabel("time stalled by others' KV loads, per request (s)")
ax.set_title("(a) grouping admissions halves the stall count, doubles the stall length\n"
             "— stall time per residence does not move (4 req/s, traced)")
ax.grid(axis="y", color="#ececea", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)

# (b) schematic
bx.set_xlim(0, 100)
bx.set_ylim(0, 100)
bx.axis("off")
bx.set_title("(b) where the loads land, schematic")


def lanes(y0, n, segs, label, note):
    bx.text(0, y0 + n * 4.2 + 2.0, label, fontsize=10.5, fontweight="bold", va="bottom")
    for k in range(n):
        for (s, e, kind) in segs(k):
            bx.add_patch(Rectangle((s, y0 + k * 4.2), e - s, 3.4, fc=C_LOAD if kind == "l" else C_DEC,
                                   ec="white", lw=0.5))
    bx.text(0, y0 - 1.2, note, fontsize=9.5, va="top", ha="left", color="#444")


# lockstep: all lanes load together at t=0 and t=50, decode in between
def lock(k):
    return [(2, 10, "l"), (10, 50, "d"), (52, 60, "l"), (60, 100, "d")]


# spread: each lane offset; loads scattered
def spread(k):
    off = (k * 11) % 50
    out = []
    t = off
    while t < 100:
        out.append((t, min(t + 4, 100), "l"))
        out.append((min(t + 4, 100), min(t + 50, 100), "d"))
        t += 50
    return [(s, e, kd) for s, e, kd in out if e > s]


lanes(60, 8, lock, "lockstep turnover (24 GiB: ~10 slots, eviction re-syncs the batch)",
      "a whole batch finishes, the next is loaded while nobody decodes → a request's decode overlaps few loads")
lanes(12, 8, spread, "spread turnover (28 GiB: 15 slots, completions scattered)",
      "loads land in the middle of other requests' decode — grouped or not")
bx.text(0, 2, "orange = a KV load (every lane stops); blue = decoding; time →", fontsize=9, color="#444")
fig.suptitle("Why 24 GiB decodes faster per token at saturation: not fewer stalls, but loads that fall "
             "between batches instead of inside them", y=1.02)
fig.savefig(REPO / "figures" / "kv_stall_check.png", dpi=150, bbox_inches="tight")
print("-> figures/kv_stall_check.png")
