#!/usr/bin/env python3
"""Storyline figure: decode alone on the 11-minute workload, GPU memory given in full against GPU memory following the load.
Reads the two existing runs of round 1 (coloc_adjust/results_600w/{decode_alone,adjust_notenant}_*); runs nothing.
  python3 storyline/make_baseline_fig.py        ->  storyline/figures/baseline_compare_timeline.png"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "coloc_adjust", "results_600w")
GiB = 1024 ** 3; TOTAL = 31.3536; FIXED = 17.7   # card, and the engine's weights + runtime as plotted in the per-run timelines of round 1
SCHED = [(0, 1), (90, 2), (180, 3), (270, 1), (360, 4), (420, 0.5), (480, 2), (510, 3), (540, 2), (570, 1), (600, 1)]; END = 660
C_KV, C_LIM, C_FREE, C_FULL, C_FOL = "#8ab4f8", "#174ea6", "#fde9c8", "#9aa0a6", "#1a73e8"

def load(arm):
    P = lambda s: os.path.join(D, f"{arm}_{s}")
    t0 = float(open(P("t0.txt")).read())
    reqs = [json.loads(l) for l in open(P("load.jsonl"))]; reqs = [r for r in reqs if not r["rid"].startswith("w")]
    R = pd.DataFrame([dict(t_send=r["t_send"] - t0, ok=bool(r["status"] == 200 and not r["error"] and r["tok_t"]),
                           tpot=((r["t_done"] - r["tok_t"][0]) / max(1, r["n_out"] - 1)) if r["tok_t"] else np.nan) for r in reqs])
    samp = pd.read_csv(P("sampler.csv")); samp["rel"] = samp.t - t0
    ctl = pd.read_csv(P("controller.csv")); ctl["rel"] = ctl.t - t0
    return R, samp, ctl
num = lambda df, c: pd.to_numeric(df[c], errors="coerce")
tt = np.arange(0, END, 1.0)
def rolling_tpot(R, win=10.0):
    d = R[R.ok].sort_values("t_send")
    return np.array([d[(d.t_send > t - win) & (d.t_send <= t)].tpot.median() * 1e3 for t in tt])

runs = {"full": load("decode_alone"), "follow": load("adjust_notenant")}
fig, ax = plt.subplots(4, 1, figsize=(12, 9.6), sharex=True, gridspec_kw=dict(height_ratios=[0.6, 1.5, 1.5, 1.3], hspace=0.30))
fig.subplots_adjust(top=0.948, bottom=0.06, left=0.075, right=0.775)
def title(a, s): a.set_title(s, loc="left", fontsize=10.5, fontweight="bold", pad=5)
def legend(a, handles): a.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9, handlelength=1.6)

# (1) the workload
rate = np.array([[v for t, v in SCHED if x >= t][-1] for x in tt])
ax[0].fill_between(tt, 0, rate, step="post", color="#bdc1c6")
for (ta, v), tb in zip(SCHED, [t for t, _ in SCHED[1:]] + [END]):
    ax[0].text((ta + tb) / 2, v + 0.25, f"{v:g}", ha="center", va="bottom", fontsize=8.5)
ax[0].set_ylim(0, 5.6); ax[0].set_yticks([0, 2, 4]); ax[0].set_ylabel("req/s")
title(ax[0], "Workload: arrival rate (the same request sequence in both runs)")

# (2, 3) GPU memory, one panel per run, same axes
for a, key, name in ((ax[1], "full", "Run A · full memory: the KV limit is fixed at 13 GiB"), (ax[2], "follow", "Run B · follows the load: the VRAM scheduler moves the KV limit")):
    R, samp, ctl = runs[key]
    kv = (num(samp, "kvc_used") / GiB).rolling(5, min_periods=1).mean()
    lim = np.interp(samp.rel, ctl.rel, num(ctl, "limit_gib"))
    a.fill_between(samp.rel, FIXED + lim, TOTAL, step="post", color=C_FREE, lw=0)
    a.fill_between(samp.rel, FIXED, FIXED + kv, color=C_KV, lw=0)
    a.step(samp.rel, FIXED + lim, where="post", color=C_LIM, lw=1.3)
    a.axhline(TOTAL, color="#5f6368", lw=0.8, ls="--"); a.axhline(FIXED, color="#5f6368", lw=0.8, ls=":")
    a.set_ylim(FIXED - 0.6, TOTAL + 0.8); a.set_yticks(range(18, 32, 2)); a.set_ylabel("GPU memory (GiB)"); title(a, name)
    legend(a, [Patch(color=C_KV, label="KV in use"), Line2D([], [], color=C_LIM, lw=1.3, label="KV limit"),
               Patch(color=C_FREE, label="above the KV limit:\nnot reserved for decode"),
               Line2D([], [], color="#5f6368", lw=0.8, ls="--", label="whole card, 31.35 GiB"),
               Line2D([], [], color="#5f6368", lw=0.8, ls=":", label="engine weights + runtime\n(17.7 GiB) below this line")])

# (4) decode TPOT, both runs on one axis
ax[3].plot(tt, rolling_tpot(runs["full"][0]), color=C_FULL, lw=4.0, solid_capstyle="round")
ax[3].plot(tt, rolling_tpot(runs["follow"][0]), color=C_FOL, lw=1.3)
ax[3].set_ylabel("decode TPOT (ms)"); ax[3].set_ylim(10, 28); title(ax[3], "Decode TPOT, 10-s median: the two runs on one axis")
two = [Line2D([], [], color=C_FULL, lw=4.0, label="Run A · full memory"), Line2D([], [], color=C_FOL, lw=1.3, label="Run B · follows the load")]
def note(s): return Line2D([], [], color="none", label=s)
def in_run(R): return R[R.ok & (R.t_send >= 0) & (R.t_send < END)]
mA, mB = in_run(runs["full"][0]).tpot.mean() * 1e3, in_run(runs["follow"][0]).tpot.mean() * 1e3
legend(ax[3], two + [note(f"mean over all {len(in_run(runs['full'][0])):,} requests:\nA {mA:.2f} ms, B {mB:.2f} ms\nB against A: {(mB / mA - 1) * 100:+.2f} %")])

ax[3].set_xlabel("time since the end of warm-up (s)"); ax[3].set_xlim(0, END)
print(f"TPOT mean A {mA:.3f} B {mB:.3f} ({(mB / mA - 1) * 100:+.3f} %), n {len(in_run(runs['full'][0]))}/{len(in_run(runs['follow'][0]))}")

for a in ax:
    a.grid(True, axis="y", alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    for t, _ in SCHED[1:]: a.axvline(t, color="k", ls=":", lw=0.5, alpha=0.5)
out = os.path.join(H, "figures", "baseline_compare_timeline.png"); fig.savefig(out, dpi=140); print(out)
