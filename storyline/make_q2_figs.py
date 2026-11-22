#!/usr/bin/env python3
"""Storyline figures for question 2 (how the tenant shares compute with decode). Runs nothing: reads the round-2 runs in
coloc_adjust/results/ (11-minute workload, Nov 9, GPU 0, 500 W).
  python3 storyline/make_q2_figs.py   ->  figures/q2_gate.png, q2_mps20.png, q2_green.png, q2_slo.png   (one schematic per method, not measured)
                                          figures/q2_compare.png   (measured: decode TPOT, average and p90, and tenant throughput)"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "coloc_adjust", "results"); F = os.path.join(H, "figures")
C_DEC, C_TEN, INK, MUTE, BLUE_D, ORG_D = "#8ab4f8", "#e37400", "#202124", "#5f6368", "#174ea6", "#8a4600"

# ------------------------------------------------------------------ 1. one schematic per method (not measured)
# Every schematic has the same two lanes: decode on top, the tenant below. In a lane, the height of the colour is the share
# of the SMs the task may use (0 to 100 %) and the horizontal direction is time.
T = 100.0
def lanes(title, extra_top=False, h=3.3):
    n = 3 if extra_top else 2
    fig, ax = plt.subplots(n, 1, figsize=(10.5, h + (0.9 if extra_top else 0)), sharex=True,
                           gridspec_kw=dict(height_ratios=([0.7, 1, 1] if extra_top else [1, 1]), hspace=0.36))
    fig.subplots_adjust(top=0.80 if not extra_top else 0.84, bottom=0.09, left=0.17, right=0.76)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905 if not extra_top else 0.925, "height = share of the SMs the task may use;  solid orange = decode's SMs,  dashed orange = the tenant's SMs", ha="left", va="center", fontsize=9, color=MUTE)
    for a, name in zip(ax[-2:], ("decode", "tenant")):
        a.set_xlim(0, T); a.set_ylim(0, 1.0); a.set_yticks([0, 1.0]); a.set_yticklabels(["0", "100 %"], fontsize=8.5); a.set_xticks([])
        a.spines[["top", "right"]].set_visible(False)
        a.text(-0.185, 0.5, name, transform=a.transAxes, ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    ax[-1].set_xlabel("time  →", fontsize=9)
    return fig, ax
def decode_steps(a, note="all SMs"):
    x = 0.0
    while x < T:
        a.add_patch(Rectangle((x, 0), 2.3, 1.0, color=C_TEN, lw=0)); x += 2.5
    side(a, note)
def ten(a, x, w, h):   # the tenant's SMs
    a.add_patch(Rectangle((x, 0), w, h, facecolor="#fde9c8", edgecolor=C_TEN, hatch="///", ls="--", lw=1.3))
def side(a, s, weight="normal"): a.text(1.03, 0.5, s, transform=a.transAxes, ha="left", va="center", fontsize=9.5, color=INK, linespacing=1.4, fontweight=weight)
def save(fig, name):
    fig.text(0.995, 0.012, "schematic, not measured", ha="right", va="bottom", fontsize=8, color=MUTE, style="italic")
    out = os.path.join(F, name); fig.savefig(out, dpi=140); plt.close(fig); print(out)

# (a) idle gate: the tenant may run only in the gaps between decode steps
fig, ax = lanes("Idle gate: the tenant runs only when no decode step is on the GPU")
decode_steps(ax[0])
ax[1].vlines(np.arange(2.4, T, 2.5), 0, 1.0, color=C_TEN, lw=1.0, ls=(0, (3, 2)))
side(ax[1], "only in the gaps\nbetween decode steps")
save(fig, "q2_gate.png")

# (b) MPS limit of 20 %: decode keeps all SMs, the tenant gets a fixed 20 %
fig, ax = lanes("MPS limit of 20 %: the tenant always runs, on a fixed 20 % of the SMs")
decode_steps(ax[0], "100 % of the SMs")
ten(ax[1], 0, T, 0.20)
ax[1].set_yticks([0, 0.2, 1.0]); ax[1].set_yticklabels(["0", "20 %", "100 %"], fontsize=8.5)
side(ax[1], "20 % of the SMs,\nall the time")
save(fig, "q2_mps20.png")

# (c) dynamic green context: the tenant's SMs go down as the arrival rate goes up
fig, ax = lanes("Dynamic green context: the tenant's SMs follow the arrival rate", extra_top=True)
SEG = ((0, 25, 0.5, 96), (25, 50, 1, 64), (50, 75, 2, 32), (75, 100, 3, 16))
ax[0].set_ylim(0, 4.2); ax[0].set_yticks([]); ax[0].spines[["top", "right", "left"]].set_visible(False)
ax[0].text(-0.185, 0.5, "arrival\nrate", transform=ax[0].transAxes, ha="left", va="center", fontsize=11, fontweight="bold", color=INK, linespacing=1.1)
for xa, xb, r, sm in SEG:
    ax[0].add_patch(Rectangle((xa, 0), xb - xa, r, color="#bdc1c6", lw=0))
    ax[0].text((xa + xb) / 2, r + 0.2, f"{r:g} req/s" + (" or more" if r == 3 else ""), ha="center", va="bottom", fontsize=9, color=INK)
    ten(ax[2], xa, xb - xa, sm / 170)
    ax[2].text((xa + xb) / 2, sm / 170 + 0.05, f"{sm} SMs", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
side(ax[0], "goes up  →")
decode_steps(ax[1], "all 170 SMs")
side(ax[2], "fewer SMs as the\nrate goes up (of 170)")
save(fig, "q2_green.png")

# (d) latency SLO: the tenant uses all SMs but sleeps; the sleep follows decode's measured slowdown
fig, ax = lanes("Latency SLO: the tenant uses all SMs but sleeps between steps", extra_top=True)
xs = np.linspace(0, T, 400); slow = 1.0 + 0.55 * np.tanh((38 - xs) / 9.0) + 0.06 * np.sin(xs / 3.1)   # above the target at first, under it later
ax[0].plot(xs, slow, color="#3c4043", lw=1.6); ax[0].axhline(1.0, color=INK, lw=0.9, ls="--")
ax[0].set_ylim(-0.6, 2.3); ax[0].set_yticks([1.0]); ax[0].set_yticklabels(["target"], fontsize=8.5); ax[0].spines[["top", "right"]].set_visible(False)
ax[0].text(-0.185, 0.5, "decode\nslow-\ndown", transform=ax[0].transAxes, ha="left", va="center", fontsize=11, fontweight="bold", color=INK, linespacing=1.05)
ax[0].text(16, 2.05, "over the target", ha="center", va="center", fontsize=9, color=INK); ax[0].text(72, -0.2, "under the target", ha="center", va="center", fontsize=9, color=INK)
side(ax[0], "measured every 2 s\n(target: +15 % or +5 %)")
decode_steps(ax[1], "all SMs")
for xa, w in ((0, 4), (13, 4), (26, 4), (39, 6), (49, 7), (59, 8), (70, 8), (81, 8), (92, 8)): ten(ax[2], xa, w, 1.0)
ax[2].text(19, 1.06, "long sleeps", ha="center", va="bottom", fontsize=9, color=INK); ax[2].text(72, 1.06, "short sleeps", ha="center", va="bottom", fontsize=9, color=INK)
side(ax[2], "all SMs while it works;\nover the target → sleep more,\nunder the target → sleep less")
save(fig, "q2_slo.png")

# ------------------------------------------------------------------ 2. measured comparison
SCHED = [(0, 1), (90, 2), (180, 3), (270, 1), (360, 4), (420, 0.5), (480, 2), (510, 3), (540, 2), (570, 1), (600, 1)]; END = 660; SKIP = 10
PH = [(ta, tb, v) for (ta, v), tb in zip(SCHED, [t for t, _ in SCHED[1:]] + [END])]
def tpot(arm):
    t0 = float(open(os.path.join(D, f"{arm}_t0.txt")).read()); rows = []
    for l in open(os.path.join(D, f"{arm}_load.jsonl")):
        r = json.loads(l)
        if r["rid"].startswith("w") or not r["tok_t"] or r["status"] != 200 or r["error"]: continue
        rows.append((r["t_send"] - t0, (r["t_done"] - r["tok_t"][0]) / max(1, r["n_out"] - 1) * 1e3))
    return np.array(rows)
def by_rate(arm, q):   # mean over the phases at one arrival rate of the per-phase statistic (the method of reports/1, sections 2.4 and 2.5); q = "mean" or a percentile
    R = tpot(arm); out = {}
    stat = (lambda v: float(np.mean(v))) if q == "mean" else (lambda v: float(np.percentile(v, q)))
    for rate in (0.5, 1, 2):
        v = [stat(R[(R[:, 0] >= a + SKIP) & (R[:, 0] < b), 1]) for a, b, r in PH if r == rate and ((R[:, 0] >= a + SKIP) & (R[:, 0] < b)).sum() > 3]
        out[rate] = float(np.mean(v))
    return out
def tenant(arm):
    p = pd.read_csv(os.path.join(D, f"{arm}_phases.csv")); return {r: float(p[p.rps == r].ft_tok_s.mean()) / 1e3 for r in (0.5, 1, 2)}
ARMS = [("adjust_mps20", "MPS limit of 20 %", "#0072B2", "MPS\n20 %"), ("green_dyn2", "dynamic green context", "#009E73", "green\ncontext"),
        ("tpot_slo15", "latency SLO, 15 %", "#E69F00", "SLO\n15 %"), ("tpot_slo5", "latency SLO, 5 %", "#CC79A7", "SLO\n5 %")]
rates = [0.5, 1, 2]
bm, b90 = by_rate("adjust_notenant", "mean"), by_rate("adjust_notenant", 90)
fig, ax = plt.subplots(1, 4, figsize=(15, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15])); fig.subplots_adjust(top=0.82, bottom=0.27, left=0.045, right=0.995, wspace=0.22)
w = 0.38
for k, (arm, name, col, short) in enumerate(ARMS):
    sm, s90, ft = by_rate(arm, "mean"), by_rate(arm, 90), tenant(arm)
    avg = [(sm[r] / bm[r] - 1) * 100 for r in rates]; p90 = [(s90[r] / b90[r] - 1) * 100 for r in rates]
    print(f"{arm:14s} avg " + " ".join(f"{v:+.1f}%" for v in avg) + "   p90 " + " ".join(f"{v:+.1f}%" for v in p90) + "   tenant " + " ".join(f"{ft[r]:.1f}k" for r in rates))
    for j, a in enumerate(ax[:3]):
        a.bar(k - w / 2 - 0.01, avg[j], w, color=col, lw=0)
        a.bar(k + w / 2 + 0.01, p90[j], w, color=col, lw=0, alpha=0.45, hatch="///", edgecolor=col)
        a.text(k - w / 2 - 0.01, avg[j] + 0.5, f"{avg[j]:+.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        a.text(k + w / 2 + 0.01, p90[j] + 0.5, f"{p90[j]:+.0f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    x = np.arange(3) + (k - 1.5) * 0.2
    ax[3].bar(x, [ft[r] for r in rates], 0.185, color=col, lw=0)
    for xi, r in zip(x, rates): ax[3].text(xi, ft[r] + 0.2, f"{ft[r]:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
for a, r in zip(ax[:3], rates):
    a.set_title(f"Decode TPOT at {r:g} req/s", loc="left", fontsize=10.5, fontweight="bold"); a.set_ylim(0, 34)
    a.set_xticks(range(4)); a.set_xticklabels([sh for *_, sh in ARMS], fontsize=9)
ax[0].set_ylabel("slower than with no tenant (%)")
ax[3].set_title("Tenant throughput", loc="left", fontsize=10.5, fontweight="bold"); ax[3].set_ylim(0, 14); ax[3].set_ylabel("thousand tokens / s")
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["0.5 req/s", "1 req/s", "2 req/s"])
for a in ax: a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
fig.legend(handles=[Patch(color="#5f6368", label="average (mean TPOT)"),
                    Patch(facecolor="#5f6368", alpha=0.45, hatch="///", edgecolor="#5f6368", label="tail (p90 TPOT)")],
           loc="upper left", bbox_to_anchor=(0.04, 0.99), ncol=2, frameon=False, fontsize=10, title="each against the same statistic with no tenant", title_fontsize=8.5, alignment="left")
fig.legend(handles=[Patch(color=c, label=n) for _, n, c, _ in ARMS], loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False, fontsize=9.5)
fig.text(0.5, 0.02, "3 and 4 req/s are left out: decode needs the whole GPU memory there, so the tenant is paused in every method.", ha="center", fontsize=8.5, color=MUTE)
out = os.path.join(F, "q2_compare.png"); fig.savefig(out, dpi=140); print(out)
