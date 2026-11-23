#!/usr/bin/env python3
"""Storyline figures for question 3 (what else the two tasks contend for). Runs nothing; reads existing results.
  python3 storyline/make_q3_figs.py  ->  figures/q3_observation.png   tenant on 32 SMs of its own: decode's memory and decode's speed
                                         figures/q3_host_copy.png     how much of the host <-> GPU link is used
                                         figures/q3_shared_path.png   schematic: separate SMs, one path to GPU memory
                                         figures/q3_bandwidth.png     decode's memory traffic and decode's speed against its SMs"""
import glob, json, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch, FancyArrowPatch

H = os.path.dirname(os.path.abspath(__file__)); CA = os.path.join(H, "..", "coloc_adjust"); F = os.path.join(H, "figures")
ORG, ORG_L, ORG_D, BLUE, BLUE_L, INK, MUTE, GREY = "#e37400", "#fde9c8", "#8a4600", "#1a73e8", "#8ab4f8", "#202124", "#5f6368", "#bdc1c6"
def clean(a): a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
def title(a, s): a.set_title(s, loc="left", fontsize=10.5, fontweight="bold")

# ---------------------------------------------------------------- 1. the observation (round 1: Nov 5, GPU 1, 600 W; tenant on a fixed 32-SM green context)
D = os.path.join(CA, "results_600w")
SCHED = [(0, 1), (90, 2), (180, 3), (270, 1), (360, 4), (420, 0.5), (480, 2), (510, 3), (540, 2), (570, 1), (600, 1)]; END = 660
PH = [(ta, tb, v) for (ta, v), tb in zip(SCHED, [t for t, _ in SCHED[1:]] + [END])]; RATES = [0.5, 1, 2]
def tpot(arm):   # TPOT with the queue wait in it: (last token - request sent) / output tokens
    t0 = float(open(os.path.join(D, f"{arm}_t0.txt")).read()); R = []
    for l in open(os.path.join(D, f"{arm}_load.jsonl")):
        r = json.loads(l)
        if r["rid"].startswith("w") or not r["tok_t"] or r["status"] != 200 or r["error"]: continue
        R.append((r["t_send"] - t0, (r["t_done"] - r["t_send"]) / max(1, r["n_out"]) * 1e3))
    return np.array(R)
def by_rate(arm, q):
    R = tpot(arm); st = (lambda v: float(np.mean(v))) if q == "mean" else (lambda v: float(np.percentile(v, q)))
    return {rate: float(np.mean([st(R[(R[:, 0] >= a + 10) & (R[:, 0] < b), 1]) for a, b, r in PH if r == rate])) for rate in RATES}
def phases(arm, col):
    p = pd.read_csv(os.path.join(D, f"{arm}_phases.csv")); return {r: float(p[p.rps == r][col].mean()) for r in RATES}
base, ten = "adjust_notenant", "green32"
avg = [(by_rate(ten, "mean")[r] / by_rate(base, "mean")[r] - 1) * 100 for r in RATES]
p90 = [(by_rate(ten, 90)[r] / by_rate(base, 90)[r] - 1) * 100 for r in RATES]
use_b, use_t, lim_b, lim_t = phases(base, "kvc_used_gib"), phases(ten, "kvc_used_gib"), phases(base, "limit_gib"), phases(ten, "limit_gib")
ft = phases(ten, "ft_tok_s")
print("observation  avg " + " ".join(f"{v:+.1f}%" for v in avg) + "  p90 " + " ".join(f"{v:+.1f}%" for v in p90))
print("  KV in use, no tenant / tenant: " + "  ".join(f"{use_b[r]:.2f}/{use_t[r]:.2f}" for r in RATES) + "   KV limit: " + "  ".join(f"{lim_b[r]:.1f}/{lim_t[r]:.1f}" for r in RATES)
      + "   tenant tok/s: " + " ".join(f"{ft[r]:.0f}" for r in RATES))
# what decode's own SM count predicts: the curve of step 7.0 (vLLM engine, decode alone, by the SMs it may use)
def sm_curve():
    runs = {}
    for p in sorted(glob.glob(os.path.join(CA, "results_sm", "eager_gpu*_*.json"))):
        gi, n = re.match(r"eager_gpu(\d)_(\w+)\.json", os.path.basename(p)).groups()
        runs.setdefault(int(gi), {})[170 if n == "full" else int(n)] = {int(c): v["median_ms"] for c, v in json.load(open(p)).items()}
    pts = {}
    for gi, dd in runs.items():
        for n, v in dd.items():
            if n != 170 or 170 not in pts: pts[n] = {c: (dd[170][c] / v[c] * 100 if v[c] else None) for c in v}
    return pts
pts = sm_curve(); speed = [100 / (1 + v / 100) for v in avg]   # decode's speed next to the tenant, % of its speed with no tenant
print("  decode speed next to the tenant: " + " ".join(f"{v:.1f}%" for v in speed))
fig = plt.figure(figsize=(13.5, 6.0)); gs = fig.add_gridspec(2, 2, height_ratios=[0.13, 1], width_ratios=[1.75, 1], hspace=0.06, wspace=0.16, top=0.875, bottom=0.115, left=0.06, right=0.985)
a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[1, 0], sharex=a0); a2 = fig.add_subplot(gs[1, 1])
# who has which SMs, on the same axis as the curve below
a0.add_patch(Rectangle((0, 0), 138, 1, color=ORG, lw=0)); a0.add_patch(Rectangle((138, 0), 32, 1, facecolor=ORG_L, edgecolor=ORG, hatch="///", ls="--", lw=1.3))
a0.text(69, 0.5, "decode's SMs: 138", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
a0.text(154, 0.5, "tenant: 32", ha="center", va="center", fontsize=10, color=ORG_D, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
a0.set_ylim(0, 1); a0.axis("off")
a0.set_title("Decode's speed: what its SMs predict, and what is measured next to the tenant", loc="left", fontsize=11, fontweight="bold", pad=8)
# the curve of step 7.0, the prediction at 138 SMs, and the measurement
for B, col in ((1, "#c6dbef"), (4, "#9ecae1"), (8, "#6baed6")):
    xs = sorted(n for n in pts if pts[n].get(B)); a1.plot(xs, [pts[n][B] for n in xs], "-", color=col, lw=2)
a1.axhline(100, color=MUTE, lw=0.8, ls="--"); a1.axvline(138, color=MUTE, lw=0.8, ls=":")
lo, hi = min(speed), max(speed); mid = (lo + hi) / 2
a1.annotate("", xy=(138, mid + 1.6), xytext=(138, 98.6), arrowprops=dict(arrowstyle="-|>", color="#c5221f", lw=2.2), zorder=6)
a1.plot([138], [100], "o", ms=14, mfc="white", mec=INK, mew=2, zorder=7)
a1.plot([138], [mid], "o", ms=14, mfc="#c5221f", mec="white", mew=1.5, zorder=7)
a1.text(132, 105.5, "predicted from its SMs alone: full speed", ha="right", va="center", fontsize=10.5, color=INK)
a1.text(168, mid - 6.5, f"achieved perf when colocated:\n{lo:.0f}–{hi:.0f} % of full speed", ha="right", va="center", fontsize=10.5, color="#c5221f", fontweight="bold", linespacing=1.3,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.0), zorder=6)
a1.text(143, (100 + mid) / 2, f"−{100 - hi:.0f} to −{100 - lo:.0f} %", ha="left", va="center", fontsize=10.5, color="#c5221f", fontweight="bold")
a1.text(66, 62, "decode alone, by the number of SMs\nit may use (the curve of step 7.0)", ha="left", va="center", fontsize=9, color="#4292c6", linespacing=1.3)
a1.set_xlim(0, 170); a1.set_ylim(40, 112); a1.set_yticks([40, 60, 80, 100]); a1.set_xticks([32, 56, 80, 96, 138, 170])
a1.set_xlabel("SMs"); a1.set_ylabel("decode speed, % of full speed"); clean(a1)
# decode's KV memory
x = np.arange(3); w = 0.36
a2.bar(x - w / 2 - 0.01, [lim_b[r] for r in RATES], w, facecolor="white", edgecolor=MUTE, lw=1); a2.bar(x - w / 2 - 0.01, [use_b[r] for r in RATES], w, color=GREY, lw=0)
a2.bar(x + w / 2 + 0.01, [lim_t[r] for r in RATES], w, facecolor="white", edgecolor=BLUE, lw=1); a2.bar(x + w / 2 + 0.01, [use_t[r] for r in RATES], w, color=BLUE_L, lw=0)
for xi, r in zip(x, RATES):
    a2.text(xi - w / 2 - 0.01, lim_b[r] + 0.12, f"{lim_b[r]:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK); a2.text(xi + w / 2 + 0.01, lim_t[r] + 0.12, f"{lim_t[r]:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
a2.set_xticks(x); a2.set_xticklabels([f"{r:g} req/s" for r in RATES]); a2.set_ylabel("GiB"); a2.set_ylim(0, 12.5); clean(a2)
a2.set_title("Decode's KV memory:\nits limit (outline) and what it uses (filled)", loc="left", fontsize=10.5, fontweight="bold")
a2.legend(handles=[Patch(facecolor=GREY, edgecolor=MUTE, label="no tenant"), Patch(facecolor=BLUE_L, edgecolor=BLUE, label="next to the tenant")], loc="upper left", frameon=False, fontsize=9)
fig.text(0.5, 0.015, "Measured on vLLM with the 11-minute workload, 0.5 to 2 req/s, tenant on a fixed 32 SMs and training at " + " / ".join(f"{ft[r] / 1e3:.1f}k" for r in RATES) + " tokens/s. Speed from the TPOT with the queue wait in it.", ha="center", fontsize=8.5, color=MUTE)
out = os.path.join(F, "q3_observation.png"); fig.savefig(out, dpi=140); plt.close(fig); print(out)

# ---------------------------------------------------------------- 2. the host <-> GPU link (round 2, MPS co-run without a limit, the run with the most copies)
t0 = float(open(os.path.join(CA, "results", "adjust_nogate_t0.txt")).read()); GiB = 2 ** 30; LINK = 13.5
d = pd.concat([pd.read_csv(os.path.join(CA, "results", f"adjust_nogate_xfer_{k}.csv")).assign(src=k) for k in ("inf", "ft")], ignore_index=True)   # src set from the file: pandas reads the text "inf" as infinity
d = d[(d.ts_end - t0 > 0) & (d.ts_start - t0 < END)]
parts = [("decode: KV loaded from host RAM", d[(d.src == "inf") & (d.dir == "h2d")], BLUE), ("decode: KV stored back", d[(d.src == "inf") & (d.dir == "d2h")], BLUE_L),
         ("tenant: all its copies", d[d.src == "ft"], ORG)]
rates_ = [(n, x.bytes.sum() / GiB / END, c) for n, x, c in parts]; used = sum(v for _, v, _ in rates_); busy = d.gpu_ms.sum() / 1e3 / END * 100
print("host link  " + "  ".join(f"{n}: {v:.2f}" for n, v, _ in rates_) + f"   total {used:.2f} GiB/s of {LINK}; time with a copy on the link {busy:.0f} %")
fig, ax = plt.subplots(figsize=(10.5, 2.9)); fig.subplots_adjust(top=0.80, bottom=0.30, left=0.155, right=0.985)
ax.barh(1, LINK, 0.55, facecolor="white", edgecolor=MUTE, lw=1); ax.text(LINK / 2, 1, f"{LINK:g} GiB/s", ha="center", va="center", fontsize=10, color=INK)
left = 0.0
for n, v, c in rates_: ax.barh(0, v, 0.55, left=left, color=c, lw=0, label=f"{n}, {v:.2f} GiB/s"); left += v
ax.text(left + 0.15, 0, f"{used:.1f} GiB/s  ({used / LINK * 100:.0f} % of the link)", ha="left", va="center", fontsize=10, color=INK, fontweight="bold")
ax.set_yticks([1, 0]); ax.set_yticklabels(["what the link\ncan carry", "what was used,\naverage"], fontsize=9.5); ax.set_xlim(0, 14.2); ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("GiB per second between host RAM and the GPU"); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_title("Host RAM ↔ GPU link over the 11-minute run", loc="left", fontsize=10.5, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3, frameon=False, fontsize=8.5)
out = os.path.join(F, "q3_host_copy.png"); fig.savefig(out, dpi=140); plt.close(fig); print(out)

# ---------------------------------------------------------------- 3. schematic: separate SMs, one path to GPU memory
fig, ax = plt.subplots(figsize=(9.6, 4.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); fig.subplots_adjust(top=0.9, bottom=0.03, left=0.02, right=0.98)
def box(x, y, w, h, text, fc, ec, hatch=None, ls="-", fs=11, bold=True, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", facecolor=fc, edgecolor=ec, lw=1.8, hatch=hatch, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc,
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0) if hatch else None, linespacing=1.35)
def arrow(x, y0, y1, col, lw): ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="<->", mutation_scale=16, lw=lw, color=col, shrinkA=0, shrinkB=0))
box(6, 74, 36, 16, "decode's SMs", ORG, ORG_D, tc="white"); box(58, 74, 36, 16, "tenant's SMs", ORG_L, ORG, hatch="///", ls="--", tc=ORG_D)
box(6, 37, 88, 18, "path between the SMs and GPU memory\n(L2 cache and memory interface: one for the whole GPU)", "#f1f3f4", MUTE, fs=10.5)
box(6, 5, 88, 14, "GPU memory:  decode's weights and KV   |   tenant's data", "#d2e3fc", BLUE, fs=10.5)
arrow(24, 73.5, 55.5, ORG_D, 3.0); arrow(76, 73.5, 55.5, ORG, 3.0); arrow(50, 36.5, 19.5, MUTE, 3.0)
ax.text(50, 96, "SMs can be given to one task or the other", ha="center", va="center", fontsize=9.5, color=MUTE)
ax.set_title("Separate SMs, one path to GPU memory", loc="left", fontsize=12, fontweight="bold")
fig.text(0.985, 0.012, "schematic", ha="right", va="bottom", fontsize=8, color=MUTE, style="italic")
out = os.path.join(F, "q3_shared_path.png"); fig.savefig(out, dpi=140); plt.close(fig); print(out)

# ---------------------------------------------------------------- 4. decode's memory traffic and decode's speed against the SMs it may use
k = pd.read_csv(os.path.join(CA, "results", "knee_probe.csv")); k = k[k.kind == "decode"]
g = k.groupby(["B", "sms"]).agg(step_ms=("step_ms", "median"), gb=("gb_per_step", "first")).reset_index(); g["gbps"] = g.gb / g.step_ms * 1e3
pts = sm_curve()
COL = {1: "#9ecae1", 4: "#4292c6", 8: "#08306b"}
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.7)); fig.subplots_adjust(top=0.88, bottom=0.24, left=0.065, right=0.985, wspace=0.2)
for B in (1, 4, 8):
    s = g[g.B == B].sort_values("sms"); ax[0].plot(s.sms, s.gbps, "-o", color=COL[B], lw=2, ms=5, markeredgecolor="white", markeredgewidth=0.8)
    xs = sorted(n for n in pts if pts[n].get(B)); ax[1].plot(xs, [pts[n][B] for n in xs], "-o", color=COL[B], lw=2, ms=5, markeredgecolor="white", markeredgewidth=0.8, label=f"{B} request{'s' if B > 1 else ''} in the batch")
    print(f"traffic, {B} in batch: " + "  ".join(f"{int(r.sms)}: {r.gbps:.0f}" for r in s.itertuples() if r.sms in (32, 48, 64, 80, 96, 128, 170)))
ax[0].axhline(1780, color=MUTE, lw=0.8, ls="--"); ax[0].text(4, 1800, "the card's nominal 1,780 GB/s", ha="left", va="bottom", fontsize=8.5, color=MUTE)
ax[0].set_ylim(0, 2000); ax[0].set_ylabel("GB per second read from GPU memory"); title(ax[0], "Decode's GPU-memory traffic (simulated decode step)")
ax[1].axhline(100, color=MUTE, lw=0.8, ls="--"); ax[1].set_ylim(0, 120); ax[1].set_yticks([0, 20, 40, 60, 80, 100]); ax[1].set_ylabel("decode speed, % of the speed with all 170 SMs"); title(ax[1], "Decode's speed (vLLM)")
for a in ax:
    a.axvline(80, color=MUTE, lw=1.0, ls="--"); a.set_xlim(0, 176); a.set_xticks([8, 32, 56, 80, 96, 128, 170]); a.set_xlabel("SMs decode may use (the card has 170)"); clean(a)
h, l = ax[1].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.045))
fig.text(0.5, 0.012, "Left: bytes one decode step has to read ÷ the measured step time, decode alone. Right: the vLLM measurement of step 7.0. Vertical lines at 80 SMs.", ha="center", fontsize=8.5, color=MUTE)
out = os.path.join(F, "q3_bandwidth.png"); fig.savefig(out, dpi=140); plt.close(fig); print(out)

# ---------------------------------------------------------------- 5. a synthetic tenant that only moves bytes: decode's slowdown against its traffic
# (simulated decode step, coloc_adjust/results/aggr_corun.csv; the slowdown is against the same block with the second task idle)
# 64 and 96 SMs: the second task on its own SMs, decode on the others (aggr_corun.csv, Nov 2026 run of report 2);
# all 170 SMs: neither task confined, both on the full device through MPS (aggr_corun_shared.csv, Nov 22)
def aggr(path, series, m):
    """[(second task's GB/s, decode's slowdown %)] per batch size. The slowdown is against decode alone on the same SMs, measured before the
    second task's process exists and again after it has exited; the point at 0 GB/s is the second task present but idle."""
    a_ = pd.read_csv(os.path.join(CA, "results", path), dtype={"target_gbps": str}); co = a_[(a_.series == series) & (a_.aggr_sms == m)]; out = {}
    for B in (1, 8):
        s_ = co[co.B == B]; dsm = s_.decode_sms.iloc[0]
        base = a_[a_.series.isin(["alone", "alone_end"]) & (a_.decode_sms == dsm) & (a_.B == B)].groupby("series").step_ms.median().mean()
        pts_ = [(0.0, (s_[s_.target_gbps == "0"].step_ms.median() / base - 1) * 100)]
        pts_ += [(float(r.aggr_gbps.iloc[0]), (r.step_ms.median() / base - 1) * 100) for t, r in s_[~s_.target_gbps.isin(["0", "max"])].groupby("target_gbps")]
        out[B] = sorted(pts_)
    return out
LINES = [("second task on 64 SMs, decode on the other 106", aggr("aggr_corun.csv", "corun", 64), "#6baed6"),
         ("second task on 96 SMs, decode on the other 74", aggr("aggr_corun.csv", "corun", 96), "#08306b"),
         ("both tasks on all 170 SMs", aggr("aggr_corun_shared.csv", "corun_shared", 0), ORG)]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True); fig.subplots_adjust(top=0.88, bottom=0.27, left=0.075, right=0.985, wspace=0.08)
for a, B in zip(ax, (1, 8)):
    for name, d_, col in LINES:
        xy = d_[B]; a.plot([p[0] for p in xy], [p[1] for p in xy], "-o", color=col, lw=2, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=name)
        print(f"{name}, {B} in batch: " + "  ".join(f"{x:.0f} GB/s: {y:+.1f}%" for x, y in xy))
    title(a, f"Decode with {B} request{'s' if B > 1 else ''} in the batch"); a.set_xlabel("GPU-memory traffic of the second task (GB/s)"); xmax = min(d_[B][-1][0] for _, d_, _ in LINES)   # the plot ends where all three lines still have a value
    a.set_xlim(-20, xmax + 14); a.set_ylim(-2, 85); a.axhline(0, color=MUTE, lw=0.6); clean(a); print(f"  {B} in batch: x axis ends at {xmax:.0f} GB/s")
ax[0].set_ylabel("decode slower than with no second task (%)")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.055))
fig.text(0.5, 0.012, "Simulated decode step; the second task only reads GPU memory at a set rate. At 0 GB/s the second task is there but idle.", ha="center", fontsize=8.5, color=MUTE)
out = os.path.join(F, "q3_aggressor.png"); fig.savefig(out, dpi=140); plt.close(fig); print(out)
