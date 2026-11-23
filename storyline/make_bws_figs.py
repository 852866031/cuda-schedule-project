#!/usr/bin/env python3
"""Storyline figure for direction one (limiting the tenant's use of GPU internal bandwidth): the tenant's measured traffic over
time under three ways of running it, for two tenants. Runs nothing: reads the 1 ms traces of coloc_adjust/pace_check.py
(coloc_adjust/results_prof/pace_<tenant>_<mode>_trace.csv; tenant alone on 32 SMs, bandwidth counter sampled at 10 kHz).
  python3 storyline/make_bws_figs.py   ->  storyline/figures/bws_pacing.png"""
import os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "coloc_adjust", "results_prof")
ORG, INK, MUTE = "#e37400", "#202124", "#5f6368"; BUDGET = 50
import numpy as np
S = pd.read_csv(os.path.join(D, "pace_summary.csv")).set_index("case")
# right column: the tenant with both changes once its traces exist (cb_vocab4096f, measured Nov 22), else the tenant with change 1 only
RIGHT = ("cb_vocab4096f", "Tenant with changes ① and ②") if os.path.exists(os.path.join(D, "pace_cb_vocab4096f_w12_trace.csv")) else ("vocab4096", "Tenant with change ① only")
COLS = [("base", "Usual tenant"), RIGHT]; YMAX = 200     # one vertical scale for all panels; taller peaks are cut off and their height is written next to them
ROWS = [("free", "no limit"), ("step", f"budget {BUDGET} GB/s,\none pause\nafter each step"), ("w12", f"budget {BUDGET} GB/s,\npauses inside\nthe step")]
fig, ax = plt.subplots(3, 2, figsize=(13.5, 8.0), sharex="col", sharey=True); fig.subplots_adjust(top=0.865, bottom=0.115, left=0.215, right=0.985, hspace=0.40, wspace=0.09)
for j, (ten, name) in enumerate(COLS):
    xmax = min(len(pd.read_csv(os.path.join(D, f"pace_{ten}_{m}_trace.csv"))) for m, _ in ROWS)      # the length of the shortest trace of the column
    for i, (mode, rname) in enumerate(ROWS):
        a = ax[i][j]; d = pd.read_csv(os.path.join(D, f"pace_{ten}_{mode}_trace.csv")); g = d.gbps.values; r12 = d.gbps.rolling(12, center=True, min_periods=1).mean()
        a.fill_between(d.t_ms, 0, d.gbps, step="mid", color=ORG, lw=0); a.plot(d.t_ms, r12, color="#5b2e00", lw=1.1)
        if mode != "free": a.axhline(BUDGET, color=INK, lw=1.2, ls="--", zorder=5)
        a.set_xlim(0, xmax); a.set_ylim(0, YMAX); a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
        hi = np.where((g > YMAX) & (d.t_ms.values < xmax))[0]                                         # peaks taller than the axis: a mark at the top of each, the height written at the first
        note = ""
        if len(hi):
            grp = np.split(hi, np.where(np.diff(hi) > 20)[0] + 1); tc = [d.t_ms.values[k].mean() for k in grp]
            a.plot(tc, [YMAX * 1.035] * len(tc), marker="^", ls="", color=INK, ms=5.5, clip_on=False, zorder=6); note = f"   ·   ▲ peaks cut off: up to {g[hi].max():,.0f} GB/s in 1 ms"
        a.text(1.0, 1.10, f"mean {int(S.mean_gbps[f'{ten}_{mode}'] + 0.5)} GB/s   ·   highest 12 ms window: {S.w12_max[f'{ten}_{mode}']:.0f} GB/s{note}", transform=a.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
        if j == 0: a.text(-0.42, 0.5, rname, transform=a.transAxes, ha="left", va="center", fontsize=10, fontweight="bold", color=INK, linespacing=1.3); a.set_ylabel("GB/s", fontsize=9)
        if i == 0: a.text(0.0, 1.26, name, transform=a.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold", color="black")
        if i == 2: a.set_xlabel("time (ms)")
        print(f"{ten:14s} {mode:5s} mean {S.mean_gbps[f'{ten}_{mode}']:6.1f}  max 1 ms {g.max():5.0f}  worst 12 ms {S.w12_max[f'{ten}_{mode}']:.0f}  tok/s {S.tok_s[f'{ten}_{mode}']:.0f}  samples {len(d)}")
fig.legend(handles=[Patch(color=ORG, label="GPU internal bandwidth used by the tenant, 1 ms bins (measured, tenant alone)"), Line2D([], [], color="#5b2e00", lw=1.1, label="the same, 12 ms average"),
                    Line2D([], [], color=INK, lw=1.2, ls="--", label=f"the budget, {BUDGET} GB/s"), Line2D([], [], marker="^", color=INK, ls="", ms=5.5, label="a peak taller than the axis")],
           loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.012), columnspacing=1.6)
out = os.path.join(H, "figures", "bws_pacing.png"); fig.savefig(out, dpi=140); print(out)

# ---------------------------------------------------------------- how a budget is applied: segments -> chunks of equal data -> pauses
# Built from the measured segment table of a tenant (coloc_adjust/results_prof/seg_<variant>.csv; one step computed as 4 batches of
# 2 sequences). One of the 4 batches is drawn; the other three look the same. The chunk rule is the one of coloc_adjust/pacing.py.
import math
import numpy as np
from matplotlib.patches import Rectangle
WINDOW_MS = 12; CHUNK = BUDGET * WINDOW_MS / 1e3      # GB
KCOL = {"fwd": "#9ecae1", "head": "#fdae6b", "bwd": "#3182bd", "other": "#bdc1c6"}
def chunk_figure(variant, out_name, names):
    """names: {kind: text written on that part of the segment bar}"""
    t = pd.read_csv(os.path.join(D, f"seg_{variant}.csv")); end = t.index[t.label == "mb_end"][0]; t = t.iloc[1:end + 1].reset_index(drop=True)   # the first batch
    def kind_of(r):
        if r.label.startswith("head") or r.total_gb > 1.0: return "head"   # the usual tenant: output layer and loss sit in one segment of 7.96 GB
        return "fwd" if r.label.endswith(":fwd") or r.label == "step" else "bwd" if r.label.endswith(":bwd") else "other"
    kind = t.apply(kind_of, axis=1); cum = np.concatenate([[0.0], t.total_gb.cumsum().values]); tot = cum[-1]
    cuts, nxt = [], CHUNK                                # pause at the first stopping point past each multiple of the chunk size
    for k in range(1, len(t) + 1):
        if cum[k] >= nxt: cuts.append(k); nxt = (math.floor(cum[k] / CHUNK) + 1) * CHUNK
    if not cuts or cuts[-1] != len(t): cuts.append(len(t))
    chunks = list(zip([0] + cuts[:-1], cuts))            # [first segment, one past the last segment)
    print(f"{variant}: one batch = {len(t)} segments, {tot:.2f} GB, largest {t.total_gb.max():.3f} GB; chunk {CHUNK:.2f} GB -> {len(chunks)} chunks: " + " ".join(f"{cum[b] - cum[a]:.2f}" for a, b in chunks))
    fig = plt.figure(figsize=(13.5, 6.9)); gs = fig.add_gridspec(4, 1, height_ratios=[0.34, 1, 1, 1], hspace=0.92, top=0.94, bottom=0.14, left=0.045, right=0.985)
    a0, a1, a2, a3 = (fig.add_subplot(g) for g in gs)
    def strip(a, name):
        a.set_ylim(0, 1); a.set_yticks([]); a.spines[["top", "right", "left"]].set_visible(False); a.set_title(name, loc="left", fontsize=10.5, fontweight="bold", pad=6)
    for b in range(4):                                   # 0. the whole step
        a0.add_patch(Rectangle((b + 0.02, 0), 0.96, 1, facecolor="#fde9c8" if b else ORG, edgecolor=ORG, lw=1.2))
        a0.text(b + 0.5, 0.5, f"batch {b + 1}" + (" (drawn below)" if b == 0 else ""), ha="center", va="center", fontsize=9.5, color="white" if b == 0 else "#8a4600", fontweight="bold" if b == 0 else "normal")
    a0.set_xlim(0, 4); a0.set_xticks([]); a0.spines["bottom"].set_visible(False); strip(a0, "One fine-tuning step = 4 batches, one after another")
    for k in range(len(t)): a1.add_patch(Rectangle((cum[k], 0), cum[k + 1] - cum[k], 1, facecolor=KCOL[kind[k]], edgecolor="white", lw=0.5))   # 1. segments
    for kd, tc in (("fwd", INK), ("head", INK), ("bwd", "white")):
        ix = np.where(kind.values == kd)[0]; gbs = t.total_gb.iloc[ix].sum()
        a1.text((cum[ix[0]] + cum[ix[-1] + 1]) / 2, 0.5, f"{names[kd]}\n{len(ix)} segment{'s' if len(ix) > 1 else ''}, {gbs:.2f} GB", ha="center", va="center", fontsize=9.5, color=tc, fontweight="bold",
                linespacing=1.25, bbox=dict(facecolor=KCOL[kd], edgecolor="none", pad=2.5))
    strip(a1, f"1. The profile cuts the batch into {len(t)} segments and measures the data each one moves   (width = GB moved)")
    a1.set_xlim(0, tot); a1.set_xlabel("data moved since the start of the batch (GB)", fontsize=9, labelpad=2)
    for n, (i, j) in enumerate(chunks):                  # 2. chunks
        a2.add_patch(Rectangle((cum[i], 0), cum[j] - cum[i], 1, facecolor="#e8eaed" if n % 2 else "#cfd3d7", edgecolor=INK, lw=1.0))
        wide = (cum[j] - cum[i]) / tot > 0.055       # a narrow chunk gets its GB only
        if wide or (cum[j] - cum[i]) / tot > 0.03:
            a2.text((cum[i] + cum[j]) / 2, 0.5, f"{cum[j] - cum[i]:.2f} GB\n{j - i} segment{'s' if j - i > 1 else ''}" if wide else f"{cum[j] - cum[i]:.2f}", ha="center", va="center",
                    fontsize=8.5 if wide else 8, color=INK, linespacing=1.25)
        if j < len(t): a2.plot([cum[j], cum[j]], [0, 1], color="#c5221f", lw=2.2)
    strip(a2, f"2. Segments are grouped into chunks of about {CHUNK:.1f} GB each   (budget {BUDGET} GB/s × window {WINDOW_MS} ms);  red = where the tenant may pause")
    a2.set_xlim(0, tot); a2.set_xlabel("data moved since the start of the batch (GB)", fontsize=9, labelpad=2)
    now = 0.0                                            # 3. in time
    for n, (i, j) in enumerate(chunks):
        work = t.ms.iloc[i:j].sum(); due = cum[j] / BUDGET * 1e3
        a3.add_patch(Rectangle((now, 0), work, 1, facecolor="#e8eaed" if n % 2 else "#cfd3d7", edgecolor=INK, lw=1.0))
        if due > now + work: a3.add_patch(Rectangle((now + work, 0), due - now - work, 1, facecolor="white", edgecolor="#c5221f", hatch="////", lw=1.0))
        now = max(due, now + work)
    strip(a3, "3. In time: after each chunk the tenant sleeps until  data moved ÷ budget  has passed;  a chunk that already took that long gets no pause")
    a3.set_xlim(0, now); a3.set_xlabel("time since the start of the batch (ms)", fontsize=9, labelpad=2)
    fig.legend(handles=[Patch(color=KCOL["fwd"], label="forward through the layers"), Patch(color=KCOL["head"], label="output layer and loss"), Patch(color=KCOL["bwd"], label="backward through the layers"),
                        Patch(facecolor="#cfd3d7", edgecolor=INK, label="a chunk"), Patch(facecolor="white", edgecolor="#c5221f", hatch="////", label="pause")],
               loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.005))
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
# direction one is told with the usual tenant; the same drawing for the tenant with all three changes belongs to direction two
chunk_figure("base", "bws_chunks.png", {"fwd": "layers, forward", "head": "last layer, then the output layer and the loss: one piece that cannot be cut", "bwd": "layers, backward"})
chunk_figure("cb_vocab4096f", "bwr_chunks_new.png", {"fwd": "24 layers, forward", "head": "output layer and loss,\nin vocabulary blocks", "bwd": "24 layers, backward"})

# ---------------------------------------------------------------- the first, simple way to apply a budget (Nov 16): checks at the layer borders, equal data assumed per check
# A schematic of coloc_adjust/pacing.py, BlockPacer, in three separate parts: what is known (one measured average), what is assumed
# (the same data for every layer), how the pauses follow. It shows only what this method knows. Part 3 is drawn like the method
# schematics of make_q2_figs.py (lanes over time, height = share of the SMs, the tenant's SMs dashed).
# Numbers of the usual tenant: 13.95 MB per token, 4,096 tokens and 197 checks per step; a layer takes about 1.3 ms (seg_base.csv).
def naive_figure(out_name="bws_naive.png", n_slot=6):
    t = pd.read_csv(os.path.join(D, "seg_base.csv")); n_checks = len(t) - 1; per = 13.95 * 4096 / 1e3 / n_checks; slot = per / BUDGET * 1e3; work_ms = float(t.ms[t.label.str.endswith(":fwd")].iloc[:8].median())
    T = 100.0; fig = plt.figure(figsize=(10.5, 7.9)); gs = fig.add_gridspec(4, 1, height_ratios=[0.50, 0.62, 0.8, 1.0], hspace=0.0, top=0.93, bottom=0.07, left=0.17, right=0.74)
    a0, a1, a2, a3 = (fig.add_subplot(g) for g in gs)
    for a, (y0, y1) in zip((a0, a1, a2, a3), ((0.865, 0.925), (0.655, 0.745), (0.335, 0.495), (0.075, 0.265))):      # explicit places: the three parts are kept apart
        a.set_position([0.17, y0, 0.57, y1 - y0])
    def head(y, text): fig.text(0.02, y, text, ha="left", va="center", fontsize=12, fontweight="bold", color="black")
    def side(a, text): a.text(1.035, 0.5, text, transform=a.transAxes, ha="left", va="center", fontsize=9.5, color=INK, linespacing=1.4)
    def lab(a, text): a.text(-0.26, 0.5, text, transform=a.transAxes, ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    # 1. known
    head(0.965, "① Known: the data the tenant moves per token")
    a0.set_xlim(0, T); a0.set_ylim(0, 1); a0.axis("off")
    a0.add_patch(Rectangle((0, 0), 26, 1, facecolor="#f1f3f4", edgecolor=INK, lw=1.0)); a0.text(13, 0.5, "1 token trained", ha="center", va="center", fontsize=10.5, color=INK)
    a0.annotate("", xy=(40, 0.5), xytext=(27, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    a0.text(42, 0.5, "13.95 MB", ha="left", va="center", fontsize=14, fontweight="bold", color=INK); a0.text(62, 0.5, "moved between the\nSMs and GPU memory", ha="left", va="center", fontsize=9.5, color=INK, linespacing=1.3)
    side(a0, "measured once, beforehand;\nan average")
    # 2. assumed
    head(0.795, "② Assumed: every layer moves the same amount of data")
    a1.set_xlim(0, T); a1.set_ylim(0, 1); a1.axis("off"); W, G = 22.0, 6.0; x = 0.0
    for it in (("layer 1", "fwd"), None, ("layer 24", "fwd"), ("layer 24", "bwd"), None, ("layer 1", "bwd")):
        if it is None: a1.text(x + G / 2, 0.5, "···", ha="center", va="center", fontsize=16, color=INK); x += G; continue
        name, kd = it; a1.add_patch(Rectangle((x, 0), W, 1, facecolor=KCOL[kd], edgecolor="white", lw=1.5)); tc = INK if kd == "fwd" else "white"
        a1.text(x + W / 2, 0.70, f"{name}, {'forward' if kd == 'fwd' else 'backward'}", ha="center", va="center", fontsize=8.5, color=tc)
        a1.text(x + W / 2, 0.30, f"{per:.2f} GB", ha="center", va="center", fontsize=11.5, color=tc, fontweight="bold"); x += W
    side(a1, f"13.95 MB × 4,096 tokens in a step\n÷ {n_checks} layer passes in a step\n= {per:.2f} GB each")
    # 3. the schedule, as lanes over time
    head(0.585, f"③ Schedule: a budget of {BUDGET} GB/s gives every layer a time slot"); fig.text(0.02, 0.550, "height = share of the SMs the task may use;  solid orange = decode's SMs,  dashed orange = the tenant's SMs", ha="left", va="center", fontsize=9, color=MUTE)
    for a in (a2, a3): a.set_xlim(0, T); a.set_xticks([]); a.spines[["top", "right"]].set_visible(False)
    a2.set_ylim(0, 1); a2.set_yticks([0, 1]); a2.set_yticklabels(["0", "100 %"], fontsize=8.5); lab(a2, "decode")
    sw = T / n_slot; dstep = 12.0 / slot * sw; x = 0.0
    while x < T: a2.add_patch(Rectangle((x, 0), min(dstep - 0.6, T - x), 1.0, color=ORG, lw=0)); x += dstep
    side(a2, "decode steps;\nits SMs are not limited")
    h = 32 / 170; a3.set_ylim(0, 1.0); a3.set_yticks([0, h, 1]); a3.set_yticklabels(["0", "32 SMs", "100 %"], fontsize=8.5); lab(a3, "tenant"); bw = work_ms / slot * sw
    for k in range(n_slot):
        a3.add_patch(Rectangle((k * sw, 0), bw, h, facecolor="#fde9c8", edgecolor=ORG, hatch="///", ls="--", lw=1.3)); a3.text(k * sw + 0.4, h + 0.04, f"layer {k + 1}", ha="left", va="bottom", fontsize=9, color=INK)
        a3.text(k * sw + (bw + sw) / 2, h / 2, "sleep", ha="center", va="center", fontsize=9, color=MUTE, style="italic")
        a3.plot([k * sw, k * sw], [0, 0.56], color=MUTE, lw=0.8, ls=":")
    a3.plot([T, T], [0, 0.56], color=MUTE, lw=0.8, ls=":", clip_on=False)
    a3.annotate("", xy=(sw, 0.52), xytext=(0, 0.52), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, shrinkA=0, shrinkB=0))
    a3.text(0.4, 0.60, f"time slot of a layer  =  {per:.2f} GB ÷ {BUDGET} GB/s  =  {slot:.1f} ms", ha="left", va="bottom", fontsize=10, color=INK, fontweight="bold")
    a3.set_xlabel("time  →", fontsize=9)
    side(a3, "32 SMs while it works;\nafter each layer it sleeps\nuntil the layer's time slot\nis over")
    fig.text(0.995, 0.012, "schematic", ha="right", va="bottom", fontsize=8, color=MUTE, style="italic")
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out, f"checks {n_checks}, {per:.3f} GB per layer, slot {slot:.2f} ms, layer {work_ms:.2f} ms")
naive_figure()

# ---------------------------------------------------------------- the simple budget, measured
# (A) on vLLM (11-minute workload, one run each): the usual tenant on a 32-SM green context without a budget (results_600w/green32, Nov 5, the run
#     of section 8.1) and with a budget of 50 GB/s kept by a pause at each layer check (results_o3/c1_fine, Nov 16). Each against decode alone of
#     its own round. TPOT with the queue wait in it, as in make_q2_figs.py. Reference: the bandwidth workload of section 8.5 at a steady 50 GB/s
#     next to the simulated decode step (results/aggr_corun_d98_122.csv; decode on 98 and 122 SMs, 1 and 8 requests).
# (B) the tenant alone, Nov 21 (results_prof/pace_base_*_trace.csv): its traffic over time without a budget and with the budget kept at each layer check.
import json
CA = os.path.join(H, "..", "coloc_adjust")
SCHED = [(0, 1), (90, 2), (180, 3), (270, 1), (360, 4), (420, 0.5), (480, 2), (510, 3), (540, 2), (570, 1), (600, 1)]; END = 660; SKIP = 10
PH = [(ta, tb, v) for (ta, v), tb in zip(SCHED, [x for x, _ in SCHED[1:]] + [END])]
def tpot(rd, arm):
    t0 = float(open(os.path.join(CA, rd, f"{arm}_t0.txt")).read()); rows = []
    for l in open(os.path.join(CA, rd, f"{arm}_load.jsonl")):
        r = json.loads(l)
        if r["rid"].startswith("w") or not r["tok_t"] or r["status"] != 200 or r["error"]: continue
        rows.append((r["t_send"] - t0, (r["t_done"] - r["t_send"]) / max(1, r["n_out"]) * 1e3))
    return np.array(rows)
def by_rate(rd, arms, q, rates):   # mean over the phases at one arrival rate (and over the runs) of the per-phase statistic, first 10 s of a phase left out
    stat = (lambda v: float(np.mean(v))) if q == "mean" else (lambda v: float(np.percentile(v, q))); out = {}
    for rate in rates:
        vv = []
        for arm in arms:
            R = tpot(rd, arm); vv += [stat(R[(R[:, 0] >= a + SKIP) & (R[:, 0] < b), 1]) for a, b, r in PH if r == rate and ((R[:, 0] >= a + SKIP) & (R[:, 0] < b)).sum() > 3]
        out[rate] = float(np.mean(vv))
    return out
def steady_ref(gbps=BUDGET):
    """slowdown of the simulated decode step next to the bandwidth workload at `gbps` GB/s: lowest and highest of decode on 98 / 122 SMs, 1 / 8 requests"""
    a_ = pd.read_csv(os.path.join(CA, "results", "aggr_corun_d98_122.csv"), dtype={"target_gbps": str}); v = []
    for m in (72, 48):
        for B in (1, 8):
            s_ = a_[(a_.series == "corun") & (a_.aggr_sms == m) & (a_.B == B)]; dsm = s_.decode_sms.iloc[0]
            base = a_[a_.series.isin(["alone", "alone_end"]) & (a_.decode_sms == dsm) & (a_.B == B)].groupby("series").step_ms.median().mean()
            pts = sorted([(0.0, s_[s_.target_gbps == "0"].step_ms.median())] + [(float(r.aggr_gbps.iloc[0]), r.step_ms.median()) for t, r in s_[~s_.target_gbps.isin(["0", "max"])].groupby("target_gbps")])
            v.append((np.interp(gbps, [x for x, _ in pts], [y for _, y in pts]) / base - 1) * 100)
    return min(v), max(v)
def naive_result(out_name="bws_naive_result.png"):
    rates = [0.5, 1]; lo, hi = steady_ref(); print(f"bandwidth workload at {BUDGET} GB/s: {lo:+.2f} to {hi:+.2f} %")
    ARMS = [("results_600w", ["adjust_notenant"], "green32", "tenant on 32 SMs, no budget", "#9aa0a6"),
            ("results_o3", ["adjust_notenant", "adjust_notenant_end"], "c1_fine", f"tenant on 32 SMs, budget of {BUDGET} GB/s", "#E69F00")]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1.75, 1])); fig.subplots_adjust(top=0.76, bottom=0.27, left=0.07, right=0.985, wspace=0.20)
    fig.suptitle("The simple budget next to decode on vLLM", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, "usual tenant on a 32-SM green context; decode TPOT includes the queue wait; each bar against decode alone (no tenant) of its own round:\n"
             "no budget on Nov 5 (600 W), budget on Nov 16 (500 W); one run each", ha="left", va="center", fontsize=9, color=MUTE, linespacing=1.4)
    w = 0.19; gap = 0.015
    for k, (rd, base, arm, name, col) in enumerate(ARMS):
        bm, b90 = by_rate(rd, base, "mean", rates), by_rate(rd, base, 90, rates); m, p = by_rate(rd, [arm], "mean", rates), by_rate(rd, [arm], 90, rates); ph = pd.read_csv(os.path.join(CA, rd, f"{arm}_phases.csv"))
        avg = [(m[r] / bm[r] - 1) * 100 for r in rates]; p90 = [(p[r] / b90[r] - 1) * 100 for r in rates]; ten = [float(ph[ph.rps == r].ft_tok_s.mean()) / 1e3 for r in rates]
        print(f"{arm:8s} avg " + " ".join(f"{v:+.1f}%" for v in avg) + "  p90 " + " ".join(f"{v:+.1f}%" for v in p90) + "  tenant " + " ".join(f"{v:.2f}k" for v in ten) + "   decode alone " + " ".join(f"{bm[r]:.2f}" for r in rates) + " ms")
        for i in range(len(rates)):                      # in a group: [average, p90] of the first arm, then of the second
            xa = i + (2 * k - 1.5) * (w + gap) - (0.02 if k == 0 else -0.02); xp = xa + w + gap
            ax[0].bar(xa, avg[i], w, color=col, lw=0, zorder=2); ax[0].bar(xp, p90[i], w, color=col, lw=0, alpha=0.45, hatch="///", edgecolor=col, zorder=2)
            for x, v in ((xa, avg[i]), (xp, p90[i])): ax[0].text(x, v + 0.2, f"{v:+.1f} %", ha="center", va="bottom", fontsize=9, color=INK)
            xt = i + (k - 0.5) * 0.39; ax[1].bar(xt, ten[i], 0.36, color=col, lw=0, zorder=2); ax[1].text(xt, ten[i] + 0.12, f"{ten[i]:.2f}k", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax[0].axhspan(lo, hi, color=INK, alpha=0.18, lw=0, zorder=3); ax[0].axhline((lo + hi) / 2, color=INK, lw=1.2, ls="--", zorder=4)
    for a, name, ymax, yl in zip(ax, ("Decode TPOT: average and p90", "Tenant progress"), (14, 9), ("slowdown against decode alone (%)", "thousand tokens/s")):
        a.set_title(name, loc="left", fontsize=11, fontweight="bold"); a.set_xticks(range(len(rates))); a.set_xticklabels([f"{r:g} req/s" for r in rates], fontsize=10); a.set_ylim(0, ymax); a.set_ylabel(yl, fontsize=9.5)
        a.set_xlim(-0.5, len(rates) - 0.5); a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=n) for _, _, _, n, c in ARMS] + [Patch(facecolor="#5f6368", label="average"), Patch(facecolor="#5f6368", alpha=0.45, hatch="///", edgecolor="#5f6368", label="p90")],
               loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.075), columnspacing=2.2)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.2, ls="--", label=f"for reference: the bandwidth workload of section 8.5 at a steady {BUDGET} GB/s slows the simulated decode step by {lo:+.1f} to {hi:+.1f} %")],
               loc="lower center", ncol=1, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
def naive_trace(out_name="bws_naive_trace.png", xmax=1050, win=12):
    ROWS = [("base_free", "No budget"), ("base_block1", f"Budget of {BUDGET} GB/s, a pause at each layer check")]
    fig, ax = plt.subplots(2, 1, figsize=(12.5, 6.4), sharex=True, sharey=True); fig.subplots_adjust(top=0.84, bottom=0.14, left=0.065, right=0.985, hspace=0.30)
    fig.suptitle("The tenant's traffic over time, without and with the simple budget", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.925, f"usual tenant alone on 32 SMs; GPU internal bandwidth it uses, from the hardware counter sampled 10,000 times a second; a {win} ms window is about one decode step", ha="left", va="center", fontsize=9, color=MUTE)
    for a, (case, name) in zip(ax, ROWS):
        d = pd.read_csv(os.path.join(D, f"pace_{case}_trace.csv")); g = d.gbps; r = g.rolling(win, center=True, min_periods=win).mean(); rv = r.dropna()
        a.fill_between(d.t_ms, 0, g, step="mid", color=ORG, lw=0); a.plot(d.t_ms, r, color="#5b2e00", lw=1.2)
        if case != "base_free": a.axhline(BUDGET, color=INK, lw=1.5, ls="--", zorder=5)
        txt = f"{name}:   mean {g.mean():.0f} GB/s,   highest {win} ms window {rv.max():.0f} GB/s,   {100 * (rv > 200).mean():.0f} % of the time above 200 GB/s"
        a.set_title(txt, loc="left", fontsize=10.5, fontweight="bold"); a.set_xlim(0, xmax); a.set_ylim(0, 1650); a.set_ylabel("GB/s", fontsize=9.5)
        a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
        print(f"{case:12s} mean {g.mean():.1f}  max 1 ms {g.max():.0f}  {win} ms windows: max {rv.max():.0f}, over budget {100 * (rv > BUDGET).mean():.1f} %, over 200: {100 * (rv > 200).mean():.1f} %, data moved in windows over 2x budget {100 * rv[rv > 2 * BUDGET].sum() / rv.sum():.0f} %")
    ax[1].set_xlabel("time (ms)", fontsize=9.5)
    fig.legend(handles=[Patch(color=ORG, label="traffic in 1 ms bins"), Line2D([], [], color="#5b2e00", lw=1.2, label=f"the same, {win} ms average"), Line2D([], [], color=INK, lw=1.1, ls="--", label=f"the budget, {BUDGET} GB/s")],
               loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01), columnspacing=2.5)
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
naive_result(); naive_trace()

# ---------------------------------------------------------------- direction two: the two changes to the tenant (vocabulary blocks; torch.compile on the layers and on the output layer), data moved per step
# From coloc_adjust/results_prof/seg_summary.md (tenant alone on 32 SMs, Nov 21/22). The usual tenant's output layer cannot be told apart
# in its own row (no check inside), so its split comes from the variant with extra marks between the operators ("split").
def seg_table():
    rows = {}
    for l in open(os.path.join(D, "seg_summary.md")):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 13 and c[1].replace(".", "").isdigit(): rows[c[0]] = dict(tok=float(c[2]), gb=float(c[3]), head=float(c[7]) if c[7] else 0.0, big=float(c[9]))
    return rows
def changes_bytes(out_name="bwr_bytes.png"):
    S_ = seg_table(); BL, OR = "#3182bd", "#fdae6b"
    ROWS = [("base", "usual tenant"), ("vocab4096", "+ ① output layer and loss in vocabulary blocks"), ("cb_vocab4096", "+ ② torch.compile on the 24 transformer layers"), ("cb_vocab4096f", "+ ② torch.compile on the output layer as well")]
    fig, a = plt.subplots(figsize=(12.5, 3.9)); fig.subplots_adjust(top=0.74, bottom=0.17, left=0.335, right=0.70)
    fig.suptitle("Data moved in one fine-tuning step, after each change", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.885, "tenant alone on 32 SMs, one step of 4,096 tokens in 4 batches; the changes add up from top to bottom", ha="left", va="center", fontsize=9, color=MUTE)
    base = S_["base"]["gb"]
    for i, (v, name) in enumerate(ROWS):
        r = S_[v]; head = S_["split"]["head"] if v == "base" else r["head"]; rest = r["gb"] - head; y = len(ROWS) - 1 - i
        a.barh(y, rest, 0.62, color=BL, lw=0); a.barh(y, head, 0.62, left=rest + 0.25, color=OR, lw=0)
        a.text(rest / 2, y, f"{rest:.1f}", ha="center", va="center", fontsize=9.5, color="white"); a.text(rest + 0.25 + head / 2, y, f"{head:.1f}", ha="center", va="center", fontsize=9.5, color=INK)
        a.text(r["gb"] + 1.2, y, f"{r['gb']:.1f} GB" + ("" if v == "base" else f"   ({(r['gb'] / base - 1) * 100:+.0f} %)"), ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
        a.text(-1.0, y, name, ha="right", va="center", fontsize=10, color=INK)
        fig.text(0.80, a.get_position().y0 + (y + 0.5) / len(ROWS) * a.get_position().height, f"{r['big']:.2f} GB", ha="center", va="center", fontsize=10.5, color=INK)
        fig.text(0.93, a.get_position().y0 + (y + 0.5) / len(ROWS) * a.get_position().height, f"{r['tok'] / 1e3:.1f}k", ha="center", va="center", fontsize=10.5, color=INK)
        print(f"{v:16s} total {r['gb']:.2f}  layers+rest {rest:.2f}  head {head:.2f}  largest {r['big']:.3f}  tok/s {r['tok']:.0f}")
    top = a.get_position().y1 + 0.035
    fig.text(0.80, top, "largest piece\nwithout a pause", ha="center", va="bottom", fontsize=9.5, color=MUTE, linespacing=1.3); fig.text(0.93, top, "tokens/s,\ntenant alone", ha="center", va="bottom", fontsize=9.5, color=MUTE, linespacing=1.3)
    a.set_xlim(0, 60); a.set_ylim(-0.5, len(ROWS) - 0.5); a.set_yticks([]); a.set_xlabel("GB per step", fontsize=9.5); a.spines[["top", "right", "left"]].set_visible(False); a.grid(True, axis="x", alpha=0.25); a.set_axisbelow(True)
    fig.legend(handles=[Patch(color=BL, label="the 24 layers (and the rest)"), Patch(color=OR, label="output layer and loss")], loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01), columnspacing=2.5)
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.15); plt.close(fig); print(out)
changes_bytes()

# ---------------------------------------------------------------- direction two: what merging small operators does (change 2, in its two places), schematics
# On a transformer layer: one part of a layer, its RMS normalisation, as recorded by coloc_adjust/compile_demo.py on the CPU (reports/3, section 6.2):
#   eager 8 operators, 5 full-size tensors written; compiled 1 generated kernel, 1 tensor written.
# On the output layer: the second pass over one vocabulary block (coloc_adjust/head_loss.py, vocab_lm_loss, plain and fused); the sizes follow from
#   the tensor shapes: 1,024 positions x 4,096 words (bf16 8.4 MB, float32 16.8 MB), 1,024 x 896 hidden values (bf16 1.8 MB).
from matplotlib.patches import FancyBboxPatch
def merge_figure(out_name, title, sub, ops, merged, head_a, head_b):
    """ops: [(name, size text, big?, library call?)]; merged: [(first op, one past the last op, text, size text, library call?)]"""
    MEM, MEDGE, LIB = "#e8f0fe", "#1a73e8", "#e8eaed"; n = len(ops); W = 100.0 / n; g = 1.2
    fig = plt.figure(figsize=(13.0, 6.2)); fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold"); fig.text(0.02, 0.925, sub, ha="left", va="center", fontsize=9, color=MUTE)
    def row(a, items, head):
        a.set_xlim(0, 100); a.set_ylim(0, 10); a.axis("off"); a.text(0, 10.3, head, ha="left", va="bottom", fontsize=11, fontweight="bold", color=INK)
        a.add_patch(FancyBboxPatch((0.3, 0.3), 99.4, 2.6, boxstyle="round,pad=0,rounding_size=0.5", facecolor=MEM, edgecolor=MEDGE, lw=1.2))
        for x0, x1, name, size, big, lib in items:
            a.add_patch(FancyBboxPatch((x0 + g / 2, 6.0), x1 - x0 - g, 3.2, boxstyle="round,pad=0,rounding_size=0.5", facecolor=LIB if lib else "#fff4e5", edgecolor=MUTE if lib else ORG, lw=1.3))
            a.text((x0 + x1) / 2, 7.6, name, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.25)
            xa = x1 - W / 2                                # the result leaves at the right end of the box
            a.annotate("", xy=(xa, 3.0), xytext=(xa, 5.9), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6 if big else 0.9, mutation_scale=16 if big else 9))
            a.text(xa, 1.6, size, ha="center", va="center", fontsize=10.5 if big else 9.5, color=INK if big else MUTE, fontweight="bold" if big else "normal")
    a = fig.add_axes([0.03, 0.50, 0.94, 0.30]); row(a, [(i * W, (i + 1) * W, nm, sz, big, lib) for i, (nm, sz, big, lib) in enumerate(ops)], head_a)
    b = fig.add_axes([0.03, 0.10, 0.94, 0.30]); row(b, [(i0 * W, i1 * W, nm, sz, True, lib) for i0, i1, nm, sz, lib in merged], head_b)
    fig.legend(handles=[Patch(facecolor="#fff4e5", edgecolor=ORG, label="an operator, run on the SMs"), Patch(facecolor=LIB, edgecolor=MUTE, label="a matrix product (a library call, not merged)"),
                        Patch(facecolor=MEM, edgecolor=MEDGE, label="GPU memory: the size of what the operator writes there"), Line2D([], [], color=INK, lw=2.6, label="writes a full-size tensor"),
                        Line2D([], [], color=INK, lw=0.9, label="writes a few KB")], loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5, handletextpad=0.5)
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
NORM = [("to float32", "3.67 MB", True, False), ("square", "3.67 MB", True, False), ("mean", "4 KB", False, False), ("add ε", "4 KB", False, False), ("1 / square root", "4 KB", False, False),
        ("multiply", "3.67 MB", True, False), ("to bf16", "1.84 MB", True, False), ("× weight", "1.84 MB", True, False)]
merge_figure("bwr_compile_layer.png", "Change ②, on the transformer layers: torch.compile, shown on one part of a layer (its normalisation, forward pass)",
             "one batch of 2 × 512 tokens; operators and sizes as recorded on the CPU; a layer has several such runs of small operators between its matrix products",
             NORM, [(0, len(NORM), "one kernel generated by torch.compile:   to float32 · square · mean · add ε · 1 / square root · multiply · to bf16 · × weight\nreads the input once; the values in between stay inside the SMs", "1.84 MB", False)],
             "Without torch.compile: 8 operators, one after another; each reads its input from GPU memory and writes its result back", "With torch.compile: the 8 operators become one")
VOC = [("matrix product\n(the block's scores)", "8.4 MB", True, True), ("to float32", "16.8 MB", True, False), ("subtract the\nlog of the sum", "16.8 MB", True, False), ("exp", "16.8 MB", True, False),
       ("subtract 1 at the\nright answer", "4 KB", False, False), ("× 1 / n", "16.8 MB", True, False), ("to bf16", "8.4 MB", True, False), ("matrix product\n(the gradient)", "1.8 MB", True, True)]
merge_figure("bwr_fuse_block.png", "Change ②, on the output layer: torch.compile, shown on one vocabulary block (second pass over the block)",
             "one batch of 1,024 positions and one block of 4,096 words; sizes follow from the tensor shapes; a step has 4 batches × 2 passes × 38 blocks",
             VOC, [(0, 1, "matrix product\n(the block's scores)", "8.4 MB", True), (1, 7, "one kernel generated by torch.compile:   to float32 · subtract · exp · right answer · × 1 / n · to bf16\nreads the scores once; the values in between stay inside the SMs", "8.4 MB", False),
                   (7, 8, "matrix product\n(the gradient)", "1.8 MB", True)],
             "Without torch.compile (after change ①): 6 small operators between the two matrix products; each rewrites the block's scores", "With torch.compile: the 6 small operators become one")

# ---------------------------------------------------------------- direction two, measured next to decode on vLLM (Nov 21/22, results_n21)
# The three tenants at about the same training progress: each gets the budget that 3,600 tok/s needs at its MB per token (50, 30.7, 19.2 GB/s),
# kept by one pause after each step. Against decode alone of the same night (five runs pooled). TPOT with the queue wait in it.
def changes_result(out_name="bwr_result.png"):
    rates = [0.5, 1]; rd = "results_n21"; base = ["adjust_notenant", "adjust_notenant_b", "adjust_notenant_c", "adjust_notenant_d", "adjust_notenant_e"]
    ARMS = [(["c1_fix", "c1_fix_r2"], "usual tenant (budget 50 GB/s)", "#E69F00"), (["v_step", "v_step_r2"], "+ change ① (30.7 GB/s)", "#56B4E9"),
            (["cvfm_step_r2", "cvfm_step_r3", "cvfm_step_r4"], "+ changes ① and ② (19.2 GB/s)", "#0072B2")]
    bm, b90 = by_rate(rd, base, "mean", rates), by_rate(rd, base, 90, rates)
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw=dict(width_ratios=[1.9, 1])); fig.subplots_adjust(top=0.76, bottom=0.22, left=0.065, right=0.985, wspace=0.18)
    fig.suptitle("The same training progress for all tenants: the budget each one needs, next to decode on vLLM", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, "tenant on 32 SMs; each tenant gets the budget that about 3,600 tokens/s needs, with one pause after each step; decode TPOT includes the queue wait\n"
             "and is compared with decode alone (no tenant) on the same night; Nov 21, two to three runs each", ha="left", va="center", fontsize=9, color=MUTE, linespacing=1.4)
    w = 0.125; gap = 0.01; n = len(ARMS)
    for k, (arms, name, col) in enumerate(ARMS):
        m, p = by_rate(rd, arms, "mean", rates), by_rate(rd, arms, 90, rates)
        ten = [float(np.mean([pd.read_csv(os.path.join(CA, rd, f"{a}_phases.csv")).query("rps == @r").ft_tok_s.mean() for a in arms])) / 1e3 for r in rates]
        avg = [(m[r] / bm[r] - 1) * 100 for r in rates]; p90 = [(p[r] / b90[r] - 1) * 100 for r in rates]
        print(f"{name:32s} avg " + " ".join(f"{v:+.1f}%" for v in avg) + "  p90 " + " ".join(f"{v:+.1f}%" for v in p90) + "  tenant " + " ".join(f"{v:.2f}k" for v in ten))
        for i in range(len(rates)):
            xa = i + (k - (n - 1) / 2) * (2 * (w + gap) + 0.03) - (w + gap) / 2; xp = xa + w + gap
            ax[0].bar(xa, avg[i], w, color=col, lw=0, zorder=2); ax[0].bar(xp, p90[i], w, color=col, lw=0, alpha=0.45, hatch="///", edgecolor=col, zorder=2)
            for x, v in ((xa, avg[i]), (xp, p90[i])): ax[0].text(x, v + 0.12, f"{v:+.1f}", ha="center", va="bottom", fontsize=9, color=INK)
            xt = i + (k - (n - 1) / 2) * 0.27; ax[1].bar(xt, ten[i], 0.25, color=col, lw=0, zorder=2); ax[1].text(xt, ten[i] + 0.06, f"{ten[i]:.2f}k", ha="center", va="bottom", fontsize=9, color=INK)
    for a, name, ymax, yl in zip(ax, ("Decode TPOT: average and p90", "Tenant progress"), (10, 4.8), ("slowdown against decode alone (%)", "thousand tokens/s")):
        a.set_title(name, loc="left", fontsize=11, fontweight="bold"); a.set_xticks(range(len(rates))); a.set_xticklabels([f"{r:g} req/s" for r in rates], fontsize=10); a.set_ylim(0, ymax); a.set_ylabel(yl, fontsize=9.5)
        a.set_xlim(-0.5, len(rates) - 0.5); a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=nm) for _, nm, c in ARMS] + [Patch(facecolor="#5f6368", label="average"), Patch(facecolor="#5f6368", alpha=0.45, hatch="///", edgecolor="#5f6368", label="p90")],
               loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01), columnspacing=1.8)
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
changes_result()

# ---------------------------------------------------------------- pauses by the table: the batch of the changed tenant, its chunks, and what is inside a chunk
# Table: coloc_adjust/results_prof/seg_cb_vocab4096f.csv (tenant with both changes, alone on 32 SMs, 4 batches of 2 sequences; one batch drawn).
# Chunk rule of coloc_adjust/pacing.py (BytePacer): chunk = budget x window; a chunk ends at the first segment border past each multiple.
def chunk_detail(variant="cb_vocab4096f", out_name="bwr_chunks_detail.png", zoom=(2, 4, 7)):
    t = pd.read_csv(os.path.join(D, f"seg_{variant}.csv")); end = t.index[t.label == "mb_end"][0]; t = t.iloc[1:end + 1].reset_index(drop=True); RED = "#c5221f"
    COL = {"fwd": "#9ecae1", "p1": "#fdd0a2", "p2": "#fdae6b", "bwd": "#3182bd", "other": "#bdc1c6"}
    def kind_of(l): return "fwd" if l.endswith(":fwd") else "bwd" if l.endswith(":bwd") else "p1" if l == "head:pass1" else "p2" if l == "head:pass2" else "other"
    kind = t.label.map(kind_of).values; gb = t.total_gb.values; cum = np.concatenate([[0.0], gb.cumsum()]); tot = cum[-1]; n = len(t)
    def name_of(k):          # a segment's name in the words of the storyline
        l = t.label[k]
        if kind[k] in ("fwd", "bwd"): return f"layer {int(l[3:].split(':')[0]) + 1}"
        return {"step": "start", "mb_end": "end", "head:surrogate": "loss"}.get(l, l)
    cuts, nxt = [], CHUNK
    for k in range(1, n + 1):
        if cum[k] >= nxt: cuts.append(k); nxt = (math.floor(cum[k] / CHUNK) + 1) * CHUNK
    if cuts[-1] != n: cuts.append(n)
    chunks = list(zip([0] + cuts[:-1], cuts))
    fig = plt.figure(figsize=(13.5, 10.4)); X0, XW = 0.045, 0.94
    def head(y, text, sub=None):
        fig.text(0.02, y, text, ha="left", va="center", fontsize=11.5, fontweight="bold", color="black")
        if sub: fig.text(0.02, y - 0.022, sub, ha="left", va="center", fontsize=9.5, color=MUTE)
    def strip(y, h, xmax, xlabel=None):
        a = fig.add_axes([X0, y, XW, h]); a.set_xlim(0, xmax); a.set_ylim(0, 1); a.set_yticks([]); a.spines[["top", "right", "left"]].set_visible(False)
        if xlabel: a.set_xlabel(xlabel, fontsize=9, labelpad=2)
        else: a.set_xticks([]); a.spines["bottom"].set_visible(False)
        return a
    # 1. the table
    head(0.972, "① The table from the profile: every segment of a batch and the data it moves", f"one batch of the tenant with both changes: {n} segments, {tot:.2f} GB in all; width = GB moved; the tenant can pause between any two segments")
    a = strip(0.850, 0.05, tot, "data moved since the start of the batch (GB)")
    for k in range(n): a.add_patch(Rectangle((cum[k], 0), gb[k], 1, facecolor=COL[kind[k]], edgecolor="white", lw=0.4))
    for kd, txt, tc in (("fwd", "24 layers, forward\n{n} segments of about {e:.2f} GB", INK), ("bwd", "24 layers, backward\n{n} segments of about {e:.2f} GB", "white")):
        ix = np.where(kind == kd)[0]; a.text((cum[ix[0]] + cum[ix[-1] + 1]) / 2, 0.5, txt.format(n=len(ix), e=gb[ix].mean()), ha="center", va="center", fontsize=9.5, color=tc, fontweight="bold", linespacing=1.25, bbox=dict(facecolor=COL[kd], edgecolor="none", pad=2))
    ix = np.where((kind == "p1") | (kind == "p2"))[0]; xm = (cum[ix[0]] + cum[ix[-1] + 1]) / 2
    a.annotate(f"output layer and loss: 38 vocabulary blocks × 2 passes = {len(ix)} segments of about {gb[ix].mean():.2f} GB", xy=(xm, 1.0), xytext=(xm, 1.32), ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold", linespacing=1.25,
               arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8), annotation_clip=False)
    # 2. the chunks
    head(0.775, f"② Chunks: budget × window = {BUDGET} GB/s × {WINDOW_MS} ms = {CHUNK:.1f} GB", "a chunk ends at the first segment border after each further 0.6 GB; red = where the tenant may pause; the three chunks with a dark frame are opened up below")
    a = strip(0.665, 0.05, tot, "data moved since the start of the batch (GB)")
    for c, (i, j) in enumerate(chunks, 1):
        z = c in zoom; a.add_patch(Rectangle((cum[i], 0), cum[j] - cum[i], 1, facecolor="#cfd3d7" if c % 2 else "#e8eaed", edgecolor=INK, lw=2.4 if z else 0.8, zorder=3 if z else 2))
        if (cum[j] - cum[i]) / tot > 0.05: a.text((cum[i] + cum[j]) / 2, 0.5, f"chunk {c}\n{cum[j] - cum[i]:.2f} GB", ha="center", va="center", fontsize=9, color=INK, fontweight="bold" if z else "normal", linespacing=1.25, zorder=4)
        if j < n: a.plot([cum[j], cum[j]], [0, 1.18], color=RED, lw=2.0, zorder=5, clip_on=False)
    # 3. inside three chunks
    head(0.585, "③ What is inside a chunk", "each chunk drawn at full width; a box is one segment of the table")
    for r, c in enumerate(zoom):
        i, j = chunks[c - 1]; w = cum[j] - cum[i]; y = 0.475 - r * 0.092; a = strip(y, 0.045, w)
        for k in range(i, j):
            small = gb[k] / w < 0.035; a.add_patch(Rectangle((cum[k] - cum[i], 0), gb[k], 1, facecolor=COL[kind[k]], edgecolor="white", lw=0.5 if small else 1.2))
            if not small and kind[k] in ("fwd", "bwd"): a.text(cum[k] - cum[i] + gb[k] / 2, 0.5, f"{name_of(k)}, {'forward' if kind[k] == 'fwd' else 'backward'}\n{gb[k]:.3f} GB", ha="center", va="center", fontsize=8.5, color=INK if kind[k] == "fwd" else "white", linespacing=1.25)
        for kd, word in (("p1", "first pass"), ("p2", "second pass")):
            ix = [k for k in range(i, j) if kind[k] == kd]
            if len(ix) > 3:
                a.text((cum[ix[0]] + cum[ix[-1] + 1]) / 2 - cum[i], 0.5, f"output layer, {word}: vocabulary blocks {1 if kd == 'p1' else 39 - len(ix)} to 38,  {len(ix)} segments of {gb[ix].min():.3f} to {gb[ix].max():.3f} GB",
                       ha="center", va="center", fontsize=8.5, color=INK, bbox=dict(facecolor=COL[kd], edgecolor="none", pad=1.5, alpha=0.9))
        cnt = {kd: int((kind[i:j] == kd).sum()) for kd in COL}; parts = []
        if cnt["fwd"]: parts.append(f"{cnt['fwd']} layers forward")
        if cnt["p1"]: parts.append(f"the first pass over {'all 38' if cnt['p1'] == 38 else cnt['p1']} vocabulary blocks")
        if cnt["p2"]: parts.append(f"{cnt['p2']} vocabulary block{'s' if cnt['p2'] > 1 else ''} of the second pass")
        if cnt["bwd"]: parts.append(f"{cnt['bwd']} layers backward")
        fig.text(X0, y + 0.052, f"chunk {c}:   {j - i} segments,   {w:.2f} GB,   {t.ms[i:j].sum():.1f} ms to compute", ha="left", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
        fig.text(X0 + 0.34, y + 0.052, "=  " + ",  ".join(parts), ha="left", va="bottom", fontsize=9.5, color=INK)
    # 4. in time
    head(0.215, "④ In time: after each chunk the tenant sleeps until  data moved so far ÷ budget", "computing times are those of the tenant alone; a chunk that already took longer than that gets no pause, and the next pauses are shorter until the time is made up")
    now, spans = 0.0, []
    for c, (i, j) in enumerate(chunks, 1):
        work = t.ms[i:j].sum(); due = cum[j] / BUDGET * 1e3; spans.append((c, now, now + work, max(due, now + work))); now = max(due, now + work)
    a = strip(0.10, 0.05, now, "time since the start of the batch (ms)")
    for c, s0, s1, s2 in spans:
        a.add_patch(Rectangle((s0, 0), s1 - s0, 1, facecolor="#cfd3d7" if c % 2 else "#e8eaed", edgecolor=INK, lw=2.4 if c in zoom else 0.8, zorder=3 if c in zoom else 2))
        if (s1 - s0) / now > 0.03: a.text((s0 + s1) / 2, 0.5, str(c), ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold" if c in zoom else "normal", zorder=4)
        if s2 > s1 + 1e-6: a.add_patch(Rectangle((s1, 0), s2 - s1, 1, facecolor="white", edgecolor=RED, hatch="////", lw=1.0))
    print("chunks:", " | ".join(f"{c}: {cum[j] - cum[i]:.2f} GB {j - i} seg {t.ms[i:j].sum():.1f} ms, pause {s2 - s1:.1f}" for (c, s0, s1, s2), (i, j) in zip(spans, chunks)), f"| batch {now:.0f} ms")
    fig.legend(handles=[Patch(color=COL["fwd"], label="a layer, forward"), Patch(color=COL["p1"], label="output layer, first pass"), Patch(color=COL["p2"], label="output layer, second pass"), Patch(color=COL["bwd"], label="a layer, backward"),
                        Patch(facecolor="#cfd3d7", edgecolor=INK, label="a chunk"), Patch(facecolor="white", edgecolor=RED, hatch="////", label="pause")],
               loc="lower center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.008), columnspacing=1.8)
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
chunk_detail()

# ---------------------------------------------------------------- one budget for both tenants (50 GB/s), next to decode on vLLM
# Usual tenant: results_n21/c1_fix (+ _r2), Nov 21. Tenant with both changes: results_b50/cvfm_step_b50, Nov 22. Both pause once after
# each step; each against decode alone of its own night (TPOT alone 12.92 / 13.81 ms and 12.92 / 13.82 ms at 0.5 / 1 req/s).
def same_budget_result(out_name="bwr_same_budget.png"):
    rates = [0.5, 1]
    ARMS = [("results_n21", ["adjust_notenant", "adjust_notenant_b", "adjust_notenant_c", "adjust_notenant_d", "adjust_notenant_e"], ["c1_fix", "c1_fix_r2"], "usual tenant", "#E69F00"),
            ("results_b50", ["adjust_notenant_a"], ["cvfm_step_b50"], "tenant with changes ① and ②", "#0072B2")]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1.75, 1])); fig.subplots_adjust(top=0.76, bottom=0.20, left=0.07, right=0.985, wspace=0.20)
    fig.suptitle(f"The same budget for both tenants ({BUDGET} GB/s), next to decode on vLLM", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, "tenant on 32 SMs, one pause after each step; decode TPOT includes the queue wait; each bar against decode alone (no tenant) of its own night:\n"
             "usual tenant on Nov 21 (two runs), tenant with both changes on Nov 22 (one run)", ha="left", va="center", fontsize=9, color=MUTE, linespacing=1.4)
    w = 0.19; gap = 0.015
    for k, (rd, base, arms, name, col) in enumerate(ARMS):
        bm, b90 = by_rate(rd, base, "mean", rates), by_rate(rd, base, 90, rates); m, p = by_rate(rd, arms, "mean", rates), by_rate(rd, arms, 90, rates)
        ten = [float(np.mean([pd.read_csv(os.path.join(CA, rd, f"{a}_phases.csv")).query("rps == @r").ft_tok_s.mean() for a in arms])) / 1e3 for r in rates]
        avg = [(m[r] / bm[r] - 1) * 100 for r in rates]; p90 = [(p[r] / b90[r] - 1) * 100 for r in rates]
        print(f"{name:30s} avg " + " ".join(f"{v:+.1f}%" for v in avg) + "  p90 " + " ".join(f"{v:+.1f}%" for v in p90) + "  tenant " + " ".join(f"{v:.2f}k" for v in ten) + "   decode alone " + " ".join(f"{bm[r]:.2f}" for r in rates) + " ms")
        for i in range(len(rates)):
            xa = i + (2 * k - 1.5) * (w + gap) - (0.02 if k == 0 else -0.02); xp = xa + w + gap
            ax[0].bar(xa, avg[i], w, color=col, lw=0, zorder=2); ax[0].bar(xp, p90[i], w, color=col, lw=0, alpha=0.45, hatch="///", edgecolor=col, zorder=2)
            for x, v in ((xa, avg[i]), (xp, p90[i])): ax[0].text(x, v + 0.15, f"{v:+.1f} %", ha="center", va="bottom", fontsize=9, color=INK)
            xt = i + (k - 0.5) * 0.39; ax[1].bar(xt, ten[i], 0.36, color=col, lw=0, zorder=2); ax[1].text(xt, ten[i] + 0.12, f"{ten[i]:.2f}k", ha="center", va="bottom", fontsize=9.5, color=INK)
    for a, name, ymax, yl in zip(ax, ("Decode TPOT: average and p90", "Tenant progress"), (12, 11), ("slowdown against decode alone (%)", "thousand tokens/s")):
        a.set_title(name, loc="left", fontsize=11, fontweight="bold"); a.set_xticks(range(len(rates))); a.set_xticklabels([f"{r:g} req/s" for r in rates], fontsize=10); a.set_ylim(0, ymax); a.set_ylabel(yl, fontsize=9.5)
        a.set_xlim(-0.5, len(rates) - 0.5); a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=n) for _, _, _, n, c in ARMS] + [Patch(facecolor="#5f6368", label="average"), Patch(facecolor="#5f6368", alpha=0.45, hatch="///", edgecolor="#5f6368", label="p90")],
               loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01), columnspacing=2.2)
    out = os.path.join(H, "figures", out_name); fig.savefig(out, dpi=140); plt.close(fig); print(out)
same_budget_result()
