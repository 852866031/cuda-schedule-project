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
S = pd.read_csv(os.path.join(D, "pace_summary.csv")); print(S.columns.tolist()[:14])
COLS = [("base", "Usual tenant", 1050, 1650), ("vocab4096", "Tenant whose output layer is computed in vocabulary blocks", 1190, 165)]   # x range: the length of the shortest trace of the column
ROWS = [("free", "no limit"), ("step", f"budget {BUDGET} GB/s,\none pause\nafter each step"), ("w12", f"budget {BUDGET} GB/s,\npauses inside\nthe step")]
# worst 12 ms window of each case, GB/s (pace_summary.md, column "12 ms windows: ... max")
WORST = {("base", "free"): 610, ("base", "step"): 610, ("base", "w12"): 610, ("vocab4096", "free"): 108, ("vocab4096", "step"): 107, ("vocab4096", "w12"): 72}
fig, ax = plt.subplots(3, 2, figsize=(13.5, 7.6), sharex="col", sharey="col"); fig.subplots_adjust(top=0.90, bottom=0.115, left=0.215, right=0.985, hspace=0.20, wspace=0.13)
for j, (ten, name, xmax, ymax) in enumerate(COLS):
    for i, (mode, rname) in enumerate(ROWS):
        a = ax[i][j]; d = pd.read_csv(os.path.join(D, f"pace_{ten}_{mode}_trace.csv"))
        a.fill_between(d.t_ms, 0, d.gbps, step="mid", color=ORG, lw=0)
        a.plot(d.t_ms, d.gbps.rolling(12, center=True, min_periods=1).mean(), color="#5b2e00", lw=1.1)
        if mode != "free": a.axhline(BUDGET, color=INK, lw=1.0, ls="--")
        a.set_xlim(0, xmax); a.set_ylim(0, ymax); a.grid(True, axis="y", alpha=0.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
        a.text(0.985, 0.93, f"mean {d.gbps.mean():.0f} GB/s · worst 12 ms: {WORST[(ten, mode)]} GB/s", transform=a.transAxes, ha="right", va="top", fontsize=9, color=INK,
               bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.9))
        if j == 0: a.text(-0.42, 0.5, rname, transform=a.transAxes, ha="left", va="center", fontsize=10, fontweight="bold", color=INK, linespacing=1.3)
        if i == 0: a.set_title(name, loc="left", fontsize=10.5, fontweight="bold")
        if i == 2: a.set_xlabel("time (ms)")
        a.set_ylabel("GB/s", fontsize=9)
        print(f"{ten:10s} {mode:5s} mean {d.gbps.mean():6.1f}  max 1 ms {d.gbps.max():5.0f}  samples {len(d)}")
fig.legend(handles=[Patch(color=ORG, label="GPU internal bandwidth used by the tenant, 1 ms bins (measured, tenant alone)"), Line2D([], [], color="#5b2e00", lw=1.1, label="the same, 12 ms average"),
                    Line2D([], [], color=INK, lw=1.0, ls="--", label=f"the budget, {BUDGET} GB/s")],
           loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.012))
fig.text(0.985, 0.955, "the two columns have different vertical scales", ha="right", va="center", fontsize=8.5, color=MUTE, style="italic")
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
