#!/usr/bin/env python3
"""Figure of the Nov 16 night (overnight plan 3): C1 and C2 measured.  python plot_o3.py  -> results_o3/o3_c1_c2.png (and report/figures/)
(a) R1, (b) R3 from <plan>_points.csv; (c), (d) R4 from r4_summary.md (analyze_o3.py)."""
import csv, os, re, shutil
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "results_o3")
INK, MUTED, GREY, BLUE, ORANGE, AQUA, RED = "#18212f", "#465166", "#8a93a0", "#2a78d6", "#eb6834", "#1baf7a", "#cf3a3a"
def fl(x):
    try: return float(x)
    except Exception: return float("nan")
def table(plan):
    R = list(csv.DictReader(open(f"{D}/{plan}_points.csv"))); alone = defaultdict(list); T = defaultdict(list)
    for r in R:
        if r["label"] == "alone": alone[(r["decode_sms"], r["B"])].append(r)
    def base(r, k): v = [fl(a[k]) for a in alone[(r["decode_sms"], r["B"])]]; return sum(v) / len(v)
    for r in R:
        if r["label"] not in ("alone", "calib", "idle") and r.get("mean_ms"):
            T[(r["label"], r["B"])].append(((fl(r["mean_ms"]) / base(r, "mean_ms") - 1) * 100, (fl(r["p90_ms"]) / base(r, "p90_ms") - 1) * 100, fl(r["tenant_tok_s"])))
    return {k: tuple(np.mean(v, axis=0)) for k, v in T.items()}
def style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK); ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if ylabel: ax.set_ylabel(ylabel)
fig, ax = plt.subplots(2, 2, figsize=(15, 12.5))
LEG = dict(loc="upper center", frameon=False, fontsize=9.5)
# ---- (a) R1
T = table("r1"); labs = [("a", "8 SMs\nall the time"), ("c", "32 SMs\npause after\neach step"), ("h", "32 SMs\nslices of\n8 blocks"), ("g", "32 SMs\nslices of\n1 block"), ("f", "all SMs\npause after\neach step")]
x = np.arange(len(labs)); a = ax[0, 0]
for j, (k, col, lab) in enumerate(((0, GREY, "mean of decode steps"), (1, ORANGE, "p90 of decode steps"))):
    v = [T[(l, "4")][k] for l, _ in labs]; a.bar(x + (j - 0.5) * 0.38, v, 0.36, color=col, label=lab)
    for xi, vi in zip(x, v): a.text(xi + (j - 0.5) * 0.38, vi + 0.3, f"{vi:+.1f}", ha="center", fontsize=9)
a.set_xticks(x); a.set_xticklabels([n for _, n in labs], fontsize=9.5); a.set_ylim(0, 23)
style(a, "(a) C1 on the synthetic decode step (4 requests):\nthe same tenant work, 2.8 k tok/s, limited in five ways", "decode step % slower than alone")
a.legend(bbox_to_anchor=(0.5, -0.2), ncol=2, **LEG)
# ---- (b) R3
T3 = table("r3"); a = ax[0, 1]; ser = [("p", GREY, "usual tenant"), ("r1", BLUE, "torch.compile (41 % fewer bytes per token)"), ("r2", AQUA, "LoRA on the top 4 layers (31 % fewer)")]
x = np.arange(3)
for j, (l, col, lab) in enumerate(ser):
    v = [T3[(l, B)][0] for B in ("1", "4", "8")] if (l, "1") in T3 else [np.nan] * 3
    a.bar(x + (j - 1) * 0.27, v, 0.25, color=col, label=lab)
    for xi, vi in zip(x, v): a.text(xi + (j - 1) * 0.27, vi + 0.25, f"{vi:+.1f}", ha="center", fontsize=9)
a.set_xticks(x); a.set_xticklabels([f"{B} request{'s' if B != '1' else ''}\ntenant {T3[('p', B)][2] / 1e3:.1f} k tok/s" for B in ("1", "4", "8")], fontsize=9.5); a.set_ylim(0, 17.5)
style(a, "(b) C2 on the synthetic decode step:\na tenant that moves fewer bytes, at the same tenant tokens/s", "decode step % slower than alone")
a.legend(bbox_to_anchor=(0.5, -0.14), ncol=2, **LEG)
# ---- R4 tables
md = open(f"{D}/r4_summary.md").read()
sec1 = md.split("## 1.")[1].split("## 2.")[0]; sec2 = md.split("## 2.")[1].split("## 3.")[0]
P = {}
for line in sec1.splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 7 and "(" in c[1]: P[c[0]] = [float(re.search(r"\(([-+]\d+) %\)", x).group(1)) for x in c[1:]]
G = {}
for line in sec2.splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 10 and c[1] in ("0.5", "1", "2") and c[2] != "–": G[(c[0], c[1])] = (float(c[2].replace(",", "")), float(c[4].replace("%", "")))
NAME = {"tpot_slo15": "15 % latency SLO\n(fast loop, all SMs)", "green_dyn2": "dynamic\ngreen context", "c1_fix": "C1 fixed\n50 GB/s", "c1_fine": "C1 fixed,\n1-block slices",
        "c1_slow15": "C1 slow loop\n15 %", "c1_slow5": "C1 slow loop\n5 %", "c2a": "C1 + C2\ntorch.compile", "c2b": "C1 + C2\ntop 4 layers"}
DNAME = {"tpot_slo15": "15 % SLO\nfast loop\nall SMs", "green_dyn2": "dynamic\ngreen\ncontext", "c1_fix": "C1\nfixed\n50 GB/s", "c1_fine": "C1 fixed\n1-block\nslices",
         "c1_slow15": "C1\nslow loop\n15 %", "c1_slow5": "C1\nslow loop\n5 %", "c2a": "C1 + C2\ntorch.\ncompile", "c2b": "C1 + C2\ntop 4\nlayers"}
COL = {"tpot_slo15": GREY, "green_dyn2": GREY, "c1_fix": ORANGE, "c1_fine": ORANGE, "c1_slow15": ORANGE, "c1_slow5": ORANGE, "c2a": BLUE, "c2b": AQUA}
order = [t for t in NAME if t in P]
# ---- (c) cost per 1 k tenant tokens/s
a = ax[1, 0]; y = np.arange(len(order))[::-1]
for j, (r, col, lab) in enumerate((("0.5", BLUE, "engine at 0.5 req/s"), ("1", ORANGE, "engine at 1 req/s"))):
    v = [G[(t, r)][1] / G[(t, r)][0] * 1e3 if (t, r) in G else np.nan for t in order]
    a.barh(y + (0.5 - j) * 0.38, v, 0.36, color=col, label=lab)
    for yi, vi, t in zip(y, v, order):
        if vi == vi: a.text(vi + 0.04, yi + (0.5 - j) * 0.38, f"{vi:.2f} %   (decode {G[(t, r)][1]:+.1f} % at {G[(t, r)][0] / 1e3:.1f} k tok/s)", va="center", fontsize=8.8, color=INK)
a.set_yticks(y); a.set_yticklabels([NAME[t].replace("\n", " ") for t in order], fontsize=9.5); a.set_xlim(0, 5.6); a.set_xlabel("decode % slower per 1 k tenant tokens/s")
a.set_title("(c) Real engine: what 1 k tenant tokens/s cost decode\n(mean token gap against the baseline of the same night)", loc="left", fontsize=11.5, fontweight="bold", color=INK)
a.grid(True, axis="x", alpha=0.3); a.set_axisbelow(True)
for sp in ("top", "right"): a.spines[sp].set_visible(False)
a.legend(bbox_to_anchor=(0.5, -0.13), ncol=2, **LEG)
# ---- (d) per-request TPOT: median and p90
a = ax[1, 1]; x = np.arange(len(order))
for j, (idx, col, lab) in enumerate(((1, GREY, "median, 1 req/s"), (4, ORANGE, "p90, 1 req/s"), (3, RED, "p90, 0.5 req/s"))):
    v = [P[t][idx] for t in order]; a.bar(x + (j - 1) * 0.27, v, 0.25, color=col, label=lab)
    for xi, vi in zip(x, v): a.text(xi + (j - 1) * 0.27, max(vi, 0) + 0.4, f"{vi:+.0f}", ha="center", fontsize=8.5)
a.set_xticks(x); a.set_xticklabels([DNAME[t] for t in order], fontsize=8.8); a.set_ylim(-2, 33); a.axhline(0, color=MUTED, lw=0.8)
style(a, "(d) Real engine: per-request TPOT against the baseline,\nmedian and tail", "TPOT % higher than engine alone")
a.legend(bbox_to_anchor=(0.5, -0.2), ncol=3, **LEG)
fig.suptitle("Night of Nov 16: limiting the tenant's bytes (C1) and a tenant that moves fewer bytes (C2)", fontsize=13.5, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.5); out = f"{D}/o3_c1_c2.png"; fig.savefig(out, dpi=130)
shutil.copy(out, os.path.join(H, "report", "figures", "o3_c1_c2.png")); print("wrote", out)
