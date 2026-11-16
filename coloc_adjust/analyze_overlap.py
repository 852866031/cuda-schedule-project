#!/usr/bin/env python3
"""Decode inter-token time split by what the fine-tune tenant was doing at that moment.
  python analyze_overlap.py [--dir results] [--schemes a,b,...]
For every decode token gap (request log, per-token timestamps) the tenant's state comes from its step log:
a step row covers [ts - dt, ts]; it ran during [ts - dt, ts - sleep] and slept (tokens/s cap) or was paused
for the rest. A gap counts as "tenant running" if >= 90 % of it lies in running time, "tenant not running"
if <= 10 %, and is dropped otherwise. Two baselines from the no-tenant run (adjust_notenant), one table each:
  A. the same phases of the same seeded schedule: the total effect, including that slower steps keep more
     requests running at once. Valid where the tenant's state does not follow the load (0.5 and 1 req/s).
  B. the same number of running requests: the per-step effect only. Needed at 2 req/s, where the tenant
     pauses when the load bursts, so a phase mean would mix load and tenant state.
The probe uses its own no-tenant window. Writes <dir>/overlap_summary.md."""
import argparse, json, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results")
ap.add_argument("--schemes", default="adjust_nogate,adjust_mps20,green_split,green_dyn2,tpot_slo15,tpot_slo5")
ap.add_argument("--schedule", default="0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"); ap.add_argument("--end", type=float, default=660)
args = ap.parse_args(); D = args.dir
sched = [(float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(","))]
phases = [(t, (sched[i + 1][0] if i + 1 < len(sched) else args.end), r) for i, (t, r) in enumerate(sched)]

def gaps(tag):
    """(t_mid, gap_ms, t_start, t_end) of every inter-token gap, times relative to the run's t0"""
    t0 = float(open(f"{D}/{tag}_t0.txt").read()); out = []; st, en = [], []
    for l in open(f"{D}/{tag}_load.jsonl"):
        r = json.loads(l)
        if r.get("status") != 200 or len(r.get("tok_t", [])) < 3: continue
        t = np.array(r["tok_t"]) - t0; st.append(t[0]); en.append(t[-1])
        if r["rid"].startswith("w"): continue
        out += [((a + b) / 2, (b - a) * 1e3, a, b) for a, b in zip(t[:-1], t[1:])]
    g = pd.DataFrame(out, columns=["t", "ms", "a", "b"]); st, en = np.sort(st), np.sort(en)
    g["n"] = np.searchsorted(st, g.t.values, side="right") - np.searchsorted(en, g.t.values, side="right")   # requests decoding at that moment
    return g, t0

def run_intervals(csv, t0):
    """sorted, merged [start, end) spans in which the tenant was computing"""
    f = pd.read_csv(csv); f = f[pd.to_numeric(f.dt_ms, errors="coerce").notna()]
    ts = f.ts.values - t0; dt = pd.to_numeric(f.dt_ms).values / 1e3; sl = pd.to_numeric(f.sleep_ms, errors="coerce").fillna(0).values / 1e3
    iv = sorted(zip(ts - dt, ts - sl)); m = []
    for a, b in iv:
        if m and a <= m[-1][1] + 1e-3: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    return np.array(m) if m else np.zeros((0, 2))

def run_frac(g, iv):
    """fraction of each gap covered by running spans"""
    if not len(iv): return np.zeros(len(g))
    s, e = iv[:, 0], iv[:, 1]; cum = np.concatenate([[0], np.cumsum(e - s)])
    def covered(x):   # running time in (-inf, x)
        i = np.searchsorted(s, x, side="right"); c = cum[i].copy(); j = i - 1; ok = j >= 0
        c[ok] -= np.maximum(0, e[j[ok]] - x[ok]); return c
    return (covered(g.b.values) - covered(g.a.values)) / np.maximum(1e-9, (g.b - g.a).values)

def by_rate(g):
    g = g.copy(); g["rps"] = np.nan
    for a, b, r in phases: g.loc[(g.t >= a + 10) & (g.t < b), "rps"] = r
    return g.dropna(subset=["rps"])

base, _ = gaps("adjust_notenant"); base = by_rate(base); BP = base.groupby("rps").ms.mean(); BN = base.groupby("n").ms.mean(); BN = BN[base.groupby("n").ms.size() >= 200]
def slow(w):
    """mean slowdown against the no-tenant run at the same number of running requests"""
    w = w[w.n.isin(BN.index)]
    return np.nan if len(w) <= 30 else (w.ms.values / BN.loc[w.n].values).mean() - 1
HEAD = ["| scheme | req/s | tenant running: share of time | all gaps | while tenant runs | while tenant does not run | n run / not |", "|---|---|---|---|---|---|---|"]
pct = lambda x: "–" if not np.isfinite(x) else f"{x * 100:+.1f} %"
mean_or_nan = lambda w, b: np.nan if len(w) <= 300 else w.ms.mean() / b - 1
enough = lambda sub, w: sub if len(sub) >= max(300, 0.05 * len(w)) else sub.iloc[:0]   # a state seen in under 5 % of the gaps is a few transitions, not a regime
A, Bt, FIG = [], [], {}
for tag in args.schemes.split(","):
    if not os.path.exists(f"{D}/{tag}_load.jsonl"): continue
    g, t0 = gaps(tag); iv = run_intervals(f"{D}/{tag}_ft.csv", t0); g["rf"] = run_frac(g, iv); g = by_rate(g)
    for r in (0.5, 1.0, 2.0):
        w = g[g.rps == r]
        if not len(w): continue
        run, idle = enough(w[w.rf >= 0.9], w), enough(w[w.rf <= 0.1], w); k = f"| {tag} | {r:g} | {w.rf.mean() * 100:.0f} % | "
        if r < 2: FIG[(tag, r)] = (w.rf.mean(), mean_or_nan(w, BP[r]), mean_or_nan(run, BP[r]), mean_or_nan(idle, BP[r]))
        if r < 2: A.append(k + f"{pct(mean_or_nan(w, BP[r]))} | {pct(mean_or_nan(run, BP[r]))} | {pct(mean_or_nan(idle, BP[r]))} | {len(run)} / {len(idle)} |")
        Bt.append(k + f"{pct(slow(w))} | {pct(slow(run) if len(run) > 300 else np.nan)} | {pct(slow(idle) if len(idle) > 300 else np.nan)} | {len(run)} / {len(idle)} |")
md = ["# Decode inter-token time by tenant state", "", "Mean decode inter-token time against the no-tenant run; `n` = token gaps; a cell needs at least 300 gaps and 5 % of the gaps of its row.", "",
      "## A. Against the same phases (total effect), 0.5 and 1 req/s", ""] + HEAD + A + ["", "## B. Against the same number of running requests (per-step effect)", ""] + HEAD + Bt
# the shape probe: its own no-tenant window as the baseline
if os.path.exists(f"{D}/probe_segments.json"):
    segs = json.load(open(f"{D}/probe_segments.json")); g, t0 = gaps("probe")
    b0 = [s for s in segs if s["shape"] < 0][0]; pb = g[(g.t >= b0["t_from"] + 10) & (g.t < b0["t_to"])]; PN = pb.groupby("n").ms.mean(); PN = PN[pb.groupby("n").ms.size() >= 100]
    def pslow(w):
        w = w[w.n.isin(PN.index)]
        return np.nan if len(w) <= 30 else (w.ms.values / PN.loc[w.n].values).mean() - 1
    pm = pb.ms.mean(); plain = lambda w: np.nan if len(w) <= 30 else w.ms.mean() / pm - 1
    md += ["", "## Shape probe (1 req/s, 4k tok/s cap), against its own no-tenant window", "",
           "`plain` = against the window mean (total effect); `same n` = against the same number of running requests. Tenant tok/s while running = tokens / time spent computing.", "",
           "| shape | tenant running: share of time | tenant tok/s while running | all gaps, plain | while tenant runs, plain | while tenant does not run, plain | while tenant runs, same n | while not, same n | n run / not |", "|---|---|---|---|---|---|---|---|---|"]
    for s_ in segs:
        if s_["shape"] < 0: continue
        iv = run_intervals(f"{D}/probe_ft{s_['shape']}.csv", t0); w = g[(g.t >= s_["t_from"] + 10) & (g.t < s_["t_to"])].copy(); w["rf"] = run_frac(w, iv)
        f = pd.read_csv(f"{D}/probe_ft{s_['shape']}.csv"); f["rel"] = f.ts - t0; f = f[(f.rel >= s_["t_from"] + 10) & (f.rel < s_["t_to"])]; f = f[pd.to_numeric(f.tok_s, errors="coerce").notna()]
        dt = pd.to_numeric(f.dt_ms) / 1e3; sl = pd.to_numeric(f.sleep_ms, errors="coerce").fillna(0) / 1e3; toks = (pd.to_numeric(f.tok_s) * dt).sum()
        run, idle = w[w.rf >= 0.9], w[w.rf <= 0.1]; idle = idle if len(idle) >= max(300, 0.05 * len(w)) else idle.iloc[:0]
        md.append(f"| cfg {s_['shape']} | {w.rf.mean() * 100:.0f} % | {toks / (dt - sl).sum():,.0f} | {pct(plain(w))} | {pct(plain(run))} | {pct(plain(idle))} | {pct(pslow(run))} | {pct(pslow(idle))} | {len(run)} / {len(idle)} |")
# per-request TPOT quantiles per scheme and rate (the same per-request TPOT as analyze_coloc.py)
def req_tpot(tag):
    t0 = float(open(f"{D}/{tag}_t0.txt").read()); R = []
    for l in open(f"{D}/{tag}_load.jsonl"):
        r = json.loads(l)
        if r["rid"].startswith("w") or r.get("status") != 200 or r.get("n_out", 0) < 2: continue
        R.append((r["t_send"] - t0, (r["t_done"] - r["tok_t"][0]) / max(1, r["n_out"] - 1) * 1e3))
    R = np.array(R); out = {}
    for q in (50, 90):
        for rate in (0.5, 1.0, 2.0):
            v = [np.percentile(R[(R[:, 0] >= a_ + 10) & (R[:, 0] < b_), 1], q) for a_, b_, r in phases if r == rate and ((R[:, 0] >= a_ + 10) & (R[:, 0] < b_)).sum() > 3]
            out[(q, rate)] = np.mean(v) if v else np.nan
    return out
QB = req_tpot("adjust_notenant")
md += ["", "## Per-request TPOT, median and p90 (ms; change against the no-tenant run)", "",
       "| scheme | p50 @0.5 | p50 @1 | p50 @2 | p90 @0.5 | p90 @1 | p90 @2 |", "|---|---|---|---|---|---|---|",
       "| adjust_notenant | " + " | ".join(f"{QB[(q, r)]:.1f}" for q in (50, 90) for r in (0.5, 1.0, 2.0)) + " |"]
for tag in ["static_split", "adjust"] + args.schemes.split(","):
    if not os.path.exists(f"{D}/{tag}_load.jsonl"): continue
    Q = req_tpot(tag); md.append(f"| {tag} | " + " | ".join(f"{Q[(q, r)]:.1f} ({(Q[(q, r)] / QB[(q, r)] - 1) * 100:+.0f} %)" for q in (50, 90) for r in (0.5, 1.0, 2.0)) + " |")
open(f"{D}/overlap_summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

# figure: phase-baseline slowdowns at 0.5 and 1 req/s, average against tenant running / not running
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NAMES = {"adjust_nogate": "MPS\nco-run", "green_split": "disjoint SMs\n128 / 42", "green_dyn2": "dynamic\ngreen ctx", "adjust_mps20": "MPS\n20 %",
         "tpot_slo15": "15 % latency\nSLO cap", "tpot_slo5": "5 % latency\nSLO cap"}
order = [t for t in NAMES if (t, 1.0) in FIG]; GREY, ORANGE, AQUA = "#8a93a0", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for a_, r in zip(ax, (0.5, 1.0)):
    x = np.arange(len(order))
    for j, (col, lab) in enumerate(((GREY, "average over the phase"), (ORANGE, "while the tenant is running"), (AQUA, "while the tenant is not running"))):
        v = np.array([FIG[(t, r)][j + 1] * 100 for t in order]); ok = np.isfinite(v)
        a_.bar(x[ok] + (j - 1) * 0.3, v[ok], 0.27, color=col, label=lab)
        for xi, vi in zip(x[ok], v[ok]): a_.text(xi + (j - 1) * 0.3, max(vi, 0) + 0.7, f"{vi:+.0f}", ha="center", va="bottom", fontsize=8.5)
    for xi, t in zip(x, order): a_.text(xi, -5.4, f"runs {FIG[(t, r)][0] * 100:.0f} %\nof the time", ha="center", va="top", fontsize=8.5, color="#465166")
    a_.axhline(0, color="#465166", lw=0.8); a_.set_xticks(x); a_.set_xticklabels([NAMES[t] for t in order], fontsize=9.5); a_.tick_params(axis="x", pad=30, length=0)
    a_.set_title(f"Engine at {r:g} req/s", loc="left", fontsize=11.5, fontweight="bold"); a_.set_ylim(-4, 34); a_.grid(True, axis="y", alpha=0.3); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
ax[0].set_ylabel("decode % slower than engine alone")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925), fontsize=10)
fig.suptitle("Decode slowdown: the phase average, and split by what the tenant is doing at that moment", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(f"{D}/overlap_schemes.png", dpi=140); print("figure:", f"{D}/overlap_schemes.png")
