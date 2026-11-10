#!/usr/bin/env python
"""Figure and tables for knee_corun.py (co-run version of the knee figure).
  main figure   (a) the knee curve of plot_knee.py (decode step time vs decode's SMs) again, one line per
                    tenant SM count; (b) decode on all 170 SMs: slowdown vs the tenant's SMs;
                    (c) the tenant's throughput vs its SMs, alone and next to decode
  <out>_all.png panel (a) for every measured number of running requests
    python plot_knee_corun.py --grid results/knee_corun_grid.csv --slices results/knee_corun.csv results/knee_corun_r2.csv --out results/knee_corun.png
--slices are the earlier shared/disjoint runs; their decode-on-170 points are pooled into (b) and (c)
(mean, bars = range over the runs).
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="results/knee_corun_grid.csv"); ap.add_argument("--slices", nargs="*", default=[])
ap.add_argument("--out", default="results/knee_corun.png"); ap.add_argument("--main-b", type=int, default=8)
args = ap.parse_args()
FULL = 170

def load(path, run):
    d = pd.read_csv(path)
    d["m"] = d.tenant_sms.where(d.tenant_sms != 0, FULL)                   # tenant on the full device -> 170
    d["n"] = d.decode_sms.where(d.decode_sms != 0, FULL)                   # decode on the full device -> 170
    start = d[d.series == "alone"].groupby(["n", "B"]).step_ms.median(); end = d[d.series == "alone_end"].groupby(["n", "B"]).step_ms.median()
    base = ((start + end) / 2).rename("alone_ms").reset_index()           # decode alone: mean of the run's start and end
    drift = ((end / start - 1) * 100).abs().max()
    ta = d[d.series == "tenant_alone"][["m", "tenant_tok_s"]].rename(columns=dict(tenant_tok_s="alone_tok_s"))
    co = d[d.series.isin(["grid", "shared", "disjoint"])].groupby(["m", "n", "B"]).agg(step_ms=("step_ms", "median"), tok_s=("tenant_tok_s", "first")).reset_index()
    co = co.merge(base, on=["n", "B"]).merge(base[base.n == FULL][["B", "alone_ms"]].rename(columns=dict(alone_ms="alone_full_ms")), on="B").merge(ta, on="m")
    co["slow"] = (co.step_ms / co.alone_full_ms - 1) * 100                 # against decode alone on all 170 SMs
    co["slow_same"] = (co.step_ms / co.alone_ms - 1) * 100                 # against decode alone on the same SMs
    co["overlap"] = (co.n + co.m - FULL).clip(lower=0); co["run"] = run
    print(f"{path}: decode alone start-to-end drift at most {drift:.1f} %")
    return co, base.assign(run=run), ta.assign(run=run)

g, gbase, gta = load(args.grid, "grid")
extra = [load(p, f"slice{i + 1}") for i, p in enumerate(args.slices)]
Ms = sorted(g.m.unique()); Ns = sorted(g.n.unique()); Bs = sorted(g.B.unique())
tcol = {m: c for m, c in zip(Ms, plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(Ms))))}
bcol = {1: "#1f77b4", 4: "#2ca02c", 8: "#9467bd"}

def knee_panel(a, B, legend=True):
    b = gbase[gbase.B == B].sort_values("n")
    a.plot(b.n, b.alone_ms, "ko-", ms=5, lw=2, label="decode alone")
    for m in Ms:
        s = g[(g.m == m) & (g.B == B)].sort_values("n"); c = tcol[m]
        a.plot(s.n, s.step_ms, "-", color=c, lw=1.6, label=f"tenant on {int(m)} SMs" + (" (all)" if m == FULL else ""))
        dis = s[s.overlap == 0]; ov = s[s.overlap > 0]
        a.plot(dis.n, dis.step_ms, "o", color=c, ms=5); a.plot(ov.n, ov.step_ms, "o", color=c, ms=5, mfc="white")
    a.set_xlabel(f"SMs given to decode (of {FULL})"); a.set_ylabel("decode step time (ms)"); a.set_xlim(30, FULL + 6); a.set_ylim(0, None); a.grid(True, alpha=0.3)
    if legend:
        a.plot([], [], "o", color="gray", ms=5, mfc="white", label="hollow: decode and tenant share SMs")
        a.legend(fontsize=8, loc="upper right")

# plateau: decode on all 170 SMs, pooled over the grid and the slice runs
pl = pd.concat([g[g.n == FULL]] + [e[0][e[0].n == FULL] for e in extra])
pl = pl.groupby(["m", "B"]).agg(slow=("slow", "mean"), lo=("slow", "min"), hi=("slow", "max"), tok_s=("tok_s", "mean"), tlo=("tok_s", "min"), thi=("tok_s", "max"),
                                step_ms=("step_ms", "mean"), runs=("run", "count")).reset_index()
ta = pd.concat([gta] + [e[2] for e in extra]).groupby("m").agg(tok_s=("alone_tok_s", "mean"), lo=("alone_tok_s", "min"), hi=("alone_tok_s", "max")).reset_index()

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
knee_panel(ax[0], args.main_b)
ax[0].set_title(f"(a) the knee curve with the fine-tune tenant running\n({args.main_b} running requests)", fontsize=10)
for B in Bs:
    s = pl[pl.B == B].sort_values("m"); lab = f"{B} running request{'s' if B > 1 else ''}"
    ax[1].errorbar(s.m, s.slow, yerr=[s.slow - s.lo, s.hi - s.slow], fmt="o-", color=bcol.get(B, "k"), ms=5, capsize=2, label=lab)
    ax[2].errorbar(s.m, s.tok_s / 1e3, yerr=[(s.tok_s - s.tlo) / 1e3, (s.thi - s.tok_s) / 1e3], fmt="o-", color=bcol.get(B, "k"), ms=5, capsize=2, label="next to decode, " + lab)
ax[2].errorbar(ta.m, ta.tok_s / 1e3, yerr=[(ta.tok_s - ta.lo) / 1e3, (ta.hi - ta.tok_s) / 1e3], fmt="k^-", ms=6, capsize=2, label="tenant alone")
ax[1].set_title("(b) decode on all 170 SMs (on its plateau):\nslowdown against decode alone", fontsize=10); ax[1].set_ylabel("decode step slowdown (%)"); ax[1].set_ylim(0, None)
ax[2].set_title("(c) what the tenant gets:\nits throughput, decode on all 170 SMs", fontsize=10); ax[2].set_ylabel("tenant throughput (k tok/s)"); ax[2].set_ylim(0, None)
for a in ax[1:]:
    a.set_xlabel(f"SMs given to the tenant (of {FULL})"); a.set_xlim(0, FULL + 6); a.grid(True, alpha=0.3); a.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(args.out, dpi=150, bbox_inches="tight"); print("figure:", args.out)

fig, ax = plt.subplots(1, len(Bs), figsize=(5.2 * len(Bs), 4.6), squeeze=False)
for a, B in zip(ax[0], Bs):
    knee_panel(a, B, legend=(B == Bs[-1])); a.set_title(f"{B} running request{'s' if B > 1 else ''}", fontsize=10)
fig.tight_layout(); fig.savefig(args.out.replace(".png", "_all.png"), dpi=150, bbox_inches="tight"); print("figure:", args.out.replace(".png", "_all.png"))

pd.set_option("display.width", 250)
g.round(3).to_csv(args.out.replace(".png", "_grid_summary.csv"), index=False); pl.round(3).to_csv(args.out.replace(".png", "_plateau_summary.csv"), index=False)
print(pl.round(2).to_string(index=False)); print(ta.round(0).to_string(index=False))
md = []
for B in Bs:                                                               # step time (slowdown against decode alone on 170 SMs); * = SM sets overlap
    md += [f"*{B} running request{'s' if B > 1 else ''}: decode step time in ms (slowdown against decode alone on all 170 SMs; \\* = decode and tenant share SMs)*", "",
           "| decode SMs | decode alone | " + " | ".join(f"tenant on {int(m)}" + (" (all)" if m == FULL else "") for m in Ms) + " |", "|---|---|" + "---|" * len(Ms)]
    for n in Ns:
        b = gbase[(gbase.B == B) & (gbase.n == n)].alone_ms.iloc[0]; bf = gbase[(gbase.B == B) & (gbase.n == FULL)].alone_ms.iloc[0]
        cells = []
        for m in Ms:
            r = g[(g.B == B) & (g.n == n) & (g.m == m)]
            cells.append("" if r.empty else f"{r.step_ms.iloc[0]:.1f} (+{r.slow.iloc[0]:.0f} %){chr(92) + '*' if r.overlap.iloc[0] > 0 else ''}")
        md.append(f"| {int(n)}{' (all)' if n == FULL else ''} | {b:.1f} ({(b / bf - 1) * 100:+.0f} %) | " + " | ".join(cells) + " |")
    md.append("")
md += ["*Decode on all 170 SMs: tenant throughput and decode slowdown (mean over the runs)*", "",
       "| tenant SMs | tenant alone (tok/s) | tenant next to decode (tok/s), " + " / ".join(map(str, Bs)) + " requests | decode slowdown, " + " / ".join(map(str, Bs)) + " requests | % per 1k tenant tok/s | runs |", "|---|---|---|---|---|---|"]
for m in sorted(pl.m.unique()):
    s = pl[pl.m == m].sort_values("B")
    md.append(f"| {int(m)}{' (all)' if m == FULL else ''} | {ta[ta.m == m].tok_s.iloc[0]:,.0f} | " + " / ".join(f"{v:,.0f}" for v in s.tok_s) + " | " + " / ".join(f"+{v:.0f} %" for v in s.slow)
              + " | " + " / ".join(f"{v:.1f}" for v in s.slow / (s.tok_s / 1e3)) + f" | {int(s.runs.max())} |")
open(args.out.replace(".png", ".md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
