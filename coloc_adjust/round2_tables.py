#!/usr/bin/env python
"""Round-2 summary tables for the report (markdown to stdout and report/round2_tables.md):
  1. baseline 600 W vs 500 W (TPOT, queue) per phase
  2. decode per scheme (TPOT p50 / queue wait) and tenant per scheme (tok/s)
  3. accounting metrics per scheme from the VRAM scheduler log: time the tenant held more than its budget,
     physical oversubscription, grow_wait time, max limit, engine fixed
  4. GPU-memory busy / power per scheme at 1 and 2 req/s
"""
import argparse, os
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_names import NAMES

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results"); ap.add_argument("--old", default="results_600w")
ap.add_argument("--schemes", nargs="+", default=["adjust_notenant", "static_split", "static_split_gate", "adjust", "adjust_nogate", "adjust_mps20", "green_split", "tpot_slo5", "tpot_slo15", "green_dyn2", "pf_base", "pf_mps", "pf_mps20", "pf_split"])
ap.add_argument("--out", default="report/round2_tables.md")
args = ap.parse_args()
TOTAL, FIXED_OLD = 31.3536, 17.7
L = []
def P(a, d=None):
    f = os.path.join(d or args.dir, f"{a}_phases.csv"); return pd.read_csv(f) if os.path.exists(f) else None
def C(a):
    f = os.path.join(args.dir, f"{a}_controller.csv"); return pd.read_csv(f) if os.path.exists(f) else None
schemes = [a for a in args.schemes if P(a) is not None]
lab = lambda r: f"{r.rps:g} req/s ({r.t_from:.0f} s)"
# 1. baseline old vs new
o, n = P("adjust_notenant", args.old), P("adjust_notenant")
if o is not None and n is not None:
    L += ["### Baseline (inference alone, KV follows load): 600 W round 1 vs 500 W round 2", "", "| phase | TPOT p50 600 W | 500 W | queue 600 W | 500 W | GPU-memory busy % | power W |", "|---|---|---|---|---|---|---|"]
    for i in range(min(len(o), len(n))):
        L.append(f"| {lab(n.iloc[i])} | {o.tpot50.iloc[i]:.1f} | {n.tpot50.iloc[i]:.1f} | {o.queue_ms.iloc[i]:.0f} | {n.queue_ms.iloc[i]:.0f} | "
                 f"{n.get('dram_busy', pd.Series([np.nan]*len(n))).iloc[i]:.0f} | {n.get('power_w', pd.Series([np.nan]*len(n))).iloc[i]:.0f} |")
    L.append("")
# 2. decode and tenant per scheme
ref = P("adjust_notenant")
L += ["### Decode per scheme (TPOT p50 ms / scheduler queue wait ms), 500 W", "", "| phase | " + " | ".join(NAMES.get(a, a) for a in schemes) + " |", "|---|" + "---|" * len(schemes)]
for i, r in ref.iterrows():
    cells = []
    for a in schemes:
        d = P(a); cells.append(f"{d.tpot50.iloc[i]:.1f} / {d.queue_ms.iloc[i]:.0f}" if d is not None and i < len(d) and not pd.isna(d.tpot50.iloc[i]) else "")
    L.append(f"| {lab(r)} | " + " | ".join(cells) + " |")
L.append("")
L += ["### Tenant per scheme (tok/s; prefill schemes: prompt tok/s), 500 W", "", "| phase | " + " | ".join(NAMES.get(a, a) for a in schemes if a != "adjust_notenant") + " |", "|---|" + "---|" * (len(schemes) - 1)]
for i, r in ref.iterrows():
    cells = []
    for a in schemes:
        if a == "adjust_notenant": continue
        d = P(a); v = d.ft_tok_s.iloc[i] if d is not None and "ft_tok_s" in d and i < len(d) else np.nan
        cells.append("" if pd.isna(v) else f"{v/1e3:.1f}k")
    L.append(f"| {lab(r)} | " + " | ".join(cells) + " |")
L.append("")
# 3. accounting
L += ["### VRAM-scheduler accounting per scheme (from its log)", "", "| scheme | engine fixed GiB | max limit GiB | tenant > budget (s) | KV used + tenant + fixed > total (s) | grow_wait (s) | grow_partial | shrinks | tenant steps (preempted) |", "|---|---|---|---|---|---|---|---|---|"]
for a in schemes:
    c = C(a)
    if c is None or "ft_budget_gib" not in c: continue
    t0 = float(open(os.path.join(args.dir, f"{a}_t0.txt")).read()); c = c[(c.t - t0 >= 0) & (c.t - t0 <= 660)]
    ft = pd.to_numeric(c.ft_footprint_gib, errors="coerce").ffill().fillna(0)
    fixed = float(pd.to_numeric(c.get("engine_fixed_gib", pd.Series([FIXED_OLD])), errors="coerce").dropna().iloc[0]) if "engine_fixed_gib" in c else FIXED_OLD
    over = ((ft - c.ft_budget_gib) > 0.05).sum() * 0.1; phys = ((c.kvc_used_gib + ft + fixed) > TOTAL).sum() * 0.1
    gw = (c.action == "grow_wait").sum() * 0.1; gp = (c.action == "grow_partial").sum(); sh = c.action.astype(str).str.startswith("shrink").sum()
    steps = ""
    fcsv = os.path.join(args.dir, f"{a}_ft.csv")
    if os.path.exists(fcsv):
        f = pd.read_csv(fcsv); steps = f"{pd.to_numeric(f.tok_s, errors='coerce').notna().sum()} ({f.event.astype(str).str.contains('preempted').sum()})"
    L.append(f"| {NAMES.get(a, a)} | {fixed:.2f} | {c.limit_gib.max():.2f} | {over:.1f} | {phys:.1f} | {gw:.1f} | {gp} | {sh} | {steps} |")
L.append("")
# 4. GPU-memory busy / power at 1 and 2 req/s
L += ["### GPU-memory-controller busy and power at 1 and 2 req/s (phase means)", "", "| scheme | GPU-memory busy % @1 | @2 | power W @1 | @2 | tenant tok/s @1 | @2 |", "|---|---|---|---|---|---|---|"]
for a in schemes:
    d = P(a)
    if d is None or "dram_busy" not in d: continue
    r1 = d[d.rps == 1.0].iloc[0] if (d.rps == 1.0).any() else None; r2 = d[d.rps == 2.0].iloc[0] if (d.rps == 2.0).any() else None
    g = lambda r, k: ("" if r is None or k not in r or pd.isna(r[k]) else f"{r[k]:.0f}")
    L.append(f"| {NAMES.get(a, a)} | {g(r1,'dram_busy')} | {g(r2,'dram_busy')} | {g(r1,'power_w')} | {g(r2,'power_w')} | {g(r1,'ft_tok_s')} | {g(r2,'ft_tok_s')} |")
L.append("")
txt = "\n".join(L); open(args.out, "w").write(txt); print(txt)

# 5. compact per-rate tables, spliced into the reports between markers
SHORT = {"adjust_notenant": "Inference alone (baseline)", "pf_base": "Inference alone, 25.5 GiB tier (prefill baseline)",
         "static_split": "Static split", "static_split_gate": "Static split + idle gate", "adjust": "Adaptive KV + gate",
         "adjust_nogate": "Adaptive KV + MPS", "adjust_mps20": "Adaptive KV + MPS 20 %", "green_split": "Adaptive KV + disjoint SMs 128/42",
         "tpot_slo5": "Adaptive KV + 5 % latency SLO", "tpot_slo15": "Adaptive KV + 15 % latency SLO", "green_dyn2": "Adaptive KV + dynamic green ctx (prebuilt)",
         "pf_mps": "3B prefill tenant, MPS", "pf_mps20": "3B prefill tenant, MPS 20 %", "pf_split": "3B prefill tenant, disjoint SMs"}
RATES = [0.5, 1.0, 2.0, 3.0, 4.0]
def rate_means(d, col):
    return [d[d.rps == r][col].mean() if (d.rps == r).any() and col in d else np.nan for r in RATES]
def compact(family, base):
    b = P(base); out = ["| scheme | " + " | ".join(f"{r:g}" for r in RATES) + " | queue @3 / @4 (ms) | tenant k tok/s @0.5 / 1 / 2 |", "|---|" + "---|" * (len(RATES) + 2)]
    bm = rate_means(b, "tpot50")
    for a in family:
        d = P(a)
        if d is None: continue
        tm = rate_means(d, "tpot50"); qm = rate_means(d, "queue_ms"); fm = rate_means(d, "ft_tok_s")
        cells = [f"{t:.1f}" if a == base else f"{t:.1f} ({(t/bb-1)*100:+.0f} %)" for t, bb in zip(tm, bm)]
        ft = "" if a == base else " / ".join("–" if pd.isna(v) else f"{v/1e3:.1f}" for v in fm[:3])
        out.append(f"| {SHORT.get(a, a)} | " + " | ".join(cells) + f" | {qm[3]:.0f} / {qm[4]:.0f} | {ft} |")
    return out
train = [a for a in ["adjust_notenant", "static_split", "static_split_gate", "adjust", "adjust_nogate", "adjust_mps20", "green_split", "tpot_slo5", "tpot_slo15", "green_dyn2"] if P(a) is not None]
pf = [a for a in ["pf_base", "pf_mps", "pf_mps20", "pf_split"] if P(a) is not None]
C1 = ["*Decode TPOT p50 in ms at each arrival rate (mean over the phases that share the rate; in brackets the change against the baseline), the scheduler queue wait at 3 and 4 req/s, and the tenant's throughput at 0.5 / 1 / 2 req/s (k tok/s; for the prefill tenant, k prompt tok/s). Round 2, 500 W, GPU 0. Full per-phase tables: [round2_tables.md](round2_tables.md).*", ""]
C1 += ["**Fine-tune tenant family**", ""] + compact(train, "adjust_notenant") + [""]
if len(pf) > 1: C1 += ["**Prefill tenant family** (the 8B's host tier is 25.5 GiB here instead of 28, hence its own baseline)", ""] + compact(pf, "pf_base") + [""]
ctext = "\n".join(C1)
for rp in ("report/REPORT.md", "report/REPORT_zh.md"):
    if os.path.exists(rp):
        s = open(rp).read(); a, b = "<!-- round2-compact-start -->", "<!-- round2-compact-end -->"
        if a in s and b in s:
            s = s[:s.index(a) + len(a)] + "\n" + ctext + "\n" + s[s.index(b):]; open(rp, "w").write(s); print("spliced into", rp)
print(ctext)
