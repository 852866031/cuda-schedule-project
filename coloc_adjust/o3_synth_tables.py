#!/usr/bin/env python
"""Overnight plan 3: tables of a corun2.py run with the two passes averaged, and R1's decision on the slice length.
  python o3_synth_tables.py r1 [--decide]     reads results_o3/<plan>_points.csv, writes results_o3/<plan>_table.md
  --decide (R1): writes results_o3/r1_slice.txt = the longest of 1 and 8 blocks whose p90 is within 2 points of the
  mean at every batch size with less than 10 % loss of tenant speed while running (against point c); 8 if neither passes."""
import csv, os, sys
from collections import defaultdict
from o3_common import O3
plan = sys.argv[1]; R = list(csv.DictReader(open(os.path.join(O3, f"{plan}_points.csv"))))
def fl(x):
    try: return float(x)
    except Exception: return float("nan")
def mean(v): v = [x for x in v if x == x]; return sum(v) / len(v) if v else float("nan")
alone = defaultdict(list)
for r in R:
    if r["label"] == "alone": alone[(r["decode_sms"], r["B"])].append(r)
def base(r, key): return mean([fl(a[key]) for a in alone[(r["decode_sms"], r["B"])]])
pts = defaultdict(list)
for r in R:
    if r["label"] not in ("alone", "calib", "idle") and r.get("mean_ms"): pts[(r["label"], r["B"])].append(r)
order = []
for r in R:
    if r["label"] not in ("alone", "calib", "idle") and r["label"] not in order: order.append(r["label"])
T = {}
for (lab, B), rows in pts.items():
    def sl(key, bkey=None): return mean([(fl(r.get(key, "")) / base(r, bkey or key) - 1) * 100 for r in rows])
    T[(lab, B)] = dict(variant=rows[0]["variant"], sms=rows[0]["tenant_sms"], pace=rows[0]["pace"] + (" " + rows[0]["slice"] if rows[0]["pace"] == "block" else ""),
                       tok=mean([fl(r["tenant_tok_s"]) for r in rows]), tok_run=mean([fl(r["tenant_tok_s_running"]) for r in rows]), share=mean([fl(r["tenant_run_share"]) for r in rows]),
                       mean=sl("mean_ms"), p90=sl("p90_ms"), p99=sl("p99_ms"), run=sl("run_ms", "mean_ms"), idle=sl("idle_ms", "mean_ms"),
                       spread=max([(fl(r["mean_ms"]) / base(r, "mean_ms") - 1) * 100 for r in rows]) - min([(fl(r["mean_ms"]) / base(r, "mean_ms") - 1) * 100 for r in rows]),
                       bw=mean([fl(r["bw_util"]) for r in rows]), n=len(rows))
f = lambda x, d=1: "–" if x != x else f"{x:+.{d}f} %"
md = [f"# {plan}: both passes averaged; slowdown of the decode step against decode alone on the same SMs\n",
      "| point | variant | tenant SMs | pacing | B | tenant tok/s | tok/s while running | running share | decode mean | p90 | p99 | while tenant runs | while it pauses | spread between passes (points) | per 1 k tok/s | passes |", "|" + "---|" * 16]
Bs = sorted({B for _, B in T}, key=int)
for lab in order:
    for B in Bs:
        t = T.get((lab, B))
        if not t: continue
        md.append(f"| {lab} | {t['variant']} | {t['sms']} | {t['pace']} | {B} | {t['tok']:,.0f} | {t['tok_run']:,.0f} | {t['share']:.2f} | {f(t['mean'])} | {f(t['p90'])} | {f(t['p99'])} | {f(t['run'])} | {f(t['idle'])} | {t['spread']:.1f} | {t['mean'] / t['tok'] * 1e3:.2f} % | {t['n']} |")
if "--decide" in sys.argv:
    md.append("\n## Slice length for `c1_fine`\n")
    ok = {}
    for lab, n in (("g", 1), ("h", 8)):
        tail = [abs(T[(lab, B)]["p90"] - T[(lab, B)]["mean"]) for B in Bs if (lab, B) in T]
        loss = [(1 - T[(lab, B)]["tok_run"] / T[("c", B)]["tok_run"]) * 100 for B in Bs if (lab, B) in T and ("c", B) in T]
        ok[n] = bool(tail) and max(tail) <= 2.0 and max(loss) < 10.0
        md.append(f"- {n} block(s) per slice (point {lab}): p90 minus mean at most {max(tail):.1f} points, tenant speed while running {max(loss):+.1f} % lower than with pauses after a step at worst: {'passes' if ok[n] else 'fails'}")
    pick = 8 if ok[8] else (1 if ok[1] else 8)
    md.append(f"- chosen: **{pick}** block(s){'' if ok[pick] else ' (neither passed; 8 by the rule of the plan, reported as failed on the synthetic test)'}")
    open(os.path.join(O3, "r1_slice.txt"), "w").write(str(pick) + "\n")
open(os.path.join(O3, f"{plan}_table.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))
