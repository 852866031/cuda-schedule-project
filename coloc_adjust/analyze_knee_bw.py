#!/usr/bin/env python
"""Byte accounting for a knee_corun.py run with the GPM counters on.   analyze_knee_bw.py results/knee_bw2.csv
Writes <run>_summary.md: counter calibration, tenant alone per SM count, and for every co-run point the decode
read rate, the tenant's traffic (tokens/s x its bytes per token measured alone on the same SM count), their sum
next to the counter, and decode's lost GB/s over the tenant's GB/s."""
import csv, sys, statistics as st
from collections import defaultdict
path = sys.argv[1]; R = list(csv.DictReader(open(path)))
def f(x): return float(x) if x not in ("", "nan") else float("nan")
G = defaultdict(list)
for r in R: G[(r["series"], r["tenant_sms_req"], r["decode_sms"], r["B"])].append(r)
def med(rows): return st.median(f(r["step_ms"]) for r in rows)
def p(rows, q): v = sorted(f(r["step_ms"]) for r in rows); return v[min(len(v) - 1, int(q * len(v)))]
# calibration: decode alone, known GB/s against the counter (least squares through the origin)
xs, ys = [], []
for (s, m, n, B), rows in G.items():
    if s in ("alone", "calib", "alone_end"):
        xs.append(f(rows[0]["bw_util"])); ys.append(f(rows[0]["gb_per_step"]) / med(rows) * 1e3)
K = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
err = [abs(K * x - y) / y * 100 for x, y in zip(xs, ys)]
out = [f"# {path}: byte accounting\n", f"Counter calibration over {len(xs)} decode-alone points: 1 % = {K:.2f} GB/s "
       f"(100 % = {K * 100 / 1e3:.2f} TB/s); error median {st.median(err):.1f} %, max {max(err):.1f} %.\n"]
out += ["## Tenant alone\n", "| Tenant SMs | tokens/s | Counter % | GB/s | GB/s per SM | MB per token | sm_util per allowed SM % | sm_occ per allowed SM % |", "|---|---|---|---|---|---|---|---|"]
BPT, TOK = {}, {}
for (s, m, n, B), rows in G.items():
    if s != "tenant_alone": continue
    r = rows[0]; M = int(m) or 170; gb = K * f(r["bw_util"]); tok = f(r["tenant_tok_s"]); BPT[m] = gb / tok; TOK[m] = tok
    out.append(f"| {M} | {tok:,.0f} | {f(r['bw_util']):.2f} | {gb:.0f} | {gb / M:.2f} | {gb / tok * 1e3:.1f} | {f(r['sm_util']) * 170 / M:.0f} | {f(r['sm_occ']) * 170 / M:.1f} |")
def alone(n, B):
    v = [med(rows) for (s, m, nn, BB), rows in G.items() if s in ("alone", "alone_end") and nn == n and BB == B]
    return sum(v) / len(v)
out += ["\n## Co-run\n", "Decode alone = same decode context, mean of the start and end series. Tenant GB/s = tokens/s × MB per token from the table above.\n",
        "| Series | Tenant SMs | Decode SMs | B | Step ms (p95) | Alone ms | Slowdown % | Decode GB/s | Tenant tok/s (% of alone) | Tenant GB/s | Sum GB/s | Counter GB/s | Decode lost GB/s | Lost ÷ tenant | % per 1k tok/s |", "|" + "---|" * 15]
for (s, m, n, B), rows in G.items():
    if s not in ("shared", "disjoint"): continue
    r = rows[0]; M = int(m) or 170; ms = med(rows); a = alone(n, B); gbs = f(r["gb_per_step"])
    d = gbs / ms * 1e3; da = gbs / a * 1e3; tok = f(r["tenant_tok_s"]); t = tok * BPT[m]; slow = (ms / a - 1) * 100
    out.append(f"| {s} | {M} | {int(n) or 170} | {B} | {ms:.2f} ({p(rows, .95):.2f}) | {a:.2f} | {slow:+.1f} | {d:.0f} | {tok:,.0f} ({tok / TOK[m] * 100:.0f}) | {t:.0f} | {d + t:.0f} | "
               f"{K * f(r['bw_util']):.0f} | {da - d:.0f} | {(da - d) / t:.2f} | {slow / tok * 1e3:.2f} |")
open(path.replace(".csv", "_summary.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
