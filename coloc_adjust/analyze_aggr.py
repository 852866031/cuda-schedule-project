#!/usr/bin/env python
"""Summary of an aggr_corun.py run (E1.2):  analyze_aggr.py results/aggr_corun.csv [more runs ...]  ->  <run>_summary.md
Per point: median decode step, slowdown against the aggressor-idle point of the same aggressor SM count, decode's
known read rate (gb_per_step / step), the aggressor's achieved GB/s, decode's lost GB/s per aggressor GB/s, and
decode + aggressor against the device counter (1 % = 17.79 GB/s, calibrated in knee_bw2)."""
import csv, statistics as st, sys
from collections import defaultdict
K = 17.79
def load(path):
    pts = defaultdict(list)
    for r in csv.DictReader(open(path)):
        pts[(r["series"], r["aggr_sms"], r["target_gbps"], r["decode_sms"], r["B"])].append(r)
    out = {}
    for k, rows in pts.items():
        f = lambda c: [float(r[c]) for r in rows if r[c] not in ("", "nan")]
        ms = f("step_ms")
        out[k] = dict(ms=st.median(ms) if ms else None, n=len(ms), spread=(max(ms) - min(ms)) / st.median(ms) * 100 if ms else None,
                      aggr=(f("aggr_gbps") or [0.0])[0], bw=(f("bw_util") or [float("nan")])[0], sm=(f("sm_util") or [float("nan")])[0],
                      gb=(f("gb_per_step") or [None])[0])
    return out
def order(t): return (0, 0) if t == "0" else (2, 0) if t == "max" else (1, float(t))
def report(path):
    P = load(path); L = [f"# {path}", ""]
    L += ["## Aggressor alone", "", "| aggr SMs | target GB/s | achieved GB/s | counter % | counter GB/s | achieved / counter | GB/s per SM | SM busy (of its SMs) |", "|---|---|---|---|---|---|---|---|"]
    for (s, m, t, _, _), p in P.items():
        if s == "aggr_alone":
            L.append(f"| {m} | {t} | {p['aggr']:.0f} | {p['bw']:.2f} | {p['bw'] * K:.0f} | {p['aggr'] / (p['bw'] * K):.3f} | {p['aggr'] / int(m):.1f} | {p['sm'] * 170 / int(m):.0f} % |")
    for series in ("corun", "corun_fixed"):
        keys = [k for k in P if k[0] == series]
        if not keys: continue
        L += ["", f"## {series}: decode step against the aggressor's achieved GB/s", "",
              "| aggr SMs | decode SMs | B | target | aggr GB/s | step ms | slowdown | decode GB/s | decode lost GB/s | lost / aggr | decode + aggr | counter GB/s | sum / counter | spread |", "|" + "---|" * 14]
        for k in sorted(keys, key=lambda k: (-int(k[1]), int(k[4]), order(k[2]))):
            _, m, t, d, B = k; p = P[k]; base = P.get((series, m, "0", d, B))
            if not base: continue
            dg = p["gb"] / p["ms"] * 1e3; lost = base["gb"] / base["ms"] * 1e3 - dg; slow = (p["ms"] / base["ms"] - 1) * 100
            L.append(f"| {m} | {d} | {B} | {t} | {p['aggr']:.0f} | {p['ms']:.2f} | {slow:+.1f} % | {dg:.0f} | {lost:.0f} | "
                     + (f"{lost / p['aggr']:.2f}" if p["aggr"] > 5 else "") + f" | {dg + p['aggr']:.0f} | {p['bw'] * K:.0f} | {(dg + p['aggr']) / (p['bw'] * K):.3f} | {p['spread']:.1f} % |")
        for B in sorted({k[4] for k in keys}, key=int):
            ts = sorted({k[2] for k in keys if k[2] != "0"}, key=order)
            L += ["", f"### {series}, B = {B}: slowdown (aggressor's achieved GB/s) by aggressor SM count and target", "", "| aggr SMs | decode SMs | " + " | ".join(f"{t} GB/s" if t != "max" else "unpaced" for t in ts) + " |", "|" + "---|" * (len(ts) + 2)]
            for m in sorted({k[1] for k in keys}, key=lambda x: -int(x)):
                d = next(k[3] for k in keys if k[1] == m); base = P[(series, m, "0", d, B)]; cells = []
                for t in ts:
                    p = P.get((series, m, t, d, B))
                    cells.append(f"{(p['ms'] / base['ms'] - 1) * 100:+.1f} % ({p['aggr']:.0f})" if p else "")
                L.append(f"| {m} | {d} | " + " | ".join(cells) + " |")
    L += ["", "## Decode alone: before the aggressor process, with it idle (target 0), after it exited", "", "| decode SMs | B | alone ms | aggressor idle ms | alone_end ms | idle vs alone | end vs alone |", "|---|---|---|---|---|---|---|"]
    for (s, m, t, d, B), p in sorted(P.items(), key=lambda kv: (-int(kv[0][3] or 0), int(kv[0][4]))):
        if s != "alone": continue
        idle = [q["ms"] for (s2, m2, t2, d2, B2), q in P.items() if s2 in ("corun", "corun_fixed") and t2 == "0" and d2 == d and B2 == B]
        e = P.get(("alone_end", "", "", d, B))
        L.append(f"| {d} | {B} | {p['ms']:.2f} | " + ", ".join(f"{x:.2f}" for x in idle) + f" | {e['ms']:.2f} | " + ", ".join(f"{(x / p['ms'] - 1) * 100:+.1f} %" for x in idle) + f" | {(e['ms'] / p['ms'] - 1) * 100:+.1f} % |" if e else "")
    out = path.replace(".csv", "_summary.md"); open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("wrote", out)
for p in sys.argv[1:]: report(p)
