#!/usr/bin/env python
"""Tables for the E2.7 mode runs of aggr_corun.py (--modes): what kind of aggressor activity slows decode.
    python analyze_modes.py results/aggr_modes.csv [more.csv ...]      -> <name>_summary.md next to each
Baseline of a (aggressor SMs, B) block = mean of the idle rows before and after it (kernel "idle", "idle_end").
Byte modes (gemv32 read, write32, copy32): slowdown per 100 GB/s and decode GB/s lost per aggressor GB/s.
Call modes (tiny, compute): slowdown per 10 k calls/s.  Power, clock and throttle share are device-wide."""
import sys, csv, statistics as st
from collections import defaultdict

BYTE = ("gemv32", "write32", "copy32"); NAME = dict(gemv32="read", write32="write", copy32="copy", tiny="tiny kernel", compute="compute in L2")
def f(x, d=float("nan")):
    try: return float(x)
    except Exception: return d
def load(path):
    g = defaultdict(list)
    for r in csv.DictReader(open(path)): g[(r["series"], r["kernel"], r["aggr_sms"], r["target_gbps"], r["B"])].append(r)
    out = {}
    for k, rs in g.items():
        ms = [f(r["step_ms"]) for r in rs if r["step_ms"]]
        out[k] = dict(ms=st.median(ms) if ms else float("nan"), spread=(max(ms) - min(ms)) / st.median(ms) * 100 if ms else float("nan"),
                      aggr=f(rs[0]["aggr_gbps"]), calls=f(rs[0]["aggr_calls_s"]), gb=f(rs[0]["gb_per_step"]), bw=f(rs[0]["bw_util"]), sm=f(rs[0]["sm_util"]),
                      w=f(rs[0].get("power_w")), mhz=f(rs[0].get("clock_mhz")), thr=f(rs[0].get("throttled")), dsm=rs[0]["decode_sms"])
    return out
def summarise(path):
    d = load(path); L = [f"# {path}: aggressor modes (E2.7)", ""]
    L += ["## Each mode alone", "", "| Aggressor SMs | Mode | Target | GB/s | k calls/s | Counter % | GB/s per counter % | SM busy % | Power W | Clock MHz | Throttled |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (s, k, m, t, B), v in d.items():
        if s != "mode_alone": continue
        L.append(f"| {m} | {NAME.get(k, k)} | {t} | {v['aggr']:.0f} | {v['calls'] / 1e3:.1f} | {v['bw']:.1f} | {v['aggr'] / v['bw']:.2f} | {v['sm']:.1f} | {v['w']:.0f} | {v['mhz']:.0f} | {v['thr']:.2f} |"
                 if v["bw"] > 0.5 else f"| {m} | {NAME.get(k, k)} | {t} | {v['aggr']:.0f} | {v['calls'] / 1e3:.1f} | {v['bw']:.1f} | | {v['sm']:.1f} | {v['w']:.0f} | {v['mhz']:.0f} | {v['thr']:.2f} |")
    L += ["", "## Next to decode (decode on the SMs the aggressor does not use)", "",
          "| Aggressor SMs | B | Mode | Target | Aggr GB/s | k calls/s | Step ms | Slowdown % | % per 100 GB/s | % per 10 k calls/s | Decode GB/s lost ÷ aggr GB/s | Counter % | Sum ÷ counter | Power W | Clock MHz | Throttled |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = {}
    for (s, k, m, t, B), v in d.items():
        if s == "mode_corun" and k in ("idle", "idle_end"): base.setdefault((m, B), {})[k] = v
    drift = []
    for (m, B), b in base.items():
        a, e = b["idle"]["ms"], b.get("idle_end", b["idle"])["ms"]; drift.append((m, B, a, e, (e / a - 1) * 100, b["idle"]["w"], b["idle"]["mhz"], b["idle"]["bw"]))
        b["ms"] = (a + e) / 2
    for (s, k, m, t, B), v in d.items():
        if s != "mode_corun" or k in ("idle", "idle_end"): continue
        b0 = base[(m, B)]["ms"]; slow = (v["ms"] / b0 - 1) * 100; dec = v["gb"] / v["ms"] * 1e3; lost = v["gb"] / b0 * 1e3 - dec
        byte = k in BYTE and v["aggr"] > 1
        L.append(f"| {m} | {B} | {NAME.get(k, k)} | {t} | {v['aggr']:.0f} | {v['calls'] / 1e3:.1f} | {v['ms']:.2f} | {slow:+.1f} | "
                 + (f"{slow / v['aggr'] * 100:.1f} | | {lost / v['aggr']:.2f}" if byte else f" | {slow / v['calls'] * 1e4:.2f} | ")
                 + f" | {v['bw']:.1f} | {(dec + v['aggr']) / (v['bw'] * 17.79):.3f} | {v['w']:.0f} | {v['mhz']:.0f} | {v['thr']:.2f} |")
    L += ["", "## Decode baseline (aggressor idle) before and after each block", "", "| Aggressor SMs | B | Before ms | After ms | Drift % | Power W | Clock MHz | Counter % |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {m} | {B} | {a:.3f} | {e:.3f} | {dr:+.2f} | {w:.0f} | {mhz:.0f} | {bw:.1f} |" for m, B, a, e, dr, w, mhz, bw in drift]
    al = {k: v for k, v in d.items() if k[0] in ("alone", "alone_end")}
    if al:
        L += ["", "## Decode alone on the full device (no aggressor process), start and end", "", "| Series | B | Step ms | Counter % | Power W | Clock MHz |", "|---|---|---|---|---|---|"]
        L += [f"| {s} | {B} | {v['ms']:.3f} | {v['bw']:.1f} | {v['w']:.0f} | {v['mhz']:.0f} |" for (s, k, m, t, B), v in al.items()]
    out = path[:-4] + "_summary.md"; open(out, "w").write("\n".join(L) + "\n"); print("wrote", out)
for p in sys.argv[1:]: summarise(p)
