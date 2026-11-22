#!/usr/bin/env python3
"""All vLLM-engine runs of one night in one table, every number against the engine alone on the same night.
  python analyze_night.py [--dir results_n21] [--rates 0.5,1]
Baseline = the runs named adjust_notenant* in the directory, pooled. Requests counted: status 200, at least 20 tokens,
sent in the phases of the given arrival rates, not in the first 10 s of a phase and not in the first 30 s of the run
(a tenant with compiled parts is still compiling then); the tenant's tokens are counted over the same stretches.
Columns (all "against the baseline's same statistic"):
  mean gap      mean time between two tokens of a request, all gaps
  TPOT p50/p90  per request: (last token - first token) / (tokens - 1); the median and the 90th percentile over requests
  step p50/p90/p99  single decode steps = single token gaps, without the gaps during which another request was admitted and
                without the engine's own double and near-zero gaps (over 1.6x / under 0.4x the request's median gap);
                grouped by how many requests were generating (1, 2, 3), each group against the baseline's same group, then averaged
  all-gap p90/p95/p99   percentiles of every token gap, nothing left out
  tenant        tokens/s it trained in those phases; share of the time it was computing; tokens/s over the whole 11 minutes
Runs whose names differ only by _r2, _r3 are repeats: the second table gives their mean and their range.
Writes <dir>/night_summary.md and night_summary.csv."""
import argparse, bisect, csv, json, os, re, statistics as st
ap = argparse.ArgumentParser(); H = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--dir", default=os.path.join(H, "results_n21")); ap.add_argument("--rates", default="0.5,1")
ap.add_argument("--schedule", default="0:1,90:2,180:3,270:1,360:4,420:0.5,480:2,510:3,540:2,570:1,600:1"); ap.add_argument("--end", type=float, default=660)
args = ap.parse_args(); D = args.dir
sch = [(float(a), float(b)) for a, b in (x.split(":") for x in args.schedule.split(","))]
PH = [(t, sch[i + 1][0] if i + 1 < len(sch) else args.end, r) for i, (t, r) in enumerate(sch)]; RATES = [float(x) for x in args.rates.split(",")]
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
def rate_at(t):
    for a, b, r in PH:
        if a <= t < b: return r, a
    return None, None
def load(tag):
    t0 = float(open(f"{D}/{tag}_t0.txt").read().split()[0]); reqs = []
    for line in open(f"{D}/{tag}_load.jsonl"):
        d = json.loads(line); tt = d.get("tok_t") or []
        if d.get("status") == 200 and len(tt) >= 20 and not str(d.get("rid", "")).startswith("w"): reqs.append((d["t_send"] - t0, [x - t0 for x in tt]))
    starts = sorted(r[1][0] for r in reqs); ends = sorted(r[1][-1] for r in reqs); sends = sorted(r[0] for r in reqs)
    by = {c: [] for c in (1, 2, 3)}; allg = []; tpot = []
    for t, tt in reqs:
        r, a = rate_at(t)
        if r not in RATES or t < max(a + 10, 30): continue
        tpot.append((tt[-1] - tt[0]) / (len(tt) - 1) * 1e3); med = st.median(y - x for x, y in zip(tt, tt[1:]))
        for x, y in zip(tt, tt[1:]):
            allg.append((y - x) * 1e3)
            if bisect.bisect_right(sends, y) - bisect.bisect_right(starts, x) > 0 or y - x > 1.6 * med or y - x < 0.4 * med: continue
            c = bisect.bisect_right(starts, (x + y) / 2) - bisect.bisect_left(ends, (x + y) / 2)
            if c in by: by[c].append((y - x) * 1e3)
    return dict(by=by, allg=allg, tpot=tpot, n=len(tpot))
def tenant(tag):
    p = f"{D}/{tag}_ft.csv"
    if not os.path.exists(p): return None
    t0 = float(open(f"{D}/{tag}_t0.txt").read().split()[0]); tok = comp = tok_all = 0.0; n = 0; pauses = late = 0.0
    for r in csv.DictReader(open(p)):
        if not r.get("dt_ms"): continue
        te = float(r["ts"]) - t0; dt = float(r["dt_ms"])
        if 0 <= te <= args.end: tok_all += float(r["tok_s"]) * dt / 1e3
        ra, a = rate_at(te)
        if ra not in RATES or te - dt / 1e3 < max(a + 10, 30): continue
        tok += float(r["tok_s"]) * dt / 1e3; comp += dt - float(r.get("sleep_ms") or 0) - float(r.get("slice_sleep_ms") or 0); n += 1
        pauses += float(r.get("slices") or 0); late += float(r.get("late") or 0)
    span = sum(b - max(a + 10, 30) for a, b, r in PH if r in RATES)
    return dict(tok_s=tok / span, computing=comp / 1e3 / span * 100, tok_all=tok_all / args.end, pauses=pauses / max(n, 1), late=late / max(n, 1))
done = sorted(f[:-7] for f in os.listdir(D) if f.endswith("_t0.txt") and os.path.exists(f"{D}/{f[:-7]}_load.jsonl"))
bases = [t for t in done if t.startswith("adjust_notenant")]; runs = [t for t in done if t not in bases]
L = {t: load(t) for t in done}; B = dict(by={c: sum((L[b]["by"][c] for b in bases), []) for c in (1, 2, 3)}, allg=sum((L[b]["allg"] for b in bases), []), tpot=sum((L[b]["tpot"] for b in bases), []))
def metrics(t):
    x = L[t]; m = {}
    m["mean"] = (st.mean(x["allg"]) / st.mean(B["allg"]) - 1) * 100
    m["tpot50"] = (q(x["tpot"], .5) / q(B["tpot"], .5) - 1) * 100; m["tpot90"] = (q(x["tpot"], .9) / q(B["tpot"], .9) - 1) * 100
    for n, p in (("s50", .5), ("s90", .9), ("s99", .99)):
        v = [(q(x["by"][c], p) / q(B["by"][c], p) - 1) * 100 for c in (1, 2, 3) if len(x["by"][c]) > 50]; m[n] = st.mean(v)
    for n, p in (("g90", .9), ("g95", .95), ("g99", .99)): m[n] = (q(x["allg"], p) / q(B["allg"], p) - 1) * 100
    T = tenant(t)
    if T: m.update(tok=T["tok_s"], comp=T["computing"], tok_all=T["tok_all"], pauses=T["pauses"], late=T["late"])
    return m
M = {t: metrics(t) for t in done}; K = ("mean", "tpot50", "tpot90", "s50", "s90", "s99", "g90", "g95", "g99")
def cells(m): return " | ".join(f"{m[k]:+.1f} %" for k in K) + (f" | {m['tok']:,.0f} | {m['comp']:.0f} % | {m['tok_all']:,.0f} | {m['pauses']:.0f}" if "tok" in m else " | | | |")
head = "| mean gap | TPOT p50 | TPOT p90 | step p50 | step p90 | step p99 | all-gap p90 | all-gap p95 | all-gap p99 | tenant tok/s | computing | tenant tok/s, whole run | pauses per step |"
out = [f"# vLLM-engine runs in {os.path.basename(D)}: every number against the engine alone on the same night, arrival rates {args.rates} req/s\n",
       f"Baseline runs pooled: {', '.join(bases)}. Baseline values: mean gap {st.mean(B['allg']):.2f} ms, median gap {q(B['allg'], .5):.2f} ms; TPOT median {q(B['tpot'], .5):.2f} ms, p90 {q(B['tpot'], .9):.2f} ms; "
       f"single steps with 1 / 2 / 3 requests generating: median " + " / ".join(f"{q(B['by'][c], .5):.2f}" for c in (1, 2, 3)) + " ms, p99 " + " / ".join(f"{q(B['by'][c], .99):.2f}" for c in (1, 2, 3)) + f" ms; all-gap p90 / p95 / p99 {q(B['allg'], .9):.2f} / {q(B['allg'], .95):.2f} / {q(B['allg'], .99):.2f} ms.\n",
       "## Every run\n", "| run " + head, "|---" * 14 + "|"]
for t in runs + bases: out.append(f"| {t} | " + cells(M[t]) + " |")
groups = {}
for t in runs: groups.setdefault(re.sub(r"_r\d+$", "", t), []).append(t)
out += ["\n## Repeats: mean (lowest to highest)\n", "| run | n " + head, "|---" * 15 + "|"]
rows_csv = []
for g, ts in groups.items():
    def cell(k, fmt="{:+.1f} %"):
        v = [M[t][k] for t in ts if k in M[t]]
        if not v: return ""
        return fmt.format(st.mean(v)) + (f" ({fmt.format(min(v))} to {fmt.format(max(v))})" if len(v) > 1 else "")
    out.append(f"| {g} | {len(ts)} | " + " | ".join(cell(k) for k in K) + " | " + " | ".join(cell(k, f) for k, f in (("tok", "{:,.0f}"), ("comp", "{:.0f} %"), ("tok_all", "{:,.0f}"), ("pauses", "{:.0f}"))) + " |")
    rows_csv.append(dict(run=g, n=len(ts), **{k: round(st.mean(M[t][k] for t in ts if k in M[t]), 2) for k in list(K) + ["tok", "comp", "tok_all", "pauses"] if any(k in M[t] for t in ts)}))
open(f"{D}/night_summary.md", "w").write("\n".join(out) + "\n")
with open(f"{D}/night_runs.csv", "w", newline="") as f:          # one row per run, for figures
    keys = ["run", "group"] + list(K) + ["tok", "comp", "tok_all", "pauses", "late"]; w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
    for t in runs + bases: w.writerow(dict(run=t, group=re.sub(r"_r\d+$", "", t), **{k: round(v, 3) for k, v in M[t].items()}))
with open(f"{D}/night_summary.csv", "w", newline="") as f:
    keys = ["run", "n"] + list(K) + ["tok", "comp", "tok_all", "pauses"]; w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_csv)
print("\n".join(out))
